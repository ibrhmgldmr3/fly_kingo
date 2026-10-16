"""Veritabanı şeması.

Akış: Job → JobFeatures (LLM) → Decision (sinek) → Application (insan).
Label tabloları kalibrasyon (Faz 3) ve değerlendirme (Faz 5) için altın standart veriyi tutar.
"""

import enum
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 768  # intfloat/multilingual-e5-base


class Base(DeclarativeBase):
    pass


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    # Veritabanında üye adları (EXTEND) yerine değerler (extend) saklanır.
    return [member.value for member in enum_cls]


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(50))
    url: Mapped[str] = mapped_column(Text)
    company: Mapped[str] = mapped_column(String(255))
    title: Mapped[str] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text)
    # başlık + şirket + metin özetinin hash'i; tekrar ayıklama bununla yapılır
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    features: Mapped["JobFeatures | None"] = relationship(back_populates="job")
    decisions: Mapped[list["Decision"]] = relationship(back_populates="job")


class JobFeatures(Base):
    """LLM'in ilandan çıkardığı yapılandırılmış özellikler (senin katmanın)."""

    __tablename__ = "job_features"

    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[dict] = mapped_column(JSONB)
    llm_model: Mapped[str] = mapped_column(String(100))
    # extract.llm.PROMPT_VERSION; istem değişince eski çıkarımlar ayırt edilebilsin
    prompt_version: Mapped[str | None] = mapped_column(String(30))
    cv_similarity: Mapped[float | None] = mapped_column(Float)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    job: Mapped[Job] = relationship(back_populates="features")


class Verdict(enum.StrEnum):
    EXTEND = "extend"  # hortum uzatıldı
    TURN_AWAY = "turn_away"  # kafasını çevirdi
    UNDECIDED = "undecided"  # kararsız, kanatlarını temizliyor


class Decision(Base):
    """Sineğin kararı. Girdiler, çıktı ve hangi sürümlerle üretildiği birlikte saklanır."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    sugar_hz: Mapped[float] = mapped_column(Float)
    water_hz: Mapped[float] = mapped_column(Float)
    bitter_hz: Mapped[float] = mapped_column(Float)
    hunger_gain: Mapped[float] = mapped_column(Float, default=1.0)
    mn9_rate_hz: Mapped[float] = mapped_column(Float)
    mn9_rate_std: Mapped[float | None] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(20))  # "surface" | "simulation"
    threshold_hz: Mapped[float] = mapped_column(Float)
    verdict: Mapped[Verdict] = mapped_column(
        Enum(Verdict, name="verdict", values_callable=_enum_values)
    )
    brain_version: Mapped[str] = mapped_column(String(50))
    encoder_version: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="decisions")


class Label(Base):
    """Kullanıcının etiketi: elle etiketleme (Faz 3) veya paneldeki "Sinek yanılıyor" (Faz 4)."""

    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    would_apply: Mapped[bool] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String(20))  # "manual" | "fly_wrong"
    split: Mapped[str | None] = mapped_column(String(10))  # "train" | "test"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ApplicationStatus(enum.StrEnum):
    FOUND = "found"
    APPROVED = "approved"
    SENT = "sent"
    INTERVIEW = "interview"
    REJECTED = "rejected"
    SILENT = "silent"


class Application(Base):
    """Başvuru takibi. Gönderme eylemi her zaman insanda; sistem yalnızca durumu kaydeder."""

    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), unique=True)
    status: Mapped[ApplicationStatus] = mapped_column(
        Enum(ApplicationStatus, name="application_status", values_callable=_enum_values),
        default=ApplicationStatus.FOUND,
    )
    cover_letter_draft: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
