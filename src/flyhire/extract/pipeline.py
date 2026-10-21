"""Veritabanındaki ilanlardan özellik çıkarımı: `jobs` → `job_features`.

Özelliği olmayan (veya `redo` ile, eski istem sürümüyle çıkarılmış) ilanlar sırayla LLM'e verilir.
Her ilan ayrı commit edilir: yarıda kesilen bir koşu kaldığı yerden devam eder, bir ilandaki hata
diğerlerini durdurmaz.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from flyhire.db.models import Job, JobFeatures
from flyhire.extract.llm import PROMPT_VERSION, StructuredClient, extract_features
from flyhire.extract.schema import JobPosting


@dataclass
class ExtractResult:
    done: int = 0
    failed: dict[int, str] = field(default_factory=dict)  # job_id -> hata


def pending_jobs(
    session: Session, redo: bool = False, limit: int | None = None, model: str | None = None
) -> list[Job]:
    """Özelliği olmayan ilanlar; `redo` ile güncel istemden (veya `model`'den) farklı
    sürümle çıkarılanlar da."""
    query = select(Job).outerjoin(JobFeatures).order_by(Job.id)
    condition = JobFeatures.job_id.is_(None)
    if redo:
        stale = [JobFeatures.prompt_version.is_(None), JobFeatures.prompt_version != PROMPT_VERSION]
        if model:
            stale.append(JobFeatures.llm_model != model[:100])
        condition = or_(condition, *stale)
    return list(session.scalars(query.where(condition).limit(limit)))


def to_posting(job: Job) -> JobPosting:
    return JobPosting(
        text=job.text, title=job.title, company=job.company, location=job.location, url=job.url
    )


def extract_jobs(
    session: Session,
    jobs: Iterable[Job],
    client: StructuredClient,
    model: str,
    progress: Callable[[Job, str | None], None] | None = None,
) -> ExtractResult:
    """Verilen ilanları işler (genelde `pending_jobs` sonucu)."""
    result = ExtractResult()
    for job in jobs:
        try:
            features = extract_features(to_posting(job), client=client)
        except Exception as error:  # LLM/ağ hatası: bu ilanı atla, sonraki koşuda tekrar denenir
            message = f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"
            result.failed[job.id] = message
            if progress:
                progress(job, message)
            continue
        row = session.get(JobFeatures, job.id) or JobFeatures(job_id=job.id)
        row.data = features.model_dump(mode="json")
        row.llm_model = model[:100]
        row.prompt_version = PROMPT_VERSION
        row.extracted_at = func.now()  # server_default yalnızca ilk eklemede çalışır
        session.add(row)
        session.commit()
        result.done += 1
        if progress:
            progress(job, None)
    return result
