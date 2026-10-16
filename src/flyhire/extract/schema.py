"""İlan özellik şeması (senin katmanın, turuncu).

LLM yalnızca ilan metnini görür ve bu şemaya uyan bir yanıt üretir. Şema ilanda yazanı kaydeder;
CV ile karşılaştırma, puanlama ve tat kanallarına çeviri sonraki katmanlarda (embedding, kodlayıcı)
ayrı ve ölçülebilir adımlarla yapılır.

Kırmızı bayrak alıntıları ilan metninde birebir geçmek zorundadır. Doğrulama, Pydantic'in
`context={"text": ...}` girdisiyle yapılır: alıntı metinde yoksa doğrulama hatası oluşur ve
Instructor LLM'den düzeltmesini ister. Context verilmezse (örneğin veritabanından okurken)
kontrol atlanır. Deneyim yılı da bir alıntıya dayanmak zorundadır; dayanmıyorsa hata yerine null
yapılır (bkz. `JobFeatures.years_need_quote`).
"""

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# Deneyim alıntısı kontrolü: 0'dan büyük yıl sayısı alıntıda yazmalı; 0 ise alıntı yeni mezun /
# deneyimsiz adayı kabul ettiğini söylemeli.
_NUMBER = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|bir|iki|üç|dört|beş|altı|yedi|"
    r"sekiz|dokuz|on)\b",
    re.IGNORECASE,
)
_NO_EXPERIENCE = re.compile(
    r"graduate|mezun|entry[- ]level|no (prior )?experience|without (prior )?experience|"
    r"deneyimsiz|deneyim (şartı )?aranmaz|tecrübe (şartı )?aranmaz|\b0\b",
    re.IGNORECASE,
)
_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "`": "'"})


def normalize_text(text: str) -> str:
    """Alıntı karşılaştırması için: tırnak, boşluk ve büyük/küçük harf farklarını yok sayar."""
    return re.sub(r"\s+", " ", text.translate(_QUOTE_CHARS)).strip().casefold()


class JobPosting(BaseModel):
    """Çıkarımın girdisi: toplanan ham ilan."""

    text: str
    title: str | None = None
    company: str | None = None
    location: str | None = None
    url: str | None = None

    @property
    def full_text(self) -> str:
        """LLM'e verilen ve alıntıların arandığı metin."""
        header = [
            f"{name}: {value}"
            for name, value in (
                ("Title", self.title),
                ("Company", self.company),
                ("Location", self.location),
            )
            if value
        ]
        return "\n".join([*header, "", self.text]) if header else self.text


def _check_quote(quote: str, info: ValidationInfo, fallback: str) -> str:
    text = (info.context or {}).get("text")
    if text is not None and normalize_text(quote) not in normalize_text(text):
        raise ValueError(
            f"Quote {quote!r} does not appear in the posting. Copy the phrase exactly as it is "
            f"written in the posting, {fallback}."
        )
    return quote


class Seniority(StrEnum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    UNSPECIFIED = "unspecified"


class WorkMode(StrEnum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"
    UNSPECIFIED = "unspecified"


class RedFlagKind(StrEnum):
    """Proje dokümanı §8'deki kırmızı bayrak sözlüğünün kategorileri."""

    FAMILY_ENVIRONMENT = "family_environment"  # "aile ortamı"
    HIGH_STRESS = "high_stress"  # "dinamik ve stres toleransı yüksek"
    JUNIOR_WITH_SENIOR_EXPERIENCE = "junior_with_senior_experience"  # "junior, 5+ yıl deneyim"
    VAGUE_SALARY = "vague_salary"  # "rekabetçi maaş" (rakam yok)
    FLEXIBLE_BUT_OVERTIME = "flexible_but_overtime"  # "esnek saat" + "gerektiğinde fazla mesai"
    UNICORN_ROLE = "unicorn_role"  # "her şeyi yapabilen full stack rockstar ninja"
    OTHER = "other"


class Salary(BaseModel):
    min: float | None = Field(None, description="Lower bound as a number, if stated")
    max: float | None = Field(None, description="Upper bound as a number, if stated")
    currency: str | None = Field(None, description="ISO code such as TRY, EUR, USD")
    period: Literal["hour", "month", "year"] | None = None


class RedFlag(BaseModel):
    kind: RedFlagKind
    quote: str = Field(
        description=(
            "The exact phrase from the posting that shows this red flag, copied character for "
            "character in the posting's own language. Keep it short (a few words to one sentence)."
        )
    )

    @field_validator("quote")
    @classmethod
    def quote_must_appear_in_posting(cls, quote: str, info: ValidationInfo) -> str:
        return _check_quote(quote, info, "or drop this red flag")


class JobFeatures(BaseModel):
    """İlanın yapılandırılmış özeti. Yalnızca ilanda açıkça yazanlar; tahmin yok."""

    language: Literal["tr", "en", "other"] = Field(description="Main language of the posting")
    required_skills: list[str] = Field(
        description="Skills, tools or qualifications the posting states as required",
    )
    preferred_skills: list[str] = Field(
        description="Skills stated as a plus, nice to have, or preferred",
    )
    # Sayı ancak ilandan alıntılanan bir ifadeye dayanıyorsa kabul edilir (modeller boşluğu 0 ile
    # doldurmaya eğilimli; bkz. `years_need_quote`).
    experience_quote: str | None = Field(
        None,
        description=(
            "Exact phrase from the posting that states the required years of experience, or that "
            "says new graduates or candidates without experience are welcome. null if the posting "
            "says neither."
        ),
    )
    min_years_experience: float | None = Field(
        None,
        description=(
            "Minimum years from experience_quote ('3+ yıl' -> 3; new graduates welcome -> 0). "
            "null when experience_quote is null. Values the quote does not support are discarded."
        ),
    )
    seniority: Seniority = Field(
        description="Level as stated or clearly implied by the title; unspecified otherwise"
    )
    work_mode: WorkMode
    location: str | None = Field(None, description="Work location as written; null if absent")
    salary: Salary | None = Field(
        None, description="Only when the posting gives at least one number; null otherwise"
    )
    flexible_hours: bool = Field(
        description="True only if the posting explicitly mentions flexible working hours"
    )
    # Varsayılan değer yok: alan isteğe bağlı olunca bazı modeller onu hiç doldurmuyor.
    red_flags: list[RedFlag] = Field(
        description=(
            "Phrases that signal poor working conditions, each with an exact quote. "
            "Check every category in the instructions; use an empty list only if none apply."
        ),
    )

    @field_validator("experience_quote")
    @classmethod
    def experience_quote_must_appear(cls, quote: str | None, info: ValidationInfo) -> str | None:
        """Boş veya ilanda geçmeyen alıntı yok sayılır (o zaman yıl da desteksiz kalır)."""
        if not quote or not quote.strip():
            return None
        text = (info.context or {}).get("text")
        if text is not None and normalize_text(quote) not in normalize_text(text):
            return None
        return quote

    @model_validator(mode="after")
    def years_need_quote(self) -> "JobFeatures":
        """Alıntıyla desteklenmeyen yıl sayısı null yapılır.

        Hata yerine düzeltme: modeller (ör. Llama 3.3) "yıl yazmıyor" durumunda ısrarla 0 verip
        yeniden denemede de değiştirmiyor; hata verilse ilanın tamamı kaybolurdu. Kural: > 0 yıl
        alıntıda bir sayıyla, 0 yıl alıntıda yeni mezun / deneyimsiz kabulüyle desteklenmeli.
        """
        years, quote = self.min_years_experience, self.experience_quote or ""
        supported = years is not None and (
            (years > 0 and _NUMBER.search(quote)) or (years == 0 and _NO_EXPERIENCE.search(quote))
        )
        if not supported:
            self.min_years_experience = None
        return self
