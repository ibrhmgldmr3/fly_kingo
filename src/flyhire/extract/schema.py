"""İlan özellik şeması (senin katmanın, turuncu).

LLM yalnızca ilan metnini görür ve bu şemaya uyan bir yanıt üretir. Şema ilanda yazanı kaydeder;
CV ile karşılaştırma, puanlama ve tat kanallarına çeviri sonraki katmanlarda (embedding, kodlayıcı)
ayrı ve ölçülebilir adımlarla yapılır.

Her iddia ilandan bir alıntıya dayanır. Alıntılar Pydantic'in `context={"text": ...}` girdisiyle
ilan metninde aranır (context yoksa, örneğin veritabanından okurken, kontrol atlanır):

- kırmızı bayrak: alıntı ilanda yoksa doğrulama hatası, Instructor LLM'den düzeltmesini ister;
- gereksinim (zorunlu / tercih): alıntısı ilanda olmayan madde atılır;
- deneyim yılı ve esnek saat: alıntı değeri desteklemiyorsa değer null / False yapılır.

Son ikisinde hata yerine düzeltme seçildi: modeller bu alanlarda yeniden denemede de ısrar ediyor
ve hata verilse ilanın tamamı kaybolurdu. Kurallar elle kontrol sonuçlarına göre konuldu
(`flyhire review`, docs/ROADMAP.md Faz 2).
"""

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

# 0'dan büyük yıl sayısı alıntıda yazmalı. Türkçe "bir" ve "on" bilerek yok: İngilizce "on" ve
# Türkçe "bir ekip" gibi ifadelerle karışıyor.
_NUMBER = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|iki|üç|dört|beş|altı|yedi|sekiz|"
    r"dokuz)\b",
    re.IGNORECASE,
)
# 0 yıl: alıntı yeni mezun / deneyimsiz adayı kabul ettiğini söylemeli.
_NO_EXPERIENCE = re.compile(
    r"graduate|mezun|entry[- ]level|no (prior )?experience|without (prior )?experience|"
    r"deneyimsiz|deneyim (şartı )?aranmaz|tecrübe (şartı )?aranmaz|\b0\b",
    re.IGNORECASE,
)
# Esnek çalışma *saati*. Uzaktan çalışma, "flexible paid time off", "Flexible Work" ve "flexible
# working model" esnek saat değildir (kullanıcı kararı, 2. tur öncesi). Kendi saatini yönetmek ve
# async-first çalışma esnek saattir (2. tur kontrol). Testlerde sabit.
_FLEXIBLE_HOURS = re.compile(
    r"flexib\w* (working |work )?(hours|schedule|start)|flexi-?time|flex-?time|"
    r"esnek (çalışma )?(saat|mesai)|manage your (own )?(time|schedule|hours)|"
    r"async(hronous)?[- ]first|operate asynchronously|work(s|ing)? asynchronously",
    re.IGNORECASE,
)
_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "`": "'"})


def normalize_text(text: str) -> str:
    """Alıntı karşılaştırması için: tırnak, boşluk ve büyük/küçük harf farklarını yok sayar."""
    return re.sub(r"\s+", " ", text.translate(_QUOTE_CHARS)).strip().casefold()


def _in_posting(quote: str, info: ValidationInfo) -> bool:
    """Alıntı ilanda geçiyor mu? Context yoksa (veritabanından okuma) kontrol yapılmaz."""
    text = (info.context or {}).get("text")
    return text is None or normalize_text(quote) in normalize_text(text)


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
    """Proje dokümanı §8'deki sözlük, elle kontrolde eklenen kategorilerle (on-call ve sonrası)."""

    FAMILY_ENVIRONMENT = "family_environment"  # "aile ortamı"
    HIGH_STRESS = "high_stress"  # "dinamik ve stres toleransı yüksek"
    JUNIOR_WITH_SENIOR_EXPERIENCE = "junior_with_senior_experience"  # "junior, 5+ yıl deneyim"
    # "rekabetçi maaş" (rakam yok) bilerek yok: maaş yazmayan ilan çok yaygın, ayırt edici değil;
    # bunu `salary` alanının boş olması zaten gösteriyor (2. tur kontrol kararı).
    FLEXIBLE_BUT_OVERTIME = "flexible_but_overtime"  # "esnek saat" + "gerektiğinde fazla mesai"
    UNICORN_ROLE = "unicorn_role"  # "her şeyi yapabilen full stack rockstar ninja"
    ON_CALL = "on_call"  # zorunlu nöbet / on-call rotasyonu
    MANDATORY_TRAVEL = "mandatory_travel"  # zorunlu, düzenli seyahat
    INTRUSIVE_APPLICATION = "intrusive_application"  # lise notları, sınav puanları gibi talepler
    OTHER = "other"


class Salary(BaseModel):
    min: float | None = Field(None, description="Lower bound as a number, if stated")
    max: float | None = Field(None, description="Upper bound as a number, if stated")
    currency: str | None = Field(None, description="ISO code such as TRY, EUR, USD")
    period: Literal["hour", "month", "year"] | None = None


class Requirement(BaseModel):
    """Bir gereksinim. "Go veya C++" ya da "PyTorch gibi" seçenekleri tek maddedir: biri yeter."""

    any_of: list[str] = Field(
        min_length=1,
        description=(
            "Short names of the accepted alternatives; one is enough ('Go or C++' -> ['Go', "
            "'C++']). A single item when there is no alternative."
        ),
    )
    quote: str = Field(description="The exact phrase from the posting that states this")


class RedFlag(BaseModel):
    kind: RedFlagKind
    quote: str = Field(
        description=(
            "The exact phrase from the posting that shows this red flag, copied character for "
            "character in the posting's own language. Keep it short (a few words to one sentence)."
        )
    )
    # LLM'e gösterilmez (şemada yok); kural tabanlı bayraklar "rule" olarak işaretlenir.
    source: SkipJsonSchema[Literal["llm", "rule"]] = "llm"

    @field_validator("quote")
    @classmethod
    def quote_must_appear_in_posting(cls, quote: str, info: ValidationInfo) -> str:
        if not _in_posting(quote, info):
            raise ValueError(
                f"Quote {quote!r} does not appear in the posting. Copy the phrase exactly as it "
                "is written in the posting, or drop this red flag."
            )
        return quote


class JobFeatures(BaseModel):
    """İlanın yapılandırılmış özeti. Yalnızca ilanda açıkça yazanlar; tahmin yok."""

    language: Literal["tr", "en", "other"] = Field(description="Main language of the posting")
    required: list[Requirement] = Field(
        description=(
            "Every requirement the posting states as mandatory: skills, tools, degrees, spoken "
            "languages, domain experience. Not responsibilities or tasks of the role."
        ),
    )
    preferred: list[Requirement] = Field(
        description="Requirements stated as a plus, nice to have, preferred or a strong signal",
    )
    experience_quote: str | None = Field(
        None,
        description=(
            "Exact phrase from the posting that states the required years of experience, or that "
            "says new graduates or candidates without experience are welcome. null if neither."
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
    locations: list[str] = Field(
        description=(
            "Every work location in the posting or its Location header, as written, including "
            "remote scopes such as 'Home based - Worldwide' or 'Remote, EMEA'. Empty if none."
        ),
    )
    salary: Salary | None = Field(
        None, description="Only when the posting gives at least one number; null otherwise"
    )
    flexible_hours_quote: str | None = Field(
        None,
        description=(
            "Exact phrase that offers flexible working hours or schedule, including managing your "
            "own time or async-first work. Remote work, hybrid work, 'flexible work' without hours "
            "or flexible time off are not flexible hours. null if absent."
        ),
    )
    flexible_hours: bool = Field(description="True only when flexible_hours_quote is given")
    # Varsayılan değer yok: alan isteğe bağlı olunca bazı modeller onu hiç doldurmuyor. Sonda
    # duruyor: diğer alanlar doldurulduktan sonra üretilince bayrakları daha iyi buluyordu
    # (Llama 3.3, örnek ilanlarda ölçüldü).
    red_flags: list[RedFlag] = Field(
        description=(
            "Phrases that signal poor working conditions, each with an exact quote. "
            "Check every category in the instructions; use an empty list only if none apply."
        ),
    )

    @field_validator("required", "preferred")
    @classmethod
    def drop_unquoted_requirements(
        cls, items: list[Requirement], info: ValidationInfo
    ) -> list[Requirement]:
        """Alıntısı ilanda geçmeyen gereksinim atılır (uydurulmuş beceri); tekrarlar birleşir."""
        kept, seen = [], set()
        for item in items:
            by_key = {}
            for name in (n.strip() for n in item.any_of):
                if name:
                    by_key.setdefault(name.casefold(), name)
            names, key = list(by_key.values()), frozenset(by_key)
            if names and key not in seen and _in_posting(item.quote, info):
                seen.add(key)
                kept.append(item.model_copy(update={"any_of": names}))
        return kept

    @field_validator("experience_quote", "flexible_hours_quote")
    @classmethod
    def quote_or_none(cls, quote: str | None, info: ValidationInfo) -> str | None:
        """Boş veya ilanda geçmeyen alıntı yok sayılır (o zaman değer de desteksiz kalır)."""
        if not quote or not quote.strip() or not _in_posting(quote, info):
            return None
        return quote

    @model_validator(mode="after")
    def values_need_support(self) -> "JobFeatures":
        """Alıntıyla desteklenmeyen değerler düzeltilir (bkz. modül açıklaması)."""
        years, quote = self.min_years_experience, self.experience_quote or ""
        supported = years is not None and (
            (years > 0 and _NUMBER.search(quote)) or (years == 0 and _NO_EXPERIENCE.search(quote))
        )
        if not supported:
            self.min_years_experience = None

        flex = self.flexible_hours_quote or ""
        self.flexible_hours = bool(_FLEXIBLE_HOURS.search(flex))
        if not self.flexible_hours:
            self.flexible_hours_quote = None

        if self.salary and not any(v and v > 0 for v in (self.salary.min, self.salary.max)):
            self.salary = None
        return self
