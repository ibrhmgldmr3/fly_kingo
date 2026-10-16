"""İlan metninden `JobFeatures` çıkarımı (Instructor ile, sağlayıcıdan bağımsız).

Model `.env`'deki `LLM_MODEL` ile seçilir (`anthropic/...`, `openai/...`, `together/...`,
`ollama/...`). Together, model adını kendi biçimiyle alır (`together/<kurum>/<model>`). LLM'e CV
verilmez; yalnızca ilan. Şemaya uymayan veya ilanda geçmeyen alıntı içeren yanıtlar Instructor
tarafından hata mesajıyla birlikte yeniden istenir.
"""

from typing import Any, Protocol

from flyhire.config import get_settings
from flyhire.extract.schema import JobFeatures, JobPosting

# İstem değişirse artırılır; veritabanında hangi istemle çıkarıldığı bununla izlenir.
PROMPT_VERSION = "features-v2"

SYSTEM_PROMPT = """\
You extract structured facts from one job posting. The posting may be in Turkish or English.

Rules:
- Use only what the posting states. Do not guess, infer from the company, or fill gaps.
- Skills: short names as written in the posting (e.g. "Python", "SQL", "Docker"); no duplicates.
- Years of experience: first quote the phrase that states it (experience_quote), then give the
  minimum. "3+ yıl" or "at least 3 years" is 3. Use 0 only if the posting explicitly welcomes new
  graduates or no experience. If no number of years is written, both are null; "experience with
  X" alone is not a number.
- Seniority: only if the title or text states the level (intern, junior, senior, lead, ...).
  Do not infer it from the skills or responsibilities; otherwise use unspecified.
- Salary: only when at least one number is given. "Competitive salary" or "rekabetçi maaş" without
  a number is not a salary; report it as a vague_salary red flag instead.
- Red flags: phrases that signal poor working conditions. Categories:
  family_environment ("aile ortamı", "we are a family"),
  high_stress ("stres toleransı yüksek", "fast-paced, high pressure"),
  junior_with_senior_experience (junior or entry title but many years required),
  vague_salary (salary praised but no number),
  flexible_but_overtime (flexible hours together with expected overtime),
  unicorn_role (one person expected to cover many unrelated roles, "rockstar", "ninja"),
  other (anything else clearly negative for the employee).
  Each red flag needs a quote copied exactly from the posting. If you cannot quote it, leave it out.
"""


class StructuredClient(Protocol):
    def create(self, *, response_model: type, messages: list[dict], **kwargs: Any) -> Any: ...


def _api_key(model: str) -> str | None:
    settings = get_settings()
    provider = model.split("/", 1)[0]
    secret = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "together": settings.together_api_key,
    }.get(provider)
    return secret.get_secret_value() if secret else None


def make_client(model: str | None = None) -> StructuredClient:
    import instructor

    model = model or get_settings().llm_model
    kwargs = {}
    if key := _api_key(model):
        kwargs["api_key"] = key
    return instructor.from_provider(model, **kwargs)


def build_messages(posting: JobPosting) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Job posting:\n\n{posting.full_text}"},
    ]


def extract_features(
    posting: JobPosting,
    client: StructuredClient | None = None,
    max_retries: int = 2,
) -> JobFeatures:
    client = client or make_client()
    return client.create(
        response_model=JobFeatures,
        messages=build_messages(posting),
        max_retries=max_retries,
        context={"text": posting.full_text},
    )
