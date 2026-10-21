"""İlan metninden `JobFeatures` çıkarımı (Instructor ile, sağlayıcıdan bağımsız).

Model `.env`'deki `LLM_MODEL` ile seçilir (`anthropic/...`, `openai/...`, `together/...`,
`ollama/...`). Together, model adını kendi biçimiyle alır (`together/<kurum>/<model>`). LLM'e CV
verilmez; yalnızca ilan. Şemaya uymayan veya ilanda geçmeyen alıntı içeren yanıtlar Instructor
tarafından hata mesajıyla birlikte yeniden istenir.
"""

from typing import Any, Protocol

from flyhire.config import get_settings
from flyhire.extract.rules import apply_rules
from flyhire.extract.schema import JobFeatures, JobPosting

# İstem değişirse artırılır; veritabanında hangi istemle çıkarıldığı bununla izlenir.
PROMPT_VERSION = "features-v5"

SYSTEM_PROMPT = """\
You extract structured facts from one job posting. The posting may be in Turkish or English.

Rules:
- Use only what the posting states. Do not guess, infer from the company, or fill gaps.
- Requirements: each item has the accepted alternatives (any_of) and the exact quote.
  - "X or Y", "X/Y", "X or similar", "one of X, Y" is ONE item with all alternatives, because one
    of them is enough. Never list such alternatives as separate items.
  - "X such as Y, Z": one item whose alternatives are the general X and the examples
    ("relational databases such as PostgreSQL" -> ["Relational databases", "PostgreSQL"]).
  - "X and Y" or "X, Y and Z" are SEPARATE items, because all are needed. Exception: when a
    sentence lists the parts of one competence, make one item named after the competence
    ("golden sets, judge rubrics and CI gates" -> ["LLM evaluation design"]).
  - Required: what the posting states as mandatory, including spoken languages ("English") and
    domain experience ("fintech"). Use short names ("Python", "SQL", "English").
  - Preferred: only what is marked as a plus, nice to have, preferred, bonus or strong signal, and
    what is required only in some cases ("for more senior positions"). Do not move a mandatory
    item to preferred or a preferred item to required.
  - Not requirements: personal or soft skills (communication, ownership, problem solving,
    mindset, openness to learning), responsibilities of the role ("you will build..."), tools the
    team uses, and anything already reported as a red flag (travel, academic records, on-call).
- Locations: every place in the title, the Location header and the text (all cities and
  countries), as written. Time zones and working hours are not locations.
- Flexible hours: an explicit offer of flexible working hours or schedule, managing your own
  time, or async-first work. Remote or hybrid work, "flexible work" without hours and flexible
  paid time off are not flexible hours.
- Years of experience: first quote the phrase that states it (experience_quote), then give the
  minimum. "3+ yıl" or "at least 3 years" is 3. Use 0 only if the posting explicitly welcomes new
  graduates or no experience. If no number of years is written, both are null; "experience with
  X" alone is not a number.
- Seniority: only if the title or text states the level (intern, junior, senior, lead, ...).
  Do not infer it from the skills or responsibilities; otherwise use unspecified.
- Salary: only when at least one number is given; "competitive salary" alone is not a salary and
  not a red flag.
- Red flags: phrases that signal poor working conditions. Categories:
  family_environment ("aile ortamı", "we are a family"),
  high_stress ("stres toleransı yüksek", "fast-paced, high pressure"),
  junior_with_senior_experience (junior or entry title but many years required),
  flexible_but_overtime (flexible hours together with expected overtime),
  unicorn_role (one person expected to cover many unrelated roles, "rockstar", "ninja"),
  on_call (mandatory on-call or pager rotation, "nöbet"),
  mandatory_travel (travel stated as a requirement or expectation, e.g. "ability to travel 4
  times a year"; a company retreat described as a perk is not a red flag),
  intrusive_application (asks for high school grades, test scores or similar),
  other (clearly negative conditions only; an honest description of the work is not a flag).
  Always check: is one person expected to cover many unrelated roles (unicorn_role)? Does a
  junior or entry title require several years?
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
    features = client.create(
        response_model=JobFeatures,
        messages=build_messages(posting),
        max_retries=max_retries,
        context={"text": posting.full_text},
    )
    return apply_rules(features)
