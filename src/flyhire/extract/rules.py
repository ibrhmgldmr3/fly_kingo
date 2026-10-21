"""LLM çıkarımından sonra uygulanan, kesin (deterministik) kurallar.

Bazı kararlar ilanın diğer alanlarından ya da sabit ifadelerden kesin olarak verilebilir; bunlar
için LLM'e güvenmek gerekmez ve kurallar istem değişse de aynı kalır. Kurallar elle kontrolde
(`flyhire review`) bulunan hatalara göre konuldu:

- "junior + çok yıl" bayrağı alanlardan hesaplanır (features-v3'te LLM 3 denemenin en fazla 1'inde
  buldu). Kuralla eklenen bayrak `source="rule"` taşır; LLM aynı türü bulduysa tekrar eklenmez.
- Kişisel beceriler (iletişim, sahiplenme, problem çözme...) gereksinim sayılmaz (2. tur kontrol).
- Bir kırmızı bayrağın konusu (zorunlu seyahat, lise/üniversite notları) gereksinim olarak tekrar
  yazılmaz; yalnızca bayrakta kalır (2. tur kontrol).

Faz 3'teki kırmızı bayrak sözlüğü (`encoder/redflags.py`) bu dosyanın devamı olacak.
"""

import re

from flyhire.extract.schema import (
    JobFeatures,
    RedFlag,
    RedFlagKind,
    Requirement,
    Seniority,
    normalize_text,
)

_JUNIOR = {Seniority.INTERN, Seniority.JUNIOR}
JUNIOR_MAX_YEARS = 3  # bu ve üstü yıl isteyen junior/stajyer ilanı çelişkili

# Tüm alternatifleri bu desenlere uyan gereksinim kişisel beceridir ve atılır. "English" gibi dil
# gereksinimleri etkilenmez ("communication in English" yerine LLM'den "English" istenir).
_SOFT_SKILL = re.compile(
    r"communicat|collaborat|ownership|accountab|problem[- ]?solving|analytical (thinking|skills)|"
    r"mindset|self[- ]?(motivated|starter|driven)|proactiv|organi[sz](ed|ational)|"
    r"attention to detail|work ethic|openness to learn|learning agility|continuous(ly)? learn|"
    r"curiosity|curious|influenc|team ?player|results?[- ]oriented|above[- ]and[- ]beyond|"
    r"initiative|adaptab|fast learner|learn fast|willingness to learn|passion|"
    r"iletişim|takım oyuncusu|sorumluluk|problem çözme|analitik düşünce",
    re.IGNORECASE,
)
# Bayrak türü -> o bayrağın konusunu anlatan gereksinim
_COVERED_BY_FLAG = {
    RedFlagKind.MANDATORY_TRAVEL: re.compile(r"travel|seyahat", re.IGNORECASE),
    RedFlagKind.INTRUSIVE_APPLICATION: re.compile(
        r"high school|academic (track record|results|record)|lise|sat score|test scores",
        re.IGNORECASE,
    ),
    RedFlagKind.ON_CALL: re.compile(r"on[- ]call|nöbet", re.IGNORECASE),
}


def is_soft_skill(requirement: Requirement) -> bool:
    return all(_SOFT_SKILL.search(name) for name in requirement.any_of)


def _covered(requirement: Requirement, flags: list[RedFlag]) -> bool:
    quote = normalize_text(requirement.quote)
    for flag in flags:
        flag_quote = normalize_text(flag.quote)
        if flag_quote and (flag_quote in quote or quote in flag_quote):
            return True
        pattern = _COVERED_BY_FLAG.get(flag.kind)
        text = " ".join([requirement.quote, *requirement.any_of])
        if pattern and pattern.search(text):
            return True
    return False


def clean_requirements(items: list[Requirement], flags: list[RedFlag]) -> list[Requirement]:
    return [item for item in items if not is_soft_skill(item) and not _covered(item, flags)]


def rule_flags(features: JobFeatures) -> list[RedFlag]:
    years = features.min_years_experience
    if features.seniority in _JUNIOR and years is not None and years >= JUNIOR_MAX_YEARS:
        return [
            RedFlag(
                kind=RedFlagKind.JUNIOR_WITH_SENIOR_EXPERIENCE,
                quote=features.experience_quote or "",
                source="rule",
            )
        ]
    return []


def apply_rules(features: JobFeatures) -> JobFeatures:
    """Kural bayraklarını ekler, gereksinimleri temizler (bkz. modül açıklaması)."""
    found = {flag.kind for flag in features.red_flags}
    flags = [*features.red_flags, *(f for f in rule_flags(features) if f.kind not in found)]
    return features.model_copy(
        update={
            "red_flags": flags,
            "required": clean_requirements(features.required, flags),
            "preferred": clean_requirements(features.preferred, flags),
        }
    )
