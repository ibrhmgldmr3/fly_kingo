"""İlan özellik çıkarımı: şema doğrulaması (LLM'siz) ve isteğe bağlı gerçek LLM testi."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from flyhire.extract.llm import PROMPT_VERSION, SYSTEM_PROMPT, extract_features
from flyhire.extract.schema import (
    JobFeatures,
    JobPosting,
    RedFlagKind,
    Seniority,
    WorkMode,
    normalize_text,
)

POSTINGS = Path(__file__).parent / "fixtures" / "postings"
RED_FLAGS_TEXT = (POSTINGS / "tr_red_flags.txt").read_text(encoding="utf-8")


def _features(**overrides) -> dict:
    data = {
        "language": "tr",
        "required": [{"any_of": ["SQL", "Excel"], "quote": "İleri düzeyde SQL ve Excel bilgisi"}],
        "preferred": [],
        "experience_quote": "En az 5 yıl veri analizi deneyimi",
        "min_years_experience": 5,
        "seniority": "junior",
        "work_mode": "onsite",
        "locations": ["İstanbul, Maslak"],
        "salary": None,
        "flexible_hours_quote": "Esnek çalışma saatleri",
        "flexible_hours": True,
        "red_flags": [],
    }
    return data | overrides


def test_normalize_text_ignores_case_spacing_and_quote_style():
    assert normalize_text("  “Aile   ortamı”\n") == normalize_text('"aile ortamı"')


def test_quote_found_in_posting_is_accepted():
    flags = [
        {"kind": "family_environment", "quote": "sıcak bir aile ortamıyız"},
        # büyük/küçük harf ve satır sonu farkı sorun değil
        {"kind": "high_stress", "quote": "Dinamik ve stres toleransı yüksek"},
    ]
    features = JobFeatures.model_validate(
        _features(red_flags=flags), context={"text": RED_FLAGS_TEXT}
    )
    assert [f.kind for f in features.red_flags] == [
        RedFlagKind.FAMILY_ENVIRONMENT,
        RedFlagKind.HIGH_STRESS,
    ]


def test_invented_quote_is_rejected():
    flags = [{"kind": "unicorn_role", "quote": "rockstar ninja arıyoruz"}]
    with pytest.raises(ValidationError, match="does not appear in the posting"):
        JobFeatures.model_validate(_features(red_flags=flags), context={"text": RED_FLAGS_TEXT})


@pytest.mark.parametrize(
    ("quote", "years", "kept"),
    [
        ("En az 5 yıl veri analizi deneyimi", 5, True),
        ("Üniversitelerin ilgili bölümlerinden mezun", 0, True),  # "mezun"
        ("İleri düzeyde SQL ve Excel bilgisi", 3, False),  # alıntıda sayı yok
        ("İleri düzeyde SQL ve Excel bilgisi", 0, False),  # deneyimsiz kabul edildiği yazmıyor
        (None, 5, False),  # alıntı yok
        ("", 0, False),  # boş alıntı
        ("En az 3 yıl deneyim", 3, False),  # ilanda geçmeyen alıntı
    ],
)
def test_years_must_be_supported_by_quote(quote, years, kept):
    """Desteklenmeyen yıl sayısı hata vermez, null yapılır (ilan kaybolmasın)."""
    features = JobFeatures.model_validate(
        _features(experience_quote=quote, min_years_experience=years),
        context={"text": RED_FLAGS_TEXT},
    )
    assert features.min_years_experience == (years if kept else None)


def test_years_quote_with_english_on_is_not_a_number():
    """Elle kontrol #29: "depended on" içindeki "on" Türkçe 10 sanılmıştı."""
    text = "Have shipped production LLM agent systems that other people depended on"
    features = JobFeatures.model_validate(
        _features(experience_quote=text, min_years_experience=2), context={"text": text}
    )
    assert features.min_years_experience is None


def test_unquoted_requirements_are_dropped():
    """Elle kontrol #13: ilanda geçmeyen "pytest" zorunlu beceri olarak yazılmıştı."""
    required = [
        {"any_of": ["SQL"], "quote": "İleri düzeyde SQL ve Excel bilgisi"},
        {"any_of": ["pytest"], "quote": "pytest deneyimi"},
    ]
    features = JobFeatures.model_validate(
        _features(required=required), context={"text": RED_FLAGS_TEXT}
    )
    assert [r.any_of for r in features.required] == [["SQL"]]


def test_duplicate_requirements_are_merged():
    """DeepSeek denemesi: "Python; Python", "Linux; Linux"."""
    quote = "İleri düzeyde SQL ve Excel bilgisi"
    required = [
        {"any_of": ["SQL", "Excel", "sql"], "quote": quote},
        {"any_of": ["excel", "sql"], "quote": quote},
        {"any_of": [" ", "Excel"], "quote": quote},
    ]
    features = JobFeatures.model_validate(_features(required=required))
    assert [r.any_of for r in features.required] == [["SQL", "Excel"], ["Excel"]]


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("Esnek çalışma saatleri", True),
        ("flexible working hours", True),
        ("Flexible Paid Time Off", False),  # elle kontrol #12: esnek izin
        ("Fully remote role", False),  # elle kontrol #2: uzaktan çalışma
        # kullanıcı kararı (21 Ekim 2026): yalnızca açıkça esnek saat/program sayılır
        ("Flexible Work", False),  # Supabase
        ("our flexible working model called iyzico Homequarters", False),  # iyzico
        ("flexible schedule", True),
        # 2. tur kontrol (21 Ekim 2026): kendi saatini yönetmek ve async-first esnek saattir
        ("We operate asynchronously and trust you to manage your own time.", True),  # Supabase
        ("Our engineers work async-first across EMEA and AMER timezones", True),  # GitLab
        (None, False),
    ],
)
def test_flexible_hours_follow_the_quote(quote, expected):
    text = (
        f"{RED_FLAGS_TEXT}\nflexible working hours\nFlexible Paid Time Off\nFully remote role\n"
        "Flexible Work\nour flexible working model called iyzico Homequarters\nflexible schedule\n"
        "We operate asynchronously and trust you to manage your own time.\n"
        "Our engineers work async-first across EMEA and AMER timezones"
    )
    features = JobFeatures.model_validate(
        _features(flexible_hours_quote=quote, flexible_hours=True), context={"text": text}
    )
    assert features.flexible_hours is expected
    assert (features.flexible_hours_quote is not None) is expected


@pytest.mark.parametrize("salary", [{"min": 0, "max": 0}, {"min": None, "max": None}])
def test_salary_without_numbers_is_none(salary):
    """Elle kontrol #24/#25/#31: maaş yazmayan ilanda "0 - 0" çıkmıştı."""
    assert JobFeatures.model_validate(_features(salary=salary)).salary is None


def test_rule_flags_are_added_once_and_marked():
    from flyhire.extract.rules import apply_rules

    flags = apply_rules(JobFeatures.model_validate(_features())).red_flags
    assert [(f.kind.value, f.source, f.quote) for f in flags] == [
        ("junior_with_senior_experience", "rule", "En az 5 yıl veri analizi deneyimi"),
    ]
    llm_flag = {"kind": "junior_with_senior_experience", "quote": "Junior Veri Analisti"}
    flags = apply_rules(JobFeatures.model_validate(_features(red_flags=[llm_flag]))).red_flags
    assert [f.source for f in flags] == ["llm"]  # LLM zaten buldu: tekrar eklenmez


def test_rule_flags_need_their_conditions():
    from flyhire.extract.rules import rule_flags

    assert rule_flags(JobFeatures.model_validate(_features(seniority="mid"))) == []


def test_vague_salary_is_not_a_red_flag():
    """2. tur kontrol kararı: maaş yazmamak ayırt edici değil, `salary` alanı bunu gösteriyor."""
    with pytest.raises(ValidationError):
        JobFeatures.model_validate(
            _features(red_flags=[{"kind": "vague_salary", "quote": "Rekabetçi maaş"}])
        )


def _req(*names, quote="x"):
    return {"any_of": list(names), "quote": quote}


@pytest.mark.parametrize(
    ("names", "soft"),
    [
        (["Written and verbal communication"], True),
        (["Proactive", "Self-motivated", "Organized"], True),
        (["Openness to learning"], True),
        (["Analytical thinking", "Problem-solving"], True),
        (["English"], False),  # dil gereksinimi kalır
        (["Python"], False),
        (["Communication", "SQL"], False),  # bir alternatif teknik: kalır
    ],
)
def test_soft_skills_are_detected(names, soft):
    from flyhire.extract.rules import is_soft_skill
    from flyhire.extract.schema import Requirement

    assert is_soft_skill(Requirement.model_validate(_req(*names))) is soft


def test_requirements_covered_by_red_flags_are_dropped():
    """2. tur kontrol: seyahat ve akademik geçmiş yalnızca bayrakta kalsın."""
    from flyhire.extract.rules import apply_rules

    features = JobFeatures.model_validate(
        _features(
            required=[
                _req("Python", quote="Python"),
                _req("Travel", quote="Ability to travel twice a year"),
                _req("Exceptional academic track record", quote="track record from high school"),
                _req("Communication skills", quote="Excellent communication"),
            ],
            red_flags=[
                {"kind": "mandatory_travel", "quote": "travel internationally twice a year"},
                {"kind": "intrusive_application", "quote": "academic track record"},
            ],
        )
    )
    assert [r.any_of for r in apply_rules(features).required] == [["Python"]]


def test_source_is_hidden_from_llm_schema():
    assert "source" not in str(JobFeatures.model_json_schema())


def test_quote_check_skipped_without_context():
    """Veritabanından okurken ilan metni verilmez; kayıtlı alıntılar yeniden denetlenmez."""
    flags = [{"kind": "other", "quote": "metinde olmayan"}]
    assert JobFeatures.model_validate(_features(red_flags=flags)).red_flags[0].quote


def test_full_text_includes_header_fields():
    posting = JobPosting(text="İlan metni", title="Junior ML", company="Örnek A.Ş.")
    assert posting.full_text == "Title: Junior ML\nCompany: Örnek A.Ş.\n\nİlan metni"
    assert JobPosting(text="yalnız metin").full_text == "yalnız metin"


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_extract_features_sends_only_the_posting_and_quote_context():
    expected = JobFeatures.model_validate(_features())
    client = FakeClient(expected)
    posting = JobPosting(text=RED_FLAGS_TEXT, title="Junior Veri Analisti")

    result = extract_features(posting, client=client)
    assert result.required == expected.required
    # kural bayrağı: junior + 5 yıl
    assert {(f.kind, f.source) for f in result.red_flags} == {
        (RedFlagKind.JUNIOR_WITH_SENIOR_EXPERIENCE, "rule"),
    }
    (call,) = client.calls
    assert call["response_model"] is JobFeatures
    assert call["context"] == {"text": posting.full_text}
    system, user = call["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["role"] == "user" and posting.full_text in user["content"]
    assert PROMPT_VERSION


@pytest.mark.parametrize("provider", ["anthropic", "openai", "together"])
def test_api_key_is_read_per_provider(monkeypatch, provider):
    from flyhire.config import get_settings
    from flyhire.extract.llm import _api_key

    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "test-key")
    get_settings.cache_clear()
    try:
        assert _api_key(f"{provider}/some-org/some-model") == "test-key"
        assert _api_key("ollama/llama3.1") is None
    finally:
        get_settings.cache_clear()


def test_together_client_builds_without_network(monkeypatch):
    pytest.importorskip("instructor")
    pytest.importorskip("openai")
    from flyhire.config import get_settings
    from flyhire.extract.llm import make_client

    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    get_settings.cache_clear()
    try:
        client = make_client("together/meta-llama/Llama-3.3-70B-Instruct-Turbo")
        assert hasattr(client, "create")
    finally:
        get_settings.cache_clear()


# --- gerçek LLM (API anahtarı ister, varsayılan olarak atlanır: pytest -m llm) ---------------


@pytest.fixture(scope="module")
def live_client():
    from flyhire.config import get_settings
    from flyhire.extract.llm import _api_key, make_client

    if not _api_key(get_settings().llm_model) and not get_settings().llm_model.startswith("ollama"):
        pytest.skip("LLM API anahtarı yok (.env)")
    return make_client()


def _live(client, name: str) -> JobFeatures:
    text = (POSTINGS / name).read_text(encoding="utf-8")
    return extract_features(JobPosting(text=text), client=client)


@pytest.mark.llm
def test_live_turkish_red_flags(live_client):
    f = _live(live_client, "tr_red_flags.txt")
    kinds = {flag.kind for flag in f.red_flags}
    assert f.language == "tr"
    assert f.min_years_experience == 5
    assert f.seniority == Seniority.JUNIOR
    assert f.work_mode == WorkMode.ONSITE
    assert f.salary is None
    assert {
        RedFlagKind.FAMILY_ENVIRONMENT,
        RedFlagKind.HIGH_STRESS,
        RedFlagKind.JUNIOR_WITH_SENIOR_EXPERIENCE,
    } <= kinds
    # "Power BI veya Tableau deneyimi": tek gereksinim, iki seçenek; Python tercih sebebi
    bi = [set(r.any_of) for r in f.required if {"Power BI", "Tableau"} & set(r.any_of)]
    assert bi == [{"Power BI", "Tableau"}]
    assert any("Python" in r.any_of for r in f.preferred)
    assert not any("Python" in r.any_of for r in f.required)


@pytest.mark.llm
def test_live_turkish_remote_junior(live_client):
    f = _live(live_client, "tr_remote_junior_ml.txt")
    assert f.work_mode == WorkMode.REMOTE
    assert f.seniority == Seniority.JUNIOR
    assert f.salary is not None and f.salary.min == 60_000 and f.salary.max == 80_000
    assert f.flexible_hours
    assert not f.red_flags


@pytest.mark.llm
def test_live_english_hybrid_mid(live_client):
    f = _live(live_client, "en_hybrid_mid.txt")
    assert f.language == "en"
    assert f.min_years_experience == 3
    assert f.work_mode == WorkMode.HYBRID
    assert f.salary is not None and f.salary.currency == "EUR"
    assert RedFlagKind.UNICORN_ROLE in {flag.kind for flag in f.red_flags}


# --- elle kontrol CSV'si (LLM'siz) ---------------------------------------------------------


def test_review_values_are_readable():
    from flyhire.extract.review import _value

    data = _features(
        red_flags=[{"kind": "family_environment", "quote": "aile ortamı"}],
        salary={"min": 60000, "max": 80000, "currency": "TRY", "period": "month"},
    )
    assert _value(data, "experience") == "5 yıl: “En az 5 yıl veri analizi deneyimi”"
    assert _value(data, "salary") == "60,000 - 80,000 TRY / month"
    assert _value(data, "red_flags") == "family_environment: “aile ortamı”"
    assert _value(data, "required") == "SQL veya Excel"
    assert _value(data, "preferred") == "(yok)"
    assert _value(data, "locations") == "İstanbul, Maslak"
    assert _value(data, "flexible_hours") == "True: “Esnek çalışma saatleri”"
    assert _value(_features(min_years_experience=None), "experience") == "(yazmıyor)"


def test_score_review_counts_marks_per_field(tmp_path):
    from flyhire.extract.review import score_review

    path = tmp_path / "review.csv"
    path.write_text(
        "job_id;sirket;baslik;url;alan;deger;dogru;not\n"
        "1;A;T;u;seniority;junior;e;\n"
        "1;A;T;u;salary;(yok);H;maaş yazıyor\n"
        "2;A;T;u;seniority;mid;evet;\n"
        "2;A;T;u;salary;(yok);;\n",
        encoding="utf-8-sig",
    )
    score = score_review(path)
    assert score.correct["seniority"] == 2 and score.wrong["salary"] == 1
    assert score.unmarked == 1
    assert score.lines()[-2].startswith("TOPLAM") and "2/3" in score.lines()[-2]
