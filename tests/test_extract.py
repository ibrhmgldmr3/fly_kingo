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
        "required_skills": ["SQL"],
        "preferred_skills": [],
        "experience_quote": "En az 5 yıl veri analizi deneyimi",
        "min_years_experience": 5,
        "seniority": "junior",
        "work_mode": "onsite",
        "location": "İstanbul",
        "salary": None,
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

    assert extract_features(posting, client=client) is expected
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
        RedFlagKind.VAGUE_SALARY,
    } <= kinds


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
