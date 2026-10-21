"""İlan toplama: HTML temizliği, ATS ayrıştırıcıları, filtreler, tekrar ayıklama (ağsız)."""

import html
from datetime import UTC, datetime, timedelta

import pytest

from flyhire.ingest.ats import FetchedJob, parse_ashby, parse_greenhouse, parse_lever
from flyhire.ingest.clean import content_hash, html_to_text
from flyhire.ingest.sources import Board, Filters, SourcesConfig
from flyhire.ingest.store import collect, read_jsonl, write_jsonl


def test_html_to_text_keeps_paragraphs_and_bullets():
    fragment = (
        "<h2>Aranan nitelikler</h2><p>Ekibimize   katılacak</p>"
        "<ul><li>SQL</li><li><p>Python&nbsp;bilgisi</p></li><li></li></ul>"
        "<script>alert(1)</script><p>Son&amp;paragraf</p>"
    )
    assert html_to_text(fragment) == (
        "Aranan nitelikler\n\nEkibimize katılacak\n\n- SQL\n- Python bilgisi\n\nSon&paragraf"
    )


def test_html_to_text_unescapes_greenhouse_content():
    assert html_to_text(html.escape("<p>Hello <b>world</b></p>"), escaped=True) == "Hello world"


def test_content_hash_ignores_case_and_whitespace():
    assert content_hash("Trendyol", "Data Scientist", "Metin  bir") == content_hash(
        "trendyol", " data scientist", "metin bir\n"
    )
    assert content_hash("A", "B", "x") != content_hash("A", "B", "y")


def test_parse_greenhouse():
    data = {
        "jobs": [
            {
                "id": 1,
                "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1",
                "company_name": "Acme",
                "title": " Data Engineer ",
                "location": {"name": "Remote, EMEA"},
                "first_published": "2026-10-02T11:31:50-04:00",
                "content": html.escape("<p>Build pipelines</p>"),
            }
        ]
    }
    (job,) = parse_greenhouse(data, "acme", "fallback")
    assert job.source == "greenhouse:acme"
    assert job.company == "Acme" and job.title == "Data Engineer"
    assert job.location == "Remote, EMEA"
    assert job.text == "Build pipelines"
    assert job.posted_at.year == 2026


def test_parse_lever_uses_html_when_plain_is_empty_and_adds_workplace():
    data = [
        {
            "id": "abc",
            "text": "Data Analyst",
            "hostedUrl": "https://jobs.lever.co/acme/abc",
            "categories": {"location": "Istanbul"},
            "country": "TR",
            "workplaceType": "hybrid",
            "createdAt": 1790754381724,
            "descriptionPlain": "",
            "description": "<div>Analiz yapacaksın</div>",
            "lists": [{"text": "Requirements", "content": "<li>SQL</li><li>Excel</li>"}],
            "additionalPlain": "Yemek kartı",
        }
    ]
    (job,) = parse_lever(data, "acme", "Acme")
    assert job.location == "Istanbul, TR (hybrid)"
    assert job.text == "Analiz yapacaksın\n\nRequirements\n\n- SQL\n- Excel\n\nYemek kartı"
    assert job.posted_at == datetime.fromtimestamp(1790754381.724, UTC)


def test_parse_ashby_joins_locations_and_skips_unlisted():
    data = {
        "jobs": [
            {
                "id": "1",
                "title": "ML Engineer",
                "jobUrl": "https://jobs.ashbyhq.com/acme/1",
                "location": "Remote (EMEA)",
                "secondaryLocations": [{"location": "Berlin"}],
                "workplaceType": "Remote",
                "publishedAt": "2026-09-18T08:41:19.683+00:00",
                "descriptionPlain": "Train models",
                "isListed": True,
            },
            {"id": "2", "title": "Hidden", "jobUrl": "x", "isListed": False},
        ]
    }
    (job,) = parse_ashby(data, "acme", "Acme")
    assert job.location == "Remote (EMEA); Berlin"  # "remote" zaten yazıyor, tekrar eklenmez
    assert job.text == "Train models"


def _job(title="Data Scientist", location="Istanbul, TR", text="metin", company="Acme"):
    return FetchedJob(
        source="lever:acme",
        external_id=title,
        url="https://example.com",
        company=company,
        title=title,
        location=location,
        text=text,
        posted_at=None,
    )


FILTERS = Filters(
    title_include=[r"\bdata\b", r"\bml\b", "analyst"],
    title_exclude=[r"\bsales\b"],
    location_include=["istanbul", "remote", "emea"],
    location_exclude=["united states", r"\bus\b", "canada"],
)


@pytest.mark.parametrize(
    ("title", "location", "kept"),
    [
        ("Data Scientist", "İstanbul / Maslak", True),
        ("Senior ML Engineer", "Remote, EMEA", True),  # kıdem elenmez
        ("HTML Developer", "Istanbul", False),  # "ml" kelime içinde değil
        ("Sales Data Analyst", "Istanbul", False),
        ("Data Engineer", "Remote, United States", False),
        ("Data Engineer", "Remote, Canada; Remote, Poland", True),  # bir yer uygun
        ("Data Engineer", "Remote, US", False),
        ("Data Engineer", "Austin, Remote", True),
        ("Data Engineer", "Berlin", False),
        ("Data Engineer", None, False),
    ],
)
def test_filters(title, location, kept):
    assert FILTERS.accepts(_job(title=title, location=location)) is kept


def test_collect_filters_dedupes_and_reports_errors():
    config = SourcesConfig(
        filters=FILTERS,
        boards=[
            Board(provider="lever", slug="a", company="Acme"),
            Board(provider="greenhouse", slug="broken", company="Broken"),
            Board(provider="ashby", slug="b", company="Acme"),
        ],
    )
    boards = {
        "a": [_job(), _job(title="Sales Manager"), _job(title="Data Analyst", text="  ")],
        "b": [_job()],  # aynı ilan başka kaynaktan
    }

    def fake_fetch(provider, slug, company):
        if slug == "broken":
            raise ValueError("bozuk yanıt")
        return boards[slug]

    result = collect(config, fetch=fake_fetch)
    assert result.fetched == 4
    assert [j.title for j in result.kept] == ["Data Scientist"]
    assert result.duplicates == 1 and result.empty == 1
    assert result.errors == {"greenhouse:broken": "ValueError: bozuk yanıt"}


def test_jsonl_roundtrip(tmp_path):
    jobs = [_job(), _job(title="ML Engineer", text="Türkçe metin")]
    path = write_jsonl(jobs, tmp_path / "jobs" / "x.jsonl")
    assert read_jsonl(path) == jobs


@pytest.fixture
def db_session():
    """Çalışan PostgreSQL (docker compose up -d, alembic upgrade head) yoksa atlanır."""
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from flyhire.db.session import make_session_factory

    session = make_session_factory()()
    try:
        session.execute(text("select 1 from jobs limit 1"))
    except OperationalError:
        session.close()
        pytest.skip("PostgreSQL çalışmıyor")
    yield session
    session.close()


def test_save_jobs_skips_existing(db_session):
    from sqlalchemy import delete

    from flyhire.db.models import Job
    from flyhire.ingest.store import save_jobs

    jobs = [_job(title="DB test 1", text="pytest-a"), _job(title="DB test 2", text="pytest-b")]
    hashes = [j.content_hash for j in jobs]
    try:
        assert save_jobs(db_session, jobs[:1]) == (1, 0)
        assert save_jobs(db_session, jobs) == (1, 1)
    finally:
        db_session.execute(delete(Job).where(Job.content_hash.in_(hashes)))
        db_session.commit()


def test_extract_jobs_saves_features_and_skips_failures(db_session):
    from sqlalchemy import delete, select, update

    from flyhire.db.models import Job, JobFeatures
    from flyhire.extract.llm import PROMPT_VERSION
    from flyhire.extract.pipeline import extract_jobs, pending_jobs
    from flyhire.extract.schema import JobFeatures as Features
    from flyhire.ingest.store import save_jobs

    fetched = [_job(title="DB pipe ok", text="pytest-ok"), _job(title="DB pipe x", text="pytest-x")]
    hashes = [j.content_hash for j in fetched]
    save_jobs(db_session, fetched)
    ok, bad = (db_session.scalar(select(Job).where(Job.content_hash == h)) for h in hashes)
    features = Features(
        language="en",
        required=[{"any_of": ["SQL"], "quote": "SQL"}],
        preferred=[],
        seniority="unspecified",
        work_mode="remote",
        locations=[],
        flexible_hours=False,
        red_flags=[],
    )

    class Client:
        def create(self, *, messages, **kwargs):
            if "pytest-x" in messages[1]["content"]:
                raise RuntimeError("LLM yanıt vermedi")
            return features

    try:
        # yalnızca bu testin ilanları işlenir; veritabanındaki gerçek ilanlara dokunulmaz
        result = extract_jobs(db_session, [ok, bad], Client(), "test/model")
        assert result.done == 1
        assert list(result.failed) == [bad.id] and "LLM yanıt vermedi" in result.failed[bad.id]
        row = db_session.get(JobFeatures, ok.id)
        assert row.prompt_version == PROMPT_VERSION and row.llm_model == "test/model"
        assert row.data["required"] == [{"any_of": ["SQL"], "quote": "SQL"}]
        # başarısız ilan bir sonraki koşuda yine bekleyenler arasında, başarılı olan değil
        waiting = {j.id for j in pending_jobs(db_session)}
        assert bad.id in waiting and ok.id not in waiting
        # yeniden çıkarımda zaman damgası güncellenir (server_default yalnızca ilk eklemede)
        first = row.extracted_at
        db_session.execute(
            update(JobFeatures)
            .where(JobFeatures.job_id == ok.id)
            .values(extracted_at=first - timedelta(days=1))
        )
        db_session.commit()
        extract_jobs(db_session, [ok], Client(), "test/model")
        db_session.refresh(row)
        assert row.extracted_at > first - timedelta(days=1)
    finally:
        db_session.execute(delete(Job).where(Job.content_hash.in_(hashes)))
        db_session.commit()


def test_embed_jobs_and_cv_similarity(db_session):
    """pgvector'da kosinüs benzerliği; yalnızca bu testin ilanları (job_ids) işlenir."""
    import numpy as np
    from sqlalchemy import delete, select

    from flyhire.db.models import EMBEDDING_DIM, Job, JobFeatures
    from flyhire.extract.embed import embed_jobs, update_similarity
    from flyhire.ingest.store import save_jobs

    class AxisEmbedder:
        """İlan metnindeki etikete göre sabit eksen vektörü."""

        def encode(self, texts):
            rows = np.zeros((len(texts), EMBEDDING_DIM))
            for i, text in enumerate(texts):
                rows[i, 0 if "emb-near" in text else 1] = 1.0
            return rows

    fetched = [_job(title="Emb near", text="emb-near"), _job(title="Emb far", text="emb-far")]
    hashes = [j.content_hash for j in fetched]
    save_jobs(db_session, fetched)
    jobs = [db_session.scalar(select(Job).where(Job.content_hash == h)) for h in hashes]
    ids = [job.id for job in jobs]
    for job in jobs:
        db_session.add(JobFeatures(job_id=job.id, data={}, llm_model="test"))
    db_session.commit()
    try:
        assert embed_jobs(db_session, AxisEmbedder(), job_ids=ids) == 2
        assert embed_jobs(db_session, AxisEmbedder(), job_ids=ids) == 0  # zaten var
        cv = np.zeros(EMBEDDING_DIM)
        cv[0], cv[1] = 0.6, 0.8  # near ile 0.6, far ile 0.8 benzerlik
        assert update_similarity(db_session, cv, job_ids=ids) == 2
        scores = [db_session.get(JobFeatures, i).cv_similarity for i in ids]
        assert scores == pytest.approx([0.6, 0.8])
    finally:
        db_session.execute(delete(Job).where(Job.content_hash.in_(hashes)))
        db_session.commit()


def test_default_config_loads():
    config = SourcesConfig.load()
    assert config.boards and config.filters.title_include
    assert len({(b.provider, b.slug) for b in config.boards}) == len(config.boards)
