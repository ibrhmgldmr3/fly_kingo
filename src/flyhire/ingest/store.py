"""Toplanan ilanların saklanması: JSONL anlık görüntü ve PostgreSQL.

JSONL dosyası veritabanı olmadan da çalışır; elle kontrol ve tekrarlanabilirlik için ham hâli
tutar. Veritabanına yazarken tekrar ayıklama `content_hash` ile yapılır: aynı ilan tekrar
gelirse yeni satır açılmaz.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from flyhire.db.models import Job
from flyhire.ingest.ats import FetchedJob, fetch_board
from flyhire.ingest.sources import SourcesConfig


@dataclass
class CollectResult:
    fetched: int = 0
    kept: list[FetchedJob] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # "lever:slug" -> hata
    duplicates: int = 0  # aynı çalıştırmada birden fazla kaynaktan gelen ilan
    empty: int = 0  # filtreden geçen ama metni boş ilan (LLM'e verilecek bir şey yok)


def collect(
    config: SourcesConfig,
    fetch: Callable[..., list[FetchedJob]] = fetch_board,
    progress: Callable[[str, int, int], None] | None = None,
) -> CollectResult:
    """Tüm board'ları çeker, filtreler ve aynı çalıştırma içindeki tekrarları ayıklar.

    Bir board'daki hata diğerlerini durdurmaz; `errors` içinde raporlanır.
    """
    result = CollectResult()
    seen: set[str] = set()
    for board in config.boards:
        key = f"{board.provider}:{board.slug}"
        try:
            jobs = fetch(board.provider, board.slug, board.company)
        except (httpx.HTTPError, ValueError, KeyError) as error:
            result.errors[key] = f"{type(error).__name__}: {error}"
            continue
        kept = 0
        for job in jobs:
            result.fetched += 1
            if not config.filters.accepts(job):
                continue
            if not job.text.strip():
                result.empty += 1
                continue
            if job.content_hash in seen:
                result.duplicates += 1
                continue
            seen.add(job.content_hash)
            result.kept.append(job)
            kept += 1
        if progress:
            progress(key, len(jobs), kept)
    return result


def write_jsonl(jobs: Iterable[FetchedJob], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(job.model_dump_json() + "\n")
    return path


def read_jsonl(path: Path) -> list[FetchedJob]:
    with path.open(encoding="utf-8") as f:
        return [FetchedJob.model_validate(json.loads(line)) for line in f if line.strip()]


def save_jobs(session: Session, jobs: Iterable[FetchedJob]) -> tuple[int, int]:
    """İlanları `jobs` tablosuna yazar. Dönüş: (yeni, zaten var olan)."""
    rows = [
        {
            "source": job.source,
            "url": job.url,
            "company": job.company[:255],
            "title": job.title[:255],
            "location": job.location[:255] if job.location else None,
            "text": job.text,
            "content_hash": job.content_hash,
            "posted_at": job.posted_at,
        }
        for job in jobs
    ]
    if not rows:
        return 0, 0
    hashes = [row["content_hash"] for row in rows]
    before = set(session.scalars(select(Job.content_hash).where(Job.content_hash.in_(hashes))))
    statement = insert(Job).values(rows).on_conflict_do_nothing(index_elements=["content_hash"])
    session.execute(statement)
    session.commit()
    new = len({h for h in hashes if h not in before})
    return new, len(rows) - new
