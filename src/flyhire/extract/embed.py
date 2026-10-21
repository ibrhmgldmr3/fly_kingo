"""CV–ilan anlamsal benzerliği (doküman §4.2): çok dilli embedding ve kosinüs benzerliği.

LLM'den ayrı, ölçülebilir bir adım: LLM'e CV verilmez; benzerlik yalnızca embedding'lerden
hesaplanır. Model `EMBEDDING_MODEL` (varsayılan `intfloat/multilingual-e5-large`, 1024 boyut).

e5 en fazla 512 token okur, ilanlar çoğunlukla daha uzundur: metin kelime sınırıyla örtüşen
parçalara bölünür, parçaların vektörlerinin ortalaması (yeniden normalize) metnin vektörüdür.
e5'in benzerlik görevleri için önerdiği "query: " öneki CV'ye de ilana da eklenir.

Hesap CPU'da yapılır ve çekirdeklerin yarısıyla sınırlanır (makine kilitlenmesin).
"""

import os
import re
from pathlib import Path
from typing import Protocol

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from flyhire.config import Settings, get_settings
from flyhire.db.models import Job, JobFeatures

# Türkçede kelime başına token sayısı yüksek; 200 kelime e5'in 512 token sınırının altında kalır.
CHUNK_WORDS = 200
CHUNK_OVERLAP = 30
PREFIX = "query: "


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Satır başına bir normalize vektör."""
        ...


class E5Embedder:
    def __init__(self, model: str | None = None, threads: int | None = None):
        self.model_name = model or get_settings().embedding_model
        self.threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(self.threads)
            try:  # önbellekteyse ağa sorma (yüklemeyi ~10 s kısaltıyor)
                self._model = SentenceTransformer(
                    self.model_name, device="cpu", local_files_only=True
                )
            except OSError:
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model.encode(
            [PREFIX + text for text in texts],
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


def chunk_text(text: str, words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, words - overlap)
    return [" ".join(tokens[i : i + words]) for i in range(0, max(1, len(tokens) - overlap), step)]


def embed_document(embedder: Embedder, text: str) -> np.ndarray:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("boş metin")
    mean = embedder.encode(chunks).mean(axis=0)
    return mean / np.linalg.norm(mean)


def read_cv(settings: Settings | None = None) -> tuple[str, Path, bool]:
    """(metin, yol, sahte_mi). Gerçek CV yoksa repodaki sahte CV kullanılır."""
    settings = settings or get_settings()
    path, example = settings.cv_path, False
    if not path.exists():
        path, example = settings.example_cv_path, True
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    return text.strip(), path, example


def embed_jobs(
    session: Session, embedder: Embedder, redo: bool = False, job_ids: list[int] | None = None
) -> int:
    """Vektörü olmayan (veya `redo` ile tüm) ilanların vektörünü hesaplar; her ilan ayrı commit.

    `job_ids` verilirse yalnızca o ilanlar (testler gerçek ilanlara dokunmasın diye).
    """
    query = select(Job).order_by(Job.id)
    if not redo:
        query = query.where(Job.embedding.is_(None))
    if job_ids is not None:
        query = query.where(Job.id.in_(job_ids))
    done = 0
    for job in session.scalars(query):
        parts = (job.title, job.company, job.location, job.text)
        posting = "\n".join(part for part in parts if part)
        job.embedding = embed_document(embedder, posting).tolist()
        session.commit()
        done += 1
    return done


def update_similarity(
    session: Session, cv_vector: np.ndarray, job_ids: list[int] | None = None
) -> int:
    """Özelliği çıkarılmış ve vektörü olan ilanlarda `cv_similarity`'yi yeniden hesaplar.

    Kosinüs benzerliği pgvector'da hesaplanır (`1 - kosinüs uzaklığı`); CV değişince yalnızca bu
    adım tekrar çalışır, ilan vektörleri yeniden hesaplanmaz.
    """
    statement = (
        update(JobFeatures)
        .where(JobFeatures.job_id == Job.id, Job.embedding.is_not(None))
        .values(cv_similarity=1 - Job.embedding.cosine_distance(cv_vector.tolist()))
    )
    if job_ids is not None:
        statement = statement.where(Job.id.in_(job_ids))
    result = session.execute(statement)
    session.commit()
    return result.rowcount
