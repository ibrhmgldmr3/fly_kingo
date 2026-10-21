"""CV–ilan benzerliği: parçalama, ortalama vektör, CV okuma (modelsiz) ve gerçek model (slow)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from flyhire.config import Settings
from flyhire.extract.embed import chunk_text, embed_document, read_cv

POSTINGS = Path(__file__).parent / "fixtures" / "postings"


class HashEmbedder:
    """Modelsiz, deterministik sahte embedder: aynı metin aynı vektör."""

    dim = 16

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(texts)
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).normal(size=self.dim)
            rows.append(vector / np.linalg.norm(vector))
        return np.array(rows)


def test_chunks_cover_every_word_with_overlap():
    words = [f"w{i}" for i in range(450)]
    chunks = chunk_text(" ".join(words), words=200, overlap=30)
    assert [len(c.split()) for c in chunks] == [200, 200, 110]
    assert chunks[1].split()[0] == "w170"  # 30 kelime örtüşme
    assert {w for c in chunks for w in c.split()} == set(words)


@pytest.mark.parametrize(("n", "count"), [(1, 1), (200, 1), (201, 2)])
def test_chunk_counts_at_boundaries(n, count):
    assert len(chunk_text(" ".join(["x"] * n), words=200, overlap=30)) == count


def test_empty_text_has_no_chunks():
    assert chunk_text("  \n ") == []
    with pytest.raises(ValueError):
        embed_document(HashEmbedder(), "")


def test_document_vector_is_normalized_mean_of_chunks():
    embedder = HashEmbedder()
    text = " ".join(f"w{i}" for i in range(450))
    vector = embed_document(embedder, text)
    (chunks,) = embedder.calls
    expected = embedder.encode(chunks).mean(axis=0)
    np.testing.assert_allclose(vector, expected / np.linalg.norm(expected))
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_read_cv_falls_back_to_example_and_strips_comments(tmp_path):
    example = tmp_path / "cv.example.md"
    example.write_text("<!-- not -->\n# Aday\nPython", encoding="utf-8")
    settings = Settings(cv_path=tmp_path / "yok.md", example_cv_path=example)
    assert read_cv(settings) == ("# Aday\nPython", example, True)
    real = tmp_path / "cv.md"
    real.write_text("Gerçek CV", encoding="utf-8")
    assert read_cv(Settings(cv_path=real, example_cv_path=example)) == ("Gerçek CV", real, False)


def test_repo_example_cv_exists():
    text, _, example = read_cv(Settings(cv_path=Path("yok/cv.md")))
    assert example and "Python" in text and "<!--" not in text


# --- gerçek model (önbellekteki multilingual-e5-large, CPU): pytest -m slow -------------------


@pytest.mark.slow
def test_ml_cv_is_closest_to_ml_posting():
    pytest.importorskip("sentence_transformers")
    from flyhire.extract.embed import E5Embedder

    embedder = E5Embedder()
    cv, _, _ = read_cv(Settings(cv_path=Path("yok/cv.md")))
    cv_vector = embed_document(embedder, cv)
    scores = {
        path.stem: float(cv_vector @ embed_document(embedder, path.read_text(encoding="utf-8")))
        for path in POSTINGS.glob("*.txt")
    }
    # sahte CV junior ML adayı: en yakın ilan junior ML ilanı olmalı
    assert max(scores, key=scores.get) == "tr_remote_junior_ml", scores
