"""FlyWire v783 verisinin indirilmesi ve doğrulanması.

Kaynaklar sabit commit'lere bağlıdır; böylece aynı commit her zaman aynı dosyaları indirir.
SHA-256 değerleri `MANIFEST` içinde tutulur ve her indirmeden sonra kontrol edilir.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx

SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNOTATIONS_COMMIT = "a83b2776d60d5764cef36b927f5f9679c16c47a2"

_SHIU_RAW = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_COMMIT}"
_ANN_RAW = (
    f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOTATIONS_COMMIT}"
    "/supplemental_files"
)


@dataclass(frozen=True)
class DataFile:
    name: str
    url: str
    sha256: str
    description: str


MANIFEST: tuple[DataFile, ...] = (
    DataFile(
        name="Completeness_783.csv",
        url=f"{_SHIU_RAW}/Completeness_783.csv",
        sha256="bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
        description="Modeldeki tüm nöronlar (Shiu, v783)",
    ),
    DataFile(
        name="Connectivity_783.parquet",
        url=f"{_SHIU_RAW}/Connectivity_783.parquet",
        sha256="efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
        description="İşaretli sinaptik bağlantılar (Shiu, v783)",
    ),
    DataFile(
        name="neuron_annotations.tsv",
        url=f"{_ANN_RAW}/Supplemental_file1_neuron_annotations.tsv",
        sha256="b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd",
        description="Nöron tipleri ve etiketleri (Schlegel et al. 2024, v783)",
    ),
)


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(item: DataFile, target_dir: Path, force: bool = False) -> Path:
    """Dosyayı indirir (gerekirse) ve checksum'ı doğrular. Uyuşmazlıkta ValueError atar."""
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / item.name

    if force or not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        with httpx.stream("GET", item.url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        tmp.replace(path)

    if item.sha256:
        actual = sha256_of(path)
        if actual != item.sha256:
            raise ValueError(f"{item.name}: checksum uyuşmuyor (beklenen {item.sha256}, {actual})")
    return path


def missing_files(target_dir: Path) -> list[str]:
    return [item.name for item in MANIFEST if not (target_dir / item.name).exists()]
