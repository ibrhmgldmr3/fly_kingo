"""Connectome'un yüklenmesi.

Shiu verisinde her satır bir pre→post bağlantıdır; `Excitatory x Connectivity` sütunu sinaps sayısı
ile nörotransmitter işaretinin (+1 uyarıcı / -1 baskılayıcı) çarpımıdır. Sinaptik ağırlık
`sign * count * w_syn` olarak hesaplanır (bkz. params.py).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COMPLETENESS_FILE = "Completeness_783.csv"
CONNECTIVITY_FILE = "Connectivity_783.parquet"


@dataclass(frozen=True)
class Connectome:
    flywire_ids: np.ndarray  # (N,) int64, model indeksi → FlyWire kök ID'si
    pre: np.ndarray  # (E,) int64
    post: np.ndarray  # (E,) int64
    signed_count: np.ndarray  # (E,) float32, işaretli sinaps sayısı
    version: str = "v783"

    @property
    def n_neurons(self) -> int:
        return len(self.flywire_ids)

    def index_of(self, ids) -> np.ndarray:
        """FlyWire ID'lerini model indekslerine çevirir. Bilinmeyen ID'de KeyError atar."""
        lookup = {int(fid): i for i, fid in enumerate(self.flywire_ids)}
        return np.array([lookup[int(i)] for i in ids], dtype=np.int64)


def load_connectome(directory: Path) -> Connectome:
    comp = pd.read_csv(directory / COMPLETENESS_FILE, index_col=0)
    con = pd.read_parquet(
        directory / CONNECTIVITY_FILE,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    return Connectome(
        flywire_ids=comp.index.to_numpy(dtype=np.int64),
        pre=con["Presynaptic_Index"].to_numpy(dtype=np.int64),
        post=con["Postsynaptic_Index"].to_numpy(dtype=np.int64),
        signed_count=con["Excitatory x Connectivity"].to_numpy(dtype=np.float32),
    )
