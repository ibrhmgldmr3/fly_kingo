"""Sinek beyninin dışarıya açılan tek arayüzü.

Diğer katmanlar (kodlayıcı, eşik, panel) beyne yalnızca `FlyBrain.simulate` üzerinden erişir:
"şu tat kanallarına şu frekansları ver, şu kadar süre ilerlet, okunan nöronların oranlarını döndür".
"""

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from flyhire.brain import neurons
from flyhire.brain.connectome import Connectome, load_connectome
from flyhire.brain.lif_torch import TorchLIF
from flyhire.brain.params import DEFAULT_PARAMS, LIFParams


@dataclass
class SimulationResult:
    inputs_hz: dict[str, float]
    duration_ms: float
    rates_hz: dict[str, np.ndarray]  # okunan nöron → denemelere göre oranlar (n_trials,)
    # (deneme, okunan nöron sırası, zaman ms); sıra `readout_names` ile eşleşir
    raster: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    readout_names: tuple[str, ...] = ()

    def mean(self, name: str = "mn9") -> float:
        return float(self.rates_hz[name].mean())

    def std(self, name: str = "mn9") -> float:
        return float(self.rates_hz[name].std())


class FlyBrain:
    def __init__(
        self,
        connectome: Connectome,
        params: LIFParams = DEFAULT_PARAMS,
        device: str = "cpu",
    ):
        self.connectome = connectome
        self.model = TorchLIF.from_connectome(connectome, params, device)
        available = connectome.flywire_ids
        self.channels: dict[str, np.ndarray] = {}
        self.missing: dict[str, list[int]] = {}
        for name, ids in neurons.TASTE_CHANNELS.items():
            present, missing = neurons.resolve(ids, available)
            self.channels[name] = connectome.index_of(present)
            self.missing[name] = missing
        self.readout_names = tuple(neurons.READOUT)
        self.readout = connectome.index_of(neurons.READOUT.values())

        # Sürüm etiketi kanal tanımını da içerir: kanallar değişirse eski karar yüzeyi reddedilir.
        digest = hashlib.sha256()
        for name in sorted(self.channels):
            digest.update(name.encode() + self.channels[name].tobytes())
        digest.update(self.readout.tobytes())
        sizes = "-".join(str(len(self.channels[k])) for k in ("sugar", "water", "bitter"))
        self.version = f"shiu-lif/{connectome.version}/grn{sizes}/{digest.hexdigest()[:8]}"

    @classmethod
    def from_directory(cls, directory: Path, **kwargs) -> "FlyBrain":
        return cls(load_connectome(directory), **kwargs)

    def simulate(
        self,
        inputs_hz: dict[str, float],
        duration_ms: float = 1000.0,
        n_trials: int = 30,
        seed: int = 0,
        record_raster: bool = False,
    ) -> SimulationResult:
        """Tat kanallarını (sugar / water / bitter) verilen frekanslarla uyarır.

        Bir kanaldaki tüm GRN'ler aynı frekansla, bağımsız Poisson süreçleriyle uyarılır.
        Verilmeyen kanallar 0 Hz kabul edilir.
        """
        return self.simulate_batch([inputs_hz], duration_ms, n_trials, seed, record_raster)[0]

    def simulate_batch(
        self,
        conditions: list[dict[str, float]],
        duration_ms: float = 1000.0,
        n_trials: int = 30,
        seed: int = 0,
        record_raster: bool = False,
    ) -> list[SimulationResult]:
        """Birden fazla girdi koşulunu tek bir batch'te çalıştırır (karar yüzeyi için)."""
        for inputs in conditions:
            unknown = set(inputs) - set(self.channels)
            if unknown:
                raise ValueError(f"Bilinmeyen tat kanalı: {sorted(unknown)}")

        # Satır sırası: koşul 0'ın denemeleri, koşul 1'in denemeleri, ...
        rates: dict[int, np.ndarray] = {}
        for name, indices in self.channels.items():
            per_row = np.repeat([float(c.get(name, 0.0)) for c in conditions], n_trials)
            for idx in indices:
                rates[int(idx)] = per_row

        run = self.model.run(
            rates,
            duration_ms=duration_ms,
            n_trials=n_trials * len(conditions),
            seed=seed,
            record=self.readout.tolist() if record_raster else None,
        )
        per_row = run.rates_hz()[:, self.readout]

        results = []
        for k, inputs in enumerate(conditions):
            rows = slice(k * n_trials, (k + 1) * n_trials)
            raster = run.raster[(run.raster[:, 0] >= rows.start) & (run.raster[:, 0] < rows.stop)]
            raster[:, 0] -= rows.start
            results.append(
                SimulationResult(
                    inputs_hz=dict(inputs),
                    duration_ms=duration_ms,
                    rates_hz={name: per_row[rows, j] for j, name in enumerate(self.readout_names)},
                    raster=raster,
                    readout_names=self.readout_names,
                )
            )
        return results


@lru_cache(maxsize=1)
def get_brain(directory: str | None = None, device: str = "cpu") -> FlyBrain:
    """Uygulama boyunca tek bir beyin örneği (yükleme birkaç saniye sürer)."""
    from flyhire.config import get_settings

    return FlyBrain.from_directory(Path(directory or get_settings().flywire_dir), device=device)
