"""Karar yüzeyi: MN9 oranının (şeker, su, acı) frekanslarına göre önceden hesaplanmış tablosu.

Beynin girdisi yalnızca üç frekans olduğu için MN9 oranı 3 boyutlu bir fonksiyondur. Yüzey bir kez
gerçek simülasyonla hesaplanır, günlük kullanımda ızgara noktaları arasında doğrusal interpolasyonla
okunur. Bu bir yaklaşımdır: interpolasyon hatası, ilanların bir alt kümesinde gerçek simülasyonla
ölçülür ve raporlanır (bkz. docs/ROADMAP.md).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

AXES = ("sugar", "water", "bitter")

# Acı, şeker yanıtını çok güçlü baskılıyor (şeker 100 + acı 100 Hz → MN9 ≈ 3 Hz); bu yüzden acı
# ekseni düşük frekanslarda daha sık örneklenir.
DEFAULT_GRID: dict[str, tuple[float, ...]] = {
    "sugar": tuple(float(x) for x in range(0, 201, 20)),
    "water": (0.0, 40.0, 80.0, 120.0, 160.0, 200.0),
    "bitter": (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0),
}


@dataclass
class DecisionSurface:
    grid: dict[str, np.ndarray]
    mean: np.ndarray  # (sugar, water, bitter) → MN9 ortalama oranı (Hz)
    std: np.ndarray  # denemeler arası standart sapma
    done: np.ndarray  # bool, hesaplanmış noktalar (yarım kalan hesap devam ettirilebilir)
    meta: dict

    @property
    def complete(self) -> bool:
        return bool(self.done.all())

    def mn9(self, sugar: float, water: float, bitter: float) -> float:
        """Izgara aralığına kırpılmış doğrusal interpolasyon."""
        if not self.complete:
            raise RuntimeError("Yüzey henüz tamamlanmadı")
        point = [
            np.clip(x, self.grid[a][0], self.grid[a][-1])
            for a, x in zip(AXES, (sugar, water, bitter), strict=True)
        ]
        interp = RegularGridInterpolator([self.grid[a] for a in AXES], self.mean)
        return float(interp(point)[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            **{f"grid_{a}": self.grid[a] for a in AXES},
            mean=self.mean,
            std=self.std,
            done=self.done,
            meta=json.dumps(self.meta),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "DecisionSurface":
        with np.load(path) as f:
            return cls(
                grid={a: f[f"grid_{a}"] for a in AXES},
                mean=f["mean"],
                std=f["std"],
                done=f["done"],
                meta=json.loads(str(f["meta"])),
            )

    @classmethod
    def empty(cls, grid: dict[str, tuple[float, ...]], meta: dict) -> "DecisionSurface":
        shape = tuple(len(grid[a]) for a in AXES)
        return cls(
            grid={a: np.asarray(grid[a], dtype=np.float64) for a in AXES},
            mean=np.zeros(shape),
            std=np.zeros(shape),
            done=np.zeros(shape, dtype=bool),
            meta=meta,
        )


def compute_surface(
    brain,
    path: Path,
    grid: dict[str, tuple[float, ...]] = DEFAULT_GRID,
    n_trials: int = 5,
    duration_ms: float = 1000.0,
    seed: int = 0,
    conditions_per_batch: int = 6,
    progress: Callable[[int, int], None] | None = None,
) -> DecisionSurface:
    """Yüzeyi hesaplar ve her batch'ten sonra `path`'e kaydeder.

    `path` varsa ve ayarlar aynıysa kaldığı yerden devam eder.
    """
    meta = {
        "brain_version": brain.version,
        "n_trials": n_trials,
        "duration_ms": duration_ms,
        "seed": seed,
        "readout": "mn9",
    }
    surface = DecisionSurface.load(path) if path.exists() else DecisionSurface.empty(grid, meta)
    same_grid = all(np.array_equal(surface.grid[a], np.asarray(grid[a])) for a in AXES)
    if surface.meta != meta or not same_grid:
        raise ValueError(f"{path} farklı ayarlarla hesaplanmış; yeni bir dosya adı kullanın")

    todo = [ijk for ijk in product(*(range(len(grid[a])) for a in AXES)) if not surface.done[ijk]]
    total = surface.done.size
    for start in range(0, len(todo), conditions_per_batch):
        chunk = todo[start : start + conditions_per_batch]
        conditions = [
            {a: float(grid[a][i]) for a, i in zip(AXES, ijk, strict=True)} for ijk in chunk
        ]
        # Her batch'in seed'i ilk noktasına bağlı: devam eden hesap aynı sonucu üretir.
        batch_seed = seed * 1_000_003 + int(np.ravel_multi_index(chunk[0], surface.done.shape))
        results = brain.simulate_batch(conditions, duration_ms, n_trials, seed=batch_seed)
        for ijk, result in zip(chunk, results, strict=True):
            surface.mean[ijk] = result.mean("mn9")
            surface.std[ijk] = result.std("mn9")
            surface.done[ijk] = True
        surface.save(path)
        if progress:
            progress(int(surface.done.sum()), total)
    return surface


def interpolation_error(
    brain,
    surface: DecisionSurface,
    n_points: int = 24,
    n_trials: int | None = None,
    seed: int = 12345,
) -> list[dict[str, float]]:
    """Izgara dışındaki rastgele noktalarda yüzeyi gerçek simülasyonla karşılaştırır.

    Her satır: girdiler, `surface` (interpolasyon), `simulated` (ortalama), `simulated_std`.
    """
    rng = np.random.default_rng(seed)
    n_trials = n_trials or int(surface.meta["n_trials"])
    points = [
        {a: float(rng.uniform(surface.grid[a][0], surface.grid[a][-1])) for a in AXES}
        for _ in range(n_points)
    ]
    rows = []
    for start in range(0, n_points, 6):
        chunk = points[start : start + 6]
        results = brain.simulate_batch(
            chunk, surface.meta["duration_ms"], n_trials, seed=seed + start
        )
        for point, result in zip(chunk, results, strict=True):
            rows.append(
                {
                    **point,
                    "surface": surface.mn9(point["sugar"], point["water"], point["bitter"]),
                    "simulated": result.mean("mn9"),
                    "simulated_std": result.std("mn9"),
                }
            )
    return rows
