"""Bir GRN kaybının etkisi: kanaldan rastgele bir nöron çıkarıp tam kanalla karşılaştırma.

Shiu listelerindeki iki GRN v783'te farklı kök ID'ye sahip (bkz. docs/BRAIN_VALIDATION.md §1). Bir
nöron kaybının MN9'a etkisini tahmin etmek için kanaldan rastgele bir nöron çıkarılır (N → N−1) ve
aynı girdilerde MN9 oranı karşılaştırılır. Üç farklı nöron için tekrarlanır. N, kanalın o anki
boyutudur (eşleme öncesi 20, sonrası 21).

Kanallar `FlyBrain.channels` geçici olarak değiştirilerek küçültülür; neurons.py değişmez.

    python scripts/grn_ablation.py --device cuda
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

from flyhire.brain.api import FlyBrain

POINTS: dict[str, list[dict[str, float]]] = {
    "sugar": [
        {"sugar": 60.0},
        {"sugar": 100.0},
        {"sugar": 200.0},
        {"sugar": 100.0, "bitter": 30.0},
        {"sugar": 60.0, "water": 80.0},
    ],
    "bitter": [
        {"sugar": 100.0, "bitter": 30.0},
        {"sugar": 100.0, "bitter": 60.0},
    ],
}


@contextlib.contextmanager
def channel_without(brain: FlyBrain, channel: str, position: int):
    """`channel` kanalından `position` sırasındaki nöronu geçici olarak çıkarır."""
    saved = brain.channels
    brain.channels = dict(saved)
    brain.channels[channel] = np.delete(saved[channel], position)
    try:
        yield
    finally:
        brain.channels = saved


def pick_removals(size: int, count: int = 3, first_seed: int = 0) -> list[tuple[int, int]]:
    """Ardışık seed'lerle `count` farklı nöron sırası seçer: [(seed, sıra), ...]."""
    picks: list[tuple[int, int]] = []
    seed = first_seed
    while len(picks) < count:
        position = int(np.random.default_rng(seed).integers(size))
        if position not in {p for _, p in picks}:
            picks.append((seed, position))
        seed += 1
    return picks


def label(point: dict[str, float]) -> str:
    names = {"sugar": "şeker", "water": "su", "bitter": "acı"}
    return " + ".join(f"{names[k]} {v:g}" for k, v in point.items())


def summarize(rates: np.ndarray) -> tuple[float, float]:
    return float(rates.mean()), float(rates.std(ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100, help="simülasyon seed'i")
    parser.add_argument("--out", type=Path, default=Path("data/grn_ablation.json"))
    args = parser.parse_args()

    brain = FlyBrain.from_directory(Path("data/flywire"), device=args.device)
    n = args.trials
    start = time.perf_counter()

    unique = list({json.dumps(p, sort_keys=True): p for ps in POINTS.values() for p in ps}.values())
    baseline = {
        json.dumps(p, sort_keys=True): r.rates_hz["mn9"]
        for p, r in zip(unique, brain.simulate_batch(unique, 1000.0, n, args.seed), strict=True)
    }

    rows = []
    for channel, points in POINTS.items():
        size = len(brain.channels[channel])
        for removal_seed, position in pick_removals(size):
            removed = int(brain.connectome.flywire_ids[brain.channels[channel][position]])
            with channel_without(brain, channel, position):
                assert len(brain.channels[channel]) == size - 1
                results = brain.simulate_batch(points, 1000.0, n, args.seed)
            for point, result in zip(points, results, strict=True):
                full_mean, full_sd = summarize(baseline[json.dumps(point, sort_keys=True)])
                abl_mean, abl_sd = summarize(result.rates_hz["mn9"])
                rows.append(
                    {
                        "channel": channel,
                        "channel_size": size,
                        "removal_seed": removal_seed,
                        "removed_id": removed,
                        "point": label(point),
                        "full_mean": full_mean,
                        "full_sd": full_sd,
                        "ablated_mean": abl_mean,
                        "ablated_sd": abl_sd,
                        "diff": abl_mean - full_mean,
                        "se": float(np.sqrt(full_sd**2 / n + abl_sd**2 / n)),
                    }
                )
    elapsed = time.perf_counter() - start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"device": args.device, "trials": n, "seed": args.seed, "elapsed_s": elapsed}
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")

    print(f"cihaz={args.device}  süre={elapsed:.0f} s")
    print("| Kanal (N) | Çıkarılan | Nokta | N nöron (Hz) | N−1 nöron (Hz) | Fark ± SE (Hz) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['channel']} ({r['channel_size']}) | `{r['removed_id']}` | {r['point']} | "
            f"{r['full_mean']:.1f} ± {r['full_sd']:.1f} | "
            f"{r['ablated_mean']:.1f} ± {r['ablated_sd']:.1f} | "
            f"{r['diff']:+.1f} ± {r['se']:.1f} |"
        )


if __name__ == "__main__":
    main()
