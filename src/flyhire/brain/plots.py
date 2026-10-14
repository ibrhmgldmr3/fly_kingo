from pathlib import Path

import numpy as np

from flyhire.brain.surface import DecisionSurface


def plot_sugar_bitter(surface: DecisionSurface, out: Path, water: float = 0.0) -> None:
    """Sabit su frekansında şeker × acı → MN9 ısı haritası."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    water_axis = surface.grid["water"]
    matches = np.flatnonzero(np.isclose(water_axis, water))
    if not len(matches):
        raise ValueError(f"su={water} Hz ızgarada yok; seçenekler: {water_axis.tolist()}")
    k = int(matches[0])
    if not surface.done[:, k, :].all():
        raise RuntimeError("Bu kesit henüz tamamlanmadı")

    sugar, bitter = surface.grid["sugar"], surface.grid["bitter"]
    rates = surface.mean[:, k, :]  # (şeker, acı)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    # Acı ekseni eşit aralıklı değil; hücreleri sıra numarasıyla çizip etiketleri gerçek değer yap.
    im = ax.imshow(rates, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(bitter)), [f"{b:g}" for b in bitter])
    ax.set_yticks(range(len(sugar)), [f"{s:g}" for s in sugar])
    ax.set_xlabel("Acı GRN frekansı (Hz)")
    ax.set_ylabel("Şeker GRN frekansı (Hz)")
    title = "Sineğin karar yüzeyi: MN9 ateşleme oranı"
    ax.set_title(f"{title}\n(su = {water:g} Hz, {surface.meta['n_trials']} deneme/nokta)")
    fig.colorbar(im, ax=ax, label="MN9 (Hz)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
