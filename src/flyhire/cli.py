import time
from pathlib import Path

import typer

from flyhire import __version__
from flyhire.brain import data as brain_data
from flyhire.config import get_settings

app = typer.Typer(help="Hortum Uzatma Kariyer Danışmanlığı", no_args_is_help=True)
data_app = typer.Typer(help="Connectome verisi", no_args_is_help=True)
brain_app = typer.Typer(help="Sinek beyni simülasyonu", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(brain_app, name="brain")

DEFAULT_SURFACE = Path("data/surface_v783.npz")


@app.command()
def info() -> None:
    """Sürümü, ayarları ve veri durumunu gösterir."""
    settings = get_settings()
    missing = brain_data.missing_files(settings.flywire_dir)
    typer.echo(f"flyhire {__version__}")
    typer.echo(f"Veri klasörü : {settings.flywire_dir.resolve()}")
    typer.echo(f"Eksik veri   : {', '.join(missing) if missing else 'yok'}")
    typer.echo(f"LLM modeli   : {settings.llm_model}")


@data_app.command("download")
def data_download(
    force: bool = typer.Option(False, help="Var olan dosyaları yeniden indir"),
) -> None:
    """FlyWire v783 verisini indirir ve checksum'ları doğrular."""
    target = get_settings().flywire_dir
    for item in brain_data.MANIFEST:
        typer.echo(f"- {item.name} ({item.description})")
        path = brain_data.download_file(item, target, force=force)
        typer.echo(f"  {path}  sha256={brain_data.sha256_of(path)}")


@brain_app.command("simulate")
def brain_simulate(
    sugar: float = typer.Option(0.0, help="Şeker GRN frekansı (Hz)"),
    water: float = typer.Option(0.0, help="Su GRN frekansı (Hz)"),
    bitter: float = typer.Option(0.0, help="Acı GRN frekansı (Hz)"),
    trials: int = typer.Option(10, help="Deneme sayısı"),
    duration: float = typer.Option(1000.0, help="Sanal süre (ms)"),
    seed: int = typer.Option(0),
) -> None:
    """Tek bir tat girdisi için MN9 oranını hesaplar."""
    from flyhire.brain.api import get_brain

    brain = get_brain()
    start = time.perf_counter()
    result = brain.simulate(
        {"sugar": sugar, "water": water, "bitter": bitter}, duration, trials, seed
    )
    elapsed = time.perf_counter() - start
    for name in result.readout_names:
        typer.echo(f"{name:<11}: {result.mean(name):6.1f} ± {result.std(name):5.1f} Hz")
    typer.echo(f"süre       : {elapsed:.1f} s ({elapsed / trials:.2f} s/deneme)")


@brain_app.command("surface")
def brain_surface(
    out: Path = typer.Option(DEFAULT_SURFACE, help="Çıktı dosyası (yarım kaldıysa devam eder)"),
    trials: int = typer.Option(5, help="Izgara noktası başına deneme sayısı"),
    seed: int = typer.Option(0),
    device: str = typer.Option("cpu", help="cpu veya cuda"),
    batch_conditions: int = typer.Option(
        6, help="Bir batch'teki ızgara noktası sayısı (GPU'da 24+ daha hızlı)"
    ),
) -> None:
    """Şeker × su × acı karar yüzeyini hesaplar."""
    from flyhire.brain.api import get_brain
    from flyhire.brain.surface import compute_surface

    start = time.perf_counter()

    def progress(done: int, total: int) -> None:
        elapsed = time.perf_counter() - start
        typer.echo(f"{done}/{total} nokta  ({elapsed / 60:.1f} dk)")

    surface = compute_surface(
        get_brain(device=device),
        out,
        n_trials=trials,
        seed=seed,
        conditions_per_batch=batch_conditions,
        progress=progress,
    )
    typer.echo(f"Tamamlandı: {out}  (tam: {surface.complete})")


@brain_app.command("check-surface")
def brain_check_surface(
    surface_path: Path = typer.Option(DEFAULT_SURFACE, "--surface"),
    points: int = typer.Option(24, help="Rastgele kontrol noktası sayısı"),
    device: str = typer.Option("cpu", help="cpu veya cuda"),
) -> None:
    """Yüzey interpolasyonunu ızgara dışı noktalarda gerçek simülasyonla karşılaştırır."""
    import numpy as np

    from flyhire.brain.api import get_brain
    from flyhire.brain.surface import DecisionSurface, interpolation_error

    rows = interpolation_error(get_brain(device=device), DecisionSurface.load(surface_path), points)
    typer.echo(f"{'şeker':>6} {'su':>6} {'acı':>6} | {'yüzey':>6} {'sim':>6} {'±':>5}")
    for r in rows:
        typer.echo(
            f"{r['sugar']:6.1f} {r['water']:6.1f} {r['bitter']:6.1f} | "
            f"{r['surface']:6.1f} {r['simulated']:6.1f} {r['simulated_std']:5.1f}"
        )
    err = np.array([r["surface"] - r["simulated"] for r in rows])
    typer.echo(
        f"ortalama mutlak hata: {np.abs(err).mean():.2f} Hz, en büyük: {np.abs(err).max():.2f} Hz"
    )


@brain_app.command("heatmap")
def brain_heatmap(
    surface_path: Path = typer.Option(DEFAULT_SURFACE, "--surface"),
    out: Path = typer.Option(Path("docs/figures/decision_surface.png")),
    water: float = typer.Option(0.0, help="Sabit tutulacak su frekansı (ızgarada olmalı)"),
) -> None:
    """Şeker × acı → MN9 ısı haritasını çizer (README görseli)."""
    from flyhire.brain.plots import plot_sugar_bitter
    from flyhire.brain.surface import DecisionSurface

    plot_sugar_bitter(DecisionSurface.load(surface_path), out, water=water)
    typer.echo(f"Kaydedildi: {out}")


if __name__ == "__main__":
    app()
