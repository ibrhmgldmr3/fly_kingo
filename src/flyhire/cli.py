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


@app.command()
def ingest(
    config: Path = typer.Option(Path("configs/sources.yaml"), exists=True, dir_okay=False),
    out: Path | None = typer.Option(
        None, help="JSONL çıktısı (varsayılan: data/jobs/ats-YYYYMMDD-HHMM.jsonl)"
    ),
    db: bool = typer.Option(False, "--db", help="PostgreSQL'e de yaz (docker compose up -d)"),
) -> None:
    """Herkese açık ATS ilan sayfalarından ilanları toplar, filtreler, kaydeder."""
    from datetime import datetime

    from flyhire.ingest.sources import SourcesConfig
    from flyhire.ingest.store import collect, write_jsonl

    def progress(key: str, fetched: int, kept: int) -> None:
        typer.echo(f"{key:<28} {fetched:>4} ilan, {kept:>3} tutuldu")

    result = collect(SourcesConfig.load(config), progress=progress)
    for key, error in result.errors.items():
        typer.echo(f"{key:<28} HATA: {error}", err=True)

    out = out or Path(f"data/jobs/ats-{datetime.now():%Y%m%d-%H%M}.jsonl")
    write_jsonl(result.kept, out)
    typer.echo(
        f"Toplam {result.fetched} ilan, {len(result.kept)} tutuldu"
        f" ({result.duplicates} tekrar, {result.empty} boş metin ayıklandı) → {out}"
    )
    if db:
        from flyhire.db.session import make_session_factory
        from flyhire.ingest.store import save_jobs

        with make_session_factory()() as session:
            new, existing = save_jobs(session, result.kept)
        typer.echo(f"Veritabanı: {new} yeni, {existing} zaten vardı")


@app.command()
def extract(
    path: Path | None = typer.Argument(
        None, exists=True, dir_okay=False, help="İlan metni (.txt); --db ile verilmez"
    ),
    db: bool = typer.Option(False, "--db", help="Veritabanındaki özelliği olmayan ilanları işle"),
    limit: int | None = typer.Option(None, help="--db: en fazla bu kadar ilan"),
    redo: bool = typer.Option(
        False, help="--db: başka istem sürümü veya modelle çıkarılanları yenile"
    ),
    title: str | None = typer.Option(None),
    company: str | None = typer.Option(None),
    location: str | None = typer.Option(None),
    model: str | None = typer.Option(None, help="Varsayılan: .env'deki LLM_MODEL"),
) -> None:
    """İlanlardan yapılandırılmış özellikleri LLM ile çıkarır.

    Dosya verilirse sonucu JSON olarak yazar; --db ile `jobs` tablosunu `job_features`'a işler.
    """
    from flyhire.extract.llm import extract_features, make_client
    from flyhire.extract.schema import JobPosting

    model = model or get_settings().llm_model
    if db:
        from flyhire.db.session import make_session_factory
        from flyhire.extract.pipeline import extract_jobs, pending_jobs

        def progress(job, error: str | None) -> None:
            status = f"HATA {error}" if error else "tamam"
            typer.echo(f"#{job.id:<5} {job.company[:16]:<16} {job.title[:44]:<44} {status}")

        with make_session_factory()() as session:
            jobs = pending_jobs(session, redo=redo, limit=limit, model=model)
            typer.echo(f"{len(jobs)} ilan bekliyor")
            result = extract_jobs(session, jobs, make_client(model), model, progress=progress)
        typer.echo(f"{result.done} ilan işlendi, {len(result.failed)} hata ({model})")
        return
    if path is None:
        raise typer.BadParameter("ilan dosyası ya da --db verin")

    posting = JobPosting(
        text=path.read_text(encoding="utf-8"), title=title, company=company, location=location
    )
    features = extract_features(posting, client=make_client(model))
    typer.echo(features.model_dump_json(indent=2))


@app.command()
def embed(
    redo: bool = typer.Option(False, help="Tüm ilanların vektörünü yeniden hesapla"),
    top: int = typer.Option(10, help="En benzer kaç ilan gösterilsin"),
) -> None:
    """İlan vektörlerini hesaplar ve CV–ilan benzerliğini `job_features`'a yazar (CPU)."""
    from sqlalchemy import select

    from flyhire.db.models import Job, JobFeatures
    from flyhire.db.session import make_session_factory
    from flyhire.extract.embed import (
        E5Embedder,
        embed_document,
        embed_jobs,
        read_cv,
        update_similarity,
    )

    cv_text, cv_path, example = read_cv()
    typer.echo(f"CV: {cv_path}" + ("  (sahte CV; gerçek CV için private/cv.md)" if example else ""))
    embedder = E5Embedder()
    start = time.perf_counter()
    with make_session_factory()() as session:
        done = embed_jobs(session, embedder, redo=redo)
        typer.echo(f"{done} ilan vektörü hesaplandı ({time.perf_counter() - start:.0f} s)")
        updated = update_similarity(session, embed_document(embedder, cv_text))
        typer.echo(f"{updated} ilanın CV benzerliği güncellendi")
        rows = session.execute(
            select(JobFeatures.cv_similarity, Job.company, Job.title)
            .join(Job)
            .where(JobFeatures.cv_similarity.is_not(None))
            .order_by(JobFeatures.cv_similarity.desc())
        ).all()
    for score, company, title in rows[:top]:
        typer.echo(f"  {score:.3f}  {company[:16]:<16} {title[:60]}")
    if len(rows) > top:
        typer.echo(f"  ... en düşük: {rows[-1][0]:.3f}  {rows[-1][1][:16]:<16} {rows[-1][2][:60]}")


review_app = typer.Typer(help="Özellik çıkarımının elle kontrolü", no_args_is_help=True)
app.add_typer(review_app, name="review")


@review_app.command("export")
def review_export(
    n: int = typer.Option(20, help="Kontrol edilecek ilan sayısı"),
    seed: int = typer.Option(0),
    out: Path = typer.Option(Path("data/review/features.csv")),
) -> None:
    """Rastgele ilanların çıkarılan alanlarını kontrol için CSV'ye yazar (Excel'de açılır)."""
    from flyhire.db.session import make_session_factory
    from flyhire.extract.review import export_review

    with make_session_factory()() as session:
        ids = export_review(session, out, n=n, seed=seed)
    typer.echo(f"{len(ids)} ilan → {out}")
    typer.echo("'dogru' sütununa e (doğru) veya h (yanlış) yazın, sonra: flyhire review score")


@review_app.command("score")
def review_score(
    path: Path = typer.Argument(Path("data/review/features.csv"), exists=True),
) -> None:
    """Doldurulmuş kontrol CSV'sinden alan bazında doğruluk oranını hesaplar."""
    from flyhire.extract.review import score_review

    for line in score_review(path).lines():
        typer.echo(line)


if __name__ == "__main__":
    app()
