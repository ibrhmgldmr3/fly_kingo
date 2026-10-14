import typer

from flyhire import __version__
from flyhire.brain import data as brain_data
from flyhire.config import get_settings

app = typer.Typer(help="Hortum Uzatma Kariyer Danışmanlığı", no_args_is_help=True)
data_app = typer.Typer(help="Connectome verisi", no_args_is_help=True)
app.add_typer(data_app, name="data")


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


if __name__ == "__main__":
    app()
