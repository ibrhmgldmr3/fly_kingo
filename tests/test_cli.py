from typer.testing import CliRunner

from flyhire.cli import app


def test_info_runs():
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "flyhire" in result.output
