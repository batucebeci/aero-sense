from pathlib import Path

from click.testing import CliRunner

from src.__main__ import cli


def test_cli_generate(tmp_path):
    runner = CliRunner()
    output = tmp_path / "synth.csv"
    result = runner.invoke(
        cli, ["generate", "--samples-per-class", "40", "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert output.stat().st_size > 0


def test_cli_train_then_predict(tmp_path):
    runner = CliRunner()
    data_path = tmp_path / "synth.csv"
    model_path = tmp_path / "model.pkl"
    preds_path = tmp_path / "preds.csv"

    r1 = runner.invoke(cli, ["generate", "--samples-per-class", "60", "--output", str(data_path)])
    assert r1.exit_code == 0

    r2 = runner.invoke(cli, ["train", "--data", str(data_path), "--output", str(model_path)])
    assert r2.exit_code == 0, r2.output
    assert model_path.exists()

    r3 = runner.invoke(
        cli,
        [
            "predict",
            "--data",
            str(data_path),
            "--model",
            str(model_path),
            "--output",
            str(preds_path),
        ],
    )
    assert r3.exit_code == 0, r3.output
    assert preds_path.exists()
