from typer.testing import CliRunner
from src.rice.cli import app  # Import your CLI app

runner = CliRunner()

def test_train_command():
    result = runner.invoke(app, ["train", "--lr", "0.01", "--epochs", "5"])
    assert result.exit_code == 0
    assert "Training complete" in result.stdout

def test_evaluate_command():
    result = runner.invoke(app, ["evaluate", "--model-checkpoint", "models/model.pth"])
    assert result.exit_code == 0
    assert "Evaluation complete" in result.stdout
