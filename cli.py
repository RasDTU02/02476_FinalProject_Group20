import typer
from pathlib import Path
import subprocess

app = typer.Typer()

@app.command()
def train(
    root_dir: str = typer.Option(None, help="Path to the dataset root directory"),
    resize: list[int] = typer.Option(None, help="Resize dimensions for the images (width, height)"),
    subset: int = typer.Option(None, help="Number of samples to use for training (evenly distributed across classes)"),
    batch_size: int = typer.Option(None, help="Batch size for training"),
    epochs: int = typer.Option(None, help="Number of epochs for training"),
    lr: float = typer.Option(None, help="Learning rate for training"),
    save_path: str = typer.Option(None, help="Path to save the trained model"),
):
    """Train the rice classification model using default configurations with CLI overrides."""
    typer.echo("Initializing training...")

    typer.echo("Starting training process using Hydra...")

    # Byg kommandoen med kun de parametre, der er angivet af brugeren
    command = ["python", "-m", "src.rice.train", "+experiment=exp1"]
    
    if root_dir is not None:
        command.extend([f"data.root_dir={root_dir}"])
    if resize is not None:
        command.extend([f"data.resize[0]={resize[0]}", f"data.resize[1]={resize[1]}"])
    if subset is not None:
        command.extend([f"data.subset={subset}"])
    if batch_size is not None:
        command.extend([f"training_conf.batch_size={batch_size}"])
    if epochs is not None:
        command.extend([f"training_conf.num_epochs={epochs}"])
    if lr is not None:
        command.extend([f"training_conf.learning_rate={lr}"])
    if save_path is not None:
        command.extend([f"training.save_path={save_path}"])
    
    subprocess.run(command)

if __name__ == "__main__":
    app()