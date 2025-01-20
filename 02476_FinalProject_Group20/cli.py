import typer
from pathlib import Path
from src.rice.data import RiceDataLoader
from torch.utils.data import DataLoader
from torchvision import transforms as T
import subprocess

app = typer.Typer()

@app.command()
def train(
    root_dir: str = typer.Option("data/archive/Rice_Image_Dataset", help="Path to the dataset root directory"),
    resize: list[int] = typer.Option([128, 128], help="Resize dimensions for the images (width, height)"),
    subset: int = typer.Option(None, help="Number of samples to use for training (evenly distributed across classes)"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    epochs: int = typer.Option(10, help="Number of epochs for training"),
    lr: float = typer.Option(0.001, help="Learning rate for training"),
    save_path: str = typer.Option("model_parameters/model.pth", help="Path to save the trained model"),
):
    """Train the rice classification model."""
    typer.echo("Initializing training...")

    # Dataset and DataLoader setup
    transform = T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = RiceDataLoader(root_dir=root_dir, transform=transform, subset=subset, resize=resize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    typer.echo("Starting training process using Hydra...")

    # Run the train.py script with Hydra CLI command
    subprocess.run(["python", "-m", "src.rice.train", "+experiment=exp1"])


if __name__ == "__main__":
    app()
