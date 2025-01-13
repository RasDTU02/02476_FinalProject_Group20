import typer
from pathlib import Path
from src.rice.train import train_model
from src.rice.data import RiceDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

app = typer.Typer()

@app.command()
def train(
    root_dir: str = typer.Option("data/archive/Rice_Image_Dataset", help="Path to the dataset root directory"),
    resize: tuple = typer.Option((128, 128), help="Resize dimensions for the images (width, height)"),
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
    dataset = RiceDataset(root_dir=root_dir, transform=transform, subset=subset, resize=resize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Start training
    train_model(dataloader, epochs, lr, save_path)
    typer.echo(f"Training completed. Model saved at {save_path}")

if __name__ == "__main__":
    app()
