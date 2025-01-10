import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import RiceDataset, MLOPSModel
from torchvision import transforms as T

def train(
    root_dir="data/archive/Rice_Image_Dataset",
    batch_size=32,
    epochs=10,
    lr=1e-3,
    resize=(128, 128),
    save_path="model.pth",
):
    """Train the rice classification model."""
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations and dataset
    transform = T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = RiceDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss function, and optimizer
    model = MLOPSModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if _name_ == "_main_":
    train()