# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import get_pretrained_model
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image

from datetime import datetime
from pathlib import Path

# Define paths
PROCESSED_DATA_PATH = Path("data/raw/Rice_Image_Dataset")

def load_data(max_images: int = None):
    """Load processed JPG images and optionally limit the number of images."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data path '{PROCESSED_DATA_PATH}' does not exist.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Filtrér kun mapper som ikke er tekstfiler eller skjulte filer
    class_dirs = [d for d in PROCESSED_DATA_PATH.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for class_idx, class_dir in enumerate(class_dirs):
        print(f"Processing class {class_idx}: {class_dir.name}")
        images = list(class_dir.glob("*.jpg"))
        if len(images) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue

        if max_images:
            images = images[:max_images]  # Begræns antallet af billeder pr. klasse

        split_idx = int(0.8 * len(images))  # 80% train, 20% test
        for i, img_path in enumerate(images):
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            if i < split_idx:
                train_data.append(img)
                train_labels.append(class_idx)
            else:
                test_data.append(img)
                test_labels.append(class_idx)

    if not train_data:
        raise RuntimeError("No training data loaded. Check your processed data folder.")

    print("Unique train labels:", set(train_labels))
    print("Unique test labels:", set(test_labels))

    train_set = torch.utils.data.TensorDataset(torch.stack(train_data), torch.tensor(train_labels))
    test_set = torch.utils.data.TensorDataset(torch.stack(test_data), torch.tensor(test_labels))

    return DataLoader(train_set, batch_size=32, shuffle=True), DataLoader(test_set, batch_size=32, shuffle=False)


def train_model(max_images: int = None):
    """Train the pre-trained model on the rice dataset."""
    train_loader, test_loader = load_data(max_images=max_images)
    model = get_pretrained_model(num_classes=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 2
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
    
    print("Training complete.")
    
    # Opret stien til models mappen
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    
    # Opret en undermappe med dato og tid for denne kørsel
    now = datetime.now()
    run_folder = now.strftime("%Y%m%d_%H%M%S")
    run_path = models_path / run_folder
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Gem modellen i den nye mappe
    model_path = run_path / "rice_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

if __name__ == "__main__":
    # Brug x billeder pr. klasse til træning/test
    train_model(max_images=20)