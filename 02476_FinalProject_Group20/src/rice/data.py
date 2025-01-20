from __future__ import annotations

import os
import requests
import zipfile
import random
from pathlib import Path
from tqdm import tqdm
from shutil import move
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RiceDataLoader(Dataset):
    RAW_DATA_PATH = Path("data/raw")
    PROCESSED_DATA_PATH = Path("data/processed")
    ZIP_PATH = RAW_DATA_PATH / "rice_dataset.zip"
    DOWNLOAD_URL = "https://www.muratkoklu.com/datasets/vtdhnd09.php"

    def __init__(self, root_dir: str = "data/raw", sample_ratio: float = 0.001, transform=None, subset: int = None,resize: tuple = (128, 128)):
        self.root_dir = root_dir
        self.sample_ratio = sample_ratio
        self.resize = resize
        self.transform = transform
        self.dataset_path = self.PROCESSED_DATA_PATH / "Rice_Image_Dataset"
        self.image_paths = list(self.dataset_path.glob("*/*.png")) + list(self.dataset_path.glob("*/*.jpg"))
        
        if subset and subset < len(self.image_paths):
            self.image_paths = random.sample(self.image_paths, subset)
        
        self.labels = [p.parent.name for p in self.image_paths]
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(set(self.labels))}
        self.labels = [self.class_to_idx[label] for label in self.labels]
    def download_dataset(self):
        """Download the rice dataset zip file with a progress bar if it doesn't already exist."""
        if self.ZIP_PATH.exists():
            print("Zip file already exists. Skipping download.")
            return

        print("Downloading the dataset...")
        response = requests.get(self.DOWNLOAD_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(self.ZIP_PATH, "wb") as f, tqdm(
            desc="Downloading", total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))

        print(f"Dataset downloaded to {self.ZIP_PATH}")

    def extract_and_sample_data(self):
        """Extract the dataset and sample a fraction of images for faster processing if not already done."""
        dataset_path = self.RAW_DATA_PATH / "Rice_Image_Dataset"
        processed_path = self.PROCESSED_DATA_PATH / "Rice_Image_Dataset"

        if processed_path.exists():
            print("Processed data already exists. Skipping extraction and sampling.")
            return

        if not dataset_path.exists():
            if self.ZIP_PATH.exists():
                print("Extracting the dataset...")
                with zipfile.ZipFile(self.ZIP_PATH, "r") as zip_ref:
                    zip_ref.extractall(self.RAW_DATA_PATH)
                print("Extraction complete.")
            else:
                print("Zip file not found. Please download the dataset first.")
                return

        print("Sampling images...")
        for class_folder in tqdm(dataset_path.iterdir(), desc="Sampling classes"):
            if class_folder.is_dir():
                images = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg"))
                sampled_images = random.sample(images, max(1, int(len(images) * self.sample_ratio)))

                dest_class_folder = processed_path / class_folder.name
                dest_class_folder.mkdir(parents=True, exist_ok=True)

                for img_path in sampled_images:
                    move(img_path, dest_class_folder)

        print("Sampling and processing complete.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    data_loader = RiceDataLoader(sample_ratio=0.01, transform=transform)
    data_loader.download_dataset()
    data_loader.extract_and_sample_data()

    train_loader = DataLoader(data_loader, batch_size=32, shuffle=True)
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
    print("Data preparation complete.")
