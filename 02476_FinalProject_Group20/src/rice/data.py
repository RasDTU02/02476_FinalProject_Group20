from __future__ import annotations

import os
import requests
import zipfile
import random
from pathlib import Path
from tqdm import tqdm
from shutil import copy2

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
ZIP_PATH = RAW_DATA_PATH / "rice_dataset.zip"
DOWNLOAD_URL = "https://www.muratkoklu.com/datasets/vtdhnd09.php"

def download_rice_dataset():
    """Download the rice dataset zip file with a progress bar if it doesn't already exist."""
    if ZIP_PATH.exists():
        print("Zip file already exists. Skipping download.")
        return

    print("Downloading the dataset...")
    response = requests.get(DOWNLOAD_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(ZIP_PATH, "wb") as f, tqdm(
        desc="Downloading", total=total_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))

    print(f"Dataset downloaded to {ZIP_PATH}")

def extract_and_sample_data(sample_ratio: float = 0.1): # 10% of the data is sampled
    """Extract the dataset and sample a fraction of images for faster processing if not already done."""
    dataset_path = RAW_DATA_PATH / "Rice_Image_Dataset"
    if dataset_path.exists() and (PROCESSED_DATA_PATH / "Rice_Image_Dataset").exists():
        print("Dataset already extracted and sampled. Skipping extraction and sampling.")
        return

    if not dataset_path.exists():
        if ZIP_PATH.exists():
            print("Extracting the dataset...")
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(RAW_DATA_PATH)
            print("Extraction complete.")
        else:
            print("Zip file not found. Please download the dataset first.")
            return

    print("Sampling images...")
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            images = list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpg"))
            sampled_images = random.sample(images, max(1, int(len(images) * sample_ratio)))

            dest_class_folder = PROCESSED_DATA_PATH / "Rice_Image_Dataset" / class_folder.name
            dest_class_folder.mkdir(parents=True, exist_ok=True)

            for img_path in sampled_images:
                copy2(img_path, dest_class_folder)

    print("Sampling and processing complete.")

if __name__ == "__main__":
    download_rice_dataset()
    extract_and_sample_data(sample_ratio=0.1)
    print("Data preparation complete.")
