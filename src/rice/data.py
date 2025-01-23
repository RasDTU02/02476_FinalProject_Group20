# data.py
from __future__ import annotations

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

class RiceDataLoader:
    RAW_DATA_PATH = Path("data/raw")
    ZIP_PATH = RAW_DATA_PATH / "rice_dataset.zip"
    DOWNLOAD_URL = "https://www.muratkoklu.com/datasets/vtdhnd09.php"

    def __init__(self, root_dir: str = "data/raw"):
        self.root_dir = root_dir

    def download_dataset(self):
        """Download the rice dataset zip file with a progress bar if it doesn't already exist."""
        if self.ZIP_PATH.exists():
            print("Zip file already exists. Skipping download.")
            return

        print("Downloading the dataset...")
        response = requests.get(self.DOWNLOAD_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(self.ZIP_PATH, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))

        print(f"Dataset downloaded to {self.ZIP_PATH}")

    def extract_data(self):
        """Extract the dataset if not already done."""
        dataset_path = self.RAW_DATA_PATH / "Rice_Image_Dataset"

        if dataset_path.exists():
            print("Dataset already extracted. Skipping extraction.")
            return

        if not self.ZIP_PATH.exists():
            print("Zip file not found. Please download the dataset first.")
            return

        print("Extracting the dataset...")
        with zipfile.ZipFile(self.ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(self.RAW_DATA_PATH)
        print("Extraction complete.")

if __name__ == "__main__":
    data_loader = RiceDataLoader()
    data_loader.download_dataset()
    data_loader.extract_data()

    print("Data preparation complete.")