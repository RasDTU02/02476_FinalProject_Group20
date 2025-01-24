# data.py
from __future__ import annotations

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd

class RiceDataLoader:
    RAW_DATA_PATH = Path("data/raw")
    CSV_FILES_PATH = Path("data/csv_files")  # New folder for CSV files
    ZIP_PATH = RAW_DATA_PATH / "rice_dataset.zip"
    DOWNLOAD_URL = "https://www.muratkoklu.com/datasets/vtdhnd09.php"

    def __init__(self, root_dir: str = "data/raw"):
        self.root_dir = root_dir
        self.CSV_FILES_PATH.mkdir(parents=True, exist_ok=True)  # Create csv_files directory

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

    def create_csv_files(self):
        """Create CSV files for training and test sets from the extracted dataset."""
        dataset_path = self.RAW_DATA_PATH / "Rice_Image_Dataset"
        if not dataset_path.exists():
            print("Dataset not found. Please extract the dataset first.")
            return

        # Collect all image paths and their labels
        data = []
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                label = class_dir.name
                for img_path in class_dir.glob("*.jpg"):
                    data.append((str(img_path), label))

        # Shuffle the data
        import random
        random.shuffle(data)

        # Split into train and test
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        # Create DataFrames
        train_df = pd.DataFrame(train_data, columns=['image_path', 'label'])
        test_df = pd.DataFrame(test_data, columns=['image_path', 'label'])

        # Save to CSV in the new directory
        train_csv_path = self.CSV_FILES_PATH / "train.csv"
        test_csv_path = self.CSV_FILES_PATH / "test.csv"

        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        print(f"Training CSV saved to {train_csv_path}")
        print(f"Test CSV saved to {test_csv_path}")

if __name__ == "__main__":
    data_loader = RiceDataLoader()
    data_loader.download_dataset()
    data_loader.extract_data()
    data_loader.create_csv_files()

    print("Data preparation complete.")