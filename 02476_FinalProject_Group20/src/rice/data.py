from pathlib import Path

import typer
from torch.utils.data import Dataset


import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from typing import Optional, Tuple, List


class RiceDataset(Dataset):
    """Custom dataset for rice classification."""

    def __init__(
        self,
        root_dir: str ="data/archive/Rice_Image_Dataset",
        transform: Optional[T.compose]=None,
        subset: int = None,
        resize: tuple = (128, 128),
    ):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the root directory containing the class folders.
            transform (callable, optional): Transformation to apply to images. Defaults to None.
            subset (int, optional): Number of total samples to load (evenly distributed across classes). Defaults to None.
            resize (tuple): Desired dimensions (width, height) to resize the images. Defaults to (128, 128).
        """
        self.root_dir = root_dir
        self.transform = transform or T.Compose(
            [T.Resize(resize), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
        )
        self.subset = subset
        self.resize = resize
        self.samples = []
        self.class_to_idx = {}
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset samples and optionally subset data."""
        # Map class names to indices and load samples
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                class_samples = [
                    os.path.join(class_path, fname)
                    for fname in os.listdir(class_path)
                    if fname.endswith(".jpg")
                ]

                # Subsample evenly from each class if subset is provided
                if self.subset:
                    per_class_count = self.subset // len(self.class_to_idx)
                    class_samples = random.sample(
                        class_samples, min(per_class_count, len(class_samples))
                    )

                self.samples.extend([(sample, idx) for sample in class_samples])

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return a given sample from the dataset."""
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # Example usage
    root_dir = "data/archive/Rice_Image_Dataset"
    resize = (64, 64)  # Change to (128, 128) or other dimensions as needed
    subset = 100  # Total number of samples to load (evenly split across classes)

    transform = T.Compose(
        [T.Resize(resize), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    )

    # Load the dataset with optional resizing and subsampling
    dataset = RiceDataset(
        root_dir=root_dir, transform=transform, subset=subset, resize=resize
    )

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Inspect dataset
    for images, labels in dataloader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels: {labels}")
        break
