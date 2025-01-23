import pytest
from pathlib import Path
from src.rice.data import RiceDataLoader


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a mock processed dataset directory with fake images for testing."""
    processed_dir = tmp_path / "data/processed/Rice_Image_Dataset"
    processed_dir.mkdir(parents=True, exist_ok=True)

    class1_dir = processed_dir / "Class1"
    class1_dir.mkdir()
    (class1_dir / "image1.jpg").touch()
    (class1_dir / "image2.jpg").touch()

    class2_dir = processed_dir / "Class2"
    class2_dir.mkdir()
    (class2_dir / "image1.jpg").touch()

    return tmp_path


def test_rice_dataloader_getitem(sample_data_dir):
    """Test the __getitem__ method."""
    data_loader = RiceDataLoader(root_dir=str(sample_data_dir), sample_ratio=1.0)

    image, label = data_loader[0]
    assert image is not None
    assert isinstance(label, int)


def test_transform_application(sample_data_dir):
    """Test the transform function on images."""
    from torchvision import transforms

    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    data_loader = RiceDataLoader(root_dir=str(sample_data_dir), transform=transform)
    image, _ = data_loader[0]

    assert image.shape == (3, 128, 128)


def test_dataset_subset(sample_data_dir):
    """Test using a subset of the dataset."""
    data_loader = RiceDataLoader(
        root_dir=str(sample_data_dir), sample_ratio=1.0, subset=1
    )
    assert len(data_loader) == 1
