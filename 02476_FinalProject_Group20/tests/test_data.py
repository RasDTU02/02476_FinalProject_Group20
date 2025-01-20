import pytest
from pathlib import Path
from src.rice.data import RiceDataLoader
from unittest.mock import patch, MagicMock
import shutil

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

def test_rice_dataloader_initialization(sample_data_dir):
    """Test the initialization of the RiceDataLoader with sample data."""
    data_loader = RiceDataLoader(root_dir=str(sample_data_dir), sample_ratio=1.0)

    assert len(data_loader) == 3
    assert isinstance(data_loader.image_paths, list)
    assert data_loader.class_to_idx["Class1"] == 0 or data_loader.class_to_idx["Class2"] == 1

def test_rice_dataloader_getitem(sample_data_dir):
    """Test the __getitem__ method."""
    data_loader = RiceDataLoader(root_dir=str(sample_data_dir), sample_ratio=1.0)

    image, label = data_loader[0]
    assert image is not None
    assert isinstance(label, int)

def test_download_dataset(mocker):
    """Test downloading the dataset."""
    data_loader = RiceDataLoader()
    mock_requests_get = mocker.patch("requests.get", return_value=MagicMock(status_code=200, headers={"content-length": "100"}))

    with patch("builtins.open", mocker.mock_open()) as mock_file:
        with patch("tqdm.tqdm") as mock_tqdm:
            data_loader.download_dataset()

            mock_requests_get.assert_called_once_with(data_loader.DOWNLOAD_URL, stream=True)
            mock_file.assert_called_once()
            mock_tqdm.assert_called_once()

def test_extract_and_sample_data(sample_data_dir, mocker):
    """Test the extract_and_sample_data method."""
    data_loader = RiceDataLoader()
    mocker.patch("zipfile.ZipFile.extractall")
    mocker.patch("random.sample", side_effect=lambda seq, num: seq[:num])

    data_loader.RAW_DATA_PATH = sample_data_dir / "data/raw"
    data_loader.PROCESSED_DATA_PATH = sample_data_dir / "data/processed"
    
    dataset_path = data_loader.RAW_DATA_PATH / "Rice_Image_Dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    class_dir = dataset_path / "ClassA"
    class_dir.mkdir()
    (class_dir / "imageA.jpg").touch()
    (class_dir / "imageB.jpg").touch()

    data_loader.extract_and_sample_data()

    processed_class_path = data_loader.PROCESSED_DATA_PATH / "Rice_Image_Dataset/ClassA"
    assert processed_class_path.exists()
    assert len(list(processed_class_path.iterdir())) > 0

def test_len_method(sample_data_dir):
    """Test the length of the dataset."""
    data_loader = RiceDataLoader(root_dir=str(sample_data_dir), sample_ratio=1.0)
    assert len(data_loader) == 3

def test_transform_application(sample_data_dir):
    """Test the transform function on images."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    data_loader = RiceDataLoader(root_dir=str(sample_data_dir), transform=transform)
    image, _ = data_loader[0]

    assert image.shape == (3, 128, 128)

def test_handle_missing_dataset(mocker):
    """Test behavior when dataset directory is missing."""
    data_loader = RiceDataLoader()
    
    with pytest.raises(FileNotFoundError):
        data_loader.extract_and_sample_data()

def test_invalid_sample_ratio():
    """Test behavior when an invalid sample ratio is provided."""
    with pytest.raises(ValueError):
        RiceDataLoader(sample_ratio=-0.5)

def test_dataset_subset(sample_data_dir):
    """Test using a subset of the dataset."""
    data_loader = RiceDataLoader(root_dir=str(sample_data_dir), sample_ratio=1.0, subset=1)
    assert len(data_loader) == 1
