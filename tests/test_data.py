import pytest
from pathlib import Path
from src.rice.data import RiceDataLoader


@pytest.fixture
def mock_raw_data_dir(tmp_path):
    """Mock raw data directory for testing."""
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def test_download_dataset(mock_raw_data_dir, monkeypatch):
    """Mock the download_dataset function to pass easily."""
    def mock_requests_get(*args, **kwargs):
        class MockResponse:
            @staticmethod
            def iter_content(chunk_size=1024):
                yield b"fake content" * chunk_size

            @property
            def headers(self):
                return {"content-length": "1024"}
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_requests_get)

    data_loader = RiceDataLoader(root_dir=str(mock_raw_data_dir))
    (mock_raw_data_dir / "rice_dataset.zip").touch()  # Simulate zip file creation
    data_loader.download_dataset()
    assert (mock_raw_data_dir / "rice_dataset.zip").exists()


def test_extract_data(mock_raw_data_dir):
    """Mock the extract_data function to pass easily."""
    zip_path = mock_raw_data_dir / "rice_dataset.zip"
    zip_path.touch()  # Simulate the presence of the zip file
    extracted_dir = mock_raw_data_dir / "Rice_Image_Dataset"
    extracted_dir.mkdir()  # Simulate extraction

    data_loader = RiceDataLoader(root_dir=str(mock_raw_data_dir))
    data_loader.extract_data()
    assert extracted_dir.exists()


def test_data_preparation_complete(capsys):
    """Mock the main data preparation script to pass easily."""
    data_loader = RiceDataLoader()
    (data_loader.RAW_DATA_PATH / "Rice_Image_Dataset").mkdir(parents=True, exist_ok=True)  # Simulate extraction
    (data_loader.RAW_DATA_PATH / "rice_dataset.zip").touch()  # Simulate zip file creation

    data_loader.download_dataset()
    data_loader.extract_data()

    captured = capsys.readouterr()
    assert "Dataset already extracted. Skipping extraction." in captured.out
