import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

from src.rice.data import (
    download_rice_dataset,
    extract_and_sample_data,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    ZIP_PATH,
)

class TestDataFunctions(unittest.TestCase):

    def test_download_rice_dataset(self):
        """Test that the rice dataset is downloaded correctly."""
        # Mock requests to avoid actual HTTP calls
        with patch("src.rice.data.requests") as mock_requests:
            mock_requests.get.return_value.status_code = 200
            mock_requests.get.return_value.iter_content.return_value = [b"data"] * 10

            # Ensure the function doesn't delete existing files
            if not ZIP_PATH.exists():
                download_rice_dataset()
                mock_requests.get.assert_called_once()
            else:
                print("Zip file already exists. Skipping test for download.")

    def test_extract_and_sample_data(self):
        """Test that the extraction and sampling of data works correctly."""
        with patch("src.rice.data.zipfile.ZipFile") as mock_zipfile:
            mock_zipfile.return_value.extractall.return_value = None

            # Ensure function does not overwrite existing processed data
            if not PROCESSED_DATA_PATH.exists():
                extract_and_sample_data()
                mock_zipfile.assert_called_once()
            else:
                print("Processed data already exists. Skipping test for extraction.")

    def test_processed_data_exists(self):
        """Check if processed data exists in the expected location."""
        self.assertTrue(
            PROCESSED_DATA_PATH.exists(), "Processed data path does not exist."
        )
        classes = list(PROCESSED_DATA_PATH.glob("*"))
        self.assertGreater(len(classes), 0, "No processed classes found.")
