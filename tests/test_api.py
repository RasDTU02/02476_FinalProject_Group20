from fastapi.testclient import TestClient
from src.rice.api import app
import os

client = TestClient(app)

# Path to the test image
TEST_IMAGE_PATH = "data/raw/Rice_Image_Dataset/Arborio/Arborio (1).jpg"

def test_predict_valid_file():
    # Ensure the test image exists
    assert os.path.exists(TEST_IMAGE_PATH), f"Test image not found at {TEST_IMAGE_PATH}"

    # Open the test image and send it to the API
    with open(TEST_IMAGE_PATH, "rb") as test_file:
        response = client.post(
            "/predict",
            files={"file": (os.path.basename(TEST_IMAGE_PATH), test_file, "image/jpeg")},
        )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

def test_predict_invalid_file():
    # Test with a non-image file
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"This is not an image", "text/plain")},
    )
    assert response.status_code == 500
    assert "detail" in response.json()
