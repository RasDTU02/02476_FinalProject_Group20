from locust import HttpUser, task, between

class APIUser(HttpUser):
    host = "http://127.0.0.1:8000"  # FastAPI server address
    wait_time = between(1, 3)

    @task
    def predict(self):
        test_image_path = "data/raw/Rice_Image_Dataset/Arborio/Arborio (1).jpg"
        with open(test_image_path, "rb") as file:
            files = {"file": ("Arborio (1).jpg", file, "image/jpeg")}
            response = self.client.post("/predict", files=files)
        assert response.status_code == 200
