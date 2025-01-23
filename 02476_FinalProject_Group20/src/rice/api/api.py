from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from torchvision import transforms as T
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Path to the saved model
MODEL_PATH = Path("models/model.pth")


# Define input schema using Pydantic
class PredictionRequest(BaseModel):
    image: bytes


# Define model inference logic
def load_model():
    """Load the trained PyTorch model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = torch.load(MODEL_PATH)
    model.eval()
    return model


def preprocess_image(image_bytes):
    """Preprocess the uploaded image for inference."""
    transform = T.Compose(
        [T.Resize((128, 128)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    )
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, image_tensor):
    """Perform inference and return predictions."""
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()


# Load the model once
model = load_model()


# API Endpoints
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"message": "API is running"}


@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    """Endpoint to accept an image and return predictions."""
    try:
        image_tensor = preprocess_image(file.file)
        prediction = predict(model, image_tensor)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
