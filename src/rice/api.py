from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Dict
import io
from torchvision.models import ResNet
# To use the API website use
# http://127.0.0.1:8000/docs#/default/predict_rice_predict_post

# Initialize FastAPI app
app = FastAPI()

# Define class names for rice categories
CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# Define the model path (use the latest saved model)
MODEL_PATH = "models/20250123_232456/rice_model_full.pth"

# Load the PyTorch model
def load_model(model_path: str):
    #torch.serialization.add_safe_globals([ResNet])
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()  # Set model to evaluation mode
    return model

# Preprocess image
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match model input
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize values
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load the model
model = load_model(MODEL_PATH)

# Define inference route
@app.post("/predict")
async def predict_rice(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")  # Ensure 3 channels

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()

        # Return prediction
        return {"prediction": CLASS_NAMES[predicted_class]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
