from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from PIL import Image
import numpy as np
from torchvision import transforms as T

app = FastAPI()

MODEL_PATH = "models/model.onnx"

# Load ONNX runtime session
ort_session = ort.InferenceSession(MODEL_PATH)

def preprocess_image(image_bytes):
    """Preprocess image for ONNX inference."""
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_bytes).convert("RGB")
    image = transform(image).unsqueeze(0).numpy()
    return image

@app.post("/predict_onnx/")
async def predict_onnx(file: UploadFile = File(...)):
    """ONNX inference endpoint."""
    try:
        image = preprocess_image(file.file)
        inputs = {"input": image}
        outputs = ort_session.run(None, inputs)
        prediction = np.argmax(outputs[0], axis=1)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def health_check():
    return {"message": "ONNX API is running"}
