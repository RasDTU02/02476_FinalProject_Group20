import torch
import os
from torchvision.models import resnet18  # Update if you are using a different model

# Model and ONNX paths
MODEL_PATH = "models/20250123_232456/rice_model_full.pth"
ONNX_PATH = "models/optimized_rice_model.onnx"

# Define class names
CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# Load the PyTorch model
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

# Export the model to ONNX
def export_to_onnx(model, onnx_path):
    dummy_input = torch.randn(1, 3, 224, 224)  # Dummy input to match model input shape
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    export_to_onnx(model, ONNX_PATH)
