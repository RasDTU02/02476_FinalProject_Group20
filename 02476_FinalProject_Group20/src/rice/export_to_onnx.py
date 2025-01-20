import torch
from model import MLOPSModel  # Adjust to your model's class and path

MODEL_PATH = "models/model.pth"
ONNX_PATH = "models/model.onnx"

def export_to_onnx():
    model = MLOPSModel()  # Adjust to your model class
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)  # Adjust dimensions
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"Model exported to ONNX at {ONNX_PATH}")

if __name__ == "__main__":
    export_to_onnx()
