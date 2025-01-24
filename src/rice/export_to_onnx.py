import torch
from model import get_pretrained_model  # Adjust to your model's function or class

MODEL_PATH = "models/model.pth"
ONNX_PATH = "models/model.onnx"

def export_to_onnx():
    """Export PyTorch model to ONNX format."""
    model = get_pretrained_model(num_classes=5)  # Adjust the number of classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)  # Adjust dimensions
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"Model exported to ONNX at {ONNX_PATH}")

if __name__ == "__main__":
    export_to_onnx()