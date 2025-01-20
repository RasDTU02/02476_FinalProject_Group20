import pytest
import torch
import torch.nn as nn
from src.rice.model import get_pretrained_model

@pytest.fixture
def model():
    """Fixture to initialize a model with a predefined number of classes."""
    return get_pretrained_model(num_classes=5)

def test_model_initialization(model):
    """Test if the model initializes correctly with the given number of classes."""
    assert isinstance(model, nn.Module), "The model should be an instance of nn.Module"
    assert hasattr(model, "fc"), "Model should have a fully connected layer"

def test_model_forward_pass(model):
    """Test if the model can perform a forward pass with random input data."""
    dummy_input = torch.randn(1, 3, 224, 224)  # Simulating a batch of 1 RGB image of size 224x224
    output = model(dummy_input)
    
    assert output.shape == torch.Size([1, 5]), f"Expected output shape [1,5], got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"

def test_model_training_step(model):
    """Test if the model can run a single training step without errors."""
    dummy_input = torch.randn(4, 3, 224, 224)  # Simulating a batch of 4 images
    dummy_target = torch.randint(0, 5, (4,))  # Random target labels
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_target)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0, "Loss should be greater than zero after a training step"

def test_model_inference(model):
    """Test if the model can run inference mode correctly."""
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
    
    assert torch.argmax(output).item() in range(5), "Output class should be within expected range"

def test_model_saving_and_loading(model, tmp_path):
    """Test saving and loading the model correctly."""
    model_path = tmp_path / "model.pth"
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # Load model
    loaded_model = get_pretrained_model(num_classes=5)
    loaded_model.load_state_dict(torch.load(model_path))
    
    assert torch.equal(model.fc.weight, loaded_model.fc.weight), "Loaded model weights should match the original"
