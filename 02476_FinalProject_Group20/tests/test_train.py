import pytest
import torch
from src.rice.model import get_pretrained_model
from torch.utils.data import DataLoader, TensorDataset

@pytest.fixture
def model():
    """Fixture to initialize a model."""
    return get_pretrained_model(num_classes=5)

@pytest.fixture
def dataloader():
    """Fixture to create a simple DataLoader."""
    inputs = torch.randn(8, 3, 224, 224)  # Simulating a dataset of 8 RGB images
    labels = torch.randint(0, 5, (8,))   # Random labels for 5 classes
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=4)

def test_training_step(model, dataloader):
    """Test if a single training step runs without errors."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0, "Loss should be greater than 0 after a training step"
        break  # Test only one batch to ensure functionality
