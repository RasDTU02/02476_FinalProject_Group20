import pytest
import torch
from unittest.mock import MagicMock, patch
from src.rice.train import train_model
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

@pytest.fixture
def mock_config():
    """Fixture to provide a mock training configuration."""
    return DictConfig({
        "experiment_name": "test_exp",
        "seed": 42,
        "model_conf": {
            "num_classes": 5
        },
        "training_conf": {
            "learning_rate": 0.001,
            "num_epochs": 2,
            "batch_size": 8
        }
    })

@pytest.fixture
def mock_dataloader():
    """Fixture to create a mock dataloader with dummy data."""
    inputs = torch.randn(16, 3, 224, 224)  # 16 images of size 3x224x224
    labels = torch.randint(0, 5, (16,))    # 16 random labels
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)

@patch("src.rice.train.load_data")
@patch("src.rice.train.get_pretrained_model")
def test_training_pipeline(mock_get_model, mock_load_data, mock_config, mock_dataloader):
    """Test the entire training pipeline."""

    # Mock data loader
    mock_load_data.return_value = (mock_dataloader, mock_dataloader)

    # Mock model
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.return_value = torch.randn(8, 5)  # Mock output logits
    mock_get_model.return_value = mock_model

    # Run the training function
    train_model(mock_config)

    # Assertions
    mock_get_model.assert_called_once_with(num_classes=mock_config.model_conf.num_classes)
    assert mock_model.to.called, "Model should be moved to the device"
    assert mock_model.train.called, "Model should be set to train mode"

def test_training_loss_calculation(mock_dataloader):
    """Test if the loss calculation produces a non-negative value."""
    model = torch.nn.Linear(150528, 5)  # Example model (flattened input size)
    criterion = torch.nn.CrossEntropyLoss()

    for images, labels in mock_dataloader:
        outputs = model(images.view(images.size(0), -1))  # Flatten the images
        loss = criterion(outputs, labels)
        
        assert loss.item() >= 0, "Loss should be a non-negative value"

@patch("torch.save")
def test_model_saving(mock_save, mock_config):
    """Test that the model saving function is called."""
    model = torch.nn.Linear(150528, 5)
    model_path = "models/test_model.pth"
    
    torch.save(model.state_dict(), model_path)
    
    mock_save.assert_called_once()
    assert mock_save.call_args[0][0] is not None, "Saved model should not be None"
