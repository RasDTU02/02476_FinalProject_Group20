# models.py
from torchvision import models
import torch.nn as nn

def get_pretrained_model(num_classes: int):
    """Return a lightweight pre-trained model (ResNet18) modified for the rice dataset."""
    model = models.resnet18(pretrained=True)  # Load pre-trained ResNet18
    # Replace the final fully connected layer for our dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
