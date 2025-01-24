# models.py
import torch.nn as nn
from torchvision import models


def get_pretrained_model(num_classes: int): # we are using the ResNet18 model
    """Return a lightweight pre-trained model (ResNet18)
    modified for the rice dataset."""
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
