import unittest
import torch
from src.rice.model import RiceModel  # Replace with your actual model class


class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up for model tests."""
        self.model = RiceModel(num_classes=5)  # Assuming 5 rice classes
        self.input_tensor = torch.randn(8, 3, 128, 128)  # Batch of 8 RGB images of size 128x128

    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, RiceModel)

    def test_forward_pass(self):
        """Test if the forward pass produces the correct output shape."""
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (8, 5))  # Batch size 8, num_classes 5


if __name__ == "__main__":
    unittest.main()
