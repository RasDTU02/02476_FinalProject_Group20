import unittest
import torch
from torch.utils.data import DataLoader
from src.rice.model import RiceModel
from src.rice.train import train_model
from src.rice.data import RiceDataset  # Assuming this is your dataset class


class TestTraining(unittest.TestCase):
    def setUp(self):
        """Set up for training tests."""
        self.model = RiceModel(num_classes=5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Mock a small dataset for testing
        self.dataset = RiceDataset(root_dir="data/archive/Rice_Image_Dataset", subset=10, resize=(128, 128))
        self.dataloader = DataLoader(self.dataset, batch_size=2)

        # Optimizer and loss function
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

    def test_training_step(self):
        """Test a single training step."""
        for batch in self.dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.assertGreater(loss.item(), 0)  # Ensure loss is computed

            # Break after first batch
            break

    def test_training_loop(self):
        """Test the full training loop."""
        initial_loss = None

        # Run training for 2 epochs
        train_model(
            model=self.model,
            dataloader=self.dataloader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            num_epochs=2,
            device=self.device,
        )

        # Validate that training decreases loss (basic check for learning)
        for batch in self.dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            if initial_loss is None:
                initial_loss = loss.item()
            else:
                self.assertLess(loss.item(), initial_loss)
                break


if __name__ == "__main__":
    unittest.main()
