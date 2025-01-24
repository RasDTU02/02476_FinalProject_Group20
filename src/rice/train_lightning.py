from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_pretrained_model
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb


class RiceLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = get_pretrained_model(
            num_classes=cfg.model_conf.num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Log batch loss and accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.cfg.training_conf.learning_rate
        )
        return optimizer


def load_data(cfg: DictConfig):
    """Load processed JPG images with transformation."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    class_dirs = [
        d
        for d in Path("data/raw/Rice_Image_Dataset").iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    for class_idx, class_dir in enumerate(class_dirs):
        images = list(class_dir.glob("*.jpg"))
        if not images:
            continue

        if cfg.training_conf.max_images:
            images = images[: cfg.training_conf.max_images]

        split_idx = int(0.8 * len(images))  # 80% train, 20% val
        for i, img_path in enumerate(images):
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            if i < split_idx:
                train_data.append(img)
                train_labels.append(class_idx)
            else:
                val_data.append(img)
                val_labels.append(class_idx)

    train_set = torch.utils.data.TensorDataset(
        torch.stack(train_data), torch.tensor(train_labels)
    )
    val_set = torch.utils.data.TensorDataset(
        torch.stack(val_data), torch.tensor(val_labels)
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.training_conf.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.training_conf.batch_size, shuffle=False
    )

    return train_loader, val_loader


@hydra.main(
    config_path="../../configs", config_name="config", version_base=None
)
def train_model(cfg: DictConfig):
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Prepare data
    train_loader, val_loader = load_data(cfg)

    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project="rice-classification",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.get("experiment_name", "default_experiment"),
    )

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="rice-model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        monitor="val_acc",
        mode="max",
    )

    # Initialize Lightning module
    model = RiceLightningModule(cfg)

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training_conf.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Finish WandB run
    wandb.finish()

    # Save final model
    now = datetime.now()
    run_folder = now.strftime("%Y%m%d_%H%M%S")
    run_path = Path("models") / run_folder
    run_path.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    model_path = run_path / "rice_model_learned_parameters.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model state_dict saved as {model_path}")

    # Save full model
    model_path_full = run_path / "rice_model_lightning.pth"
    torch.save(model, model_path_full)
    print(f"Full model saved as {model_path_full}")


def main():
    train_model()


if __name__ == "__main__":
    main()
