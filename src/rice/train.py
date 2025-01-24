# train.py
# command line interface: 'train'
# To run default training: python src/rice/train.py (remember to set cd to the root of the project)
# To run experiment: python src/rice/train.py experiment=exp1
# To add a parameter: python src/rice/train.py +experiment.new_param=42
# To change a parameter: python src/rice/train.py training_conf.learning_rate=0.0001

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from .model import get_pretrained_model
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from .logger import get_logger
import wandb
from torch.profiler import profile, record_function, ProfilerActivity

# Create logger
log = get_logger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def initialize_wandb(cfg: DictConfig): # Initialize Weights & Biases
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    safe_cfg = {k: v for k, v in cfg_dict.items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
    
    wandb.init(
        project="rice-classification",
        config=safe_cfg, # Save hydra config
        name=safe_cfg.get('experiment_name', 'default_experiment'),
    )

@hydra.main(config_path="../../configs", config_name="config", version_base=None) # Load configuration
def train_model(cfg: DictConfig):
    log.info("Training configuration:")
    log.info(f"Batch size: {cfg.training_conf.batch_size}")
    log.info(f"Learning rate: {cfg.training_conf.learning_rate}")
    log.info(f"Number of epochs: {cfg.training_conf.num_epochs}")
    log.info(f"Experiment name: {cfg.experiment_name}")

    # Sæt seed for reproducibility
    set_seed(cfg.seed)
    
    initialize_wandb(cfg)
    
    # Load data
    train_loader, test_loader = load_data(cfg)

    # Get model
    model = get_pretrained_model(num_classes=cfg.model_conf.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training_conf.learning_rate)

    # Training loop with profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                 record_shapes=True, 
                 profile_memory=True, 
                 with_stack=True) as prof:
        for epoch in range(cfg.training_conf.num_epochs):
            running_loss = 0.0
            model.train()
            total = 0
            correct = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training_conf.num_epochs}"):
                with record_function("model_inference"):
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    wandb.log({
                        "batch_loss": loss.item(),
                        "running_accuracy": 100 * correct / total
                    })
            
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total

            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": epoch_loss,
                "epoch_accuracy": epoch_accuracy
            })

            log.info(f"Epoch {epoch+1}: Loss {epoch_loss:.4f}, Accuracy {epoch_accuracy:.2f}%")

            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
            log.info(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
            prof.step()  # Step the profiler at the end of each epoch

    print("Training complete.")
    log.info("Training complete.")
    wandb.finish()  # Finish the run
    
    # Save model
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)

    now = datetime.now()
    run_folder = now.strftime("%Y%m%d_%H%M%S")
    run_path = models_path / run_folder
    run_path.mkdir(parents=True, exist_ok=True)

    model_path = run_path / "rice_model_learned_parameters.pth"
    torch.save(model.state_dict(), model_path) # Saves only the learned parameters
    print(f"Model state_dict saved as {model_path}")
    log.info(f"Model state_dict saved as {model_path}")

    model_path_full = run_path / "rice_model_full.pth"
    torch.save(model, model_path_full) # Saves the full model
    print(f"Full model saved as {model_path_full}")
    log.info(f"Full model saved as {model_path_full}")

    # Print profiling results
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


def load_data(cfg: DictConfig):
    """Load processed JPG images and optionally limit the number of images based on sample_ratio and max_images."""
    if not Path("data/raw/Rice_Image_Dataset").exists():
        raise FileNotFoundError("Processed data path 'data/raw/Rice_Image_Dataset' does not exist.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # A list comprehension that filters out non-directory files and hidden files
    class_dirs = [d for d in Path("data/raw/Rice_Image_Dataset").iterdir() if d.is_dir() and not d.name.startswith('.')]

    for class_idx, class_dir in enumerate(class_dirs):
        print(f"Processing class {class_idx}: {class_dir.name}")
        log.info(f"Processing class {class_idx}: {class_dir.name}")
        images = list(class_dir.glob("*.jpg"))
        if len(images) == 0:
            print(f"Warning: No images found in {class_dir}")
            log.warning(f"Warning: No images found in {class_dir}")
            continue

        # Limit the number of images based on sample_ratio and max_images
        num_samples = min(cfg.training_conf.max_images, int(len(images) * cfg.training_conf.sample_ratio))
        images = images[:num_samples]

        split_idx = int(0.8 * len(images))  # 80% train, 20% test
        for i, img_path in enumerate(images):
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            if i < split_idx:
                train_data.append(img)
                train_labels.append(class_idx)
            else:
                test_data.append(img)
                test_labels.append(class_idx)

    if not train_data:
        raise RuntimeError("No training data loaded. Check your processed data folder.")

    print("Unique train labels:", set(train_labels))
    log.info(f"Unique train labels: {set(train_labels)}")
    print("Unique test labels:", set(test_labels))
    log.info(f"Unique test labels: {set(test_labels)}")

    train_set = torch.utils.data.TensorDataset(torch.stack(train_data), torch.tensor(train_labels))
    test_set = torch.utils.data.TensorDataset(torch.stack(test_data), torch.tensor(test_labels))

    return DataLoader(train_set, batch_size=cfg.training_conf.batch_size, shuffle=True), DataLoader(test_set, batch_size=cfg.training_conf.batch_size, shuffle=False)

def main():
    train_model()

if __name__ == "__main__":
    main()