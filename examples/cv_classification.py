"""Image classification — reference implementation.

Demonstrates the project code quality contract:
- Complete type hints on all signatures
- Typed config via dataclass
- argparse CLI entry point
- logging (no print)
- Modular structure: load_data / build_model / train / evaluate / export
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    data_dir: Path = Path("data/images")
    output_dir: Path = Path("outputs/cv")
    num_classes: int = 10
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-2
    img_size: int = 224
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    label_smoothing: float = 0.1
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


def load_data(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """Return train and validation DataLoaders."""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(cfg.img_size + 32),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])

    train_ds = datasets.ImageFolder(cfg.data_dir / "train", transform=train_transforms)
    val_ds = datasets.ImageFolder(cfg.data_dir / "val", transform=val_transforms)

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))
    logger.info("Classes: %s", train_ds.classes)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def build_model(cfg: Config) -> nn.Module:
    """ResNet-50 pretrained on ImageNet, head replaced for num_classes."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, cfg.num_classes)
    model = model.to(cfg.device)
    logger.info("Model: ResNet-50 | Classes: %d | Device: %s", cfg.num_classes, cfg.device)
    return model


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
) -> nn.Module:
    """Train model and return best checkpoint by val accuracy."""
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_acc: float = 0.0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, cfg)
        scheduler.step()

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_acc=%.4f | lr=%.2e",
            epoch, cfg.epochs, train_loss, val_acc,
            scheduler.get_last_lr()[0],
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.output_dir / "best.pt")
            logger.info("Checkpoint saved (val_acc=%.4f)", best_acc)

    logger.info("Training complete. Best val_acc=%.4f", best_acc)
    return model


def evaluate(model: nn.Module, loader: DataLoader, cfg: Config) -> float:
    """Return top-1 accuracy on loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def export(model: nn.Module, cfg: Config) -> Path:
    """Export best checkpoint to TorchScript for serving."""
    model.load_state_dict(torch.load(cfg.output_dir / "best.pt", map_location=cfg.device))
    model.eval()
    example = torch.randn(1, 3, cfg.img_size, cfg.img_size).to(cfg.device)
    scripted = torch.jit.trace(model, example)
    export_path = cfg.output_dir / "model.torchscript"
    scripted.save(str(export_path))
    logger.info("Exported TorchScript model to %s", export_path)
    return export_path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Image classification training")
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cv"))
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        num_workers=args.num_workers,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    logger.info("Config: %s", cfg)
    train_loader, val_loader = load_data(cfg)
    model = build_model(cfg)
    train(model, train_loader, val_loader, cfg)
    export(model, cfg)


if __name__ == "__main__":
    main()
