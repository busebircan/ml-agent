"""Thermal image labelling — reference implementation.

Multi-task CV pipeline: classifies radiator alert level and alert type
from thermal images, and regresses heating efficiency. Fuses image
features (EfficientNet-B0) with tabular room metadata.

Key patterns demonstrated:
  - Multi-task head (two classifiers + one regressor)
  - Image + tabular feature fusion
  - Class imbalance: FocalLoss + WeightedRandomSampler + minority augmentation
  - MLflow experiment tracking
  - Azure Blob Storage image loading with local cache

Follows the project code quality contract:
  - Complete type hints on all signatures
  - Typed config via dataclass
  - argparse CLI entry point
  - logging (no print)
  - Modular structure: load_data / build_model / train / evaluate / export
"""

import argparse
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, f1_score, mean_absolute_error
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Paths
    train_csv: Path = Path("data/thermal/train.csv")
    val_csv: Path = Path("data/thermal/val.csv")
    test_csv: Path = Path("data/thermal/test.csv")
    cache_dir: Path = Path("data/thermal/cache")
    output_dir: Path = Path("outputs/thermal")

    # Label classes (order must match integer label encoding in CSVs)
    alert_classes: list[str] = field(
        default_factory=lambda: ["No Alert", "Low", "Medium", "High"]
    )
    alerttext_classes: list[str] = field(
        default_factory=lambda: [
            "OK", "Needs Balancing", "Bleeding Required",
            "Check Thermostat", "Check Boiler", "Cold Spot", "Other",
        ]
    )
    room_classes: list[str] = field(
        default_factory=lambda: [
            "Living Room", "Bedroom", "Kitchen", "Bathroom",
            "Hallway", "Other",
        ]
    )
    floor_classes: list[str] = field(
        default_factory=lambda: ["Ground", "First", "Second", "Third+"]
    )

    # Model
    backbone: str = "efficientnet_b0"
    tabular_emb_dim: int = 32
    fusion_hidden_dim: int = 256
    dropout: float = 0.3

    # Training
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-2
    early_stopping_patience: int = 5
    num_workers: int = 4
    seed: int = 42

    # Loss weights (multi-task)
    loss_weight_alert: float = 1.0
    loss_weight_alerttext: float = 2.0   # harder task, weighted higher
    loss_weight_efficiency: float = 0.5

    # Class imbalance
    focal_gamma: float = 2.0
    minority_threshold: int = 50         # alerttext classes below this get stronger augmentation

    # Image normalisation (ImageNet defaults — fine for thermal after RGB conversion)
    img_size: int = 224
    img_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    img_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # MLflow
    mlflow_experiment: str = "thermal-labelling"
    mlflow_run_name: str = "baseline"

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(cfg: Config) -> tuple[
    transforms.Compose, transforms.Compose, transforms.Compose
]:
    """Return (train_transform, minority_transform, eval_transform)."""
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size + 32, cfg.img_size + 32)),
        transforms.RandomCrop(cfg.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(cfg.img_mean, cfg.img_std),
    ])
    # Stronger augmentation for minority alerttext classes
    minority_tf = transforms.Compose([
        transforms.Resize((cfg.img_size + 32, cfg.img_size + 32)),
        transforms.RandomCrop(cfg.img_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(cfg.img_mean, cfg.img_std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.img_mean, cfg.img_std),
    ])
    return train_tf, minority_tf, eval_tf


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ThermalDataset(Dataset):
    """Thermal radiator image dataset.

    Expects a CSV with columns:
        image_url, room_name_idx, floor_idx, avg_temp_norm,
        alert_label, alerttext_label, efficiency  (0–100 float)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: transforms.Compose,
        cache_dir: Path,
        minority_indices: set[int] | None = None,
        minority_transform: transforms.Compose | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.minority_indices: set[int] = minority_indices or set()
        self.minority_transform = minority_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        alerttext_label = int(row["alerttext_label"])

        tf = (
            self.minority_transform
            if alerttext_label in self.minority_indices and self.minority_transform
            else self.transform
        )
        image = self._load_image(str(row["image_url"]), tf)

        return {
            "image": image,
            "tabular": torch.tensor(
                [int(row["room_name_idx"]), int(row["floor_idx"])], dtype=torch.long
            ),
            "avg_temp": torch.tensor([float(row["avg_temp_norm"])], dtype=torch.float32),
            "alert_label": torch.tensor(int(row["alert_label"]), dtype=torch.long),
            "alerttext_label": torch.tensor(alerttext_label, dtype=torch.long),
            "efficiency": torch.tensor(float(row["efficiency"]) / 100.0, dtype=torch.float32),
        }

    def _load_image(self, url: str, tf: transforms.Compose) -> torch.Tensor:
        cache_path = self.cache_dir / url.split("/")[-1]
        if cache_path.exists():
            try:
                return tf(Image.open(cache_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Corrupt cache, re-downloading %s: %s", cache_path, exc)
                cache_path.unlink(missing_ok=True)
        try:
            data = requests.get(url, timeout=10).content
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.save(cache_path)
            return tf(img)
        except Exception as exc:
            logger.error("Failed to load image %s: %s", url, exc)
            return torch.zeros(3, 224, 224)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class Focal Loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        focal = (1.0 - torch.exp(-ce)) ** self.gamma * ce
        return focal.mean()


class ThermalModel(nn.Module):
    """EfficientNet-B0 + tabular embedding fusion with three output heads.

    Inputs:
        image    (B, 3, H, W)
        tabular  (B, 2)    — [room_name_idx, floor_idx]
        avg_temp (B, 1)    — normalised temperature

    Outputs dict:
        alert_logits      (B, n_alert)
        alerttext_logits  (B, n_alerttext)
        efficiency_pred   (B,)   — in [0, 1], multiply by 100 for %
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        emb = cfg.tabular_emb_dim
        n_rooms = len(cfg.room_classes)
        n_floors = len(cfg.floor_classes)
        n_alert = len(cfg.alert_classes)
        n_alerttext = len(cfg.alerttext_classes)

        # Image backbone — EfficientNet-B0 outputs 1280-d after global avg pool
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        img_dim = 1280

        # Tabular branch
        self.room_emb = nn.Embedding(n_rooms, emb)
        self.floor_emb = nn.Embedding(n_floors, emb)
        self.temp_enc = nn.Sequential(nn.Linear(1, emb), nn.ReLU())

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + emb * 3, cfg.fusion_hidden_dim),
            nn.BatchNorm1d(cfg.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        # Heads
        fd = cfg.fusion_hidden_dim
        self.alert_head = nn.Linear(fd, n_alert)
        self.alerttext_head = nn.Linear(fd, n_alerttext)
        self.efficiency_head = nn.Sequential(
            nn.Linear(fd, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        avg_temp: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        img_feat = self.image_encoder(image).flatten(1)
        tab_feat = torch.cat([
            self.room_emb(tabular[:, 0]),
            self.floor_emb(tabular[:, 1]),
            self.temp_enc(avg_temp),
        ], dim=1)
        fused = self.fusion(torch.cat([img_feat, tab_feat], dim=1))
        return {
            "alert_logits": self.alert_head(fused),
            "alerttext_logits": self.alerttext_head(fused),
            "efficiency_pred": self.efficiency_head(fused).squeeze(1),
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: Config) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """Build train and val DataLoaders with class-imbalance handling.

    Returns (train_loader, val_loader, alerttext_class_weights).
    """
    train_df = pd.read_csv(cfg.train_csv)
    val_df = pd.read_csv(cfg.val_csv)
    logger.info("Train: %d rows | Val: %d rows", len(train_df), len(val_df))

    # Compute inverse-frequency weights for alerttext (most imbalanced task)
    n_classes = len(cfg.alerttext_classes)
    counts = train_df["alerttext_label"].value_counts().sort_index()
    all_counts = torch.zeros(n_classes)
    for cls_idx, cnt in counts.items():
        all_counts[int(cls_idx)] = cnt
    inv_freq = 1.0 / all_counts.clamp(min=1.0)
    class_weights = inv_freq / inv_freq.mean()

    minority_indices = {
        int(i) for i, cnt in counts.items() if cnt < cfg.minority_threshold
    }
    logger.info("Minority alerttext classes (<%d samples): %s",
                cfg.minority_threshold, minority_indices)

    sample_weights = [class_weights[int(l)].item() for l in train_df["alerttext_label"]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_tf, minority_tf, eval_tf = build_transforms(cfg)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        ThermalDataset(train_df, train_tf, cfg.cache_dir, minority_indices, minority_tf),
        batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        ThermalDataset(val_df, eval_tf, cfg.cache_dir),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, class_weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion_ce: nn.CrossEntropyLoss,
    criterion_focal: FocalLoss,
    criterion_mse: nn.MSELoss,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    alert_loss = criterion_ce(outputs["alert_logits"], batch["alert_label"])
    alerttext_loss = criterion_focal(outputs["alerttext_logits"], batch["alerttext_label"])
    efficiency_loss = criterion_mse(outputs["efficiency_pred"], batch["efficiency"])
    total = (
        cfg.loss_weight_alert * alert_loss
        + cfg.loss_weight_alerttext * alerttext_loss
        + cfg.loss_weight_efficiency * efficiency_loss
    )
    return total, {
        "alert": alert_loss.item(),
        "alerttext": alerttext_loss.item(),
        "efficiency": efficiency_loss.item(),
    }


def train(
    model: ThermalModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    cfg: Config,
) -> ThermalModel:
    """Train with early stopping, cosine LR schedule, and MLflow tracking."""
    device = torch.device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = cfg.output_dir / "best_model.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_focal = FocalLoss(gamma=cfg.focal_gamma, weight=class_weights.to(device))
    criterion_mse = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        mlflow.log_params({
            "backbone": cfg.backbone, "epochs": cfg.epochs,
            "batch_size": cfg.batch_size, "lr": cfg.lr,
            "focal_gamma": cfg.focal_gamma, "dropout": cfg.dropout,
        })

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running: dict[str, float] = {"total": 0, "alert": 0, "alerttext": 0, "efficiency": 0}

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(batch["image"], batch["tabular"], batch["avg_temp"])
                total_loss, components = _compute_loss(
                    outputs, batch, criterion_ce, criterion_focal, criterion_mse, cfg
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running["total"] += total_loss.item()
                for k, v in components.items():
                    running[k] += v

            n = len(train_loader)
            train_loss = running["total"] / n
            val_metrics = evaluate(model, val_loader, criterion_ce, criterion_mse, cfg)
            scheduler.step()

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "alert_acc=%.3f | alerttext_acc=%.3f | eff_mae=%.2f",
                epoch, cfg.epochs, train_loss, val_metrics["val_loss"],
                val_metrics["alert_acc"], val_metrics["alerttext_acc"],
                val_metrics["efficiency_mae"],
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}},
                step=epoch,
            )

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                torch.save({"model_state_dict": model.state_dict(), "cfg": cfg}, best_ckpt)
                logger.info("Checkpoint saved (val_loss=%.4f)", best_val_loss)
                mlflow.log_artifact(str(best_ckpt))
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: ThermalModel,
    loader: DataLoader,
    criterion_ce: nn.CrossEntropyLoss,
    criterion_mse: nn.MSELoss,
    cfg: Config,
) -> dict[str, float]:
    """Return val_loss, alert_acc, alerttext_acc, efficiency_mae."""
    device = torch.device(cfg.device)
    model.eval()
    total_loss = 0.0
    alert_preds, alert_labels = [], []
    alerttext_preds, alerttext_labels = [], []
    eff_preds, eff_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["image"], batch["tabular"], batch["avg_temp"])
            loss = (
                cfg.loss_weight_alert * criterion_ce(outputs["alert_logits"], batch["alert_label"])
                + cfg.loss_weight_alerttext * criterion_ce(outputs["alerttext_logits"], batch["alerttext_label"])
                + cfg.loss_weight_efficiency * criterion_mse(outputs["efficiency_pred"], batch["efficiency"])
            )
            total_loss += loss.item()
            alert_preds += outputs["alert_logits"].argmax(1).cpu().tolist()
            alert_labels += batch["alert_label"].cpu().tolist()
            alerttext_preds += outputs["alerttext_logits"].argmax(1).cpu().tolist()
            alerttext_labels += batch["alerttext_label"].cpu().tolist()
            eff_preds += (outputs["efficiency_pred"] * 100).cpu().tolist()
            eff_labels += (batch["efficiency"] * 100).cpu().tolist()

    alert_acc = sum(p == l for p, l in zip(alert_preds, alert_labels)) / len(alert_labels)
    alerttext_acc = sum(p == l for p, l in zip(alerttext_preds, alerttext_labels)) / len(alerttext_labels)
    return {
        "val_loss": total_loss / len(loader),
        "alert_acc": alert_acc,
        "alerttext_acc": alerttext_acc,
        "efficiency_mae": float(mean_absolute_error(eff_labels, eff_preds)),
    }


def full_eval(model: ThermalModel, cfg: Config) -> dict[str, float]:
    """Final evaluation on the held-out test set with full classification reports."""
    _, _, eval_tf = build_transforms(cfg)
    test_df = pd.read_csv(cfg.test_csv)
    test_loader = DataLoader(
        ThermalDataset(test_df, eval_tf, cfg.cache_dir),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )
    device = torch.device(cfg.device)
    model.eval()
    alert_preds, alert_labels, alerttext_preds, alerttext_labels = [], [], [], []
    eff_preds, eff_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["image"], batch["tabular"], batch["avg_temp"])
            alert_preds += outputs["alert_logits"].argmax(1).cpu().tolist()
            alert_labels += batch["alert_label"].cpu().tolist()
            alerttext_preds += outputs["alerttext_logits"].argmax(1).cpu().tolist()
            alerttext_labels += batch["alerttext_label"].cpu().tolist()
            eff_preds += (outputs["efficiency_pred"] * 100).cpu().tolist()
            eff_labels += (batch["efficiency"] * 100).cpu().tolist()

    logger.info("\n=== Alert ===\n%s",
                classification_report(alert_labels, alert_preds, target_names=cfg.alert_classes))
    logger.info("\n=== Alert Text ===\n%s",
                classification_report(alerttext_labels, alerttext_preds, target_names=cfg.alerttext_classes))
    eff_mae = mean_absolute_error(eff_labels, eff_preds)
    logger.info("Efficiency MAE: %.2f pp", eff_mae)

    return {
        "alert_f1": f1_score(alert_labels, alert_preds, average="weighted"),
        "alerttext_f1": f1_score(alerttext_labels, alerttext_preds, average="weighted"),
        "efficiency_mae": eff_mae,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(model: ThermalModel, cfg: Config) -> Path:
    """Load best checkpoint and save final model for serving."""
    best_ckpt = cfg.output_dir / "best_model.pt"
    checkpoint = torch.load(best_ckpt, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    export_path = cfg.output_dir / "final_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg}, export_path)
    logger.info("Final model saved to %s", export_path)
    return export_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Thermal image labelling — multi-task training")
    parser.add_argument("--train-csv", type=Path, default=Path("data/thermal/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/thermal/val.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/thermal/test.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/thermal/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/thermal"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--minority-threshold", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--mlflow-experiment", type=str, default="thermal-labelling")
    parser.add_argument("--mlflow-run-name", type=str, default="baseline")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return Config(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
        minority_threshold=args.minority_threshold,
        early_stopping_patience=args.early_stopping_patience,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    logger.info("Config: %s", cfg)
    torch.manual_seed(cfg.seed)

    train_loader, val_loader, class_weights = load_data(cfg)
    model = ThermalModel(cfg).to(torch.device(cfg.device))
    train(model, train_loader, val_loader, class_weights, cfg)
    full_eval(model, cfg)
    export(model, cfg)


if __name__ == "__main__":
    main()
