"""NLP text classification — reference implementation.

Demonstrates the project code quality contract:
- Complete type hints on all signatures
- Typed config via dataclass
- argparse CLI entry point
- logging (no print)
- Modular structure: load_data / build_model / train / evaluate / export
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "imdb"
    text_col: str = "text"
    label_col: str = "label"
    output_dir: Path = Path("outputs/nlp")
    hub_model_id: str = ""          # set to push to HF Hub
    num_labels: int = 2
    max_length: int = 256
    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


def load_data(cfg: Config) -> tuple[Dataset, Dataset]:
    """Load dataset from HF Hub, return (train_ds, eval_ds)."""
    raw = load_dataset(cfg.dataset_name)
    train_ds = raw["train"]
    eval_ds = raw.get("test", raw.get("validation"))
    logger.info(
        "Dataset: %s | Train: %d | Eval: %d",
        cfg.dataset_name, len(train_ds), len(eval_ds),
    )
    logger.info("Columns: %s", train_ds.column_names)
    return train_ds, eval_ds


def build_model(cfg: Config) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load pretrained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels,
    )
    logger.info("Model: %s | Labels: %d", cfg.model_name, cfg.num_labels)
    return model, tokenizer


def tokenize(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    cfg: Config,
) -> Dataset:
    """Tokenize dataset in-place with truncation and padding."""
    def _tok(batch: dict) -> dict:
        return tokenizer(
            batch[cfg.text_col],
            truncation=True,
            max_length=cfg.max_length,
        )
    return dataset.map(_tok, batched=True, remove_columns=[cfg.text_col])


def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    """Accuracy and macro-F1 for Trainer."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
    }


def train(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    eval_ds: Dataset,
    cfg: Config,
) -> Trainer:
    """Fine-tune model with HF Trainer, return fitted Trainer."""
    train_tok = tokenize(train_ds, tokenizer, cfg)
    eval_tok = tokenize(eval_ds, tokenizer, cfg)
    collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        disable_tqdm=True,
        seed=cfg.seed,
        push_to_hub=bool(cfg.hub_model_id),
        hub_model_id=cfg.hub_model_id or None,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


def evaluate(trainer: Trainer) -> dict[str, float]:
    """Run final evaluation and log results."""
    results = trainer.evaluate()
    logger.info("Final eval | accuracy=%.4f | f1=%.4f",
                results["eval_accuracy"], results["eval_f1"])
    return results


def export(trainer: Trainer, cfg: Config) -> Path:
    """Save model and tokenizer locally (and push to Hub if configured)."""
    save_path = cfg.output_dir / "final"
    trainer.save_model(str(save_path))
    logger.info("Model saved to %s", save_path)
    if cfg.hub_model_id:
        trainer.push_to_hub()
        logger.info("Pushed to Hub: https://huggingface.co/%s", cfg.hub_model_id)
    return save_path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="NLP text classification fine-tuning")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset-name", type=str, default="imdb")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nlp"))
    parser.add_argument("--hub-model-id", type=str, default="")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return Config(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        text_col=args.text_col,
        label_col=args.label_col,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        num_labels=args.num_labels,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    logger.info("Config: %s", cfg)
    train_ds, eval_ds = load_data(cfg)
    model, tokenizer = build_model(cfg)
    trainer = train(model, tokenizer, train_ds, eval_ds, cfg)
    evaluate(trainer)
    export(trainer, cfg)


if __name__ == "__main__":
    main()
