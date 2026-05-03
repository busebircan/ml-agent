"""Tabular classification — reference implementation.

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

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    data_path: Path = Path("data/tabular/train.csv")
    test_path: Path = Path("data/tabular/test.csv")
    output_dir: Path = Path("outputs/tabular")
    target_col: str = "target"
    n_folds: int = 5
    seed: int = 42
    # LightGBM hyperparameters
    n_estimators: int = 1000
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 50
    n_jobs: int = -1


def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs, return (train_df, test_df)."""
    train_df = pd.read_csv(cfg.data_path)
    test_df = pd.read_csv(cfg.test_path)
    logger.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)
    logger.info("Target distribution:\n%s", train_df[cfg.target_col].value_counts())
    return train_df, test_df


def engineer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Encode categoricals, return (train_df, test_df, feature_cols)."""
    combined = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0)
    cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    n_train = len(train_df)
    train_features = combined.iloc[:n_train].copy()
    test_features = combined.iloc[n_train:].copy()

    feature_cols = [c for c in train_features.columns]
    logger.info("Features: %d | Categorical encoded: %d", len(feature_cols), len(cat_cols))
    return train_features, test_features, feature_cols


def build_model(cfg: Config) -> LGBMClassifier:
    """Return configured LightGBM classifier."""
    return LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbose=-1,
    )


def train(
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
) -> tuple[list[LGBMClassifier], np.ndarray]:
    """Stratified k-fold training. Returns (models, oof_predictions)."""
    X = train_features[feature_cols].values
    y = train_df[cfg.target_col].values
    oof_preds = np.zeros(len(X))
    models: list[LGBMClassifier] = []

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = build_model(cfg)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[],
        )
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        logger.info("Fold %d/%d | val_auc=%.4f", fold, cfg.n_folds, fold_auc)
        models.append(model)

    oof_auc = roc_auc_score(y, oof_preds)
    logger.info("OOF AUC: %.4f", oof_auc)
    return models, oof_preds


def evaluate(
    models: list[LGBMClassifier],
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
) -> None:
    """Log full classification report on OOF predictions."""
    X = train_features[feature_cols].values
    y = train_df[cfg.target_col].values
    oof_preds = np.mean(
        [m.predict_proba(X)[:, 1] for m in models], axis=0
    )
    oof_labels = (oof_preds > 0.5).astype(int)
    logger.info("Classification report:\n%s", classification_report(y, oof_labels))
    logger.info("ROC-AUC: %.4f", roc_auc_score(y, oof_preds))


def export(
    models: list[LGBMClassifier],
    test_features: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
) -> Path:
    """Save ensemble predictions and model artefacts."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    X_test = test_features[feature_cols].values
    test_preds = np.mean(
        [m.predict_proba(X_test)[:, 1] for m in models], axis=0
    )

    submission = pd.DataFrame({"prediction": test_preds})
    submission_path = cfg.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    logger.info("Submission saved to %s", submission_path)

    for i, model in enumerate(models):
        model_path = cfg.output_dir / f"lgbm_fold{i + 1}.joblib"
        joblib.dump(model, model_path)
    logger.info("Saved %d model files to %s", len(models), cfg.output_dir)
    return submission_path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Tabular classification with LightGBM")
    parser.add_argument("--data-path", type=Path, default=Path("data/tabular/train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/tabular/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tabular"))
    parser.add_argument("--target-col", type=str, default="target")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()
    return Config(
        data_path=args.data_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        n_folds=args.n_folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
    )


def main() -> None:
    cfg = parse_args()
    logger.info("Config: %s", cfg)
    train_df, test_df = load_data(cfg)
    train_features, test_features, feature_cols = engineer_features(
        train_df, test_df, cfg.target_col
    )
    models, _ = train(train_df, train_features, feature_cols, cfg)
    evaluate(models, train_df, train_features, feature_cols, cfg)
    export(models, test_features, feature_cols, cfg)


if __name__ == "__main__":
    main()
