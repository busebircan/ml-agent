"""Time-series forecasting — reference implementation.

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
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    data_path: Path = Path("data/timeseries/series.csv")
    output_dir: Path = Path("outputs/timeseries")
    date_col: str = "date"
    target_col: str = "value"
    id_col: str = "series_id"
    horizon: int = 28          # steps to forecast
    n_lags: int = 28           # lag features
    n_rolling: list[int] = None  # rolling window sizes; set in __post_init__
    val_frac: float = 0.2
    seed: int = 42
    # LightGBM
    n_estimators: int = 1000
    learning_rate: float = 0.05
    num_leaves: int = 63
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.n_rolling is None:
            self.n_rolling = [7, 14, 28]


def load_data(cfg: Config) -> pd.DataFrame:
    """Load CSV, parse dates, sort by id and date."""
    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.date_col])
    df = df.sort_values([cfg.id_col, cfg.date_col]).reset_index(drop=True)
    logger.info(
        "Loaded %d rows | %d series | date range: %s to %s",
        len(df),
        df[cfg.id_col].nunique(),
        df[cfg.date_col].min().date(),
        df[cfg.date_col].max().date(),
    )
    return df


def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add lag, rolling, and calendar features per series."""
    dfs: list[pd.DataFrame] = []
    for series_id, group in df.groupby(cfg.id_col):
        group = group.copy().sort_values(cfg.date_col)

        for lag in range(1, cfg.n_lags + 1):
            group[f"lag_{lag}"] = group[cfg.target_col].shift(lag)

        for window in cfg.n_rolling:
            group[f"roll_mean_{window}"] = (
                group[cfg.target_col].shift(1).rolling(window).mean()
            )
            group[f"roll_std_{window}"] = (
                group[cfg.target_col].shift(1).rolling(window).std()
            )

        dt = group[cfg.date_col]
        group["dayofweek"] = dt.dt.dayofweek
        group["month"] = dt.dt.month
        group["dayofyear"] = dt.dt.dayofyear
        group["weekofyear"] = dt.dt.isocalendar().week.astype(int)
        dfs.append(group)

    out = pd.concat(dfs).dropna().reset_index(drop=True)
    logger.info("Feature matrix shape after engineering: %s", out.shape)
    return out


def build_model(cfg: Config) -> LGBMRegressor:
    """Return configured LightGBM regressor."""
    return LGBMRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbose=-1,
    )


def train(
    df: pd.DataFrame,
    cfg: Config,
) -> tuple[LGBMRegressor, list[str], pd.Timestamp]:
    """Time-based train/val split, fit model, return (model, features, cutoff)."""
    cutoff = df[cfg.date_col].max() - pd.Timedelta(days=cfg.horizon)
    train_df = df[df[cfg.date_col] <= cutoff]
    val_df = df[df[cfg.date_col] > cutoff]

    feature_cols = [
        c for c in df.columns
        if c not in [cfg.date_col, cfg.target_col, cfg.id_col]
    ]
    X_train, y_train = train_df[feature_cols], train_df[cfg.target_col]
    X_val, y_val = val_df[feature_cols], val_df[cfg.target_col]

    model = build_model(cfg)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = mean_squared_error(y_val, val_preds) ** 0.5
    logger.info("Val MAE=%.4f | Val RMSE=%.4f", mae, rmse)
    return model, feature_cols, cutoff


def evaluate(
    model: LGBMRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
    cutoff: pd.Timestamp,
    cfg: Config,
) -> dict[str, float]:
    """Compute and log metrics on the held-out horizon."""
    val_df = df[df[cfg.date_col] > cutoff].copy()
    preds = model.predict(val_df[feature_cols])
    actual = val_df[cfg.target_col].values
    metrics = {
        "mae": float(mean_absolute_error(actual, preds)),
        "rmse": float(mean_squared_error(actual, preds) ** 0.5),
        "mape": float(np.mean(np.abs((actual - preds) / (actual + 1e-8))) * 100),
    }
    logger.info("Evaluation | MAE=%.4f | RMSE=%.4f | MAPE=%.2f%%", *metrics.values())
    return metrics


def export(
    model: LGBMRegressor,
    feature_cols: list[str],
    cfg: Config,
) -> Path:
    """Save model and feature list for serving."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.output_dir / "lgbm_forecast.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    logger.info("Model saved to %s", model_path)
    return model_path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Time-series forecasting with LightGBM")
    parser.add_argument("--data-path", type=Path, default=Path("data/timeseries/series.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/timeseries"))
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--target-col", type=str, default="value")
    parser.add_argument("--id-col", type=str, default="series_id")
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--n-lags", type=int, default=28)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()
    return Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        date_col=args.date_col,
        target_col=args.target_col,
        id_col=args.id_col,
        horizon=args.horizon,
        n_lags=args.n_lags,
        val_frac=args.val_frac,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
    )


def main() -> None:
    cfg = parse_args()
    logger.info("Config: %s", cfg)
    df = load_data(cfg)
    df = engineer_features(df, cfg)
    model, feature_cols, cutoff = train(df, cfg)
    evaluate(model, df, feature_cols, cutoff, cfg)
    export(model, feature_cols, cfg)


if __name__ == "__main__":
    main()
