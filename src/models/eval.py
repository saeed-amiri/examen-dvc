"""
Model Evaluation Script

This script:
1. Loads the trained model.
2. Loads processed test data.
3. Generates predictions.
4. Saves predictions to data/processed.
5. Computes evaluation metrics (MSE, MAE, R2).
6. Saves metrics into metrics/scores.json.
"""

from pathlib import Path
import logging
import json
from dataclasses import dataclass

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class Paths:
    """Set the paths"""
    processed: Path
    models: Path
    metrics: Path
    logs: Path
    log_file: Path


def _set_paths() -> Paths:
    """Set all project paths"""
    project_root: Path = Path(__file__).parents[2]
    data_path: Path = project_root / "data"

    return Paths(
        processed=data_path / "processed",
        models=project_root / "models",
        metrics=project_root / "metrics",
        logs=project_root / "logs",
        log_file=project_root / "logs" / "eval.log",
    )


def _mk_dirs(paths: Paths) -> None:
    """Create required directories"""
    paths.metrics.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)


def _set_logger(paths: Paths) -> None:
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(paths.log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _load_model(paths: Paths):
    """Load trained model"""
    model_path = paths.models / "gbr_model.pkl"
    logging.info("Loading model from: %s", model_path)

    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError as err:
        logging.error("Model file not found!")
        raise IOError(f"{model_path} cannot be loaded.") from err


def _load_test_data(paths: Paths) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed test datasets"""
    logging.info("Loading processed test data")

    try:
        X_test = pd.read_csv(paths.processed / "X_test_scaled.csv")
        y_test = pd.read_csv(paths.processed / "y_test.csv").squeeze("columns")
        return X_test, y_test
    except FileNotFoundError as err:
        logging.error("Processed test data not found!")
        raise IOError("Test datasets cannot be loaded.") from err


def _save_predictions(
    paths: Paths,
    y_test: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Save prediction results"""
    pred_path = paths.processed / "predictions.csv"

    pd.DataFrame(
        {"y_true": y_test, "y_pred": y_pred}
    ).to_csv(pred_path, index=False)

    logging.info("Predictions saved to: %s", pred_path)


def _compute_metrics(
    y_test: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute regression metrics"""
    logging.info("Computing evaluation metrics")

    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return {"mse": mse, "mae": mae, "r2": r2}


def _save_metrics(paths: Paths, metrics: dict[str, float]) -> None:
    """Save metrics as JSON"""
    metrics_path = paths.metrics / "scores.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Metrics saved to: %s | %s", metrics_path, metrics)


def main() -> None:
    """
    Evaluate trained model on test dataset.
    """
    paths: Paths = _set_paths()
    _mk_dirs(paths)
    _set_logger(paths)

    model = _load_model(paths)
    X_test, y_test = _load_test_data(paths)

    logging.info("Generating predictions")
    y_pred = model.predict(X_test)

    _save_predictions(paths, y_test, y_pred)

    metrics = _compute_metrics(y_test, y_pred)
    _save_metrics(paths, metrics)

    logging.info("Model evaluation completed successfully.")


if __name__ == "__main__":
    main()
