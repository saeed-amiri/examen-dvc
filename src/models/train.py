"""
Model Training Script

This script:
1. Loads processed training data.
2. Loads best hyperparameters from models/best_params.pkl.
3. Trains a RandomForestRegressor using the best parameters.
4. Saves the trained model to models/gbr_model.pkl.
"""

from pathlib import Path
import logging
from dataclasses import dataclass
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class Paths:
    """Set the paths"""
    processed: Path
    models: Path
    logs: Path
    log_file: Path


def _set_paths() -> Paths:
    """Set all project paths"""
    project_root: Path = Path(__file__).parents[2]
    data_path: Path = project_root / "data"

    return Paths(
        processed=data_path / "processed",
        models=project_root / "models",
        logs=project_root / "logs",
        log_file=project_root / "logs" / "train.log",
    )


def _mk_dirs(paths: Paths) -> None:
    """Create required directories"""
    paths.models.mkdir(parents=True, exist_ok=True)
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


def _load_training_data(paths: Paths) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed training dataset"""
    logging.info("Loading processed training data")

    try:
        X_train = pd.read_csv(paths.processed / "X_train_scaled.csv")
        y_train = pd.read_csv(paths.processed / "y_train.csv").squeeze("columns")
        return X_train, y_train
    except FileNotFoundError as err:
        logging.error("Training data not found!")
        raise IOError("Training datasets cannot be loaded.") from err


def _load_best_params(paths: Paths) -> dict:
    """Load best hyperparameters"""
    best_path = paths.models / "best_params.pkl"
    logging.info("Loading best parameters from: %s", best_path)

    try:
        return joblib.load(best_path)
    except FileNotFoundError as err:
        logging.error("Best parameters file not found!")
        raise IOError(f"{best_path} cannot be loaded.") from err


def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best: dict,
) -> RandomForestRegressor:
    """Train model using best hyperparameters"""
    logging.info("Initializing RandomForestRegressor with best parameters")

    model = RandomForestRegressor(
        **best.get("model_base_params", {}),
        **best.get("best_params", {}),
    )

    logging.info("Fitting model")
    model.fit(X_train, y_train)

    return model


def _save_model(paths: Paths, model: RandomForestRegressor) -> None:
    """Save trained model"""
    model_path = paths.models / "gbr_model.pkl"
    joblib.dump(model, model_path)

    logging.info("Model saved to: %s", model_path)


def main() -> None:
    """
    Train model using best hyperparameters.
    """
    paths: Paths = _set_paths()
    _mk_dirs(paths)
    _set_logger(paths)

    X_train, y_train = _load_training_data(paths)
    best = _load_best_params(paths)

    model = _train_model(X_train, y_train, best)
    _save_model(paths, model)

    logging.info("Model training completed successfully.")


if __name__ == "__main__":
    main()