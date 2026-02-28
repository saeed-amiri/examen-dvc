"""
Grid Search Script

This script:
1. Loads processed training data.
2. Loads hyperparameters from params.yaml (if available).
3. Performs GridSearchCV on RandomForestRegressor.
4. Saves best parameters into models/best_params.pkl.
"""

from pathlib import Path
import logging
from dataclasses import dataclass
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


@dataclass
class Paths:
    """Set the paths"""
    processed: Path
    models: Path
    logs: Path
    params_file: Path
    log_file: Path


def _set_paths() -> Paths:
    """Set all project paths"""
    project_root: Path = Path(__file__).parents[2]
    data_path: Path = project_root / "data"

    return Paths(
        processed=data_path / "processed",
        models=project_root / "models",
        logs=project_root / "logs",
        params_file=project_root / "params.yaml",
        log_file=project_root / "logs" / "gridsearch.log",
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


def _load_params(paths: Paths) -> dict:
    """Load parameters from YAML file (with fallback defaults)"""
    if paths.params_file.exists():
        logging.info("Loading parameters from %s", paths.params_file)
        with open(paths.params_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    logging.warning("params.yaml not found. Using default parameters.")

    return {
        "model": {"random_state": 42},
        "search": {
            "cv": 5,
            "n_jobs": -1,
            "param_grid": {
                "n_estimators": [100, 300],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
    }


def _run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> GridSearchCV:
    """Run GridSearchCV"""
    model_base = params.get("model", {})
    search_cfg = params.get("search", {})

    param_grid = search_cfg.get("param_grid", {})
    cv = search_cfg.get("cv", 5)
    n_jobs = search_cfg.get("n_jobs", -1)

    logging.info("Base model parameters: %s", model_base)
    logging.info("Grid: %s | CV=%s | n_jobs=%s", param_grid, cv, n_jobs)

    base_model = RandomForestRegressor(**model_base)

    gs = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
    )

    logging.info("Starting GridSearchCV")
    gs.fit(X_train, y_train)

    logging.info("GridSearch completed")
    return gs


def _save_best_params(paths: Paths, gs: GridSearchCV, params: dict) -> None:
    """Save best grid search results"""
    search_cfg = params.get("search", {})
    cv = search_cfg.get("cv", 5)
    model_base = params.get("model", {})

    best = {
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
        "cv": cv,
        "model_class": "RandomForestRegressor",
        "model_base_params": model_base,
    }

    best_path = paths.models / "best_params.pkl"
    joblib.dump(best, best_path)

    logging.info("Best parameters saved to: %s", best_path)


def main() -> None:
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    paths: Paths = _set_paths()
    _mk_dirs(paths)
    _set_logger(paths)

    X_train, y_train = _load_training_data(paths)
    params = _load_params(paths)

    gs = _run_grid_search(X_train, y_train, params)
    _save_best_params(paths, gs, params)

    logging.info("Grid search completed successfully.")


if __name__ == "__main__":
    main()
