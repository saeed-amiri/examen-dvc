"""
Data Normalization: As you may notice, the data varies widely in scale,
so normalization is necessary. You can use existing functions to construct
this script. As output, this script will create two new datasets
(X_train_scaled, X_test_scaled) which you will also save in data/processed.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Paths:
    """Set the paths"""
    processed: Path
    logs: Path
    log_file: Path
    x_train: Path
    x_test: Path
    out_train: Path    
    out_test: Path

def _set_logger(paths: Paths) -> None:
    """set up the logger"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(paths.log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


def _set_paths() -> Paths:
    """Set all the paths"""
    project_root: Path = Path(__file__).parents[2]
    data_path: Path = Path(project_root / "data")
    processed=Path(data_path / "processed")
    logs=Path(project_root / 'logs')
    return Paths(
        processed=processed,
        logs=logs,
        log_file=Path(logs / "scale.log"),
        x_train=Path(processed / "X_train.csv"),
        x_test=Path(processed / "X_test.csv"),
        out_train=Path(processed / "X_train_scaled.csv"),
        out_test=Path(processed / "X_test_scaled.csv"),
    )


def _mk_dirs(paths: Paths) -> None:
    """Make the required dirs"""
    paths.logs.mkdir(parents=True, exist_ok=True)


def _load_data(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        X_train = pd.read_csv(paths.x_train)
        X_test = pd.read_csv(paths.x_test)
    except (FileNotFoundError, FileExistsError) as err:
        logging.error('Proccessed data files not found')
        raise IOError(f'{paths.processed} not found or is empty!') from err
    return X_train,X_test


def _export_normalized_data(
        paths: Paths, X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame
        ) -> None:
    logging.info("Sauvegarde X_train_scaled / X_test_scaled")
    X_train_scaled.to_csv(paths.out_train, index=False)
    X_test_scaled.to_csv(paths.out_test, index=False)


def _merge_scaled_features(
        x_train: pd.DataFrame, x_test: pd.DataFrame,non_num_cols: list,
        x_train_num_scaled: pd.DataFrame, x_test_num_scaled: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train_scaled = pd.concat([x_train_num_scaled, x_train[non_num_cols]], axis=1)
    x_test_scaled = pd.concat([x_test_num_scaled, x_test[non_num_cols]], axis=1)
    x_train_scaled = x_train_scaled[x_train.columns]
    x_test_scaled = x_test_scaled[x_test.columns]
    return x_train_scaled,x_test_scaled


def main():
    """
    normalization of the data
    """
    paths: Paths = _set_paths()
    _set_logger(paths)
    _mk_dirs(paths)


    X_train, X_test = _load_data(paths)

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    non_num_cols = [c for c in X_train.columns if c not in num_cols]

    logging.info("Scaled columns (%d): %s ", len(num_cols), num_cols)

    if non_num_cols:
        logging.info("Non-numeric columns left unchanged (%d): %s",
                     len(non_num_cols), non_num_cols)

    logging.info("StandardScaler fit/transform on numeric columns")
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]),
        columns=num_cols,
        index=X_train.index
    )
    X_test_num_scaled = pd.DataFrame(
        scaler.transform(X_test[num_cols]),
        columns=num_cols,
        index=X_test.index
    )

    X_train_scaled, X_test_scaled = _merge_scaled_features(
        X_train, X_test, non_num_cols, X_train_num_scaled, X_test_num_scaled)


    _export_normalized_data(paths, X_train_scaled, X_test_scaled)

    logging.info("Normalization finished.")





if __name__ == "__main__":
    main()