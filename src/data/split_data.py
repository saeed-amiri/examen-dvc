"""
1- Data Splitting: Split the data into training and testing sets. Our
target variable is silica_concentrate, located in the last column of
the dataset. This script will produce 4 datasets:
    (X_test, X_train, y_test, y_train) that you can store in data/processed.
"""


from pathlib import Path
import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Paths:
    """Set the paths"""
    raw_data: Path
    processed: Path
    logs: Path
    log_file: Path

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
    logs=Path(project_root / 'logs')
    return Paths(
        raw_data=Path(data_path / "raw_data" / "raw.csv"),
        processed=Path(data_path / "processed"),
        logs=logs,
        log_file=Path(logs / "split.log")
    )


def _mk_dirs(paths: Paths) -> None:
    """Make the required dirs"""
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)


def _split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target_col = "silica_concentrate"
    date_col = "date"
    X = df.drop(columns=[target_col, date_col])
    y = df[target_col]
    return X, y


def _export_split_datasets(
        paths: Paths,
        x_train: pd.DataFrame,
        x_test:pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    logging.info("Writing split data into %s", paths.processed)
    x_train.to_csv(Path(paths.processed / "X_train.csv"), index=False)
    x_test.to_csv(Path(paths.processed / "X_test.csv"), index=False)
    y_train.to_csv(Path(paths.processed / "y_train.csv"), index=False, header=True)
    y_test.to_csv(Path(paths.processed / "y_test.csv"), index=False, header=True)


def _load_raw_data(paths: Paths) -> pd.DataFrame:
    logging.info("Loading raw data: %s", paths.raw_data)
    try:
        df = pd.read_csv(paths.raw_data)
        return df
    except (FileExistsError, FileNotFoundError) as err:
        logging.error('Raw data cannot be load!')
        raise IOError(f"{paths.raw_data} cannot be load!") from err


def main() -> None:
    """
    Split the raw data
    """
    paths: Paths = _set_paths()
    _mk_dirs(paths)
    _set_logger(paths)

    df: pd.DataFrame = _load_raw_data(paths)

    X, y = _split_features_and_target(df)

    logging.info("Split based on '80%/20%' rule")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    _export_split_datasets(paths, X_train, X_test, y_train, y_test)

    logging.info("Data is splitted.")





if __name__ == "__main__":

    main()
