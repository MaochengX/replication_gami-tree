from os import environ as oenv
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from gami_tree_reproduce.utils import get_project_paths

project_paths = get_project_paths()
config = Path(oenv["PROJECT_ROOT"])
config = OmegaConf.load(Path("conf/config.yaml"))
OmegaConf.resolve(config)
seed = config.seed


def make_train_val_test(
    source_file: Path, destination_folder: Path, test_size, val_size
) -> None:
    data_name = source_file.stem

    test_size = config["test_size"]
    val_size = config["val_size"]

    data = pd.read_parquet(source_file)
    data_train_val, data_test = train_test_split(
        data, test_size=test_size, shuffle=True, random_state=seed
    )

    val_size = val_size / (1 - test_size)
    data_train, data_val = train_test_split(
        data_train_val, test_size=val_size, shuffle=True, random_state=seed
    )

    destination_subfolder = destination_folder / data_name
    Path(destination_subfolder).mkdir(exist_ok=True, parents=True)

    data_train.to_parquet(
        Path(destination_subfolder / Path(data_name + "_train")).with_suffix(".pq")
    )
    data_test.to_parquet(
        Path(destination_subfolder / Path(data_name + "_test")).with_suffix(".pq")
    )
    data_val.to_parquet(
        Path(destination_subfolder / Path(data_name + "_val")).with_suffix(".pq")
    )


def get_train_val_test(
    subfolder: str | Path | None = None,
    source: Path = project_paths["data_preprocessed"],
    tuples=True,
):
    if isinstance(subfolder, str):  # only name provided and default path assumed
        assert Path(source / subfolder).exists()
        source = source / subfolder
    else:  # explicitly folder path given
        assert source.exists()

    train_file = source / f"{subfolder}_train.pq"
    test_file = source / f"{subfolder}_test.pq"
    val_file = source / f"{subfolder}_val.pq"

    assert train_file.exists()
    assert test_file.exists()
    assert val_file.exists()

    data_train = pd.read_parquet(train_file)
    data_val = pd.read_parquet(val_file)
    data_test = pd.read_parquet(test_file)

    if not tuples:
        return data_train, data_val, data_test

    X_train, y_train = data_train.drop(columns="y"), data_train["y"]
    X_val, y_val = data_val.drop(columns="y"), data_val["y"]
    X_test, y_test = data_test.drop(columns="y"), data_test["y"]

    return (
        (X_train.to_numpy(), y_train.to_numpy()),
        (X_val.to_numpy(), y_val.to_numpy()),
        (X_test.to_numpy(), y_test.to_numpy()),
    )


def assert_preprocess_data_available(
    folder: Path = project_paths["data_preprocessed"],
) -> Path:
    """
    For each pq file in raw folder there should be a corresponding train, test, val pq file in preprocessed folder
    """
    subfolders = [p for p in folder.iterdir() if p.is_dir()]

    msg = f"Expected subfolders (with train, val, test splits) inside {folder} but found none."
    assert len(subfolders) >= 1, msg

    for subfolder in subfolders:
        train_files = list(subfolder.glob("*_train.pq"))
        val_files = list(subfolder.glob("*_val.pq"))
        test_files = list(subfolder.glob("*_test.pq"))

        assert len(train_files) == 1, f"Expected exactly one train file in {subfolder}"
        assert len(val_files) == 1, f"Expected exactly one val file in {subfolder}"
        assert len(test_files) == 1, f"Expected exactly one test file in {subfolder}"

    return folder
