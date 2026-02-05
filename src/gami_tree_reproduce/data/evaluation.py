import re
from pathlib import Path

import pandas as pd
import yaml

from gami_tree_reproduce.utils import get_project_paths

project_paths = get_project_paths()


def get_metadata_from_experiment(
    config_name: str, source_folder=project_paths["assets_conf_data"]
):
    if config_name not in [folder.stem for folder in source_folder.iterdir()]:
        msg = f"No configuration asset for dataset '{config_name}'"
        raise KeyError(msg)

    with Path(source_folder, config_name).with_suffix(".yaml").open("r") as f:
        conf = yaml.safe_load(f)

    if "size" not in conf:
        msg = f"Expected to find entry 'size' in config asset {config_name}"
        raise KeyError(msg)

    samplesize = conf["size"]
    correlation = conf["cor"]
    return samplesize, correlation


def gather_results(result_directories: list[Path]) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            "model": pd.Series(dtype="string"),
            "task": pd.Series(dtype="string"),
            "inducer": pd.Series(dtype="string"),
            "samplesize": pd.Series(dtype="int64"),
            "correlation": pd.Series(dtype="float64"),
            "loss_train": pd.Series(dtype="float64"),
            "loss_test": pd.Series(dtype="float64"),
            "time_train": pd.Series(dtype="float64"),
            "time_inference": pd.Series(dtype="float64"),
        }
    )

    for folder in result_directories:
        with Path(folder, "results.yaml").open("r") as f:
            conf = yaml.safe_load(f)

        # naming convention for regex
        match = re.search(r"_([^_]+)_config", str(folder.stem))
        inducer = match.group(1)

        match = re.search(r"mod(\d)(\w)", str(folder.stem))
        model, task = (int(match.group(1)), match.group(2))
        if task == "c":
            task = "classification"
        elif task == "r":
            task = "regression"
        else:
            msg = f"folder {folder.stem!s} name not in format 'mod<modelnr><task-character>'"
            raise KeyError(msg)

        match = re.search(r"(sim\d+)", str(folder.stem))
        simulation_data = match.group(1)

        n, corr = get_metadata_from_experiment(simulation_data)
        loss_train = conf["loss_train"]
        loss_test = conf["loss_test"]

        time_train = conf["time_train"]
        time_inference = conf["time_predict"]

        experiment = pd.DataFrame(
            {
                "model": [model],
                "task": [task],
                "inducer": [inducer],
                "samplesize": [n],
                "correlation": [corr],
                "loss_train": [loss_train],
                "loss_test": [loss_test],
                "time_train": [time_train],
                "time_inference": [time_inference],
            }
        )
        metadata = pd.concat([metadata, experiment], ignore_index=True)
    return metadata


experiment_results = [
    item for item in project_paths["assets_conf_experiments"].iterdir() if item.is_dir()
]
metadata = gather_results(experiment_results)
result_folder = Path(project_paths["assets"], "experiment")
result_folder.mkdir(exist_ok=True, parents=True)
metadata.write_csv(Path(result_folder, "results.csv"))
