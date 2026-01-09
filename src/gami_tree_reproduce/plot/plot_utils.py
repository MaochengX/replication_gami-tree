import math
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from gami_tree_reproduce.utils import get_project_paths

project_paths = get_project_paths()


def get_plt_grid(len_objects: int, n_cols: int = 3, figsize=(12, 8)):
    """
    _summary_

    Args:
        len_objects (int): _description_
        n_cols (int, optional): _description_. Defaults to 3.
        figsize (tuple, optional): _description_. Defaults to (12, 8).

    Returns:
        _type_: _description_
    """
    n_rows = math.ceil(len_objects / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len_objects > 1 else [axes]

    return fig, axes


def get_modelname_from_path(path: Path) -> str:
    pass


def plot_response(
    dataset_paths,
    plot_method: str = "hist",
    ncols=2,
    plot_method_kwargs=None,
    subplots_kwargs=None,
) -> tuple[plt.figure, list[plt.axis]]:
    if subplots_kwargs is None:
        subplots_kwargs = {}
        # TODO: else check arg valid for subplots signature
    if plot_method_kwargs is None:
        plot_method_kwargs = {}

    nrows = math.ceil(len(dataset_paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, **subplots_kwargs)
    dataset_paths.sort()
    for counter, (dataset_path, ax) in enumerate(
        zip(dataset_paths, axes.flatten(), strict=True)
    ):
        data = pd.read_parquet(dataset_path)
        plot_func = getattr(ax, plot_method)
        plot_func(data["y"], **plot_method_kwargs)
        ax.set_title(f"Model {counter + 1}")
    return fig, axes


def save_fig(
    fig: plt.figure,
    filename: str,
    destination_folder: Path = project_paths["assets_plots"],
    save_kwargs=None,
) -> None:
    if save_kwargs is None:
        save_kwargs = {}

    destination_folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(destination_folder / Path(filename).with_suffix(".png"))
