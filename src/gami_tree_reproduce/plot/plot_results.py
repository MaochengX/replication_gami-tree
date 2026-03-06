import math
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from gami_tree_reproduce.utils import get_project_paths

mpl.use("Agg")
NCOL = 3
FSIZE = (16, 12)
project_paths = get_project_paths()

ebm_effects = list(Path(project_paths["assets_effects"], "ebm").glob("*.pq"))
gaminet_effects = list(Path(project_paths["assets_effects"], "gaminet").glob("*.pq"))

ebm_importance = list(Path(project_paths["assets_importance"], "ebm").glob("*.pq"))
gaminet_importance = list(
    Path(project_paths["assets_importance"], "gaminet").glob("*.pq")
)


def plot_topk_importance(df_importance: pd.DataFrame, ax=None, k: int = 15) -> tuple:
    to_plot = df_importance.sort_values("importance", ascending=False).head(k)
    to_plot = to_plot.sort_values("importance", ascending=True)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.barh(to_plot["feature"], to_plot["importance"])

    return fig, ax


def save_importance(data_path_list: list[Path], full_out_path: Path) -> None:
    nrow = math.ceil(len(ebm_importance) / NCOL)
    fig, axes = plt.subplots(nrow, NCOL, figsize=FSIZE)
    axes = axes.flatten()
    for i, df_path in enumerate(data_path_list):
        df_importance = pd.read_parquet(df_path)
        ax = axes[i]
        plot_topk_importance(df_importance, ax=ax)
    fig.savefig(str(full_out_path), dpi=300)
    plt.close()


plots_importance_ebm = Path(project_paths["assets_plots_importance"], "ebm")
plots_importance_gaminet = Path(project_paths["assets_plots_importance"], "gaminet")

plots_importance_ebm.mkdir(exist_ok=True, parents=True)
plots_importance_gaminet.mkdir(exist_ok=True, parents=True)


def get_metadata(simulation_data_name: str):
    data_name, response_model = simulation_data_name.split("_")
    conf = Path(project_paths["assets_conf_data"], data_name).with_suffix(".yaml")
    with conf.open() as f:
        config = yaml.safe_load(f)

    sample_size = config["size"]
    correlation = config["cor"]
    task = response_model[4]
    task = "Regression" if task == "r" else "Classififcation"
    model_number = response_model[3]

    return (
        model_number,
        task,
        sample_size,
        correlation,
    )


def format_k(n):
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)


for path in ebm_importance:
    model_number, task, sample_size, correlation = get_metadata(path.stem)
    sample_size = format_k(sample_size)
    df_importance = pd.read_parquet(path)
    interaction_importance = df_importance[df_importance["feature"].str.contains("&")]
    main_importance = df_importance[~df_importance["feature"].str.contains("&")]

    fig, ax = plot_topk_importance(main_importance)
    title = f"Importance main effect (Model {model_number}, ebm)"
    subtitle = f"corr={correlation}, n={sample_size}"
    ax.set_title(subtitle)
    fig.suptitle(title)
    fig.savefig(
        Path(
            plots_importance_ebm,
            "ebm_main_" + path.stem + "_" + subtitle.replace(", ", "_"),
        ).with_suffix(".png"),
        dpi=300,
    )
    plt.close()

    fig, ax = plot_topk_importance(interaction_importance)
    title = f"Importance interaction effect (Model {model_number}, ebm)"
    ax.set_title(subtitle)
    fig.suptitle(title)
    fig.savefig(
        Path(
            plots_importance_ebm,
            "ebm_interact_" + path.stem + "_" + subtitle.replace(", ", "_"),
        ).with_suffix(".png"),
        dpi=300,
    )
    plt.close()


for path in gaminet_importance:
    model_number, task, sample_size, correlation = get_metadata(path.stem)

    df_importance = pd.read_parquet(path)
    interaction_importance = df_importance[df_importance["feature"].str.contains("&")]
    main_importance = df_importance[~df_importance["feature"].str.contains("&")]

    fig, ax = plot_topk_importance(main_importance)
    title = f"Importance main effect (Model {model_number}, gaminet)"
    subtitle = f"corr={correlation}, n={sample_size}"

    ax.set_title(subtitle)
    fig.suptitle(title)
    fig.savefig(
        Path(
            plots_importance_gaminet,
            "gaminet_main_" + path.stem + subtitle.replace(", ", "_"),
        ).with_suffix(".png"),
        dpi=300,
    )
    plt.close()

    fig, ax = plot_topk_importance(interaction_importance)
    title = f"Importance interaction effect (Model {model_number}, gaminet)"
    ax.set_title(subtitle)
    fig.suptitle(title)
    fig.savefig(
        Path(
            plots_importance_gaminet,
            "gaminet_interact_" + path.stem + subtitle.replace(", ", "_"),
        ).with_suffix(".png"),
        dpi=300,
    )
    plt.close()


plots_effects_ebm = Path(project_paths["assets_plots_effects"], "ebm")
plots_effects_gaminet = Path(project_paths["assets_plots_effects"], "gaminet")

df_effect_main = next(path for path in ebm_effects if "main" in str(path))
df_effect_main = pd.read_parquet(df_effect_main)
