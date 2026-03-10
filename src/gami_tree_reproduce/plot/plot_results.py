import math
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from gami_tree_reproduce.utils import get_project_paths

mpl.use("Agg")
NCOL = 4
FSIZE = (16, 17)
SUBPLOT_W = 4
SUBPLOT_H = 3
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
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

    to_plot["importance"] = (to_plot["importance"] - to_plot["importance"].min()) / (
        to_plot["importance"].max() - to_plot["importance"].min()
    )
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
    return str(int(n))


# ------------------------------
#            Importance plots
# -----------------------------
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
        Path(plots_importance_ebm, "ebm_main_" + path.stem).with_suffix(".png"),
        dpi=300,
    )
    plt.close()

    fig, ax = plot_topk_importance(interaction_importance)
    title = f"Importance interaction effect (Model {model_number}, ebm)"
    ax.set_title(subtitle)
    fig.suptitle(title)
    fig.savefig(
        Path(plots_importance_ebm, "ebm_interact_" + path.stem).with_suffix(".png"),
        dpi=300,
    )
    plt.close()


for path in gaminet_importance:
    model_number, task, sample_size, correlation = get_metadata(path.stem)
    sample_size = format_k(sample_size)
    df_importance = pd.read_parquet(path)
    interaction_importance = df_importance[df_importance["feature"].str.contains("&")]
    main_importance = df_importance[~df_importance["feature"].str.contains("&")]

    fig, ax = plot_topk_importance(main_importance)
    title = f"Importance main effect (Model {model_number}, gaminet)"
    subtitle = f"corr={correlation}, n={sample_size}"

    ax.set_title(subtitle)
    fig.suptitle(title)
    fig.savefig(
        Path(plots_importance_gaminet, "gaminet_main_" + path.stem).with_suffix(".png"),
        dpi=300,
    )
    plt.close()

    fig, ax = plot_topk_importance(interaction_importance)
    title = f"Importance interaction effect (Model {model_number}, gaminet)"
    ax.set_title(subtitle)
    fig.suptitle(title)
    fig.savefig(
        Path(plots_importance_gaminet, "gaminet_interact_" + path.stem).with_suffix(
            ".png"
        ),
        dpi=300,
    )
    plt.close()


plots_effects_ebm = Path(project_paths["assets_plots_effects"], "ebm")
plots_effects_ebm.mkdir(parents=True, exist_ok=True)
plots_effects_gaminet = Path(project_paths["assets_plots_effects"], "gaminet")
plots_effects_gaminet.mkdir(parents=True, exist_ok=True)


def get_metadata(foldername: str) -> tuple:
    with (
        Path(project_paths["assets_conf_data"], foldername.split("_", maxsplit=1)[0])
        .with_suffix(".yaml")
        .open() as f
    ):
        data_config = yaml.safe_load(f)

    correlation = data_config["cor"]
    samplesize = data_config["size"]
    task = "Regression" if foldername[-1] == "r" else "Classificarion"
    model_number = foldername[-2]

    return correlation, samplesize, model_number, task


def plot_interaction_effect(df_effects_row, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    feature_names = df_effects_row["feature"].split("&")
    feature_1_name = feature_names[0]
    feature_2_name = feature_names[1]
    interact_matrix = np.array([elem.tolist() for elem in df_effects_row["effect"]])
    grid_feature_2 = df_effects_row["grid"][1]
    selection = [-2.3, -1.3, -0.3, 0.3, 1.3, 2.3]
    # get entries closest to selection values
    idx_feature_1 = np.abs(df_effects_row.grid[0][:, None] - selection).argmin(axis=0)
    interact_selection = interact_matrix[idx_feature_1]

    for feature1_value, interaction_effects in zip(
        selection, interact_selection.tolist(), strict=False
    ):
        ax.plot(
            grid_feature_2,
            interaction_effects,
            label=f"{feature_1_name}={feature1_value}",
        )
        ax.set_xlabel(f"{feature_2_name}")
        ax.set_ylabel(f"f({feature_1_name}, {feature_2_name})")

    return fig, ax


def plot_main_effect(df_effects_row, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    feature_name = df_effects_row["feature"].iloc[0]
    grid = df_effects_row["grid"].iloc[0][0]
    grid = grid[1:-1]
    effect = df_effects_row["effect"].iloc[0][0]
    effect = effect[1:-1]
    ax.plot(grid, effect)
    ax.set_ylabel(feature_name)

    return fig, ax


def remove_no_effects(df_effects):
    idx_blacklist = []
    for idx, row in df_effects.iterrows():
        effects = row["effect"].flatten()
        # in case of matrix concatenate for testing all entries
        effects = np.concatenate(effects) if len(effects) > 1 else effects[0]

        if np.all(effects == 0):
            idx_blacklist.append(idx)

    return df_effects.drop(index=idx_blacklist)


# ------------------------------
#            Effect plots
# -----------------------------
for path in gaminet_effects + ebm_effects:
    # ---------------------------------------------
    # general setup
    # --------------------------------------------
    inducer_name = path.parent.name
    df_effect = pd.read_parquet(path)
    df_effect = remove_no_effects(df_effect)
    df_interaction_effects = df_effect[df_effect["feature"].str.contains("&")]
    df_main_effects = df_effect[~df_effect["feature"].str.contains("&")]
    corr, samplesize, model_number, task = get_metadata(path.stem)

    samplesize = format_k(samplesize)
    subtitle = f"corr={corr}, n={samplesize}"
    folder_name = path.stem  #  + "_" + subtitle
    if inducer_name == "gaminet":
        plot_folder = Path(plots_effects_gaminet, folder_name.replace(", ", "_"))
    elif inducer_name == "ebm":
        plot_folder = Path(plots_effects_ebm, folder_name.replace(", ", "_"))
    plot_folder.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------
    # interaction effects
    # --------------------------------------------
    nrow = max(1, math.ceil(len(df_interaction_effects) / NCOL))
    title = f"Interaction effect (Model {model_number}, {inducer_name})"
    fig, axes = plt.subplots(
        nrow,
        NCOL,
        figsize=(SUBPLOT_W * NCOL, SUBPLOT_H * nrow),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for row_idx in range(df_interaction_effects.shape[0]):
        axis = axes[row_idx]
        plot_interaction_effect(df_interaction_effects.iloc[row_idx], axis)

        fig_single, ax_single = plot_interaction_effect(
            df_interaction_effects.iloc[row_idx], None
        )
        ax_single.legend(loc="upper right")
        ax_single.set_title(title + "\n" + subtitle)

        feature1, feature2 = df_interaction_effects.iloc[row_idx]["feature"].split("&")
        plotname = f"interact_{feature1}-{feature2}"
        fig_single.savefig(Path(plot_folder, plotname).with_suffix(".png"), dpi=300)
        plt.close()

    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    for ax in axes:
        if not ax.has_data():
            ax.set_visible(False)
    fig.suptitle(title + "\n" + subtitle)
    fig.savefig(Path(plot_folder, "interact_all").with_suffix(".png"), dpi=300)
    plt.close()
    # #---------------------------------------------
    # # main effects
    # #--------------------------------------------
    nrow = max(1, math.ceil(len(df_main_effects) / NCOL))
    title = f"Main effect (Model {model_number}, {inducer_name})"
    fig, axes = plt.subplots(
        nrow,
        NCOL,
        figsize=(SUBPLOT_W * NCOL, SUBPLOT_H * nrow),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for row_idx in range(df_main_effects.shape[0]):
        axis = axes[row_idx]
        df_subset = df_main_effects.iloc[[row_idx]]

        plot_main_effect(df_subset, axis)
        fig_single, ax_single = plot_main_effect(df_subset, ax=None)
        feature_name = df_subset["feature"].iloc[0]
        plotname = f"main_{feature_name}"
        fig_single.savefig(Path(plot_folder, plotname).with_suffix(".png"), dpi=300)
        plt.close()

    fig.suptitle(title + "\n" + subtitle)
    for ax in axes:
        if not ax.has_data():
            fig.delaxes(ax)
    fig.savefig(Path(plot_folder, "main_all").with_suffix(".png"))
    plt.close()
