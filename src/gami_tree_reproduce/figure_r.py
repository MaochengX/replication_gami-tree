import time
from pathlib import Path
from typing import Optional, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.gami_tree_reproduce.training_r import cache_path, load_xy, split_50_25_25


ROOT = Path.cwd()
OUTDIR = ROOT / "figures_regression"


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def load_cached(tag: str, task: str = "regression"):
    cp = cache_path(tag, task)
    if not cp.exists():
        raise FileNotFoundError(f"Missing cache for tag={tag}: {cp}. Run training first.")
    obj = joblib.load(cp)
    return obj["model"], obj["meta"]


def load_X_fit_from_meta(meta):
    X, y, x_cols = load_xy(Path(meta["data_path"]), task=meta["task"])
    tr, va, te = split_50_25_25(X.shape[0], seed=123)
    idx_fit = np.concatenate([tr, va])
    X_fit = X[idx_fit]
    return X_fit, x_cols


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def idx(x_cols: Sequence[str], xname: str) -> int:
    if xname not in x_cols:
        raise KeyError(f"{xname} not in columns")
    return x_cols.index(xname)


def pretty_n(n_text: str) -> str:
    if n_text.endswith("k"):
        return f"{float(n_text[:-1]):.1f}K"
    return n_text


def rho_text(rho_token: str) -> str:
    return "0.5" if rho_token == "rho05" else "0"


def plot_top15_main_importance(model, X, x_cols, outpath: Path, title: str):
    imp = np.asarray(model.main_importance(X), float)
    order = np.argsort(imp)[::-1][:15]
    labels = [x_cols[i].replace("_", "") for i in order]
    vals = imp[order]

    plt.figure(figsize=(8, 6))
    y = np.arange(len(labels))
    plt.barh(y, vals[::-1])
    plt.yticks(y, labels[::-1])
    plt.title(title)
    savefig(outpath)
    return imp


def plot_main_effect(model, X, x_cols, xname: str, outpath: Path, title: str = None):
    j = idx(x_cols, xname)

    x = X[:, j]
    order = np.argsort(x)
    xs = x[order]
    fx = model.main_component(X, j)[order]

    if title is None:
        title = f"Main effect plot for GAMI-Tree: {xname.replace('_','')}"

    plt.figure(figsize=(7, 4))
    plt.plot(xs, fx)

    plt.xlabel(xname.replace("_", ""))
    plt.ylabel("main effect")
    plt.title(title)

    savefig(outpath)


def plot_top10_interaction_importance(model, X, x_cols, outpath: Path, title: str):
    imp = model.interaction_importance(X)
    items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:10]
    labels = [f"{x_cols[a].replace('_', '')}&{x_cols[b].replace('_', '')}" for (a, b), _v in items]
    vals = np.array([v for (_pair, v) in items], float)

    plt.figure(figsize=(9, 6))
    y = np.arange(len(labels))
    plt.barh(y, vals[::-1])
    plt.yticks(y, labels[::-1])
    plt.title(title)
    savefig(outpath)
    return items


def plot_interaction_slices(model, X, x_cols, a: int, b: int, outpath: Path, n_slices: int = 7, title: Optional[str] = None, swap_axes: bool = False,):
    if not swap_axes:
        x_axis, slice_axis = a, b
    else:
        x_axis, slice_axis = b, a

    xj = X[:, x_axis]
    xk = X[:, slice_axis]

    qs = np.linspace(0.1, 0.9, n_slices)
    qvals = np.quantile(xk, qs)

    xj_grid = np.linspace(
        np.quantile(xj, 0.01),
        np.quantile(xj, 0.99),
        300,
    )

    base = np.mean(X, axis=0, keepdims=True)

    plt.figure(figsize=(8, 5))
    for qv in qvals:
        Xg = np.repeat(base, xj_grid.size, axis=0)
        Xg[:, x_axis] = xj_grid
        Xg[:, slice_axis] = qv
        fg = model.interaction_component(Xg, a, b)
        plt.plot(xj_grid, fg, label=f"{x_cols[slice_axis]}={qv:.2g}")

    plt.xlabel(x_cols[x_axis])
    plt.ylabel("interaction effect")
    plt.legend()

    if title:
        plt.title(title)

    savefig(outpath)


def main() -> None:
    ensure_dirs()
    print(f"[{ts()}] figures dir: {OUTDIR}")

    models = ["model1", "model2", "model3", "model4"]
    ns = ["n50k", "n500k"]
    rhos = ["rho0", "rho05"]
    task = "regression"

    tags = []
    for model_name in models:
        for n_name in ns:
            for rho_name in rhos:
                tags.append(
                    dict(
                        model=model_name,
                        n=n_name,
                        rho=rho_name,
                        task=task,
                        tag=f"{model_name}_{n_name}_{rho_name}",
                    )
                )

    pbar = tqdm(tags, desc="Generating figures", unit="model")

    for spec in pbar:
        model_name = spec["model"]
        n_name = spec["n"]
        rho_name = spec["rho"]
        task_name = spec["task"]
        tag = spec["tag"]

        pbar.set_postfix({"tag": tag})

        model, meta = load_cached(tag, task=task_name)
        X_fit, x_cols = load_X_fit_from_meta(meta)

        fig_dir = OUTDIR / tag
        fig_dir.mkdir(parents=True, exist_ok=True)

        model_num = model_name.replace("model", "")
        corr = rho_text(rho_name)
        n_text = pretty_n(n_name.replace("n", ""))

        main_title = f"Importance main effect (Model {model_num}, gamitree)\ncorr={corr}, n={n_text}"
        inter_title = f"Importance interaction effect (Model {model_num}, gamitree)\ncorr={corr}, n={n_text}"
        slice_title = "Interaction effect plot for GAMI-Tree"

        plot_top15_main_importance(
            model,
            X_fit,
            x_cols,
            fig_dir / f"{tag}_main_importance.png",
            main_title,
        )

        top_pairs = plot_top10_interaction_importance(
            model,
            X_fit,
            x_cols,
            fig_dir / f"{tag}_interaction_importance.png",
            inter_title,
        )

        for xname in x_cols:
            plot_main_effect(
                model,
                X_fit,
                x_cols,
                xname,
                fig_dir / f"{tag}_main_effect_{xname}.png",
                title=None,
            )

        for (a, b), _v in top_pairs:
            plot_interaction_slices(
                model,
                X_fit,
                x_cols,
                a,
                b,
                fig_dir / f"{tag}_interaction_{x_cols[a]}_{x_cols[b]}.png",
                n_slices=7,
                title=slice_title,
                swap_axes=True,
            )

        print(f"[{ts()}] done: {tag} -> {fig_dir}")

    print(f"[{ts()}] all done.")


if __name__ == "__main__":
    main()