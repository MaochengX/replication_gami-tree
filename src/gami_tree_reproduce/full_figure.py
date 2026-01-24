import time
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.gami_tree_reproduce.training import CACHE_DIR, cache_path, load_X_fit_from_meta


ROOT = Path.cwd()
OUTDIR = ROOT / "figures"


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def load_cached(tag: str):
    cp = cache_path(tag)
    if not cp.exists():
        raise FileNotFoundError(f"Missing cache for tag={tag}: {cp}. Run training first.")
    obj = joblib.load(cp)
    return obj["model"], obj["meta"]


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTDIR / name, dpi=200)
    plt.close()


def idx(x_cols, xname: str) -> int:
    if xname not in x_cols:
        raise KeyError(f"{xname} not in columns")
    return x_cols.index(xname)


def rel01(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, float)
    m = float(np.max(vals)) if vals.size else 0.0
    if m <= 0.0:
        return np.zeros_like(vals)
    return vals / m


def plot_top15_main_importance(model, X, x_cols, fname: str, title: str):
    imp = model.main_importance(X)
    order = np.argsort(imp)[::-1][:15]
    labels = [x_cols[i] for i in order]
    vals = rel01(imp[order])
    y = np.arange(len(labels))[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(y, vals[::-1])
    plt.yticks(y, labels[::-1])
    plt.xlim(0.0, 1.0)
    plt.xlabel("relative importance")
    plt.ylabel("x")
    plt.title(title)
    savefig(fname)
    return imp


def plot_main_effect(model, X, x_cols, xname: str, fname: str, title: Optional[str]):
    j = idx(x_cols, xname)
    x = X[:, j]
    order = np.argsort(x)
    xs = x[order]
    fx = model.main_component(X, j)[order]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, fx)
    plt.xlabel(xname.lower())
    plt.ylabel(f"f({xname.lower()})")
    if title is not None and str(title).strip():
        plt.title(title)
    savefig(fname)


def plot_top10_interaction_importance(model, X, x_cols, fname: str, title: str):
    imp = model.interaction_importance(X)
    items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:10]
    labels = [f"{x_cols[a]}×{x_cols[b]}" for (a, b), _v in items]
    vals = rel01(np.array([v for (_p, v) in items], float))
    y = np.arange(len(labels))[::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(y, vals[::-1])
    plt.yticks(y, labels[::-1])
    plt.xlim(0.0, 1.0)
    plt.xlabel("relative importance")
    plt.ylabel("pair")
    plt.title(title)
    savefig(fname)
    return items


def plot_interaction_slices(model, X, x_cols, a: int, b: int, fname: str, n_slices: int = 7, title: Optional[str] = None):
    xj = X[:, a]
    xk = X[:, b]
    qs = np.linspace(0.1, 0.9, n_slices)
    qvals = np.quantile(xk, qs)
    xj_grid = np.linspace(np.quantile(xj, 0.01), np.quantile(xj, 0.99), 300)
    base = np.mean(X, axis=0, keepdims=True)
    plt.figure(figsize=(8, 5))
    for qv in qvals:
        Xg = np.repeat(base, xj_grid.size, axis=0)
        Xg[:, a] = xj_grid
        Xg[:, b] = qv
        fg = model.interaction_component(Xg, a, b)
        plt.plot(xj_grid, fg, label=f"{x_cols[b]}={qv:.2g}")
    plt.xlabel(x_cols[a])
    plt.ylabel("interaction effect")
    plt.legend()
    if title is not None and str(title).strip():
        plt.title(title)
    savefig(fname)


def main() -> None:
    ensure_dirs()

    tags = dict(
        model4_rho0="model4_n50k_rho0",
        model4_rho05="model4_n50k_rho05",
        model2_rho05="model2_n50k_rho05",
        model3_rho05="model3_n50k_rho05",
    )

    tqdm.write(f"[{ts()}] Cache dir: {CACHE_DIR}")

    model_r0, meta_r0 = load_cached(tags["model4_rho0"])
    X_fit_r0, xcols_r0 = load_X_fit_from_meta(meta_r0)

    plot_top15_main_importance(
        model_r0,
        X_fit_r0,
        xcols_r0,
        "Figure2_main_importance_model4_rho0.png",
        "the top 15 main effects of Gamitree",
    )
    plot_main_effect(model_r0, X_fit_r0, xcols_r0, "X_11", "Figure3_fx11_model4_rho0.png", title=None)
    plot_main_effect(model_r0, X_fit_r0, xcols_r0, "X_19", "Figure3_fx19_model4_rho0.png", title=None)
    plot_main_effect(model_r0, X_fit_r0, xcols_r0, "X_26", "Figure3_fx26_model4_rho0.png", title=None)

    model_r05, meta_r05 = load_cached(tags["model4_rho05"])
    X_fit_r05, xcols_r05 = load_X_fit_from_meta(meta_r05)

    plot_top15_main_importance(
        model_r05,
        X_fit_r05,
        xcols_r05,
        "Figure4_main_importance_model4_rho05.png",
        "top 15 main effects of Gamitree",
    )
    plot_main_effect(model_r05, X_fit_r05, xcols_r05, "X_18", "Figure5_fx18_model4_rho05.png", title=None)
    plot_main_effect(model_r05, X_fit_r05, xcols_r05, "X_9", "Figure6_fx9_model4_rho05.png", title=None)
    plot_main_effect(model_r05, X_fit_r05, xcols_r05, "X_10", "Figure6_fx10_model4_rho05.png", title=None)

    model_m2, meta_m2 = load_cached(tags["model2_rho05"])
    X_fit_m2, xcols_m2 = load_X_fit_from_meta(meta_m2)

    top_pairs_m2 = plot_top10_interaction_importance(
        model_m2,
        X_fit_m2,
        xcols_m2,
        "Figure7_interaction_importance_model2_rho05.png",
        "top 10 interaction pairs",
    )

    model_m3, meta_m3 = load_cached(tags["model3_rho05"])
    X_fit_m3, xcols_m3 = load_X_fit_from_meta(meta_m3)

    plot_top10_interaction_importance(
        model_m3,
        X_fit_m3,
        xcols_m3,
        "Figure8_interaction_importance_model3_rho05.png",
        "top 10 interaction pairs",
    )

    pbar = tqdm(top_pairs_m2, desc="Figure9 Model2 interaction effects", unit="pair")
    for (a, b), _v in pbar:
        name = f"Figure9_model2_{xcols_m2[a]}_{xcols_m2[b]}.png"
        pbar.set_postfix({"pair": f"{xcols_m2[a]}x{xcols_m2[b]}"})
        plot_interaction_slices(
            model_m2,
            X_fit_m2,
            xcols_m2,
            a,
            b,
            name,
            n_slices=7,
            title="Interaction effect plot for GAMI-Tree",
        )

    tqdm.write(f"[{ts()}] Done. Figures in: {OUTDIR}")


if __name__ == "__main__":
    main()
