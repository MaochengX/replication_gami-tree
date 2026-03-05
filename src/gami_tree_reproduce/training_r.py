import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.gami_tree_reproduce.model.gamitree import GAMITree

ROOT = Path.cwd()
CACHE_DIR = ROOT / "src" / "gami_tree_reproduce" / "cache"


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_xy(data_path: Path, task: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_parquet(data_path)
    x_cols = [c for c in df.columns if c.startswith("X_")]
    x_cols = sorted(x_cols, key=lambda s: int(s.split("_")[1]))
    X = df[x_cols].to_numpy(float)
    y = df["y"].to_numpy(float)

    if task == "classification":
        y_unique = np.unique(y[~np.isnan(y)])
        if not np.all(np.isin(y_unique, [0.0, 1.0])):
            raise ValueError(f"Classification expects y in {{0,1}}; got {y_unique[:20]}")
    return X, y, x_cols


def split_50_25_25(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n))
    rng.shuffle(idx)
    n_train = int(round(0.50 * n))
    n_valid = int(round(0.25 * n))
    tr = idx[:n_train]
    va = idx[n_train : n_train + n_valid]
    te = idx[n_train + n_valid :]
    return tr, va, te


def cache_path(tag: str, task: str) -> Path:
    return CACHE_DIR / f"gamitree_{tag}_{task}.joblib"


def paper_params(seed: int = 42) -> Dict:
    return dict(
        task="regression", 
        M=1000,
        R=5,
        lam=0.2,
        max_depth=2, 
        q=10,
        nknots=5,
        d=50,
        valid_frac=0.25,
        seed=seed,
        n_bins=64,
        min_leaf=20,
        alpha_grid=tuple(np.exp(np.linspace(-8.0, 0.0, 9))),
        max_coef=1.0,
        filter_max_depth=1,
        filter_n_bins=16,
        filter_alpha_grid=tuple(np.exp(np.linspace(-8.0, 0.0, 9))),
        filter_subsample=10**9,
        filter_prescreen_topk=0,
        verbose=True,
        show_progress=True,
    )


def _fmt_metrics(metrics: Dict) -> str:
    keys = ["loss", "mse", "rmse", "accuracy", "auc"]
    parts = []
    for k in keys:
        if k in metrics and metrics[k] is not None:
            v = metrics[k]
            if isinstance(v, float) and np.isnan(v):
                continue
            parts.append(f"{k}={float(v):.6f}")
    extras = sorted([k for k in metrics.keys() if k not in keys])
    for k in extras:
        v = metrics[k]
        if isinstance(v, float) and np.isnan(v):
            continue
        try:
            parts.append(f"{k}={float(v):.6f}")
        except Exception:
            parts.append(f"{k}={v}")
    return " ".join(parts) if parts else "(no metrics)"


def main() -> None:
    ensure_cache_dir()
    params = paper_params(seed=42)
    task = str(params["task"])

    TASKS = [
        ("model1_n500k_rho0", ROOT / "data" / "sim1_model1r.pq"),
        ("model2_n500k_rho0", ROOT / "data" / "sim1_model2r.pq"),
        ("model3_n500k_rho0", ROOT / "data" / "sim1_model3r.pq"),
        ("model4_n500k_rho0", ROOT / "data" / "sim1_model4r.pq"),

        ("model1_n500k_rho05", ROOT / "data" / "sim2_model1r.pq"),
        ("model2_n500k_rho05", ROOT / "data" / "sim2_model2r.pq"),
        ("model3_n500k_rho05", ROOT / "data" / "sim2_model3r.pq"),
        ("model4_n500k_rho05", ROOT / "data" / "sim2_model4r.pq"),

        ("model1_n50k_rho0", ROOT / "data" / "sim3_model1r.pq"),
        ("model2_n50k_rho0", ROOT / "data" / "sim3_model2r.pq"),
        ("model3_n50k_rho0", ROOT / "data" / "sim3_model3r.pq"),
        ("model4_n50k_rho0", ROOT / "data" / "sim3_model4r.pq"),

        ("model1_n50k_rho05", ROOT / "data" / "sim4_model1r.pq"),
        ("model2_n50k_rho05", ROOT / "data" / "sim4_model2r.pq"),
        ("model3_n50k_rho05", ROOT / "data" / "sim4_model3r.pq"),
        ("model4_n50k_rho05", ROOT / "data" / "sim4_model4r.pq"),
    ]

    rows = []
    for tag, path in TASKS:
        cp = cache_path(tag, task)
        tqdm.write(f"[{ts()}] === {tag} ({task}) ===")
        tqdm.write(f"[{ts()}] data={path}")
        tqdm.write(f"[{ts()}] cache={cp}")

        if cp.exists():
            obj = joblib.load(cp)
            model = obj["model"]
            meta = obj["meta"]
            loaded = True
        else:
            X, y, _x_cols = load_xy(path, task=task)
            tr, va, te = split_50_25_25(X.shape[0], seed=123)

            X_fit = X[np.concatenate([tr, va])]
            y_fit = y[np.concatenate([tr, va])]
            X_train, y_train = X[tr], y[tr]
            X_test, y_test = X[te], y[te]

            t0 = time.time()
            model = GAMITree(log=tqdm.write, **params).fit(X_fit, y_fit)
            seconds = float(time.time() - t0)

            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)

            meta = dict(
                tag=tag,
                task=task,
                params=dict(params),
                train_seconds=seconds,
                train_metrics=dict(train_metrics),
                test_metrics=dict(test_metrics),
                created_at=ts(),
                data_path=str(path),
            )
            joblib.dump({"model": model, "meta": meta}, cp)
            loaded = False

        train_metrics = meta.get("train_metrics", {})
        test_metrics = meta.get("test_metrics", {})

        tqdm.write(
            f"[{ts()}] {'LOADED' if loaded else 'TRAINED'} "
            f"train: {_fmt_metrics(train_metrics)} | "
            f"test: {_fmt_metrics(test_metrics)}"
        )

        rows.append(
            dict(
                dataset=tag,
                task=task,
                cached=loaded,
                seconds=float(meta.get("train_seconds", np.nan)),
                train_metrics=_fmt_metrics(train_metrics),
                test_metrics=_fmt_metrics(test_metrics),
            )
        )

    df = pd.DataFrame(rows)
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
