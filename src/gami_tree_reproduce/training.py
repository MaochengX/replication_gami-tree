import time
from dataclasses import dataclass
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


def load_xy(data_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")
    df = pd.read_parquet(data_path)
    x_cols = [c for c in df.columns if c.startswith("X_")]
    x_cols = sorted(x_cols, key=lambda s: int(s.split("_")[1]))
    X = df[x_cols].to_numpy(float)
    y = df["y"].to_numpy(float)
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


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def cache_path(tag: str) -> Path:
    return CACHE_DIR / f"gamitree_{tag}.joblib"


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


@dataclass(frozen=True)
class TrainResult:
    model: GAMITree
    meta: Dict
    loaded: bool


def train_or_load(
    tag: str,
    data_path: Path,
    params: Dict,
    split_seed: int = 123,
) -> TrainResult:
    ensure_cache_dir()
    cp = cache_path(tag)
    if cp.exists():
        obj = joblib.load(cp)
        return TrainResult(model=obj["model"], meta=obj["meta"], loaded=True)

    X, y, x_cols = load_xy(data_path)
    tr, va, te = split_50_25_25(X.shape[0], seed=int(split_seed))

    X_fit = X[np.concatenate([tr, va])]
    y_fit = y[np.concatenate([tr, va])]
    X_train = X[tr]
    y_train = y[tr]
    X_test = X[te]
    y_test = y[te]

    t0 = time.time()
    model = GAMITree(log=tqdm.write, **params).fit(X_fit, y_fit)
    train_seconds = float(time.time() - t0)

    train_mse = mse(y_train, model.predict(X_train))
    test_mse = mse(y_test, model.predict(X_test))

    meta = dict(
        tag=str(tag),
        params=dict(params),
        x_cols=list(x_cols),
        train_seconds=float(train_seconds),
        train_mse=float(train_mse),
        test_mse=float(test_mse),
        data_path=str(data_path),
        split_seed=int(split_seed),
        created_at=ts(),
    )

    joblib.dump({"model": model, "meta": meta}, cp)
    return TrainResult(model=model, meta=meta, loaded=False)


def load_X_fit_from_meta(meta: Dict) -> Tuple[np.ndarray, List[str]]:
    X, _y, x_cols = load_xy(Path(meta["data_path"]))
    tr, va, _te = split_50_25_25(X.shape[0], seed=int(meta["split_seed"]))
    X_fit = X[np.concatenate([tr, va])]
    return X_fit, x_cols


def main() -> None:
    tasks = [
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

    params = paper_params(seed=42)

    rows = []
    for tag, path in tasks:
        tqdm.write(f"[{ts()}] === {tag} ===")
        tqdm.write(f"[{ts()}] data={path}")
        tqdm.write(f"[{ts()}] cache={cache_path(tag)}")

        res = train_or_load(tag, path, params, split_seed=123)

        rows.append(
            dict(
                dataset=tag,
                cached=res.loaded,
                train_mse=res.meta["train_mse"],
                test_mse=res.meta["test_mse"],
                seconds=res.meta["train_seconds"],
            )
        )

        tqdm.write(
            f"[{ts()}] {'LOADED' if res.loaded else 'TRAINED'} "
            f"train_mse={res.meta['train_mse']:.6f} "
            f"test_mse={res.meta['test_mse']:.6f} "
            f"seconds={res.meta['train_seconds']:.2f}"
        )

    df = pd.DataFrame(rows)
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
