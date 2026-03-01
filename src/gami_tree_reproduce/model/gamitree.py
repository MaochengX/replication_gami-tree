from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import SplineTransformer

Task = Literal["regression", "classification"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def train_valid_split(n: int, valid_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n))
    rng.shuffle(idx)
    n_valid = int(np.round(n * float(valid_frac)))
    n_valid = max(1, min(n - 1, n_valid))
    return idx[n_valid:], idx[:n_valid]


def safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * np.eye(A.shape[0], dtype=A.dtype)
        return np.linalg.solve(A + jitter, b)


def loss_derivatives(task: Task, y: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y, float).reshape(-1)
    f = np.asarray(f, float).reshape(-1)

    if task == "regression":
        r = f - y
        G = 2.0 * r
        H = np.full_like(G, 2.0, dtype=float)
        L = float(np.mean(r * r))
        return G, H, L

    y01 = y
    if not np.all((y01 == 0.0) | (y01 == 1.0)):
        raise ValueError("For task='classification', y must be binary in {0,1}.")

    softplus = np.maximum(f, 0.0) + np.log1p(np.exp(-np.abs(f)))
    L = float(np.mean(softplus - y01 * f))
    p = sigmoid(f)
    G = p - y01
    H = p * (1.0 - p)
    H = np.clip(H, 1e-6, None)
    return G, H, L


def pseudo_response(G: np.ndarray, H: np.ndarray) -> np.ndarray:
    return -np.asarray(G, float) / np.asarray(H, float)


@dataclass(frozen=True)
class Gram:
    XtWX: np.ndarray
    XtWz: np.ndarray
    ztWz: float
    n: int


def gram(Phi: np.ndarray, z: np.ndarray, w: np.ndarray) -> Gram:
    Phi = np.asarray(Phi, float)
    z = np.asarray(z, float).reshape(-1)
    w = np.asarray(w, float).reshape(-1)
    sw = np.sqrt(w)
    Xw = Phi * sw[:, None]
    zw = z * sw
    XtWX = Xw.T @ Xw
    XtWz = Xw.T @ zw
    ztWz = float(zw @ zw)
    return Gram(XtWX=XtWX, XtWz=XtWz, ztWz=ztWz, n=int(z.size))


def gcv_from_gram(XtWX: np.ndarray, XtWz: np.ndarray, ztWz: float, lam: float, n: float) -> Tuple[float, np.ndarray, float]:
    d = XtWX.shape[0]
    A = XtWX + float(lam) * np.eye(d)
    beta = safe_solve(A, XtWz)
    SSE = float(ztWz - 2.0 * (beta @ XtWz) + beta @ (XtWX @ beta))
    M = safe_solve(A, XtWX)
    edf = float(np.trace(M))
    denom = max(float(n) - edf, 1e-8)
    return float(SSE / (denom * denom)), beta, float(SSE)


def maxcoef_ok(beta: np.ndarray, sd: np.ndarray, max_coef: float) -> bool:
    if not np.isfinite(max_coef):
        return True
    beta = np.asarray(beta, float)
    if beta.size <= 1:
        return True
    scaled = np.abs(beta[1:]) * np.asarray(sd, float)
    return bool(np.all(scaled <= float(max_coef) + 1e-12))


def fit_leaf(Phi: np.ndarray, z: np.ndarray, w: np.ndarray, alpha_grid: Sequence[float], max_coef: float) -> Tuple[np.ndarray, float]:
    g = gram(Phi, z, w)
    best_beta: Optional[np.ndarray] = None
    best_gcv = float("inf")
    best_sse = float("inf")
    sd = np.std(np.asarray(Phi, float)[:, 1:], axis=0, ddof=0) if Phi.shape[1] > 1 else np.array([], float)
    for lam in alpha_grid:
        score, beta, sse = gcv_from_gram(g.XtWX, g.XtWz, g.ztWz, float(lam), float(g.n))
        if not np.isfinite(score):
            continue
        if not maxcoef_ok(beta, sd, max_coef):
            continue
        if score < best_gcv:
            best_gcv = float(score)
            best_beta = beta
            best_sse = float(sse)
    if best_beta is None:
        lam = float(alpha_grid[-1])
        _score, best_beta, best_sse = gcv_from_gram(g.XtWX, g.XtWz, g.ztWz, lam, float(g.n))
    return best_beta, float(best_sse)


def bin_edges_quantile(x: np.ndarray, B: int) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([-np.inf, np.inf], float)
    qs = np.linspace(0.0, 1.0, int(B) + 1)
    edges = np.quantile(x, qs, method="linear")
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([-np.inf, np.inf], float)
    return edges


def bin_assign(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1)
    b = np.searchsorted(edges, x, side="right") - 1
    return np.clip(b, 0, edges.size - 2).astype(int)


def binned_stats(
    Phi: np.ndarray, z: np.ndarray, w: np.ndarray, bins: np.ndarray, nb: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Phi = np.asarray(Phi, float)
    z = np.asarray(z, float).reshape(-1)
    w = np.asarray(w, float).reshape(-1)
    bins = np.asarray(bins, int).reshape(-1)
    _n, d = Phi.shape
    wz = w * z
    wzz = w * z * z
    Sxz = np.zeros((nb, d), float)
    Szz = np.zeros(nb, float)
    cnt = np.zeros(nb, int)
    sum1 = np.zeros((nb, d - 1), float) if d > 1 else np.zeros((nb, 0), float)
    sum2 = np.zeros((nb, d - 1), float) if d > 1 else np.zeros((nb, 0), float)
    Sxx = np.zeros((nb, d, d), float)
    np.add.at(Sxz, bins, (Phi * wz[:, None]))
    np.add.at(Szz, bins, wzz)
    np.add.at(cnt, bins, 1)
    if d > 1:
        P1 = Phi[:, 1:]
        np.add.at(sum1, bins, P1)
        np.add.at(sum2, bins, P1 * P1)
    for a in range(d):
        for b in range(d):
            np.add.at(Sxx[:, a, b], bins, w * Phi[:, a] * Phi[:, b])
    return Sxx, Sxz, Szz, cnt, sum1, sum2


def sd_from_sums(sum1: np.ndarray, sum2: np.ndarray, n: int) -> np.ndarray:
    if sum1.size == 0:
        return np.array([], float)
    n = max(int(n), 1)
    mean = sum1 / float(n)
    var = (sum2 / float(n)) - mean * mean
    var = np.maximum(var, 0.0)
    return np.sqrt(var)


def fit_leaf_from_gram(
    XtWX: np.ndarray,
    XtWz: np.ndarray,
    ztWz: float,
    n: int,
    alpha_grid: Sequence[float],
    max_coef: float,
    sd: np.ndarray,
) -> Tuple[np.ndarray, float]:
    best_beta: Optional[np.ndarray] = None
    best_gcv = float("inf")
    best_sse = float("inf")
    for lam in alpha_grid:
        score, beta, sse = gcv_from_gram(XtWX, XtWz, ztWz, float(lam), float(n))
        if not np.isfinite(score):
            continue
        if not maxcoef_ok(beta, sd, max_coef):
            continue
        if score < best_gcv:
            best_gcv = float(score)
            best_beta = beta
            best_sse = float(sse)
    if best_beta is None:
        lam = float(alpha_grid[-1])
        _score, best_beta, best_sse = gcv_from_gram(XtWX, XtWz, ztWz, lam, float(n))
    return best_beta, float(best_sse)


def best_split_binned(
    x_split: np.ndarray,
    Phi: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    n_bins: int,
    min_leaf: int,
    alpha_grid: Sequence[float],
    max_coef: float,
) -> Tuple[float, Optional[float]]:
    x_split = np.asarray(x_split, float).reshape(-1)
    Phi = np.asarray(Phi, float)
    z = np.asarray(z, float).reshape(-1)
    w = np.asarray(w, float).reshape(-1)

    edges = bin_edges_quantile(x_split, int(n_bins))
    bins = bin_assign(x_split, edges)
    nb = int(edges.size - 1)
    if nb <= 1:
        return float("inf"), None

    Sxx, Sxz, Szz, cnt, sum1, sum2 = binned_stats(Phi, z, w, bins, nb)

    cSxx = np.cumsum(Sxx, axis=0)
    cSxz = np.cumsum(Sxz, axis=0)
    cSzz = np.cumsum(Szz, axis=0)
    ccnt = np.cumsum(cnt, axis=0)
    csum1 = np.cumsum(sum1, axis=0) if sum1.size else sum1
    csum2 = np.cumsum(sum2, axis=0) if sum2.size else sum2

    totSxx = cSxx[-1]
    totSxz = cSxz[-1]
    totSzz = float(cSzz[-1])
    totcnt = int(ccnt[-1])
    totsum1 = csum1[-1] if sum1.size else sum1
    totsum2 = csum2[-1] if sum2.size else sum2

    best = float("inf")
    best_thr: Optional[float] = None

    for cut in range(nb - 1):
        nL = int(ccnt[cut])
        nR = int(totcnt - nL)
        if nL < int(min_leaf) or nR < int(min_leaf):
            continue

        XL = cSxx[cut]
        XzL = cSxz[cut]
        zzL = float(cSzz[cut])

        XR = totSxx - XL
        XzR = totSxz - XzL
        zzR = float(totSzz - zzL)

        sdL = sd_from_sums(csum1[cut] if sum1.size else sum1, csum2[cut] if sum2.size else sum2, nL)
        sdR = sd_from_sums(
            (totsum1 - (csum1[cut] if sum1.size else sum1)) if sum1.size else sum1,
            (totsum2 - (csum2[cut] if sum2.size else sum2)) if sum2.size else sum2,
            nR,
        )

        _bL, sseL = fit_leaf_from_gram(XL, XzL, zzL, nL, alpha_grid, max_coef, sdL)
        _bR, sseR = fit_leaf_from_gram(XR, XzR, zzR, nR, alpha_grid, max_coef, sdR)
        sse = float(sseL + sseR)
        if sse < best:
            best = sse
            best_thr = float(edges[cut + 1])

    return float(best), best_thr


@dataclass(frozen=True)
class Node:
    t: Optional[float]
    beta: Optional[np.ndarray]
    L: Optional["Node"] = None
    R: Optional["Node"] = None


@dataclass(frozen=True)
class MainTree:
    j: int
    root: Node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        xj = X[:, self.j]
        out = np.zeros(X.shape[0], dtype=float)
        stack: List[Tuple[Node, np.ndarray]] = [(self.root, np.arange(X.shape[0]))]
        while stack:
            v, I = stack.pop()
            if I.size == 0:
                continue
            if v.t is None:
                Phi = np.column_stack([np.ones(I.size, float), xj[I]])
                out[I] = Phi @ v.beta
                continue
            thr = float(v.t)
            IL = I[xj[I] <= thr]
            IR = I[xj[I] > thr]
            if v.L is not None:
                stack.append((v.L, IL))
            if v.R is not None:
                stack.append((v.R, IR))
        return out


@dataclass(frozen=True)
class IntTree:
    j: int
    k: int
    Bj: SplineTransformer
    root: Node
    def basis(self, xj: np.ndarray) -> np.ndarray:
        Bj = self.Bj.transform(np.asarray(xj, float).reshape(-1, 1))
        return np.concatenate([np.ones((Bj.shape[0], 1), float), Bj], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        xj = X[:, self.j]
        xk = X[:, self.k]
        out = np.zeros(X.shape[0], dtype=float)
        stack: List[Tuple[Node, np.ndarray]] = [(self.root, np.arange(X.shape[0]))]
        while stack:
            v, I = stack.pop()
            if I.size == 0:
                continue
            if v.t is None:
                Phi = self.basis(xj[I])
                out[I] = Phi @ v.beta
                continue
            thr = float(v.t)
            IL = I[xk[I] <= thr]
            IR = I[xk[I] > thr]
            if v.L is not None:
                stack.append((v.L, IL))
            if v.R is not None:
                stack.append((v.R, IR))
        return out


def fit_main_tree(
    xj: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    max_depth: int,
    n_bins: int,
    min_leaf: int,
    alpha_grid: Sequence[float],
    max_coef: float,
) -> Tuple[Node, float]:
    xj = np.asarray(xj, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    w = np.asarray(w, float).reshape(-1)
    Phi = np.column_stack([np.ones(xj.size, float), xj])
    beta, sse_leaf = fit_leaf(Phi, z, w, alpha_grid, max_coef)
    if int(max_depth) <= 0 or xj.size < 2 * int(min_leaf):
        return Node(t=None, beta=beta), float(sse_leaf)
    best_sse, thr = best_split_binned(xj, Phi, z, w, n_bins, min_leaf, alpha_grid, max_coef)
    if thr is None or best_sse >= sse_leaf:
        return Node(t=None, beta=beta), float(sse_leaf)
    m = xj <= float(thr)
    L, sseL = fit_main_tree(xj[m], z[m], w[m], max_depth - 1, n_bins, min_leaf, alpha_grid, max_coef)
    R, sseR = fit_main_tree(xj[~m], z[~m], w[~m], max_depth - 1, n_bins, min_leaf, alpha_grid, max_coef)
    return Node(t=float(thr), beta=None, L=L, R=R), float(sseL + sseR)


def fit_int_tree(
    x_model: np.ndarray,
    x_split: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    tr: SplineTransformer,
    max_depth: int,
    n_bins: int,
    min_leaf: int,
    alpha_grid: Sequence[float],
    max_coef: float,
) -> Tuple[Node, float]:
    x_model = np.asarray(x_model, float).reshape(-1)
    x_split = np.asarray(x_split, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    w = np.asarray(w, float).reshape(-1)
    B = tr.transform(x_model.reshape(-1, 1))
    Phi = np.concatenate([np.ones((B.shape[0], 1), float), B], axis=1)
    beta, sse_leaf = fit_leaf(Phi, z, w, alpha_grid, max_coef)
    if int(max_depth) <= 0 or x_split.size < 2 * int(min_leaf):
        return Node(t=None, beta=beta), float(sse_leaf)
    best_sse, thr = best_split_binned(x_split, Phi, z, w, n_bins, min_leaf, alpha_grid, max_coef)
    if thr is None or best_sse >= sse_leaf:
        return Node(t=None, beta=beta), float(sse_leaf)
    m = x_split <= float(thr)
    L, sseL = fit_int_tree(x_model[m], x_split[m], z[m], w[m], tr, max_depth - 1, n_bins, min_leaf, alpha_grid, max_coef)
    R, sseR = fit_int_tree(x_model[~m], x_split[~m], z[~m], w[~m], tr, max_depth - 1, n_bins, min_leaf, alpha_grid, max_coef)
    return Node(t=float(thr), beta=None, L=L, R=R), float(sseL + sseR)


@dataclass(frozen=True)
class Spline1D:
    j: int
    tr: SplineTransformer
    gamma: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        x = X[:, self.j]
        B = self.tr.transform(x.reshape(-1, 1))
        Phi = np.concatenate([np.ones((B.shape[0], 1), float), B], axis=1)
        return Phi @ self.gamma


@dataclass(frozen=True)
class RoundSnapshot:
    main_end: int
    int_end: int
    pair_splines: Dict[Tuple[int, int], Tuple[Spline1D, Spline1D]]


@dataclass
class GAMITreeParams:
    M: int = 1000
    max_depth: int = 2
    lam: float = 0.2
    R: int = 5
    q: int = 10
    nknots: int = 5
    alpha_grid: Sequence[float] = tuple(np.exp(np.linspace(-8.0, 0.0, 9)))
    max_coef: float = 1.0
    d: int = 50
    valid_frac: float = 0.25
    seed: int = 0
    n_bins: int = 64
    min_leaf: int = 20
    filter_subsample: int = 10**9
    filter_max_depth: int = 1
    filter_n_bins: int = 16
    filter_alpha_grid: Sequence[float] = tuple(np.exp(np.linspace(-8.0, 0.0, 9)))
    filter_prescreen_topk: int = 0
    verbose: bool = True
    show_progress: bool = True


def tqdm_factory():
    try:
        from tqdm import tqdm

        return tqdm
    except Exception:
        return None


class GAMITree:
    def __init__(self, task: Task = "regression", log: Optional[Callable[[str], None]] = None, **params):
        self.task: Task = task
        if 'max_depth' not in params or params.get('max_depth') is None:
            params = dict(params)
            params['max_depth'] = 2 if task == 'regression' else 1
        self.params = GAMITreeParams(**params)
        self.log = log if log is not None else (lambda s: print(s, flush=True))
        self.base: float = 0.0
        self.main_trees: List[MainTree] = []
        self.int_trees: List[IntTree] = []
        self.pair_splines: Dict[Tuple[int, int], Tuple[Spline1D, Spline1D]] = {}
        self.rounds: List[RoundSnapshot] = []
        self.fitted: bool = False
        self.tqdm = tqdm_factory()

    def say(self, msg: str) -> None:
        if bool(self.params.verbose):
            self.log(str(msg))

    def iter_wrap(self, it, **kwargs):
        if bool(self.params.show_progress) and self.tqdm is not None:
            return self.tqdm(it, **kwargs)
        return it

    def init_base(self, y: np.ndarray) -> float:
        y = np.asarray(y, float).reshape(-1)
        if self.task == "regression":
            return float(np.mean(y))
        p = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        return float(np.log(p / (1.0 - p)))

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        f = np.full(X.shape[0], float(self.base), dtype=float)
        lam = float(self.params.lam)
        for t in self.main_trees:
            f += lam * t.predict(X)
        for t in self.int_trees:
            f += lam * t.predict(X)
        for hj, hk in self.pair_splines.values():
            f += hj.predict(X)
            f += hk.predict(X)
        return f

    def predict(self, X: np.ndarray) -> np.ndarray:
        f = self.predict_raw(X)
        if self.task == "regression":
            return f
        return (sigmoid(f) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        f = self.predict_raw(X)
        if self.task == "regression":
            return f
        return sigmoid(f)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        f = self.predict_raw(X)
        _G, _H, L = loss_derivatives(self.task, y, f)
        return float(L)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X = np.asarray(X, float)
        y = np.asarray(y).reshape(-1)
        out: Dict[str, float] = {}
        out["loss"] = self.loss(X, y)

        if self.task == "regression":
            pred = self.predict_raw(X)
            mse = float(np.mean((pred - y.astype(float)) ** 2))
            out["mse"] = mse
            out["rmse"] = float(np.sqrt(mse))
            return out

        proba = self.predict_proba(X).reshape(-1)
        pred = (proba >= 0.5).astype(int)
        y01 = y.astype(int)
        out["accuracy"] = float(np.mean(pred == y01))

        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y01)) == 2:
                out["auc"] = float(roc_auc_score(y01, proba))
        except Exception:
            pass
        return out

    def main_component_round(self, X: np.ndarray, j: int, r: int) -> np.ndarray:
        X = np.asarray(X, float)
        lam = float(self.params.lam)
        out = np.zeros(X.shape[0], dtype=float)
        main_end = len(self.main_trees) if r <= 0 or not self.rounds else self.rounds[r - 1].main_end
        pair_splines = self.pair_splines if r <= 0 or not self.rounds else self.rounds[r - 1].pair_splines
        for t in self.main_trees[:main_end]:
            if int(t.j) == int(j):
                out += lam * t.predict(X)
        for (a, b), (ha, hb) in pair_splines.items():
            if int(a) == int(j):
                out += ha.predict(X)
            if int(b) == int(j):
                out += hb.predict(X)
        return out

    def main_component(self, X: np.ndarray, j: int) -> np.ndarray:
        return self.main_component_round(X, j, r=len(self.rounds) if self.rounds else 0)

    def interaction_component_round(self, X: np.ndarray, j: int, k: int, r: int) -> np.ndarray:
        X = np.asarray(X, float)
        a, b = sorted((int(j), int(k)))
        lam = float(self.params.lam)
        out = np.zeros(X.shape[0], dtype=float)
        int_end = len(self.int_trees) if r <= 0 or not self.rounds else self.rounds[r - 1].int_end
        pair_splines = self.pair_splines if r <= 0 or not self.rounds else self.rounds[r - 1].pair_splines
        for t in self.int_trees[:int_end]:
            aa, bb = sorted((int(t.j), int(t.k)))
            if (aa, bb) == (a, b):
                out += lam * t.predict(X)
        if (a, b) in pair_splines:
            ha, hb = pair_splines[(a, b)]
            out -= ha.predict(X)
            out -= hb.predict(X)
        return out

    def interaction_component(self, X: np.ndarray, j: int, k: int) -> np.ndarray:
        return self.interaction_component_round(X, j, k, r=len(self.rounds) if self.rounds else 0)

    def main_importance(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        p = int(X.shape[1])
        imp = np.zeros(p, dtype=float)
        for j in range(p):
            imp[j] = float(np.std(self.main_component(X, j), ddof=0))
        return imp

    def interaction_importance(self, X: np.ndarray) -> Dict[Tuple[int, int], float]:
        X = np.asarray(X, float)
        keys = set()
        for t in self.int_trees:
            a, b = sorted((int(t.j), int(t.k)))
            keys.add((a, b))
        out: Dict[Tuple[int, int], float] = {}
        for a, b in sorted(keys):
            out[(a, b)] = float(np.std(self.interaction_component(X, a, b), ddof=0))
        return out

    def fit_main_stage(self, X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, r: int, R: int) -> int:
        p = int(X.shape[1])
        M = int(self.params.M)
        d = int(self.params.d)
        f_tr = self.predict_raw(X[tr])
        f_va = self.predict_raw(X[va])
        start = len(self.main_trees)
        losses: List[float] = []
        loop = self.iter_wrap(range(1, M + 1), desc=f"Round {r}/{R} FitMain", unit="tree", leave=False)
        for m in loop:
            G, H, _ = loss_derivatives(self.task, y[tr], f_tr)
            z = pseudo_response(G, H)
            best_tree = None
            best_sse = float("inf")
            for j in range(p):
                node, sse = fit_main_tree(
                    X[tr, j],
                    z,
                    H,
                    max_depth=int(self.params.max_depth),
                    n_bins=int(self.params.n_bins),
                    min_leaf=int(self.params.min_leaf),
                    alpha_grid=self.params.alpha_grid,
                    max_coef=float(self.params.max_coef),
                )
                if sse < best_sse:
                    best_sse = float(sse)
                    best_tree = MainTree(j=int(j), root=node)
            if best_tree is None:
                return 0
            self.main_trees.append(best_tree)
            lam = float(self.params.lam)
            f_tr = f_tr + lam * best_tree.predict(X[tr])
            f_va = f_va + lam * best_tree.predict(X[va])
            _, _, Lm = loss_derivatives(self.task, y[va], f_va)
            losses.append(float(Lm))
            if hasattr(loop, "set_postfix"):
                loop.set_postfix({"m": m, "main": len(self.main_trees), "val": f"{Lm:.4f}"})
            if len(losses) >= d + 1:
                L_m_d = losses[-(d + 1)]
                if L_m_d < min(losses[-d:]):
                    keep = len(losses) - d
                    self.main_trees = self.main_trees[: start + keep]
                    return int(keep)
        return int(len(self.main_trees) - start)

    def corr_abs(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, float).reshape(-1)
        b = np.asarray(b, float).reshape(-1)
        a = a - np.mean(a)
        b = b - np.mean(b)
        da = float(np.sqrt(np.mean(a * a)))
        db = float(np.sqrt(np.mean(b * b)))
        if da <= 0.0 or db <= 0.0:
            return 0.0
        return float(abs(np.mean(a * b) / (da * db)))

    def filter_interactions(self, X: np.ndarray, y: np.ndarray, tr: np.ndarray, r: int, R: int) -> List[Tuple[int, int]]:
        p = int(X.shape[1])
        rng = np.random.default_rng(int(self.params.seed) + 999)
        m = min(int(tr.size), int(self.params.filter_subsample))
        tr_sub = tr if m >= tr.size else rng.choice(tr, size=m, replace=False)
        f_sub = self.predict_raw(X[tr_sub])
        G, H, _ = loss_derivatives(self.task, y[tr_sub], f_sub)
        z = pseudo_response(G, H)
        trs: List[SplineTransformer] = []
        for j in range(p):
            trj = SplineTransformer(degree=1, n_knots=int(self.params.nknots), knots="quantile", include_bias=False)
            trj.fit(X[tr_sub, j].reshape(-1, 1))
            trs.append(trj)
        pairs0: List[Tuple[float, int, int]] = []
        topk = int(self.params.filter_prescreen_topk)
        for j in range(p - 1):
            xj = X[tr_sub, j]
            for k in range(j + 1, p):
                xk = X[tr_sub, k]
                s0 = self.corr_abs(z, xj * xk)
                pairs0.append((float(s0), int(j), int(k)))
        pairs0.sort(key=lambda t: t[0], reverse=True)
        if topk > 0:
            pairs0 = pairs0[: min(topk, len(pairs0))]
        loop = self.iter_wrap(pairs0, desc=f"Round {r}/{R} FilterInt", unit="pair", leave=False)
        scores: List[Tuple[float, int, int]] = []
        for idx, (_s0, j, k) in enumerate(loop, start=1):
            trj = trs[j]
            trk = trs[k]
            _node, sse_jk = fit_int_tree(
                X[tr_sub, j],
                X[tr_sub, k],
                z,
                H,
                trj,
                max_depth=int(self.params.filter_max_depth),
                n_bins=int(self.params.filter_n_bins),
                min_leaf=int(self.params.min_leaf),
                alpha_grid=self.params.filter_alpha_grid,
                max_coef=float(self.params.max_coef),
            )
            _node, sse_kj = fit_int_tree(
                X[tr_sub, k],
                X[tr_sub, j],
                z,
                H,
                trk,
                max_depth=int(self.params.filter_max_depth),
                n_bins=int(self.params.filter_n_bins),
                min_leaf=int(self.params.min_leaf),
                alpha_grid=self.params.filter_alpha_grid,
                max_coef=float(self.params.max_coef),
            )
            sse = float(min(sse_jk, sse_kj))
            scores.append((sse, int(j), int(k)))
            if hasattr(loop, "set_postfix"):
                loop.set_postfix({"done": idx, "topk": len(pairs0), "best_sse": f"{min(s[0] for s in scores):.2f}"})
        scores.sort(key=lambda t: t[0])
        top = scores[: int(self.params.q)]
        Q: List[Tuple[int, int]] = []
        for _sse, j, k in top:
            Q.append((int(j), int(k)))
            Q.append((int(k), int(j)))
        return Q

    def fit_int_stage(self, X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, Q: List[Tuple[int, int]], r: int, R: int) -> int:
        if not Q:
            return 0
        M = int(self.params.M)
        d = int(self.params.d)
        f_tr = self.predict_raw(X[tr])
        f_va = self.predict_raw(X[va])
        start = len(self.int_trees)
        losses: List[float] = []
        trs: Dict[int, SplineTransformer] = {}
        for j, _k in Q:
            if int(j) not in trs:
                trj = SplineTransformer(degree=1, n_knots=int(self.params.nknots), knots="quantile", include_bias=False)
                trj.fit(X[tr, int(j)].reshape(-1, 1))
                trs[int(j)] = trj
        loop = self.iter_wrap(range(1, M + 1), desc=f"Round {r}/{R} FitInt", unit="tree", leave=False)
        for m in loop:
            G, H, _ = loss_derivatives(self.task, y[tr], f_tr)
            z = pseudo_response(G, H)
            best_tree = None
            best_sse = float("inf")
            for j, k in Q:
                trj = trs[int(j)]
                node, sse = fit_int_tree(
                    X[tr, j],
                    X[tr, k],
                    z,
                    H,
                    trj,
                    max_depth=int(self.params.max_depth),
                    n_bins=int(self.params.n_bins),
                    min_leaf=int(self.params.min_leaf),
                    alpha_grid=self.params.alpha_grid,
                    max_coef=float(self.params.max_coef),
                )
                if sse < best_sse:
                    best_sse = float(sse)
                    best_tree = IntTree(j=int(j), k=int(k), Bj=trj, root=node)
            if best_tree is None:
                return 0
            self.int_trees.append(best_tree)
            lam = float(self.params.lam)
            f_tr = f_tr + lam * best_tree.predict(X[tr])
            f_va = f_va + lam * best_tree.predict(X[va])
            _, _, Lm = loss_derivatives(self.task, y[va], f_va)
            losses.append(float(Lm))
            if hasattr(loop, "set_postfix"):
                loop.set_postfix({"m": m, "int": len(self.int_trees), "val": f"{Lm:.4f}"})
            if len(losses) >= d + 1:
                L_m_d = losses[-(d + 1)]
                if L_m_d < min(losses[-d:]):
                    keep = len(losses) - d
                    self.int_trees = self.int_trees[: start + keep]
                    return int(keep)
        return int(len(self.int_trees) - start)

    def purify_from_state(self, X: np.ndarray, tr: np.ndarray, int_end: int) -> Dict[Tuple[int, int], Tuple[Spline1D, Spline1D]]:
        if int_end <= 0:
            return {}
        groups: Dict[Tuple[int, int], List[IntTree]] = {}
        for t in self.int_trees[:int_end]:
            a, b = sorted((int(t.j), int(t.k)))
            groups.setdefault((a, b), []).append(t)
        Xt = np.asarray(X[tr], float)
        lam = float(self.params.lam)
        out: Dict[Tuple[int, int], Tuple[Spline1D, Spline1D]] = {}
        keys = list(groups.keys())
        loop = self.iter_wrap(keys, desc="Purify", unit="pair", leave=False)
        for (a, b) in loop:
            trees = groups[(a, b)]
            yhat = np.zeros(Xt.shape[0], dtype=float)
            for t in trees:
                yhat += lam * t.predict(Xt)
            tr_a = SplineTransformer(degree=1, n_knots=int(self.params.nknots), knots="quantile", include_bias=False)
            tr_b = SplineTransformer(degree=1, n_knots=int(self.params.nknots), knots="quantile", include_bias=False)
            Ba = tr_a.fit_transform(Xt[:, a].reshape(-1, 1))
            Bb = tr_b.fit_transform(Xt[:, b].reshape(-1, 1))
            Phi = np.concatenate([np.ones((Xt.shape[0], 1), float), Ba, Bb], axis=1)
            beta, _sse = fit_leaf(Phi, yhat, np.ones(Xt.shape[0], float), self.params.alpha_grid, float(self.params.max_coef))
            na = Ba.shape[1]
            ga = beta[: 1 + na]
            gb = np.concatenate([[0.0], beta[1 + na :]], axis=0)
            ha = Spline1D(j=int(a), tr=tr_a, gamma=ga)
            hb = Spline1D(j=int(b), tr=tr_b, gamma=gb)
            out[(int(a), int(b))] = (ha, hb)
            if hasattr(loop, "set_postfix"):
                loop.set_postfix({"pairs": len(out)})
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAMITree":
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        n = int(X.shape[0])
        tr, va = train_valid_split(n, float(self.params.valid_frac), int(self.params.seed))
        self.base = self.init_base(y[tr])
        self.rounds = []
        R = int(self.params.R)
        for r in range(1, R + 1):
            self.say(f"Round {r}/{R}: FitMain")
            m1 = self.fit_main_stage(X, y, tr, va, r=r, R=R)
            self.say(f"Round {r}/{R}: FilterInt")
            Q = self.filter_interactions(X, y, tr, r=r, R=R)
            self.say(f"Round {r}/{R}: FitInt  Q={len(Q)}")
            m2 = self.fit_int_stage(X, y, tr, va, Q, r=r, R=R)
            self.say(f"Round {r}/{R}: Purify")
            main_end = len(self.main_trees)
            int_end = len(self.int_trees)
            pair_splines_r = self.purify_from_state(X, tr, int_end=int_end)
            self.rounds.append(RoundSnapshot(main_end=main_end, int_end=int_end, pair_splines=pair_splines_r))
            if m1 == 0 and m2 == 0:
                break
        self.pair_splines = self.rounds[-1].pair_splines if self.rounds else {}
        self.fitted = True
        return self
