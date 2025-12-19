from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit


def model1(X) -> np.ndarray:
    term1 = X[:, 0:5].sum(axis=1)

    term2 = 0.5 * np.sum(X[:, 5:8] ** 2, axis=1)

    idx = np.where(X[:, 8:10] > 0, X[:, 8:10], 0)
    term3 = idx.sum(axis=1)

    term4 = 0.2 * np.sum(X * X, axis=1)

    return term1 + term2 + term3 + term4


def model2(X) -> np.ndarray:
    # term1: sum_{j=1..5} x_j
    term1 = X[:, 0:5].sum(axis=1)

    # term2: sum_{j=6..8} 0.5 * x_j^2
    term2 = 0.5 * np.sum(X[:, 5:8] ** 2, axis=1)

    # term3: sum_{j=9..10} x_j * 1(x_j > 0)
    term3 = np.sum(np.where(X[:, 8:10] > 0, X[:, 8:10], 0), axis=1)

    # term4: 0.25 * x1*x2
    term4 = 0.25 * X[:, 0] * X[:, 1]

    # term5: 0.25 * x1 * x3^2
    term5 = 0.25 * X[:, 0] * (X[:, 2] ** 2)

    # term6: 0.25 * x4^2 * x5^2
    term6 = 0.25 * (X[:, 3] ** 2) * (X[:, 4] ** 2)

    # term7: exp(x4 * x6 / 3)
    term7 = np.exp(X[:, 3] * X[:, 5] / 3)

    # term8: x5*x6 * 1(x5>0) * 1(x6>0)
    term8 = X[:, 4] * X[:, 5] * (X[:, 4] > 0) * (X[:, 5] > 0)

    # term9: clip(x7 + x8, -1, 0)
    term9 = np.clip(X[:, 6] + X[:, 7], -1, 0)

    # term10: clip(x7 * x9, -1, 1)
    term10 = np.clip(X[:, 6] * X[:, 8], -1, 1)

    # term11: 1(x8>0) * 1(x9>0)
    term11 = (X[:, 7] > 0) * (X[:, 8] > 0)

    return (
        term1
        + term2
        + term3
        + term4
        + term5
        + term6
        + term7
        + term8
        + term9
        + term10
        + term11
    )


def model3(X) -> np.ndarray:
    # term1: sum_{j=1..5} x_j
    term1 = X[:, 0:5].sum(axis=1)

    # term2: sum_{j=6..8} 0.5 * x_j^2
    term2 = 0.5 * np.sum(X[:, 5:8] ** 2, axis=1)

    # term3: sum_{j=9..10} x_j * 1(x_j > 0)
    term3 = np.sum(np.where(X[:, 8:10] > 0, X[:, 8:10], 0), axis=1)

    # term4: 0.25 * x1^2 * x2^2
    term4 = 0.25 * (X[:, 0] ** 2) * (X[:, 1] ** 2)

    # term5: 2 * (x3 - 0.5)_+
    term5 = 2 * np.maximum(X[:, 2] - 0.5, 0)

    # term6: (x4 - 0.5)_+
    term6 = np.maximum(X[:, 3] - 0.5, 0)

    # term7: 0.5 * sin(pi*x5) * sin(pi*x6)
    term7 = 0.5 * np.sin(np.pi * X[:, 4]) * np.sin(np.pi * X[:, 5])

    # term8: 0.5 * sin(pi*(x7 + x8))
    term8 = 0.5 * np.sin(np.pi * (X[:, 6] + X[:, 7]))

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8


def model4(X) -> np.ndarray:
    # term1: sum_{j=1..5} x_j
    term1 = X[:, 0:5].sum(axis=1)

    # term2: sum_{j=6..8} 0.5 * x_j**2
    term2 = 0.5 * np.sum(X[:, 5:8] ** 2, axis=1)

    # term3: sum_{j=9..10} x_j * 1(x_j > 0)
    term3 = np.sum(np.where(X[:, 8:10] > 0, X[:, 8:10], 0), axis=1)

    # shorthand variables for readability
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    x4, x5, x6 = X[:, 3], X[:, 4], X[:, 5]

    # term4: interactions x1x2 + x1x3 + x2x3
    term4 = x1 * x2 + x1 * x3 + x2 * x3

    # term5: 0.5 * x1*x2*x3
    term5 = 0.5 * x1 * x2 * x3

    # term6: x4x5 + x4x6 + x5x6
    term6 = x4 * x5 + x4 * x6 + x5 * x6

    # term7: 0.5 * 1(x4>0) * x5 * x6
    term7 = 0.5 * (x4 > 0) * x5 * x6

    return term1 + term2 + term3 + term4 + term5 + term6 + term7


def set_y(
    X: pd.DataFrame,
    task: str,
    model_func: Callable,
    y_generator,
    y_generator_params,
    rng: np.random.default_rng,
) -> pd.DataFrame:
    """
    Depending whether the task is 'classification' or 'regression' the response has to be adjusted.

    Args:
        y (pd.Series): _description_
        task (str): _description_

    Returns:
        pd.Series: _description_
    """
    gx = model_func(X)
    if task == "regression":
        y_generator = getattr(
            rng, y_generator
        )  # get generator constructor by name, eg. "normal"
        error_term = y_generator(**y_generator_params)
        y = gx + error_term
    elif task == "classification":
        pi = 0.5

        def f(beta_0):
            p = expit(gx + beta_0)  # log sigmoid
            return np.average(p) - pi

        # since sigmoid +- 50 is roughly 0,1 in between must be the optimal value
        optimal_betal_0 = brentq(f, -50.0, 50)
        p = expit(gx + optimal_betal_0)
        y = rng.binomial(n=1, p=p, size=gx.size)
    else:
        raise KeyError

    data = pd.DataFrame(X, columns=[f"X_{idx}" for idx in range(1, X.shape[1] + 1)])
    data["y"] = y
    return data
