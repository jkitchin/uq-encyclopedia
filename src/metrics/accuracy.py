"""Accuracy metrics for point predictions."""

import numpy as np
from typing import Optional


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    mse : float
        Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    rmse : float
        Root mean squared error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    mae : float
        Mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Median Absolute Error (MdAE).

    More robust to outliers than MAE.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    mdae : float
        Median absolute error
    """
    return np.median(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot

    where:
    - SS_res = sum of squared residuals
    - SS_tot = total sum of squares

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    r2 : float
        R² score (1.0 is perfect, can be negative)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0 if ss_res == 0 else -np.inf

    return 1.0 - (ss_res / ss_tot)


def adjusted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_params: int,
) -> float:
    """
    Compute adjusted R² score.

    Accounts for the number of parameters in the model.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    n_params : int
        Number of model parameters

    Returns
    -------
    adj_r2 : float
        Adjusted R² score
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)

    if n <= n_params + 1:
        return -np.inf

    return 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1)


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    epsilon : float
        Small constant to avoid division by zero

    Returns
    -------
    mape : float
        Mean absolute percentage error (in percentage)
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute maximum absolute error.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    max_err : float
        Maximum absolute error
    """
    return np.max(np.abs(y_true - y_pred))


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute explained variance score.

    EV = 1 - Var(y_true - y_pred) / Var(y_true)

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    ev : float
        Explained variance score
    """
    var_residual = np.var(y_true - y_pred)
    var_true = np.var(y_true)

    if var_true == 0:
        return 0.0 if var_residual == 0 else -np.inf

    return 1.0 - (var_residual / var_true)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute bias (mean error).

    Positive bias indicates systematic over-prediction.
    Negative bias indicates systematic under-prediction.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    bias : float
        Bias (mean error)
    """
    return np.mean(y_pred - y_true)


def normalized_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str = 'range',
) -> float:
    """
    Compute normalized RMSE.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    normalization : str
        Normalization method: 'range', 'std', or 'mean'

    Returns
    -------
    nrmse : float
        Normalized RMSE
    """
    rmse = root_mean_squared_error(y_true, y_pred)

    if normalization == 'range':
        norm = np.ptp(y_true)  # Peak-to-peak
    elif normalization == 'std':
        norm = np.std(y_true)
    elif normalization == 'mean':
        norm = np.mean(y_true)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if norm == 0:
        return 0.0 if rmse == 0 else np.inf

    return rmse / norm
