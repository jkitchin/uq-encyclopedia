"""Coverage and calibration metrics for uncertainty quantification."""

import numpy as np
from typing import Tuple, Optional
from src.uq_methods.base import UncertaintyResult


def prediction_interval_coverage_probability(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Compute Prediction Interval Coverage Probability (PICP).

    PICP is the fraction of true values that fall within the prediction intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals

    Returns
    -------
    picp : float
        Coverage probability (between 0 and 1)
    """
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(covered)


def picp(y_true: np.ndarray, result: UncertaintyResult) -> float:
    """
    Compute PICP from UncertaintyResult.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    result : UncertaintyResult
        Uncertainty quantification results

    Returns
    -------
    picp : float
        Coverage probability
    """
    return prediction_interval_coverage_probability(
        y_true, result.y_lower, result.y_upper
    )


def coverage_by_bin(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    X: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute coverage probability in bins across the input space.

    Useful for checking if coverage is uniform across the domain.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals
    X : np.ndarray, optional
        Input features. If None, bins are based on data order
    n_bins : int
        Number of bins

    Returns
    -------
    bin_edges : np.ndarray
        Edges of bins
    bin_coverage : np.ndarray
        Coverage probability in each bin
    bin_counts : np.ndarray
        Number of samples in each bin
    """
    if X is None:
        # Use indices as bins
        X = np.arange(len(y_true)).reshape(-1, 1)

    X_flat = X.flatten()

    # Create bins
    bin_edges = np.linspace(X_flat.min(), X_flat.max(), n_bins + 1)
    bin_indices = np.digitize(X_flat, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute coverage in each bin
    bin_coverage = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    covered = (y_true >= y_lower) & (y_true <= y_upper)

    for i in range(n_bins):
        mask = bin_indices == i
        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_coverage[i] = np.mean(covered[mask])
        else:
            bin_coverage[i] = np.nan

    return bin_edges, bin_coverage, bin_counts


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve for uncertainty estimates.

    For well-calibrated uncertainties, the observed coverage at each confidence
    level should match the expected coverage.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray
        Predicted standard deviations
    n_bins : int
        Number of confidence levels to test

    Returns
    -------
    expected_coverage : np.ndarray
        Expected coverage levels
    observed_coverage : np.ndarray
        Observed coverage levels
    """
    from scipy import stats

    # Test different confidence levels
    expected_coverage = np.linspace(0.1, 0.99, n_bins)
    observed_coverage = np.zeros(n_bins)

    for i, conf_level in enumerate(expected_coverage):
        # Compute z-score for this confidence level
        alpha = 1 - conf_level
        z = stats.norm.ppf(1 - alpha/2)

        # Compute intervals
        y_lower = y_pred - z * y_std
        y_upper = y_pred + z * y_std

        # Compute observed coverage
        observed_coverage[i] = prediction_interval_coverage_probability(
            y_true, y_lower, y_upper
        )

    return expected_coverage, observed_coverage


def miscalibration_area(
    expected_coverage: np.ndarray,
    observed_coverage: np.ndarray,
) -> float:
    """
    Compute area between expected and observed coverage (miscalibration).

    A well-calibrated model should have miscalibration area close to 0.

    Parameters
    ----------
    expected_coverage : np.ndarray
        Expected coverage levels
    observed_coverage : np.ndarray
        Observed coverage levels

    Returns
    -------
    area : float
        Miscalibration area
    """
    return np.trapz(np.abs(observed_coverage - expected_coverage), expected_coverage)


def coverage_width_criterion(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    eta: float = 0.05,
) -> float:
    """
    Compute Coverage Width-based Criterion (CWC).

    CWC penalizes both poor coverage and wide intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals
    eta : float
        Penalty parameter for coverage violations

    Returns
    -------
    cwc : float
        Coverage width criterion (lower is better)
    """
    n = len(y_true)
    coverage = prediction_interval_coverage_probability(y_true, y_lower, y_upper)
    mean_width = np.mean(y_upper - y_lower)

    # Penalty for insufficient coverage
    penalty = np.exp(-eta * (coverage - 0.95)) if coverage < 0.95 else 1.0

    return mean_width * penalty
