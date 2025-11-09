"""Efficiency metrics for uncertainty quantification.

These metrics measure how narrow/efficient the prediction intervals are,
given that they achieve the desired coverage.
"""

import numpy as np
from typing import Optional
from src.uq_methods.base import UncertaintyResult


def mean_interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Compute mean prediction interval width.

    Parameters
    ----------
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals

    Returns
    -------
    width : float
        Mean interval width
    """
    return np.mean(y_upper - y_lower)


def median_interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Compute median prediction interval width.

    Parameters
    ----------
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals

    Returns
    -------
    width : float
        Median interval width
    """
    return np.median(y_upper - y_lower)


def interval_width_percentile(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    percentile: float = 90,
) -> float:
    """
    Compute a percentile of prediction interval widths.

    Parameters
    ----------
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals
    percentile : float
        Percentile to compute (0-100)

    Returns
    -------
    width : float
        Percentile of interval widths
    """
    widths = y_upper - y_lower
    return np.percentile(widths, percentile)


def normalized_interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Compute mean interval width normalized by the range of true values.

    This makes the metric scale-independent.

    Parameters
    ----------
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals
    y_true : np.ndarray
        True target values (for normalization)

    Returns
    -------
    width : float
        Normalized mean interval width
    """
    y_range = np.ptp(y_true)  # Peak-to-peak (max - min)
    if y_range == 0:
        return 0.0
    return mean_interval_width(y_lower, y_upper) / y_range


def interval_sharpness(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Compute interval sharpness (inverse of mean width).

    Higher values indicate sharper (narrower) intervals.

    Parameters
    ----------
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals

    Returns
    -------
    sharpness : float
        Interval sharpness
    """
    width = mean_interval_width(y_lower, y_upper)
    return 1.0 / width if width > 0 else np.inf


def width_variance(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Compute variance of interval widths.

    Measures how variable the interval widths are across predictions.

    Parameters
    ----------
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals

    Returns
    -------
    variance : float
        Variance of interval widths
    """
    widths = y_upper - y_lower
    return np.var(widths)


def coefficient_of_variation_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Compute coefficient of variation of interval widths.

    CV = std / mean, measures relative variability.

    Parameters
    ----------
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals

    Returns
    -------
    cv : float
        Coefficient of variation
    """
    widths = y_upper - y_lower
    mean_width = np.mean(widths)
    if mean_width == 0:
        return 0.0
    return np.std(widths) / mean_width


def interval_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Compute interval score (Gneiting & Raftery, 2007).

    The interval score is a proper scoring rule that rewards both
    narrow intervals and good coverage.

    IS = (upper - lower) + (2/α)(lower - y) if y < lower
                          + (2/α)(y - upper) if y > upper

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals
    alpha : float
        Miscoverage rate (1 - confidence_level)

    Returns
    -------
    score : float
        Mean interval score (lower is better)
    """
    width = y_upper - y_lower

    # Penalty for observations outside the interval
    penalty_lower = (2.0 / alpha) * np.maximum(0, y_lower - y_true)
    penalty_upper = (2.0 / alpha) * np.maximum(0, y_true - y_upper)

    scores = width + penalty_lower + penalty_upper
    return np.mean(scores)


def winkler_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Compute Winkler score (another name for interval score).

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals
    alpha : float
        Miscoverage rate (1 - confidence_level)

    Returns
    -------
    score : float
        Mean Winkler score (lower is better)
    """
    return interval_score(y_true, y_lower, y_upper, alpha)


def calibration_sharpness_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    target_coverage: float = 0.95,
) -> dict:
    """
    Compute a combined calibration-sharpness score.

    This metric penalizes both:
    1. Deviation from target coverage (miscalibration)
    2. Unnecessarily wide intervals (lack of sharpness)

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals
    target_coverage : float
        Target coverage level (e.g., 0.95 for 95%)

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'calibration_error': |observed_coverage - target_coverage|
        - 'sharpness': 1 / normalized_width (higher is better)
        - 'score': Combined score (lower is better)
        - 'observed_coverage': Actual coverage achieved
        - 'mean_width': Mean interval width
    """
    from src.metrics.coverage import prediction_interval_coverage_probability

    # Compute observed coverage
    observed_coverage = prediction_interval_coverage_probability(y_true, y_lower, y_upper)

    # Calibration error: absolute deviation from target
    calibration_error = abs(observed_coverage - target_coverage)

    # Sharpness: normalized by data range (higher is better)
    width = mean_interval_width(y_lower, y_upper)
    y_range = np.ptp(y_true)
    normalized_width = width / y_range if y_range > 0 else width
    sharpness = 1.0 / normalized_width if normalized_width > 0 else np.inf

    # Combined score: calibration error + penalty for wide intervals
    # Use normalized width so score is scale-independent
    # Weight calibration error more heavily (10x) since it's the primary goal
    score = 10 * calibration_error + normalized_width

    return {
        'calibration_error': calibration_error,
        'sharpness': sharpness,
        'score': score,
        'observed_coverage': observed_coverage,
        'mean_width': width,
        'normalized_width': normalized_width
    }
