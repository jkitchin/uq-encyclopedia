"""Metrics module for UQ Encyclopedia.

This module provides comprehensive evaluation metrics for uncertainty quantification.
"""

from src.metrics.coverage import (
    prediction_interval_coverage_probability,
    picp,
    coverage_by_bin,
    calibration_curve,
    miscalibration_area,
    coverage_width_criterion,
)

from src.metrics.efficiency import (
    mean_interval_width,
    median_interval_width,
    interval_width_percentile,
    normalized_interval_width,
    interval_sharpness,
    width_variance,
    coefficient_of_variation_width,
    interval_score,
    winkler_score,
    calibration_sharpness_score,
)

from src.metrics.accuracy import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    adjusted_r2_score,
    mean_absolute_percentage_error,
    max_error,
    explained_variance_score,
    bias,
    normalized_rmse,
)

__all__ = [
    # Coverage metrics
    'prediction_interval_coverage_probability',
    'picp',
    'coverage_by_bin',
    'calibration_curve',
    'miscalibration_area',
    'coverage_width_criterion',
    # Efficiency metrics
    'mean_interval_width',
    'median_interval_width',
    'interval_width_percentile',
    'normalized_interval_width',
    'interval_sharpness',
    'width_variance',
    'coefficient_of_variation_width',
    'interval_score',
    'winkler_score',
    'calibration_sharpness_score',
    # Accuracy metrics
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_error',
    'median_absolute_error',
    'r2_score',
    'adjusted_r2_score',
    'mean_absolute_percentage_error',
    'max_error',
    'explained_variance_score',
    'bias',
    'normalized_rmse',
]
