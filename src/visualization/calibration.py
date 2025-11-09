"""Calibration plots for uncertainty quantification."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from src.metrics.coverage import calibration_curve, miscalibration_area
from config import global_config


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    ax: Optional[plt.Axes] = None,
    show_miscalibration: bool = True,
) -> plt.Axes:
    """
    Plot calibration curve for uncertainty estimates.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray
        Predicted standard deviations
    n_bins : int
        Number of confidence levels to test
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes
    show_miscalibration : bool
        Whether to show miscalibration area in title

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Compute calibration curve
    expected_cov, observed_cov = calibration_curve(y_true, y_pred, y_std, n_bins)

    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Plot observed calibration
    ax.plot(expected_cov, observed_cov, 'bo-', linewidth=2, markersize=8,
            label='Observed calibration')

    # Shade the miscalibration area
    ax.fill_between(expected_cov, expected_cov, observed_cov,
                     alpha=0.2, color='red')

    # Compute and show miscalibration area
    if show_miscalibration:
        misc_area = miscalibration_area(expected_cov, observed_cov)
        title += f"\nMiscalibration Area: {misc_area:.4f}"

    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    return ax


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot reliability diagram (same as calibration curve with different style).

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray
        Predicted standard deviations
    n_bins : int
        Number of bins
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes
    """
    return plot_calibration_curve(
        y_true, y_pred, y_std, n_bins, title, ax, show_miscalibration=True
    )


def plot_sharpness_vs_calibration(
    y_true: np.ndarray,
    results: dict,
    title: str = "Sharpness vs Calibration",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot sharpness vs calibration for multiple UQ methods.

    Good methods are sharp (narrow intervals) and well-calibrated.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    results : dict
        Dictionary mapping method names to UncertaintyResult objects
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes
    """
    from src.metrics.coverage import picp
    from src.metrics.efficiency import mean_interval_width

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    coverages = []
    widths = []
    labels = []

    for name, result in results.items():
        coverage = picp(y_true, result)
        width = mean_interval_width(result.y_lower, result.y_upper)

        coverages.append(coverage)
        widths.append(width)
        labels.append(name)

    # Plot points
    ax.scatter(coverages, widths, s=100, alpha=0.6)

    # Annotate points
    for i, label in enumerate(labels):
        ax.annotate(label, (coverages[i], widths[i]),
                    xytext=(5, 5), textcoords='offset points')

    # Add target coverage line
    target_coverage = results[labels[0]].confidence_level
    ax.axvline(x=target_coverage, color='r', linestyle='--',
               label=f'Target coverage: {target_coverage*100:.0f}%')

    ax.set_xlabel('Coverage Probability')
    ax.set_ylabel('Mean Interval Width')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_pit_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 20,
    title: str = "PIT Histogram",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot Probability Integral Transform (PIT) histogram.

    For well-calibrated predictions, PIT values should be uniform.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray
        Predicted standard deviations
    n_bins : int
        Number of histogram bins
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes
    """
    from scipy import stats

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Compute PIT values
    # PIT = CDF(y_true | y_pred, y_std)
    pit_values = stats.norm.cdf(y_true, loc=y_pred, scale=y_std + 1e-10)

    # Plot histogram
    ax.hist(pit_values, bins=n_bins, density=True, alpha=0.7,
            edgecolor='black', label='PIT histogram')

    # Add uniform reference line
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2,
               label='Uniform (ideal)')

    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([0, 1])

    return ax
