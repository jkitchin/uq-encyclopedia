"""Core plotting functions for UQ Encyclopedia."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from src.uq_methods.base import UncertaintyResult
from config import global_config


def setup_plot_style():
    """Set up consistent plotting style."""
    try:
        plt.style.use(global_config.PLOT_STYLE)
    except:
        plt.style.use('seaborn-v0_8-darkgrid')

    plt.rcParams['figure.dpi'] = global_config.FIGURE_DPI
    plt.rcParams['figure.figsize'] = global_config.FIGURE_SIZE


def plot_predictions_with_intervals(
    X: np.ndarray,
    y_true: np.ndarray,
    result: UncertaintyResult,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    title: str = "Predictions with Uncertainty Intervals",
    xlabel: str = "X",
    ylabel: str = "y",
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
) -> plt.Axes:
    """
    Plot predictions with uncertainty intervals.

    Parameters
    ----------
    X : np.ndarray
        Test input values
    y_true : np.ndarray
        True test values
    result : UncertaintyResult
        Uncertainty quantification results
    X_train : np.ndarray, optional
        Training input values
    y_train : np.ndarray, optional
        Training target values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    show_legend : bool
        Whether to show legend

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=global_config.FIGURE_SIZE)

    # Flatten arrays for plotting
    X_flat = X.flatten()
    sort_idx = np.argsort(X_flat)
    X_sorted = X_flat[sort_idx]
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = result.y_pred[sort_idx]
    y_lower_sorted = result.y_lower[sort_idx]
    y_upper_sorted = result.y_upper[sort_idx]

    # Plot training data if provided
    if X_train is not None and y_train is not None:
        ax.scatter(X_train.flatten(), y_train, alpha=0.5, s=20,
                   label='Training data', color='gray', zorder=1)

    # Plot true function
    ax.plot(X_sorted, y_true_sorted, 'k-', linewidth=2,
            label='True function', zorder=3)

    # Plot predictions
    ax.plot(X_sorted, y_pred_sorted, 'b--', linewidth=2,
            label='Prediction', zorder=4)

    # Plot uncertainty intervals
    ax.fill_between(X_sorted, y_lower_sorted, y_upper_sorted,
                     alpha=0.3, color='blue',
                     label=f'{result.confidence_level*100:.0f}% PI', zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    title: str = "Residual Plot",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot residuals with optional uncertainty bands.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray, optional
        Prediction standard deviations
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=global_config.FIGURE_SIZE)

    residuals = y_true - y_pred

    ax.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)

    if y_std is not None:
        # Plot ±2σ bands
        sort_idx = np.argsort(y_pred)
        y_pred_sorted = y_pred[sort_idx]
        y_std_sorted = y_std[sort_idx]

        ax.fill_between(y_pred_sorted,
                        -2*y_std_sorted, 2*y_std_sorted,
                        alpha=0.2, color='blue',
                        label='±2σ')
        ax.legend()

    ax.set_xlabel('Predicted value')
    ax.set_ylabel('Residual')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_coverage_by_region(
    X: np.ndarray,
    y_true: np.ndarray,
    result: UncertaintyResult,
    regions: dict,
    title: str = "Coverage by Region",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot coverage probability across different regions.

    Parameters
    ----------
    X : np.ndarray
        Input values
    y_true : np.ndarray
        True values
    result : UncertaintyResult
        Uncertainty results
    regions : dict
        Dictionary mapping region names to boolean masks
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    from src.metrics.coverage import picp

    region_names = list(regions.keys())
    coverages = []

    for name, mask in regions.items():
        if mask.sum() == 0:
            coverages.append(0)
            continue

        # Compute coverage for this region
        coverage = picp(y_true[mask],
                       UncertaintyResult(
                           y_pred=result.y_pred[mask],
                           y_lower=result.y_lower[mask],
                           y_upper=result.y_upper[mask],
                           confidence_level=result.confidence_level
                       ))
        coverages.append(coverage)

    # Plot bars
    bars = ax.bar(region_names, coverages, alpha=0.7)

    # Add target coverage line
    ax.axhline(y=result.confidence_level, color='r', linestyle='--',
               linewidth=2, label=f'Target ({result.confidence_level*100:.0f}%)')

    # Color bars by how close they are to target
    for bar, cov in zip(bars, coverages):
        if abs(cov - result.confidence_level) < 0.05:
            bar.set_color('green')
        elif abs(cov - result.confidence_level) < 0.10:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    ax.set_ylabel('Coverage Probability')
    ax.set_title(title)
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def plot_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    title: str = "Q-Q Plot of Normalized Residuals",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Q-Q plot of normalized residuals.

    Checks if residuals are normally distributed.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray
        Prediction standard deviations
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
        fig, ax = plt.subplots(figsize=(6, 6))

    # Compute normalized residuals
    normalized_residuals = (y_true - y_pred) / (y_std + 1e-10)

    # Create Q-Q plot
    stats.probplot(normalized_residuals, dist="norm", plot=ax)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def plot_interval_widths(
    X: np.ndarray,
    result: UncertaintyResult,
    title: str = "Prediction Interval Widths",
    xlabel: str = "X",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot prediction interval widths across input space.

    Parameters
    ----------
    X : np.ndarray
        Input values
    result : UncertaintyResult
        Uncertainty results
    title : str
        Plot title
    xlabel : str
        X-axis label
    ax : plt.Axes, optional
        Matplotlib axes

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=global_config.FIGURE_SIZE)

    X_flat = X.flatten()
    widths = result.y_upper - result.y_lower

    # Sort for plotting
    sort_idx = np.argsort(X_flat)
    X_sorted = X_flat[sort_idx]
    widths_sorted = widths[sort_idx]

    ax.plot(X_sorted, widths_sorted, 'b-', linewidth=2)
    ax.axhline(y=np.mean(widths), color='r', linestyle='--',
               label=f'Mean: {np.mean(widths):.3f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Interval Width')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def save_figure(
    fig: plt.Figure,
    filename: str,
    formats: Optional[List[str]] = None,
    dpi: Optional[int] = None,
):
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    formats : list of str, optional
        File formats to save (default from config)
    dpi : int, optional
        DPI for raster formats (default from config)
    """
    if formats is None:
        formats = global_config.EXPORT_FORMATS

    if dpi is None:
        dpi = global_config.FIGURE_DPI

    from pathlib import Path
    from config.global_config import RESULTS_FIGURES_DIR

    for fmt in formats:
        filepath = RESULTS_FIGURES_DIR / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")
