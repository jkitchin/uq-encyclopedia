"""Comparison plots for multiple UQ methods."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from src.uq_methods.base import UncertaintyResult
from src.metrics.coverage import picp
from src.metrics.efficiency import mean_interval_width
from src.metrics.accuracy import root_mean_squared_error
from config import global_config


def plot_method_comparison_bars(
    y_true: np.ndarray,
    results: Dict[str, UncertaintyResult],
    metrics: List[str] = ['Coverage', 'RMSE', 'Width'],
    title: str = "UQ Method Comparison",
    figsize: tuple = None,
) -> plt.Figure:
    """
    Create bar plots comparing different UQ methods across metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    results : dict
        Dictionary mapping method names to UncertaintyResult objects
    metrics : list
        List of metrics to compare
    title : str
        Overall title
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    if figsize is None:
        figsize = (15, 5)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    method_names = list(results.keys())

    for ax, metric in zip(axes, metrics):
        values = []

        for method_name, result in results.items():
            if metric.lower() == 'coverage':
                value = picp(y_true, result)
            elif metric.lower() == 'rmse':
                value = root_mean_squared_error(y_true, result.y_pred)
            elif metric.lower() == 'width':
                value = mean_interval_width(result.y_lower, result.y_upper)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            values.append(value)

        bars = ax.bar(method_names, values, alpha=0.7)

        # Add target line for coverage
        if metric.lower() == 'coverage':
            target = results[method_names[0]].confidence_level
            ax.axhline(y=target, color='r', linestyle='--',
                      label=f'Target: {target*100:.0f}%')
            ax.legend()

        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x-labels if needed
        if len(method_names) > 3:
            ax.set_xticklabels(method_names, rotation=45, ha='right')

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    return fig


def plot_radar_comparison(
    y_true: np.ndarray,
    results: Dict[str, UncertaintyResult],
    title: str = "UQ Method Radar Comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Create radar plot comparing UQ methods across multiple metrics.

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
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Metrics to compare (normalized to [0, 1])
    metrics = ['Coverage', 'Sharpness', 'Accuracy']
    n_metrics = len(metrics)

    # Compute angles for each metric
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for method_name, result in results.items():
        # Compute normalized metrics
        coverage = picp(y_true, result)
        width = mean_interval_width(result.y_lower, result.y_upper)
        sharpness = 1.0 / (width + 1e-10)  # Inverse of width
        rmse = root_mean_squared_error(y_true, result.y_pred)
        accuracy = 1.0 / (1.0 + rmse)  # Normalize accuracy

        # Normalize all to [0, 1] relative to this set
        values = [coverage, sharpness, accuracy]

        # Close the polygon
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=method_name)
        ax.fill(angles, values, alpha=0.15)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, 1])
    ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    return ax


def plot_interval_comparison(
    X: np.ndarray,
    y_true: np.ndarray,
    results: Dict[str, UncertaintyResult],
    title: str = "Prediction Interval Comparison",
    xlabel: str = "X",
    ylabel: str = "y",
    n_methods_per_plot: int = 2,
) -> plt.Figure:
    """
    Plot prediction intervals from multiple methods side-by-side.

    Parameters
    ----------
    X : np.ndarray
        Input values
    y_true : np.ndarray
        True values
    results : dict
        Dictionary mapping method names to UncertaintyResult objects
    title : str
        Overall title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    n_methods_per_plot : int
        Number of methods to show per subplot

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    from src.visualization.plots import plot_predictions_with_intervals

    n_methods = len(results)
    n_plots = int(np.ceil(n_methods / n_methods_per_plot))

    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    method_items = list(results.items())

    for i, ax in enumerate(axes):
        start_idx = i * n_methods_per_plot
        end_idx = min(start_idx + n_methods_per_plot, n_methods)

        # Plot true function
        X_flat = X.flatten()
        sort_idx = np.argsort(X_flat)
        ax.plot(X_flat[sort_idx], y_true[sort_idx], 'k-',
                linewidth=2, label='True', zorder=10)

        # Plot intervals for each method
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for j, (method_name, result) in enumerate(method_items[start_idx:end_idx]):
            color = colors[j % len(colors)]

            y_pred_sorted = result.y_pred[sort_idx]
            y_lower_sorted = result.y_lower[sort_idx]
            y_upper_sorted = result.y_upper[sort_idx]

            ax.plot(X_flat[sort_idx], y_pred_sorted, '--',
                    color=color, linewidth=1.5, label=f'{method_name} (pred)')
            ax.fill_between(X_flat[sort_idx], y_lower_sorted, y_upper_sorted,
                           alpha=0.2, color=color,
                           label=f'{method_name} (PI)')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    return fig


def plot_metrics_heatmap(
    y_true: np.ndarray,
    results: Dict[str, UncertaintyResult],
    title: str = "Metrics Heatmap",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Create heatmap of metrics for different UQ methods.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    results : dict
        Dictionary mapping method names to UncertaintyResult objects
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    import pandas as pd
    import seaborn as sns

    from src.metrics.efficiency import median_interval_width, interval_score
    from src.metrics.accuracy import mean_absolute_error, r2_score

    # Compute metrics for each method
    metrics_data = []

    for method_name, result in results.items():
        metrics = {
            'Method': method_name,
            'Coverage': picp(y_true, result),
            'RMSE': root_mean_squared_error(y_true, result.y_pred),
            'MAE': mean_absolute_error(y_true, result.y_pred),
            'R²': r2_score(y_true, result.y_pred),
            'Mean Width': mean_interval_width(result.y_lower, result.y_upper),
            'Median Width': median_interval_width(result.y_lower, result.y_upper),
            'Interval Score': interval_score(
                y_true, result.y_lower, result.y_upper,
                alpha=1-result.confidence_level
            ),
        }
        metrics_data.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    df = df.set_index('Method')

    # Normalize each column to [0, 1] for visualization
    df_norm = (df - df.min()) / (df.max() - df.min())

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_norm.T, annot=df.T, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'Normalized Value'}, ax=ax)

    ax.set_title(title)
    ax.set_xlabel('Method')
    ax.set_ylabel('Metric')

    plt.tight_layout()

    return fig
