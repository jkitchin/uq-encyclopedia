"""Learning curve plots for model performance vs training size."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from config import global_config


def plot_learning_curve(
    sample_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    metric_name: str = "Score",
    title: str = "Learning Curve",
    ax: Optional[plt.Axes] = None,
    show_std: bool = True,
    train_std: Optional[np.ndarray] = None,
    test_std: Optional[np.ndarray] = None,
) -> plt.Axes:
    """
    Plot learning curve showing performance vs training size.

    Parameters
    ----------
    sample_sizes : np.ndarray
        Training sample sizes
    train_scores : np.ndarray
        Training scores (mean across folds/bootstrap)
    test_scores : np.ndarray
        Test scores (mean across folds/bootstrap)
    metric_name : str
        Name of the metric being plotted
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes
    show_std : bool
        Whether to show standard deviation bands
    train_std : np.ndarray, optional
        Standard deviation of training scores
    test_std : np.ndarray, optional
        Standard deviation of test scores

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=global_config.FIGURE_SIZE)

    # Plot training scores
    ax.plot(sample_sizes, train_scores, 'o-', linewidth=2,
            label='Training score', color='blue')

    # Plot test scores
    ax.plot(sample_sizes, test_scores, 'o-', linewidth=2,
            label='Test score', color='orange')

    # Add standard deviation bands if requested
    if show_std and train_std is not None:
        ax.fill_between(sample_sizes,
                        train_scores - train_std,
                        train_scores + train_std,
                        alpha=0.2, color='blue')

    if show_std and test_std is not None:
        ax.fill_between(sample_sizes,
                        test_scores - test_std,
                        test_scores + test_std,
                        alpha=0.2, color='orange')

    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Use log scale for x-axis if appropriate
    if sample_sizes.max() / sample_sizes.min() > 10:
        ax.set_xscale('log')

    return ax


def plot_multiple_learning_curves(
    sample_sizes: np.ndarray,
    results: Dict[str, Dict[str, np.ndarray]],
    metric_name: str = "Score",
    title: str = "Learning Curves Comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot multiple learning curves for comparison.

    Parameters
    ----------
    sample_sizes : np.ndarray
        Training sample sizes
    results : dict
        Dictionary mapping method names to dictionaries with 'mean' and 'std' keys
    metric_name : str
        Name of the metric being plotted
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

    for method_name, scores in results.items():
        mean_scores = scores['mean']
        std_scores = scores.get('std', None)

        ax.plot(sample_sizes, mean_scores, 'o-', linewidth=2,
                label=method_name)

        if std_scores is not None:
            ax.fill_between(sample_sizes,
                            mean_scores - std_scores,
                            mean_scores + std_scores,
                            alpha=0.2)

    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Use log scale for x-axis if appropriate
    if sample_sizes.max() / sample_sizes.min() > 10:
        ax.set_xscale('log')

    return ax


def plot_coverage_vs_sample_size(
    sample_sizes: np.ndarray,
    coverages: Dict[str, np.ndarray],
    target_coverage: float = 0.95,
    title: str = "Coverage vs Sample Size",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot how coverage changes with training sample size.

    Parameters
    ----------
    sample_sizes : np.ndarray
        Training sample sizes
    coverages : dict
        Dictionary mapping method names to coverage arrays
    target_coverage : float
        Target coverage level
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

    for method_name, coverage in coverages.items():
        ax.plot(sample_sizes, coverage, 'o-', linewidth=2,
                label=method_name)

    # Add target coverage line
    ax.axhline(y=target_coverage, color='r', linestyle='--', linewidth=2,
               label=f'Target ({target_coverage*100:.0f}%)')

    # Add acceptable range
    ax.axhspan(target_coverage - 0.05, target_coverage + 0.05,
               alpha=0.1, color='green', label='±5% tolerance')

    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel('Coverage Probability')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    # Use log scale for x-axis if appropriate
    if sample_sizes.max() / sample_sizes.min() > 10:
        ax.set_xscale('log')

    return ax


def plot_width_vs_sample_size(
    sample_sizes: np.ndarray,
    widths: Dict[str, np.ndarray],
    title: str = "Interval Width vs Sample Size",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot how interval width changes with training sample size.

    Parameters
    ----------
    sample_sizes : np.ndarray
        Training sample sizes
    widths : dict
        Dictionary mapping method names to width arrays
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

    for method_name, width in widths.items():
        ax.plot(sample_sizes, width, 'o-', linewidth=2,
                label=method_name)

    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel('Mean Interval Width')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Use log scale for both axes if appropriate
    if sample_sizes.max() / sample_sizes.min() > 10:
        ax.set_xscale('log')
        ax.set_yscale('log')

    return ax


def plot_learning_curve_grid(
    sample_sizes: np.ndarray,
    results: Dict[str, Dict[str, np.ndarray]],
    metrics: List[str] = ['RMSE', 'Coverage', 'Width'],
    title: str = "Learning Curves",
    figsize: tuple = None,
) -> plt.Figure:
    """
    Plot a grid of learning curves for different metrics.

    Parameters
    ----------
    sample_sizes : np.ndarray
        Training sample sizes
    results : dict
        Nested dictionary: results[method][metric] = scores
    metrics : list
        List of metrics to plot
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

    for ax, metric in zip(axes, metrics):
        metric_results = {method: scores[metric]
                          for method, scores in results.items()}

        plot_multiple_learning_curves(
            sample_sizes,
            metric_results,
            metric_name=metric,
            title=f"{metric} vs Sample Size",
            ax=ax
        )

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    return fig
