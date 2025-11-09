"""Visualization module for UQ Encyclopedia.

This module provides comprehensive plotting functions for uncertainty quantification.
"""

from src.visualization.plots import (
    setup_plot_style,
    plot_predictions_with_intervals,
    plot_residuals,
    plot_coverage_by_region,
    plot_qq,
    plot_interval_widths,
    save_figure,
)

from src.visualization.calibration import (
    plot_calibration_curve,
    plot_reliability_diagram,
    plot_sharpness_vs_calibration,
    plot_pit_histogram,
)

from src.visualization.learning_curves import (
    plot_learning_curve,
    plot_multiple_learning_curves,
    plot_coverage_vs_sample_size,
    plot_width_vs_sample_size,
    plot_learning_curve_grid,
)

from src.visualization.comparison_plots import (
    plot_method_comparison_bars,
    plot_radar_comparison,
    plot_interval_comparison,
    plot_metrics_heatmap,
)

__all__ = [
    # Core plots
    'setup_plot_style',
    'plot_predictions_with_intervals',
    'plot_residuals',
    'plot_coverage_by_region',
    'plot_qq',
    'plot_interval_widths',
    'save_figure',
    # Calibration plots
    'plot_calibration_curve',
    'plot_reliability_diagram',
    'plot_sharpness_vs_calibration',
    'plot_pit_histogram',
    # Learning curves
    'plot_learning_curve',
    'plot_multiple_learning_curves',
    'plot_coverage_vs_sample_size',
    'plot_width_vs_sample_size',
    'plot_learning_curve_grid',
    # Comparison plots
    'plot_method_comparison_bars',
    'plot_radar_comparison',
    'plot_interval_comparison',
    'plot_metrics_heatmap',
]
