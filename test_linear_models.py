#!/usr/bin/env python
"""Quick test script to generate results from linear models."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')

from src.datasets.linear import LineDataset, PolynomialDataset
from src.models.linear_models import OLSRegression
from src.uq_methods.linear_uq import HatMatrixUQ, BayesianLinearRegressionUQ, ConformalPredictionUQ
from src.visualization.plots import plot_predictions_with_intervals, save_figure
from src.visualization.comparison_plots import plot_method_comparison_bars
from src.metrics import picp, mean_interval_width, root_mean_squared_error
from src.utils.seeds import set_global_seed
from config.global_config import RESULTS_FIGURES_DIR
import pandas as pd

print("=" * 60)
print("UQ Encyclopedia - Linear Models Test")
print("=" * 60)

# Set random seed
set_global_seed(42)
print("\n1. Setting random seed: 42")

# Create dataset
print("\n2. Generating Line Dataset...")
dataset = LineDataset(
    slope=0.8,
    intercept=0.1,
    n_samples=100,
    noise_model='homoskedastic',
    noise_level=0.05,
    seed=42
)

data = dataset.generate()
print(f"   - Training samples: {len(data.X_train)}")
print(f"   - Test samples: {len(data.X_test)}")
print(f"   - Gap samples: {len(data.X_gap)}")

# Fit model
print("\n3. Fitting OLS Regression Model...")
model = OLSRegression(degree=1, fit_intercept=True)
model.fit(data.X_train, data.y_train)
print(f"   - Fitted coefficients: {model.coef_}")
print(f"   - True: [intercept={dataset.intercept:.2f}, slope={dataset.slope:.2f}]")

# Apply UQ methods
print("\n4. Applying UQ Methods...")
hat_uq = HatMatrixUQ(confidence_level=0.95)
bay_uq = BayesianLinearRegressionUQ(confidence_level=0.95)

result_hat = hat_uq.compute_intervals(model, data.X_train, data.y_train, data.X_test)
print("   ✓ Hat Matrix UQ")

result_bay = bay_uq.compute_intervals(model, data.X_train, data.y_train, data.X_test)
print("   ✓ Bayesian Linear Regression UQ")

# Note: Conformal prediction temporarily disabled due to MAPIE API changes
print("   (Conformal prediction skipped - API update needed)")

results = {
    'Hat Matrix': result_hat,
    'Bayesian': result_bay,
}

# Compute metrics
print("\n5. Computing Metrics...")
metrics_data = []

for method_name, result in results.items():
    metrics = {
        'Method': method_name,
        'Coverage': picp(data.y_test, result),
        'RMSE': root_mean_squared_error(data.y_test, result.y_pred),
        'Mean Width': mean_interval_width(result.y_lower, result.y_upper),
    }
    metrics_data.append(metrics)

df_metrics = pd.DataFrame(metrics_data)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print("\nMetrics Comparison:")
print(df_metrics.to_string(index=False))
print(f"\nTarget coverage: 0.95")
print("=" * 60)

# Generate plots
print("\n6. Generating Plots...")

# Individual method plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (method_name, result) in zip(axes, results.items()):
    plot_predictions_with_intervals(
        data.X_test,
        data.y_test,
        result,
        X_train=data.X_train,
        y_train=data.y_train,
        title=f'{method_name} UQ',
        ax=ax
    )

plt.tight_layout()
save_figure(fig, 'linear_models_comparison', formats=['png'])
print(f"   ✓ Saved: {RESULTS_FIGURES_DIR}/linear_models_comparison.png")

# Comparison bar chart
fig2 = plot_method_comparison_bars(
    data.y_test,
    results,
    metrics=['Coverage', 'RMSE', 'Width'],
    title='UQ Method Comparison'
)
save_figure(fig2, 'linear_models_metrics_bars', formats=['png'])
print(f"   ✓ Saved: {RESULTS_FIGURES_DIR}/linear_models_metrics_bars.png")

# Save metrics to CSV
csv_path = RESULTS_FIGURES_DIR.parent / 'csv' / 'linear_models_metrics.csv'
df_metrics.to_csv(csv_path, index=False)
print(f"   ✓ Saved: {csv_path}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print(f"\nResults saved to:")
print(f"  - Figures: {RESULTS_FIGURES_DIR}")
print(f"  - CSV: {csv_path}")
print("\nYou can view the figures and metrics to see the results!")
