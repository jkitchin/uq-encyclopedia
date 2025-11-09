#!/usr/bin/env python
"""Generate PNG images for all linear model plots to replace inline Plotly scripts."""

import sys
sys.path.insert(0, '.')
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.datasets.linear import LineDataset, PolynomialDataset, LennardJonesDataset, ShomateDataset
from src.models.linear_models import OLSRegression, LennardJonesRegression, ShomateRegression
from src.uq_methods.linear_uq import HatMatrixUQ, BayesianLinearRegressionUQ, ConformalPredictionUQ
from src.utils.seeds import get_experiment_seed

print("=" * 70)
print("GENERATING LINEAR MODEL PNG PLOTS")
print("=" * 70)

# Output directory
output_dir = Path('reports/dashboard/linear_plots_png')
output_dir.mkdir(parents=True, exist_ok=True)

# Dataset configurations (matching run_linear_benchmark.py)
DATASET_CONFIGS = {
    'Line': {
        'class': LineDataset,
        'model_class': OLSRegression,
        'model_kwargs': {'degree': 1, 'fit_intercept': True},
        'params': {'slope': 0.8, 'intercept': 0.1}
    },
    'Polynomial': {
        'class': PolynomialDataset,
        'model_class': OLSRegression,
        'model_kwargs': {'degree': 3, 'fit_intercept': True},
        'params': {'coefficients': [0.1, 0.5, 0.3, 0.2]}
    },
    'LennardJones': {
        'class': LennardJonesDataset,
        'model_class': LennardJonesRegression,
        'model_kwargs': {'r_min': 0.9, 'r_max': 3.0},
        'params': {'epsilon': 1.0, 'sigma': 1.0}
    },
    'Shomate': {
        'class': ShomateDataset,
        'model_class': ShomateRegression,
        'model_kwargs': {'T_min': 298.0, 'T_max': 1000.0},
        'params': {'T_min': 298.0, 'T_max': 1000.0}
    },
}

# UQ Methods
UQ_METHODS = {
    'HatMatrix': HatMatrixUQ(confidence_level=0.95),
    'Bayesian': BayesianLinearRegressionUQ(confidence_level=0.95),
    'Conformal': ConformalPredictionUQ(confidence_level=0.95, method='plus', cv=5),
}

# Noise models and levels
NOISE_MODELS = ['homoskedastic', 'heteroskedastic']
NOISE_LEVELS = [0.01, 0.02, 0.05, 0.10]

count = 0
total = len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS) * len(UQ_METHODS)

for dataset_name, dataset_config in DATASET_CONFIGS.items():
    for noise_model in NOISE_MODELS:
        for noise_level in NOISE_LEVELS:
            # Generate experiment ID and seed
            exp_id = f"{dataset_name}_{noise_model}_noise{int(noise_level*100):02d}"
            seed = get_experiment_seed(exp_id)

            # Create dataset
            dataset = dataset_config['class'](
                n_samples=100,
                noise_model=noise_model,
                noise_level=noise_level,
                seed=seed,
                **dataset_config['params']
            )
            data = dataset.generate()

            # Fit model
            model = dataset_config['model_class'](**dataset_config['model_kwargs'])
            model.fit(data.X_train, data.y_train)

            # Prepare dense grid for plotting true function
            X_plot = np.linspace(0, 1, 200)
            y_true = dataset._generate_clean(X_plot)

            for method_name, uq_method in UQ_METHODS.items():
                count += 1
                print(f"[{count}/{total}] {dataset_name} - {noise_model} {int(noise_level*100)}% - {method_name}")

                # Compute uncertainty intervals
                result = uq_method.compute_intervals(
                    model, data.X_train, data.y_train, data.X_test
                )

                # Create plot
                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

                # Plot true function
                ax.plot(X_plot, y_true, 'k-', linewidth=2, alpha=0.3, label='True function')

                # Plot training data
                ax.scatter(data.X_train, data.y_train, c='steelblue', s=50,
                          alpha=0.6, edgecolors='none', label='Training data')

                # Plot prediction intervals
                sort_idx = np.argsort(data.X_test.flatten())
                X_sorted = data.X_test.flatten()[sort_idx]
                y_lower_sorted = result.y_lower[sort_idx]
                y_upper_sorted = result.y_upper[sort_idx]
                y_pred_sorted = result.y_pred[sort_idx]

                ax.fill_between(X_sorted, y_lower_sorted, y_upper_sorted,
                               alpha=0.3, color='lightblue', label='95% PI')
                ax.plot(X_sorted, y_pred_sorted, 'b-', linewidth=2, label='Prediction')

                # Labels and formatting
                noise_pct = int(noise_level * 100)
                title = f"{dataset_name}\n{noise_model.capitalize()} {noise_pct}% noise"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('X', fontsize=9)
                ax.set_ylabel('y', fontsize=9)
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])

                plt.tight_layout()

                # Save PNG
                filename = f"{dataset_name}_{noise_model}_{noise_pct}_{method_name}.png"
                filepath = output_dir / filename
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close()

print(f"\n✓ Generated {count} PNG plots")
print(f"✓ Saved to {output_dir}")
