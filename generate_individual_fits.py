#!/usr/bin/env python
"""Generate individual fit plots for all 64 experiments."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from src.datasets.linear import LineDataset, PolynomialDataset, LennardJonesDataset, ShomateDataset
from src.models.linear_models import OLSRegression, LennardJonesRegression, ShomateRegression
from src.uq_methods.linear_uq import HatMatrixUQ, BayesianLinearRegressionUQ
from src.utils.seeds import get_experiment_seed
from config.global_config import RESULTS_FIGURES_DIR, NOISE_MODELS, NOISE_LEVELS

print("=" * 70)
print("GENERATING INDIVIDUAL FIT PLOTS")
print("=" * 70)

# Create subdirectory for individual fits
FITS_DIR = RESULTS_FIGURES_DIR / 'individual_fits'
FITS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations
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
        'params': {'degree': 3}
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

# UQ methods
UQ_METHODS = {
    'HatMatrix': HatMatrixUQ(confidence_level=0.95),
    'Bayesian': BayesianLinearRegressionUQ(confidence_level=0.95),
}

# Configuration
N_SAMPLES = 100

total = len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS) * len(UQ_METHODS)
print(f"\nGenerating {total} individual fit plots...")
print(f"Saving to: {FITS_DIR}")

pbar = tqdm(total=total, desc="Generating plots")

for dataset_name, dataset_config in DATASET_CONFIGS.items():
    for noise_model in NOISE_MODELS:
        for noise_level in NOISE_LEVELS:

            # Generate experiment ID
            exp_id = f"{dataset_name}_{noise_model}_noise{int(noise_level*100):02d}"
            exp_seed = get_experiment_seed(exp_id)

            # Create dataset
            dataset_class = dataset_config['class']
            dataset = dataset_class(
                n_samples=N_SAMPLES,
                noise_model=noise_model,
                noise_level=noise_level,
                seed=exp_seed,
                **dataset_config['params']
            )

            # Generate data
            data = dataset.generate()

            # Fit model
            model_class = dataset_config['model_class']
            model = model_class(**dataset_config['model_kwargs'])
            model.fit(data.X_train, data.y_train)

            # Apply each UQ method and create plot
            for uq_name, uq_method in UQ_METHODS.items():

                # Compute uncertainty intervals
                result = uq_method.compute_intervals(
                    model, data.X_train, data.y_train, data.X_test
                )

                # Compute coverage
                from src.metrics import picp
                coverage = picp(data.y_test, result)

                # Create figure
                fig, ax = plt.subplots(figsize=(12, 7))

                # Sort data for plotting
                X_flat = data.X_test.flatten()
                sort_idx = np.argsort(X_flat)
                X_sorted = X_flat[sort_idx]
                y_true_sorted = data.y_test[sort_idx]
                y_pred_sorted = result.y_pred[sort_idx]
                y_lower_sorted = result.y_lower[sort_idx]
                y_upper_sorted = result.y_upper[sort_idx]

                # Plot training data
                ax.scatter(data.X_train.flatten(), data.y_train, alpha=0.5, s=30,
                          label='Training data', color='gray', zorder=1)

                # Plot gap region (if exists)
                if data.X_gap is not None and len(data.X_gap) > 0:
                    ax.axvspan(data.X_gap.min(), data.X_gap.max(),
                              alpha=0.1, color='orange', zorder=0,
                              label='Gap (interpolation)')

                # Plot TRUE extrapolation regions (outside actual training data range)
                x_train_min = data.X_train.min()
                x_train_max = data.X_train.max()

                # Only shade regions truly beyond training data
                ax.axvspan(-0.25, x_train_min, alpha=0.08, color='red', zorder=0,
                          label='Extrapolation')
                ax.axvspan(x_train_max, 1.25, alpha=0.08, color='red', zorder=0)

                # Plot true function
                ax.plot(X_sorted, y_true_sorted, 'k-', linewidth=2.5,
                       label='True function', zorder=3)

                # Plot predictions
                ax.plot(X_sorted, y_pred_sorted, 'b--', linewidth=2,
                       label='Prediction', zorder=4)

                # Plot uncertainty intervals
                ax.fill_between(X_sorted, y_lower_sorted, y_upper_sorted,
                               alpha=0.3, color='blue',
                               label=f'95% Prediction Interval', zorder=2)

                # Add title and labels
                title = f"{dataset_name} | {noise_model.capitalize()} {int(noise_level*100)}% | {uq_name}"
                title += f"\nCoverage: {coverage*100:.1f}% (target: 95%)"
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('X', fontsize=12)
                ax.set_ylabel('y', fontsize=12)
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Expand x-axis to show extrapolation better
                ax.set_xlim(-0.25, 1.25)

                # Add annotation
                textstr = f'N_train: {len(data.X_train)}\nN_test: {len(data.X_test)}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)

                plt.tight_layout()

                # Save figure
                filename = f"{exp_id}_{uq_name}.png"
                filepath = FITS_DIR / filename
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close()

                pbar.update(1)

pbar.close()

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\nGenerated {total} individual fit plots")
print(f"Location: {FITS_DIR}")
print(f"\nExample files:")
for i, f in enumerate(list(FITS_DIR.glob('*.png'))[:3]):
    print(f"  - {f.name}")
if total > 3:
    print(f"  ... and {total - 3} more")
