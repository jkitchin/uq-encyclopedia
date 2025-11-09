#!/usr/bin/env python
"""
Comprehensive benchmark for linear models.

Runs all combinations of:
- 4 datasets: Line, Polynomial, Lennard-Jones, Shomate
- 2 noise models: homoskedastic, heteroskedastic
- 4 noise levels: 1%, 2%, 5%, 10%
- 3 UQ methods: Hat Matrix, Bayesian Linear Regression, Conformal Prediction

Total: 96 experiments
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from src.datasets.linear import LineDataset, PolynomialDataset, LennardJonesDataset, ShomateDataset
from src.models.linear_models import OLSRegression, LennardJonesRegression, ShomateRegression
from src.uq_methods.linear_uq import HatMatrixUQ, BayesianLinearRegressionUQ, ConformalPredictionUQ
from src.metrics import (
    picp, mean_interval_width, root_mean_squared_error,
    mean_absolute_error, r2_score, interval_score,
    normalized_interval_width, calibration_curve, miscalibration_area
)
from src.utils.seeds import set_global_seed, get_experiment_seed
from src.utils.io import save_results_csv, save_results_json
from config.global_config import (
    NOISE_MODELS, NOISE_LEVELS,
    RESULTS_CSV_DIR, RESULTS_JSON_DIR, RESULTS_FIGURES_DIR
)

print("=" * 70)
print("UQ ENCYCLOPEDIA - COMPREHENSIVE LINEAR MODELS BENCHMARK")
print("=" * 70)

# Configuration
CONFIDENCE_LEVEL = 0.95
N_SAMPLES = 100
SEED = 42

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
    'HatMatrix': HatMatrixUQ(confidence_level=CONFIDENCE_LEVEL),
    'Bayesian': BayesianLinearRegressionUQ(confidence_level=CONFIDENCE_LEVEL),
    'Conformal': ConformalPredictionUQ(confidence_level=CONFIDENCE_LEVEL, method='plus', cv=5),
}

# Results storage
all_results = []
detailed_results = {}

# Set global seed
set_global_seed(SEED)

print(f"\nConfiguration:")
print(f"  - Datasets: {len(DATASET_CONFIGS)}")
print(f"  - Noise models: {len(NOISE_MODELS)}")
print(f"  - Noise levels: {len(NOISE_LEVELS)}")
print(f"  - UQ methods: {len(UQ_METHODS)}")
print(f"  - Total experiments: {len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS) * len(UQ_METHODS)}")
print(f"  - Confidence level: {CONFIDENCE_LEVEL}")
print(f"  - Samples per dataset: {N_SAMPLES}")

print("\n" + "=" * 70)
print("RUNNING EXPERIMENTS")
print("=" * 70)

# Track progress
total_experiments = len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS) * len(UQ_METHODS)
pbar = tqdm(total=total_experiments, desc="Overall Progress")

# Run all experiments
for dataset_name, dataset_config in DATASET_CONFIGS.items():
    for noise_model in NOISE_MODELS:
        for noise_level in NOISE_LEVELS:

            # Generate experiment ID
            exp_id = f"{dataset_name}_{noise_model}_noise{int(noise_level*100):02d}"

            # Get deterministic seed for this experiment
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

            # Apply each UQ method
            for uq_name, uq_method in UQ_METHODS.items():

                # Compute uncertainty intervals
                result = uq_method.compute_intervals(
                    model, data.X_train, data.y_train, data.X_test
                )

                # Compute metrics
                coverage = picp(data.y_test, result)
                rmse = root_mean_squared_error(data.y_test, result.y_pred)
                mae = mean_absolute_error(data.y_test, result.y_pred)
                r2 = r2_score(data.y_test, result.y_pred)
                mean_width = mean_interval_width(result.y_lower, result.y_upper)
                norm_width = normalized_interval_width(result.y_lower, result.y_upper, data.y_test)
                int_score = interval_score(
                    data.y_test, result.y_lower, result.y_upper,
                    alpha=1-CONFIDENCE_LEVEL
                )

                # Compute calibration
                expected_cov, observed_cov = calibration_curve(
                    data.y_test, result.y_pred, result.std, n_bins=10
                )
                miscal_area = miscalibration_area(expected_cov, observed_cov)

                # Compute coverage by region
                X_flat = data.X_test.flatten()
                regions = {
                    'extrap_low': X_flat < 0.2,
                    'interpolation': (X_flat >= 0.2) & (X_flat <= 0.8),
                    'extrap_high': X_flat > 0.8,
                }

                coverage_by_region = {}
                for region_name, mask in regions.items():
                    if mask.sum() > 0:
                        from src.uq_methods.base import UncertaintyResult
                        region_result = UncertaintyResult(
                            y_pred=result.y_pred[mask],
                            y_lower=result.y_lower[mask],
                            y_upper=result.y_upper[mask],
                            confidence_level=result.confidence_level
                        )
                        coverage_by_region[f'coverage_{region_name}'] = picp(
                            data.y_test[mask], region_result
                        )
                    else:
                        coverage_by_region[f'coverage_{region_name}'] = np.nan

                # Store results
                result_dict = {
                    'dataset': dataset_name,
                    'noise_model': noise_model,
                    'noise_level': noise_level,
                    'uq_method': uq_name,
                    'n_train': len(data.X_train),
                    'n_test': len(data.X_test),
                    'coverage': coverage,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mean_width': mean_width,
                    'normalized_width': norm_width,
                    'interval_score': int_score,
                    'miscalibration_area': miscal_area,
                    **coverage_by_region,
                    'seed': exp_seed,
                }

                all_results.append(result_dict)

                # Store detailed results for later visualization
                detailed_key = f"{exp_id}_{uq_name}"
                detailed_results[detailed_key] = {
                    'data': data,
                    'result': result,
                    'metrics': result_dict,
                }

                pbar.update(1)

pbar.close()

# Create DataFrame
df_results = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save comprehensive results
results_csv_path = RESULTS_CSV_DIR / 'linear_models_comprehensive.csv'
df_results.to_csv(results_csv_path, index=False)
print(f"✓ Saved: {results_csv_path}")

# Save summary statistics
summary_stats = df_results.groupby(['dataset', 'noise_model', 'noise_level', 'uq_method']).agg({
    'coverage': ['mean', 'std'],
    'rmse': ['mean', 'std'],
    'mean_width': ['mean', 'std'],
}).round(4)

summary_path = RESULTS_CSV_DIR / 'linear_models_summary.csv'
summary_stats.to_csv(summary_path)
print(f"✓ Saved: {summary_path}")

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# Print overall summary
print("\nOverall Performance by UQ Method:")
method_summary = df_results.groupby('uq_method').agg({
    'coverage': ['mean', 'std'],
    'rmse': ['mean', 'std'],
    'mean_width': ['mean', 'std'],
}).round(4)
print(method_summary)

print("\nCoverage by Dataset:")
dataset_coverage = df_results.groupby('dataset')['coverage'].agg(['mean', 'std', 'min', 'max']).round(3)
print(dataset_coverage)

print("\nCoverage by Noise Model:")
noise_coverage = df_results.groupby('noise_model')['coverage'].agg(['mean', 'std']).round(3)
print(noise_coverage)

print("\nCoverage by Noise Level:")
level_coverage = df_results.groupby('noise_level')['coverage'].agg(['mean', 'std']).round(3)
print(level_coverage)

# Check which experiments met target coverage (95% ± 5%)
target_coverage = CONFIDENCE_LEVEL
tolerance = 0.05
df_results['meets_target'] = (
    (df_results['coverage'] >= target_coverage - tolerance) &
    (df_results['coverage'] <= target_coverage + tolerance)
)

print(f"\nExperiments meeting target coverage ({target_coverage:.0%} ± {tolerance:.0%}):")
print(f"  {df_results['meets_target'].sum()} / {len(df_results)} ({df_results['meets_target'].mean()*100:.1f}%)")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE!")
print("=" * 70)
print(f"\nResults saved to:")
print(f"  - {results_csv_path}")
print(f"  - {summary_path}")
print(f"\nRun generate_linear_visualizations.py to create figures.")
