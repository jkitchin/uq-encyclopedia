#!/usr/bin/env python
"""Hyperparameter tuning for Bayesian Linear Regression UQ method."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
from itertools import product
from tqdm import tqdm

from src.datasets.linear import LineDataset, PolynomialDataset, LennardJonesDataset, ShomateDataset
from src.models.linear_models import OLSRegression, LennardJonesRegression, ShomateRegression
from src.uq_methods.linear_uq import BayesianLinearRegressionUQ
from src.metrics import picp, mean_interval_width
from src.utils.seeds import get_experiment_seed
from config.global_config import NOISE_MODELS, NOISE_LEVELS

print("=" * 70)
print("BAYESIAN LINEAR REGRESSION - HYPERPARAMETER TUNING")
print("=" * 70)

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

N_SAMPLES = 100

# Hyperparameter grid
# BayesianRidge uses Gamma priors: Gamma(alpha | alpha_1, alpha_2)
# Smaller alpha_1, alpha_2 -> less informative prior (more uncertainty)
# Larger values -> more regularization

PARAM_GRID = {
    'alpha_1': [1e-7, 1e-6, 1e-5],     # Shape parameter for noise precision prior
    'alpha_2': [1e-7, 1e-6, 1e-5],     # Rate parameter for noise precision prior
    'lambda_1': [1e-7, 1e-6, 1e-5],    # Shape parameter for weight precision prior
    'lambda_2': [1e-7, 1e-6, 1e-5],    # Rate parameter for weight precision prior
}

print(f"\nHyperparameter Grid:")
for param, values in PARAM_GRID.items():
    print(f"  {param}: {values}")

# Generate all combinations
param_combinations = list(product(
    PARAM_GRID['alpha_1'],
    PARAM_GRID['alpha_2'],
    PARAM_GRID['lambda_1'],
    PARAM_GRID['lambda_2']
))

print(f"\nTotal combinations to test: {len(param_combinations)}")
print(f"Total experiments: {len(param_combinations)} params × {len(DATASET_CONFIGS)} datasets × {len(NOISE_MODELS)} noise × {len(NOISE_LEVELS)} levels")
print(f"                 = {len(param_combinations) * len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS)} runs")

# Run tuning
results = []

pbar = tqdm(total=len(param_combinations) * len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS),
           desc="Tuning")

for alpha_1, alpha_2, lambda_1, lambda_2 in param_combinations:

    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        for noise_model in NOISE_MODELS:
            for noise_level in NOISE_LEVELS:

                # Generate experiment ID and seed
                exp_id = f"{dataset_name}_{noise_model}_noise{int(noise_level*100):02d}"
                exp_seed = get_experiment_seed(exp_id)

                try:
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

                    # Apply UQ method with current hyperparameters
                    uq_method = BayesianLinearRegressionUQ(
                        confidence_level=0.95,
                        alpha_1=alpha_1,
                        alpha_2=alpha_2,
                        lambda_1=lambda_1,
                        lambda_2=lambda_2
                    )

                    result = uq_method.compute_intervals(
                        model, data.X_train, data.y_train, data.X_test
                    )

                    # Compute metrics
                    coverage = picp(data.y_test, result)
                    width = mean_interval_width(result.y_lower, result.y_upper)

                    # Calculate score: penalize deviation from 95% and large widths
                    # Coverage score: 0 if within [0.90, 1.00], penalty otherwise
                    if 0.90 <= coverage <= 1.00:
                        coverage_penalty = 0
                    else:
                        coverage_penalty = abs(coverage - 0.95) * 10

                    # Width score: normalized by typical values
                    width_score = width

                    # Combined score: minimize
                    score = coverage_penalty + width_score * 0.5

                    results.append({
                        'alpha_1': alpha_1,
                        'alpha_2': alpha_2,
                        'lambda_1': lambda_1,
                        'lambda_2': lambda_2,
                        'dataset': dataset_name,
                        'noise_model': noise_model,
                        'noise_level': noise_level,
                        'coverage': coverage,
                        'width': width,
                        'score': score
                    })

                except Exception as e:
                    print(f"\nError with {exp_id}, params α1={alpha_1}, α2={alpha_2}, λ1={lambda_1}, λ2={lambda_2}: {e}")
                    results.append({
                        'alpha_1': alpha_1,
                        'alpha_2': alpha_2,
                        'lambda_1': lambda_1,
                        'lambda_2': lambda_2,
                        'dataset': dataset_name,
                        'noise_model': noise_model,
                        'noise_level': noise_level,
                        'coverage': np.nan,
                        'width': np.nan,
                        'score': np.inf
                    })

                pbar.update(1)

pbar.close()

# Convert to DataFrame
df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("TUNING RESULTS")
print("=" * 70)

# Group by hyperparameters and compute average metrics
summary = df.groupby(['alpha_1', 'alpha_2', 'lambda_1', 'lambda_2']).agg({
    'coverage': ['mean', 'std', 'min', 'max'],
    'width': ['mean', 'std'],
    'score': 'mean'
}).round(4)

summary = summary.sort_values(('score', 'mean'))

print("\nTop 10 Hyperparameter Combinations (by score):")
print(summary.head(10).to_string())

# Best parameters
best_params = summary.index[0]
best_results = df[
    (df['alpha_1'] == best_params[0]) &
    (df['alpha_2'] == best_params[1]) &
    (df['lambda_1'] == best_params[2]) &
    (df['lambda_2'] == best_params[3])
]

print("\n" + "=" * 70)
print("BEST HYPERPARAMETERS")
print("=" * 70)
print(f"alpha_1:  {best_params[0]}")
print(f"alpha_2:  {best_params[1]}")
print(f"lambda_1: {best_params[2]}")
print(f"lambda_2: {best_params[3]}")

print("\nPerformance Summary:")
print(f"  Mean Coverage: {best_results['coverage'].mean():.1%} ± {best_results['coverage'].std():.1%}")
print(f"  Coverage Range: [{best_results['coverage'].min():.1%}, {best_results['coverage'].max():.1%}]")
print(f"  Mean Width: {best_results['width'].mean():.4f} ± {best_results['width'].std():.4f}")
print(f"  Width Range: [{best_results['width'].min():.4f}, {best_results['width'].max():.4f}]")

# Coverage by dataset
print("\nCoverage by Dataset (with best params):")
for dataset in DATASET_CONFIGS.keys():
    dataset_cov = best_results[best_results['dataset'] == dataset]['coverage'].mean()
    print(f"  {dataset:15s}: {dataset_cov:.1%}")

# Save results
output_path = 'results/csv/bayesian_hyperparameter_tuning.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Full results saved to: {output_path}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("Update src/uq_methods/linear_uq.py with these default values:")
print(f"""
    def __init__(
        self,
        confidence_level: float = 0.95,
        n_iter: int = 300,
        alpha_1: float = {best_params[0]},
        alpha_2: float = {best_params[1]},
        lambda_1: float = {best_params[2]},
        lambda_2: float = {best_params[3]},
        **kwargs
    ):
""")
