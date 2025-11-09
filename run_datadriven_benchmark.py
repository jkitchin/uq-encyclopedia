#!/usr/bin/env python
"""Run comprehensive UQ benchmark on data-driven models."""

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Linear datasets
from src.datasets.linear import LineDataset, PolynomialDataset

# Nonlinear datasets
from src.datasets.nonlinear import (
    ExponentialDecayDataset, LogisticGrowthDataset,
    MichaelisMentenDataset, GaussianDataset
)

# Data-driven models
from src.models.data_driven_models import DPOSEModel
from src.uq_methods.data_driven_uq import EnsembleUQ, CalibratedEnsembleUQ

# Metrics
from src.metrics.coverage import picp
from src.metrics.accuracy import root_mean_squared_error, mean_absolute_error, r2_score
from src.metrics.efficiency import mean_interval_width
from src.utils.seeds import get_experiment_seed
from config.global_config import RESULTS_CSV_DIR, NOISE_MODELS, NOISE_LEVELS

print("=" * 70)
print("DATA-DRIVEN MODELS UQ BENCHMARK")
print("=" * 70)

# Configuration
CONFIDENCE_LEVEL = 0.95

# DPOSE Model Configuration (as specified)
MODEL_CONFIG = {
    'n_hidden': 20,
    'n_ensemble': 32,
    'activation': 'tanh',
    'seed': 42,
    'loss_type': 'crps',
    'optimizer': 'bfgs',
}

# Dataset configurations
DATASET_CONFIGS = {
    # Linear datasets (linear in parameters)
    'Line': {'class': LineDataset, 'params': {'slope': 0.8, 'intercept': 0.1}},
    'Quadratic': {'class': PolynomialDataset, 'params': {'coefficients': [0.5, 2.0, -1.0]}},  # ax^2 + bx + c
    'Cubic': {'class': PolynomialDataset, 'params': {'coefficients': [0.5, -1.0, 2.0, 0.5]}},  # ax^3 + bx^2 + cx + d

    # Nonlinear datasets
    'ExponentialDecay': {'class': ExponentialDecayDataset, 'params': {'a': 2.0, 'b': 3.0, 'c': 0.5}},
    'LogisticGrowth': {'class': LogisticGrowthDataset, 'params': {'L': 1.0, 'k': 10.0, 'x0': 0.5}},
    'MichaelisMenten': {'class': MichaelisMentenDataset, 'params': {'Vmax': 1.0, 'Km': 0.3}},
    'Gaussian': {'class': GaussianDataset, 'params': {'a': 1.0, 'mu': 0.5, 'sigma': 0.15}},
}

# UQ Methods
UQ_METHODS = {
    'Ensemble': EnsembleUQ(confidence_level=CONFIDENCE_LEVEL),
    'EnsembleCalibrated': CalibratedEnsembleUQ(confidence_level=CONFIDENCE_LEVEL),
}

# Generate all experiment configurations
print(f"\nConfigurations:")
print(f"  Datasets: {len(DATASET_CONFIGS)}")
print(f"  Noise models: {len(NOISE_MODELS)}")
print(f"  Noise levels: {len(NOISE_LEVELS)}")
print(f"  UQ methods: {len(UQ_METHODS)}")
total_experiments = len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS) * len(UQ_METHODS)
print(f"  Total experiments: {total_experiments}")

print(f"\nDPOSE Model Configuration:")
for key, value in MODEL_CONFIG.items():
    print(f"  {key}: {value}")

# Run experiments
results = []
experiment_count = 0

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

            for uq_method_name, uq_method in UQ_METHODS.items():
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] {exp_id} - {uq_method_name}")

                try:
                    # Create and fit model
                    model = DPOSEModel(**MODEL_CONFIG)
                    model.fit(data.X_train, data.y_train)

                    # Get predictions
                    y_pred = model.predict(data.X_test)

                    # Compute UQ intervals
                    uq_result = uq_method.compute_intervals(
                        model, data.X_train, data.y_train, data.X_test, data.y_test
                    )

                    # Compute metrics
                    coverage = picp(data.y_test, uq_result)
                    rmse = root_mean_squared_error(data.y_test, uq_result.y_pred)
                    mae = mean_absolute_error(data.y_test, uq_result.y_pred)
                    r2 = r2_score(data.y_test, uq_result.y_pred)
                    mean_width = mean_interval_width(uq_result.y_lower, uq_result.y_upper)

                    # Store results
                    result = {
                        'exp_id': exp_id,
                        'dataset': dataset_name,
                        'noise_model': noise_model,
                        'noise_level': noise_level,
                        'uq_method': uq_method_name,
                        'coverage': coverage,
                        'mean_width': mean_width,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                    }
                    results.append(result)

                    print(f"  ✓ Coverage: {coverage:.2%}, Width: {mean_width:.4f}, R²: {r2:.4f}")

                except Exception as e:
                    print(f"  ✗ Failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Store failed result
                    result = {
                        'exp_id': exp_id,
                        'dataset': dataset_name,
                        'noise_model': noise_model,
                        'noise_level': noise_level,
                        'uq_method': uq_method_name,
                        'coverage': np.nan,
                        'mean_width': np.nan,
                        'rmse': np.nan,
                        'mae': np.nan,
                        'r2': np.nan,
                    }
                    results.append(result)

# Save results
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

df = pd.DataFrame(results)
output_path = RESULTS_CSV_DIR / 'datadriven_models_comprehensive.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Results saved to: {output_path}")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

successful = df.dropna(subset=['coverage'])
print(f"\nSuccessful experiments: {len(successful)}/{len(df)}")

if len(successful) > 0:
    print(f"\nCoverage by UQ method:")
    for method in UQ_METHODS.keys():
        method_df = successful[successful['uq_method'] == method]
        if len(method_df) > 0:
            mean_cov = method_df['coverage'].mean()
            std_cov = method_df['coverage'].std()
            print(f"  {method}: {mean_cov:.1%} ± {std_cov:.1%}")

    print(f"\nMean interval width by UQ method:")
    for method in UQ_METHODS.keys():
        method_df = successful[successful['uq_method'] == method]
        if len(method_df) > 0:
            mean_width = method_df['mean_width'].mean()
            std_width = method_df['mean_width'].std()
            print(f"  {method}: {mean_width:.4f} ± {std_width:.4f}")

    print(f"\nR² by UQ method:")
    for method in UQ_METHODS.keys():
        method_df = successful[successful['uq_method'] == method]
        if len(method_df) > 0:
            mean_r2 = method_df['r2'].mean()
            std_r2 = method_df['r2'].std()
            print(f"  {method}: {mean_r2:.4f} ± {std_r2:.4f}")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE!")
print("=" * 70)
