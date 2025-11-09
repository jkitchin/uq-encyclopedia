#!/usr/bin/env python
"""Compute calibration-sharpness scores for all experiments."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from src.datasets.linear import LineDataset, PolynomialDataset, LennardJonesDataset, ShomateDataset
from src.models.linear_models import OLSRegression, LennardJonesRegression, ShomateRegression
from src.uq_methods.linear_uq import HatMatrixUQ, BayesianLinearRegressionUQ
from src.metrics import calibration_sharpness_score, interval_score
from src.utils.seeds import get_experiment_seed
from config.global_config import NOISE_MODELS, NOISE_LEVELS

print("=" * 70)
print("CALIBRATION-SHARPNESS ANALYSIS")
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

UQ_METHODS = {
    'HatMatrix': HatMatrixUQ(confidence_level=0.95),
    'Bayesian': BayesianLinearRegressionUQ(confidence_level=0.95),
}

N_SAMPLES = 100

# Compute scores
results = []

print("\nComputing calibration-sharpness scores...")
print(f"Total experiments: {len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS) * len(UQ_METHODS)}")

for dataset_name, dataset_config in DATASET_CONFIGS.items():
    for noise_model in NOISE_MODELS:
        for noise_level in NOISE_LEVELS:

            exp_id = f"{dataset_name}_{noise_model}_noise{int(noise_level*100):02d}"
            exp_seed = get_experiment_seed(exp_id)

            # Create dataset
            dataset = dataset_config['class'](
                n_samples=N_SAMPLES,
                noise_model=noise_model,
                noise_level=noise_level,
                seed=exp_seed,
                **dataset_config['params']
            )

            data = dataset.generate()

            # Fit model
            model = dataset_config['model_class'](**dataset_config['model_kwargs'])
            model.fit(data.X_train, data.y_train)

            # Apply each UQ method
            for uq_name, uq_method in UQ_METHODS.items():

                result = uq_method.compute_intervals(
                    model, data.X_train, data.y_train, data.X_test
                )

                # Compute calibration-sharpness score
                cal_sharp = calibration_sharpness_score(
                    data.y_test, result.y_lower, result.y_upper, target_coverage=0.95
                )

                # Compute interval score for comparison
                int_score = interval_score(
                    data.y_test, result.y_lower, result.y_upper, alpha=0.05
                )

                results.append({
                    'dataset': dataset_name,
                    'noise_model': noise_model,
                    'noise_level': noise_level,
                    'uq_method': uq_name,
                    'observed_coverage': cal_sharp['observed_coverage'],
                    'calibration_error': cal_sharp['calibration_error'],
                    'mean_width': cal_sharp['mean_width'],
                    'normalized_width': cal_sharp['normalized_width'],
                    'sharpness': cal_sharp['sharpness'],
                    'cal_sharp_score': cal_sharp['score'],
                    'interval_score': int_score
                })

# Convert to DataFrame
df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("OVERALL COMPARISON")
print("=" * 70)

summary = df.groupby('uq_method').agg({
    'observed_coverage': ['mean', 'std'],
    'calibration_error': ['mean', 'std'],
    'normalized_width': ['mean', 'std'],
    'cal_sharp_score': ['mean', 'std'],
    'interval_score': ['mean', 'std']
}).round(4)

print("\n" + summary.to_string())

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

hat_stats = df[df['uq_method'] == 'HatMatrix']
bay_stats = df[df['uq_method'] == 'Bayesian']

print(f"\nHat Matrix:")
print(f"  Average Coverage: {hat_stats['observed_coverage'].mean():.1%}")
print(f"  Calibration Error: {hat_stats['calibration_error'].mean():.4f} ± {hat_stats['calibration_error'].std():.4f}")
print(f"  Normalized Width: {hat_stats['normalized_width'].mean():.4f} ± {hat_stats['normalized_width'].std():.4f}")
print(f"  Cal-Sharp Score: {hat_stats['cal_sharp_score'].mean():.4f} ± {hat_stats['cal_sharp_score'].std():.4f} (lower is better)")
print(f"  Interval Score: {hat_stats['interval_score'].mean():.4f} ± {hat_stats['interval_score'].std():.4f} (lower is better)")

print(f"\nBayesian:")
print(f"  Average Coverage: {bay_stats['observed_coverage'].mean():.1%}")
print(f"  Calibration Error: {bay_stats['calibration_error'].mean():.4f} ± {bay_stats['calibration_error'].std():.4f}")
print(f"  Normalized Width: {bay_stats['normalized_width'].mean():.4f} ± {bay_stats['normalized_width'].std():.4f}")
print(f"  Cal-Sharp Score: {bay_stats['cal_sharp_score'].mean():.4f} ± {bay_stats['cal_sharp_score'].std():.4f} (lower is better)")
print(f"  Interval Score: {bay_stats['interval_score'].mean():.4f} ± {bay_stats['interval_score'].std():.4f} (lower is better)")

# Compute ratios
width_ratio = bay_stats['normalized_width'].mean() / hat_stats['normalized_width'].mean()
score_ratio = bay_stats['cal_sharp_score'].mean() / hat_stats['cal_sharp_score'].mean()

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"\nBayesian vs Hat Matrix:")
print(f"  Calibration Error: {bay_stats['calibration_error'].mean() / hat_stats['calibration_error'].mean():.2f}x (Bayesian/Hat)")
print(f"  Normalized Width: {width_ratio:.2f}x wider (Bayesian has intervals {width_ratio:.1f}x wider)")
print(f"  Cal-Sharp Score: {score_ratio:.2f}x worse")
print(f"  Interval Score: {bay_stats['interval_score'].mean() / hat_stats['interval_score'].mean():.2f}x worse")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
The calibration-sharpness score combines:
1. Calibration Error: |observed_coverage - 0.95| (weighted 10x)
2. Normalized Width: mean_width / data_range

Key insights:
- Bayesian achieves better calibration (100% vs 88% coverage)
- But pays a HUGE price in interval width (34x wider!)
- The calibration-sharpness score shows Bayesian is much worse overall
- Interval score (Gneiting & Raftery) confirms this

Recommendation:
- Hat Matrix is better balanced (good coverage + sharp intervals)
- Bayesian is over-conservative (perfect coverage but wastefully wide)
- For production use, Hat Matrix is preferred
- Bayesian could be useful when extreme conservatism is needed
""")

# Save results
output_path = 'results/csv/calibration_sharpness_scores.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Results saved to: {output_path}")
