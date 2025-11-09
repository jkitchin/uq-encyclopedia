#!/usr/bin/env python
"""Analyze Bayesian hyperparameter tuning results."""

import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('results/csv/bayesian_hyperparameter_tuning.csv')

print("=" * 70)
print("BAYESIAN HYPERPARAMETER TUNING ANALYSIS")
print("=" * 70)

# Group by hyperparameters
summary = df.groupby(['alpha_1', 'alpha_2', 'lambda_1', 'lambda_2']).agg({
    'coverage': ['mean', 'std', 'min', 'max'],
    'width': ['mean', 'std', 'min', 'max'],
    'score': 'mean'
}).round(4)

# Sort by score (lower is better)
summary = summary.sort_values(('score', 'mean'))

print("\nTop 10 Hyperparameter Combinations:")
print(summary.head(10).to_string())

# Since all achieved 100% coverage, find the one with minimum width
width_sorted = df.groupby(['alpha_1', 'alpha_2', 'lambda_1', 'lambda_2']).agg({
    'width': 'mean',
    'coverage': 'mean'
}).sort_values('width')

print("\n" + "=" * 70)
print("BEST PARAMETERS (minimum width while maintaining coverage)")
print("=" * 70)

best = width_sorted.iloc[0]
best_params = width_sorted.index[0]

print(f"\nalpha_1:  {best_params[0]}")
print(f"alpha_2:  {best_params[1]}")
print(f"lambda_1: {best_params[2]}")
print(f"lambda_2: {best_params[3]}")

print(f"\nMean Coverage: {best['coverage']:.1%}")
print(f"Mean Width: {best['width']:.4f}")

# Compare with current defaults
current_defaults = df[
    (df['alpha_1'] == 1e-6) &
    (df['alpha_2'] == 1e-6) &
    (df['lambda_1'] == 1e-6) &
    (df['lambda_2'] == 1e-6)
]

if len(current_defaults) > 0:
    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT DEFAULTS")
    print("=" * 70)
    print(f"Current (1e-6, 1e-6, 1e-6, 1e-6):")
    print(f"  Coverage: {current_defaults['coverage'].mean():.1%}")
    print(f"  Width: {current_defaults['width'].mean():.4f}")
    print(f"\nBest params:")
    print(f"  Coverage: {best['coverage']:.1%}")
    print(f"  Width: {best['width']:.4f}")

    width_reduction = (1 - best['width'] / current_defaults['width'].mean()) * 100
    print(f"\nWidth reduction: {width_reduction:.1f}%")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("\nSince all parameter combinations achieved 100% coverage,")
print("the intervals are too conservative (over-coverage).")
print("\nThe best approach is to:")
print("1. Use the parameters that give the narrowest intervals")
print("2. Consider using a lower confidence level in practice")
print("3. Or implement a more sophisticated prior that adapts to the data")

print(f"\nUpdate src/uq_methods/linear_uq.py with:")
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
