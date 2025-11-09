#!/usr/bin/env python
"""Test different approaches for Bayesian interval estimation."""

import numpy as np
import sys
sys.path.insert(0, '.')

from src.datasets.linear import LineDataset
from src.models.linear_models import OLSRegression
from src.uq_methods.linear_uq import BayesianLinearRegressionUQ, HatMatrixUQ
from src.utils.seeds import get_experiment_seed

# Create a simple test case
dataset = LineDataset(
    n_samples=100,
    noise_model='homoskedastic',
    noise_level=0.05,
    seed=42,
    slope=0.8,
    intercept=0.1
)

data = dataset.generate()

# Fit OLS model
model = OLSRegression(degree=1, fit_intercept=True)
model.fit(data.X_train, data.y_train)

print("=" * 70)
print("TESTING BAYESIAN INTERVAL WIDTHS")
print("=" * 70)

# Test current defaults
print("\n1. Current Bayesian (tuned hyperparameters):")
bay_current = BayesianLinearRegressionUQ(confidence_level=0.95)
result_bay = bay_current.compute_intervals(model, data.X_train, data.y_train, data.X_test)

print(f"   Mean width: {np.mean(result_bay.y_upper - result_bay.y_lower):.4f}")
print(f"   Coverage: {np.mean((data.y_test >= result_bay.y_lower) & (data.y_test <= result_bay.y_upper)):.1%}")
print(f"   Hyperparameters: α1={bay_current.alpha_1}, α2={bay_current.alpha_2}, λ1={bay_current.lambda_1}, λ2={bay_current.lambda_2}")

# Test with stronger priors (more regularization)
print("\n2. Bayesian with stronger priors (more regularization):")
bay_strong = BayesianLinearRegressionUQ(
    confidence_level=0.95,
    alpha_1=1e-3,  # Much stronger
    alpha_2=1e-3,
    lambda_1=1e-3,
    lambda_2=1e-3
)
result_strong = bay_strong.compute_intervals(model, data.X_train, data.y_train, data.X_test)

print(f"   Mean width: {np.mean(result_strong.y_upper - result_strong.y_lower):.4f}")
print(f"   Coverage: {np.mean((data.y_test >= result_strong.y_lower) & (data.y_test <= result_strong.y_upper)):.1%}")

# Test with 90% confidence
print("\n3. Bayesian with 90% confidence level:")
bay_90 = BayesianLinearRegressionUQ(confidence_level=0.90)
result_90 = bay_90.compute_intervals(model, data.X_train, data.y_train, data.X_test)

print(f"   Mean width: {np.mean(result_90.y_upper - result_90.y_lower):.4f}")
print(f"   Coverage: {np.mean((data.y_test >= result_90.y_lower) & (data.y_test <= result_90.y_upper)):.1%}")

# Compare with Hat Matrix
print("\n4. Hat Matrix UQ (for comparison):")
hat = HatMatrixUQ(confidence_level=0.95)
result_hat = hat.compute_intervals(model, data.X_train, data.y_train, data.X_test)

print(f"   Mean width: {np.mean(result_hat.y_upper - result_hat.y_lower):.4f}")
print(f"   Coverage: {np.mean((data.y_test >= result_hat.y_lower) & (data.y_test <= result_hat.y_upper)):.1%}")

# Check what sklearn's BayesianRidge is learning
from sklearn.linear_model import BayesianRidge

print("\n" + "=" * 70)
print("SKLEARN BAYESIANRIDGE DIAGNOSTICS")
print("=" * 70)

X_train_design = model._create_design_matrix(data.X_train)
X_test_design = model._create_design_matrix(data.X_test)

bay_model = BayesianRidge(
    max_iter=300,
    alpha_1=1e-7,
    alpha_2=1e-5,
    lambda_1=1e-5,
    lambda_2=1e-7,
)
bay_model.fit(X_train_design, data.y_train)

print(f"\nLearned parameters:")
print(f"  Coefficients: {bay_model.coef_}")
print(f"  Intercept: {bay_model.intercept_:.4f}")
print(f"  Noise precision (alpha_): {bay_model.alpha_:.4f}")
print(f"  Weight precision (lambda_): {bay_model.lambda_:.4f}")
print(f"  Estimated noise std: {1.0 / np.sqrt(bay_model.alpha_):.4f}")

# Compare with OLS
print(f"\nOLS parameters (for comparison):")
print(f"  Coefficients: {model.coef_}")
print(f"  Residual std: {np.sqrt(model.sigma_squared_):.4f}")

# Get predictions with std
y_pred, y_std = bay_model.predict(X_test_design, return_std=True)

print(f"\nPredictive distribution:")
print(f"  Mean std: {np.mean(y_std):.4f}")
print(f"  Min std: {np.min(y_std):.4f}")
print(f"  Max std: {np.max(y_std):.4f}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

ratio = np.mean(result_bay.y_upper - result_bay.y_lower) / np.mean(result_hat.y_upper - result_hat.y_lower)
print(f"\nBayesian intervals are {ratio:.1f}x wider than Hat Matrix")

print("\nPossible issue:")
print("  BayesianRidge's predictive std accounts for parameter uncertainty,")
print("  which can be large with weak priors even when data is abundant.")
print("\nRecommendations:")
print("  1. Use Hat Matrix for production (better calibrated)")
print("  2. Use 90% confidence for Bayesian if narrower intervals needed")
print("  3. Accept that Bayesian is naturally more conservative")
