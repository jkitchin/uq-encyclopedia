#!/usr/bin/env python
"""Test pycse.regress and pycse.predict integration."""

import numpy as np
import sys
sys.path.insert(0, '.')
from pycse import regress, predict

from src.datasets.linear import LineDataset
from src.models.linear_models import OLSRegression

# Create simple test case
dataset = LineDataset(
    n_samples=100,
    noise_model='homoskedastic',
    noise_level=0.05,
    seed=42,
    slope=0.8,
    intercept=0.1
)

data = dataset.generate()

# Fit OLS model to get design matrix
model = OLSRegression(degree=1, fit_intercept=True)
model.fit(data.X_train, data.y_train)

print("=" * 70)
print("TESTING PYCSE INTEGRATION")
print("=" * 70)

# Get design matrices
X_train_design = model._create_design_matrix(data.X_train)
X_test_design = model._create_design_matrix(data.X_test)

print(f"\nTraining data:")
print(f"  X_train_design shape: {X_train_design.shape}")
print(f"  y_train shape: {data.y_train.shape}")

# Use pycse.regress to fit
print(f"\n1. Fitting with pycse.regress:")
pars, pars_int, se = regress(X_train_design, data.y_train, alpha=0.05)
print(f"  Parameters: {pars}")
print(f"  Parameter intervals:\n{pars_int}")
print(f"  Standard errors: {se}")

# Compare with OLS
print(f"\n2. Compare with our OLS:")
print(f"  OLS coefficients: {model.coef_}")
print(f"  Match: {np.allclose(pars, model.coef_)}")

# Use pycse.predict for prediction intervals
print(f"\n3. Prediction intervals with pycse.predict:")
y_pred, y_int, pred_se = predict(
    X_train_design,
    data.y_train,
    pars,
    X_test_design,
    alpha=0.05
)

print(f"  y_pred shape: {y_pred.shape}")
print(f"  y_int shape: {y_int.shape}")
print(f"  pred_se shape: {pred_se.shape}")

# y_int is (2, n) where row 0 is lower, row 1 is upper
y_lower = y_int[0, :]
y_upper = y_int[1, :]
print(f"  y_lower shape: {y_lower.shape}")
print(f"  y_upper shape: {y_upper.shape}")
print(f"  Mean prediction interval width: {np.mean(y_upper - y_lower):.4f}")

# Check coverage
coverage = np.mean((data.y_test >= y_lower) & (data.y_test <= y_upper))
print(f"  Coverage: {coverage:.1%}")

# Compare with our current implementation
from src.uq_methods.linear_uq import HatMatrixUQ

hat = HatMatrixUQ(confidence_level=0.95)
result = hat.compute_intervals(model, data.X_train, data.y_train, data.X_test)

print(f"\n4. Compare with current HatMatrixUQ:")
print(f"  Current mean width: {np.mean(result.y_upper - result.y_lower):.4f}")
print(f"  Current coverage: {np.mean((data.y_test >= result.y_lower) & (data.y_test <= result.y_upper)):.1%}")
print(f"  pycse mean width: {np.mean(y_int[:, 1] - y_int[:, 0]):.4f}")
print(f"  pycse coverage: {coverage:.1%}")

# Check if they match
print(f"\n5. Do they match?")
print(f"  Predictions match: {np.allclose(y_pred, result.y_pred)}")
print(f"  Lower bounds match: {np.allclose(y_lower, result.y_lower, atol=1e-4)}")
print(f"  Upper bounds match: {np.allclose(y_upper, result.y_upper, atol=1e-4)}")
print(f"  Max difference in lower: {np.max(np.abs(y_lower - result.y_lower)):.6f}")
print(f"  Max difference in upper: {np.max(np.abs(y_upper - result.y_upper)):.6f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("pycse.predict can be used for Hat Matrix UQ!")
print("The results should match our current implementation.")
