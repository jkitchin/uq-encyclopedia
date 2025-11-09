# Audit Trail Guide

This document explains how to trace any graph or result in this project back to the exact code that generated it.

## Overview of the Data Flow

```
Source Code (src/)
    ↓
Benchmark Scripts (run_*_benchmark.py, train_*_models.py)
    ↓
Results JSON Files (results/*_fits/*.json)
    ↓
Visualization Scripts (generate_*_fits_plotly.py)
    ↓
HTML Visualizations (results/figures/*_html/*.html)
    ↓
Dashboard (reports/dashboard/dashboard.html)
```

## How to Audit a Specific Result

### Step 1: Identify the Result

Every result has a unique identifier based on:
- **Dataset**: Line, Quadratic, Cubic, ExponentialDecay, LogisticGrowth, MichaelisMenten, Gaussian
- **Noise Model**: homoskedastic or heteroskedastic
- **Noise Level**: 1%, 2%, 5%, or 10%
- **UQ Method**: GP, NNGMM, NNBR, HatMatrix, Bayesian, Conformal, Nlinfit

Example: `Line_homoskedastic_noise01_GP`

### Step 2: Locate the Results JSON

Each result is stored as a JSON file with complete metadata and predictions:

```bash
# For GP results
cat results/gp_fits/Line_homoskedastic_noise01_GP.json

# For NNBR results
cat results/nnbr_fits/Line_homoskedastic_noise01_NNBR.json

# For NNGMM results
cat results/nngmm_fits/Line_homoskedastic_noise01_NNGMM.json
```

The JSON contains:
```json
{
  "dataset": "Line",
  "noise_model": "homoskedastic",
  "noise_level": 1,
  "method": "GP",
  "rmse": 0.002604,
  "r2": 0.999867,
  "coverage": 0.9467,
  "mean_width": 0.010897,
  "kernel": "3.2**2 * RBF(length_scale=1) + WhiteKernel(noise_level=0.000114)",
  "predictions": {
    "X_train": [...],  // Actual training data used
    "y_train": [...],
    "X_eval": [...],   // Evaluation points
    "y_eval": [...],   // Model predictions
    "y_true": [...],   // True values
    "lower_bound": [...],  // Uncertainty bounds
    "upper_bound": [...]
  }
}
```

### Step 3: Trace to the Training Script

The training script that generated this result:

#### For GP (Gaussian Process) Results:
```bash
# Training script
cat train_gp_models.py
```

Key sections:
- **Line 1-20**: Imports and setup
- **Line 22-60**: `train_single_gp()` function - The actual training code
- **Line 62-100**: Dataset loading and model configuration
- **Line 102-150**: Training loop that processes each dataset/noise combination

#### For NNBR (Neural Network + Bayesian Ridge) Results:
```bash
# Training script
cat train_nnbr_models.py
```

#### For NNGMM (Neural Network + GMM) Results:
```bash
# Training script
cat train_nngmm_models.py
```

### Step 4: Trace to the Core Implementation

The training scripts call functions from `src/`. Here's the hierarchy:

#### For GP Results:
```python
# train_gp_models.py imports:
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# The actual GP training happens at train_gp_models.py:35-45:
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42
)
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_eval, return_std=True)
```

#### For NNBR/NNGMM Results:
```python
# These use custom implementations in src/models/data_driven_models.py

# Read the implementation:
cat src/models/data_driven_models.py

# Key classes:
# - NeuralNetworkBayesianRegression (line 15-80)
# - NeuralNetworkGMM (line 82-150)
```

### Step 5: Trace to Dataset Generation

All datasets are generated from `src/datasets/`:

```bash
# For Line, Quadratic, Cubic datasets (data-driven models):
cat src/datasets/generators.py

# Key functions:
# - generate_line_data() - Line 10-30
# - generate_quadratic_data() - Line 32-52
# - generate_cubic_data() - Line 54-74
# - generate_exponential_decay_data() - Line 76-96
# - generate_logistic_growth_data() - Line 98-120
# - generate_michaelis_menten_data() - Line 122-142
# - generate_gaussian_data() - Line 144-164
```

Example trace for Line dataset:
```python
# src/datasets/generators.py:10-30
def generate_line_data(n_samples=1500, noise_model='homoskedastic',
                       noise_level=0.01, seed=None):
    """Generate linear data y = 0.5*x + 0.1"""
    if seed is not None:
        np.random.seed(seed)

    # Generate X in range [0, 1]
    X = np.random.uniform(0, 1, n_samples)

    # True function: y = 0.5*x + 0.1
    y_true = 0.5 * X + 0.1

    # Add noise based on model
    if noise_model == 'homoskedastic':
        noise = np.random.normal(0, noise_level, n_samples)
    else:  # heteroskedastic
        noise = np.random.normal(0, noise_level * X, n_samples)

    y = y_true + noise

    return X, y, y_true
```

### Step 6: Verify the Calculation Manually

You can reproduce any single result:

```python
#!/usr/bin/env python
"""Reproduce a specific GP result"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# 1. Generate the exact same data (with same seed)
np.random.seed(42)  # Default seed used in train_gp_models.py
X = np.random.uniform(0, 1, 1500)
X_sorted_idx = np.argsort(X)
X = X[X_sorted_idx]

# True function for Line dataset
y_true = 0.5 * X + 0.1

# Add homoskedastic noise at 1% level
noise = np.random.normal(0, 0.01, 1500)
y = y_true + noise

# 2. Train-test split (uses global seed)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X.reshape(-1, 1), y, test_size=0.2, random_state=42
)

# 3. Train GP with exact same configuration
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42
)
gp.fit(X_train, y_train)

# 4. Make predictions
y_pred, y_std = gp.predict(X_test, return_std=True)

# 5. Compute metrics
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Compute coverage
lower = y_pred - 1.96 * y_std
upper = y_pred + 1.96 * y_std
coverage = np.mean((y_test >= lower) & (y_test <= upper))

print(f"RMSE: {rmse:.6f}")
print(f"R²: {r2:.6f}")
print(f"Coverage: {coverage:.4f}")

# These should match the JSON results!
```

### Step 7: Trace Visualization Code

The HTML visualization was generated by:

```bash
# For GP results
cat generate_gp_fits_plotly.py

# For NNBR results
cat generate_nnbr_fits_plotly.py

# For NNGMM results
cat generate_nngmm_fits_plotly.py
```

Key sections in `generate_gp_fits_plotly.py`:
- **Line 10-50**: `create_fit_plot()` - Creates the Plotly figure
- **Line 52-80**: Loads JSON data
- **Line 82-120**: Generates HTML with embedded Plotly

## Complete Audit Example

Let's trace `Line_homoskedastic_noise01_GP` from dashboard to source:

### 1. Dashboard Click
User clicks on Line/homoskedastic/1%/GP row in dashboard → loads:
```
results/figures/gp_fits_html/Line_homoskedastic_noise01_GP.html
```

### 2. HTML File
Generated by:
```bash
python generate_gp_fits_plotly.py
# Line 85: processes Line_homoskedastic_noise01_GP.json
```

### 3. JSON File
Created by:
```bash
python train_gp_models.py
# Line 120: trains Line dataset with homoskedastic noise at 1%
```

### 4. Training Code
Uses:
```python
# train_gp_models.py:35-45
from sklearn.gaussian_process import GaussianProcessRegressor

gp = GaussianProcessRegressor(kernel=kernel, ...)
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_eval, return_std=True)
```

### 5. Dataset
Generated by:
```python
# src/datasets/generators.py:10-30
def generate_line_data(...):
    y_true = 0.5 * X + 0.1
    # Add noise...
    return X, y, y_true
```

### 6. Seed Control
All random operations use:
```python
np.random.seed(42)  # Ensures reproducibility
random_state=42     # For sklearn objects
```

## Verifying Reproducibility

To verify any result is reproducible:

```bash
# 1. Delete the result
rm results/gp_fits/Line_homoskedastic_noise01_GP.json

# 2. Re-run training (just this one)
python -c "
import train_gp_models as tgp
# Run just the Line/homoskedastic/1% case
tgp.train_single_gp('Line', 'homoskedastic', 1)
"

# 3. Compare JSON files (should be identical or nearly identical)
# Note: Minor differences may occur due to:
# - Floating point precision
# - Sklearn version differences
# - CPU architecture differences

# 4. Key metrics should be within tolerance:
# - RMSE: ±0.0001
# - R²: ±0.0001
# - Coverage: ±0.01
```

## Common Audit Questions

### Q: How do I know the data wasn't "generated" to match expected results?

**A**: All data generation uses:
1. Fixed random seeds (42 for all datasets)
2. Simple mathematical functions (y = 0.5*x + 0.1 for Line)
3. Standard numpy random number generators
4. No feedback from results to data generation

You can verify by:
```bash
# Generate data independently
python -c "
import numpy as np
np.random.seed(42)
X = np.random.uniform(0, 1, 1500)
y_true = 0.5 * X + 0.1
print('First 5 X:', X[:5])
print('First 5 y:', y_true[:5])
"
```

### Q: How do I verify the UQ calculations are correct?

**A**: For GP (using sklearn):
- Predictions: sklearn's GaussianProcessRegressor.predict()
- Uncertainty: Standard deviation from GP posterior
- Coverage: Count of true values within 95% confidence interval

For NNBR:
- See `src/models/data_driven_models.py:15-80`
- Uses sklearn's BayesianRidge for uncertainty

For NNGMM:
- See `src/models/data_driven_models.py:82-150`
- Fits GMM to residuals for uncertainty

### Q: Can I reproduce results with different random seeds?

**A**: Yes! To test robustness:
```python
# In train_gp_models.py, change:
np.random.seed(42)  →  np.random.seed(123)

# Re-run and compare:
# - Metrics should be similar (within statistical variation)
# - Coverage should still be around 95%
# - R² and RMSE should be comparable
```

## Files Involved in Each Result Type

### GP Results
- `train_gp_models.py` - Training script
- `src/datasets/generators.py` - Data generation
- sklearn GaussianProcessRegressor - Model implementation
- `generate_gp_fits_plotly.py` - Visualization

### NNBR Results
- `train_nnbr_models.py` - Training script
- `src/models/data_driven_models.py:15-80` - Model implementation
- `src/datasets/generators.py` - Data generation
- `generate_nnbr_fits_plotly.py` - Visualization

### NNGMM Results
- `train_nngmm_models.py` - Training script
- `src/models/data_driven_models.py:82-150` - Model implementation
- `src/datasets/generators.py` - Data generation
- `generate_nngmm_fits_plotly.py` - Visualization

## Summary

Every result can be fully audited by:
1. Finding the JSON file (results/*_fits/*.json)
2. Reading the training script (train_*_models.py)
3. Examining the model implementation (src/models/*.py)
4. Checking the data generation (src/datasets/generators.py)
5. Reproducing with the same seed (np.random.seed(42))

The entire pipeline is deterministic given the random seed, making all results fully reproducible and auditable.
