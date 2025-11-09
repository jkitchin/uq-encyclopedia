#!/usr/bin/env python
"""Test training plot generation."""

import sys
sys.path.insert(0, '.')
import numpy as np

from src.datasets.linear import LineDataset
from src.utils.seeds import get_experiment_seed

# Test single plot
exp_id = "Line_homoskedastic_noise01"
seed = get_experiment_seed(exp_id)

dataset = LineDataset(
    n_samples=100,
    noise_model='homoskedastic',
    noise_level=0.01,
    seed=seed,
    slope=0.8,
    intercept=0.1
)
data = dataset.generate()

print("X_train shape:", data.X_train.shape)
print("y_train shape:", data.y_train.shape)
print("X_train min/max:", data.X_train.min(), data.X_train.max())
print("y_train min/max:", data.y_train.min(), data.y_train.max())
print("\nFirst 5 training points:")
print("X:", data.X_train.flatten()[:5])
print("y:", data.y_train[:5])

# Check if y_train is 1D or 2D
print("\ny_train ndim:", data.y_train.ndim)
if data.y_train.ndim > 1:
    print("y_train needs flattening!")
    print("Flattened shape:", data.y_train.flatten().shape)
