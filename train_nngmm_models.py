#!/usr/bin/env python
"""Train Neural Network GMM models on all datasets."""

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.neural_network import MLPRegressor
from pycse.sklearn.nngmm import NeuralNetworkGMM
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import dataset classes
from src.datasets.linear import LineDataset, PolynomialDataset
from src.datasets.nonlinear import (
    ExponentialDecayDataset, LogisticGrowthDataset,
    MichaelisMentenDataset, GaussianDataset
)
from src.utils.seeds import get_experiment_seed
from config.global_config import NOISE_MODELS, NOISE_LEVELS

print("=" * 70)
print("NEURAL NETWORK GMM UQ BENCHMARK")
print("=" * 70)

# Results directory
results_dir = Path('results/nngmm_fits')
results_dir.mkdir(parents=True, exist_ok=True)

# Dataset configurations (matching DPOSE benchmark)
DATASET_CONFIGS = {
    # Linear datasets
    'Line': {'class': LineDataset, 'params': {'slope': 0.8, 'intercept': 0.1}},
    'Quadratic': {'class': PolynomialDataset, 'params': {'coefficients': [0.5, 2.0, -1.0]}},
    'Cubic': {'class': PolynomialDataset, 'params': {'coefficients': [0.5, -1.0, 2.0, 0.5]}},

    # Nonlinear datasets
    'ExponentialDecay': {'class': ExponentialDecayDataset, 'params': {'a': 2.0, 'b': 3.0, 'c': 0.5}},
    'LogisticGrowth': {'class': LogisticGrowthDataset, 'params': {'L': 1.0, 'k': 10.0, 'x0': 0.5}},
    'MichaelisMenten': {'class': MichaelisMentenDataset, 'params': {'Vmax': 1.0, 'Km': 0.3}},
    'Gaussian': {'class': GaussianDataset, 'params': {'a': 1.0, 'mu': 0.5, 'sigma': 0.15}},
}

# Configuration
print(f"\nConfigurations:")
print(f"  Datasets: {len(DATASET_CONFIGS)}")
print(f"  Noise models: {len(NOISE_MODELS)}")
print(f"  Noise levels: {len(NOISE_LEVELS)}")
total_experiments = len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS)
print(f"  Total experiments: {total_experiments}")

# Storage for results
all_results = []
experiment_count = 0

for dataset_name, dataset_config in DATASET_CONFIGS.items():
    for noise_model in NOISE_MODELS:
        for noise_level in NOISE_LEVELS:
            experiment_count += 1

            # Generate experiment ID and seed
            exp_id = f"{dataset_name}_{noise_model}_noise{int(noise_level*100):02d}"
            seed = get_experiment_seed(exp_id)

            print(f"\n[{experiment_count}/{total_experiments}] {dataset_name} ({noise_model}, {int(noise_level*100)}% noise)")

            # Create dataset
            dataset = dataset_config['class'](
                n_samples=100,
                noise_model=noise_model,
                noise_level=noise_level,
                seed=seed,
                **dataset_config['params']
            )
            data = dataset.generate()

            # Extract data from DataSplit object
            X_train = data.X_train.reshape(-1, 1)
            y_train = data.y_train
            X_test = data.X_test.reshape(-1, 1)
            y_test = data.y_test
            X_gap = data.X_gap.reshape(-1, 1) if data.X_gap is not None else None
            y_gap = data.y_gap if data.y_gap is not None else None

            # Combine test and gap for evaluation
            if X_gap is not None:
                X_eval = np.vstack([X_test, X_gap])
                y_eval = np.hstack([y_test, y_gap])
            else:
                X_eval = X_test
                y_eval = y_test

            # Create MLPRegressor as the backend neural network
            # Using similar architecture to DPOSE (20 hidden neurons)
            mlp = MLPRegressor(
                hidden_layer_sizes=(20,),
                activation='tanh',
                solver='lbfgs',
                max_iter=1000,
                random_state=seed
            )

            # Create NNGMM with 3 GMM components for uncertainty estimation
            nngmm = NeuralNetworkGMM(
                nn=mlp,
                n_components=3,
                n_samples=500
            )

            print("  Fitting NNGMM model...")
            nngmm.fit(X_train, y_train)

            # Make predictions on evaluation set
            y_pred, y_std = nngmm.predict(X_eval, return_std=True)

            # Flatten y_pred to avoid broadcasting issues (it comes as shape (n, 1))
            y_pred = y_pred.flatten()

            # Compute 95% prediction intervals
            # For NNGMM, we use 1.96 * std for 95% coverage
            z_score = 1.96
            y_lower = y_pred - z_score * y_std
            y_upper = y_pred + z_score * y_std

            # Compute metrics
            rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
            r2 = r2_score(y_eval, y_pred)

            # Coverage: fraction of test points within prediction intervals
            coverage = np.mean((y_eval >= y_lower) & (y_eval <= y_upper))

            # Mean interval width
            mean_width = np.mean(y_upper - y_lower)

            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Coverage: {coverage:.3f}")
            print(f"  Mean Width: {mean_width:.4f}")

            # Save predictions
            results = {
                'dataset': dataset_name,
                'noise_model': noise_model,
                'noise_level': int(noise_level*100),
                'method': 'NNGMM',
                'rmse': float(rmse),
                'r2': float(r2),
                'coverage': float(coverage),
                'mean_width': float(mean_width),
                'predictions': {
                    'X_train': X_train.flatten().tolist(),
                    'y_train': y_train.tolist(),
                    'X_eval': X_eval.flatten().tolist(),
                    'y_eval': y_eval.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_lower': y_lower.tolist(),
                    'y_upper': y_upper.tolist(),
                    'y_std': y_std.tolist()
                }
            }

            # Save to JSON
            result_file = results_dir / f'{dataset_name}_{noise_model}_noise{int(noise_level*100):02d}_NNGMM.json'
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)

            all_results.append({
                'Dataset': dataset_name,
                'Noise Model': noise_model.capitalize(),
                'Noise Level': f'{int(noise_level*100)}%',
                'Method': 'NNGMM',
                'Coverage': f'{coverage:.3f}',
                'RMSE': f'{rmse:.4f}',
                'Mean Width': f'{mean_width:.4f}',
                'R²': f'{r2:.4f}'
            })

# Save summary CSV
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(results_dir / 'nngmm_results_summary.csv', index=False)

print(f"\n✓ Completed training {len(all_results)} NNGMM models")
print(f"✓ Results saved to {results_dir}")
print("\nSummary statistics:")
print(f"  Average Coverage: {summary_df['Coverage'].astype(float).mean():.3f}")
print(f"  Average RMSE: {summary_df['RMSE'].astype(float).mean():.4f}")
print(f"  Average Mean Width: {summary_df['Mean Width'].astype(float).mean():.4f}")
print(f"  Average R²: {summary_df['R²'].astype(float).mean():.4f}")
