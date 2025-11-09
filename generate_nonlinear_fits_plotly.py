#!/usr/bin/env python
"""Generate interactive Plotly fit plots for nonlinear model experiments."""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from src.datasets.nonlinear import (
    ExponentialDecayDataset, LogisticGrowthDataset,
    MichaelisMentenDataset, GaussianDataset
)
from src.models.nonlinear_models import (
    NonlinearModel,
    exponential_decay_func, logistic_growth_func,
    michaelis_menten_func, gaussian_func
)
from src.uq_methods.nonlinear_uq import NlinfitUQ, ConformalPredictionNonlinear
from src.utils.seeds import get_experiment_seed
from config.global_config import RESULTS_FIGURES_DIR, NOISE_MODELS, NOISE_LEVELS

print("=" * 70)
print("GENERATING NONLINEAR MODEL FIT PLOTS")
print("=" * 70)

# Create subdirectory for nonlinear fits
FITS_DIR = RESULTS_FIGURES_DIR / 'nonlinear_fits_html'
FITS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASET_CONFIGS = {
    'ExponentialDecay': {
        'class': ExponentialDecayDataset,
        'model_func': exponential_decay_func,
        'param_names': ['a', 'b', 'c'],
        'initial_guess': [2.0, 3.0, 0.5],
        'params': {'a': 2.0, 'b': 3.0, 'c': 0.5}
    },
    'LogisticGrowth': {
        'class': LogisticGrowthDataset,
        'model_func': logistic_growth_func,
        'param_names': ['L', 'k', 'x0'],
        'initial_guess': [1.0, 10.0, 0.5],
        'params': {'L': 1.0, 'k': 10.0, 'x0': 0.5}
    },
    'MichaelisMenten': {
        'class': MichaelisMentenDataset,
        'model_func': michaelis_menten_func,
        'param_names': ['Vmax', 'Km'],
        'initial_guess': [1.0, 0.3],
        'params': {'Vmax': 1.0, 'Km': 0.3}
    },
    'Gaussian': {
        'class': GaussianDataset,
        'model_func': gaussian_func,
        'param_names': ['a', 'mu', 'sigma'],
        'initial_guess': [1.0, 0.5, 0.15],
        'params': {'a': 1.0, 'mu': 0.5, 'sigma': 0.15}
    }
}

# UQ methods
UQ_METHODS = {
    'Nlinfit': NlinfitUQ(confidence_level=0.95),
    'Conformal': ConformalPredictionNonlinear(confidence_level=0.95),
}

# Generate all combinations
experiments = []
for dataset_name, dataset_config in DATASET_CONFIGS.items():
    for noise_model in NOISE_MODELS:
        for noise_level in NOISE_LEVELS:
            for uq_method_name, uq_method in UQ_METHODS.items():
                experiments.append({
                    'dataset_name': dataset_name,
                    'dataset_config': dataset_config,
                    'noise_model': noise_model,
                    'noise_level': noise_level,
                    'uq_method_name': uq_method_name,
                    'uq_method': uq_method,
                })

print(f"\nGenerating {len(experiments)} individual fit plots...")
print(f"Saving to: {FITS_DIR}\n")

# Generate plots
for exp in tqdm(experiments, desc="Generating plots"):
    dataset_name = exp['dataset_name']
    dataset_config = exp['dataset_config']
    noise_model = exp['noise_model']
    noise_level = exp['noise_level']
    uq_method_name = exp['uq_method_name']
    uq_method = exp['uq_method']

    # Get seed
    exp_id = f"{dataset_name}_{noise_model}_noise{int(noise_level*100):02d}"
    seed = get_experiment_seed(exp_id)

    try:
        # Create dataset
        dataset = dataset_config['class'](
            n_samples=100,
            noise_model=noise_model,
            noise_level=noise_level,
            seed=seed,
            **dataset_config['params']
        )
        data = dataset.generate()

        # Create and fit model
        model = NonlinearModel(
            model_func=dataset_config['model_func'],
            param_names=dataset_config['param_names'],
            initial_guess=dataset_config['initial_guess']
        )
        model.fit(data.X_train, data.y_train)

        # Compute intervals
        result = uq_method.compute_intervals(
            model, data.X_train, data.y_train, data.X_test, data.y_test
        )

        # Prepare data for plotting
        X_test_flat = data.X_test.flatten()
        y_test = data.y_test
        y_pred = result.y_pred
        y_lower = result.y_lower
        y_upper = result.y_upper

        # Sort for proper line plotting
        sort_idx = np.argsort(X_test_flat)
        X_sorted = X_test_flat[sort_idx]
        y_true_sorted = y_test[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_lower_sorted = y_lower[sort_idx]
        y_upper_sorted = y_upper[sort_idx]

        # Compute coverage
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

        # Create Plotly figure
        fig = go.Figure()

        # Gap region
        if data.X_gap is not None and len(data.X_gap) > 0:
            fig.add_vrect(
                x0=data.X_gap.min(), x1=data.X_gap.max(),
                fillcolor="orange", opacity=0.1,
                layer="below", line_width=0,
                annotation_text="Gap", annotation_position="top left"
            )

        # Extrapolation regions
        x_train_min = data.X_train.min()
        x_train_max = data.X_train.max()

        if x_train_min > 0:
            fig.add_vrect(
                x0=0, x1=x_train_min,
                fillcolor="red", opacity=0.08,
                layer="below", line_width=0,
                annotation_text="Extrap", annotation_position="top left"
            )

        if x_train_max < 1:
            fig.add_vrect(
                x0=x_train_max, x1=1.0,
                fillcolor="red", opacity=0.08,
                layer="below", line_width=0,
                annotation_text="Extrap", annotation_position="top right"
            )

        # Prediction interval (shaded band)
        fig.add_trace(go.Scatter(
            x=np.concatenate([X_sorted, X_sorted[::-1]]),
            y=np.concatenate([y_upper_sorted, y_lower_sorted[::-1]]),
            fill='toself',
            fillcolor='rgba(100, 100, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Prediction Interval',
            hoverinfo='skip'
        ))

        # Test data
        fig.add_trace(go.Scatter(
            x=X_test_flat,
            y=y_test,
            mode='markers',
            marker=dict(color='black', size=3, opacity=0.8),
            name='Test data',
            hovertemplate='Test: %{y:.4f}<extra></extra>'
        ))

        # Training data
        fig.add_trace(go.Scatter(
            x=data.X_train.flatten(),
            y=data.y_train,
            mode='markers',
            marker=dict(color='gray', size=8, opacity=0.5),
            name='Training data (noisy)',
            hovertemplate='Training: %{y:.4f}<extra></extra>'
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=X_sorted,
            y=y_pred_sorted,
            mode='lines',
            line=dict(color='blue', width=1.5),
            name='Prediction',
            hovertemplate='Prediction: %{y:.4f}<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=f"{dataset_name} | {noise_model.capitalize()} {int(noise_level*100)}% | Nlinfit<br>" +
                  f"<sub>Coverage: {coverage:.1%} (target: 95%)</sub>",
            xaxis_title="X",
            yaxis_title="y",
            xaxis=dict(range=[0, 1]),
            template='plotly_white',
            hovermode='closest',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )

        # Save as HTML
        noise_pct = int(noise_level * 100)
        filename = f"{dataset_name}_{noise_model}_noise{noise_pct:02d}_{uq_method_name}.html"
        output_path = FITS_DIR / filename

        fig.write_html(output_path, include_plotlyjs='cdn')

    except Exception as e:
        print(f"\nFailed to generate plot for {exp_id}: {str(e)}")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\nGenerated fit plots in: {FITS_DIR}")
