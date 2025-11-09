#!/usr/bin/env python
"""Generate interactive Plotly fit plots for all 64 experiments."""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from src.datasets.linear import LineDataset, PolynomialDataset, LennardJonesDataset, ShomateDataset
from src.models.linear_models import OLSRegression, LennardJonesRegression, ShomateRegression
from src.uq_methods.linear_uq import HatMatrixUQ, BayesianLinearRegressionUQ, ConformalPredictionUQ
from src.utils.seeds import get_experiment_seed
from config.global_config import RESULTS_FIGURES_DIR, NOISE_MODELS, NOISE_LEVELS

print("=" * 70)
print("GENERATING INTERACTIVE PLOTLY FIT PLOTS")
print("=" * 70)

# Create subdirectory for individual fits
FITS_DIR = RESULTS_FIGURES_DIR / 'individual_fits_html'
FITS_DIR.mkdir(parents=True, exist_ok=True)

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
        'params': {}
    }
}

# UQ method configurations
UQ_METHODS = {
    'HatMatrix': HatMatrixUQ,
    'Bayesian': BayesianLinearRegressionUQ,
    'Conformal': ConformalPredictionUQ
}

# Generate all combinations
experiments = []
for dataset_name, dataset_config in DATASET_CONFIGS.items():
    for noise_model in NOISE_MODELS:
        for noise_level in NOISE_LEVELS:
            for uq_method_name, uq_method_class in UQ_METHODS.items():
                experiments.append({
                    'dataset_name': dataset_name,
                    'dataset_config': dataset_config,
                    'noise_model': noise_model,
                    'noise_level': noise_level,
                    'uq_method_name': uq_method_name,
                    'uq_method_class': uq_method_class
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
    uq_method_class = exp['uq_method_class']

    # Get seed
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

    # Create and fit model
    model = dataset_config['model_class'](**dataset_config['model_kwargs'])
    model.fit(data.X_train, data.y_train)

    # Create UQ method
    if uq_method_name == 'Conformal':
        uq_method = uq_method_class(confidence_level=0.95, method='plus', cv=5)
    else:
        uq_method = uq_method_class(confidence_level=0.95)

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

    # Plot shaded regions first (background)
    # Gap region
    if data.X_gap is not None and len(data.X_gap) > 0:
        fig.add_vrect(
            x0=data.X_gap.min(), x1=data.X_gap.max(),
            fillcolor="orange", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Gap", annotation_position="top left"
        )

    # Extrapolation regions (only if they exist within 0-1 range)
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

    # Test data across entire range (black dots - actual noisy test data)
    fig.add_trace(go.Scatter(
        x=X_test_flat,
        y=y_test,
        mode='markers',
        marker=dict(color='black', size=3, opacity=0.8),
        name='Test data',
        hovertemplate='Test: %{y:.4f}<extra></extra>'
    ))

    # Training data (transparent grey circles - includes noise)
    fig.add_trace(go.Scatter(
        x=data.X_train.flatten(),
        y=data.y_train,
        mode='markers',
        marker=dict(color='gray', size=8, opacity=0.5),
        name='Training data (noisy)',
        hovertemplate='Training: %{y:.4f}<extra></extra>'
    ))

    # Predictions (thin solid blue line)
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
        title=f"{dataset_name} | {noise_model.capitalize()} {int(noise_level*100)}% | {uq_method_name}<br>" +
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

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\nGenerated {len(experiments)} interactive fit plots")
print(f"Location: {FITS_DIR}")
print(f"\nExample files:")
for i, exp in enumerate(experiments[:3]):
    noise_pct = int(exp['noise_level'] * 100)
    filename = f"{exp['dataset_name']}_{exp['noise_model']}_noise{noise_pct:02d}_{exp['uq_method_name']}.html"
    print(f"  - {filename}")
print(f"  ... and {len(experiments)-3} more")
