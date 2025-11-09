#!/usr/bin/env python
"""Generate interactive Plotly visualizations for GP fits - matching dashboard style."""

import sys
sys.path.insert(0, '.')
import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# Import datasets to regenerate data
from src.datasets.linear import LineDataset, PolynomialDataset
from src.datasets.nonlinear import (
    ExponentialDecayDataset, LogisticGrowthDataset,
    MichaelisMentenDataset, GaussianDataset
)
from src.utils.seeds import get_experiment_seed

# Directories
gp_results_dir = Path('results/gp_fits')
output_dir = Path('results/figures/gp_fits_html')
output_dir.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASET_CONFIGS = {
    'Line': {'class': LineDataset, 'params': {'slope': 0.8, 'intercept': 0.1}},
    'Quadratic': {'class': PolynomialDataset, 'params': {'coefficients': [0.5, 2.0, -1.0]}},
    'Cubic': {'class': PolynomialDataset, 'params': {'coefficients': [0.5, -1.0, 2.0, 0.5]}},
    'ExponentialDecay': {'class': ExponentialDecayDataset, 'params': {'a': 2.0, 'b': 3.0, 'c': 0.5}},
    'LogisticGrowth': {'class': LogisticGrowthDataset, 'params': {'L': 1.0, 'k': 10.0, 'x0': 0.5}},
    'MichaelisMenten': {'class': MichaelisMentenDataset, 'params': {'Vmax': 1.0, 'Km': 0.3}},
    'Gaussian': {'class': GaussianDataset, 'params': {'a': 1.0, 'mu': 0.5, 'sigma': 0.15}},
}

# Find all GP result files
result_files = sorted(gp_results_dir.glob('*_GP.json'))

print(f"Generating visualizations for {len(result_files)} GP fits (dashboard style)")

for result_file in result_files:
    # Load results
    with open(result_file, 'r') as f:
        results = json.load(f)

    dataset_name = results['dataset']
    noise_model = results['noise_model']
    noise_level = results['noise_level']
    coverage = results['coverage']
    rmse = results['rmse']
    mean_width = results['mean_width']
    r2 = results['r2']

    # Extract predictions
    preds = results['predictions']
    X_train = np.array(preds['X_train'])
    y_train = np.array(preds['y_train'])
    X_eval = np.array(preds['X_eval'])
    y_eval = np.array(preds['y_eval'])
    y_pred = np.array(preds['y_pred'])
    y_lower = np.array(preds['y_lower'])
    y_upper = np.array(preds['y_upper'])

    # Regenerate dataset to get gap and extrapolation regions
    exp_id = f"{dataset_name}_{noise_model}_noise{noise_level:02d}"
    seed = get_experiment_seed(exp_id)
    dataset_config = DATASET_CONFIGS[dataset_name]
    dataset = dataset_config['class'](
        n_samples=100,
        noise_model=noise_model,
        noise_level=noise_level / 100.0,
        seed=seed,
        **dataset_config['params']
    )
    data = dataset.generate()

    # Create dense grid for smooth prediction line
    X_plot = np.linspace(0, 1, 200)
    sort_idx = np.argsort(X_plot)
    X_plot_sorted = X_plot[sort_idx]

    # Sort eval data for plotting
    eval_sort_idx = np.argsort(X_eval)
    X_eval_sorted = X_eval[eval_sort_idx]
    y_eval_sorted = y_eval[eval_sort_idx]
    y_pred_sorted = y_pred[eval_sort_idx]
    y_lower_sorted = y_lower[eval_sort_idx]
    y_upper_sorted = y_upper[eval_sort_idx]

    # Create figure
    fig = go.Figure()

    # Prediction interval band (matching dashboard style)
    fig.add_trace(go.Scatter(
        x=X_eval_sorted,
        y=y_upper_sorted,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=X_eval_sorted,
        y=y_lower_sorted,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 138, 255, 0.3)',
        fill='tonexty',
        name='95% Prediction Interval',
        hovertemplate='<b>95% PI</b><br>x: %{x:.3f}<br>Lower: %{y:.3f}<extra></extra>'
    ))

    # Test data points (black circles - matching dashboard)
    fig.add_trace(go.Scatter(
        x=data.X_test.flatten(),
        y=data.y_test,
        mode='markers',
        marker=dict(size=5, color='black', symbol='circle'),
        name='Test Data',
        hovertemplate='<b>Test</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))

    # Training data points (lightgray open circles - matching dashboard)
    fig.add_trace(go.Scatter(
        x=X_train,
        y=y_train,
        mode='markers',
        marker=dict(size=6, color='lightgray', symbol='circle-open'),
        name='Training Data',
        hovertemplate='<b>Training</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))

    # Mean prediction line (blue - matching dashboard)
    fig.add_trace(go.Scatter(
        x=X_eval_sorted,
        y=y_pred_sorted,
        mode='lines',
        line=dict(color='blue', width=2),
        name='GP Prediction',
        hovertemplate='<b>Prediction</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))

    # Gap region (orange shading - matching dashboard)
    if data.X_gap is not None and len(data.X_gap) > 0:
        fig.add_vrect(
            x0=data.X_gap.min(), x1=data.X_gap.max(),
            fillcolor="orange", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Gap", annotation_position="top left"
        )

    # Extrapolation regions (red shading - matching dashboard)
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

    # Title format matching dashboard
    title = f"{dataset_name} - {noise_model.capitalize()} Noise ({noise_level}%)<br>"
    title += f"UQ Method: GP (RBF + White Noise) | Coverage: {coverage:.1%} | RMSE: {rmse:.4f} | R²: {r2:.4f}"

    # Update layout to match dashboard style
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    # Save to HTML
    output_file = output_dir / f'{dataset_name}_{noise_model}_noise{noise_level:02d}_GP.html'
    fig.write_html(str(output_file))

    print(f"  ✓ {dataset_name}_{noise_model}_noise{noise_level:02d}")

print(f"\n✓ Generated {len(result_files)} visualizations (dashboard style)")
print(f"✓ Saved to {output_dir}")
