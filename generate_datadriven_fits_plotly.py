#!/usr/bin/env python
"""Generate interactive Plotly fit plots for data-driven models."""

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm

# Import datasets
from src.datasets.linear import LineDataset, PolynomialDataset
from src.datasets.nonlinear import (
    ExponentialDecayDataset, LogisticGrowthDataset,
    MichaelisMentenDataset, GaussianDataset
)

# Import models and UQ
from src.models.data_driven_models import DPOSEModel
from src.uq_methods.data_driven_uq import EnsembleUQ, CalibratedEnsembleUQ
from src.utils.seeds import get_experiment_seed
from config.global_config import RESULTS_FIGURES_DIR

print("=" * 70)
print("GENERATING DATA-DRIVEN MODEL FIT PLOTS")
print("=" * 70)

# Load results to get list of experiments
results_csv = Path('results/csv/datadriven_models_comprehensive.csv')
df = pd.read_csv(results_csv)

# Output directory
output_dir = RESULTS_FIGURES_DIR / 'datadriven_fits_html'
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nGenerating {len(df)} individual fit plots...")
print(f"Saving to: {output_dir}\n")

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

# Model config
MODEL_CONFIG = {
    'n_hidden': 20,
    'n_ensemble': 32,
    'activation': 'tanh',
    'seed': 42,
    'loss_type': 'crps',
    'optimizer': 'bfgs',
}

# UQ methods
UQ_METHODS = {
    'Ensemble': EnsembleUQ(confidence_level=0.95),
    'EnsembleCalibrated': CalibratedEnsembleUQ(confidence_level=0.95),
}

# Generate plots
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating plots"):
    dataset_name = row['dataset']
    noise_model = row['noise_model']
    noise_level = row['noise_level']
    uq_method_name = row['uq_method']
    exp_id = row['exp_id']

    # Create dataset
    seed = get_experiment_seed(exp_id)
    dataset_config = DATASET_CONFIGS[dataset_name]
    dataset = dataset_config['class'](
        n_samples=100,
        noise_model=noise_model,
        noise_level=noise_level,
        seed=seed,
        **dataset_config['params']
    )
    data = dataset.generate()

    # Fit model
    model = DPOSEModel(**MODEL_CONFIG)
    model.fit(data.X_train, data.y_train)

    # Get UQ intervals
    uq_method = UQ_METHODS[uq_method_name]
    uq_result = uq_method.compute_intervals(
        model, data.X_train, data.y_train, data.X_test, data.y_test
    )

    # Create dense grid for smooth visualization
    X_plot = np.linspace(data.X_test.min(), data.X_test.max(), 200).reshape(-1, 1)
    uq_plot = uq_method.compute_intervals(
        model, data.X_train, data.y_train, X_plot
    )

    # Create figure
    fig = go.Figure()

    # Prediction interval band
    fig.add_trace(go.Scatter(
        x=X_plot.flatten(),
        y=uq_plot.y_upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=X_plot.flatten(),
        y=uq_plot.y_lower,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 138, 255, 0.3)',
        fill='tonexty',
        name='95% Prediction Interval',
        hovertemplate='<b>95% PI</b><br>x: %{x:.3f}<br>Lower: %{y:.3f}<extra></extra>'
    ))

    # Test data points
    fig.add_trace(go.Scatter(
        x=data.X_test.flatten(),
        y=data.y_test,
        mode='markers',
        marker=dict(size=5, color='black', symbol='circle'),
        name='Test Data',
        hovertemplate='<b>Test</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))

    # Training data points
    fig.add_trace(go.Scatter(
        x=data.X_train.flatten(),
        y=data.y_train,
        mode='markers',
        marker=dict(size=6, color='lightgray', symbol='circle-open'),
        name='Training Data',
        hovertemplate='<b>Training</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))

    # Mean prediction line
    fig.add_trace(go.Scatter(
        x=X_plot.flatten(),
        y=uq_plot.y_pred,
        mode='lines',
        line=dict(color='blue', width=2),
        name='DPOSE Prediction',
        hovertemplate='<b>Prediction</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))

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

    # Update layout
    title = f"{dataset_name} - {noise_model.capitalize()} Noise ({int(noise_level*100)}%)<br>"
    title += f"UQ Method: {uq_method_name} | Coverage: {row['coverage']:.1%} | RMSE: {row['rmse']:.4f} | R²: {row['r2']:.4f}"

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

    # Save plot
    output_file = output_dir / f"{exp_id}_{uq_method_name}.html"
    fig.write_html(output_file)

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"\nGenerated fit plots in: {output_dir}")
