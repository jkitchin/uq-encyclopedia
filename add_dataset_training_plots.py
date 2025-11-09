#!/usr/bin/env python
"""Add training data plots for each dataset/noise combination to the dashboard."""

import sys
sys.path.insert(0, '.')
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import re

from src.datasets.linear import LineDataset, PolynomialDataset, LennardJonesDataset, ShomateDataset
from src.utils.seeds import get_experiment_seed
from config.global_config import DASHBOARD_DIR, NOISE_MODELS, NOISE_LEVELS

print("Generating training data plots for dataset tab...")

# Dataset configurations
DATASET_CONFIGS = {
    'Line': {
        'class': LineDataset,
        'params': {'slope': 0.8, 'intercept': 0.1},
        'description': 'Linear: y = 0.8x + 0.1'
    },
    'Polynomial': {
        'class': PolynomialDataset,
        'params': {'degree': 3},
        'description': 'Polynomial (degree 3)'
    },
    'LennardJones': {
        'class': LennardJonesDataset,
        'params': {'epsilon': 1.0, 'sigma': 1.0},
        'description': 'Lennard-Jones Potential'
    },
    'Shomate': {
        'class': ShomateDataset,
        'params': {},
        'description': 'Shomate Equation (Heat Capacity)'
    }
}

def create_training_plot(dataset_name, dataset_config, noise_model, noise_level):
    """Create a plot of training data for a specific configuration."""
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

    # Create Plotly figure
    fig = go.Figure()

    # Plot true function (clean, no noise)
    X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    y_clean = dataset._generate_clean(X_plot).flatten()
    fig.add_trace(go.Scatter(
        x=X_plot.flatten(),
        y=y_clean,
        mode='lines',
        line=dict(color='black', width=2),
        opacity=0.3,
        name='True function',
        hovertemplate='True: %{y:.4f}<extra></extra>'
    ))

    # Plot training data
    fig.add_trace(go.Scatter(
        x=data.X_train.flatten(),
        y=data.y_train,
        mode='markers',
        marker=dict(color='steelblue', size=8, opacity=0.6),
        name='Training data',
        hovertemplate='Training: %{y:.4f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f"{dataset_config['description']}<br><sub>{noise_model.capitalize()} {int(noise_level*100)}% noise</sub>",
        xaxis_title="X",
        yaxis_title="y",
        xaxis=dict(range=[0, 1]),
        template='plotly_white',
        hovermode='closest',
        height=350,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=50, r=20, t=60, b=50)
    )

    # Return HTML - use 'cdn' to include plotlyjs separately for each plot
    # This ensures binary-encoded data works properly
    plot_html = fig.to_html(include_plotlyjs='cdn', div_id=f'plot_{dataset_name}_{noise_model}_{int(noise_level*100)}', full_html=False)
    return plot_html

# Generate all plots
print("\nGenerating plots...")
plots_by_dataset = {}

for dataset_name, dataset_config in DATASET_CONFIGS.items():
    print(f"  {dataset_name}...")
    plots_by_dataset[dataset_name] = {}

    for noise_model in NOISE_MODELS:
        plots_by_dataset[dataset_name][noise_model] = {}

        for noise_level in NOISE_LEVELS:
            plot_html = create_training_plot(dataset_name, dataset_config, noise_model, noise_level)
            plots_by_dataset[dataset_name][noise_model][noise_level] = plot_html

print("\nGenerating HTML for dataset tab...")

# Create HTML sections for each dataset
dataset_html_sections = []

for dataset_name, dataset_config in DATASET_CONFIGS.items():
    # Create grid of plots for this dataset
    plots_html = f"""
                <div class="dataset-card">
                    <h4>{dataset_name}: {dataset_config['description']}</h4>

                    <h5>Homoskedastic Noise (constant variance)</h5>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin: 1rem 0;">
"""

    for noise_level in NOISE_LEVELS:
        plot_html = plots_by_dataset[dataset_name]['homoskedastic'][noise_level]
        plots_html += f"""
                        <div style="border: 1px solid #ddd; border-radius: 4px; padding: 0.5rem;">
                            {plot_html}
                        </div>
"""

    plots_html += """
                    </div>

                    <h5>Heteroskedastic Noise (varying variance)</h5>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin: 1rem 0;">
"""

    for noise_level in NOISE_LEVELS:
        plot_html = plots_by_dataset[dataset_name]['heteroskedastic'][noise_level]
        plots_html += f"""
                        <div style="border: 1px solid #ddd; border-radius: 4px; padding: 0.5rem;">
                            {plot_html}
                        </div>
"""

    plots_html += """
                    </div>
                </div>
"""

    dataset_html_sections.append(plots_html)

# Combine all sections
all_plots_html = '\n'.join(dataset_html_sections)

# Read current dashboard
dashboard_path = DASHBOARD_DIR / 'dashboard.html'
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Find the Datasets tab and insert plots after the existing content
# Look for the marker where we should insert
marker = '<!-- Methods Tab -->'

if marker in html_content:
    # Split at the marker
    parts = html_content.split(marker, 1)

    # Insert the plots before the marker
    plots_section = '\n                <h3>Training Data Visualizations</h3>\n' + all_plots_html + '\n            '
    html_content = parts[0] + plots_section + marker + parts[1]
else:
    print("Warning: Could not find insertion marker")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✓ Training data plots added to: {dashboard_path}")
print(f"\nAdded {len(DATASET_CONFIGS)} datasets × {len(NOISE_MODELS)} noise models × {len(NOISE_LEVELS)} noise levels = {len(DATASET_CONFIGS) * len(NOISE_MODELS) * len(NOISE_LEVELS)} plots")
print("\nRefresh your browser to see the training data visualizations!")
