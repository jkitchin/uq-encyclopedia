#!/usr/bin/env python
"""Add dataset visualization plots to the guide section."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import base64
from io import BytesIO
sys.path.insert(0, '.')

from src.datasets.linear import LineDataset, PolynomialDataset, LennardJonesDataset, ShomateDataset
from config.global_config import DASHBOARD_DIR

print("Generating dataset plots...")

# Configuration
datasets_config = {
    'Line': {
        'class': LineDataset,
        'params': {'slope': 0.8, 'intercept': 0.1, 'n_samples': 100, 'noise_model': 'homoskedastic', 'noise_level': 0.02, 'seed': 42}
    },
    'Polynomial': {
        'class': PolynomialDataset,
        'params': {'degree': 3, 'n_samples': 100, 'noise_model': 'homoskedastic', 'noise_level': 0.02, 'seed': 42}
    },
    'LennardJones': {
        'class': LennardJonesDataset,
        'params': {'epsilon': 1.0, 'sigma': 1.0, 'n_samples': 100, 'noise_model': 'homoskedastic', 'noise_level': 0.02, 'seed': 42}
    },
    'Shomate': {
        'class': ShomateDataset,
        'params': {'T_min': 298.0, 'T_max': 1000.0, 'n_samples': 100, 'noise_model': 'homoskedastic', 'noise_level': 0.02, 'seed': 42}
    }
}

# Generate plots as base64
dataset_plots = {}

for name, config in datasets_config.items():
    print(f"  - {name}")

    # Create dataset
    dataset = config['class'](**config['params'])
    data = dataset.generate()

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot training data
    ax.scatter(data.X_train.flatten(), data.y_train, alpha=0.6, s=40,
              label='Training data', color='#2196F3', zorder=2)

    # Plot true function (dense) - extend to show extrapolation
    X_plot = np.linspace(-0.25, 1.25, 750).reshape(-1, 1)
    y_true = dataset._generate_clean(X_plot)
    ax.plot(X_plot, y_true, 'k-', linewidth=2.5, label='True function', zorder=3)

    # Highlight gap region
    if data.X_gap is not None and len(data.X_gap) > 0:
        ax.axvspan(data.X_gap.min(), data.X_gap.max(), alpha=0.15, color='orange',
                  zorder=1, label='Gap (interpolation)')

    # Highlight TRUE extrapolation regions (outside actual training data)
    x_train_min = data.X_train.min()
    x_train_max = data.X_train.max()

    ax.axvspan(-0.25, x_train_min, alpha=0.08, color='red', zorder=0, label='Extrapolation')
    ax.axvspan(x_train_max, 1.25, alpha=0.08, color='red', zorder=0)

    # Add extrap labels
    ax.text((-0.25 + x_train_min)/2, ax.get_ylim()[1]*0.95, 'Extrap',
            ha='center', fontsize=9, color='red', alpha=0.7)
    ax.text((x_train_max + 1.25)/2, ax.get_ylim()[1]*0.95, 'Extrap',
            ha='center', fontsize=9, color='red', alpha=0.7)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'{name} Dataset', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Expand x-axis to show extrapolation better
    ax.set_xlim(-0.25, 1.25)
    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    dataset_plots[name] = img_base64

print("\nUpdating dashboard...")

# Read the current dashboard
dashboard_path = DASHBOARD_DIR / 'dashboard.html'
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Create the dataset plots HTML
datasets_html = ""
for name, img_base64 in dataset_plots.items():
    datasets_html += f'''
                    <div class="dataset-plot">
                        <img src="data:image/png;base64,{img_base64}" alt="{name} dataset" style="width: 100%; max-width: 800px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    </div>
'''

# Find and replace the datasets section
# Look for the datasets tab content
import re
pattern = r'(<div id="guide-datasets" class="guide-content">.*?)(</div>\s*<!-- UQ Methods Tab -->)'
replacement = r'\1' + datasets_html + r'\2'

html_content = re.sub(pattern, replacement, html_content, flags=re.DOTALL)

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ Dataset plots added to: {dashboard_path}")
print("\nThe datasets tab now includes:")
for name in dataset_plots.keys():
    print(f"  - {name} visualization")
print("\nRefresh your browser to see the plots!")
