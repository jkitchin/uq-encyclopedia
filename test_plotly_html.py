#!/usr/bin/env python
"""Test Plotly HTML generation."""

import sys
sys.path.insert(0, '.')
import numpy as np
import plotly.graph_objects as go

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

# Create figure
fig = go.Figure()

# True function
X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
y_clean = dataset._generate_clean(X_plot).flatten()

print("True function points:", len(X_plot))
print("y_clean min/max:", y_clean.min(), y_clean.max())

fig.add_trace(go.Scatter(
    x=X_plot.flatten(),
    y=y_clean,
    mode='lines',
    line=dict(color='black', width=2),
    opacity=0.3,
    name='True function'
))

# Training data
print("\nTraining data points:", len(data.X_train))
fig.add_trace(go.Scatter(
    x=data.X_train.flatten(),
    y=data.y_train,
    mode='markers',
    marker=dict(color='steelblue', size=8, opacity=0.6),
    name='Training data'
))

fig.update_layout(
    title="Test Plot",
    xaxis_title="X",
    yaxis_title="y",
    xaxis=dict(range=[0, 1]),
    template='plotly_white',
    height=350
)

# Generate HTML
html = fig.to_html(include_plotlyjs=False, div_id='test_plot', full_html=False)

print("\nHTML length:", len(html))
print("\nFirst 500 chars of HTML:")
print(html[:500])

# Check if data is in HTML
if '"x":[' in html and '"y":[' in html:
    print("\n✓ Data arrays found in HTML")
else:
    print("\n✗ Data arrays NOT found in HTML")

# Save to test file
with open('/tmp/test_plot.html', 'w') as f:
    f.write('<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>')
    f.write(html)

print("\nTest file saved to: /tmp/test_plot.html")
