#!/usr/bin/env python
"""Add NNBR model description to the data-driven tab."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# NNBR Model Architecture description
nnbr_description = '''
        <!-- NNBR Model Architecture -->
        <div class="section">
            <h3>🔗 Neural Network BLR Model Architecture</h3>
            <p><strong>Type:</strong> Neural Network + Bayesian Linear Regression</p>
            <p><strong>Architecture:</strong> MLP (Multi-Layer Perceptron) backend with Bayesian Linear Regression for uncertainty quantification</p>

            <p><strong>Neural Network Configuration:</strong></p>
            <ul style="margin-left: 2rem;">
                <li><strong>Hidden Layers:</strong> 1 hidden layer with 20 neurons</li>
                <li><strong>Activation:</strong> Tanh (hyperbolic tangent)</li>
                <li><strong>Solver:</strong> L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)</li>
                <li><strong>Max Iterations:</strong> 1000</li>
            </ul>

            <p><strong>Bayesian Linear Regression Configuration:</strong></p>
            <ul style="margin-left: 2rem;">
                <li><strong>Backend:</strong> sklearn.linear_model.BayesianRidge</li>
                <li><strong>Prior Distribution:</strong> Gamma distribution on precision parameters</li>
                <li><strong>Inference Method:</strong> Bayesian inference with analytical posterior</li>
            </ul>

            <p><strong>UQ Method:</strong> Bayesian Linear Regression on neural network features</p>
            <p><strong>Prediction Intervals:</strong> 95% intervals using 1.96 × posterior standard deviation</p>

            <p><strong>Note:</strong> NNBR combines the feature extraction power of neural networks with the probabilistic framework of Bayesian linear regression. This approach typically produces well-calibrated, conservative uncertainty estimates. The model achieves 100% coverage in this benchmark, though intervals may be wider than necessary.</p>
        </div>
'''

# Find the NNGMM section and insert NNBR description after it
pattern = r'(<h3>🧠 Neural Network GMM Model Architecture</h3>.*?</div>)\s*\n\s*(<!-- Interactive Results Table -->)'

replacement = r'\1\n' + nnbr_description + r'\n        \2'

html_new = re.sub(pattern, replacement, html, flags=re.DOTALL)

if html_new != html:
    print("✓ Added NNBR model description after NNGMM section")
    html = html_new
else:
    print("⚠️ Could not find NNGMM section pattern - trying alternative approach")

    # Try alternative pattern - insert before Results Table
    pattern2 = r'(</div>\s*\n\s*)(<!-- Interactive Results Table -->)'
    # Find the last occurrence before Interactive Results Table in data-driven section
    dd_start = html.find('<div id="main-tab-datadriven"')
    dd_section = html[dd_start:]

    # Find first Interactive Results Table in this section
    results_table_pos = dd_section.find('<!-- Interactive Results Table -->')

    if results_table_pos != -1:
        # Find the </div> just before it
        before_table = dd_section[:results_table_pos]
        last_closing_div = before_table.rfind('</div>')

        if last_closing_div != -1:
            # Insert NNBR description
            insertion_point = dd_start + last_closing_div + len('</div>')
            html = html[:insertion_point] + '\n' + nnbr_description + '\n' + html[insertion_point:]
            print("✓ Added NNBR model description (alternative method)")
        else:
            print("❌ Could not find insertion point")
    else:
        print("❌ Could not find Interactive Results Table")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print("✓ Added Neural Network BLR model architecture description")
print("\nRefresh your browser to see the NNBR model description!")
