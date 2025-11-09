#!/usr/bin/env python
"""Add NNGMM model description to the data-driven tab."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_nngmm_desc')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# NNGMM Model Architecture description
nngmm_description = '''
        <!-- NNGMM Model Architecture -->
        <div class="section">
            <h3>🧠 Neural Network GMM Model Architecture</h3>
            <p><strong>Type:</strong> Neural Network + Gaussian Mixture Model</p>
            <p><strong>Architecture:</strong> MLP (Multi-Layer Perceptron) backend with GMM-based uncertainty quantification</p>

            <p><strong>Neural Network Configuration:</strong></p>
            <ul style="margin-left: 2rem;">
                <li><strong>Hidden Layers:</strong> 1 hidden layer with 20 neurons</li>
                <li><strong>Activation:</strong> Tanh (hyperbolic tangent)</li>
                <li><strong>Solver:</strong> L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)</li>
                <li><strong>Max Iterations:</strong> 1000</li>
            </ul>

            <p><strong>GMM Configuration:</strong></p>
            <ul style="margin-left: 2rem;">
                <li><strong>Components:</strong> 3 Gaussian mixture components</li>
                <li><strong>Uncertainty Samples:</strong> 500 samples for uncertainty estimation</li>
            </ul>

            <p><strong>UQ Method:</strong> Gaussian Mixture Model over network outputs</p>
            <p><strong>Prediction Intervals:</strong> 95% intervals using 1.96 × GMM-derived standard deviation</p>

            <p><strong>Note:</strong> NNGMM combines the flexibility of neural networks with probabilistic uncertainty quantification through Gaussian mixture modeling. The model may require hyperparameter tuning for optimal performance on specific datasets.</p>
        </div>
'''

# Find the GP section and insert NNGMM description after it
pattern = r'(<h3>🔮 Gaussian Process Model Architecture</h3>.*?</div>)\s*\n\s*(<!-- Interactive Results Table -->)'

replacement = r'\1\n' + nngmm_description + r'\n        \2'

html_new = re.sub(pattern, replacement, html, flags=re.DOTALL)

if html_new != html:
    print("✓ Added NNGMM model description after GP section")
    html = html_new
else:
    print("⚠️ Could not find GP section pattern - trying alternative approach")

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
            # Insert NNGMM description
            insertion_point = dd_start + last_closing_div + len('</div>')
            html = html[:insertion_point] + '\n' + nngmm_description + '\n' + html[insertion_point:]
            print("✓ Added NNGMM model description (alternative method)")
        else:
            print("❌ Could not find insertion point")
    else:
        print("❌ Could not find Interactive Results Table")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print("✓ Added Neural Network GMM model architecture description")
print("\nRefresh your browser to see the NNGMM model description!")
