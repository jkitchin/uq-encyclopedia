#!/usr/bin/env python
"""Add GP model description to the data-driven tab."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_gp_desc')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# GP Model Architecture description
gp_description = '''
        <!-- GP Model Architecture -->
        <div class="section">
            <h3>🔮 Gaussian Process Model Architecture</h3>
            <p><strong>Kernel:</strong> RBF (Radial Basis Function) + White Noise</p>
            <p><strong>Kernel Form:</strong> k(x, x') = σ²<sub>f</sub> · exp(-||x - x'||² / (2ℓ²)) + σ²<sub>n</sub>δ(x, x')</p>
            <p><strong>Hyperparameters:</strong></p>
            <ul style="margin-left: 2rem;">
                <li><strong>Length scale (ℓ):</strong> Controls correlation distance (optimized, bounds: 0.001-1.0)</li>
                <li><strong>Signal variance (σ²<sub>f</sub>):</strong> Output scale factor (optimized, bounds: 0.001-1000)</li>
                <li><strong>Noise variance (σ²<sub>n</sub>):</strong> Observation noise level (optimized, bounds: 1e-5 to 10)</li>
            </ul>
            <p><strong>Optimization:</strong> Maximum Likelihood Estimation with 10 random restarts</p>
            <p><strong>UQ Method:</strong> GP posterior variance (exact Bayesian inference)</p>
            <p><strong>Prediction Intervals:</strong> 95% credible intervals using 1.96 × posterior standard deviation</p>
            <p><strong>Note:</strong> GP provides probabilistic predictions with uncertainty quantification based on training data proximity</p>
        </div>
'''

# Find the DPOSE section and insert GP description after it
pattern = r'(<h3>🧠 DPOSE Model Architecture</h3>.*?</div>)\s*\n\s*(<!-- Interactive Results Table -->)'

replacement = r'\1\n' + gp_description + r'\n        \2'

html_new = re.sub(pattern, replacement, html, flags=re.DOTALL)

if html_new != html:
    print("✓ Added GP model description after DPOSE section")
    html = html_new
else:
    print("⚠️ Could not find DPOSE section pattern - trying alternative approach")

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
            # Insert GP description
            insertion_point = dd_start + last_closing_div + len('</div>')
            html = html[:insertion_point] + '\n' + gp_description + '\n' + html[insertion_point:]
            print("✓ Added GP model description (alternative method)")
        else:
            print("❌ Could not find insertion point")
    else:
        print("❌ Could not find Interactive Results Table")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print("✓ Added Gaussian Process model architecture description")
print("\nRefresh your browser to see the GP model description!")
