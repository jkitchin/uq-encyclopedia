#!/usr/bin/env python
"""Fix nonlinear and data-driven tab structure - add missing sections and reorder."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Backup
backup_path = dashboard_path.with_suffix('.html.bak_structure')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# ============================================================================
# FIX 1: Add Dataset Explanations to nonlinear tab
# ============================================================================

datasets_explanation = '''
            <!-- Dataset Explanations -->
            <div class="section">
                <h3>📁 Datasets Explained</h3>

                <div class="dataset-card">
                    <h4>1. Exponential Decay</h4>
                    <p><strong>Function:</strong> y = a·exp(-bx) + c</p>
                    <p><strong>Parameters:</strong> a=2.0, b=3.0, c=0.5</p>
                    <p><strong>Type:</strong> First-order decay process</p>
                    <p><strong>Complexity:</strong> ⭐⭐ (Moderate)</p>
                    <p><strong>Applications:</strong> Radioactive decay, chemical kinetics, RC circuits</p>
                    <p><strong>Why test this?</strong> Common in science/engineering. Tests UQ methods on rapid initial decay.</p>
                    <p><strong>Expected Coverage:</strong> ~92% (Delta Method), ~86% (Conformal)</p>
                </div>

                <div class="dataset-card">
                    <h4>2. Logistic Growth</h4>
                    <p><strong>Function:</strong> y = L / (1 + exp(-k(x - x₀)))</p>
                    <p><strong>Parameters:</strong> L=1.0 (carrying capacity), k=10.0 (growth rate), x₀=0.5 (midpoint)</p>
                    <p><strong>Type:</strong> S-shaped (sigmoid) growth curve</p>
                    <p><strong>Complexity:</strong> ⭐⭐⭐ (Moderate-High)</p>
                    <p><strong>Applications:</strong> Population dynamics, viral spread, machine learning (sigmoid activation)</p>
                    <p><strong>Why test this?</strong> Non-monotonic curvature (inflection point). Tests UQ in flat + steep regions.</p>
                    <p><strong>Expected Coverage:</strong> ~94% (Delta Method), ~93% (Conformal)</p>
                </div>

                <div class="dataset-card">
                    <h4>3. Michaelis-Menten</h4>
                    <p><strong>Function:</strong> y = Vmax·x / (Km + x)</p>
                    <p><strong>Parameters:</strong> Vmax=1.0 (max rate), Km=0.3 (half-max concentration)</p>
                    <p><strong>Type:</strong> Enzyme kinetics / saturation curve</p>
                    <p><strong>Complexity:</strong> ⭐⭐ (Moderate)</p>
                    <p><strong>Applications:</strong> Biochemistry (enzyme kinetics), pharmacology (drug response)</p>
                    <p><strong>Why test this?</strong> Rapid saturation behavior. Tests UQ near asymptote.</p>
                    <p><strong>Expected Coverage:</strong> ~89% (Delta Method), ~86% (Conformal)</p>
                </div>

                <div class="dataset-card">
                    <h4>4. Gaussian</h4>
                    <p><strong>Function:</strong> y = a·exp(-(x-μ)²/(2σ²))</p>
                    <p><strong>Parameters:</strong> a=1.0 (amplitude), μ=0.5 (mean), σ=0.15 (std dev)</p>
                    <p><strong>Type:</strong> Bell curve / normal distribution</p>
                    <p><strong>Complexity:</strong> ⭐⭐⭐⭐ (Challenging)</p>
                    <p><strong>Applications:</strong> Spectroscopy (peak fitting), statistics, physics (wavepackets)</p>
                    <p><strong>Why test this?</strong> Sharp peak + rapid decay. Tests UQ on high curvature regions.</p>
                    <p><strong>Challenge:</strong> Conformal struggles with sharp features (40-70% coverage)</p>
                    <p><strong>Expected Coverage:</strong> ~83% (Delta Method), ~64% (Conformal)</p>
                </div>

                <div class="info-box">
                    <h4>Common Features</h4>
                    <ul>
                        <li>All scaled to domain [0, 1] for fair comparison</li>
                        <li>All output roughly in range [0, 1]</li>
                        <li>Same train/test split strategy with gap region</li>
                        <li>100 total samples per dataset</li>
                        <li>Tested with both homoskedastic and heteroskedastic noise</li>
                    </ul>
                </div>
            </div>
'''

# ============================================================================
# FIX 2: Add UQ Methods Explained to nonlinear tab
# ============================================================================

methods_explanation = '''
            <!-- UQ Methods Explanation -->
            <div class="section">
                <h3>🎯 UQ Methods Explained</h3>

                <div class="method-card">
                    <h4>1. Delta Method (Jacobian-based)</h4>
                    <p><strong>Type:</strong> Frequentist / Classical Statistics</p>
                    <p><strong>Average Coverage:</strong> 89.0%</p>

                    <h5>How it works:</h5>
                    <ol>
                        <li>Fit nonlinear model using nonlinear least squares</li>
                        <li>Compute the Jacobian matrix J at the fitted parameters</li>
                        <li>Estimate parameter covariance: Σ = σ²(J^T J)^(-1)</li>
                        <li>Propagate uncertainty to predictions using first-order Taylor expansion</li>
                        <li>Prediction intervals: ŷ ± t_{α/2,df} × σ_pred</li>
                    </ol>

                    <h5>Assumptions:</h5>
                    <ul>
                        <li>Errors are normally distributed</li>
                        <li>Model is locally linear (first-order approximation valid)</li>
                        <li>Constant variance (tested on both homo/heteroskedastic)</li>
                    </ul>

                    <h5>Strengths:</h5>
                    <ul class="pros">
                        <li>Fast computation</li>
                        <li>Well-established statistical theory</li>
                        <li>Reliable coverage (89.0% avg)</li>
                        <li>Works well for most nonlinear models</li>
                        <li>Provides parameter correlations</li>
                    </ul>

                    <h5>Weaknesses:</h5>
                    <ul class="cons">
                        <li>Assumes local linearity (may fail for highly nonlinear models)</li>
                        <li>Less accurate for models with strong curvature</li>
                        <li>Assumes normality of errors</li>
                        <li>Can struggle with heteroskedastic noise</li>
                    </ul>
                </div>

                <div class="method-card">
                    <h4>2. Conformal Prediction</h4>
                    <p><strong>Type:</strong> Distribution-free / Modern ML</p>
                    <p><strong>Average Coverage:</strong> 82.4%</p>

                    <h5>How it works:</h5>
                    <ol>
                        <li>Fit nonlinear model on training data</li>
                        <li>Compute residuals (non-conformity scores) on calibration set</li>
                        <li>Sort residuals and find quantile for desired coverage</li>
                        <li>Add/subtract this quantile to create prediction intervals</li>
                    </ol>

                    <h5>What makes it "Conformal":</h5>
                    <ul>
                        <li>Distribution-free: No assumptions about error distribution</li>
                        <li>Finite-sample guarantees: Coverage guaranteed for any sample size</li>
                        <li>Model-agnostic: Works with any prediction model</li>
                    </ul>

                    <h5>Strengths:</h5>
                    <ul class="pros">
                        <li>No distributional assumptions</li>
                        <li>Theoretically guaranteed coverage</li>
                        <li>Simple and intuitive</li>
                        <li>Works with any model</li>
                    </ul>

                    <h5>Weaknesses:</h5>
                    <ul class="cons">
                        <li>Lower average coverage (82.4%)</li>
                        <li>More variable across datasets (17.0% std dev)</li>
                        <li>Requires separate calibration data</li>
                        <li>Intervals have constant width (doesn't adapt locally)</li>
                        <li>Struggles with Gaussian peaks and low heteroskedastic noise</li>
                    </ul>
                </div>

                <div class="info-box warning">
                    <h4>⚠️ Key Finding</h4>
                    <p>Delta Method provides more reliable coverage across nonlinear models (89.0% avg vs 82.4%). Conformal prediction's constant-width intervals struggle with heteroskedastic noise and sharp features like Gaussian peaks.</p>
                </div>
            </div>
'''

# Find where to insert in nonlinear tab - right before "Detailed Results"
nl_detailed_results_pattern = r'(<div id="main-tab-nonlinear".*?)(            <h2>🔍 Detailed Results)'

replacement = r'\1' + datasets_explanation + '\n' + methods_explanation + '\n' + r'\2'

html, n_subs = re.subn(nl_detailed_results_pattern, replacement, html, flags=re.DOTALL)

if n_subs > 0:
    print("✓ Added Dataset and UQ Methods explanations to nonlinear tab")
else:
    print("⚠ Could not find insertion point in nonlinear tab")

# ============================================================================
# FIX 3: Filter data-driven table to only show EnsembleCalibrated rows
# ============================================================================

# Find the data-driven table and filter rows
dd_table_pattern = r'(<table id="resultsTableDD">.*?<tbody>)(.*?)(</tbody>.*?</table>)'

dd_match = re.search(dd_table_pattern, html, re.DOTALL)

if dd_match:
    table_start = dd_match.group(1)
    tbody_content = dd_match.group(2)
    table_end = dd_match.group(3)

    # Find all table rows
    rows = re.findall(r'<tr[^>]*>.*?</tr>', tbody_content, re.DOTALL)

    # Filter rows - keep only EnsembleCalibrated
    filtered_rows = []
    for row in rows:
        # Skip rows with just "Ensemble" (not calibrated)
        if '>Ensemble<' in row and 'Calibrated' not in row:
            continue
        # Keep all other rows
        filtered_rows.append(row)

    # Reconstruct table
    new_tbody = '\n                        '.join(filtered_rows)
    new_table = table_start + '\n                        ' + new_tbody + '\n                    ' + table_end

    # Replace in HTML
    html = html[:dd_match.start()] + new_table + html[dd_match.end():]

    print(f"✓ Filtered data-driven table: {len(rows)} → {len(filtered_rows)} rows (EnsembleCalibrated only)")
else:
    print("⚠ Could not find data-driven table")

# ============================================================================
# Save
# ============================================================================

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print("\nChanges made:")
print("  1. Added Dataset Explanations section to nonlinear tab")
print("  2. Added UQ Methods Explained section to nonlinear tab")
print("  3. Filtered data-driven table to show only EnsembleCalibrated rows")
print("\nRefresh your browser!")
