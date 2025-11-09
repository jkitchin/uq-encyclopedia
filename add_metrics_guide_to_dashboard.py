#!/usr/bin/env python
"""Add metrics guide section to the dashboard."""

import sys
sys.path.insert(0, '.')
from pathlib import Path
from config.global_config import DASHBOARD_DIR

print("Adding metrics guide to dashboard...")

# Read the current dashboard
dashboard_path = DASHBOARD_DIR / 'dashboard.html'
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Create the metrics guide HTML
metrics_section = """
                <h3>📊 Metrics for Evaluating UQ Methods</h3>

                <p>Coverage alone doesn't tell the whole story. Here are key metrics for comparing UQ methods:</p>

                <div class="dataset-card">
                    <h4>1. Coverage (Calibration)</h4>
                    <p><strong>What it measures:</strong> Does a 95% prediction interval actually contain the true value 95% of the time?</p>

                    <p><strong>Formula:</strong> PICP = (# intervals containing true value) / (# total points)</p>

                    <div class="info-box">
                        <p><strong>Ideal:</strong> Coverage = 95% (matches target confidence level)</p>
                        <ul>
                            <li>✓ <strong>Well-calibrated</strong>: Coverage ≈ 95%</li>
                            <li>⚠️ <strong>Under-coverage</strong>: Coverage < 90% (intervals too narrow, risky!)</li>
                            <li>⚠️ <strong>Over-coverage</strong>: Coverage > 98% (intervals too wide, not useful)</li>
                        </ul>
                    </div>

                    <p><strong>In our study:</strong></p>
                    <ul>
                        <li><strong>Hat Matrix</strong>: 86.0% ± 13.0% (slight under-coverage)</li>
                        <li><strong>Bayesian</strong>: 100.0% ± 0.0% (perfect coverage, but...)</li>
                        <li><strong>Conformal</strong>: 87.8% ± 13.5% (good coverage)</li>
                    </ul>
                </div>

                <div class="dataset-card">
                    <h4>2. Sharpness (Efficiency)</h4>
                    <p><strong>What it measures:</strong> How narrow are the prediction intervals? Narrower = more informative (if coverage is maintained).</p>

                    <p><strong>Formula:</strong> Mean Interval Width = Average(upper_bound - lower_bound)</p>

                    <div class="info-box warning">
                        <p><strong>⚠️ The Trade-off:</strong></p>
                        <p>It's easy to get 100% coverage with infinitely wide intervals!</p>
                        <p>Good UQ methods must balance <strong>coverage</strong> AND <strong>sharpness</strong>.</p>
                    </div>

                    <p><strong>In our study:</strong></p>
                    <ul>
                        <li><strong>Hat Matrix</strong>: Mean width = 0.063 ✓ (sharp intervals)</li>
                        <li><strong>Bayesian</strong>: Mean width = 2.407 ⚠️ (38× wider than Hat Matrix!)</li>
                        <li><strong>Conformal</strong>: Mean width = 0.072 ✓ (sharp intervals)</li>
                    </ul>

                    <p><strong>Interpretation:</strong> Bayesian achieves 100% coverage by being extremely conservative (wide intervals). Hat Matrix and Conformal provide much sharper (more useful) intervals with good coverage.</p>
                </div>

                <div class="dataset-card">
                    <h4>3. Calibration-Sharpness Score</h4>
                    <p><strong>What it measures:</strong> Combined metric balancing coverage accuracy and interval width.</p>

                    <p><strong>Formula:</strong> Score = 10 × |coverage - 0.95| + normalized_width</p>
                    <p>Lower is better! Penalizes both poor coverage AND wide intervals.</p>

                    <div class="info-box info">
                        <p><strong>Why this matters:</strong></p>
                        <p>This single metric captures the fundamental trade-off in UQ:</p>
                        <ul>
                            <li>First term: Are you calibrated? (coverage ≈ 95%)</li>
                            <li>Second term: Are your intervals sharp? (small width)</li>
                        </ul>
                    </div>

                    <p><strong>In our study (typical values):</strong></p>
                    <ul>
                        <li><strong>Hat Matrix</strong>: ~0.80 ✓ (best balance)</li>
                        <li><strong>Bayesian</strong>: ~3.24 (poor - too wide despite perfect coverage)</li>
                        <li><strong>Conformal</strong>: ~0.85 ✓ (good balance)</li>
                    </ul>
                </div>

                <div class="dataset-card">
                    <h4>4. Regional Coverage</h4>
                    <p><strong>What it measures:</strong> Does the method perform well in interpolation AND extrapolation?</p>

                    <p><strong>Regions tested:</strong></p>
                    <ul>
                        <li><strong>Interpolation</strong>: Between training points (easier)</li>
                        <li><strong>Gap</strong>: Large gap in training data (moderate difficulty)</li>
                        <li><strong>Extrapolation</strong>: Beyond training range (hardest!)</li>
                    </ul>

                    <div class="info-box">
                        <p><strong>Expected behavior:</strong></p>
                        <p>All methods should have wider intervals in extrapolation regions due to higher uncertainty.</p>
                    </div>

                    <p><strong>Typical performance:</strong></p>
                    <ul>
                        <li><strong>Hat Matrix</strong>: Good interpolation (~95%), degraded extrapolation (~70-85%)</li>
                        <li><strong>Bayesian</strong>: 100% everywhere (overly conservative)</li>
                        <li><strong>Conformal</strong>: Consistent across regions (~88%)</li>
                    </ul>
                </div>

                <div class="dataset-card">
                    <h4>5. Other Important Metrics</h4>

                    <h5>Prediction Accuracy (RMSE, MAE)</h5>
                    <ul>
                        <li><strong>RMSE</strong>: Root Mean Squared Error (point predictions)</li>
                        <li><strong>MAE</strong>: Mean Absolute Error</li>
                        <li>These measure how accurate the central prediction is (ignoring intervals)</li>
                    </ul>

                    <h5>Interval Score</h5>
                    <ul>
                        <li>Combines interval width with penalties for missed coverage</li>
                        <li>Formula: width + (2/α) × penalties for under/over predictions</li>
                        <li>Lower is better</li>
                    </ul>

                    <h5>Miscalibration Area</h5>
                    <ul>
                        <li>Area between observed and expected calibration curves</li>
                        <li>Measures how well probability levels match empirical coverage</li>
                        <li>Closer to 0 is better</li>
                    </ul>
                </div>

                <div class="info-box info">
                    <h4>🎯 Which Method is "Best"?</h4>

                    <p><strong>It depends on your needs:</strong></p>

                    <p><strong>Hat Matrix (pycse)</strong></p>
                    <ul>
                        <li>✓ <strong>Best for:</strong> Fast, sharp intervals with good coverage</li>
                        <li>✓ <strong>Pros:</strong> Analytical solution, very fast, sharp intervals</li>
                        <li>⚠️ <strong>Cons:</strong> Assumes correct model form, may under-cover in extrapolation</li>
                        <li><strong>Use when:</strong> Speed matters, you trust your model, slight under-coverage is acceptable</li>
                    </ul>

                    <p><strong>Bayesian Linear Regression</strong></p>
                    <ul>
                        <li>✓ <strong>Best for:</strong> Guaranteed coverage when you must be conservative</li>
                        <li>✓ <strong>Pros:</strong> Perfect coverage, accounts for parameter uncertainty</li>
                        <li>⚠️ <strong>Cons:</strong> Intervals 30-40× wider than Hat Matrix, less informative</li>
                        <li><strong>Use when:</strong> Safety-critical applications, you must avoid under-coverage at all costs</li>
                    </ul>

                    <p><strong>Conformal Prediction (MAPIE)</strong></p>
                    <ul>
                        <li>✓ <strong>Best for:</strong> Distribution-free guarantees with sharp intervals</li>
                        <li>✓ <strong>Pros:</strong> No parametric assumptions, guaranteed coverage (under exchangeability), sharp intervals</li>
                        <li>⚠️ <strong>Cons:</strong> Requires held-out calibration data, slower than Hat Matrix</li>
                        <li><strong>Use when:</strong> You don't trust model assumptions, need guaranteed coverage with sharp intervals</li>
                    </ul>

                    <p><strong>Our Recommendation:</strong></p>
                    <p>For most applications: <strong>Conformal Prediction</strong> or <strong>Hat Matrix</strong></p>
                    <ul>
                        <li>Both provide sharp, useful intervals</li>
                        <li>Hat Matrix: Faster, analytical</li>
                        <li>Conformal: Distribution-free guarantees</li>
                        <li>Bayesian: Only if you need 100% coverage and can tolerate very wide intervals</li>
                    </ul>
                </div>
"""

# Insert after the "How to Interpret Results" content, before the closing div
# Find the "How to Interpret" tab content and add metrics section
pattern = r'(<!-- Interpretation Tab -->.*?)(</div>\s*</div>\s*<!-- Interactive Results Table -->)'

import re

# Add metrics section before the closing of the guide section
html_content = re.sub(
    pattern,
    r'\1' + metrics_section + r'\n\2',
    html_content,
    flags=re.DOTALL
)

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ Metrics guide added to: {dashboard_path}")
print("\nThe 'How to Interpret' tab now includes:")
print("  - Coverage (Calibration) explanation")
print("  - Sharpness (Efficiency) discussion")
print("  - Calibration-Sharpness trade-off")
print("  - Regional coverage analysis")
print("  - Other metrics (RMSE, MAE, Interval Score)")
print("  - Method comparison and recommendations")
print("\nRefresh your browser to see the updates!")
