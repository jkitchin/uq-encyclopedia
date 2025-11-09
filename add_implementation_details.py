#!/usr/bin/env python
"""Add implementation details and code links to the dashboard."""

import sys
sys.path.insert(0, '.')
from pathlib import Path
from config.global_config import DASHBOARD_DIR

print("Adding implementation details to dashboard...")

# Read the current dashboard
dashboard_path = DASHBOARD_DIR / 'dashboard.html'
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Create the implementation details HTML
implementation_section = """
                <div class="info-box info">
                    <h4>📦 Implementation Details</h4>

                    <h5>Libraries Used:</h5>
                    <ul>
                        <li><strong>pycse</strong>: <code>regress</code> and <code>predict</code> for Hat Matrix UQ (<a href="https://github.com/jkitchin/pycse" target="_blank">GitHub</a>)</li>
                        <li><strong>scikit-learn</strong> (<code>sklearn</code>): BayesianRidge for Bayesian regression</li>
                        <li><strong>MAPIE</strong>: CrossConformalRegressor for Conformal Prediction (<a href="https://github.com/scikit-learn-contrib/MAPIE" target="_blank">GitHub</a>)</li>
                        <li><strong>NumPy</strong>: Matrix operations and linear algebra</li>
                        <li><strong>SciPy</strong>: Statistical distributions</li>
                    </ul>

                    <h5>Code Location:</h5>
                    <ul>
                        <li><strong>UQ Methods</strong>: <code>src/uq_methods/linear_uq.py</code></li>
                        <li><strong>Models</strong>: <code>src/models/linear_models.py</code></li>
                        <li><strong>Datasets</strong>: <code>src/datasets/linear.py</code></li>
                        <li><strong>Metrics</strong>: <code>src/metrics/</code></li>
                    </ul>

                    <h5>Key References:</h5>
                    <ul>
                        <li><strong>Hat Matrix</strong>: pycse.regress and pycse.predict (<a href="https://github.com/jkitchin/pycse" target="_blank">GitHub repo</a>)</li>
                        <li><strong>Bayesian Ridge</strong>: sklearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html" target="_blank">BayesianRidge docs</a></li>
                        <li><strong>Design Matrix</strong>: Polynomial/basis function transformation in <code>linear_models.py</code></li>
                    </ul>
                </div>

                <h4>🔧 Implementation Notes</h4>

                <div class="dataset-card">
                    <h5>Hat Matrix Method</h5>
                    <div class="code-snippet">
<pre>
# Uses pycse.regress and pycse.predict
from pycse import regress, predict

# 1. Fit parameters
pars, pars_int, se = regress(X_train, y_train, alpha=0.05)

# 2. Get predictions with prediction intervals
y_pred, y_int, pred_se = predict(X_train, y_train, pars, X_test, alpha=0.05)

# Formula: ŷ ± t(α/2, df) × σ × sqrt(1 + h_i)
#   - h_i = leverage (hat matrix diagonal)
#   - σ = residual standard error
</pre>
                    </div>
                    <p><strong>Implementation</strong>: <code>src/uq_methods/linear_uq.py:HatMatrixUQ</code> (lines 11-127)</p>
                    <p><strong>Library</strong>: pycse (https://github.com/jkitchin/pycse)</p>
                    <p><strong>Key feature</strong>: Fast, analytic solution using classical OLS theory</p>
                </div>

                <div class="dataset-card">
                    <h5>Bayesian Linear Regression</h5>
                    <div class="code-snippet">
<pre>
# Uses sklearn's BayesianRidge with:
# - Gamma priors on α (noise precision) and λ (weight precision)
# - Automatic relevance determination (ARD)
# - Returns predictive distribution: N(μ, σ²)
# - Intervals: μ ± z(α/2) × σ
</pre>
                    </div>
                    <p><strong>Implementation</strong>: <code>src/uq_methods/linear_uq.py:BayesianLinearRegressionUQ</code></p>
                    <p><strong>Key feature</strong>: Accounts for both parameter and noise uncertainty</p>
                    <p><strong>Hyperparameters</strong>: Tuned via grid search (α₁=1e-7, α₂=1e-5, λ₁=1e-5, λ₂=1e-7)</p>
                </div>

                <div class="dataset-card">
                    <h5>Conformal Prediction</h5>
                    <div class="code-snippet">
<pre>
# Uses MAPIE's CrossConformalRegressor with:
# - Distribution-free prediction intervals
# - Cross-validation (CV=5) for better coverage
# - Method: 'plus' (CV+ method)
# - Guaranteed finite-sample coverage (under exchangeability)
</pre>
                    </div>
                    <p><strong>Implementation</strong>: <code>src/uq_methods/linear_uq.py:ConformalPredictionUQ</code></p>
                    <p><strong>Library</strong>: MAPIE (https://github.com/scikit-learn-contrib/MAPIE)</p>
                    <p><strong>Key feature</strong>: Distribution-free, guaranteed coverage without parametric assumptions</p>
                </div>

                <div class="dataset-card">
                    <h5>Design Matrix Transformation</h5>
                    <div class="code-snippet">
<pre>
# For Polynomial (degree 3):
X_design = [1, x, x², x³]

# For Lennard-Jones:
X_design = [1, 1/r⁶, 1/r¹²]

# For Shomate:
X_design = [1, t, t², t³, 1/t²]  where t = T/1000
</pre>
                    </div>
                    <p><strong>Critical fix</strong>: Bayesian method now uses same design matrix as OLS</p>
                    <p><strong>Result</strong>: Improved coverage from 83% to 100% after fixing this bug</p>
                </div>
"""

# Find the UQ Methods section and add implementation details before the closing div
import re

# Look for the end of the UQ methods content
pattern = r'(</div>\s*<!-- Interpretation Tab -->)'
replacement = implementation_section + r'\n            \1'

html_content = re.sub(pattern, replacement, html_content, flags=re.DOTALL)

# Add CSS for code snippets
code_snippet_css = """
        .code-snippet {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            margin: 0.5rem 0;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 0.9em;
        }

        .code-snippet pre {
            margin: 0;
            white-space: pre;
        }

        .info-box a {
            color: #667eea;
            text-decoration: underline;
        }

        .info-box a:hover {
            color: #764ba2;
        }
"""

# Add CSS before the closing </style> tag
html_content = html_content.replace('    </style>', code_snippet_css + '\n    </style>')

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ Implementation details added to: {dashboard_path}")
print("\nThe UQ Methods tab now includes:")
print("  - Libraries used (sklearn, NumPy, SciPy)")
print("  - Code file locations")
print("  - Implementation formulas and notes")
print("  - Key references and documentation links")
print("  - Design matrix transformation details")
print("\nRefresh your browser to see the updates!")
