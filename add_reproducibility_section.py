#!/usr/bin/env python
"""Add reproducibility section with links to code and scripts."""

import sys
sys.path.insert(0, '.')
from pathlib import Path
from config.global_config import DASHBOARD_DIR

print("Adding reproducibility section to dashboard...")

# Read the current dashboard
dashboard_path = DASHBOARD_DIR / 'dashboard.html'
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Create the reproducibility section HTML
reproducibility_section = """
                <h4>🔄 Reproducibility</h4>

                <div class="info-box success">
                    <h5>How to Reproduce These Results</h5>
                    <p>All results in this dashboard can be reproduced by running the scripts below.
                    The codebase uses deterministic seeding for full reproducibility.</p>
                </div>

                <div class="dataset-card">
                    <h5>1. Run Complete Benchmark (All 64 Experiments)</h5>
                    <div class="code-snippet">
<pre>
# Run all experiments (4 datasets × 2 noise models × 4 levels × 2 UQ methods)
python run_linear_benchmark.py

# This generates:
# - results/csv/linear_models_comprehensive.csv
# - results/csv/linear_models_summary.csv
</pre>
                    </div>
                    <p><strong>Script</strong>: <code>run_linear_benchmark.py</code></p>
                    <p><strong>Runtime</strong>: ~2-3 minutes</p>
                    <p><strong>Output</strong>: CSV files with all metrics (coverage, RMSE, width, etc.)</p>
                </div>

                <div class="dataset-card">
                    <h5>2. Generate Visualizations</h5>
                    <div class="code-snippet">
<pre>
# Create all static visualizations (11 figures)
python generate_linear_visualizations.py

# Generate individual fit plots (64 plots, one per experiment)
python generate_individual_fits.py

# Create this interactive dashboard
python create_interactive_dashboard.py
python add_guide_to_dashboard.py
python add_dataset_plots_to_guide.py
python add_implementation_details.py
</pre>
                    </div>
                    <p><strong>Output</strong>: PNG figures in <code>results/figures/</code> and HTML dashboard</p>
                </div>

                <div class="dataset-card">
                    <h5>3. Compute Calibration-Sharpness Scores</h5>
                    <div class="code-snippet">
<pre>
# Analyze calibration vs efficiency trade-off
python compute_calibration_sharpness.py

# This computes the combined metric showing:
# - Bayesian: 100% coverage but 37x wider intervals
# - Hat Matrix: 88% coverage with sharp intervals
</pre>
                    </div>
                    <p><strong>Output</strong>: <code>results/csv/calibration_sharpness_scores.csv</code></p>
                </div>

                <div class="dataset-card">
                    <h5>4. Hyperparameter Tuning (Optional)</h5>
                    <div class="code-snippet">
<pre>
# Run grid search for Bayesian hyperparameters (81 combinations × 32 experiments)
python tune_bayesian_hyperparameters.py

# Analyze tuning results
python analyze_tuning_results.py
</pre>
                    </div>
                    <p><strong>Runtime</strong>: ~5-10 minutes</p>
                    <p><strong>Note</strong>: Already completed; results show minimal difference between parameters</p>
                </div>

                <h4>📁 Source Code Reference</h4>

                <table class="code-table">
                    <thead>
                        <tr>
                            <th>Component</th>
                            <th>File Path</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Datasets</strong></td>
                            <td><code>src/datasets/linear.py</code></td>
                            <td>Line, Polynomial, Lennard-Jones, Shomate</td>
                        </tr>
                        <tr>
                            <td><strong>Models</strong></td>
                            <td><code>src/models/linear_models.py</code></td>
                            <td>OLS regression with basis functions</td>
                        </tr>
                        <tr>
                            <td><strong>UQ Methods</strong></td>
                            <td><code>src/uq_methods/linear_uq.py</code></td>
                            <td>Hat Matrix, Bayesian, Conformal</td>
                        </tr>
                        <tr>
                            <td><strong>Metrics</strong></td>
                            <td><code>src/metrics/</code></td>
                            <td>Coverage, efficiency, accuracy metrics</td>
                        </tr>
                        <tr>
                            <td><strong>Seed Management</strong></td>
                            <td><code>src/utils/seeds.py</code></td>
                            <td>Deterministic experiment seeding</td>
                        </tr>
                        <tr>
                            <td><strong>Configuration</strong></td>
                            <td><code>config/global_config.py</code></td>
                            <td>Global parameters and paths</td>
                        </tr>
                    </tbody>
                </table>

                <div class="info-box info">
                    <h5>🔗 Quick Links to Example Code</h5>
                    <ul>
                        <li><strong>Run a single experiment</strong>: See <code>test_linear_models.py</code></li>
                        <li><strong>Hat Matrix implementation</strong>: <code>src/uq_methods/linear_uq.py:HatMatrixUQ</code> (lines 46-112)</li>
                        <li><strong>Bayesian implementation</strong>: <code>src/uq_methods/linear_uq.py:BayesianLinearRegressionUQ</code> (lines 115-229)</li>
                        <li><strong>Design matrix transformation</strong>: <code>src/models/linear_models.py:_create_design_matrix</code> (lines 52-84)</li>
                        <li><strong>Coverage metric</strong>: <code>src/metrics/coverage.py:prediction_interval_coverage_probability</code></li>
                        <li><strong>Calibration-sharpness score</strong>: <code>src/metrics/efficiency.py:calibration_sharpness_score</code> (lines 259-328)</li>
                    </ul>
                </div>

                <div class="info-box warning">
                    <h5>⚠️ Requirements</h5>
                    <p>Install dependencies before running:</p>
                    <div class="code-snippet">
<pre>
pip install numpy scipy scikit-learn matplotlib seaborn pandas plotly tqdm h5py
</pre>
                    </div>
                    <p>Or use the provided requirements file:</p>
                    <div class="code-snippet">
<pre>
pip install -r requirements.txt
</pre>
                    </div>
                </div>
"""

# Find the end of the "How to Interpret" tab and insert before closing div
import re
# Look for the last content in the interpret tab, then insert before its closing div
pattern = r'(Notice Bayesian.*?</ol>\s*</div>\s*</div>)'
replacement = r'\1\n' + reproducibility_section

html_content = re.sub(pattern, replacement, html_content, flags=re.DOTALL)

# Add CSS for the code table
code_table_css = """
        .code-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9em;
        }

        .code-table thead {
            background: #667eea;
            color: white;
        }

        .code-table th,
        .code-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }

        .code-table tbody tr:hover {
            background: #f5f5f5;
        }

        .code-table code {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
        }

        .info-box.success {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
        }

        .info-box.warning {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
        }
"""

# Add CSS before the closing </style> tag
html_content = html_content.replace('    </style>', code_table_css + '\n    </style>')

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ Reproducibility section added to: {dashboard_path}")
print("\nThe 'How to Interpret' tab now includes:")
print("  - Complete reproduction instructions")
print("  - Command-line examples for all scripts")
print("  - Source code reference table")
print("  - Links to specific implementations")
print("  - Runtime estimates")
print("  - Requirements and dependencies")
print("\nRefresh your browser to see the updates!")
