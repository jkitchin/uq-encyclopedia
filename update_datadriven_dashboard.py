#!/usr/bin/env python
"""Update dashboard with data-driven models results."""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from config.global_config import RESULTS_CSV_DIR, DASHBOARD_DIR

print("Updating dashboard with data-driven results...")

# Load data-driven results
results_path = RESULTS_CSV_DIR / 'datadriven_models_comprehensive.csv'
if not results_path.exists():
    print(f"Error: Results file not found at {results_path}")
    print("Please run run_datadriven_benchmark.py first")
    sys.exit(1)

df = pd.read_csv(results_path)

# Convert to numeric (handles JAX arrays)
for col in ['coverage', 'mean_width', 'rmse', 'mae', 'r2']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Loaded {len(df)} experiments from {results_path}")

# Calculate summary statistics
total_experiments = len(df)
successful = df.dropna(subset=['coverage'])
coverage_mean_ensemble = successful[successful['uq_method'] == 'Ensemble']['coverage'].mean()
coverage_mean_calibrated = successful[successful['uq_method'] == 'EnsembleCalibrated']['coverage'].mean()
mean_width = successful['mean_width'].mean()
mean_r2 = successful['r2'].mean()

# Prepare data
df['noise_pct'] = (df['noise_level'] * 100).astype(int)
df['fit_file'] = df['exp_id'] + '_' + df['uq_method'] + '.html'
df['fit_path'] = '../../results/figures/datadriven_fits_html/' + df['fit_file']

# Prepare table rows (show both UQ methods)
table_rows = []
for idx, row in df.sort_values(['dataset', 'noise_model', 'noise_pct', 'uq_method']).iterrows():
    coverage_class = 'good' if row['coverage'] >= 0.90 else ('warning' if row['coverage'] >= 0.80 else 'poor')
    fit_link = row['fit_path']

    # Handle potentially negative R²
    r2_display = f"{row['r2']:.4f}" if row['r2'] >= 0 else f"<span style='color:red'>{row['r2']:.4f}</span>"

    table_rows.append(f"""
        <tr class="table-row" onclick="showFit('{fit_link}', '{row['dataset']}', '{row['noise_model']}', {row['noise_pct']}, '{row['uq_method']}', {row['coverage']:.3f})">
            <td>{row['dataset']}</td>
            <td>{row['noise_model'].capitalize()}</td>
            <td>{row['noise_pct']}%</td>
            <td>{row['uq_method']}</td>
            <td class="coverage-{coverage_class}">{row['coverage']:.3f}</td>
            <td>{row['rmse']:.4f}</td>
            <td>{row['mean_width']:.4f}</td>
            <td>{r2_display}</td>
            <td><button class="view-btn">View Fit</button></td>
        </tr>
    """)

table_html = '\n'.join(table_rows)

# Read current dashboard
dashboard_path = DASHBOARD_DIR / 'dashboard.html'
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Replace placeholder with actual content
datadriven_content = f"""
        <div class="section">
            <h2>📊 Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Total Experiments</h4>
                    <div class="value">{total_experiments}</div>
                </div>
                <div class="stat-card">
                    <h4>Ensemble Coverage</h4>
                    <div class="value">{coverage_mean_ensemble:.1%}</div>
                </div>
                <div class="stat-card">
                    <h4>Calibrated Ensemble Coverage</h4>
                    <div class="value">{coverage_mean_calibrated:.1%}</div>
                </div>
                <div class="stat-card">
                    <h4>Mean R²</h4>
                    <div class="value">{mean_r2:.3f}</div>
                </div>
            </div>
        </div>

        <!-- Model Architecture Info -->
        <div class="section">
            <h3>🧠 DPOSE Model Architecture</h3>
            <p><strong>Hidden Layer:</strong> 20 neurons with tanh activation</p>
            <p><strong>Output Layer:</strong> 32 ensemble members</p>
            <p><strong>Loss Function:</strong> CRPS (Continuous Ranked Probability Score)</p>
            <p><strong>Optimizer:</strong> BFGS</p>
            <p><strong>Seed:</strong> 42 (consistent across all experiments)</p>
        </div>

        <!-- Interactive Results Table -->
        <div class="section">
            <h2>🔍 Detailed Results - Click Any Row to View Fit</h2>

            <div class="filter-section">
                <select id="filterDatasetDD" onchange="filterTableDD()">
                    <option value="">All Datasets</option>
                    <option value="Line">Line</option>
                    <option value="Quadratic">Quadratic</option>
                    <option value="Cubic">Cubic</option>
                    <option value="ExponentialDecay">Exponential Decay</option>
                    <option value="LogisticGrowth">Logistic Growth</option>
                    <option value="MichaelisMenten">Michaelis-Menten</option>
                    <option value="Gaussian">Gaussian</option>
                </select>

                <select id="filterNoiseDD" onchange="filterTableDD()">
                    <option value="">All Noise Models</option>
                    <option value="Homoskedastic">Homoskedastic</option>
                    <option value="Heteroskedastic">Heteroskedastic</option>
                </select>

                <select id="filterMethodDD" onchange="filterTableDD()">
                    <option value="">All UQ Methods</option>
                    <option value="Ensemble">Ensemble</option>
                    <option value="EnsembleCalibrated">Ensemble Calibrated</option>
                </select>

                <input type="text" id="searchBoxDD" placeholder="Search..." onkeyup="filterTableDD()">
            </div>

            <div style="overflow-x: auto;">
                <table id="resultsTableDD">
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Noise Model</th>
                            <th>Noise Level</th>
                            <th>UQ Method</th>
                            <th>Coverage</th>
                            <th>RMSE</th>
                            <th>Mean Width</th>
                            <th>R²</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_html}
                    </tbody>
                </table>
            </div>
        </div>
"""

# Find and replace the data-driven placeholder
marker_start = '<div id="main-tab-datadriven" class="main-tab-content">'
marker_end = '        </div>\n\n    </div>\n    <!-- End Container -->'

if marker_start in html_content and marker_end in html_content:
    # Extract before and after
    before = html_content.split(marker_start)[0]
    after = html_content.split(marker_end)[1]

    # Reconstruct with new content
    html_content = before + marker_start + '\\n' + datadriven_content + '\\n        ' + marker_end + after

    # Add filter function for data-driven table
    filter_function = """
        // Filter data-driven table function
        function filterTableDD() {
            var datasetFilter = document.getElementById('filterDatasetDD').value;
            var noiseFilter = document.getElementById('filterNoiseDD').value.toLowerCase();
            var methodFilter = document.getElementById('filterMethodDD').value;
            var searchText = document.getElementById('searchBoxDD').value.toLowerCase();

            var table = document.getElementById('resultsTableDD');
            var rows = table.getElementsByTagName('tr');

            for (var i = 1; i < rows.length; i++) {
                var row = rows[i];
                var cells = row.getElementsByTagName('td');

                if (cells.length > 0) {
                    var dataset = cells[0].textContent;
                    var noiseModel = cells[1].textContent;
                    var method = cells[3].textContent;
                    var rowText = row.textContent.toLowerCase();

                    var showRow = true;

                    if (datasetFilter && dataset !== datasetFilter) showRow = false;
                    if (noiseFilter && noiseModel.toLowerCase() !== noiseFilter) showRow = false;
                    if (methodFilter && method !== methodFilter) showRow = false;
                    if (searchText && !rowText.includes(searchText)) showRow = false;

                    row.style.display = showRow ? '' : 'none';
                }
            }
        }
"""

    # Add before closing script tag
    html_content = html_content.replace('    </script>\\n</body>', filter_function + '\\n    </script>\\n</body>')

    # Save updated dashboard
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\\n✓ Dashboard updated: {dashboard_path}")
    print(f"\\nData-Driven Models tab now includes:")
    print(f"  - Summary statistics ({total_experiments} experiments)")
    print(f"  - Model architecture details")
    print(f"  - Interactive results table")
    print(f"  - Filter by dataset, noise model, and UQ method")
    print(f"  - Clickable rows to view fit plots")
else:
    print("Error: Could not find data-driven tab markers in dashboard")
