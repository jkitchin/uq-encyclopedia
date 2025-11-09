#!/usr/bin/env python
"""Update dashboard with nonlinear models results."""

import sys
sys.path.insert(0, '.')
import pandas as pd
from pathlib import Path
from config.global_config import RESULTS_CSV_DIR, DASHBOARD_DIR

print("Updating dashboard with nonlinear results...")

# Load nonlinear results
results_path = RESULTS_CSV_DIR / 'nonlinear_models_comprehensive.csv'
if not results_path.exists():
    print(f"Error: Results file not found at {results_path}")
    print("Please run run_nonlinear_benchmark.py first")
    sys.exit(1)

df = pd.read_csv(results_path)
print(f"Loaded {len(df)} experiments from {results_path}")

# Calculate summary statistics
total_experiments = len(df)
successful = df.dropna(subset=['coverage'])
coverage_mean = successful['coverage'].mean()
mean_width = successful['mean_width'].mean()

# Prepare data
df['noise_pct'] = (df['noise_level'] * 100).astype(int)
df['fit_file'] = df['exp_id'] + '_Nlinfit.html'
df['fit_path'] = '../../results/figures/nonlinear_fits_html/' + df['fit_file']

# Prepare table rows
table_rows = []
for idx, row in df.sort_values('coverage', ascending=False).iterrows():
    coverage_class = 'good' if row['coverage'] >= 0.90 else ('warning' if row['coverage'] >= 0.80 else 'poor')
    fit_link = row['fit_path']

    table_rows.append(f"""
        <tr class="table-row" onclick="showFit('{fit_link}', '{row['dataset']}', '{row['noise_model']}', {row['noise_pct']}, '{row['uq_method']}', {row['coverage']:.3f})">
            <td>{row['dataset']}</td>
            <td>{row['noise_model'].capitalize()}</td>
            <td>{row['noise_pct']}%</td>
            <td>{row['uq_method']}</td>
            <td class="coverage-{coverage_class}">{row['coverage']:.3f}</td>
            <td>{row['rmse']:.4f}</td>
            <td>{row['mean_width']:.4f}</td>
            <td><button class="view-btn">View Fit</button></td>
        </tr>
    """)

table_html = '\n'.join(table_rows)

# Read current dashboard
dashboard_path = DASHBOARD_DIR / 'dashboard.html'
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Replace placeholder with actual content
nonlinear_content = f"""
        <div class="section">
            <h2>📊 Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Total Experiments</h4>
                    <div class="value">{total_experiments}</div>
                </div>
                <div class="stat-card">
                    <h4>Nlinfit Coverage</h4>
                    <div class="value">{coverage_mean:.1%}</div>
                </div>
                <div class="stat-card">
                    <h4>Mean Interval Width</h4>
                    <div class="value">{mean_width:.3f}</div>
                </div>
            </div>
        </div>

        <!-- Interactive Results Table -->
        <div class="section">
            <h2>🔍 Detailed Results - Click Any Row to View Fit</h2>

            <div class="filter-section">
                <select id="filterDatasetNL" onchange="filterTableNL()">
                    <option value="">All Datasets</option>
                    <option value="ExponentialDecay">Exponential Decay</option>
                    <option value="LogisticGrowth">Logistic Growth</option>
                    <option value="MichaelisMenten">Michaelis-Menten</option>
                    <option value="Gaussian">Gaussian</option>
                </select>

                <select id="filterNoiseNL" onchange="filterTableNL()">
                    <option value="">All Noise Models</option>
                    <option value="Homoskedastic">Homoskedastic</option>
                    <option value="Heteroskedastic">Heteroskedastic</option>
                </select>

                <input type="text" id="searchBoxNL" placeholder="Search..." onkeyup="filterTableNL()">
            </div>

            <div style="overflow-x: auto;">
                <table id="resultsTableNL">
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Noise Model</th>
                            <th>Noise Level</th>
                            <th>UQ Method</th>
                            <th>Coverage</th>
                            <th>RMSE</th>
                            <th>Mean Width</th>
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

# Find and replace the nonlinear placeholder
marker_start = '<div id="main-tab-nonlinear" class="main-tab-content">'
marker_end = '</div>\n\n        <!-- Data-Driven Models Tab -->'

if marker_start in html_content and marker_end in html_content:
    # Extract before and after
    before = html_content.split(marker_start)[0]
    after = html_content.split(marker_end)[1]

    # Reconstruct with new content
    html_content = before + marker_start + '\n' + nonlinear_content + '\n        ' + marker_end + after

    # Add filter function for nonlinear table
    filter_function = """
        // Filter nonlinear table function
        function filterTableNL() {
            var datasetFilter = document.getElementById('filterDatasetNL').value;
            var noiseFilter = document.getElementById('filterNoiseNL').value.toLowerCase();
            var searchText = document.getElementById('searchBoxNL').value.toLowerCase();

            var table = document.getElementById('resultsTableNL');
            var rows = table.getElementsByTagName('tr');

            for (var i = 1; i < rows.length; i++) {
                var row = rows[i];
                var cells = row.getElementsByTagName('td');

                if (cells.length > 0) {
                    var dataset = cells[0].textContent;
                    var noiseModel = cells[1].textContent;
                    var rowText = row.textContent.toLowerCase();

                    var showRow = true;

                    if (datasetFilter && dataset !== datasetFilter) showRow = false;
                    if (noiseFilter && noiseModel.toLowerCase() !== noiseFilter) showRow = false;
                    if (searchText && !rowText.includes(searchText)) showRow = false;

                    row.style.display = showRow ? '' : 'none';
                }
            }
        }
"""

    # Add before closing script tag
    html_content = html_content.replace('    </script>\n</body>', filter_function + '\n    </script>\n</body>')

    # Save updated dashboard
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✓ Dashboard updated: {dashboard_path}")
    print(f"\nNonlinear Models tab now includes:")
    print(f"  - Summary statistics ({total_experiments} experiments)")
    print(f"  - Interactive results table")
    print(f"  - Filter by dataset and noise model")
    print(f"  - Clickable rows to view fit plots")
else:
    print("Error: Could not find nonlinear tab markers in dashboard")
