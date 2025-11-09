#!/usr/bin/env python
"""Generate fully interactive HTML dashboard with Plotly visualizations."""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
import sys
sys.path.insert(0, '.')

from config.global_config import RESULTS_CSV_DIR, RESULTS_FIGURES_DIR, DASHBOARD_DIR

print("=" * 70)
print("CREATING INTERACTIVE DASHBOARD")
print("=" * 70)

# Load results
results_path = RESULTS_CSV_DIR / 'linear_models_comprehensive.csv'
df = pd.read_csv(results_path)
print(f"\nLoaded {len(df)} experiment results")

# Create fits directory path (now HTML instead of PNG)
FITS_DIR = RESULTS_FIGURES_DIR / 'individual_fits_html'

# Prepare data for visualizations
df['noise_pct'] = (df['noise_level'] * 100).astype(int)
df['exp_id'] = (df['dataset'] + '_' + df['noise_model'] + '_noise' +
                df['noise_pct'].astype(str).str.zfill(2))
df['fit_file'] = df['exp_id'] + '_' + df['uq_method'] + '.html'
df['fit_path'] = '../../results/figures/individual_fits_html/' + df['fit_file']

# Calculate summary statistics
print("\nCalculating summary statistics...")
total_experiments = len(df)
coverage_by_method = df.groupby('uq_method')['coverage'].mean()
target_achievement = (df['coverage'] >= 0.90).sum() / len(df) * 100

# Get all unique methods for dropdown
all_methods = sorted(df['uq_method'].unique())

print(f"  Total experiments: {total_experiments}")
print(f"  Coverage by method:")
for method in all_methods:
    print(f"    - {method}: {coverage_by_method[method]:.1%}")
print(f"  Target achievement (≥90%): {target_achievement:.1f}%")

# Create HTML with interactive table
print("\nBuilding HTML...")

# Prepare table data with links
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

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UQ Encyclopedia - Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}

        .container {{
            max-width: 1600px;
            margin: 2rem auto;
            padding: 0 2rem;
        }}

        .section {{
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .section h2 {{
            color: #667eea;
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
        }}

        .plot-container {{
            margin: 2rem 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 1rem;
        }}

        th {{
            background: #667eea;
            color: white;
            padding: 1rem;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #e0e0e0;
        }}

        .table-row {{
            cursor: pointer;
            transition: background 0.2s;
        }}

        .table-row:hover {{
            background: #f0f4ff;
        }}

        .coverage-good {{ color: #4caf50; font-weight: bold; }}
        .coverage-warning {{ color: #ff9800; font-weight: bold; }}
        .coverage-poor {{ color: #f44336; font-weight: bold; }}

        .view-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
        }}

        .view-btn:hover {{
            background: #5568d3;
        }}

        /* Modal styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
        }}

        .modal-content {{
            background-color: white;
            margin: 2% auto;
            padding: 2rem;
            width: 90%;
            max-width: 1400px;
            border-radius: 8px;
            position: relative;
        }}

        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}

        .close:hover {{
            color: #000;
        }}

        .modal-content iframe {{
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 4px;
        }}

        .filter-section {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}

        .filter-section select, .filter-section input {{
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
        }}

        .stat-card h4 {{
            font-size: 0.9rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }}

        .stat-card .value {{
            font-size: 2rem;
            font-weight: bold;
        }}

        /* Main Tab Navigation */
        .main-tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            border-bottom: 2px solid #e0e0e0;
        }}

        .main-tab-btn {{
            padding: 1rem 2rem;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 600;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }}

        .main-tab-btn:hover {{
            color: #667eea;
            background: #f5f5f5;
        }}

        .main-tab-btn.active {{
            color: #667eea;
            border-bottom-color: #667eea;
        }}

        .main-tab-content {{
            display: none;
        }}

        .main-tab-content.active {{
            display: block;
        }}

        .placeholder-section {{
            padding: 4rem 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            margin: 2rem 0;
        }}

        .placeholder-section h2 {{
            font-size: 2rem;
            color: #667eea;
            margin-bottom: 1rem;
        }}

        .placeholder-section p {{
            font-size: 1.1rem;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 UQ Encyclopedia - Results Dashboard</h1>
        <p>Comprehensive Benchmark Results with Interactive Fit Visualizations</p>
    </div>

    <div class="container">
        <!-- Main Tabs Navigation -->
        <div class="main-tabs">
            <button class="main-tab-btn active" onclick="showMainTab('linear')">📈 Linear Models</button>
            <button class="main-tab-btn" onclick="showMainTab('nonlinear')">🔄 Nonlinear Models</button>
            <button class="main-tab-btn" onclick="showMainTab('datadriven')">🤖 Data-Driven Models</button>
        </div>

        <!-- Linear Models Tab -->
        <div id="main-tab-linear" class="main-tab-content active">
        <!-- Summary Stats -->
        <div class="section">
            <h2>📊 Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Total Experiments</h4>
                    <div class="value">{total_experiments}</div>
                </div>
                <div class="stat-card">
                    <h4>Hat Matrix Coverage</h4>
                    <div class="value">{coverage_by_method.get('HatMatrix', 0):.1%}</div>
                </div>
                <div class="stat-card">
                    <h4>Bayesian Coverage</h4>
                    <div class="value">{coverage_by_method.get('Bayesian', 0):.1%}</div>
                </div>
                <div class="stat-card">
                    <h4>Conformal Coverage</h4>
                    <div class="value">{coverage_by_method.get('Conformal', 0):.1%}</div>
                </div>
                <div class="stat-card">
                    <h4>Target Achievement</h4>
                    <div class="value">{target_achievement:.1f}%</div>
                </div>
            </div>
        </div>

        <!-- Interactive Results Table -->
        <div class="section">
            <h2>🔍 Detailed Results - Click Any Row to View Fit</h2>

            <div class="filter-section">
                <select id="filterDataset" onchange="filterTable()">
                    <option value="">All Datasets</option>
                    <option value="Line">Line</option>
                    <option value="Polynomial">Polynomial</option>
                    <option value="LennardJones">Lennard-Jones</option>
                    <option value="Shomate">Shomate</option>
                </select>

                <select id="filterMethod" onchange="filterTable()">
                    <option value="">All Methods</option>
                    <option value="HatMatrix">Hat Matrix</option>
                    <option value="Bayesian">Bayesian</option>
                    <option value="Conformal">Conformal</option>
                </select>

                <select id="filterNoise" onchange="filterTable()">
                    <option value="">All Noise Models</option>
                    <option value="Homoskedastic">Homoskedastic</option>
                    <option value="Heteroskedastic">Heteroskedastic</option>
                </select>

                <input type="text" id="searchBox" placeholder="Search..." onkeyup="filterTable()">
            </div>

            <div style="overflow-x: auto;">
                <table id="resultsTable">
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
    </div>
        </div>
        <!-- End Linear Models Tab -->

        <!-- Nonlinear Models Tab -->
        <div id="main-tab-nonlinear" class="main-tab-content">
            <div class="placeholder-section">
                <h2>🔄 Nonlinear Models</h2>
                <p>Nonlinear model benchmarks coming soon...</p>
                <p style="margin-top: 1rem; color: #999;">This section will include Gaussian Processes, Neural Networks, and other nonlinear UQ methods.</p>
            </div>
        </div>

        <!-- Data-Driven Models Tab -->
        <div id="main-tab-datadriven" class="main-tab-content">
            <div class="placeholder-section">
                <h2>🤖 Data-Driven Models</h2>
                <p>Data-driven model benchmarks coming soon...</p>
                <p style="margin-top: 1rem; color: #999;">This section will include Deep Ensembles, Bayesian Neural Networks, and other data-driven UQ methods.</p>
            </div>
        </div>

    </div>
    <!-- End Container -->

    <!-- Modal for displaying fits -->
    <div id="fitModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeFit()">&times;</span>
            <h2 id="modalTitle"></h2>
            <iframe id="fitFrame" src=""></iframe>
        </div>
    </div>

    <script>
        // Show fit function
        function showFit(fitPath, dataset, noiseModel, noiseLevel, method, coverage) {{
            var modal = document.getElementById('fitModal');
            var iframe = document.getElementById('fitFrame');
            var title = document.getElementById('modalTitle');

            iframe.src = fitPath;
            title.innerHTML = dataset + ' | ' + noiseModel.charAt(0).toUpperCase() + noiseModel.slice(1) +
                             ' ' + noiseLevel + '% | ' + method +
                             ' (Coverage: ' + (coverage * 100).toFixed(1) + '%)';

            modal.style.display = 'block';
        }}

        function closeFit() {{
            var modal = document.getElementById('fitModal');
            var iframe = document.getElementById('fitFrame');
            modal.style.display = 'none';
            // Clear iframe to stop any animations/interactions
            iframe.src = '';
        }}

        // Close modal when clicking outside
        window.onclick = function(event) {{
            var modal = document.getElementById('fitModal');
            if (event.target == modal) {{
                modal.style.display = 'none';
            }}
        }}

        // Filter table function
        function filterTable() {{
            var datasetFilter = document.getElementById('filterDataset').value;
            var methodFilter = document.getElementById('filterMethod').value;
            var noiseFilter = document.getElementById('filterNoise').value.toLowerCase();
            var searchText = document.getElementById('searchBox').value.toLowerCase();

            var table = document.getElementById('resultsTable');
            var rows = table.getElementsByTagName('tr');

            for (var i = 1; i < rows.length; i++) {{
                var row = rows[i];
                var cells = row.getElementsByTagName('td');

                if (cells.length > 0) {{
                    var dataset = cells[0].textContent;
                    var noiseModel = cells[1].textContent;
                    var method = cells[3].textContent;
                    var rowText = row.textContent.toLowerCase();

                    var showRow = true;

                    if (datasetFilter && dataset !== datasetFilter) showRow = false;
                    if (methodFilter && method !== methodFilter) showRow = false;
                    if (noiseFilter && noiseModel.toLowerCase() !== noiseFilter) showRow = false;
                    if (searchText && !rowText.includes(searchText)) showRow = false;

                    row.style.display = showRow ? '' : 'none';
                }}
            }}
        }}

        // Main tab switching function
        function showMainTab(tabName) {{
            // Hide all tab contents
            var contents = document.getElementsByClassName('main-tab-content');
            for (var i = 0; i < contents.length; i++) {{
                contents[i].classList.remove('active');
            }}

            // Remove active class from all tab buttons
            var buttons = document.getElementsByClassName('main-tab-btn');
            for (var i = 0; i < buttons.length; i++) {{
                buttons[i].classList.remove('active');
            }}

            // Show selected tab content
            document.getElementById('main-tab-' + tabName).classList.add('active');

            // Activate clicked button
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""

# Save HTML
output_path = DASHBOARD_DIR / 'dashboard.html'
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✓ Interactive dashboard created: {output_path}")
print(f"\nTo view the dashboard:")
print(f"  open {output_path}")
