#!/usr/bin/env python
"""Add Gaussian Process results to the dashboard."""

import json
from pathlib import Path
import re

dashboard_path = Path('reports/dashboard/dashboard.html')
gp_results_dir = Path('results/gp_fits')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_gp')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# Load all GP results
gp_files = sorted(gp_results_dir.glob('*_GP.json'))
print(f"\nFound {len(gp_files)} GP results")

# Generate table rows
gp_rows = []
for gp_file in gp_files:
    with open(gp_file, 'r') as f:
        result = json.load(f)

    dataset = result['dataset']
    noise_model = result['noise_model'].capitalize()
    noise_level = result['noise_level']
    coverage = result['coverage']
    rmse = result['rmse']
    mean_width = result['mean_width']
    r2 = result['r2']

    # Determine coverage class
    if 0.93 <= coverage <= 0.97:
        coverage_class = "coverage-good"
    elif 0.90 <= coverage <= 0.99:
        coverage_class = "coverage-warning"
    else:
        coverage_class = "coverage-poor"

    # Create table row
    fit_path = f'../../results/figures/gp_fits_html/{dataset}_{noise_model.lower()}_noise{noise_level:02d}_GP.html'

    row = f'''        <tr class="table-row" onclick="showFit('{fit_path}', '{dataset}', '{noise_model.lower()}', {noise_level}, 'GP', {coverage:.3f})">
            <td>{dataset}</td>
            <td>{noise_model}</td>
            <td>{noise_level}%</td>
            <td>GP</td>
            <td class="{coverage_class}">{coverage:.3f}</td>
            <td>{rmse:.4f}</td>
            <td>{mean_width:.4f}</td>
            <td>{r2:.3f}</td>
            <td><button class="view-btn">View Fit</button></td>
        </tr>
    '''
    gp_rows.append(row)

print(f"Generated {len(gp_rows)} table rows")

# Find the data-driven table tbody
dd_table_pattern = r'(<table id="resultsTableDD">.*?<tbody>)(.*?)(</tbody>)'
dd_match = re.search(dd_table_pattern, html, re.DOTALL)

if dd_match:
    table_start = dd_match.group(1)
    tbody_content = dd_match.group(2)
    table_end = dd_match.group(3)

    # Add GP rows to the existing content
    new_tbody = tbody_content + '\n' + '\n'.join(gp_rows) + '\n                    '
    new_table = table_start + '\n' + new_tbody + table_end

    # Replace in HTML
    html = html[:dd_match.start()] + new_table + html[dd_match.end():]
    print("✓ Added GP rows to data-driven table")
else:
    print("⚠️ Could not find data-driven table")

# Update the UQ Method filter to include GP
filter_method_pattern = r'(<select id="filterMethodDD".*?<option value="EnsembleCalibrated">Ensemble Calibrated</option>)'

if re.search(filter_method_pattern, html, re.DOTALL):
    filter_replacement = r'\1\n                    <option value="GP">Gaussian Process</option>'
    html = re.sub(filter_method_pattern, filter_replacement, html, flags=re.DOTALL)
    print("✓ Added GP to UQ Method filter")
else:
    print("⚠️ Could not find UQ Method filter")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print(f"✓ Added {len(gp_rows)} Gaussian Process results to data-driven tab")
print("\nRefresh your browser to see the GP results!")
