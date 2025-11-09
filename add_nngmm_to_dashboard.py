#!/usr/bin/env python
"""Add Neural Network GMM results to the dashboard."""

import json
from pathlib import Path
import re

dashboard_path = Path('reports/dashboard/dashboard.html')
nngmm_results_dir = Path('results/nngmm_fits')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_nngmm')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# Load all NNGMM results
nngmm_files = sorted(nngmm_results_dir.glob('*_NNGMM.json'))
print(f"\nFound {len(nngmm_files)} NNGMM results")

# Generate table rows
nngmm_rows = []
for nngmm_file in nngmm_files:
    with open(nngmm_file, 'r') as f:
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
    fit_path = f'../../results/figures/nngmm_fits_html/{dataset}_{noise_model.lower()}_noise{noise_level:02d}_NNGMM.html'

    row = f'''        <tr class="table-row" onclick="showFit('{fit_path}', '{dataset}', '{noise_model.lower()}', {noise_level}, 'NNGMM', {coverage:.3f})">
            <td>{dataset}</td>
            <td>{noise_model}</td>
            <td>{noise_level}%</td>
            <td>NNGMM</td>
            <td class="{coverage_class}">{coverage:.3f}</td>
            <td>{rmse:.4f}</td>
            <td>{mean_width:.4f}</td>
            <td>{r2:.3f}</td>
            <td><button class="view-btn">View Fit</button></td>
        </tr>
    '''
    nngmm_rows.append(row)

print(f"Generated {len(nngmm_rows)} table rows")

# Find the data-driven table tbody
dd_table_pattern = r'(<table id="resultsTableDD">.*?<tbody>)(.*?)(</tbody>)'
dd_match = re.search(dd_table_pattern, html, re.DOTALL)

if dd_match:
    table_start = dd_match.group(1)
    tbody_content = dd_match.group(2)
    table_end = dd_match.group(3)

    # Add NNGMM rows to the existing content
    new_tbody = tbody_content + '\n' + '\n'.join(nngmm_rows) + '\n                    '
    new_table = table_start + '\n' + new_tbody + table_end

    # Replace in HTML
    html = html[:dd_match.start()] + new_table + html[dd_match.end():]
    print("✓ Added NNGMM rows to data-driven table")
else:
    print("⚠️ Could not find data-driven table")

# Update the UQ Method filter to include NNGMM
filter_method_pattern = r'(<select id="filterMethodDD".*?<option value="GP">Gaussian Process</option>)'

if re.search(filter_method_pattern, html, re.DOTALL):
    filter_replacement = r'\1\n                    <option value="NNGMM">Neural Network GMM</option>'
    html = re.sub(filter_method_pattern, filter_replacement, html, flags=re.DOTALL)
    print("✓ Added NNGMM to UQ Method filter")
else:
    print("⚠️ Could not find UQ Method filter")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print(f"✓ Added {len(nngmm_rows)} Neural Network GMM results to data-driven tab")
print("\nRefresh your browser to see the NNGMM results!")
