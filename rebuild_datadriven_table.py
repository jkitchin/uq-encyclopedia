#!/usr/bin/env python
"""Rebuild all data-driven results in the dashboard."""

import json
from pathlib import Path
import re

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

print("Rebuilding data-driven table...")

# Collect all results
all_rows = []

# Load GP results
gp_results_dir = Path('results/gp_fits')
gp_files = sorted(gp_results_dir.glob('*_GP.json'))
print(f"Found {len(gp_files)} GP results")

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

    if 0.93 <= coverage <= 0.97:
        coverage_class = "coverage-good"
    elif 0.90 <= coverage <= 0.99:
        coverage_class = "coverage-warning"
    else:
        coverage_class = "coverage-poor"

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
        </tr>'''
    all_rows.append(row)

# Load NNGMM results
nngmm_results_dir = Path('results/nngmm_fits')
nngmm_files = sorted(nngmm_results_dir.glob('*_NNGMM.json'))
print(f"Found {len(nngmm_files)} NNGMM results")

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

    if 0.93 <= coverage <= 0.97:
        coverage_class = "coverage-good"
    elif 0.90 <= coverage <= 0.99:
        coverage_class = "coverage-warning"
    else:
        coverage_class = "coverage-poor"

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
        </tr>'''
    all_rows.append(row)

# Load NNBR results
nnbr_results_dir = Path('results/nnbr_fits')
nnbr_files = sorted(nnbr_results_dir.glob('*_NNBR.json'))
print(f"Found {len(nnbr_files)} NNBR results")

for nnbr_file in nnbr_files:
    with open(nnbr_file, 'r') as f:
        result = json.load(f)

    dataset = result['dataset']
    noise_model = result['noise_model'].capitalize()
    noise_level = result['noise_level']
    coverage = result['coverage']
    rmse = result['rmse']
    mean_width = result['mean_width']
    r2 = result['r2']

    if 0.93 <= coverage <= 0.97:
        coverage_class = "coverage-good"
    elif 0.90 <= coverage <= 0.99:
        coverage_class = "coverage-warning"
    else:
        coverage_class = "coverage-poor"

    fit_path = f'../../results/figures/nnbr_fits_html/{dataset}_{noise_model.lower()}_noise{noise_level:02d}_NNBR.html'

    row = f'''        <tr class="table-row" onclick="showFit('{fit_path}', '{dataset}', '{noise_model.lower()}', {noise_level}, 'NNBR', {coverage:.3f})">
            <td>{dataset}</td>
            <td>{noise_model}</td>
            <td>{noise_level}%</td>
            <td>NNBR</td>
            <td class="{coverage_class}">{coverage:.3f}</td>
            <td>{rmse:.4f}</td>
            <td>{mean_width:.4f}</td>
            <td>{r2:.3f}</td>
            <td><button class="view-btn">View Fit</button></td>
        </tr>'''
    all_rows.append(row)

print(f"\nGenerated {len(all_rows)} total table rows")

# Find the data-driven table tbody and replace content
dd_table_pattern = r'(<table id="resultsTableDD">.*?<thead>.*?</thead>\s*<tbody>)(.*?)(</tbody>)'
dd_match = re.search(dd_table_pattern, html, re.DOTALL)

if dd_match:
    table_start = dd_match.group(1)
    table_end = dd_match.group(3)

    # Create new tbody content
    new_tbody = '\n' + '\n'.join(all_rows) + '\n                    '
    new_table = table_start + new_tbody + table_end

    # Replace in HTML
    html = html[:dd_match.start()] + new_table + html[dd_match.end():]
    print("✓ Rebuilt data-driven table")
else:
    print("❌ Could not find data-driven table")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print(f"✓ Added {len(all_rows)} results ({len(gp_files)} GP + {len(nngmm_files)} NNGMM + {len(nnbr_files)} NNBR)")
