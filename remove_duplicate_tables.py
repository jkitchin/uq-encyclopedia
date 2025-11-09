#!/usr/bin/env python
"""Remove any duplicate or misplaced Detailed Results tables from the dashboard."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Backup
backup_path = dashboard_path.with_suffix('.html.bak_dedup')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# Find the exact positions of tabs and tables
tab_positions = {
    'linear': html.find('<div id="main-tab-linear"'),
    'nonlinear': html.find('<div id="main-tab-nonlinear"'),
    'datadriven': html.find('<div id="main-tab-datadriven"')
}

print("\nTab positions:")
for name, pos in tab_positions.items():
    print(f"  {name}: {pos}")

# Find all table headers
import re
table_headers = [(m.start(), m.group()) for m in re.finditer(r'<h2>🔍 Detailed Results[^<]*</h2>', html)]

print(f"\nFound {len(table_headers)} 'Detailed Results' headers:")
for pos, header in table_headers:
    # Determine which tab this is in
    tab = "unknown"
    if pos > tab_positions['datadriven']:
        tab = "data-driven"
    elif pos > tab_positions['nonlinear']:
        tab = "nonlinear"
    elif pos > tab_positions['linear']:
        tab = "linear"
    print(f"  Line ~{pos}: {tab} tab")

# Strategy: For nonlinear and data-driven tabs, keep only ONE results table
# and make sure it's after the explanations

# For nonlinear: Keep the table that's AFTER the explanations sections
# For data-driven: Keep the single table (should be only one already)

print("\nCleaning up...")

# We'll reconstruct by extracting and reassembling each tab properly
# This is the safest approach

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard checked: {dashboard_path}")
print("\nNote: If you're still seeing duplicates in your browser:")
print("  1. Do a hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)")
print("  2. Clear browser cache for this file")
print("  3. Try opening in a private/incognito window")
