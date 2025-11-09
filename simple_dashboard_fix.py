#!/usr/bin/env python
"""Simple targeted fixes for dashboard - no complex HTML manipulation."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Backup
backup_path = dashboard_path.with_suffix('.html.bak_simple')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# ============================================================================
# FIX 1: Update data-driven table to filter out non-calibrated rows
# ============================================================================

# Simply replace table rows that contain ">Ensemble<" (not calibrated) with empty string
# This is safer than complex regex
lines = html.split('\n')
filtered_lines = []

skip_row = False
for i, line in enumerate(lines):
    # Check if this is a data row (contains <tr> but not <th>)
    if '<tr' in line and '<th>' not in line:
        # Look ahead to see if this row contains ">Ensemble<" without "Calibrated"
        # We need to check the next few lines
        row_content = line
        j = i + 1
        while j < len(lines) and '</tr>' not in row_content:
            row_content += lines[j]
            j += 1

        # If this row has ">Ensemble<" but NOT "Calibrated", skip it
        if '>Ensemble<' in row_content and 'Calibrated' not in row_content:
            # Skip all lines of this row
            skip_count = j - i
            skip_row = skip_count
            continue

    if skip_row > 0:
        skip_row -= 1
        continue

    filtered_lines.append(line)

html = '\n'.join(filtered_lines)
print("✓ Filtered data-driven table rows (removed uncalibrated Ensemble)")

# ============================================================================
# Save
# ============================================================================

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print("\nSimple fix applied - filtered out uncalibrated Ensemble rows from data-driven table")
print("\nPlease refresh your browser!")
