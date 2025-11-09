#!/usr/bin/env python
"""Remove all debug labels from the dashboard."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_remove_debug')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# Pattern to match all debug labels (with various background colors)
# Matches: <p style="background: XXX; ...">...DEBUG...</p>
debug_patterns = [
    r'<p style="background: lightblue; padding: 5px; font-size: 0\.9em;">ℹ️ DEBUG:.*?</p>\n?',
    r'<p style="background: lightcoral; padding: 5px; font-size: 0\.9em;">ℹ️ DEBUG:.*?</p>\n?',
    r'<p style="background: lightgreen; padding: 5px; font-size: 0\.9em;">ℹ️ DEBUG:.*?</p>\n?',
    r'<p style="background: yellow; padding: 10px; font-weight: bold; border: 3px solid red;">⚠️ DEBUG:.*?</p>\n?',
]

count = 0
for pattern in debug_patterns:
    matches = re.findall(pattern, html, flags=re.DOTALL)
    count += len(matches)
    html = re.sub(pattern, '', html, flags=re.DOTALL)

print(f"✓ Removed {count} debug labels")

# Save updated dashboard
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✓ Dashboard updated: {dashboard_path}")
print("\nRefresh your browser to see the clean dashboard!")
