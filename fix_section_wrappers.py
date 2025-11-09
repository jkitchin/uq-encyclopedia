#!/usr/bin/env python
"""Fix missing section wrappers around Detailed Results sections."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read the dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_wrappers')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# ============================================================================
# FIX 1: Wrap nonlinear Detailed Results section
# ============================================================================

# Pattern: Find the h2 header that appears directly without a section wrapper
# Look for closing </div> followed by the h2, and wrap everything until the next main tab
nl_pattern = r'(</div>\s*</div>\s*\n\s*)(<h2>🔍 Detailed Results - Click Any Row to View Fit</h2>\s*<div class="filter-section">.*?)(</div>\s*<div id="main-tab-datadriven")'

def wrap_with_section(match):
    before = match.group(1)
    content = match.group(2)
    after = match.group(3)

    return f'{before}\n            <div class="section">\n                {content}\n            </div>\n\n        {after}'

html_new = re.sub(nl_pattern, wrap_with_section, html, flags=re.DOTALL)

if html_new != html:
    print("✓ Wrapped nonlinear Detailed Results section")
    html = html_new
else:
    print("⚠ Could not find nonlinear Detailed Results pattern")

# ============================================================================
# FIX 2: Wrap data-driven Detailed Results section
# ============================================================================

# Similar pattern for data-driven tab
dd_pattern = r'(</div>\s*</div>\s*\n\s*)(<h2>🔍 Detailed Results - Click Any Row to View Fit</h2>\s*<div class="filter-section">.*?</table>)'

def wrap_dd_section(match):
    before = match.group(1)
    content = match.group(2)

    return f'{before}\n            <div class="section">\n                {content}\n            </div>'

html_new = re.sub(dd_pattern, wrap_dd_section, html, flags=re.DOTALL)

if html_new != html:
    print("✓ Wrapped data-driven Detailed Results section")
    html = html_new
else:
    print("⚠ Could not find data-driven Detailed Results pattern")

# ============================================================================
# Save updated dashboard
# ============================================================================

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print("\nChanges made:")
print("  • Added proper <div class=\"section\"> wrapper around nonlinear Detailed Results")
print("  • Added proper <div class=\"section\"> wrapper around data-driven Detailed Results")
print("\nThis should fix the duplicate table appearance issue.")
print("Please refresh your browser to verify the fix!")
