#!/usr/bin/env python
"""Comprehensive cleanup to remove any duplicate sections in nonlinear and data-driven tabs."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read the dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_cleanup')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

changes_made = []

# ============================================================================
# STRATEGY: Extract each tab and ensure only ONE instance of each section
# ============================================================================

# Find tab boundaries
linear_start = html.find('<div id="main-tab-linear"')
nonlinear_start = html.find('<div id="main-tab-nonlinear"')
datadriven_start = html.find('<div id="main-tab-datadriven"')

# Extract sections
before_linear = html[:linear_start]
linear_section = html[linear_start:nonlinear_start]
nonlinear_section = html[nonlinear_start:datadriven_start]
datadriven_section = html[datadriven_start:]

# ============================================================================
# FIX 1: Clean nonlinear tab - ensure only ONE Detailed Results section
# ============================================================================

# Count occurrences of "Detailed Results" in nonlinear
detailed_count_nl = nonlinear_section.count('🔍 Detailed Results')
print(f"\nNonlinear tab: Found {detailed_count_nl} 'Detailed Results' header(s)")

if detailed_count_nl > 1:
    # Find all positions
    positions = []
    start_pos = 0
    while True:
        pos = nonlinear_section.find('🔍 Detailed Results', start_pos)
        if pos == -1:
            break
        positions.append(pos)
        start_pos = pos + 1

    print(f"  Positions: {positions}")

    # Keep only the LAST occurrence (most likely to be correct)
    # Remove all others
    for i in range(len(positions) - 1):
        # Find the section div that contains this occurrence
        pos = positions[i]
        # Find the <div class="section"> before this
        section_start = nonlinear_section.rfind('<div class="section">', 0, pos)
        if section_start != -1:
            # Find the corresponding closing </div>
            # We'll find where the next section starts
            next_section = nonlinear_section.find('<div class="section">', section_start + 1)
            if next_section != -1 and next_section < positions[-1]:
                # Remove this section
                nonlinear_section = nonlinear_section[:section_start] + nonlinear_section[next_section:]
                print(f"  Removed duplicate Detailed Results section at position {section_start}")
                changes_made.append("Removed duplicate Detailed Results from nonlinear tab")
                # Update positions for next iteration
                offset = next_section - section_start
                positions = [p - offset if p > section_start else p for p in positions]

# ============================================================================
# FIX 2: Clean data-driven tab - ensure only ONE Detailed Results section
# ============================================================================

detailed_count_dd = datadriven_section.count('🔍 Detailed Results')
print(f"\nData-driven tab: Found {detailed_count_dd} 'Detailed Results' header(s)")

if detailed_count_dd > 1:
    # Similar approach - keep the last one
    positions_dd = []
    start_pos = 0
    while True:
        pos = datadriven_section.find('🔍 Detailed Results', start_pos)
        if pos == -1:
            break
        positions_dd.append(pos)
        start_pos = pos + 1

    print(f"  Positions: {positions_dd}")

    for i in range(len(positions_dd) - 1):
        pos = positions_dd[i]
        section_start = datadriven_section.rfind('<div class="section">', 0, pos)
        if section_start != -1:
            next_section = datadriven_section.find('<div class="section">', section_start + 1)
            if next_section != -1 and next_section < positions_dd[-1]:
                datadriven_section = datadriven_section[:section_start] + datadriven_section[next_section:]
                print(f"  Removed duplicate Detailed Results section at position {section_start}")
                changes_made.append("Removed duplicate Detailed Results from data-driven tab")
                offset = next_section - section_start
                positions_dd = [p - offset if p > section_start else p for p in positions_dd]

# ============================================================================
# FIX 3: Remove any orphaned or malformed section divs
# ============================================================================

# Check for consecutive </div></div> patterns that might indicate structural issues
nl_before = nonlinear_section.count('</div>\n</div>')
nonlinear_section = re.sub(r'(</div>)\s*\n\s*(</div>)\s*\n\s*(<div class="section">)',
                           r'\1\n\2\n\n\3',
                           nonlinear_section)
nl_after = nonlinear_section.count('</div>\n</div>')

# ============================================================================
# FIX 4: Ensure proper section wrapper around all Detailed Results
# ============================================================================

# Make sure Detailed Results is properly wrapped
if '<h2>🔍 Detailed Results' in nonlinear_section:
    # Check if it's already wrapped
    if nonlinear_section.find('<div class="section">\n                <h2>🔍 Detailed Results') == -1:
        # Not properly wrapped - fix it
        nonlinear_section = re.sub(
            r'(<h2>🔍 Detailed Results.*?</table>)',
            r'<div class="section">\n                \1\n            </div>',
            nonlinear_section,
            flags=re.DOTALL
        )
        changes_made.append("Ensured Detailed Results is wrapped in section div (nonlinear)")

if '<h2>🔍 Detailed Results' in datadriven_section:
    if datadriven_section.find('<div class="section">\n            <h2>🔍 Detailed Results') == -1:
        datadriven_section = re.sub(
            r'(<h2>🔍 Detailed Results.*?</table>)',
            r'<div class="section">\n            \1\n        </div>',
            datadriven_section,
            flags=re.DOTALL
        )
        changes_made.append("Ensured Detailed Results is wrapped in section div (data-driven)")

# ============================================================================
# Reassemble and save
# ============================================================================

html_clean = before_linear + linear_section + nonlinear_section + datadriven_section

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html_clean)

print(f"\n✓ Dashboard updated: {dashboard_path}")

if changes_made:
    print("\nChanges made:")
    for change in changes_made:
        print(f"  • {change}")
else:
    print("\n  No structural duplicates found - HTML already clean")

print("\nDiagnostic summary:")
print(f"  • Nonlinear tab: {detailed_count_nl} Detailed Results section(s)")
print(f"  • Data-driven tab: {detailed_count_dd} Detailed Results section(s)")
print("\nPlease refresh your browser (hard refresh: Cmd+Shift+R) and check again!")
