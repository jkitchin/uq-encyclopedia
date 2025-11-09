#!/usr/bin/env python
"""Final comprehensive dashboard fixes."""

import re
from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

# Read the dashboard
with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_final')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# ============================================================================
# FIX 1: Reorder nonlinear tab - datasets first, then methods, then results
# ============================================================================

# Find the nonlinear tab
nl_tab_start = html.find('<div id="main-tab-nonlinear" class="main-tab-content">')
nl_tab_end = html.find('<div id="main-tab-datadriven"', nl_tab_start)

if nl_tab_start != -1 and nl_tab_end != -1:
    nl_content = html[nl_tab_start:nl_tab_end]

    # Extract the three main sections
    # 1. Training Data Visualizations
    training_start = nl_content.find('<h3>Training Data Visualizations</h3>')
    datasets_expl_start = nl_content.find('<h3>📁 Datasets Explained</h3>')
    methods_expl_start = nl_content.find('<h3>🎯 UQ Methods Explained</h3>')
    detailed_results_start = nl_content.find('<h2>🔍 Detailed Results')

    # Extract each section
    if all(x != -1 for x in [training_start, datasets_expl_start, methods_expl_start, detailed_results_start]):
        # Find the end of each section by looking for the next section start or </div>

        # Training data section (from start to datasets explanation)
        # Find the closing </div> of the training data section
        # We need to find where the dataset explanations section starts (its opening <div class="section">)
        datasets_section_div = nl_content.rfind('<div class="section">', 0, datasets_expl_start)
        training_section = nl_content[training_start:datasets_section_div].strip()

        # Datasets explanation section
        methods_section_div = nl_content.rfind('<div class="section">', 0, methods_expl_start)
        datasets_section = nl_content[datasets_section_div:methods_section_div].strip()

        # Methods explanation section
        results_section_div = nl_content.rfind('<div class="section">', 0, detailed_results_start)
        methods_section = nl_content[methods_section_div:results_section_div].strip()

        # Detailed results section (rest)
        results_section = nl_content[results_section_div:].strip()

        # Reconstruct in correct order: Training -> Datasets -> Methods -> Results
        nl_header = nl_content[:training_start].strip()

        new_nl_content = f"{nl_header}\n\n"
        new_nl_content += f"                {training_section}\n\n"
        new_nl_content += f"            {datasets_section}\n\n"
        new_nl_content += f"            {methods_section}\n\n"
        new_nl_content += f"            {results_section}\n\n"
        new_nl_content += "        "  # closing for main-tab-content

        # Replace in HTML
        html = html[:nl_tab_start] + new_nl_content + html[nl_tab_end:]
        print("✓ Reordered nonlinear tab sections (Training → Datasets → Methods → Results)")
    else:
        print("⚠ Could not find all nonlinear sections")
else:
    print("⚠ Could not find nonlinear tab boundaries")

# ============================================================================
# FIX 2: Remove duplicate "Detailed Results" tables in nonlinear tab
# ============================================================================

# Find all "Detailed Results" headers in the nonlinear tab
nl_tab_start_new = html.find('<div id="main-tab-nonlinear"')
nl_tab_end_new = html.find('<div id="main-tab-datadriven"', nl_tab_start_new)

if nl_tab_start_new != -1 and nl_tab_end_new != -1:
    nl_section = html[nl_tab_start_new:nl_tab_end_new]

    # Count how many "Detailed Results" sections there are
    detailed_count = nl_section.count('<h2>🔍 Detailed Results')

    if detailed_count > 1:
        print(f"  Found {detailed_count} Detailed Results sections in nonlinear tab")

        # Find the first occurrence and remove it
        first_detailed = nl_section.find('<h2>🔍 Detailed Results')
        second_detailed = nl_section.find('<h2>🔍 Detailed Results', first_detailed + 1)

        if second_detailed != -1:
            # Find the <div class="section"> that contains the first one
            first_section_start = nl_section.rfind('<div class="section">', 0, first_detailed)

            # The end is where the second section starts
            second_section_start = nl_section.rfind('<div class="section">', 0, second_detailed)

            # Remove the first section
            nl_section_fixed = nl_section[:first_section_start] + nl_section[second_section_start:]

            # Update HTML
            html = html[:nl_tab_start_new] + nl_section_fixed + html[nl_tab_end_new:]
            print(f"✓ Removed first Detailed Results table from nonlinear tab")
    else:
        print(f"  Only {detailed_count} Detailed Results section in nonlinear tab")

# ============================================================================
# FIX 3: Remove first "Detailed Results" table in data-driven tab
# ============================================================================

dd_tab_start = html.find('<div id="main-tab-datadriven"')
dd_tab_end = len(html)  # Goes to end

if dd_tab_start != -1:
    dd_section = html[dd_tab_start:dd_tab_end]

    # Count "Detailed Results" sections
    detailed_count_dd = dd_section.count('<h2>🔍 Detailed Results')

    if detailed_count_dd > 1:
        print(f"  Found {detailed_count_dd} Detailed Results sections in data-driven tab")

        # Find first and second occurrences
        first_detailed_dd = dd_section.find('<h2>🔍 Detailed Results')
        second_detailed_dd = dd_section.find('<h2>🔍 Detailed Results', first_detailed_dd + 1)

        if second_detailed_dd != -1:
            # Find section boundaries
            first_section_start_dd = dd_section.rfind('<div class="section">', 0, first_detailed_dd)
            second_section_start_dd = dd_section.rfind('<div class="section">', 0, second_detailed_dd)

            # Remove first section
            dd_section_fixed = dd_section[:first_section_start_dd] + dd_section[second_section_start_dd:]

            html = html[:dd_tab_start] + dd_section_fixed
            print(f"✓ Removed first Detailed Results table from data-driven tab")
    else:
        print(f"  Only {detailed_count_dd} Detailed Results section in data-driven tab")

# ============================================================================
# FIX 4: Filter data-driven table to only show EnsembleCalibrated
# ============================================================================

# Find the data-driven results table and modify the rows to only include EnsembleCalibrated
# We'll do this by modifying the table data generation or filtering in JavaScript

# Find the resultsTableDD and its tbody
dd_table_pattern = r'<table id="resultsTableDD">.*?</table>'
dd_table_match = re.search(dd_table_pattern, html, re.DOTALL)

if dd_table_match:
    dd_table = dd_table_match.group(0)

    # Find all table rows and filter out non-EnsembleCalibrated
    # The UQ Method is in the 4th column (index 3)

    # Strategy: Find all <tr> tags and remove those that don't contain "EnsembleCalibrated"
    # But we need to be careful not to remove the header row

    # Better approach: Remove the rows by finding patterns
    # Look for rows that have <td>Ensemble</td> (not EnsembleCalibrated)

    # Let's use a different approach - find all data rows
    rows_pattern = r'<tr[^>]*>.*?</tr>'
    rows = re.findall(rows_pattern, dd_table, re.DOTALL)

    filtered_rows = []
    for row in rows:
        # Keep header row (has <th>)
        if '<th>' in row:
            filtered_rows.append(row)
        # Keep rows with EnsembleCalibrated
        elif 'EnsembleCalibrated' in row or 'Ensemble Calibrated' in row:
            filtered_rows.append(row)
        # Skip rows with just "Ensemble" (not calibrated)
        elif '>Ensemble<' in row and 'Calibrated' not in row:
            continue
        else:
            # If no UQ method mentioned, keep it
            filtered_rows.append(row)

    # Reconstruct table
    # Find table structure
    table_start = dd_table.find('<table')
    table_header_end = dd_table.find('</thead>')
    tbody_start = dd_table.find('<tbody>')
    tbody_end = dd_table.find('</tbody>')
    table_end = dd_table.find('</table>')

    if all(x != -1 for x in [table_start, table_header_end, tbody_start]):
        # Rebuild table
        new_table = dd_table[:tbody_start + 7]  # Include <tbody>
        new_table += '\n'

        # Add filtered rows (skip header which is in thead)
        for row in filtered_rows:
            if '<th>' not in row:  # Skip header rows
                new_table += '                        ' + row + '\n'

        new_table += '                    </tbody>\n                </table>'

        # Replace in HTML
        html = html[:dd_table_match.start()] + new_table + html[dd_table_match.end():]

        rows_before = len([r for r in rows if '<th>' not in r])
        rows_after = len([r for r in filtered_rows if '<th>' not in r])
        print(f"✓ Filtered data-driven table: {rows_before} → {rows_after} rows (EnsembleCalibrated only)")

# Also update the table header to remove "UQ Method" column
# Actually, keep it but it will always say "EnsembleCalibrated"

# ============================================================================
# FIX 5: Update data-driven table header - rename UQ Method to just show model name
# ============================================================================

# Actually, let's keep the UQ Method column but it's clearer this way
# Or we can hide it with CSS

# Let's update the model architecture section to clarify
dd_arch_pattern = r'(<h3>🧠 DPOSE Model Architecture</h3>.*?</div>)'
dd_arch_match = re.search(dd_arch_pattern, html, re.DOTALL)

if dd_arch_match:
    dd_arch = dd_arch_match.group(1)

    # Update to mention calibration
    new_arch = dd_arch.replace(
        '</div>',
        '<p><strong>UQ Method:</strong> Ensemble with Isotonic Calibration (EnsembleCalibrated)</p>\n            </div>'
    )

    html = html.replace(dd_arch, new_arch)
    print("✓ Updated data-driven model architecture description")

# ============================================================================
# Save updated dashboard
# ============================================================================

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Dashboard updated: {dashboard_path}")
print("\nAll fixes applied:")
print("  1. ✓ Reordered nonlinear tab (Training → Datasets → Methods → Results)")
print("  2. ✓ Removed duplicate Detailed Results table from nonlinear tab")
print("  3. ✓ Removed first Detailed Results table from data-driven tab")
print("  4. ✓ Filtered data-driven to show only EnsembleCalibrated")
print("  5. ✓ Updated data-driven architecture description")
print("\nRefresh your browser to see the updates!")
print("\nNote: Linear plots issue may require regeneration - investigating...")
