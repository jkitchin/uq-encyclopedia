#!/usr/bin/env python
"""Fix the extra closing div in the linear tab that breaks tab switching."""

from pathlib import Path

dashboard_path = Path('reports/dashboard/dashboard.html')

with open(dashboard_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Create backup
backup_path = dashboard_path.with_suffix('.html.bak_tabfix')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✓ Created backup: {backup_path}")

# The problem: After the linear table closes, there are 4 closing </div> tags
# but there should only be 3:
# 1. </div> closes the overflow-x div
# 2. </div> closes the section div
# 3. </div> closes main-tab-linear
#
# There's an extra </div> that's breaking the structure

# Find the problematic section
# Pattern: table closing, then 4 consecutive div closings before "End Linear Models Tab" comment
pattern = (
    r'(</table>\s*'
    r'</div>\s*'  # closes overflow-x
    r'</div>\s*'  # closes section
    r'</div>\s*'  # EXTRA - this is the problem
    r'</div>\s*'  # closes main-tab-linear
    r'<!-- End Linear Models Tab -->)'
)

replacement = (
    r'</table>\n'
    r'            </div>\n'  # closes overflow-x
    r'        </div>\n'      # closes section
    r'        </div>\n'      # closes main-tab-linear
    r'        <!-- End Linear Models Tab -->'
)

import re

# Let's be more surgical - find the exact location
# Search for the closing tags after </table> in the linear tab
linear_table_close = html.find('</table>', html.find('id="resultsTable"'))
print(f"\nLinear table closes at position: {linear_table_close}")

# Get the next 500 characters to see the structure
context = html[linear_table_close:linear_table_close+500]
print("\nContext after linear table closing:")
print(context[:300])

# Count the divs
divs_after = context[:200]
closing_div_count = divs_after.count('</div>')
print(f"\nClosing </div> tags in next 200 chars: {closing_div_count}")

# Strategy: Replace the specific pattern of 4 closing divs with 3
# between </table> and <!-- End Linear Models Tab -->

# Find from linear table close to comment
end_linear_comment_pos = html.find('<!-- End Linear Models Tab -->', linear_table_close)
section_to_fix = html[linear_table_close:end_linear_comment_pos]

print(f"\nSection from table close to comment:")
print(repr(section_to_fix))

# Now replace: We need to remove ONE of the 4 closing divs
# The pattern is roughly: </table>\n    </div>\n</div>\n    </div>\n        </div>
# We want: </table>\n    </div>\n</div>\n        </div>

# More specifically, looking at the actual structure:
# </table>
#             </div>      (closes overflow-x)
#         </div>          (closes section)
#     </div>              (REMOVE THIS - it's extra)
#         </div>          (closes main-tab-linear)

# Pattern to match
old_pattern = r'(</table>\s*</div>\s*</div>\s*)</div>(\s*</div>\s*<!-- End Linear Models Tab -->)'
new_pattern = r'\1\2'  # Remove the middle </div>

html_fixed = re.sub(old_pattern, new_pattern, html)

if html_fixed != html:
    print("\n✓ Successfully removed extra closing div")

    # Save
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_fixed)

    print(f"✓ Fixed dashboard saved: {dashboard_path}")

    # Verify the fix
    linear_start = html_fixed.find('<div id="main-tab-linear"')
    nonlinear_start = html_fixed.find('<div id="main-tab-nonlinear"')
    linear_section = html_fixed[linear_start:nonlinear_start]

    opening = linear_section.count('<div')
    closing = linear_section.count('</div>')
    balance = opening - closing

    print(f"\nPost-fix verification:")
    print(f"  Linear tab div balance: {balance} (should be 1)")

    if balance == 1:
        print("  ✓ Tab structure is now correct!")
    else:
        print(f"  ⚠️ Still has issues (balance = {balance})")
else:
    print("\n⚠️ Pattern did not match - manual inspection needed")
    print("Looking for alternative pattern...")

    # Try a more lenient pattern
    # Just look for 4 consecutive </div> tags before End Linear Models comment
    alt_pattern = r'(</table>.*?)(</div>\s*</div>\s*</div>\s*</div>)(\s*<!-- End Linear Models Tab -->)'

    def replace_four_divs(match):
        before = match.group(1)
        after = match.group(3)
        # Replace 4 divs with 3
        return before + '\n            </div>\n        </div>\n        </div>' + after

    html_fixed = re.sub(alt_pattern, replace_four_divs, html, flags=re.DOTALL)

    if html_fixed != html:
        print("✓ Alternative pattern worked!")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_fixed)
        print(f"✓ Fixed dashboard saved: {dashboard_path}")
    else:
        print("❌ No pattern matched - needs manual fix")
