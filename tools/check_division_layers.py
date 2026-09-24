#!/usr/bin/env python3
"""Guard the arithmetic/product boundary introduced by the division refactor."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
files = [
    'src/algorithms/reciprocal.hpp', 'src/algorithms/refinement.hpp',
    'src/algorithms/local_inverse.cpp', 'src/algorithms/local_divrem.cpp',
    'src/algorithms/divrem_service.cpp', 'src/algorithms/inverse_rung.cpp',
    'src/algorithms/divide_terminal.cpp',
]
forbidden = re.compile(r'pq16::|fixed_ntt_geometry|FixedNttGeometry|backend_lookup|'
                       r'SBN3_MUL_(?:PQ16|FLAT|BAILEY)|_mm(?:256|512)?_|__m(?:256|512)')
errors = []
for name in files:
    for line, text in enumerate((root / name).read_text().splitlines(), 1):
        if forbidden.search(text):
            errors.append(f'{name}:{line}: physical backend choice/intrinsic in arithmetic')
# The shared service also hosts the separately scoped legacy rsqrt policy.
# Check the inverse/quotient dispatch branch, not that unrelated selector.
name = 'src/algorithms/newton_planner_impl.hpp'
source = (root / name).read_text()
marker = 'if(kind==Cycle::Inverse||kind==Cycle::Division)'
start = source.index(marker)
first = source.index('{', start)
depth = 1
end = first + 1
while depth and end < len(source):
    depth += (source[end] == '{') - (source[end] == '}')
    end += 1
branch = source[first:end]
if depth or forbidden.search(branch) or 'product::compile_window_group' not in branch:
    errors.append(f'{name}: inverse/quotient must delegate physical selection to product')
for name in ['src/product/local_windows.hpp', 'src/product/window_group.hpp',
             'src/product/repeated_product.hpp', 'src/product/difference.hpp',
             'src/product/compact_windows.hpp', 'src/product/local_program.hpp']:
    if re.search(r'#include\s+"algorithms/', (root / name).read_text()):
        errors.append(f'{name}: product implementation depends on arithmetic implementation')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'division/product boundary: {len(files)} arithmetic units and product dispatch PASS')
