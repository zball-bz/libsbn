#!/usr/bin/env python3
"""Generate the scalar short high-half kernels (MULX/ADCX/ADOX, BMI2 and ADX at run time).

Contract, n u64 limbs: 0 <= floor(A*B / 2^(64n)) - r <= 1 (squares: B = A).
Kept terms: every product a_i b_j with i+j >= n-1, plus the high limb of each product with
i+j = n-2 (carried into column n-1). Exact geometric sum with B = 2^64 (n >= 2): the dropped
columns k <= n-3 hold k+1 products <= (B-1)^2, together
(B-1)^2 sum_{k=0..n-3} (k+1) B^k = (n-2) B^(n-1) - (n-1) B^(n-2) + 1, and the n-1 dropped low
limbs of column n-2 add at most (n-1)(B-1)B^(n-2); the omission is at most
(2n-3) B^(n-1) - 2(n-1) B^(n-2) + 1 < B^n. The kept sum is exact, so r = floor(kept / B^n) is
the exact high half or one less. n = 1 is the exact mulhi. A square keeps the same terms, so
every kernel here gives the same words as any other for equal values.

Rows: rdx = multiplier, MULX per product, ADCX chains the product highs, ADOX adds the old
column values; column n-1 stays in r11 and is never stored, the row's top column takes the
previous row's pending high (r8). Squares sum the cross triangle first, then one pass doubles
each column with ADCX (t + t + CF) and adds the diagonal with ADOX; no operand is doubled.
Only caller-saved registers.

src/core/x86_64/gen/mulhigh.hpp, namespace sbn::v3::l0, n = 1..8, inlinable (L0):
  mulhigh_asm<n>, sqrhigh_asm<n>   n = 3..8, GNU extended asm; operands are the exact r/a/b
    arrays, no memory clobber.
  mulhigh_u128<n>, sqrhigh_u128<n> n = 1..4, portable C++: column sums of 64-bit halves in a
    u128 (at most 2n-1 terms < 2^64 per column); squares double the cross column sum before
    the diagonal is added.
  mulhigh<n>, sqrhigh<n>: n <= 2 u128, n >= 3 asm (u128 is 3..10% slower at n = 3..4; u128
    n = 3..4 stays for portable builds and as an independent cross-check).
src/core/x86_64/high_fixed.S: sbn3i_mulhigh_<n>(rp, ap, bp) for n in MUL and
  sbn3i_sqrhigh_<n>(rp, ap) for n in SQR, out of line: register columns (below) to REGM / REGQ, the rows
  above beyond. The upper ends are the measured crossovers against the u52 kernels (write + call +
  readback, both producer models).
  One body per size, shared with the fp kernel (src/fp/x86_64/mul.S, tools/generate_fp_mul.py), whose
  dispatch jumps to the hidden sbn3i_mulhigh_fp_<n> / sbn3i_sqrhigh_fp_<n> (rsi a, rcx b, rdi r, its
  e_top and ns in the red zone): the body keeps the fp kernel's red-zone layout (FP_SLOTS; callee-saved
  column registers from FP_SAVE down) and ends in its exit (fp_exit: rax = e_top, rdx = ns from the red
  zone, a zero top limb strips it). The integer entry is a head right before the body (falls through):
  movq %rdx, %rcx for products, then ns = 1, so that a zero top limb leaves the strip at once (it reads
  no memory); rax and rdx are clobbered. The heads sit just below the 16-byte aligned bodies (padding
  before the head, not executed). Until 2026-10-17 both files carried a copy of every body (40 KB).
Aliasing: r must not overlap a or b; a may equal b.
"""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
HPP = ROOT / 'src/core/x86_64/gen/mulhigh.hpp'
ASM = ROOT / 'src/core/x86_64/high_fixed.S'
MUL, SQR, REGM, REGQ = tuple(range(5, 15)), tuple(range(5, 21)), 10, 11   # out of line; register columns to REGM / REGQ
# the fp kernel's red-zone slots (tools/generate_fp_mul.py and the shared bodies here): ns, e_top, the callee-saved column
# registers from FP_SAVE down (at most 6), the b and r pointers of the spilled register columns (products 9, 10, squares 11)
FP_SLOTS = (('S_NS', -8), ('S_ETOP', -16), ('S_B', -72), ('S_Z', -80))
FP_SAVE = -24


class Emit:
    """One instruction stream as .S lines or as GNU extended-asm strings."""
    def __init__(self, inline):
        self.inline, self.lines = inline, []

    def reg(self, r):
        return ('%%' if self.inline else '%') + r

    def mem(self, base, off):                       # base: r (output), a, b
        return f'{off}(%[{base}])' if self.inline else f'{off}(%{dict(r="rdi", a="rsi", b="rcx")[base]})'

    def __call__(self, mnemonic, *args):
        if self.inline: self.lines.append(f'"{mnemonic} {", ".join(args)}\\n\\t"')
        else: self.lines.append(f'    {mnemonic:<8}{", ".join(args)}')


def row(e, n, rdx_src, a_lo, terms, first, top_pending):
    # terms: (a index, column) of the full products in column order; a_lo: the a index whose
    # product contributes only its high limb (or None); first: no old column values
    H, R = (e.reg('r9'), e.reg('r10')), e.reg
    e('movq', rdx_src, R('rdx'))
    e('xorl', R('eax'), R('eax'))                  # CF = OF = 0
    t, have = 0, False
    if a_lo is not None:
        e('mulxq', e.mem('a', 8 * a_lo), R('rax'), H[0]); have, t = True, 1
    for idx, (ai, col) in enumerate(terms):
        h = H[t & 1]
        e('mulxq', e.mem('a', 8 * ai), R('rax'), h)
        if have: e('adcxq', H[(t - 1) & 1], R('rax'))
        if not first:
            if idx == len(terms) - 1 and top_pending: e('adoxq', R('r8'), R('rax'))
            elif col == n - 1: e('adoxq', R('r11'), R('rax'))
            else: e('adoxq', e.mem('r', 8 * (col - n)), R('rax'))
        e('movq', R('rax'), R('r11') if col == n - 1 else e.mem('r', 8 * (col - n)))
        have, t = True, t + 1
    h = H[(t - 1) & 1]
    e('movl', '$0', R('eax'))                       # zero, flags kept
    if not first: e('adoxq', R('rax'), h)
    e('adcxq', R('rax'), h)
    e('movq', h, R('r8'))


def mulhigh_rows(e, n):
    for j in range(n):
        lo = n - 2 - j
        row(e, n, e.mem('b', 8 * j), lo if lo >= 0 else None, [(i, i + j) for i in range(n - 1 - j, n)], j == 0, j > 0)
    e('movq', e.reg('r8'), e.mem('r', 8 * (n - 1)))  # column 2n-1


def sqrhigh_rows(e, n):
    R = e.reg
    for i in range(n - 1):                          # cross rows: a_i * a_j, j > i
        j0 = n - 2 - i
        row(e, n, e.mem('a', 8 * i), j0 if j0 > i else None, [(j, i + j) for j in range(max(i + 1, n - 1 - i), n)], i == 0, i > 0)
    # cross columns: n-1 in r11, n..2n-3 in r[0..n-3], 2n-2 in r8, 2n-1 zero
    e('xorl', R('eax'), R('eax'))
    for col in range(n - 1, 2 * n):
        i = col // 2
        if col == n - 1 or col % 2 == 0:            # a_i^2 when its first kept column comes
            e('movq', e.mem('a', 8 * i), R('rdx'))
            e('mulxq', R('rdx'), R('rax'), R('rcx'))
        sq = R('rax') if col % 2 == 0 else R('rcx')
        if col == n - 1: t = R('r11')
        elif col <= 2 * n - 3: t = R('r9'); e('movq', e.mem('r', 8 * (col - n)), t)
        elif col == 2 * n - 2: t = R('r8')
        else: t = R('r9'); e('movl', '$0', R('r9d'))
        e('adcxq', t, t)
        e('adoxq', sq, t)
        if col >= n: e('movq', t, e.mem('r', 8 * (col - n)))


# Register columns (n <= REGM / REGQ): columns n-1 .. 2n-1 live in registers for the whole product, so a
# product costs MULX, ADOX (low word) and ADCX (high word) and no column goes through memory. The
# same kept terms as the rows above, hence the same words. Leaf code: callee-saved column registers
# are saved in the red zone. (FLINT's mulhigh_hard.asm, Albin Ahlbaeck 2024, LGPL-3.0-or-later, uses
# the same register-column scheme; this is independent generator code.) Shared with
# tools/generate_fp_mul.py, which passes its own red-zone layout and zero constant.
CALLEE = ('%rbx', '%rbp', '%r12', '%r13', '%r14', '%r15')
XOR_MUL = 7     # product rows clear the flags from this n on (2026-10-12, fp mul: thr +1..+3 at n <= 6 with
                # the clears, lat equal; from 7 lat -4..-17, from 8 thr -1..-9)


def r32(r):
    return {'%rax': '%eax', '%rbx': '%ebx', '%rcx': '%ecx', '%rdx': '%edx', '%rsi': '%esi', '%rdi': '%edi',
            '%rbp': '%ebp'}.get(r, r + 'd')


def saves(regs, base):
    """red-zone saves of the callee-saved registers among regs at base, base - 8, ..."""
    used = [r for r in regs if r in CALLEE]
    return ([f'    movq    {r}, {base - 8 * i}(%rsp)' for i, r in enumerate(used)],
            [f'    movq    {base - 8 * i}(%rsp), {r}' for i, r in enumerate(used)])


def reg_row(L, load, hi_only, prods, top_col, col, zero, xor):
    """One row: load = the lines that set rdx = multiplier; hi_only = (a operand, column) of the
    hi-only product or None; prods = [(a operand, column of its low word)] ascending, the last one the
    row's top (its high word starts column top_col and takes both chains' carries). xor: clear CF / OF
    first (both are zero after the previous row); otherwise every row's first ADCX / ADOX waits for the
    previous row's last one and the two chains run through all rows."""
    L += load
    if xor:
        L.append('    xorl    %r8d, %r8d')
    if hi_only:
        src, c = hi_only
        L += [f'    mulxq   {src}, %r8, %r9', f'    adcxq   %r9, {col[c]}']
    for m, (src, c) in enumerate(prods):
        if m < len(prods) - 1:
            L += [f'    mulxq   {src}, %r8, %r9', f'    adoxq   %r8, {col[c]}', f'    adcxq   %r9, {col[c + 1]}']
        else:
            assert c + 1 == top_col
            L += [f'    mulxq   {src}, %r8, {col[c + 1]}', f'    adoxq   %r8, {col[c]}',
                  f'    adcxq   {zero}, {col[c + 1]}', f'    adoxq   {zero}, {col[c + 1]}']


def mul_regs(n, zero, save_base, slot_b, slot_z):
    """register-column high product, 2 <= n <= 10: a = rsi, b = rcx, r = rdi; returns (lines, the top
    column's register, the restore lines). n = 9, 10: the b and r pointers go to the red-zone slots
    and rcx, rdi hold columns n+2, n+3; rows 0..2 still read b through rcx (column n+2 is first
    written by row 2's top product, after its multiplier load). Copying the windows into a stack frame
    to free rsi for n = 11 was rejected (2026-10-12, fp mul: thr -6%, lat +18 cycles: the 32-byte
    copies read the previous call's 8-byte stores, no forwarding)."""
    spill = n + 1 > 3 + len(CALLEE)
    pool = ['%rax', '%r10', '%r11'] + (['%rcx', '%rdi'] if spill else []) + list(CALLEE)
    col = {n - 1 + i: pool[i] for i in range(n + 1)}
    sv, rs = saves(col.values(), save_base)
    L, zslot = list(sv), slot_z
    if spill:
        L += [f'    movq    %rcx, {slot_b}', f'    movq    %rdi, {slot_z}']
    a = lambda i: f'{8 * i}(%rsi)'
    load = lambda j: [f'    movq    {8 * j}(%rcx), %rdx'] if not spill or j <= 2 else \
        [f'    movq    {slot_b}, %rdx', f'    movq    {8 * j}(%rdx), %rdx']
    # row 0: hi(a_{n-2} b_0) and a_{n-1} b_0 start columns n-1, n
    L += load(0) + ['    xorl    %r8d, %r8d',
                    f'    mulxq   {a(n - 2)}, %r8, {col[n - 1]}', f'    mulxq   {a(n - 1)}, %r8, {col[n]}',
                    f'    adcxq   %r8, {col[n - 1]}', f'    adcxq   {zero}, {col[n]}']
    for j in range(1, n):
        i0 = n - 2 - j
        reg_row(L, load(j), (a(i0), n - 1) if i0 >= 0 else None,
                [(a(i), i + j) for i in range(n - 1 - j, n)], n + j, col, zero, n >= XOR_MUL)
    if spill:
        L += [f'    movq    {zslot}, %r9'] + [f'    movq    {col[n + i]}, {8 * i}(%r9)' for i in range(n)] + ['    movq    %r9, %rdi']
    else:
        L += [f'    movq    {col[n + i]}, {8 * i}(%rdi)' for i in range(n)]
    return L, col[2 * n - 1], rs


def sqr_regs(n, zero, save_base, slot_z):
    """register-column high square, 2 <= n <= 11: a = rsi, r = rdi; column 2n-1 ends in r9. n = 11:
    the r pointer goes to the red-zone slot and rdi holds a column (the stack-frame copy of the window
    for n = 12 was 2..3 cycles slower than the rows, 2026-10-12).
    Cross rows by the larger index j (multiplier a_j): products a_i a_j, max(0, n-1-j) <= i < j, and
    the hi-only a_{n-2-j} a_j. A row's new columns are zeroed by XOR (which also clears CF / OF),
    except its top column 2j, which the top product's high word starts: about n/2 rows (rows by the
    smaller index, n - 1 of them, were 2..3 cycles slower at n = 8, 9 in the fp kernel). Then one pass
    doubles each column (ADCX t, t) and adds the diagonal words (ADOX), following the last row's
    chains (CF = OF = 0 after its top; clearing the flags first was 3 cycles slower at n = 8)."""
    spill = n > 4 + len(CALLEE)
    pool = ['%rax', '%rcx', '%r10', '%r11'] + (['%rdi'] if spill else []) + list(CALLEE)
    col = {n - 1 + i: pool[i] for i in range(n)}
    sv, rs = saves(col.values(), save_base)
    L, zslot = list(sv), slot_z
    if spill:
        L += [f'    movq    %rdi, {slot_z}']
    a = lambda i: f'{8 * i}(%rsi)'
    top = n - 2
    for j in range(1, n):
        i0, h = max(0, n - 1 - j), n - 2 - j
        prods, has_h = list(range(i0, j)), 0 <= h < j
        if not prods and not has_h:
            continue
        last = 2 * j if prods else n - 1
        zeroed = [c for c in range(max(top + 1, n - 1), last + 1) if not (prods and c == last)]
        L.append(f'    movq    {a(j)}, %rdx')
        L += [f'    xorl    {r32(col[c])}, {r32(col[c])}' for c in zeroed] or ['    xorl    %r8d, %r8d']
        if has_h:
            L += [f'    mulxq   {a(h)}, %r8, %r9', f'    adcxq   %r9, {col[n - 1]}']
        for i in prods:
            if i < j - 1:
                L += [f'    mulxq   {a(i)}, %r8, %r9', f'    adoxq   %r8, {col[i + j]}', f'    adcxq   %r9, {col[i + j + 1]}']
            else:
                L += [f'    mulxq   {a(i)}, %r8, {col[2 * j]}', f'    adoxq   %r8, {col[2 * j - 1]}',
                      f'    adcxq   {zero}, {col[2 * j]}', f'    adoxq   {zero}, {col[2 * j]}']
        top = max(top, last)
    for c in range(n - 1, 2 * n - 1):                       # 2 cross + diagonal, columns n-1 .. 2n-2
        if c % 2 == 0 or c == n - 1:
            L += [f'    movq    {a(c // 2)}, %rdx', '    mulxq   %rdx, %r8, %r9']
        L += [f'    adcxq   {col[c]}, {col[c]}', f'    adoxq   {"%r8" if c % 2 == 0 else "%r9"}, {col[c]}']
    L += [f'    adcxq   {zero}, %r9', f'    adoxq   {zero}, %r9']   # column 2n-1: hi(a_{n-1}^2) + carries
    if spill:
        L += [f'    movq    {zslot}, %r8'] + [f'    movq    {col[n + i]}, {8 * i}(%r8)' for i in range(n - 1)] + \
            [f'    movq    %r9, {8 * (n - 1)}(%r8)', '    movq    %r8, %rdi']
    else:
        L += [f'    movq    {col[n + i]}, {8 * i}(%rdi)' for i in range(n - 1)] + [f'    movq    %r9, {8 * (n - 1)}(%rdi)']
    return L, '%r9', rs


def fp_defines():
    return [f'#define {name} {off}(%rsp)' for name, off in FP_SLOTS]


def fp_exit(top, restore, strip):
    """the fp kernel's exit after a high product: top = the register of its top limb; restore = the red-zone restores;
    rax = e_top, rdx = ns from the red zone; a zero top limb goes to strip (rax e, rdx ns)"""
    return [f'    testq   {top}, {top}'] + restore + ['    movq    S_ETOP, %rax', '    movq    S_NS, %rdx', f'    jz      {strip}',
                                                     '    ret']


def inline_asm(n, sq):
    e = Emit(True)
    (sqrhigh_rows if sq else mulhigh_rows)(e, n)
    o = f'      : "=m"(*(uint64_t (*)[{n}])r)\n      : [r] "r"(r), [a] "r"(a), "m"(*(const uint64_t (*)[{n}])a)'
    if not sq: o += f', [b] "r"(b), "m"(*(const uint64_t (*)[{n}])b)'
    clob = '"rax", "rdx", "r8", "r9", "r10", "r11", ' + ('"rcx", ' if sq else '') + '"cc"'
    name, sig = ('sqrhigh_asm', '(uint64_t *r, const uint64_t *a)') if sq else ('mulhigh_asm', '(uint64_t *r, const uint64_t *a, const uint64_t *b)')
    return (f'template<> inline __attribute__((always_inline)) void {name}<{n}>{sig} {{\n'
            '    __asm__ volatile(\n        ' + '\n        '.join(e.lines) + '\n' + o + f'\n      : {clob});\n}}')


def u128(n, sq):
    # kept products: i+j >= n-2 (high limb only at n-2); squares take i <= j, cross terms doubled
    L = []
    pairs = [(i, j) for i in range(n) for j in range(n) if i + j >= n - 2 and (not sq or i <= j)]
    for i, j in pairs:
        L.append(f'    const unsigned __int128 p{i}_{j} = (unsigned __int128)a[{i}] * {"a" if sq else "b"}[{j}];')
    L.append('    unsigned __int128 acc = 0;')
    for col in range(n - 1, 2 * n):
        lo = [(i, j) for (i, j) in pairs if i + j == col]
        hi = [(i, j) for (i, j) in pairs if i + j == col - 1]
        if sq:
            cross = [f'(uint64_t)p{i}_{j}' for (i, j) in lo if i < j] + [f'(uint64_t)(p{i}_{j} >> 64)' for (i, j) in hi if i < j]
            diag = [f'(uint64_t)p{i}_{j}' for (i, j) in lo if i == j] + [f'(uint64_t)(p{i}_{j} >> 64)' for (i, j) in hi if i == j]
            c = ' + '.join(f'(unsigned __int128){x}' for x in cross) or '0'
            d = ' + '.join(f'(unsigned __int128){x}' for x in diag) or '0'
            L.append(f'    acc += 2 * ({c}) + {d};')
        else:
            terms = [f'(unsigned __int128)(uint64_t)p{i}_{j}' for (i, j) in lo] + [f'(unsigned __int128)(uint64_t)(p{i}_{j} >> 64)' for (i, j) in hi]
            L.append(f'    acc += {" + ".join(terms) if terms else "0"};')
        if col >= n: L.append(f'    r[{col - n}] = (uint64_t)acc;')
        L.append('    acc >>= 64;')
    name = 'sqrhigh_u128' if sq else 'mulhigh_u128'
    sig = '(uint64_t *r, const uint64_t *a)' if sq else '(uint64_t *r, const uint64_t *a, const uint64_t *b)'
    return f'template<> inline __attribute__((always_inline)) void {name}<{n}>{sig} {{\n' + '\n'.join(L) + '\n}'


def header():
    out = ['// Generated by tools/generate_high_half.py; do not edit. L0 high-half kernels, n = 1..8:',
           '// 0 <= floor(A*B / 2^(64n)) - r <= 1 (proof, variants and defaults: see the generator).',
           '// r must not overlap a or b; a may equal b. The asm variants need BMI2 and ADX.',
           '#pragma once', '#include <stdint.h>', 'namespace sbn::v3::l0 {',
           'template<int n> void mulhigh_asm(uint64_t *, const uint64_t *, const uint64_t *);',
           'template<int n> void sqrhigh_asm(uint64_t *, const uint64_t *);',
           'template<int n> void mulhigh_u128(uint64_t *, const uint64_t *, const uint64_t *);',
           'template<int n> void sqrhigh_u128(uint64_t *, const uint64_t *);']
    for n in range(1, 5):
        out += [u128(n, False), u128(n, True)]
    for n in range(3, 9):
        out += [inline_asm(n, False), inline_asm(n, True)]
    out += ['template<int n> inline __attribute__((always_inline)) void mulhigh(uint64_t *r, const uint64_t *a, const uint64_t *b) {',
            '    static_assert(n >= 1 && n <= 8);',
            '    if constexpr (n <= 2) mulhigh_u128<n>(r, a, b); else mulhigh_asm<n>(r, a, b);',
            '}',
            'template<int n> inline __attribute__((always_inline)) void sqrhigh(uint64_t *r, const uint64_t *a) {',
            '    static_assert(n >= 1 && n <= 8);',
            '    if constexpr (n <= 2) sqrhigh_u128<n>(r, a); else sqrhigh_asm<n>(r, a);',
            '}',
            '} // namespace sbn::v3::l0']
    return '\n'.join(out) + '\n'


# the integer entries' heads: (line, its encoded bytes)
HEAD_B = ('    movq    %rdx, %rcx', 3)      # products: bp to rcx (rdx is the MULX source)
HEAD_NS = ('    movq    $1, S_NS', 9)        # ns = 1 for fp_exit's strip (REX.W C7 /0, disp8, imm32)


def fixed():
    out = [f'/* Generated by tools/generate_high_half.py; do not edit. sbn3i_mulhigh_<n>(rp, ap, bp),',
           f' * n = {MUL[0]}..{MUL[-1]}, and sbn3i_sqrhigh_<n>(rp, ap), n = {SQR[0]}..{SQR[-1]}: n limbs r with',
           ' * 0 <= floor(A*B / 2^(64n)) - r <= 1, rp disjoint from ap and bp; rax and rdx clobbered. Each body is',
           ' * also the fp kernel\'s (src/fp/x86_64/mul.S enters at the hidden sbn3i_mulhigh_fp_<n> / sbn3i_sqrhigh_fp_<n>);',
           ' * see the generator. */'] + fp_defines() + [
           '    .section .rodata', '    .p2align 3', '.Lhz:', '    .quad   0', '    .text']
    def fn(kind, n, head, body):
        name, fp = f'sbn3i_{kind}high_{n}', f'sbn3i_{kind}high_fp_{n}'
        lines, top, rs = body
        # the head ends where the 16-byte aligned body starts (the padding before the entry; the second .p2align adds
        # nothing then, and keeps the body aligned should the head's size ever differ)
        pad = -sum(b for _, b in head) % 16
        out.extend(['    .p2align 4', f'    .skip   {pad}, 0xcc', f'    .globl {name}', f'    .type {name}, @function', f'{name}:']
                   + [h for h, _ in head] + ['    .p2align 4', f'    .globl {fp}', f'    .hidden {fp}', f'    .type {fp}, @function', f'{fp}:']
                   + lines + fp_exit(top, rs, '.Lhstrip') + [f'    .size {fp}, .-{fp}', f'    .size {name}, .-{name}'])
    for n in MUL:
        if n <= REGM:
            body = mul_regs(n, '.Lhz(%rip)', FP_SAVE, 'S_B', 'S_Z')
        else:
            e = Emit(False)
            mulhigh_rows(e, n)                      # top limb in r8
            body = e.lines, '%r8', []
        fn('mul', n, [HEAD_B, HEAD_NS], body)
    for n in SQR:
        if n <= REGQ:
            body = sqr_regs(n, '.Lhz(%rip)', FP_SAVE, 'S_Z')
        else:
            e = Emit(False)
            sqrhigh_rows(e, n)                      # top limb in r9
            body = e.lines, '%r9', []
        fn('sqr', n, [HEAD_NS], body)
    out += ['.Lhstrip:                                   # fp_exit with a zero top limb: rax e, rdx ns (mul.S\'s .Lstrip)',
            '    movq    %rdx, %rcx', '    shlq    $1, %rcx', '    shrq    $1, %rcx',
            '1:  decq    %rax', '    decq    %rdx', '    decq    %rcx', '    jz      2f',
            '    cmpq    $0, -8(%rdi,%rcx,8)', '    je      1b', '    ret',
            '2:  xorl    %eax, %eax', '    xorl    %edx, %edx', '    ret',
            '    .section .note.GNU-stack,"",@progbits']
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    files = {HPP: header(), ASM: fixed()}
    if args.check:
        for path, text in files.items():
            if not path.exists() or path.read_text() != text:
                raise SystemExit(f'{path.relative_to(ROOT)} is stale')
        print(f'OK: high-half kernels L0 n = 1..8, scalar mul {MUL[0]}..{MUL[-1]}, sqr {SQR[0]}..{SQR[-1]}')
    else:
        HPP.parent.mkdir(exist_ok=True)
        for path, text in files.items():
            path.write_text(text)


if __name__ == '__main__':
    main()
