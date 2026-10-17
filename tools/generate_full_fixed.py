#!/usr/bin/env python3
"""Generate the L0 inline full products and squares of at most 8 u64 limbs (MULX/ADCX/ADOX, BMI2 and ADX
at run time): the exact product of an na-limb and an nb-limb number into na+nb limbs, and the exact
square of n limbs into 2n limbs, for the S segment (fma's exact product, exact x exact short products).

src/core/x86_64/gen/mulfull.hpp, namespace sbn::v3::l0, inlinable (L0):
  mul_asm<na, 1>    na = 2..8 (one row): GNU extended asm, rows by b. Row j:
    rdx = b[j], MULX per a[i], ADCX chains the product highs, ADOX adds the old column (from r,
    the row's top column from the previous row's pending high in r8); each row leaves its high
    column pending in r8, the last one is stored. Per row the value is below B^(na+1), so the
    pending high plus the two carries fits a limb. Only caller-saved registers.
  mul_reg<na, nb>   2 <= nb <= na <= 8: the same rows with the row's columns in registers (POOL, at most BUDGET besides
    rdx): the lowest column is final after the row's first product and is stored, its register reused;
    na = 8 keeps the row's top column in r (read back by the next row's last product).
  sqr_asm<n>        n = 2..8: the cross triangle sum_{i<j} a_i a_j by the same rows (row i =
    a_i * a[i+1..n-1] at columns 2i+1..i+n-1), then one pass r = 2*cross + diag(a_i^2) with ADCX
    doubling each column with its carry (t + t + CF) and ADOX adding the square limbs; no operand
    is doubled (the structure of sqr_basecase_fixed.S, generate_mul_basecase.py).
  mul_u128<na, nb>, sqr_u128<n>   any sizes, portable C++ (not generated text), r restrict: column sums of the
    products' 64-bit halves in a u128 (at most 2*min(na, nb) terms < 2^64 per column); squares add
    the doubled cross column to the diagonal. The portable reference and the independent check.
  mul<na, nb>, sqr<n>   the defaults: either operand order (na < nb swaps); 1 x 1 and the square of 1
    limb u128, one row (nb = 1) mul_asm, nb >= 2 mul_reg, squares sqr_asm. Measured in the calling
    context (write + inline call + read back, both producer models; build/u52probe/l0full_2026-10-09):
    mul_reg wins from nb = 2 (8 x 8 0.92x of the memory rows, 1.17 c per limb product, p8), the memory
    row wins by 1-2 cycles at nb = 1, u128 is slower from 3 limbs and ties at 2. The memory rows for
    nb >= 2 are not kept (no shape where they win in both producer models).
Aliasing: r must not overlap a or b; a may equal b.
"""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
HPP = ROOT / 'src/core/x86_64/gen/mulfull.hpp'
N = 8
# (na, nb) and square sizes whose default is the u128 C++ (measured: l0full_2026-10-09)


class Emit:
    def __init__(self): self.lines = []
    def __call__(self, mnemonic, *args): self.lines.append(f'"{mnemonic} {", ".join(args)}\\n\\t"')


R = lambda r: '%%' + r
M = lambda base, off: f'{off}(%[{base}])'


def mul_rows(e, na, nb):
    H = (R('r9'), R('r10'))
    for j in range(nb):
        e('movq', M('b', 8 * j), R('rdx'))
        e('xorl', R('eax'), R('eax'))                # CF = OF = 0
        for i in range(na):
            h = H[i & 1]
            e('mulxq', M('a', 8 * i), R('rax'), h)
            if i: e('adcxq', H[(i - 1) & 1], R('rax'))
            if j: e('adoxq', R('r8') if i == na - 1 else M('r', 8 * (i + j)), R('rax'))
            e('movq', R('rax'), M('r', 8 * (i + j)))
        h = H[(na - 1) & 1]
        e('movl', '$0', R('eax'))                    # zero, flags kept
        if j: e('adoxq', R('rax'), h)
        if na > 1 or j: e('adcxq', R('rax'), h)
        e('movq', h, R('r8'))
    e('movq', R('r8'), M('r', 8 * (na + nb - 1)))


def sqr_rows(e, n):
    H = (R('r9'), R('r10'))
    for i in range(n - 1):                           # cross row i: columns 2i+1 .. i+n-1
        e('movq', M('a', 8 * i), R('rdx'))
        e('xorl', R('r11d'), R('r11d'))
        js = list(range(i + 1, n))
        for t, j in enumerate(js):
            p = i + j
            e('mulxq', M('a', 8 * j), R('rax'), H[t & 1])
            if t: e('adcxq', H[(t - 1) & 1], R('rax'))
            if i: e('adoxq', R('r8') if t == len(js) - 1 else M('r', 8 * p), R('rax'))
            e('movq', R('rax'), M('r', 8 * p))
        h = H[(len(js) - 1) & 1]
        if i: e('adoxq', R('r11'), h)
        e('adcxq', R('r11'), h)
        e('movq', h, R('r8'))
    e('movq', R('r8'), M('r', 8 * (2 * n - 2)))
    e('xorl', R('r11d'), R('r11d'))                  # r = 2*cross + diag
    for i in range(n):
        e('movq', M('a', 8 * i), R('rdx'))
        e('mulxq', R('rdx'), R('rax'), R('rcx'))
        for k, s in ((2 * i, R('rax')), (2 * i + 1, R('rcx'))):
            if k == 0: e('xorl', R('r9d'), R('r9d'))
            elif k == 2 * n - 1: e('movl', '$0', R('r9d'))   # keeps CF/OF
            else: e('movq', M('r', 8 * k), R('r9'))
            e('adcxq', R('r9'), R('r9'))
            e('adoxq', s, R('r9'))
            e('movq', R('r9'), M('r', 8 * k))


def asm_fn(name, tmpl, sig, lines, outs, ins, clob):
    return (f'template<> inline __attribute__((always_inline)) void {name}<{tmpl}>{sig} {{\n'
            '    __asm__ volatile(\n        ' + '\n        '.join(lines) + '\n'
            f'      : {outs}\n      : {ins}\n      : {clob});\n}}')


# Register window: the columns of the current row live in registers (at most BUDGET of POOL besides
# rdx, which leaves four general registers for the r/a/b pointers and the caller's addressing, without
# rbp); the lowest column is final after the row's first product and is stored, its register reused.
POOL = ('rax', 'rcx', 'r8', 'r9', 'r10', 'r11', 'rbx', 'r12', 'r13', 'r14')
BUDGET = 9                                           # registers besides rdx: 4 stay for pointers and addressing
r32 = lambda r: r + 'd' if r[1:].isdigit() else 'e' + r[1:]


def mul_rows_reg(e, na, nb):
    # A row peaks at na + 2 registers besides rdx; above BUDGET (na = 8) the row's top column goes to r
    # and the next row's last product takes it from there (one load and store per row).
    spill = na + 2 > BUDGET
    free, used = list(POOL[:BUDGET]), set()
    def take():
        r = free.pop(0); used.add(r); return r
    def give(r): free.insert(0, r)
    acc = {}                                         # column -> register
    def zero_flags():
        t = take(); e('xorl', R(r32(t)), R(r32(t))); give(t)
    for j in range(nb):
        e('movq', M('b', 8 * j), R('rdx'))
        zero_flags()                                 # CF = OF = 0
        prev = None
        for i in range(na):
            lo, h = take(), take()
            e('mulxq', M('a', 8 * i), R(lo), R(h))
            col = i + j
            if prev is not None: e('adcxq', R(prev), R(lo)); give(prev)
            if j == 0:
                if i == 0: e('movq', R(lo), M('r', 0)); give(lo)
                else: acc[col] = lo                  # row 0: the column is lo (+ previous high + CF)
            elif col not in acc:                     # the spilled top column of the previous row
                e('adoxq', M('r', 8 * col), R(lo)); acc[col] = lo
            else:
                e('adoxq', R(lo), R(acc[col])); give(lo)
                if i == 0: e('movq', R(acc[col]), M('r', 8 * col)); give(acc.pop(col))
            prev = h
        t = take()
        e('movl', '$0', R(r32(t)))                   # zero, flags kept
        if j: e('adoxq', R(t), R(prev))
        e('adcxq', R(t), R(prev))
        give(t)
        if spill: e('movq', R(prev), M('r', 8 * (j + na))); give(prev)
        else: acc[j + na] = prev
    for col in sorted(acc): e('movq', R(acc[col]), M('r', 8 * col))
    return [r for r in POOL if r in used]


def mul_reg(na, nb):
    e = Emit(); regs = mul_rows_reg(e, na, nb)
    return asm_fn('mul_reg', f'{na}, {nb}', '(uint64_t *r, const uint64_t *a, const uint64_t *b)', e.lines,
                  f'"=m"(*(uint64_t (*)[{na + nb}])r)',
                  f'[r] "r"(r), [a] "r"(a), "m"(*(const uint64_t (*)[{na}])a), [b] "r"(b), "m"(*(const uint64_t (*)[{nb}])b)',
                  ', '.join(f'"{r}"' for r in ['rdx'] + regs) + ', "cc"')


def mul_asm(na, nb):
    e = Emit(); mul_rows(e, na, nb)
    return asm_fn('mul_asm', f'{na}, {nb}', '(uint64_t *r, const uint64_t *a, const uint64_t *b)', e.lines,
                  f'"=m"(*(uint64_t (*)[{na + nb}])r)',
                  f'[r] "r"(r), [a] "r"(a), "m"(*(const uint64_t (*)[{na}])a), [b] "r"(b), "m"(*(const uint64_t (*)[{nb}])b)',
                  '"rax", "rdx", "r8", "r9", "r10", "cc"')


def sqr_asm(n):
    e = Emit(); sqr_rows(e, n)
    return asm_fn('sqr_asm', f'{n}', '(uint64_t *r, const uint64_t *a)', e.lines,
                  f'"=m"(*(uint64_t (*)[{2 * n}])r)', f'[r] "r"(r), [a] "r"(a), "m"(*(const uint64_t (*)[{n}])a)',
                  '"rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "cc"')


PORTABLE = '''// Portable column sums (any sizes): the reference and the u128 defaults. r is restrict (the contract:
// r does not overlap the inputs), so products and input loads may be reused across the stores to r;
// a and b may still be equal.
template<int na, int nb> inline __attribute__((always_inline)) void mul_u128(uint64_t *__restrict r, const uint64_t *a, const uint64_t *b) {
    unsigned __int128 acc = 0;
#pragma unroll
    for (int k = 0; k < na + nb; ++k) {
        unsigned __int128 hi = 0;                    // high halves of column k-1, low halves of column k
#pragma unroll
        for (int i = 0; i < na; ++i) {
            const int j = k - i;
            if (j >= 0 && j < nb) acc += (uint64_t)((unsigned __int128)a[i] * b[j]);
            if (j - 1 >= 0 && j - 1 < nb) hi += (uint64_t)(((unsigned __int128)a[i] * b[j - 1]) >> 64);
        }
        acc += hi;
        r[k] = (uint64_t)acc;
        acc >>= 64;
    }
}
template<int n> inline __attribute__((always_inline)) void sqr_u128(uint64_t *__restrict r, const uint64_t *a) {
    unsigned __int128 acc = 0;
#pragma unroll
    for (int k = 0; k < 2 * n; ++k) {
        unsigned __int128 cross = 0, diag = 0;
#pragma unroll
        for (int i = 0; i < n; ++i) {
            const int j = k - i, jh = k - 1 - i;
            if (j > i && j < n) cross += (uint64_t)((unsigned __int128)a[i] * a[j]);
            if (jh > i && jh < n) cross += (uint64_t)(((unsigned __int128)a[i] * a[jh]) >> 64);
            if (j == i) diag += (uint64_t)((unsigned __int128)a[i] * a[i]);
            if (jh == i) diag += (uint64_t)(((unsigned __int128)a[i] * a[i]) >> 64);
        }
        acc += 2 * cross + diag;
        r[k] = (uint64_t)acc;
        acc >>= 64;
    }
}'''


def header():
    out = ['// Generated by tools/generate_full_fixed.py; do not edit. L0 inline full products of at most 8',
           '// limbs: r[0..na+nb) = A*B and r[0..2n) = A^2 exactly (variants and defaults: see the generator).',
           '// r must not overlap a or b; a may equal b. The asm variants need BMI2 and ADX.',
           '#pragma once', '#include <stdint.h>', 'namespace sbn::v3::l0 {',
           'template<int na, int nb> void mul_asm(uint64_t *, const uint64_t *, const uint64_t *);',
           'template<int na, int nb> void mul_reg(uint64_t *, const uint64_t *, const uint64_t *);',
           'template<int n> void sqr_asm(uint64_t *, const uint64_t *);', PORTABLE]
    for na in range(1, N + 1):
        for nb in range(1, na + 1):
            if (na, nb) == (1, 1): continue
            out.append(mul_asm(na, nb) if nb == 1 else mul_reg(na, nb))
    for n in range(2, N + 1): out.append(sqr_asm(n))
    out += ['template<int na, int nb> inline __attribute__((always_inline)) void mul(uint64_t *r, const uint64_t *a, const uint64_t *b) {',
            f'    static_assert(na >= 1 && nb >= 1 && na <= {N} && nb <= {N});',
            '    if constexpr (na < nb) mul<nb, na>(r, b, a);',
            '    else if constexpr (na == 1) mul_u128<1, 1>(r, a, b);',
            '    else if constexpr (nb == 1) mul_asm<na, 1>(r, a, b);',
            '    else mul_reg<na, nb>(r, a, b);', '}',
            'template<int n> inline __attribute__((always_inline)) void sqr(uint64_t *r, const uint64_t *a) {',
            f'    static_assert(n >= 1 && n <= {N});',
            '    if constexpr (n == 1) sqr_u128<1>(r, a);',
            '    else sqr_asm<n>(r, a);', '}', '} // namespace sbn::v3::l0']
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    text = header()
    if args.check:
        if not HPP.exists() or HPP.read_text() != text: raise SystemExit(f'{HPP.relative_to(ROOT)} is stale')
        print(f'OK: L0 inline full products and squares, n <= {N}')
    else:
        HPP.write_text(text)


if __name__ == '__main__':
    main()
