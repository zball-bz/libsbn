#!/usr/bin/env python3
"""Generate src/fp/x86_64/rsqrt_rsp.S: the P-first scalar part of the fp reciprocal square root (contract: src/fp/kernel.hpp):
sbn3f_rsqrt_rsp_k7, the public k 7..10 and k 6 at s 61..63 (tail jumps from sbn3f_rsqrt, src/fp/x86_64/rsqrt.S), and
sbn3f_rsqrt_rsp_seed, the IFMA side's seeds of ns 7 / 8 limbs (tail jump from sbn3f_rsqrt_seed, rsqrt.S).
  usage: generate_fp_rsqrt_rsp.py [--check] [OUT.S] [options]; without options it writes the shipped file (SHIPPED below) to
  OUT (default src/fp/x86_64/rsqrt_rsp.S); --check generates in memory and compares with OUT (exit status 1 if they differ).

The shipped file is the research kernel G's rsp.S (experiments/results/fpk_2026-10-12/rsqrt_s4x/joint/x512g_claude,
OVH2-RSQRT, accepted by the owner 2026-10-16) with the library's symbol names: its machine code is rsp.o's. This file is that
snapshot's source/gen/gen_rsp_sl.py with the library's defaults (SHIPPED, the output path, the first header lines, the helper
tools/fp_rsqrt/gen_r2_hybrid.py) and one change that keeps every result: valid_c evaluates the bound zf only at the smallest
and the largest s of a class (zf's two ends are linear in q = 2^s / B^(n-k+1), which grows with s, so their minimum and
maximum over the class are attained there); the full scan took 7 s, the generator now runs in well under a second. The
route table (--table) and the options other than SHIPPED are research outputs and variants, not built by the library.

SLIM-RSQRT copy (Claude, 2026-10-16) of claude_scalar/proto/gen/gen_rsp.py: without options the output is the original's byte
for byte; --no-b47: public k 7 and k 8 s 0 run the 4 -> 8 body (quadratic, one shared bias; the 4 -> 7 body is not emitted);
--seed: the seed entry PREFIXrsp_seed (fpkrs_seed's interface for ns 7 / 8) on the 4 -> 8 quadratic body (see generate()).
SCALAR-PROTO (joint rsqrt, Claude, 2026-10-15). LGPL-3.0-or-later.
Generator of Claude's inv-style scalar rsqrt path for public k 7..10 (DESIGN.md, narrow variant), entered by a tail jump
from Codex's public entry with the IFMA hand-off registers (rdi z, rsi raw limbs, r8 nb, r9 e_top, rcx k, r10 c, r11 s).

  usage: gen_rsp.py OUT.S [--prefix fpkrs_] [--trace] [--table OUT.txt]

The path. Decode N = k + 2 limbs of x 2^c (Codex's rolling decode, copied), Codex's hybrid R2 (gen_r2_hybrid.emit_core,
copied; its three output stores dropped, the outputs stay in rax / r10 / r11), then one register step 2 -> 4 (mid) and
one final register step 4 -> n chosen per (k, s class):
  k 7      : 4 -> 7  (s 0: copy z = Y[1..7]; s >= 1: pack)
  k 8, s 0 : 4 -> 7  (copy z = Y[0..7])
  k 8      : 4 -> 8 quadratic for s 1 .. THR8, cubic 4 => 8 above (pack)
  k 9, s 0 : 4 => 8  (copy z = Y[0..8])
  k 9      : 4 => 10 (pack from limb 1: z = floor(Y 2^s / B^2))
  k 10     : 4 => 10 (s 0: copy z = Y[1..10]; s >= 1: pack)
Y = X_n + c (c per class, below), z = floor(Y 2^s / B^o), o = n + 1 - k, clamped to B^k - 1 (only possible at s = 63).

The step m -> n (one shared emitter, data per (m, n); "P-first": inv's register step with the divisor replaced by
P = t R_m). B = 2^64, R_m = B^m + a (a_j = R's fraction limb j, top first; F[i] = a_(m-i) from the bottom), t = D / B^N
(d_i = D's limb i, top first, zero below N), cbot = n + 2, W = n - m + 3, w = n - m:
  P stage  columns 1 .. cbot (weight B^-c) in registers: columns >= m start as d_c, columns < m as 0; rows j = m .. 1
           add a_j d_i (i + j <= cbot: lo at column i + j, hi at i + j - 1); each row's two carry chains end in column
           j - 1, still zero (inv's correction tail); then d_(m-1) .. d_1 are added into columns m-1 .. 1. P~ < 1, so the
           sum mod B^cbot is exact. Columns m .. n+1 are copied for the MP rows: into the spare registers (all four in
           2 -> 4, three in 4 -> 7, two in 4 -> 8, none in 4 => 10), the rest stored; columns < m stay register operands.
  MP       inv's window on P: Acc[t] = column n + 2 - t (P's own registers, the a_0 = 1 row), t = 0 .. W-1; rows jj = m .. 1
           add p_i a_jj at position n + 2 - i - jj (mod B^W).
  G        = ~Acc[2 .. W-1] (w + 1 limbs); G_h = G >> 1 (shrd chain).
  cubic    (4 => 8 behind the CUBF flag, 4 => 10 always): Gt = G_h's top q = n - 2m + 2 limbs, S = Gt^2 (exact),
           T = S + (S >> 1) (shrx / shlx into two carry chains), K = T's limbs cut .. 2q-1 (cut = 2m + 2q - n - 2),
           G'' = G_h + K. 4 => 10 keeps G_h / G'' in memory (registers: Cr 9 + G 7 > 12).
  corr     inv's: Cr = sum_{i + j >= m-2} F_i G''_j B^(i + j - (m-2)) (two carry columns), C = Cr[2 ..] + G''.
  update   X_n = R_m B^w + C (leading limb in rbx). Mid (2 -> 4): -2, clamp up to B^4, store X_4's fraction, jump to the
           final body. Final: + c, then copy (s = 0) or pack.
Error (exact - X in units B^-n, before the bias; DESIGN.md section 5 with Delta_2 < 2.43 from Codex's hybrid R2 window,
which holds against the full decoded t, HYBRID.md; strictly > 0): quadratic (-eps, 3 + 1.5 D^2 B^(n-2m) + eps), cubic
(-eps, 5 + 2.5 D^3 B^(n-3m) + 3 D B^(n-2m+1-q) + eps); the mid step subtracts 2 (Delta_4 in (2 - eps, hi + 2)). This
generator recomputes every class's bias and the k 8 threshold from these bounds and asserts |z - f| < 4 (table rules).
Original code; the copied R2 (gen_r2_hybrid.py, Codex, LGPL-3.0-or-later) and decode keep their origin (scalar-v3,
joint/j12_codex). No library file is read or written.
"""
import argparse
import sys
from fractions import Fraction as Fr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tools/fp_rsqrt'
sys.path.insert(0, str(HERE))
import gen_r2_hybrid as r2h  # noqa: E402  Codex's hybrid R2 (copy of j12_codex/source/gen/base/seed/gen_r2_hybrid.py)
SHIPPED = ['--prefix', 'sbn3f_rsqrt_', '--seed', '--r2v3', '--seed47', '--k6hi']     # G's RSPOPT, R2 v3 and RSPX
ARGV = sys.argv[1:]
CHECK = '--check' in ARGV
ARGV = [x for x in ARGV if x != '--check']
LIB = not any(x.startswith('-') for x in ARGV)
if LIB:
    ARGV = (ARGV or [str(ROOT / 'src/fp/x86_64/rsqrt_rsp.S')]) + SHIPPED
NO_B47 = '--no-b47' in ARGV   # SLIM-RSQRT: k 7 and k 8 s 0 through 4 -> 8 (no 4 -> 7 body)
R2V2 = '--r2v2' in ARGV     # FIX-RSQRT: R2 v2 (gen_r2_hybrid.emit_core_v2, top limb in r14 from the decode)
R2V3 = '--r2v3' in ARGV     # FIX-RSQRT: R2 v3 (gen_r2_hybrid.emit_core_v3: xmm1 = y1 from the caller's emit_fp_v3; no decode change)
SEED = '--seed' in ARGV       # SLIM-RSQRT: the seed entry PREFIXrsp_seed (PREFIXseed's interface, ns 7 / 8, 4 -> 8 quadratic)
K6HI = '--k6hi' in ARGV       # OVH2-RSQRT: public k 6 at s 61..63 (from rsqrt.S) through 2 -> 4 -> 7, pack from limb 2
SEED47 = '--seed47' in ARGV   # OVH2-RSQRT: ns 7 seeds through the 4 -> 7 body (its copy exit, o = 0; its bias C47), ns 8 as before

POOL = ['rcx', 'rsi', 'rdi', 'rbp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
R32 = {'rax': 'eax', 'rbx': 'ebx', 'rcx': 'ecx', 'rdx': 'edx', 'rsi': 'esi', 'rdi': 'edi', 'rbp': 'ebp'}


def r32(r):
    return R32.get(r, r + 'd')


# frame (rsp-relative)
Z, ET, KK, PP, BODY, CUBF, BIAS, R2S0, R2S1 = 0, 8, 16, 24, 32, 40, 48, 56, 64
XF = 72                        # X_4's fraction limbs F[0..3], F[0] lowest
PS = 104                       # P's column c (1 .. 11) at PS + 8 (c - 1)
GS = 192                       # G_h / G'' of 4 => 10 (7 limbs)
DB = 248                       # d_12: zero (k 9's 4 => 10 reads it), d_i at DT - 8 (i - 1)
DT = DB + 88
FRAME = DT + 8
assert XF + 32 == PS and PS + 88 == GS and GS + 56 == DB and FRAME % 16 == 8

# ---------------------------------------------------------------------------------------------------------------------
# error model and route table (all assertions are table rules)
BF = Fr(2) ** 64
EPS = Fr(1, 10 ** 12)
D2 = Fr(243, 100)                  # Codex's hybrid R2: 0 < exact_2(t_D) - A < 2.43 (HYBRID.md; lower margin > 0.577)
BODIES = {'b24': (2, 4, 'none'), 'b47': (4, 7, 'none'), 'b48': (4, 8, 'flag'), 'b410': (4, 10, 'always')}


def level(cub, dbase, m, n):
    """exact_n - X_n < hi after one step from 0 < Delta_m < dbase (DESIGN.md section 5)"""
    if not cub:
        return 3 + Fr(3, 2) * dbase * dbase * BF ** (n - 2 * m) + EPS
    q = n - 2 * m + 2
    return 5 + Fr(5, 2) * dbase ** 3 * BF ** (n - 3 * m) + 3 * dbase * BF ** (n - 2 * m + 1 - q) + EPS


HI4 = level(False, D2, 2, 4)
D4 = HI4 + 2 + Fr(1, 100)          # after -2: 2 - eps < Delta_4 < HI4 + 2


def zf(hi, c, s, n, k):
    """z - f in ((c - hi) q - 1, c q] (+ eps for the truncated radicand), q = 2^s / B^(n - k + 1)"""
    q = Fr(2) ** s / BF ** (n - k + 1)
    return (c - hi) * q - 1 - EPS, c * q + EPS


def final_of(k, s, thr8):
    """(body, cubic) of the final step for public (k, s)"""
    if K6HI and k == 6:                         # OVH2-RSQRT: k 6, s 61..63 only (the scalar entry sends no other s here)
        assert 61 <= s <= 63
        return 'b47', False
    if k == 7:
        return ('b48', False) if NO_B47 else ('b47', False)       # SLIM-RSQRT --no-b47: 4 -> 8, pack from limb 2
    if k == 8:
        if s == 0:
            return ('b48', False) if NO_B47 else ('b47', False)   # SLIM-RSQRT --no-b47: 4 -> 8, copy from limb 1
        return ('b48', False) if s <= thr8 else ('b48', True)
    if k == 9:
        return ('b48', True) if s == 0 else ('b410', True)
    if k == 10:
        return 'b410', True
    raise ValueError(k)


def valid_c(k, ss, body, cub):
    m, n, _ = BODIES[body]
    hi = level(cub, D4, m, n)
    out = {}
    ends = sorted({min(ss), max(ss)})              # zf is linear in q(s), q increasing: its extremes over ss are at the ends
    for c in range(0, 600):
        lo_hi = [zf(hi, c, s, n, k) for s in ends]
        zl, zh = min(x[0] for x in lo_hi), max(x[1] for x in lo_hi)
        if zl > -4 and zh < 4:
            out[c] = (zl, zh, min(zl + 4, 4 - zh))
    return hi, out


def best_c(cands, hi):
    """the largest margin, then c nearest hi / 2"""
    return max(cands, key=lambda c: (round(float(cands[c][2]), 9), -abs(c - hi / 2)))


def build_table():
    # k 8: the largest s threshold for the quadratic 4 -> 8 with one bias
    thr8 = 0
    for t in range(1, 64):
        if valid_c(8, range(1, t + 1), 'b48', False)[1]:
            thr8 = t
    assert thr8 >= 1
    classes = []
    for k in range(6 if K6HI else 7, 11):
        cur = None
        for s in (range(61, 64) if k == 6 else range(64)):
            body, cub = final_of(k, s, thr8)
            key = (body, cub, s == 0)
            if cur and cur['key'] == key:
                cur['ss'].append(s)
            else:
                cur = {'k': k, 'key': key, 'body': body, 'cub': cub, 'ss': [s]}
                classes.append(cur)
    for cl in classes:
        cl['hi'], cl['cands'] = valid_c(cl['k'], cl['ss'], cl['body'], cl['cub'])
        assert cl['cands'], cl
    # bias per body: an immediate where the body's classes share one c (b47, b410), data (BIAS slot) for b48
    bias = {}
    groups = [('b47', lambda cl: cl['body'] == 'b47'), ('b410', lambda cl: cl['body'] == 'b410')]
    if NO_B47:      # SLIM-RSQRT: k 7, k 8 s 0 .. THR8 share b48's quadratic bias (one data value per flag, as the cubic classes)
        groups = [('b410', groups[1][1]), ('b48q', lambda cl: cl['body'] == 'b48' and not cl['cub'])]
    for b, sel in groups:
        cls = [cl for cl in classes if sel(cl)]
        common = set(cls[0]['cands'])
        for cl in cls[1:]:
            common &= set(cl['cands'])
        assert common, b
        sub = {c: (0, 0, min(cl['cands'][c][2] for cl in cls)) for c in common}
        bias[b] = best_c(sub, max(cl['hi'] for cl in cls))
        for cl in cls:
            cl['c'] = bias[b]
    for cl in classes:
        if cl['body'] == 'b48' and 'c' not in cl:
            cl['c'] = best_c(cl['cands'], cl['hi'])
    # b48's cubic classes use the same c (one data value per flag)
    assert len({cl['c'] for cl in classes if cl['body'] == 'b48' and cl['cub']}) == 1
    for cl in classes:
        m, n, _ = BODIES[cl['body']]
        cl['n'] = n
        cl['o'] = n + 1 - cl['k']
        cl['exit'] = 'copy' if cl['ss'] == [0] else 'pack'
        assert (cl['exit'] == 'copy') == (0 in cl['ss']) and (cl['exit'] == 'pack') == (0 not in cl['ss'])
        assert 0 <= cl['o'] <= 2 and (cl['exit'] == 'copy' or cl['o'] >= 1)
        assert n + 2 <= cl['k'] + 2 + 1, 'P stage reads at most one zero limb below D (d_12 at k 9)'
        cl['zl'], cl['zh'], cl['margin'] = cl['cands'][cl['c']]
        assert cl['zl'] > -4 and cl['zh'] < 4
    return thr8, classes


THR8, CLASSES = build_table()
C48Q = [cl['c'] for cl in CLASSES if cl['body'] == 'b48' and not cl['cub']][0]
C48C = [cl['c'] for cl in CLASSES if cl['body'] == 'b48' and cl['cub']][0]
C47 = ([cl['c'] for cl in CLASSES if cl['body'] == 'b47'] or [None])[0]
C410 = [cl['c'] for cl in CLASSES if cl['body'] == 'b410'][0]
EXITS = {}                                          # body -> {'copy': offsets, 'pack': offsets}
for cl in CLASSES:
    EXITS.setdefault(cl['body'], {'copy': set(), 'pack': set()})[cl['exit']].add(cl['o'])
if SEED:            # SLIM-RSQRT: the seed leaves through b48's copy exit with o = 8 - ns (ns 8: 0, ns 7: 1), so that exit takes o from KK
    EXITS['b48']['copy'] |= {0, 1}
# OVH2-RSQRT: the seeds against the acked seed contract v2 (-O_ns < d = r B^ns - A < C_ns, A the seed's ns + 1 limbs; gate5c.cpp's CV2 /
# OV2): ns 8 = X_8 of 4 -> 8 quadratic without bias (copy o 0): d in (-eps, hi); ns 7 = the 4 -> 7 body's X_7 + C47 (--seed47, copy o 0):
# d in (-C47 - eps, hi - C47), or without --seed47 floor(X_8 / B): d in (-eps, hi / B + 1). hi: level() from Delta_4 (the mid 2 -> 4 step).
SEEDCV2 = {7: (Fr(4294967296), Fr(12927851561)), 8: (Fr(17592186044416), Fr(52952479993693))}
if SEED:
    _h8 = level(False, D4, 4, 8)
    assert -SEEDCV2[8][0] < -EPS and _h8 < SEEDCV2[8][1], ('seed ns 8', _h8)
    if SEED47:
        _h7 = level(False, D4, 4, 7)
        assert -SEEDCV2[7][0] < -C47 - EPS and _h7 - C47 < SEEDCV2[7][1], ('seed ns 7', _h7, C47)
    else:
        assert _h8 / BF + 1 < SEEDCV2[7][1], ('seed ns 7', _h8)


def table_text():
    L = [f'# SCALAR-PROTO route table: Delta_2 < {float(D2)}, 2 -> 4 mid: hi {float(HI4):.4f}, Delta_4 < {float(D4):.4f}; '
         f'k 8 quadratic threshold s <= {THR8}']
    for cl in CLASSES:
        L.append(f"k {cl['k']:2d} s {cl['ss'][0]:2d}..{cl['ss'][-1]:2d}  2->4 {cl['body'][1]}{'=>' if cl['cub'] else '->'}{cl['n']:<2d}"
                 f"  hi {float(cl['hi']):9.4f}  c {cl['c']:3d}  {cl['exit']} o {cl['o']}  z - f in ({float(cl['zl']):7.4f}, "
                 f"{float(cl['zh']):7.4f}]  margin {float(cl['margin']):.4f}")
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------------------------------------------------
# emitter
out = []
cold = []
TR = None                                           # trace symbol (--trace)


def e(s, to=None):
    (out if to is None else to).append('\t' + s)


def lab(s, to=None):
    (out if to is None else to).append(s + ':')


def trace(regs, slot0):
    if TR:
        for i, r in enumerate(regs):
            e(f'movq\t%{r}, {TR}+{8 * (slot0 + i)}(%rip)')


def dmem(i):
    return f'{DT - 8 * (i - 1)}(%rsp)'


def pmem(c):
    return f'{PS + 8 * (c - 1)}(%rsp)'


def xfmem(i):
    return f'{XF + 8 * i}(%rsp)'


def sq2(a, X):
    """X[0..3] = (a1 B + a0)^2 exactly; uses rax, rbx, rdx"""
    e(f'movq\t%{a[0]}, %rdx')
    e(f'mulxq\t%{a[1]}, %rax, %rbx')                # a0 a1: columns 1, 2
    e(f'mulxq\t%rdx, %{X[0]}, %{X[1]}')             # a0^2
    e(f'movq\t%{a[1]}, %rdx')
    e(f'mulxq\t%rdx, %{X[2]}, %{X[3]}')             # a1^2
    e('addq\t%rax, %rax')
    e('adcq\t%rbx, %rbx')
    e(f'adcq\t$0, %{X[3]}')
    e(f'addq\t%rax, %{X[1]}')
    e(f'adcq\t%rbx, %{X[2]}')
    e(f'adcq\t$0, %{X[3]}')


def sqq(a, X):
    """X[0..2q-1] = Gt^2 exactly for q = len(a) >= 3: cross rows (two carry chains, tails into still-zero columns), the
    doubling chain, the diagonal chain; uses rax, rbx, rdx"""
    q = len(a)
    for c in range(1, 2 * q):
        e(f'xorl\t%{r32(X[c])}, %{r32(X[c])}')
    for i in range(q - 1):
        e(f'movq\t%{a[i]}, %rdx')
        e('xorl\t%eax, %eax')
        for j in range(i + 1, q):
            e(f'mulxq\t%{a[j]}, %rax, %rbx')
            e(f'adcxq\t%rax, %{X[i + j]}')
            e(f'adoxq\t%rbx, %{X[i + j + 1]}')
        e('movl\t$0, %edx')
        e(f'adcxq\t%rdx, %{X[i + q]}')
        if i + q + 1 <= 2 * q - 1:
            e(f'adoxq\t%rdx, %{X[i + q + 1]}')
            e(f'adcxq\t%rdx, %{X[i + q + 1]}')
    e(f'addq\t%{X[1]}, %{X[1]}')
    for c in range(2, 2 * q):
        e(f'adcq\t%{X[c]}, %{X[c]}')
    e(f'movq\t%{a[0]}, %rdx')
    e(f'mulxq\t%rdx, %{X[0]}, %rbx')
    e(f'addq\t%rbx, %{X[1]}')
    for i in range(1, q):
        e(f'movq\t%{a[i]}, %rdx')
        e('mulxq\t%rdx, %rax, %rbx')
        e(f'adcq\t%rax, %{X[2 * i]}')
        e(f'adcq\t%rbx, %{X[2 * i + 1]}')


def three_halves(S, c63):
    """S += S >> 1 (all 2q limbs; the carry out is zero by the bounds): (S_i >> 1) on the CF chain, (S_(i+1) << 63) on
    the OF chain; shrx / shlx keep the flags; uses rax, rbx, rdx (= 1) and c63 (= 63)"""
    e('movl\t$1, %edx')
    e(f'movl\t$63, %{r32(c63)}')
    e('xorl\t%eax, %eax')
    for i in range(len(S)):
        e(f'shrxq\t%rdx, %{S[i]}, %rax')
        if i + 1 < len(S):
            e(f'shlxq\t%{c63}, %{S[i + 1]}, %rbx')
        e(f'adcxq\t%rax, %{S[i]}')
        if i + 1 < len(S):
            e(f'adoxq\t%rbx, %{S[i]}')


def step(tag, Fregs, ending):
    """one register step of body `tag` (BODIES); Fregs: X_m's fraction limbs in registers (low first) or None (memory XF);
    ending: 'mid' or 'final'. Returns nothing; exits by jumps."""
    m, n, cmode = BODIES[tag]
    w, W, cbot = n - m, n - m + 3, n + 2
    keepF = Fregs is not None
    aj = (lambda j: f'%{Fregs[m - j]}') if keepF else (lambda j: xfmem(m - j))
    Fs = (lambda i: f'%{Fregs[i]}') if keepF else (lambda i: xfmem(i))
    avail = [r for r in POOL if not keepF or r not in Fregs]
    assert len(avail) >= cbot
    Pc = {c: avail[c - 1] for c in range(1, cbot + 1)}
    e(f'# ---- {tag}: P stage (columns 1 .. {cbot})')
    lab(f'.Lrp_s_{tag}_p')
    for c in range(1, cbot + 1):
        if c >= m:
            e(f'movq\t{dmem(c)}, %{Pc[c]}')
        else:
            e(f'xorl\t%{r32(Pc[c])}, %{r32(Pc[c])}')
    for j in range(m, 0, -1):
        e(f'movq\t{aj(j)}, %rdx')
        e('xorl\t%eax, %eax')
        for i in range(cbot - j, 0, -1):
            e(f'mulxq\t{dmem(i)}, %rax, %rbx')
            e(f'adcxq\t%rax, %{Pc[i + j]}')
            e(f'adoxq\t%rbx, %{Pc[i + j - 1]}')
        e('movl\t$0, %edx')
        e(f'adcxq\t%rdx, %{Pc[j]}')
        if j >= 2:
            e(f'adoxq\t%rdx, %{Pc[j - 1]}')
            e(f'adcxq\t%rdx, %{Pc[j - 1]}')
    if m >= 2:
        e(f'addq\t{dmem(m - 1)}, %{Pc[m - 1]}')
        for c in range(m - 2, 0, -1):
            e(f'adcq\t{dmem(c)}, %{Pc[c]}')
    # P's columns m .. n+1 are read by the MP rows: copies in the spare registers (in the order of the MP's first reads),
    # the rest stored (columns < m stay in their registers)
    order = []
    for jj in range(m, 0, -1):
        for t in range(W):
            i = n + 2 - jj - t
            if m <= i <= n + 1 and i not in order:
                order.append(i)
    assert sorted(order) == list(range(m, n + 2))
    spare = avail[cbot:]
    preg = {c: spare[x] for x, c in enumerate(order[:len(spare)])}
    for c in range(m, n + 2):
        e(f'movq\t%{Pc[c]}, %{preg[c]}' if c in preg else f'movq\t%{Pc[c]}, {pmem(c)}')
    e(f'# ---- {tag}: MP window (W = {W})')
    lab(f'.Lrp_s_{tag}_mp')
    Acc = [Pc[n + 2 - t] for t in range(W)]
    for jj in range(m, 0, -1):
        e(f'movq\t{aj(jj)}, %rdx')
        e('xorl\t%eax, %eax')
        for t in range(W):
            i = n + 2 - jj - t
            if i == 0:
                assert jj == m and t == W - 1
                e(f'adcq\t$0, %{Acc[t]}')
                continue
            e(f'mulxq\t{"%" + Pc[i] if i < m else "%" + preg[i] if i in preg else pmem(i)}, %rax, %rbx')
            e(f'adcxq\t%rax, %{Acc[t]}')
            if t + 1 < W:
                e(f'adoxq\t%rbx, %{Acc[t + 1]}')
    G = Acc[2:]
    assert len(G) == w + 1
    e(f'# ---- {tag}: G = ~Acc[2 ..], G_h = G >> 1')
    lab(f'.Lrp_s_{tag}_g')
    for r in G:
        e(f'notq\t%{r}')
    for q_ in range(w):
        e(f'shrdq\t$1, %{G[q_ + 1]}, %{G[q_]}')
    e(f'shrq\t$1, %{G[w]}')
    gmem = cmode == 'always' and W + (w + 1) > len(POOL)    # 4 => 10: G_h / G'' in memory
    if gmem:
        for q_, r in enumerate(G):
            e(f'movq\t%{r}, {GS + 8 * q_}(%rsp)')
    if cmode != 'none':
        q = n - 2 * m + 2
        cut = 2 * m + 2 * q - n - 2
        assert q >= 2 and cut >= 0 and w + 1 >= q
        e(f'# ---- {tag}: cubic G\'\' = G_h + floor(1.5 Gt^2 / B^{cut}), Gt = top {q} limbs')
        lab(f'.Lrp_s_{tag}_cub')
        if cmode == 'flag':
            e(f'cmpq\t$0, {CUBF}(%rsp)')
            e(f'je\t.Lrp_cubd_{tag}')
        Gt = G[w + 1 - q:]
        free = [r for r in POOL if r not in G] if not gmem else [r for r in POOL if r not in Gt]
        S = free[:2 * q]
        if q == 2:
            sq2(Gt, S)
        else:
            sqq(Gt, S)
        c63 = ([r for r in free if r not in S] + ([r for r in Gt] if gmem else []))[0]
        three_halves(S, c63)
        K = S[cut:]
        if gmem:
            e(f'addq\t%{K[0]}, {GS}(%rsp)')
            for i in range(1, w + 1):
                e(f'adcq\t{"%" + K[i] if i < len(K) else "$0"}, {GS + 8 * i}(%rsp)')
        else:
            e(f'addq\t%{K[0]}, %{G[0]}')
            for i in range(1, w + 1):
                e(f'adcq\t{"%" + K[i] if i < len(K) else "$0"}, %{G[i]}')
        if cmode == 'flag':
            lab(f'.Lrp_cubd_{tag}')
    e(f'# ---- {tag}: correction (inv\'s rows, two carry columns)')
    lab(f'.Lrp_s_{tag}_corr')
    Gs = (lambda j: f'{GS + 8 * j}(%rsp)') if gmem else (lambda j: f'%{G[j]}')
    Cr = [r for r in POOL if (gmem or r not in G) and (not keepF or r not in Fregs)][:W]
    assert len(Cr) == W
    for r in Cr:
        e(f'xorl\t%{r32(r)}, %{r32(r)}')
    for i in range(m):
        e(f'movq\t{Fs(i)}, %rdx')
        e('xorl\t%eax, %eax')
        for j in range(max(0, m - 2 - i), w + 1):
            idx = i + j - (m - 2)
            e(f'mulxq\t{Gs(j)}, %rax, %rbx')
            e(f'adcxq\t%rax, %{Cr[idx]}')
            e(f'adoxq\t%rbx, %{Cr[idx + 1]}')
        top = i + w - m + 3
        e('movl\t$0, %edx')
        e(f'adcxq\t%rdx, %{Cr[top]}')
        if top + 1 < W:
            e(f'adoxq\t%rdx, %{Cr[top + 1]}')
            e(f'adcxq\t%rdx, %{Cr[top + 1]}')
    e(f'# ---- {tag}: C = Cr[2 ..] + G\'\', X_{n} = R_{m} B^{w} + C')
    lab(f'.Lrp_s_{tag}_upd')
    e(f'addq\t{Gs(0)}, %{Cr[2]}')
    for q_ in range(1, w + 1):
        e(f'adcq\t{Gs(q_)}, %{Cr[q_ + 2]}')
    if keepF:
        Xh = list(Fregs)
    else:
        spare = [r for r in POOL if r not in Cr[2:]] if gmem else [r for r in G] + [r for r in POOL if r not in G and r not in Cr]
        Xh = spare[:m]
        for i in range(m):
            e(f'movq\t{xfmem(i)}, %{Xh[i]}')
    e(f'addq\t%{Cr[w + 2]}, %{Xh[0]}')
    for i in range(1, m):
        e(f'adcq\t$0, %{Xh[i]}')
    e('movl\t$1, %ebx')
    e('adcq\t$0, %rbx')
    X = Cr[2:w + 2] + Xh
    assert len(X) == n and len(set(X)) == n and 'rbx' not in X
    if ending == 'mid':
        e(f'# ---- {tag}: mid: X - 2, clamp up to B^{n}, store, jump to the final body')
        lab(f'.Lrp_s_{tag}_mid')
        e(f'subq\t$2, %{X[0]}')
        for r in X[1:]:
            e(f'sbbq\t$0, %{r}')
        e('sbbq\t$0, %rbx')
        e(f'jz\t.Lrp_clamp_{tag}')
        lab(f'.Lrp_clampd_{tag}')
        trace(X + ['rbx'], 3)
        for i, r in enumerate(X):
            e(f'movq\t%{r}, {xfmem(i)}')
        e(f'jmp\t*{BODY}(%rsp)')
        lab(f'.Lrp_clamp_{tag}', cold)
        for r in X:
            e(f'xorl\t%{r32(r)}, %{r32(r)}', cold)
        e('movl\t$1, %ebx', cold)
        e(f'jmp\t.Lrp_clampd_{tag}', cold)
        return
    # final: + c, then copy (s = 0) or pack
    trace(X + ['rbx'], 8)
    e(f'# ---- {tag}: final: + c, copy (s = 0) or pack, z = floor(Y 2^s / B^o)')
    lab(f'.Lrp_s_{tag}_fin')
    e(f'addq\t{BIAS}(%rsp), %{X[0]}' if tag == 'b48' else f'addq\t${C47 if tag == "b47" else C410}, %{X[0]}')
    for r in X[1:]:
        e(f'adcq\t$0, %{r}')
    e('adcq\t$0, %rbx')
    Y = X + ['rbx']
    free = [r for r in POOL + ['rax'] if r not in Y]
    e(f'movq\t{PP}(%rsp), %rdx')
    e('cmpq\t$1, %rdx')
    e(f'je\t.Lrp_copy_{tag}')
    e('cmpq\t$1, %rbx')
    e(f'ja\t.Lrp_ovf_{tag}')
    lab(f'.Lrp_pk_{tag}')
    O = sorted(EXITS[tag]['pack'])
    zp, ta = free[0], free[1]
    e(f'movq\t{Z}(%rsp), %{zp}')
    if len(O) == 1:
        o = O[0]
        assert o >= 1
        e(f'mulxq\t%{Y[o - 1]}, %{ta}, %{Y[o - 1]}')
        for i in range(0, n + 1 - o):
            e(f'mulxq\t%{Y[i + o]}, %{ta}, %{Y[i + o]}')
            e(f'orq\t%{Y[i + o - 1]}, %{ta}')
            e(f'movq\t%{ta}, {8 * i}(%{zp})')
    else:
        assert O == [1, 2]
        zq = free[2]
        e(f'movq\t{KK}(%rsp), %{zq}')
        e(f'leaq\t{-8 * (n + 1)}(%{zp},%{zq},8), %{zq}')           # z - 8 o
        e(f'mulxq\t%{Y[0]}, %{ta}, %{Y[0]}')
        e(f'mulxq\t%{Y[1]}, %{ta}, %{Y[1]}')
        e(f'orq\t%{Y[0]}, %{ta}')
        e(f'movq\t%{ta}, (%{zp})')                                   # limb 1 of the product: z[0] (o = 1), overwritten (o = 2)
        for j in range(2, n + 1):
            e(f'mulxq\t%{Y[j]}, %{ta}, %{Y[j]}')
            e(f'orq\t%{Y[j - 1]}, %{ta}')
            e(f'movq\t%{ta}, {8 * j}(%{zq})')
    e('jmp\t.Lrp_ret')
    lab(f'.Lrp_copy_{tag}')
    O = sorted(EXITS[tag]['copy'])
    if O == [0, 1, 2]:      # SLIM-RSQRT --no-b47: k 7 s 0 (o = 2) packs (2^s = 1 in rdx: the pack is the copy); o 0 / 1 copy
        assert tag == 'b48' and 2 in EXITS[tag]['pack']
        e(f'cmpq\t${n - 1}, {KK}(%rsp)')
        e(f'je\t.Lrp_pk_{tag}')
        O = [0, 1]
    e(f'movq\t{Z}(%rsp), %{zp}')
    if len(O) == 1:
        o = O[0]
        for i in range(0, n + 1 - o):
            e(f'movq\t%{Y[i + o]}, {8 * i}(%{zp})')
    else:
        assert O == [0, 1]
        zq = free[2]
        e(f'movq\t{KK}(%rsp), %{zq}')
        e(f'leaq\t{-8 * (n + 1)}(%{zp},%{zq},8), %{zq}')           # z - 8 o
        e(f'movq\t%{Y[0]}, (%{zp})')                                 # z[0] (o = 0), overwritten (o = 1)
        for j in range(1, n + 1):
            e(f'movq\t%{Y[j]}, {8 * j}(%{zq})')
    e('jmp\t.Lrp_ret')
    # Y >= 2 B^n: only s = 63 overflows k limbs (then z = B^k - 1); s <= 62 packs as usual
    lab(f'.Lrp_ovf_{tag}', cold)
    e('btq\t$63, %rdx', cold)
    e(f'jnc\t.Lrp_pk_{tag}', cold)
    e('jmp\t.Lrp_fill', cold)


def generate(prefix, trace_on):
    global TR
    out.clear()
    cold.clear()
    TR = f'{prefix}rsp_trace' if trace_on else None
    ent = f'{prefix}rsp_k7'
    e0 = out.append
    if LIB:
        e0('/* Generated by tools/generate_fp_rsqrt_rsp.py; do not edit. The P-first scalar path of the fp reciprocal square root:')
        e0(f' * {prefix}rsp_k7 (public k 7..10, k 6 at s 61..63) and {prefix}rsp_seed (the IFMA side\'s seeds, ns 7 / 8). Research kernel')
        e0(' * G (experiments/results/fpk_2026-10-12/rsqrt_s4x/joint/x512g_claude/rsp.S, OVH2-RSQRT; this file is that rsp.S with the')
        e0(' * library\'s symbol names and these lines). The research generator\'s header (its paths relative to rsqrt_s4x/joint):')
        e0(' *')
        e0(f' * Generated by claude_scalar/proto/gen/gen_rsp.py (SCALAR-PROTO, joint rsqrt, 2026-10-15); do not edit. LGPL-3.0-or-later.')
    else:
        e0(f'/* Generated by claude_scalar/proto/gen/gen_rsp.py (SCALAR-PROTO, joint rsqrt, 2026-10-15); do not edit. LGPL-3.0-or-later.')
    e0(f' * Claude\'s inv-style scalar rsqrt path, public k 7..10, entered from the public entry by a tail jump with the IFMA')
    e0(f' * hand-off registers (rdi z, rsi raw limbs, r8 nb, r9 e_top, rcx k, r10 c, r11 s). Decode and the hybrid R2 are')
    e0(f' * copies of Codex\'s scalar-v3 (joint/j12_codex, gen_r2_hybrid.py, LGPL-3.0-or-later). Frame {FRAME} B + 6 pushes.')
    e0(f' * Route classes (bias c, k 8 quadratic up to s = {THR8}): see the generator\'s table. */')
    if K6HI or SEED47:
        e0('/* OVH2-RSQRT (Claude, 2026-10-16; ovh2_rsqrt/gen/gen_rsp_sl.py):' + (' --k6hi: public k 6 at s 61..63 (scalar.S sends them here)'
           ' through 2 -> 4 -> 7, pack from limb 2 (d_9 = 0);' if K6HI else '') + (' --seed47: seeds ns 7 through the 4 -> 7 body (its copy'
           ' exit, o = 0; its bias), ns 8 through 4 -> 8;' if SEED47 else '') + ' route table in the generator\'s --table output. */')
    if NO_B47 or SEED:
        e0(f'/* SLIM-RSQRT (Claude, 2026-10-16; slim_rsqrt/gen/gen_rsp_sl.py):' + (' --no-b47: k 7 and k 8 s 0 run 4 -> 8 (quadratic, the'
           f' shared bias {C48Q}; k 7 packs from limb 2, k 8 s 0 copies from limb 1; d_10 zeroed at entry for k 7), no 4 -> 7 body.' if NO_B47 else '')
           + (f' --seed: {prefix}rsp_seed(out, x, nb, c, ns), {prefix}seed\'s interface for ns 7 / 8: decode ns + 3 limbs, R2,'
              f' 2 -> 4, 4 -> 8 quadratic without bias, the copy exit with o = 8 - ns: out[0 .. ns] = X_8\'s limbs 8 - ns .. 8'
              f' (exact - X_8 in (-eps, {float(level(False, D4, 4, 8)):.2f}) units of B^-8, a floor for ns 7).' if SEED else '') + ' */')
    e0('\t.text')
    if SEED:        # SLIM-RSQRT: the seed entry (the public path's frame and slots; it joins at the decode)
        se = f'{prefix}rsp_seed'
        e0(f'\t.globl\t{se}')
        e0(f'\t.hidden\t{se}')
        e0(f'\t.type\t{se}, @function')
        e0('\t.p2align\t5')
        lab(se)
        for r in ('rbx', 'rbp', 'r12', 'r13', 'r14', 'r15'):
            e(f'pushq\t%{r}')
        e(f'subq\t${FRAME}, %rsp')
        e(f'movq\t%rdi, {Z}(%rsp)')
        e(f'movq\t$0, {ET}(%rsp)')
        e('leal\t1(%r8), %eax')                      # k' = ns + 1: decode ns + 3 limbs; the copy exit's o = n + 1 - k' = 8 - ns
        e(f'movq\t%rax, {KK}(%rsp)')
        e(f'movq\t$1, {PP}(%rsp)')                   # 2^s = 1: the copy exit
        e(f'movq\t$0, {DB}(%rsp)')
        e(f'movq\t$0, {CUBF}(%rsp)')                 # 4 -> 8 quadratic
        e(f'movq\t$0, {BIAS}(%rsp)')                 # no bias: the one-sided X_8
        e('leaq\t.Lrp_b48(%rip), %r9')
        if SEED47:                                   # OVH2-RSQRT: ns 7: the 4 -> 7 body (copy exit o = n + 1 - k' = 0: Y[0 .. 7])
            assert 0 in EXITS['b47']['copy'] and not NO_B47
            e('leaq\t.Lrp_b47(%rip), %r11')
            e('cmpl\t$7, %r8d')
            e('cmoveq\t%r11, %r9')
        e(f'movq\t%r9, {BODY}(%rsp)')
        e('movq\t%rdx, %r8')                         # nb
        e('movl\t%ecx, %r10d')                       # c
        e('movl\t%eax, %ecx')                        # k'
        e('jmp\t.Lrp_s_decode')
        out.append(f'\t.size\t{se}, .-{se}')
    e0(f'\t.globl\t{ent}')
    e0(f'\t.hidden\t{ent}')
    e0(f'\t.type\t{ent}, @function')
    e0('\t.p2align\t5')
    lab(ent)
    for r in ('rbx', 'rbp', 'r12', 'r13', 'r14', 'r15'):
        e(f'pushq\t%{r}')
    e(f'subq\t${FRAME}, %rsp')
    e(f'movq\t%rdi, {Z}(%rsp)')
    e(f'movq\t%r9, {ET}(%rsp)')
    e(f'movq\t%rcx, {KK}(%rsp)')
    e('movl\t$1, %eax')
    e('shlxq\t%r11, %rax, %rax')                     # 2^s (r11 = s, upper half zero)
    e(f'movq\t%rax, {PP}(%rsp)')
    e(f'movq\t$0, {DB}(%rsp)')                       # d_12 = 0 (k 9: 4 => 10 reads one limb below D)
    if NO_B47:
        e(f'movq\t$0, {dmem(10)}')                   # SLIM-RSQRT: d_10 = 0 (k 7: 4 -> 8 reads one limb below D; k >= 8 decode it)
    if K6HI:
        e(f'movq\t$0, {dmem(9)}')                    # OVH2-RSQRT: d_9 = 0 (k 6: 4 -> 7 reads one limb below D; k >= 7 decode it)
    e('# ---- route: the final body (BODY), b48\'s cubic flag (CUBF) and bias (BIAS)')
    lab('.Lrp_s_route')
    e(f'leaq\t.Lrp_b410(%rip), %rax')
    e('cmpl\t$9, %ecx')
    e('ja\t.Lrp_rset')                               # k 10
    e('jb\t.Lrp_r78')
    e('testl\t%r11d, %r11d')
    e('jnz\t.Lrp_rset')                              # k 9, s >= 1
    e(f'movq\t$1, {CUBF}(%rsp)')                     # k 9, s 0: 4 => 8, copy
    e(f'movq\t${C48C}, {BIAS}(%rsp)')
    e('leaq\t.Lrp_b48(%rip), %rax')
    e('jmp\t.Lrp_rset')
    lab('.Lrp_r78')
    if NO_B47:      # SLIM-RSQRT: k 7 and k 8 s 0 .. THR8: 4 -> 8 quadratic (one bias); k 8 s > THR8: 4 => 8
        e('leaq\t.Lrp_b48(%rip), %rax')
        e(f'movq\t$0, {CUBF}(%rsp)')
        e(f'movq\t${C48Q}, {BIAS}(%rsp)')
        e('cmpl\t$7, %ecx')
        e('je\t.Lrp_rset')                           # k 7
    else:
        e('leaq\t.Lrp_b47(%rip), %rax')
        e('cmpl\t$7, %ecx')
        e('jbe\t.Lrp_rset' if K6HI else 'je\t.Lrp_rset')     # k 7 (OVH2-RSQRT --k6hi: and k 6, s 61..63)
        e('testl\t%r11d, %r11d')
        e('jz\t.Lrp_rset')                               # k 8, s 0: 4 -> 7, copy
        e('leaq\t.Lrp_b48(%rip), %rax')                  # k 8, s >= 1: 4 -> 8 quadratic up to THR8, else 4 => 8
        e(f'movq\t$0, {CUBF}(%rsp)')
        e(f'movq\t${C48Q}, {BIAS}(%rsp)')
    e(f'cmpl\t${THR8}, %r11d')
    e('jbe\t.Lrp_rset')
    e(f'movq\t$1, {CUBF}(%rsp)')
    e(f'movq\t${C48C}, {BIAS}(%rsp)')
    lab('.Lrp_rset')
    e(f'movq\t%rax, {BODY}(%rsp)')
    e('# ---- decode: N = k + 2 limbs of x 2^c, top at DT (Codex\'s rolling decode, scalar-v3)')
    lab('.Lrp_s_decode')
    e('leaq\t2(%rcx), %r12')
    e('movq\t%r10, %rcx')
    e('leaq\t-8(%rsi,%r8,8), %rsi')
    e(f'leaq\t{DT}(%rsp), %rdi')
    e('movl\t$1, %edx')
    e('shlxq\t%rcx, %rdx, %rdx')
    e('movq\t(%rsi), %r9')
    e('shlxq\t%rcx, %r9, %r9')
    if R2V2:
        e('movq\t%r9, %r14')                       # FIX-RSQRT --r2v2: R2's top limb in r14 (nb = 1: no lower raw limb)
    e('decq\t%r8')
    e('movq\t%r8, %r13')
    e('cmpq\t%r12, %r13')
    e('cmovaq\t%r12, %r13')
    e('subq\t%r13, %r12')
    e('testq\t%r13, %r13')
    e('jz\t.Lrp_decode_last')
    if R2V2:
        e('mulxq\t-8(%rsi), %rax, %r14')           # FIX-RSQRT --r2v2: top limb = (x[nb-1] << c) | (x[nb-2] >> (64 - c)), as the loop's
        e('orq\t%r9, %r14')                        # first step, in a register for R2 (no store-forward wait)
    lab('.Lrp_decode_pair')
    e('mulxq\t-8(%rsi), %r10, %rax')
    e('orq\t%rax, %r9')
    e('movq\t%r9, (%rdi)')
    e('movq\t%r10, %r9')
    e('subq\t$8, %rsi')
    e('subq\t$8, %rdi')
    e('decq\t%r13')
    e('jnz\t.Lrp_decode_pair')
    lab('.Lrp_decode_last')
    e('testq\t%r12, %r12')
    e('jz\t.Lrp_decoded')
    e('movq\t%r9, (%rdi)')
    e('subq\t$8, %rdi')
    e('decq\t%r12')
    e('jz\t.Lrp_decoded')
    e('xorl\t%eax, %eax')
    lab('.Lrp_decode_zero')
    e('movq\t%rax, (%rdi)')
    e('subq\t$8, %rdi')
    e('decq\t%r12')
    e('jnz\t.Lrp_decode_zero')
    lab('.Lrp_decoded')
    e(f'leaq\t{DT - 16}(%rsp), %r15')                # T = d_1 d_2 d_3 (Codex's top3)
    e('# ---- Codex\'s hybrid R2 (gen_r2_hybrid.emit_core, copied; the three output stores dropped)')
    lab('.Lrp_s_r2')
    core = []
    if R2V3:
        r2h.emit_core_v3(core.append, '.Lrp_r2hy')                 # FIX-RSQRT --r2v3
    elif R2V2:
        r2h.emit_core_v2(core.append, '.Lrp_r2hy', top='r14')     # FIX-RSQRT --r2v2
    else:
        r2h.emit_core(core.append, '.Lrp_r2hy', f'{R2S0}(%rsp)', f'{R2S1}(%rsp)')
    stores = ['movq %rax, 0(%rbx)', 'movq %r10, 8(%rbx)', 'movq %r11, 16(%rbx)']
    assert core[-3:] == stores
    for s in core[:-3]:
        out.append(s if s.endswith(':') or s.startswith('#') else '\t' + s)
    trace(['rax', 'r10', 'r11'], 0)
    e('# ---- 2 -> 4 (mid): A = B^2 + r10 B + rax')
    lab('.Lrp_s_b24_in')
    e('movq\t%rax, %r14')
    e('movq\t%r10, %r15')
    step('b24', ['r14', 'r15'], 'mid')
    for tag in (('b48', 'b410') if NO_B47 else ('b47', 'b48', 'b410')):     # SLIM-RSQRT --no-b47
        out.append('\t.p2align\t4')
        lab(f'.Lrp_{tag}')
        step(tag, None, 'final')
    if out[-1] == '\tjmp\t.Lrp_ret':
        out.pop()                                    # the last exit falls through into the return
    lab('.Lrp_ret')
    e(f'movq\t{ET}(%rsp), %rax')
    e(f'movq\t{KK}(%rsp), %rdx')
    e(f'addq\t${FRAME}, %rsp')
    for r in ('r15', 'r14', 'r13', 'r12', 'rbp', 'rbx'):
        e(f'popq\t%{r}')
    e('ret')
    # cold paths: clamps, the s = 63 overflow fill (z = B^k - 1)
    out.extend(cold)
    lab('.Lrp_fill')
    e(f'movq\t{Z}(%rsp), %rdi')
    e(f'movq\t{KK}(%rsp), %rcx')
    e('movq\t$-1, %rax')
    lab('.Lrp_fill1')
    e('movq\t%rax, -8(%rdi,%rcx,8)')
    e('decq\t%rcx')
    e('jnz\t.Lrp_fill1')
    e('jmp\t.Lrp_ret')
    out.append(f'\t.size\t{ent}, .-{ent}')
    rod = []
    if not R2V3:                                     # FIX-RSQRT --r2v3: the core has no constants
        (r2h.emit_constants_v2 if R2V2 else r2h.emit_constants)(rod.append, '.Lrp_r2hy')
    out.extend(rod)
    if TR:
        out.append('\t.bss')
        out.append(f'\t.globl\t{TR}')
        out.append('\t.p2align\t6')
        out.append(f'{TR}:')
        out.append('\t.zero\t256')
    out.append('\t.section\t.note.GNU-stack,"",@progbits')
    text = '\n'.join(out) + '\n'
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output', type=Path)
    ap.add_argument('--prefix', default='fpkrs_')
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--table', type=Path)
    ap.add_argument('--no-b47', action='store_true')    # SLIM-RSQRT (read at import: NO_B47)
    ap.add_argument('--seed', action='store_true')      # SLIM-RSQRT (read at import: SEED)
    ap.add_argument('--r2v2', action='store_true')      # FIX-RSQRT (read at import: R2V2)
    ap.add_argument('--r2v3', action='store_true')      # FIX-RSQRT (read at import: R2V3)
    ap.add_argument('--seed47', action='store_true')    # OVH2-RSQRT (read at import: SEED47)
    ap.add_argument('--k6hi', action='store_true')      # OVH2-RSQRT (read at import: K6HI)
    a = ap.parse_args(ARGV)
    assert a.prefix.replace('_', '').isalnum()
    text = generate(a.prefix, a.trace)
    if CHECK:
        if a.output.read_text() != text:
            raise SystemExit(f'{a.output} differs from tools/generate_fp_rsqrt_rsp.py\'s output')
        print(f'OK: {a.output} is tools/generate_fp_rsqrt_rsp.py\'s output')
        return
    a.output.write_text(text)
    if a.table:
        a.table.write_text(table_text())
    print(a.output)


if __name__ == '__main__':
    main()
