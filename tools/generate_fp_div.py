#!/usr/bin/env python3
"""Generate src/fp/x86_64/div.S: the fp kernel division and reciprocal (contract: src/fp/kernel.hpp).

  fxr sbn3f_div(limb *z, const fx *a, const fx *b, int64_t leps)   rdi z, rsi a, rdx b, rcx leps
  fxr sbn3f_inv_base(limb *z, const fx *b, int64_t leps)           rdi z, rsi b, rdx leps (hidden: the reciprocal
      sbn3f_inv, src/fp/x86_64/inv.S, hands over k <= 3, k = 4 at ld > 60 and k > 512; precondition k <= 512)
  result: rax e, rdx ns

The research prototype's scalar division (experiments/results/fp_roots_2026-10-09/src/gen_fpdiv_s.py, 2026-10-10 .. 11)
ported to the limb grid; its step machinery is unchanged. n = 2 and n = 3..5 port GMP 6.3.0 udiv_qr_3by2 /
invert_pi1 / sbpi1_div_qr arithmetic via the library (dual LGPLv3+ / GPLv2+; Copyright 1991-2016 Free Software
Foundation, Inc.; notice src/core/x86_64/scalar-donor-notice.txt); n >= 6 is the prototype's register-window short
division (divsw.hpp there, Bounds (1)-(6)).

Shape. e_top = ea - eb + 1 (the quotient a / b is below B^e_top), k = e_top - floor((leps - 2) / 64) quotient limbs at
grid L(2) (k <= 0: zero; precondition k <= 512). The division from k = KMIN = 22 is sbn3f_div_km
(src/fp/x86_64/div_km.cpp: Karp-Markstein on the reciprocal's IFMA levels, the operands' whole arrays; its header has the
derivation); below, and for the reciprocal entry (an = 0), the cores here. The top k + 1 limbs of a and of b (zero below an operand that is
shorter) are shifted left together by ld = lzcnt(b's top limb) (k <= 5: in registers by mulx with 2^ld; k <= NMAX: a
VPSHLDVQ funnel into the frame; qv: inside its u64 -> u52 conversions, the windows passed unshifted):
  Y = floor(b_w 2^ld / B), k limbs, normalized;   X = floor(a_w 2^ld / B), k + 1 limbs, X < B Y (X / Y < 2^(ld+1)).
The quotient of X B^(k-1) by Y has k limbs (below B^k) on the grid L(2): no binade test, no normalization of z, no
clamp. The cores take the remainder R = X B^k (positions k .. 2k; the prototype: X' B^(k+1) with X' < Y after a
subtraction for X >= Y), which keeps their invariant 0 <= R < 2 Y B^(k+1) at the first step (R < Y B^(k+1)).
  k = 1: one divq of <x1, x0> by y0.  k = 2: two exact 3/2 steps on <x2, x1, x0> and <r1, r0, 0>.
  k = 3..5: div_ssF's 3/2 loop, the registers loaded with X[k] .. X[0] top-aligned.
  k = 6..NMAX: the register-window short division, M from position k (X[0] .. X[k-5], complemented), the window
    X[k-4] .. X[k] (the division to KMIN - 1, the reciprocal to NMAX).  The reciprocal above NMAX (k <= QVMAX = 230; not
    reached from sbn3f_inv, which hands over only k <= 4 and k > 512): sbn3f_div_qv (src/fp/x86_64/div_qv.cpp, the IFMA
    short division).
Error, in units of the last quotient limb: the cores give z in (v' - 1 - 16/B, v' + 2k/B^2) (k <= 2: the exact floor;
qv: (v' - 1 - 2^-6.9, v' + 2^-400]), v' = X B^(k-1) / Y; Y below b_w 2^ld / B by < 1 + 2^ld / B < 3/2 with
Y >= B^k / 2 and v' < B^k puts v' above the true quotient by less than 3; X's and a's truncations lower v' by
less than 3/B. So z - a / b lies in (-1 - 19/B, 3 + 2k/B^2): below 4 units = 2^(64 L(2) + 2) <= 2^leps.
The top limb of z is zero when X < Y; the strip loop takes any count.

The reciprocal. e_top = 1 - eb (1 / b <= B^(1-eb), equal only for b = B^(eb-1)), k = e_top - L(2), X = 2^ld B^k
(no window of a), Y as above: X < B Y unless Y = B^k / 2 (ld = 63: b's window is 1, k - 1 zero limbs, a last limb
<= 1), which the entry tests when b's top limb is 1 and answers with z = 1 at e_top + 1 (1 / b in (B^(1-eb) (1 - 2 B^-k),
B^(1-eb)]: error < 2 units). Otherwise the bound above holds as it is. k = 1: one divq of <2^ld, 0>. k = 2: the
prototype's two-limb inverse: v = invert_pi1(Y), r = B^3 - (B + v) Y, v0 = r (B + v) / B^2 from below (by < 2), so
W = <1, v, v0> lies in (B^4 / Y - 3, B^4 / Y] and z = floor(W 2^ld / B) in (v' - 5/2, v'] (no divq after the
inverse's). k = 3..5, 6..NMAX and qv: the division's cores on this X (registers, directly initialized complemented M/window, an = 0).
"""
import sys

OUT = 'src/fp/x86_64/div.S'
NMAX = 34                         # the scalar cores' largest n (the prototype's measured cut 34 / 35); qv beyond
KMAX = 512                        # the precondition's largest k (Karp-Markstein's range)
QVMAX = 230                       # qv's largest k (the reciprocal's na = 0 path above NMAX, unreachable from sbn3f_inv)
KMIN = 22                         # the division's k from which Karp-Markstein (div_km.cpp) beats the register-window cores
CORE_PAD = 15                     # int3 bytes before the cores: the route's code in the entries moved them by 49 B (+1 % at k 9..17)
MO = 0                            # M: positions n-6 .. 2n+1 at 8 (p - (n-6))
QO = 8 * (NMAX + 8)               # Qd[k] at QO + 8k, k <= n
V1 = QO + 8 * (NMAX + 1)
V0, ZPTR, M2, C80 = V1 + 8, V1 + 16, V1 + 32, V1 + 40   # M2, C80: constants -2 and 2^63
FRAME = (V1 + 104 + 8 + 15) // 16 * 16 - 8  # 6 pushes + the return address: rsp 16-aligned after sub

SAVE_REGS = ['rbx', 'rbp', 'r12', 'r13', 'r14', 'r15']
MB = FRAME                        # k = 6..NMAX (.Lbe): above the core's frame: 0 e_top, 8 ns, 16 z, 24 k, 32 na, 40 a.d, 48 Y
XB, YB = MB + 64, MB + 64 + 8 * 48   # X and Y, 48 limbs each (whole blocks of 8 up to k + 7)
FRAMEB = MB + 64 + 2 * 8 * 48       # FRAME = 8 mod 16, so FRAMEB too: rsp 16-aligned after the pushes

out = []
def e(s=''):
    out.append(s)

# ---- shared pieces
def inv_pi1(d1, d0, v, tag):
    """v = invert_pi1(d1, d0) (registers d1, d0 hold the limbs; uses rax, rdx, r10, r11; v must not be those)"""
    e(f'\tmov\t%{d1}, %rdx')
    e('\tnot\t%rdx')
    e('\tmov\t$-1, %rax')
    e(f'\tdiv\t%{d1}')
    e(f'\tmov\t%rax, %{v}')
    e(f'\tmov\t%{d1}, %r10')
    e(f'\timul\t%{v}, %r10')
    e(f'\tadd\t%{d0}, %r10')
    e(f'\tjnc\t.Linv1_{tag}')
    e(f'\tsub\t$1, %{v}')
    e(f'\tcmp\t%{d1}, %r10')
    e('\tsbb\t%r11, %r11')
    e('\tnot\t%r11')
    e(f'\tsub\t%{d1}, %r10')
    e(f'\tadd\t%r11, %{v}')
    e(f'\tand\t%{d1}, %r11')
    e('\tsub\t%r11, %r10')
    e(f'.Linv1_{tag}:')
    e(f'\tmov\t%{d0}, %rdx')
    e(f'\tmulx\t%{v}, %r11, %rax')
    e('\tadd\t%rax, %r10')
    e(f'\tjnc\t.Linv2_{tag}')
    e(f'\tsub\t$1, %{v}')
    e(f'\tcmp\t%{d1}, %r10')
    e(f'\tjb\t.Linv2_{tag}')
    e(f'\tja\t.Linv3_{tag}')
    e(f'\tcmp\t%{d0}, %r11')
    e(f'\tjb\t.Linv2_{tag}')
    e(f'.Linv3_{tag}:')
    e(f'\tsub\t$1, %{v}')
    e(f'.Linv2_{tag}:')

def udiv_3by2(q, r1, r0, n2, n1, n0, d1, d0, dinv, tag, cold):
    """q, (r1, r0) = <n2, n1, n0> / <d1, d0> (GMP udiv_qr_3by2); operands are registers or memory strings; uses
    rdx, rax, r11, rbx and rcx. q, r1, r0 registers distinct from those; r1 may be n2's register and r0 n1's (each is
    written after the last read of the other)."""
    e(f'\tmov\t{n2}, %rdx')
    e(f'\tmulx\t{dinv}, %rbx, %{q}')
    e(f'\tadd\t{n1}, %rbx')
    e(f'\tadc\t{n2}, %{q}')
    e(f'\tmov\t%{q}, %r11')
    e(f'\timul\t{d1}, %r11')
    e(f'\tmov\t{n1}, %{r1}')
    e(f'\tsub\t%r11, %{r1}')
    e(f'\tmov\t{n0}, %{r0}')
    e(f'\tsub\t{d0}, %{r0}')
    e(f'\tsbb\t{d1}, %{r1}')
    e(f'\tmov\t%{q}, %rdx')
    e(f'\tmulx\t{d0}, %r11, %rax')
    e(f'\tsub\t%r11, %{r0}')
    e(f'\tsbb\t%rax, %{r1}')
    e(f'\tadd\t$1, %{q}')
    e(f'\tcmp\t%rbx, %{r1}')
    e('\tsbb\t%rcx, %rcx')
    e('\tnot\t%rcx')
    e(f'\tadd\t%rcx, %{q}')
    e(f'\tmov\t{d0}, %r11')
    e('\tand\t%rcx, %r11')
    e(f'\tand\t{d1}, %rcx')
    e(f'\tadd\t%r11, %{r0}')
    e(f'\tadc\t%rcx, %{r1}')
    e(f'\tcmp\t{d1}, %{r1}')
    e(f'\tjae\t.Lfix_{tag}')
    e(f'.Lfixr_{tag}:')
    cold += [f'.Lfix_{tag}:', f'\tja\t.Lfixd_{tag}', f'\tcmp\t{d0}, %{r0}', f'\tjb\t.Lfixr_{tag}', f'.Lfixd_{tag}:',
             f'\tadd\t$1, %{q}', f'\tsub\t{d0}, %{r0}', f'\tsbb\t{d1}, %{r1}', f'\tjmp\t.Lfixr_{tag}']

# ---- k <= 5: the entry's operands go straight into the small cores. State on entry to .Lf<k>: rsi = a.d, r10 = na,
# rdx = b.d, r11 = nb, rax = e_top, r8 = ns (sign | k), rdi = z. Every window limb comes by load_top (the t-th limb
# from the operand's top, zero below it: one predictable branch per limb) and the shift by SHLD in registers.
LBL = [0]
def lbl():
    LBL[0] += 1
    return f'.Lg{LBL[0]}'

def r32(r):
    return {'rax': 'eax', 'rbx': 'ebx', 'rcx': 'ecx', 'rdx': 'edx', 'rsi': 'esi', 'rdi': 'edi', 'rbp': 'ebp'}.get(r, r + 'd')

def load_top(src, n, t, reg):
    """reg = the operand's t-th limb from the top (src = its limbs, n = its limb count, registers), 0 when t >= n"""
    if t == 0:
        e(f'\tmovq\t-8({src},{n},8), %{reg}')
        return
    skip = lbl()
    e(f'\txorl\t%{r32(reg)}, %{r32(reg)}')
    e(f'\tcmpq\t${t + 1}, {n}')
    e(f'\tjb\t{skip}')
    e(f'\tmovq\t{-8 * (t + 1)}({src},{n},8), %{reg}')
    e(f'{skip}:')

def load_window(src, n, k, regs, temp, scratch, cold):
    """regs[t] = the operand's t-th limb from the top, t = 0..k, zero below it (src = its limbs, n = its count,
    registers). Fast path n >= k: the limbs t < k plainly, t = k by load_top. Below (cold): the operand is copied into
    the zero-padded stack slots temp(%rsp) .. (k + 1 limbs; temp a displacement) and read from there (src, n change)."""
    back, slow = lbl(), lbl()
    e(f'\tcmpq\t${k}, {n}')
    e(f'\tjb\t{slow}')
    e(f'{back}:')
    for t in range(k):
        e(f'\tmovq\t{-8 * (t + 1)}({src},{n},8), %{regs[t]}')
    load_top(src, n, k, regs[k])
    c = [f'{slow}:'] + [f'\tmovq\t$0, {temp + 8 * i}(%rsp)' for i in range(k + 1)]
    done = lbl()
    for t in range(k):
        c += [f'\tcmpq\t${t + 1}, {n}', f'\tjb\t{done}', f'\tmovq\t{-8 * (t + 1)}({src},{n},8), %{scratch}',
              f'\tmovq\t%{scratch}, {temp + 8 * (k - t)}(%rsp)']
    c += [f'{done}:', f'\tleaq\t{temp}(%rsp), {src}', f'\tmovl\t${k + 1}, %{r32(n[1:])}', f'\tjmp\t{back}']
    cold += c

# The shift by ld: SHLD r, r, CL issues one per 2 cycles on Zen 5 (2026-10-12 microbenchmark), so the funnel is
# dst = (dst << ld) | ((src >> (64 - ld)) & m) by SHLX / SHRX (counts mod 64), m = 0 for ld = 0 (src >> 64 is 0):
# 0.93 cycles a limb.
def funnel_prep(c64, m):
    """c64 = 64 - ld, m = all ones unless ld = 0 (rcx = ld); registers"""
    e(f'\tmovl\t$64, %{r32(c64)}')
    e(f'\tsubl\t%ecx, %{r32(c64)}')
    e(f'\txorl\t%{r32(m)}, %{r32(m)}')
    e('\tcmpq\t$1, %rcx')
    e(f'\tadcq\t$-1, %{m}')

def shl2(dst, src, c64, m, tmp):
    """dst = shld(dst, src, ld) (rcx = ld, c64 / m from funnel_prep; m a register or memory operand string)"""
    e(f'\tshlxq\t%rcx, %{dst}, %{dst}')
    e(f'\tshrxq\t%{c64}, %{src}, %{tmp}')
    e(f'\tandq\t{m}, %{tmp}')
    e(f'\torq\t%{tmp}, %{dst}')

def shr2(dst, src, c64, m):
    """dst = shld(0, src, ld) = src >> (64 - ld), 0 for ld = 0"""
    e(f'\tshrxq\t%{c64}, %{src}, %{dst}')
    e(f'\tandq\t{m}, %{dst}')

# The shift by p = 2^ld as a product (k <= 5): mulx gives lo = w << ld and hi = w >> (64 - ld) (0 for ld = 0) at
# once, so a funnel limb is one mulx and one or, with no 64 - ld count or ld = 0 mask (shl2: four instructions, and
# funnel_prep's five a call); 2026-10-12: k = 1 / 2 reciprocal 10.7 / 27.4 -> 10.1 / 25.7 cycles.
def p_prep():
    """rdx = p = 2^ld (rcx = ld)"""
    e('\tmovl\t$1, %edx')
    e('\tshlxq\t%rcx, %rdx, %rdx')

def funnel_mulx(regs, hi0, tmp):
    """regs[0..m] = a window's limbs from the top (rdx = p): regs[t] = shld(regs[t], regs[t+1]) for t = 0..m-1 in
    place (regs[m]'s lo is left there); hi0 = a register for shld(0, regs[0]) = regs[0] >> (64 - ld), or None (regs[0]
    << ld by shlx: the top limb of Y)"""
    if hi0:
        e(f'\tmulxq\t%{regs[0]}, %{regs[0]}, %{hi0}')
    else:
        e(f'\tshlxq\t%rcx, %{regs[0]}, %{regs[0]}')
    for t in range(1, len(regs)):
        e(f'\tmulxq\t%{regs[t]}, %{regs[t]}, %{tmp}')
        e(f'\torq\t%{tmp}, %{regs[t - 1]}')

# k = 1: y = shld(b_t0, b_t1), <x1, x0> = <shld(0, a_t0), shld(a_t0, a_t1)>, one divq (x1 < y); leaf, red zone
def fused_k1():
    e('.Lf1:')
    e('\tmovq\t%rax, -16(%rsp)')
    e('\tmovq\t%r8, -8(%rsp)')
    load_top('%rdx', '%r11', 0, 'r9')
    e('\tlzcntq\t%r9, %rcx')
    load_top('%rdx', '%r11', 1, 'r8')
    load_top('%rsi', '%r10', 0, 'r11')
    load_top('%rsi', '%r10', 1, 'rax')
    p_prep()
    funnel_mulx(['r9', 'r8'], None, 'r10')        # y = r9
    funnel_mulx(['r11', 'rax'], 'r8', 'r10')      # x1 = r8, x0 = r11
    e('\tmovq\t%r11, %rax')
    e('\tmovq\t%r8, %rdx')
    e('\tdivq\t%r9')
    e('\tmovq\t%rax, (%rdi)')
    e('\ttestq\t%rax, %rax')
    e('\tmovq\t-16(%rsp), %rax')
    e('\tmovq\t-8(%rsp), %rdx')
    e('\tjz\t.Lzero')
    e('\tret')

# k = 2: two exact 3/2 steps on R = X B^2 (X = <x2, x1, x0> < B Y; quotient limbs 1, 0 into z[1], z[0]), registers:
# d1 = r12, d0 = r13, v = r14, x2 r8, x1 r9, x0 rsi, q1 = r15, q0 = r8; e_top / ns in the red zone below the pushes
N2_SAVE = ['rbx', 'r12', 'r13', 'r14', 'r15']
def fused_k2(cold):
    e('.Lf2:')
    for r in N2_SAVE:
        e(f'\tpush\t%{r}')
    e('\tmovq\t%rax, -16(%rsp)')
    e('\tmovq\t%r8, -8(%rsp)')
    load_window('%rdx', '%r11', 2, ['r12', 'r13', 'rax'], -48, 'rcx', cold)
    e('\tlzcntq\t%r12, %rcx')
    p_prep()
    funnel_mulx(['r12', 'r13', 'rax'], None, 'rbx')   # y1, y0
    load_window('%rsi', '%r10', 2, ['r9', 'r11', 'rax'], -48, 'r8', cold)
    funnel_mulx(['r9', 'r11', 'rax'], 'r8', 'rbx')    # x2 = r8, x1 = r9, x0 = r11
    e('\tmovq\t%r11, %rsi')
    inv_pi1('r12', 'r13', 'r14', '2')
    udiv_3by2('r15', 'r10', 'rsi', '%r8', '%r9', '%rsi', '%r12', '%r13', '%r14', '2_2', cold)
    udiv_3by2('r8', 'r10', 'rsi', '%r10', '%rsi', '$0', '%r12', '%r13', '%r14', '2_1', cold)
    e('\tmovq\t%r8, (%rdi)')
    e('\tmovq\t%r15, 8(%rdi)')
    e('\ttestq\t%r15, %r15')
    e('\tmovq\t-16(%rsp), %rax')
    e('\tmovq\t-8(%rsp), %rdx')
    e('\tmovl\t$2, %ecx')
    for r in reversed(N2_SAVE):
        e(f'\tpop\t%{r}')
    e('\tjz\t.Lstrip')
    e('\tret')

# the reciprocal, k = 1: z = floor(2^ld B / y) (2^ld < y: the entry took Y = B / 2); leaf, red zone
def fused_i1():
    e('.Li1:')
    e('\tmovq\t%rax, -16(%rsp)')
    e('\tmovq\t%r8, -8(%rsp)')
    load_top('%rdx', '%r11', 0, 'r9')
    e('\tlzcntq\t%r9, %rcx')
    load_top('%rdx', '%r11', 1, 'r8')
    p_prep()                                      # p = 2^ld = x1
    funnel_mulx(['r9', 'r8'], None, 'rsi')        # y
    e('\txorl\t%eax, %eax')                       # x0
    e('\tdivq\t%r9')
    e('\tmovq\t%rax, (%rdi)')                     # >= 2^ld: no strip
    e('\tmovq\t-16(%rsp), %rax')
    e('\tmovq\t-8(%rsp), %rdx')
    e('\tret')

# the reciprocal, k = 2 (the prototype's .Li2 on the limb grid): Y = <d1, d0> in rsi / r9, v = invert_pi1 in r8, then
# r = B^2 - <x1, x0> with <x1, x0> the low limbs of (B + v) Y, v0 = r1 + hi(r1 v) + carry(r0 + lo(r1 v) + hi(r0 v)),
# z = <1, v, v0> 2^ld / B (top limb >= 2^ld: no strip); leaf, no callee-saved register, red zone
def fused_i2(cold):
    e('.Li2:')
    e('\tmovq\t%rax, -16(%rsp)')
    e('\tmovq\t%r8, -8(%rsp)')
    load_window('%rdx', '%r11', 2, ['rsi', 'r9', 'rax'], -48, 'rcx', cold)
    e('\tlzcntq\t%rsi, %rcx')
    p_prep()
    funnel_mulx(['rsi', 'r9', 'rax'], None, 'r10')    # d1, d0
    inv_pi1('rsi', 'r9', 'r8', 'i2')
    e('\tmovq\t%r9, %rdx')
    e('\tmulx\t%r8, %rax, %r10')                  # v d0
    e('\tmovq\t%rsi, %r11')
    e('\timulq\t%r8, %r11')                       # lo(v d1)
    e('\taddq\t%r9, %r10')
    e('\taddq\t%r11, %r10')                       # x1
    e('\tnegq\t%rax')                             # r0 = -x0, CF = x0 != 0
    e('\tmovl\t$0, %r11d')
    e('\tsbbq\t%r10, %r11')                       # r1
    e('\tmovq\t%r11, %rdx')
    e('\tmulx\t%r8, %r10, %rsi')                  # r1 v
    e('\tmovq\t%rax, %rdx')
    e('\tmulx\t%r8, %rdx, %rdx')                  # hi(r0 v)
    e('\taddq\t%r10, %rax')
    e('\tadcq\t$0, %rsi')
    e('\taddq\t%rdx, %rax')
    e('\tadcq\t$0, %rsi')
    e('\taddq\t%r11, %rsi')                       # v0
    p_prep()
    e('\tmulxq\t%r8, %rax, %r9')                  # v p
    e('\tmulxq\t%rsi, %r10, %rsi')                # hi(v0 p)
    e('\torq\t%rsi, %rax')                        # z0
    e('\torq\t%rdx, %r9')                         # z1
    e('\tmovq\t%rax, (%rdi)')
    e('\tmovq\t%r9, 8(%rdi)')
    e('\tmovq\t-16(%rsp), %rax')
    e('\tmovq\t-8(%rsp), %rdx')
    e('\tret')

# ---- k = 3..5: div_ssF<k>'s 3/2 steps (the library's port of GMP sbpi1_div_qr arithmetic) as one loop, as in the
# prototype: steps j = k .. 1; the remainder in registers with fixed roles, shifted down one register after every step:
#   n1 rsi, np1 r8, np0 r9, lo (top-indexed: T0 = the limb below np0, T1, T2) r10 r12 r13, q r14, dinv rdi,
#   Y + 8(k-1) in r15 (d1 = (r15), d0 = -8(r15), D(k-3-j) = -16-8j(r15) for T_j), the step counter in rbp.
# R = X B^k starts as n1 .. T2 = X[k] .. X[k-5] (zero below X[0]). The update of lo has L = min(k - 2, j) elements:
# one code variant per L (1, 2, 3), reached by jmp *VT[j] from a five-entry table filled per entry (k is fixed there).
# After the update: z limb j-1 = q at (z - 8) + 8j; the shift np0 <- T0 <- T1 <- T2 <- 0. Frame (CLF below the six
# pushes): 8 z - 8, 16 e_top, 24 ns, 32 z, 40 na, 48 .. 80 VT[1..5], 88 a.d, 96 k - 2, 120 .. 159 Y (k limbs).
CLF = 264                                        # rsp 16-aligned: 8 + 48 + 264 = 320; 168 .. 215, 216 .. 263: load_window's slots (b, a)

def fused_cl(k, cold, inv=False):
    sfx = f'{k}i' if inv else f'{k}'
    e(f'.Li{k}:' if inv else f'.Lf{k}:')
    for r in SAVE_REGS:
        e(f'\tpush\t%{r}')
    e(f'\tsubq\t${CLF}, %rsp')
    e('\tmovq\t%rax, 16(%rsp)')
    e('\tmovq\t%r8, 24(%rsp)')
    e('\tmovq\t%rdi, 32(%rsp)')
    if not inv:
        e('\tmovq\t%r10, 40(%rsp)')
        e('\tmovq\t%rsi, 88(%rsp)')
    # Y[k-1-t] = shld(b_t, b_t+1): b's top k + 1 limbs in Bq[0..k], shifted in place in ascending t (each uses the next
    # one unshifted); d1 = Y[k-1] in rbx, d0 = Y[k-2] in r13 for the inverse
    Bq = ['rbx', 'r13', 'r12', 'r14', 'r15', 'rbp'][:k + 1]
    load_window('%rdx', '%r11', k, Bq, 168, 'rcx', cold)
    e('\tlzcntq\t%rbx, %rcx')
    e('\ttestq\t%rcx, %rcx')                      # ld = 0: Y is b's top k limbs as they are (no copy to forward from)
    e(f'\tjz\t.Lcly0_{sfx}')
    p_prep()
    funnel_mulx(Bq, None, 'rax')
    for t in range(k):
        e(f'\tmovq\t%{Bq[t]}, {120 + 8 * (k - 1 - t)}(%rsp)')
    e(f'\tleaq\t{120 + 8 * (k - 1)}(%rsp), %r15')  # yt
    e(f'\tjmp\t.Lcly1_{sfx}')
    e(f'.Lcly0_{sfx}:')
    e('\tleaq\t-8(%rdx,%r11,8), %r15')            # yt: b's top limb (load_window's src / n)
    e(f'.Lcly1_{sfx}:')
    e('\tleaq\t-8(%rdi), %rax')                   # z - 8: limb j-1 at 8j
    e('\tmovq\t%rax, 8(%rsp)')
    inv_pi1('rbx', 'r13', 'rdi', f'cl{sfx}')         # dinv in rdi for the whole loop (rax rdx r10 r11 clobbered)
    e(f'\tmovl\t${k}, %ebp')                      # step counter j = k
    e(f'\tmovq\t${k - 2}, 96(%rsp)')              # k - 2 (the q = B - 1 path's L)
    for j in range(1, 6):                         # VT[j] = the update variant of L = min(k - 2, j)
        e(f'\tleaq\t.Lcl_s{min(k - 2, j)}(%rip), %rax')
        e(f'\tmovq\t%rax, {40 + 8 * j}(%rsp)')
    if inv:                                       # X = 2^ld B^k (rcx = ld still)
        e('\tmovl\t$1, %esi')
        e('\tshlxq\t%rcx, %rsi, %rsi')
        for r in ('r8', 'r9', 'r10', 'r12', 'r13'):
            e(f'\txorl\t%{r32(r)}, %{r32(r)}')
        e('\tjmp\t.Lcl_st')
        return
    # X[k-j] = shld(a_t(j-1), a_tj) in R[j]: a's top k + 1 limbs into R[1..k+1], R[0] = 0, shifted in ascending j
    R = ['rsi', 'r8', 'r9', 'r10', 'r12', 'r13', 'r14'][:k + 2]
    e('\tmovq\t40(%rsp), %r11')                   # na
    e('\tmovq\t88(%rsp), %rdx')                   # a.d
    load_window('%rdx', '%r11', k, R[1:], 216, 'rax', cold)
    p_prep()
    funnel_mulx(R[1:], R[0], 'rax')
    A = ['rsi', 'r8', 'r9', 'r10', 'r12', 'r13']
    for j in range(k + 1, 6):                     # below X[0]: zero
        e(f'\txorl\t%{r32(A[j])}, %{r32(A[j])}')
    e('\tjmp\t.Lcl_st')

def block_cl(cold):
    d1, d0 = '(%r15)', '-8(%r15)'
    Dl = lambda j: f'{-16 - 8 * j}(%r15)'        # D(n-3-j), T_j's limb
    # the steps
    e('.Lcl_st:')
    e(f'\tcmp\t{d1}, %rsi')
    e('\tje\t.Lqm_cl')
    e('.Lqmr_cl:')
    udiv_3by2('r14', 'rsi', 'r8', '%rsi', '%r8', '%r9', d1, d0, '%rdi', 'cl', cold)
    e('\tjmp\t*40(%rsp,%rbp,8)')                 # VT[k]
    # the update variants: T_(L-1) .. T0 -= q D(n-2-L .. n-3), the borrow into np0 (r8 now) and n1
    for L in (3, 2, 1):
        T = ['r10', 'r12', 'r13'][:L][::-1]        # low to high
        Dd = [Dl(L - 1 - i) for i in range(L)]
        e(f'.Lcl_s{L}:')
        e('\tmov\t%r14, %rdx')
        tmp = ['rax', 'r11', 'rbx', 'rcx', 'r9']
        t = [tmp.pop(0)]
        h = tmp.pop(0)
        e(f'\tmulx\t{Dd[0]}, %{t[0]}, %{h}')
        for i in range(1, L):
            ti, hn = tmp.pop(0), tmp.pop(0)
            e(f'\tmulx\t{Dd[i]}, %{ti}, %{hn}')
            e(f'\t{"add" if i == 1 else "adc"}\t%{h}, %{ti}')
            tmp.append(h)
            t.append(ti)
            h = hn
        if L > 1:
            e(f'\tadc\t$0, %{h}')
        e(f'\tsub\t%{t[0]}, %{T[0]}')
        for i in range(1, L):
            e(f'\tsbb\t%{t[i]}, %{T[i]}')
        e(f'\tadc\t$0, %{h}')
        e(f'\tsub\t%{h}, %r8')
        e('\tsbb\t$0, %rsi')
        e(f'\tjc\t.Lab_cl{L}')
        if L != 1:
            e('\tjmp\t.Lqd_cl')
    # z limb k-1 = q, the shift, the counter
    e('.Lqd_cl:')
    e('\tmov\t8(%rsp), %r11')
    e('\tmov\t%r14, (%r11,%rbp,8)')
    e('\tmov\t%r10, %r9')
    e('\tmov\t%r12, %r10')
    e('\tmov\t%r13, %r12')
    e('\tmov\t$0, %r13d')
    e('\tsub\t$1, %rbp')
    e('\tjnz\t.Lcl_st')
    e('\tmovq\t32(%rsp), %rdi')
    e('\tmovq\t16(%rsp), %rax')
    e('\tmovq\t24(%rsp), %rdx')
    e('\tmovq\t96(%rsp), %rcx')
    e(f'\taddq\t${CLF}, %rsp')
    for r in reversed(SAVE_REGS):
        e(f'\tpop\t%{r}')
    e('\taddq\t$2, %rcx')                       # k
    e('\tcmpq\t$0, -8(%rdi,%rcx,8)')
    e('\tje\t.Lstrip')
    e('\tret')
    # cold: add-back T_(L-1) .. T0 np0 (r8) += D(n-2-L .. n-2), n1 += d1 + carry, q -= 1; entered per L
    c = ['.Lab_cl2:', '\tclc', '\tjmp\t.Labe_cl2', '.Lab_cl1:', '\tclc', '\tjmp\t.Labe_cl1', '.Lab_cl3:', '\tclc',
         f'\tmov\t{Dl(2)}, %rax', '\tadc\t%rax, %r13', '.Labe_cl2:', f'\tmov\t{Dl(1)}, %rax', '\tadc\t%rax, %r12',
         '.Labe_cl1:', f'\tmov\t{Dl(0)}, %rax', '\tadc\t%rax, %r10', f'\tmov\t{d0}, %rax', '\tadc\t%rax, %r8',
         f'\tmov\t{d1}, %rax', '\tadc\t%rax, %rsi', '\tsub\t$1, %r14', '\tjmp\t.Lqd_cl']
    # cold: n1 = d1: if np1 = d0, q = B - 1 with L = min(n - 2, k): T_(L-1) .. np1 += D(n-2-L .. n-1), then
    # T_(L-2) .. np1 -= D(n-2-L .. n-2) (one limb up); n1 = np1, np1 = np0 (r8 <- r9, the normal path's r0 register)
    c += ['.Lqm_cl:', f'\tcmp\t{d0}, %r8', '\tjne\t.Lqmr_cl', '.Lqmt_cl:', '\tmov\t96(%rsp), %rcx', '\tcmp\t%rbp, %rcx',
          '\tcmova\t%rbp, %rcx', '\tcmp\t$2, %ecx', '\tjb\t.Lqml_cl1', '\tje\t.Lqml_cl2']
    for L in (3, 2, 1):
        seg = ['r10', 'r12', 'r13'][:L][::-1] + ['r9', 'r8']          # low to high
        Dg = [Dl(L - 1 - i) for i in range(L)] + [d0, d1]
        c += [f'.Lqml_cl{L}:', f'\tmov\t{Dg[0]}, %rax', f'\tadd\t%rax, %{seg[0]}']
        for i in range(1, len(seg)):
            c += [f'\tmov\t{Dg[i]}, %rax', f'\tadc\t%rax, %{seg[i]}']
        c += [f'\tmov\t{Dg[0]}, %rax', f'\tsub\t%rax, %{seg[1]}']
        for i in range(2, len(seg)):
            c += [f'\tmov\t{Dg[i - 1]}, %rax', f'\tsbb\t%rax, %{seg[i]}']
        c += ['\tjmp\t.Lqmd_cl'] if L != 1 else []
    c += ['.Lqmd_cl:', '\tmov\t%r8, %rsi', '\tmov\t%r9, %r8', '\tmov\t$-1, %r14', '\tjmp\t.Lqd_cl']
    cold += c

# ---- n >= 6: divsw.hpp's algorithm (div_sw<n>, word for word, in its step order) as one shared code path (candidate B
# in its B-new form): register top window w0..w5 (positions k+n-4 .. k+n+1, kept complemented), product-only
# low-biased estimates q = w4 + hi(w4 v1) + carry(hi(w3 v1) + hi(w4 v0) + lo(w4 v1) + w3) from V (inv_window: one 3/2
# inverse and four products), the window update (positions k+n-5 .. k+n) in registers in two parts with two carry chains
# each (the product sum on OF with adox, the accumulation on CF with adcx): the top part (bases w2 .. w4, slack 2) right
# after the estimate, the bottom part (bases F, w0, w1) after step k+1's lower chain, which runs here (positions >= n-2,
# <= k+n-5, memory; its top limb F stays in a register) with its carry bits applied to w0 at once; the bottom part's
# carry bits correct w3 in the next step after its estimate. Quotient limbs k = n .. 1 (z uses limbs 1 .. n; step 1
# only estimates). Cold paths: negative window lower bound -> q = 0; window top limb > 0 or an estimate >= B ->
# R -= D B^(k+1), limb k+1 += 1, estimate again; carries of the corrections. Output: z = limbs 1 .. n shifted right by
# the half flag (shrd), raised to B^n / 2 when its top bit is clear.
# The remainder is kept complemented (R' = ~R) in registers and memory, so every update is an addition R' + P with
# carries (R - P - c = ~(R' + P + c)); a pending correction delta (added to R) is subtracted from R'. Step k in
# divsw.hpp's order: checks and estimate; delta_top of step k+1 at w3; the top part (j = 3..5); the lower chain of step
# k+1 (multiplier qp) and its delta at w0; the bottom part (j = 0..2), whose carry bits are delta_top for step k-1.
# cmp C80, %rdi sets CF = OF = 1 (rdi = yb is a user pointer, below 2^63); test %rdi, %rdi sets CF = OF = 0.
# Registers: rdi = yb = y + 8 (n-6) (every D access is yb-relative: D(n-6+i) at 8 i(%rdi)); r14 = P = rsp + 8 k_p, so
# every slot of step k = k_p - p (body position p) sits at a fixed displacement from P; r8 .. r13 the window, rotating
# with period 6 (W(p)[j] = RB[(j - p) mod 6]); r15 = the estimate q_k; rbx = qp, step k+1's estimate (the lower chain's
# multiplier, never re-read from Qd: an overflow fix may have raised Qd[k+1]), then q_k for step k-1; rbp = delta_top
# across steps, inside a step keepL = lo(q_k D(n-3)) from the top part for the bottom part; rax rcx rsi temporaries; the
# lower chain writes rax rbp r15 rcx; zero and the constants as memory operands.
# Steps: the first three k = n, n-1, n-2 (shared, peeled; k = n takes V from rbx / rbp and skips the top-limb test;
# their lower chain has n - 5 elements: entry and index from slots set once per call); the stable part k = n-3 .. 4 as
# a BP-step body (the lcm of the 6 window positions and the GS elements of a lower-chain group), entered once per call
# at position p0 = (6 - n) mod BP so that k = 4 falls on position BP - 1: at position p, k = k_p - p with
# k_p = 3 mod BP, the lower chain of step k+1 has k - 2 elements and starts at group element p, so each call site has
# one fixed entry (.Llcg<p>); n = 6 has no body steps (p0 = BP: straight to the tail, positions BP .. BP + 2); the
# last three k = 3 (step 4's one-element chain inline), 2, 1. The lower chain is one shared body (lc_regs; lea / jrcxz /
# jmp between groups: no flag writes, so both ADX carries cross groups), called at a fixed entry once the caller has
# zeroed that entry's hp. Cold paths: the overflow fix's memory part and Qd's carry as shared loops (ov_loops).
BP = 6                            # the body period
GS = BP                           # the lower chain's group size
RB = ['r8', 'r9', 'r10', 'r11', 'r12', 'r13']
PR = 'r14'
DTR = 'rbp'
ZEROS, NKP, NNS, LCC, LCE, OVC = V1 + 48, V1 + 56, V1 + 72, V1 + 80, V1 + 88, V1 + 96   # a zero, -k_p, n, the peel
                                  # chain's index and entry, the peel's overflow index -(n-6)

def wb(p):
    return [RB[(j - p) % 6] for j in range(6)]

def dlt_apply(W, lab, cold, zero='%rdi'):
    """the lower chain's carry bits (CF, OF) into w0: W0' += CF + OF - 2, carried through w1 .. w4 (cold)"""
    e(f'\tadcx\t{M2}(%rsp), %{W[0]}')
    e(f'\tadox\t{zero}, %{W[0]}')
    e(f'\tjnc\t.Ldl_{lab}')
    e(f'\tjo\t.Ldo_{lab}')
    e(f'.Ldlr_{lab}:')
    cold += [f'.Ldl_{lab}:', f'\tjo\t.Ldlr_{lab}', f'.Ldlb_{lab}:', f'\tsub\t$1, %{W[1]}'] + [f'\tsbb\t$0, %{W[i]}' for i in (2, 3, 4)] + \
            [f'\tjmp\t.Ldlr_{lab}']
    cold += [f'.Ldo_{lab}:', f'\tadd\t$1, %{W[1]}'] + [f'\tadc\t$0, %{W[i]}' for i in (2, 3, 4)] + [f'\tjmp\t.Ldlr_{lab}']
LCR = [('rax', 'rbp', 'r15'), ('r15', 'rax', 'rbp'), ('rbp', 'r15', 'rax')]     # B_regs: (lo, hi, hp) for j mod 3

def lc_regs(pre, gs, stubs):
    """B_regs's lower chain: as lc_shared (B addressing: rsi = the last element's M address, M and D at index rcx + j)
    with groups of gs elements (a multiple of 3) and three rotating temporaries (LCR: element j's hi is element j+1's
    hp), group element j at lcg<j>, the one-element chain at lct; writes rax rbp r15 rcx only (rbx, the window r8 .. r13,
    r14, rdi, rsi, rdx kept). F in rax, the carries in CF / OF. stubs: entries lce<j> and lcet that zero hp first (the
    probe's B_regs); without them (B-new: the routine, the probe's variants 3, 4) the caller zeroes hp"""
    L = lambda name: f'.L{pre}{name}'
    e('\t.p2align\t5')
    for j in range(gs):
        lo, hi, hp = LCR[j % 3]
        e(f'{L("lcg")}{j}:')
        e(f'\tmulx\t{8 * j}(%rdi,%rcx,8), %{lo}, %{hi}')
        e(f'\tadox\t%{hp}, %{lo}')
        e(f'\tadcx\t{8 * j}(%rsi,%rcx,8), %{lo}')
        e(f'\tmov\t%{lo}, {8 * j}(%rsi,%rcx,8)')
    e(f'\tlea\t{gs}(%rcx), %rcx')
    e(f'\tjrcxz\t{L("lct")}')
    e(f'\tjmp\t{L("lcg")}0')
    e(f'{L("lct")}:')                             # after the group's last element (hi in r15) or the one-element entry
    e('\tmulx\t(%rdi), %rax, %rbp')
    e('\tadox\t%r15, %rax')
    e('\tadcx\t(%rsi), %rax')
    e('\tret')
    if not stubs:
        return
    for j in range(gs):
        e(f'{L("lce")}{j}:')
        e(f'\tmov\t$0, %{LCR[j % 3][2]}')
        e(f'\tjmp\t{L("lcg")}{j}')
    e(f'{L("lcet")}:')
    e('\tmov\t$0, %r15')
    e(f'\tjmp\t{L("lct")}')

def ov_loops():
    # overflow fix, memory part: complemented M[i] += D(n-6+i) + CF for i = rcx .. 0 (rcx <= 0), CF out
    e('.Lovm:')
    e('\tmov\t(%rdi,%rcx,8), %rax')
    e('\tadc\t%rax, (%rsi,%rcx,8)')
    e('\tjrcxz\t.Lovmd')
    e('\tlea\t1(%rcx), %rcx')
    e('\tjmp\t.Lovm')
    e('.Lovmd:')
    e('\tret')
    # overflow fix, quotient: Qd[k+1 ..] += 1 (rsi = &Qd[k+1], rcx = n - k >= 1 limbs)
    e('.Lqinc:')
    e('\taddq\t$1, (%rsi)')
    e('\tjnc\t.Lqincd')
    e('\tlea\t8(%rsi), %rsi')
    e('\tlea\t-1(%rcx), %rcx')
    e('\tjrcxz\t.Lqincd')
    e('\tjmp\t.Lqinc')
    e('.Lqincd:')
    e('\tret')
def step_b(p, kind, cold):
    """step k = k_p - p (P = rsp + 8 k_p); kind: first (k = n), peel (k = n-1, n-2), body (k = n-3 .. 4), t3, t2, t1"""
    W = wb(p)
    lab = f'b{p}' if kind != 'first' and kind != 'peel' else f'bp{p}'
    disp = lambda off: f'{off - 8 * p}(%{PR})'
    qd = disp(QO)                                  # Qd[k]
    mend = disp(8)                                 # M at position k + n - 5 (the lower chain's last element / F)
    e(f'# B step ({kind}), position {p}: window {W}')
    e(f'.Lst_{lab}:')
    first = kind == 'first'                       # v1 / v0 still in rbx / rbp; W5 is the complemented zero of
    v1o, v0o = ('%rbx', '%rbp') if first else (f'{V1}(%rsp)', f'{V0}(%rsp)')   # position 2n+1 (no test)
    if not first:
        e(f'\tcmp\t$-1, %{W[5]}')
        e(f'\tjne\t.Lw5_{lab}')
    e(f'\tmov\t%{W[4]}, %rdx')                   # the estimate into r15 (rbx holds qp across the step)
    e('\tnot\t%rdx')
    e(f'\tmulx\t{v1o}, %rax, %r15')
    e(f'\tmulx\t{v0o}, %rcx, %rcx')
    e('\tadd\t%rdx, %r15')
    e(f'\tjc\t.Lov_{lab}')
    e(f'\tmov\t%{W[3]}, %rdx')
    e('\tnot\t%rdx')
    e(f'\tmulx\t{v1o}, %rsi, %rsi')
    e('\tcmp\t$-4, %r15')
    e(f'\tjae\t.Les_{lab}')
    e('\tadd\t%rax, %rdx')
    e('\tadc\t$0, %r15')
    e('\tadd\t%rcx, %rsi')
    e('\tadc\t$0, %r15')
    e('\tadd\t%rdx, %rsi')
    e('\tadc\t$0, %r15')
    e(f'.Lq_{lab}:')
    e(f'\tmov\t%r15, {qd}')
    # cold: window top limb != 0 (true w5 = ~W5): negative -> q = 0, else overflow
    if not first:
        cold += [f'.Lw5_{lab}:', f'\ttest\t%{W[5]}, %{W[5]}', f'\tjs\t.Lov_{lab}', f'.Lng_{lab}:', '\txor\t%r15d, %r15d',
                 f'\tjmp\t.Lq_{lab}']
    # cold: base w4 + hi(w4 v1) >= B - 4: the column carries with overflow checks
    cold += [f'.Les_{lab}:', '\tadd\t%rax, %rdx', '\tadc\t$0, %r15', f'\tjc\t.Lov_{lab}', '\tadd\t%rcx, %rsi',
             '\tadc\t$0, %r15', f'\tjc\t.Lov_{lab}', '\tadd\t%rdx, %rsi', '\tadc\t$0, %r15', f'\tjc\t.Lov_{lab}',
             f'\tjmp\t.Lq_{lab}']
    # overflow: complemented R += D B^(k+1) at positions >= n-2 (memory part, then w0 .. w5), Qd[k+1 ..] += 1, again
    c_ = [f'.Lov_{lab}:']
    if kind == 'first':
        c_ += ['\tud2']
    else:
        c_ += ['\tclc']
        if kind in ('peel', 'body', 't3'):
            if kind == 'peel':
                c_ += [f'\tmov\t{OVC}(%rsp), %rcx']                  # -(n-6): positions k+1 .. k+n-5
            elif kind == 'body':
                c_ += [f'\tmov\t{NKP}(%rsp), %rcx', f'\tlea\t{p + 3}(%rcx), %rcx']   # -(k-3): positions n-2 .. k+n-5
            else:
                c_ += ['\tmov\t$0, %ecx']                           # k = 3: position n-2 only
            c_ += [f'\tlea\t{mend}, %rsi', '\tcall\t.Lovm']
        lo = 2 if kind == 't1' else 1                                # k = 1: w0 is position n-3 (below n-2)
        if lo == 2:
            c_ += [f'\tadc\t{8 * j}(%rdi), %{W[j - 1]}' for j in range(2, 6)]
        else:
            c_ += [f'\tadc\t{8 * j}(%rdi), %{W[j - 1]}' for j in range(1, 6)]
        c_ += [f'\tadc\t$0, %{W[5]}']
        if kind == 'peel':
            c_ += [f'\tmov\t${p}, %ecx']                            # n - k = p (P = rsp + 8n in the peel)
        else:
            c_ += [f'\tmov\t{NNS}(%rsp), %rcx', f'\tadd\t{NKP}(%rsp), %rcx', f'\tadd\t${p}, %rcx']
        c_ += [f'\tlea\t{disp(QO + 8)}, %rsi', '\tcall\t.Lqinc', f'\tjmp\t.Lst_{lab}']
    cold += c_
    if kind == 't1':
        return
    e('\tmov\t%r15, %rdx')
    if kind != 'first':                           # delta_top of step k+1 at w3: W3' += dt - 2 (dt = its carry bits)
        e(f'\tcmp\t$2, %{W[3]}')
        e(f'\tjb\t.Ldt_{lab}')
        e(f'\tlea\t-2(%{W[3]},%{DTR}), %{W[3]}')
        e(f'.Ldtr_{lab}:')
        cold += [f'.Ldt_{lab}:', f'\tsub\t$2, %{W[3]}', f'\tsbb\t$0, %{W[4]}', f'\tadd\t%{DTR}, %{W[3]}', f'\tadc\t$0, %{W[4]}',
                 f'\tjmp\t.Ldtr_{lab}']
    # top part (j = 3..5, positions k+n-2 .. k+n, bases w2, w3, w4), slack 2 for the bottom part's borrow
    e(f'\tcmp\t{C80}(%rsp), %rdi')
    e(f'\tmulx\t24(%rdi), %{W[5]}, %rax')       # keepL = lo(q D(n-3)) for the bottom part (the old top limb is dead)
    e('\tmulx\t32(%rdi), %rcx, %rsi')
    e('\tadox\t%rax, %rcx')
    e(f'\tadcx\t%rcx, %{W[2]}')
    e('\tmulx\t40(%rdi), %rax, %rbp')
    e('\tadox\t%rsi, %rax')
    e(f'\tadcx\t%rax, %{W[3]}')
    e(f'\tadox\t{ZEROS}(%rsp), %rbp')
    e(f'\tadcx\t%rbp, %{W[4]}')
    # step k+1's lower chain and its delta at w0 (W0' += CF + OF - 2); F into the freed top register
    if kind in ('peel', 'body', 't3'):
        e('\tmov\t%rbx, %rdx')                     # qp: step k+1's estimate (never raised by an overflow fix)
        e('\tmov\t%r15, %rbx')                     # q_k, kept across the chain and into step k-1
        r32 = {'rax': 'eax', 'rbp': 'ebp', 'r15': 'r15d'}   # the entry's hp zeroed by xor (also CF = OF = 0)
        if kind == 'peel':                        # the entry varies with n: every hp register
            e(f'\tmov\t{LCC}(%rsp), %rcx')
            e(f'\tlea\t{mend}, %rsi')
            e('\txor\t%eax, %eax')
            e('\txor\t%ebp, %ebp')
            e('\txor\t%r15d, %r15d')
            e(f'\tcall\t*{LCE}(%rsp)')
        elif kind == 'body':
            e(f'\tmov\t{NKP}(%rsp), %rcx')
            e('\tlea\t3(%rcx), %rcx')
            e(f'\tlea\t{mend}, %rsi')
            hp = r32[LCR[p % 3][2]]
            e(f'\txor\t%{hp}, %{hp}')
            e(f'\tcall\t.Llcg{p}')
        else:                                     # t3: the one-element chain inline (hp = 0, OF = 0 from the test)
            e('\ttest\t%rdi, %rdi')
            e('\tmulx\t(%rdi), %rax, %rbp')
            e(f'\tadcx\t{mend}, %rax')
        dlt_apply(W, lab, cold, zero=f'{ZEROS}(%rsp)')
        e(f'\tmov\t%{W[5]}, %rbp')                 # keepL
        e(f'\tmov\t%rax, %{W[5]}')
        e('\tmov\t%rbx, %rdx')                     # q_k
    else:                                         # first (k = n) and t2 (k = 2): the base comes from memory
        if kind == 'first':
            e('\tmov\t%r15, %rbx')                 # q_n: step n-1's qp
        e(f'\tmov\t%{W[5]}, %rbp')                 # keepL (rdx = q_k since the top part)
        e(f'\tmov\t{mend}, %{W[5]}')
    # bottom part (j = 0..2, positions k+n-5 .. k+n-3, bases F, w0, w1; q_k in rdx and rbx, keepL in rbp); slack 2
    # when step k has a lower chain (k >= 4)
    if kind in ('t3', 't2'):
        e('\ttest\t%rdi, %rdi')
    else:
        e(f'\tcmp\t{C80}(%rsp), %rdi')
    e('\tmulx\t(%rdi), %rax, %rax')
    e('\tmulx\t8(%rdi), %rcx, %rsi')
    e('\tadox\t%rax, %rcx')
    e(f'\tadcx\t%rcx, %{W[5]}')
    e('\tmulx\t16(%rdi), %rax, %r15')
    e('\tadox\t%rsi, %rax')
    e(f'\tadcx\t%rax, %{W[0]}')
    e('\tadox\t%r15, %rbp')
    e(f'\tadcx\t%rbp, %{W[1]}')
    if kind != 't2':                              # k >= 3: delta_top for step k-1 (step 1 does not apply it)
        e('\tmov\t$0, %ebp')
        e(f'\tadcx\t{ZEROS}(%rsp), %{DTR}')
        e(f'\tadox\t{ZEROS}(%rsp), %{DTR}')

def v_setup_b(cold):
    """V for E = <D(n-1), D(n-2)> + 1 from yb-relative operands (inv_window), run first (rcx = n, rsi = x, r15 = y,
    rdi = yb live): v1 in rbx, v0 in rbp, also stored at V1 / V0; temporaries r8 r9 (and inv_pi1's)"""
    e('\tmov\t40(%rdi), %r12')
    e('\tmov\t32(%rdi), %r13')
    e('\tadd\t$1, %r13')
    e('\tadc\t$0, %r12')
    e('\tjc\t.Lvzero_b')
    inv_pi1('r12', 'r13', 'rbx', 'wb')
    e('\tmov\t%r13, %rdx')
    e('\tmulx\t%rbx, %rax, %r8')
    e('\tmov\t%r12, %r9')
    e('\timul\t%rbx, %r9')
    e('\tadd\t%r13, %r8')
    e('\tadd\t%r9, %r8')
    e('\tneg\t%rax')
    e('\tmov\t$0, %r9')
    e('\tsbb\t%r8, %r9')
    e('\tmov\t%r9, %rdx')
    e('\tmulx\t%rbx, %r8, %rbp')
    e('\tmov\t%rax, %rdx')
    e('\tmulx\t%rbx, %rdx, %rdx')
    e('\tadd\t%r8, %rax')
    e('\tadc\t$0, %rbp')
    e('\tadd\t%rdx, %rax')
    e('\tadc\t$0, %rbp')
    e('\tadd\t%r9, %rbp')
    e('.Lvdone_b:')
    e(f'\tmov\t%rbx, {V1}(%rsp)')
    e(f'\tmov\t%rbp, {V0}(%rsp)')
    cold += ['.Lvzero_b:', '\txor\t%ebx, %ebx', '\txor\t%ebp, %ebp', '\tjmp\t.Lvdone_b']
def bnew_setup(cold):
    """the setup: V first (its divq chain overlaps the rest; v1 / v0 stay in rbx / rbp), the complemented zero at
    positions n-6 .. n-1, ~X[0 .. n-5] at positions n .. 2n-5 (M) by one loop and ~X[n-4 .. n] (positions 2n-4 .. 2n)
    straight into the window registers; position 2n+1 is neither stored nor loaded (the first step does not test it
    and keepL overwrites its register)"""
    e('\tmov\t%rdx, %r15')                        # y
    v_setup_b(cold)                               # v1 = rbx, v0 = rbp
    e(f'\tcmpq\t$0, {MB + 32}(%rsp)')
    e('\tje\t.Lbs_inv')
    for i in range(6):                            # complemented zero at positions n-6 .. n-1
        e(f'\tmovq\t$-1, {8 * i}(%rsp)')
    e('\tlea\t-32(%rsi,%rcx,8), %rsi')            # X + 8 (n-4): X[n-4+j] at 8j
    e('\tlea\t16(%rsp,%rcx,8), %r11')             # the slot of position 2n-4
    e('\tmov\t$4, %eax')
    e('\tsub\t%rcx, %rax')                        # -(n-4) <= -2
    e('.Lbs:')
    e('\tmov\t(%rsi,%rax,8), %r8')
    e('\tnot\t%r8')
    e('\tmov\t%r8, (%r11,%rax,8)')
    e('\tinc\t%rax')
    e('\tjnz\t.Lbs')
    W = wb(0)
    for j in range(5):
        e(f'\tmov\t{8 * j}(%rsi), %{W[j]}')
        e(f'\tnot\t%{W[j]}')
    e('\tjmp\t.Lbs_ready')
    e('.Lbs_inv:')                                 # ~X below the top is all ones, directly in the working representation
    m_vectors = (NMAX + 2 + 7) // 8
    assert m_vectors * 64 <= QO
    e('\tvpternlogd\t$255, %zmm0, %zmm0, %zmm0')
    for i in range(m_vectors):
        e(f'\tvmovdqu64\t%zmm0, {64 * i}(%rsp)')
    e('\tvzeroupper')
    for reg in W[:4]:
        e(f'\tmovq\t$-1, %{reg}')
    e(f'\tmovq\t{MB + 40}(%rsp), %{W[4]}')
    e(f'\tnotq\t%{W[4]}')
    e('.Lbs_ready:')
    e(f'\tmovq\t$-2, {M2}(%rsp)')
    e('\tmovabs\t$0x8000000000000000, %rax')
    e(f'\tmov\t%rax, {C80}(%rsp)')
    e(f'\tmovq\t$0, {ZEROS}(%rsp)')
    # the peel's lower chain (n - 5 elements): entry .Lbp0[n] (BP for n = 6: .Llct), index -(n-6) - j0; overflow -(n-6)
    e(f'\tmov\t{NNS}(%rsp), %rcx')
    e('\tmov\t$6, %edx')
    e('\tsub\t%rcx, %rdx')
    e(f'\tmov\t%rdx, {OVC}(%rsp)')
    e('\tlea\t.Lbp0(%rip), %rax')
    e('\tmovzbl\t(%rax,%rcx), %eax')
    e('\tsub\t%rax, %rdx')
    e(f'\tmov\t%rdx, {LCC}(%rsp)')
    e('\tlea\t.Lblce(%rip), %rsi')
    e('\tmovslq\t(%rsi,%rax,4), %rax')
    e('\tadd\t%rsi, %rax')
    e(f'\tmov\t%rax, {LCE}(%rsp)')
    e(f'\tlea\t(%rsp,%rcx,8), %{PR}')            # P = rsp + 8n (k_p = n in the peel); NKP is set at the body entry

def block_b(cold):
    e('.Lb:')                                     # rcx = n, rdi = z, rsi = X, rdx = Y; frame set up by .Ldcore
    e(f'\tmov\t%rcx, {NNS}(%rsp)')
    e(f'\tmov\t%rdi, {ZPTR}(%rsp)')
    e('\tlea\t-48(%rdx,%rcx,8), %rdi')            # yb
    bnew_setup(cold)
    block_b_steps(cold)

def block_b_steps(cold):
    step_b(0, 'first', cold)
    step_b(1, 'peel', cold)
    step_b(2, 'peel', cold)
    # into the body: P = rsp + 8 k_p, k_p = n - 3 + p0, NKP = -k_p; registers from phase 3 to phase p0 mod 6 (p0 in
    # r15 and the permutation by its cycles with rax as the only temporary: rbx = qp, rbp = dt live)
    e(f'\tmov\t{NNS}(%rsp), %rcx')
    e('\tlea\t.Lbp0(%rip), %rax')
    e('\tmovzbl\t(%rax,%rcx), %r15d')
    e('\tlea\t-3(%rcx,%r15), %rax')
    e(f'\tlea\t(%rsp,%rax,8), %{PR}')
    e('\tneg\t%rax')
    e(f'\tmov\t%rax, {NKP}(%rsp)')
    e('\tlea\t.Lbent(%rip), %rax')
    e('\tmovslq\t(%rax,%r15,4), %rdx')
    e('\tadd\t%rax, %rdx')
    e('\tjmp\t*%rdx')
    for p0 in range(BP + 1):                      # p0 = BP (n = 6): phase 3 -> 0, straight to the tail step k = 3
        d = (3 - p0) % 6
        e(f'.Lbe{p0}:')
        if d:                                     # new RB[a + d] = old RB[a], one cycle at a time
            seen = set()
            for c in range(6):
                if c in seen:
                    continue
                cyc = [c]
                while (cyc[-1] + d) % 6 != c:
                    cyc.append((cyc[-1] + d) % 6)
                seen.update(cyc)
                e(f'\tmov\t%{RB[cyc[-1]]}, %rax')
                for i in range(len(cyc) - 1, 0, -1):
                    e(f'\tmov\t%{RB[cyc[i - 1]]}, %{RB[cyc[i]]}')
                e(f'\tmov\t%rax, %{RB[c]}')
        e(f'\tjmp\t.Lst_b{p0}')
    for p in range(BP):
        step_b(p, 'body', cold)
    e(f'\tcmpq\t${-(BP + 3)}, {NKP}(%rsp)')
    e('\tje\t.Lbt')
    e(f'\taddq\t${BP}, {NKP}(%rsp)')
    e(f'\tlea\t{-8 * BP}(%{PR}), %{PR}')
    e('\tjmp\t.Lst_b0')
    e('.Lbt:')
    step_b(BP, 't3', cold)
    step_b(BP + 1, 't2', cold)
    step_b(BP + 2, 't1', cold)
    bnew_output(cold)
def bnew_output(cold):
    # output: z_i = Qd[i+1], i = 0 .. n-1; the frame is released here, the fp result from the entry's slots
    e(f'\tmov\t{ZPTR}(%rsp), %r11')
    e(f'\tmov\t{NNS}(%rsp), %r9')
    e(f'\tlea\t{QO + 8}(%rsp,%r9,8), %r10')      # &Qd[n+1]
    e('\tlea\t(%r11,%r9,8), %r11')               # &z[n]
    e('\tneg\t%r9')
    e('.Lbo:')
    e('\tmov\t(%r10,%r9,8), %rax')
    e('\tmov\t%rax, (%r11,%r9,8)')
    e('\tinc\t%r9')
    e('\tjnz\t.Lbo')
    e(f'\tmovq\t{MB + 16}(%rsp), %rdi')
    e(f'\tmovq\t{MB}(%rsp), %rax')
    e(f'\tmovq\t{MB + 8}(%rsp), %rdx')
    e(f'\tmovq\t{MB + 24}(%rsp), %rcx')
    e(f'\taddq\t${FRAMEB}, %rsp')
    for r in reversed(SAVE_REGS):
        e(f'\tpop\t%{r}')
    e('\tcmpq\t$0, -8(%rdi,%rcx,8)')
    e('\tje\t.Lstrip')
    e('\tret')
    lc_regs('', GS, False)
    ov_loops()

def b_tables():
    e('\t.p2align\t2')
    e('.Lblce:')                                  # the peel chain's entry by group element j0; GS (n = 6): .Llct
    for j in range(GS):
        e(f'\t.long\t.Llcg{j}-.Lblce')
    e('\t.long\t.Llct-.Lblce')
    e('.Lbent:')
    for p0 in range(BP + 1):                      # BP: n = 6, no body steps (to the tail)
        e(f'\t.long\t.Lbe{p0}-.Lbent')
    e('.Lbp0:')                                   # p0 = (6 - n) mod BP; n = 6: BP, the peel's entry .Llct and the tail stub
    e('\t.byte\t' + ', '.join(str(BP if n == 6 else (BP + 6 - n % BP) % BP) for n in range(NMAX + 1)))

# ---- the cores: .Ldcore, rdi = z, rsi = X (n + 1 limbs), rdx = Y (n limbs, normalized), rcx = n in 1..NMAX; writes
# z[0..n), returns (called)
def cores():
    # k = 6..NMAX: state as at .Lf<k> (rsi = a.d, r10 = na, rdx = b.d, r11 = nb, rax = e_top, r8 = ns, r9 = k,
    # rdi = z); the windows go into the core's own frame: X shifted always, Y shifted unless ld = 0 and nb >= k (then
    # b's top k limbs as they are: no stores for the core's many D reads to forward from)
    e(f'\t.skip\t{CORE_PAD}, 0xcc')                 # the cores at their placement (mod 64) before the Karp-Markstein route
    e('.Lbe:')
    for r in SAVE_REGS:
        e(f'\tpush\t%{r}')
    e(f'\tsubq\t${FRAMEB}, %rsp')
    for off, r in ((0, 'rax'), (8, 'r8'), (16, 'rdi'), (24, 'r9'), (32, 'r10'), (40, 'rsi')):
        e(f'\tmovq\t%{r}, {MB + off}(%rsp)')
    e('\tlzcntq\t-8(%rdx,%r11,8), %rcx')          # ld
    e('\tvpbroadcastq\t%rcx, %zmm7')
    e('\ttestq\t%r10, %r10')
    e('\tjz\t.Lbinv')
    e('.Lbinvr:')
    e('\ttestq\t%rcx, %rcx')
    e('\tjnz\t.Lbey')
    e('\tcmpq\t%r9, %r11')
    e('\tjb\t.Lbey')
    e('\tsubq\t%r9, %r11')
    e('\tleaq\t(%rdx,%r11,8), %rax')              # b's top k limbs
    e('\tjmp\t.Lbex')
    e('.Lbey:')
    e(f'\tleaq\t{YB}(%rsp), %rdi')
    funnel_vector('%r11', top_from_b=True, pre='B')
    e(f'\tleaq\t{YB}(%rsp), %rax')
    e('.Lbex:')
    e(f'\tmovq\t%rax, {MB + 48}(%rsp)')
    e(f'\tcmpq\t$0, {MB + 32}(%rsp)')
    e('\tje\t.Lbxd')
    e(f'\tleaq\t{XB}(%rsp), %rdi')
    e(f'\tmovq\t{MB + 40}(%rsp), %rdx')
    funnel_vector(f'{MB + 32}(%rsp)', top_from_b=False, pre='B')
    e('.Lbxd:')
    e('\tvzeroupper')
    e(f'\tmovq\t{MB + 16}(%rsp), %rdi')
    e(f'\tleaq\t{XB}(%rsp), %rsi')
    e(f'\tmovq\t{MB + 48}(%rsp), %rdx')
    e(f'\tmovq\t{MB + 24}(%rsp), %rcx')
    e('\tjmp\t.Lb')
    e('.Lbinv:')                                  # X = 2^ld B^k; preserve p instead of materializing X
    e('\tmovl\t$1, %esi')
    e('\tshlxq\t%rcx, %rsi, %rsi')
    e(f'\tmovq\t%rsi, {MB + 40}(%rsp)')             # a.d is unused for the reciprocal
    e('\tjmp\t.Lbinvr')
    fused_k1()
    fused_i1()
    for name, gen in (('k = 2', fused_k2), ('the reciprocal, k = 2', fused_i2), ('the classic loop (k = 3..5)', block_cl),
                      ('k >= 6', block_b)):
        cold = []
        gen(cold)
        e(f'# cold paths, {name}')
        out.extend(cold)
    cold = []
    for k in (3, 4, 5):
        fused_cl(k, cold)
        fused_cl(k, cold, inv=True)
    e('# cold paths, the small entries')
    out.extend(cold)

# ---- the entries. Common body .Lbody: rsi = a.d, r10 = na, r9 = k (>= 1), rdx = b.d, r11 = nb, rax = e_top,
# r8 = the sign word (bit 63), rdi = z. Frame (rbp): -8 z, -16 e_top, -24 ns = sign | k, -32 k; X at rsp (k + 1
# limbs), Y at rsp + 8 (k + 1) (k limbs), rsp 64-aligned.
def entries():
    e('\t.text')
    e('\t.globl\tSYM(div)')
    e('\t.type\tSYM(div), @function')
    e('\t.p2align\t5')
    e('SYM(div):')
    e('\tmovq\t16(%rsi), %r10')                   # a.ns
    e('\tmovq\t16(%rdx), %r11')                   # b.ns
    e('\tmovq\t8(%rsi), %rax')
    e('\tsubq\t8(%rdx), %rax')
    e('\tincq\t%rax')                             # e_top = ea - eb + 1
    e('\tmovq\t(%rsi), %rsi')                     # a.d
    e('\tmovq\t(%rdx), %rdx')                     # b.d
    e('\tmovq\t%r10, %r8')
    e('\txorq\t%r11, %r8')                        # sign in bit 63
    e('\tbtrq\t$63, %r10')                        # na
    e('\tbtrq\t$63, %r11')                        # nb
    e('\tjmp\t.Lleps')
    e('\t.size\tSYM(div), .-SYM(div)')
    e('')
    e('\t.globl\tSYM(inv_base)')                  # the reciprocal's base path: sbn3f_inv (inv.S) hands over
    e('\t.hidden\tSYM(inv_base)')                 # k <= 3, k = 4 at ld > 60 and k > 256
    e('\t.type\tSYM(inv_base), @function')
    e('\t.p2align\t5')
    e('SYM(inv_base):')                           # X = 2^ld B^k
    e('\tmovq\t%rdx, %rcx')                       # leps
    e('\tmovq\t16(%rsi), %r11')                   # b.ns
    e('\tmovl\t$1, %eax')
    e('\tsubq\t8(%rsi), %rax')                    # e_top = 1 - eb
    e('\tmovq\t(%rsi), %rdx')                     # b.d
    e('\tmovq\t%r11, %r8')                        # sign | nb
    e('\tbtrq\t$63, %r11')                        # nb
    e('\tsubq\t$2, %rcx')
    e('\tsarq\t$6, %rcx')                         # L(2)
    e('\tmovq\t%rax, %r9')
    e('\tsubq\t%rcx, %r9')                        # k
    e('\tjle\t.Lzero')
    e(f'\tcmpq\t${KMAX}, %r9')
    e('\tja\t.Lbad')
    e('\tsubq\t%r11, %r8')
    e('\taddq\t%r9, %r8')                        # ns = sign | k
    e('\tcmpq\t$1, -8(%rdx,%r11,8)')
    e('\tje\t.Lip')                               # b's top limb 1: Y may be B^k / 2
    e('.Lipr:')
    e('\tcmpq\t$2, %r9')
    e('\tjb\t.Li1')
    e('\tje\t.Li2')
    e('\tcmpq\t$5, %r9')
    e('\tja\t.Ligen')
    e('\tleaq\t.Litab(%rip), %rcx')
    e('\tmovslq\t-12(%rcx,%r9,4), %r9')
    e('\taddq\t%rcx, %r9')
    e('\tjmp\t*%r9')
    # Y = B^k / 2 when b's window is 1, k - 1 zero limbs and a last limb <= 1 (zero below b): z = 1 at e_top + 1
    e('.Lip:')
    e('\tleaq\t-2(%r11), %rcx')                   # the limb below the top
    e('\tmovq\t%r11, %rsi')
    e('\tsubq\t%r9, %rsi')
    e('\tdecq\t%rsi')                             # the window's last limb: nb - 1 - k
    e('.Lipl:')
    e('\ttestq\t%rcx, %rcx')
    e('\tjs\t.Lip1')
    e('\tcmpq\t%rsi, %rcx')
    e('\tje\t.Lipw')
    e('\tcmpq\t$0, (%rdx,%rcx,8)')
    e('\tjne\t.Lipr')
    e('\tdecq\t%rcx')
    e('\tjmp\t.Lipl')
    e('.Lipw:')
    e('\tcmpq\t$1, (%rdx,%rcx,8)')
    e('\tja\t.Lipr')
    e('.Lip1:')
    e('\tmovq\t$1, (%rdi)')
    e('\tincq\t%rax')
    e('\tshrq\t$63, %r8')
    e('\tshlq\t$63, %r8')
    e('\tleaq\t1(%r8), %rdx')
    e('\tret')
    e('.Ligen:')
    e('\txorl\t%r10d, %r10d')                     # na = 0: the reciprocal for .Lbe and qv
    e('\tjmp\t.Lgen')
    e('.Lleps:')
    e('\tsubq\t$2, %rcx')
    e('\tsarq\t$6, %rcx')                         # L(2)
    e('\tmovq\t%rax, %r9')
    e('\tsubq\t%rcx, %r9')                        # k
    e('\tjle\t.Lzero')
    e(f'\tcmpq\t${KMAX}, %r9')
    e('\tja\t.Lbad')
    e('\tshrq\t$63, %r8')
    e('\tshlq\t$63, %r8')
    e('\taddq\t%r9, %r8')                        # ns = sign | k
    e('\tcmpq\t$2, %r9')
    e('\tjb\t.Lf1')
    e('\tje\t.Lf2')
    e('\tcmpq\t$5, %r9')
    e('\tja\t.Lgen')
    e('\tleaq\t.Lftab(%rip), %rcx')
    e('\tmovslq\t-12(%rcx,%r9,4), %r9')
    e('\taddq\t%rcx, %r9')
    e('\tjmp\t*%r9')
    e('.Lgen:')
    e(f'\tcmpq\t${KMIN - 1}, %r9')
    e('\tjbe\t.Lbe')
    e('\ttestq\t%r10, %r10')
    e('\tjnz\t.Lgkm')
    e(f'\tcmpq\t${NMAX}, %r9')                    # na = 0, the reciprocal: the cores to NMAX, then qv
    e('\tjbe\t.Lbe')
    e('.Lgkm:')
    e('\tpushq\t%rdi')                            # z
    e('\tpushq\t%rax')                            # e_top
    e('\tpushq\t%r8')                             # ns
    e('\tpushq\t%r9')                             # k
    e('\tsubq\t$8, %rsp')                         # rsp 16-aligned at the call
    e('\ttestq\t%r10, %r10')
    e('\tjz\t.Lgqv')                              # na = 0: the reciprocal (qv)
    # Karp-Markstein: the operands' whole arrays
    e('\tmovq\t%rdx, %rcx')                       # b.d
    e('\tmovq\t%r10, %rdx')                       # na
    e('\tmovq\t%r11, %r8')                        # nb
    e('\tcall\tsbn3f_div_km')                     # (z, a.d, na, b.d, nb, k)
    e('\tjmp\t.Lgret')
    # qv: the windows' top limbs unshifted (the shift by ld goes into its u64 -> u52 conversions), an = 0 the reciprocal
    e('.Lgqv:')
    e(f'\tcmpq\t${QVMAX}, %r9')
    e('\tja\t.Lbad')
    e('\tleaq\t1(%r9), %rcx')
    e('\tmovq\t%r11, %r8')
    e('\tcmpq\t%rcx, %r8')
    e('\tcmovaq\t%rcx, %r8')                      # bn = min(nb, k + 1)
    e('\tsubq\t%r8, %r11')
    e('\tleaq\t(%rdx,%r11,8), %rcx')              # bw
    e('\tleaq\t1(%r9), %rdx')
    e('\tcmpq\t%rdx, %r10')
    e('\tcmovbq\t%r10, %rdx')                     # an = min(na, k + 1)
    e('\tsubq\t%rdx, %r10')
    e('\tleaq\t(%rsi,%r10,8), %rsi')              # aw
    e('\tcall\tsbn3f_div_qv')                     # (z, aw, an, bw, bn, k)
    e('.Lgret:')
    e('\taddq\t$8, %rsp')
    e('\tpopq\t%rcx')                             # k
    e('\tpopq\t%rdx')                             # ns
    e('\tpopq\t%rax')                             # e_top
    e('\tpopq\t%rdi')                             # z
    e('\tcmpq\t$0, -8(%rdi,%rcx,8)')
    e('\tje\t.Lstrip')
    e('\tret')
    e('.Lstrip:')                                 # z[n-1] = 0: n - 1, e - 1, until a nonzero top or zero
    e('\tdecq\t%rax')
    e('\tdecq\t%rdx')
    e('\tdecq\t%rcx')
    e('\tjz\t.Lzero')
    e('\tcmpq\t$0, -8(%rdi,%rcx,8)')
    e('\tje\t.Lstrip')
    e('\tret')
    e('.Lzero:')
    e('\txorl\t%eax, %eax')
    e('\txorl\t%edx, %edx')
    e('\tret')
    e('.Lbad:')                                   # k beyond KMAX: outside the precondition
    e('\tud2')
    e('\t.size\tSYM(inv_base), .-SYM(inv_base)')
    out.append('')

def funnel_vector(nsrc, top_from_b, pre=''):
    # the same 8 limbs a block by VPSHLDVQ (zmm7 = ld in every lane), into a buffer padded to whole blocks (the stores
    # are unmasked, so the cores' 8-byte loads forward from them): blocks inside the window ([z0, k] for w[i], also for
    # w[i+1]) load plainly; a boundary block masks lanes outside it (fault suppression, zero). Lanes past the result
    # store zeros or the top limb's spill into the padding. rdi = dst, r9 = k; clobbers rax, rcx, rdx, rsi, r8, r10,
    # r11, k1, k2, zmm0, zmm1
    tag = pre + ('y' if top_from_b else 'x')
    e(f'\tmovq\t{nsrc}, %rax')
    e('\tsubq\t%r9, %rax')
    e('\tleaq\t-8(%rdx,%rax,8), %rdx')            # w (may point below the operand)
    e('\tnegq\t%rax')
    e('\tincq\t%rax')                             # z0 = k + 1 - n
    e('\txorl\t%r10d, %r10d')                     # i0
    e(f'.Lv{tag}:')
    e('\tcmpq\t%rax, %r10')                       # interior: i0 >= z0 and i0 + 8 <= k (w[i0 .. i0+8] all inside)
    e(f'\tjl\t.Lvb{tag}')
    e('\tleaq\t8(%r10), %r11')
    e('\tcmpq\t%r9, %r11')
    e(f'\tjg\t.Lvb{tag}')
    e('\tvmovdqu64\t(%rdx,%r10,8), %zmm0')
    e('\tvmovdqu64\t8(%rdx,%r10,8), %zmm1')
    e(f'.Lvs{tag}:')
    e('\tvpshldvq\t%zmm7, %zmm0, %zmm1')
    e('\tvmovdqu64\t%zmm1, (%rdi,%r10,8)')
    e('\taddq\t$8, %r10')
    e('\tcmpq\t%r9, %r10')
    e(f'\t{"jl" if top_from_b else "jle"}\t.Lv{tag}')
    e(f'\tjmp\t.Lve{tag}')
    e(f'.Lvb{tag}:')                              # a boundary block: lo lanes valid in [z0, k] - i0, hi = lo >> 1 and lane 7 = [w[i0+8] valid]
    e('\tmovl\t$0xff, %r8d')
    e('\tleaq\t1(%r9), %r11')
    e('\tsubq\t%r10, %r11')
    e('\tbzhiq\t%r11, %r8, %r11')                 # lanes <= k - i0
    e('\tmovq\t%rax, %rsi')
    e('\tsubq\t%r10, %rsi')
    e('\tmovq\t%rsi, %rcx')
    e('\tsarq\t$63, %rcx')
    e('\tandnq\t%rsi, %rcx, %rsi')                # max(0, z0 - i0)
    e('\tbzhiq\t%rsi, %r8, %rsi')
    e('\tandnq\t%r11, %rsi, %r11')                # lo
    e('\tkmovb\t%r11d, %k1')
    e('\tshrl\t$1, %r11d')
    e('\tleaq\t8(%r10), %rsi')                    # w[i0+8] valid: z0 <= i0 + 8 <= k
    e('\tcmpq\t%r9, %rsi')
    e('\tsetle\t%cl')
    e('\tcmpq\t%rax, %rsi')
    e('\tsetge\t%r8b')
    e('\tandb\t%r8b, %cl')
    e('\tshlb\t$7, %cl')
    e('\torb\t%cl, %r11b')
    e('\tkmovb\t%r11d, %k2')
    e('\tvmovdqu64\t(%rdx,%r10,8), %zmm0{%k1}{z}')
    e('\tvmovdqu64\t8(%rdx,%r10,8), %zmm1{%k2}{z}')
    e(f'\tjmp\t.Lvs{tag}')
    e(f'.Lve{tag}:')

def main():
    e(f'/* Generated by tools/generate_fp_div.py; do not edit. fp kernel division and reciprocal, x86-64 (BMI2, ADX,')
    e(' * LZCNT): the research prototype\'s scalar division on the limb grid to k = 21; from k = 22 Karp-Markstein on the')
    e(' * reciprocal\'s IFMA levels (sbn3f_div_km, div_km.cpp); the reciprocal entry above k = 34: IFMA qv (div_qv.cpp).')
    e(' * n = 2 and n = 3..5 port GMP 6.3.0 udiv_qr_3by2 / invert_pi1 / sbpi1_div_qr arithmetic via the library (dual')
    e(' * LGPLv3+ / GPLv2+; Copyright 1991-2016 Free Software Foundation, Inc.; notice')
    e(' * src/core/x86_64/scalar-donor-notice.txt); n >= 6 is the prototype\'s register-window short division. */')
    e('#ifndef FPK_PFX')
    e('#define FPK_PFX sbn3f_')
    e('#endif')
    e('#define FPK_CAT2(a, b) a##b')
    e('#define FPK_CAT(a, b) FPK_CAT2(a, b)')
    e('#define SYM(x) FPK_CAT(FPK_PFX, x)')
    entries()
    cores()
    e('\t.section\t.rodata')
    e('\t.p2align\t2')
    e('.Lftab:')                                  # k = 3..5: the fused small paths (k = 1, 2 by branches)
    for k in range(3, 6):
        e(f'\t.long\t.Lf{k}-.Lftab')
    e('.Litab:')                                  # the reciprocal's
    for k in range(3, 6):
        e(f'\t.long\t.Li{k}-.Litab')
    b_tables()
    e('\t.section\t.note.GNU-stack,"",@progbits')
    args = [x for x in sys.argv[1:] if x != '--check']
    path, text = args[0] if args else OUT, '\n'.join(out) + '\n'
    if '--check' in sys.argv[1:]:                 # compare instead of writing (tools/build.py)
        if open(path).read() != text:
            raise SystemExit(f'{path} differs from tools/generate_fp_div.py\'s output')
        print(f'OK: {path} is tools/generate_fp_div.py\'s output')
        return
    open(path, 'w').write(text)

if __name__ == '__main__':
    main()
