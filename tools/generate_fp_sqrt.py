#!/usr/bin/env python3
"""Generate src/fp/x86_64/sqrt.S: the fp kernel square root and reciprocal square root entries (contract:
src/fp/kernel.hpp; the grid and the cores' proofs: src/fp/x86_64/sqrt.cpp).

  fxr sbn3f_sqrt(limb *z, const fx *x, int64_t leps)     rdi z, rsi x, rdx leps
  fxr sbn3f_rsqrt(limb *z, const fx *x, int64_t leps)    rdi z, rsi x, rdx leps
  result: rax e, rdx ns

The entry decodes x, computes e_top (sqrt: ceil(e/2), rsqrt: floor((2-e)/2)), k = e_top - L2, c = lzcnt(X's top limb)
& ~1 and s = c/2 + 32 [e odd], and runs here:
  sqrt k = 1, s >= 14: z = floor(sqrt(T1) 2^(32-s)) in double, T1 = Tf's top limb (Tf = the top limbs of X << c,
    t = Tf / B^2 in [1/4, 1)): the value sqrt(t) B / 2^s < 2^(64-s) <= 2^50 carries a relative error < 2^-53 (the
    conversion and the square root's rounding) + 2^-62 (Tf's lower limb): < 2^-3 absolute; the floor < 1. Clamped up to
    2^(63-s) (the target's lower bound: the limb stays nonzero).
  sqrt k = 1, s < 14: the prototype's sqrt1b on T = Tf's top two limbs (z0 from sqrtsd, |z0 - sqrt T| <= 2^13; the
    exact residual R = T - z0^2; d = (R >> 32) rd 2^31 with rd = vrsqrt14(T): |z1 - sqrt T| < 1.501), z = z1 >> s:
    error < 1.501 + 1 + Tf's truncation (< 2^-62 of T: 1/4).
  sqrt k = 2: the prototype's sqrt2 on Tf's top three limbs (isqrt128 of the top two, exact: q, R = A - q^2 <= 2q; one
    divq of (R B + Tf[0]) / 2q as (R >> 1, (R & 1) 2^63 + Tf[0] >> 1) / q; R = 2q: (q + 1) B), Z' within 2 of sqrt(t)
    B^2, z = Z' >> s.
  rsqrt k = 1, s <= 46: z = floor(y 2^(32+s)), y ~ 1 / sqrt(T1) from vrsqrt14 and two Newton steps (relative error
    < 2^-50 with the double's rounding): the value 2^s / sqrt(t) < 2^(s+1) has an absolute error < 2^-3, Tf's lower
    limb < 2^-60. Clamped up to 2^s. s > 46: Zh >> (63 - s) from rsqrt k = 2's core (rsq2_core).
  rsqrt k = 2: rsq2_core and Tf's third limb (rsqrt2 below). sqrt k = 3..KSB: the schoolbook root (core_sb below).
Everything else (sqrt k > KSB, rsqrt k >= 3) calls sbn3f_root_core (sqrt.cpp) with the decoded values: (z, X, n, k, c,
s | rsqrt << 8); it hands sqrt to the IFMA root sbn3f_sqrt_qv (div_qv.cpp).
"""
import sys

OUT = 'src/fp/x86_64/sqrt.S'
KMAX = 230
out = []
def e(s=''):
    out.append(s)

LBL = [0]
def lbl():
    LBL[0] += 1
    return f'.Lg{LBL[0]}'

def load_top(t, reg):
    """reg = X's t-th limb from the top (rsi = X, r8 = n), 0 when t >= n; flags kept? no"""
    if t == 0:
        e(f'\tmovq\t-8(%rsi,%r8,8), %{reg}')
        return
    skip = lbl()
    e(f'\txorl\t%{reg32(reg)}, %{reg32(reg)}')
    e(f'\tcmpq\t${t + 1}, %r8')
    e(f'\tjb\t{skip}')
    e(f'\tmovq\t{-8 * (t + 1)}(%rsi,%r8,8), %{reg}')
    e(f'{skip}:')

def reg32(r):
    return {'rax': 'eax', 'rbx': 'ebx', 'rcx': 'ecx', 'rdx': 'edx', 'rsi': 'esi', 'rdi': 'edi', 'rbp': 'ebp'}.get(r, r + 'd')

def entry(name, rsqrt):
    e('\t.globl\tSYM(%s)' % name)
    e('\t.type\tSYM(%s), @function' % name)
    e('\t.p2align\t5')
    e('SYM(%s):' % name)
    e('\tmovq\t16(%rsi), %r8')
    e('\tbtrq\t$63, %r8')                         # n (the sign is the caller's: x >= 0)
    e('\ttestq\t%r8, %r8')                        # (BTR leaves ZF undefined)
    e('\tjz\t.Lzero')
    e('\tmovq\t8(%rsi), %rax')                    # e
    e('\tmovq\t(%rsi), %rsi')                     # X
    if rsqrt:
        e('\tmovl\t$2, %r9d')
        e('\tsubq\t%rax, %r9')
    else:
        e('\tleaq\t1(%rax), %r9')
    e('\tsarq\t$1, %r9')                          # e_top
    e('\tsubq\t$2, %rdx')
    e('\tsarq\t$6, %rdx')                         # L2
    e('\tmovq\t%r9, %rcx')
    e('\tsubq\t%rdx, %rcx')                       # k
    e('\tjle\t.Lzero')
    e(f'\tcmpq\t${KMAX}, %rcx')
    e('\tja\t.Lbad')
    e('\tlzcntq\t-8(%rsi,%r8,8), %r10')
    e('\tandl\t$62, %r10d')                       # c
    e('\tandl\t$1, %eax')
    e('\tshll\t$5, %eax')
    e('\tmovl\t%r10d, %r11d')
    e('\tshrl\t$1, %r11d')
    e('\taddl\t%eax, %r11d')                      # s
    e('\tcmpq\t$1, %rcx')
    if rsqrt:
        e('\tje\t.Lr1')
        e('\tcmpq\t$2, %rcx')
        e('\tje\t.Lr2')
        e('\tmovl\t$256, %eax')                   # the rsqrt flag for the core
    else:
        e('\tje\t.Lq1')
        e('\tcmpq\t$2, %rcx')
        e('\tje\t.Lq2')
        e(f'\tcmpq\t${KSB}, %rcx')
        e('\tjbe\t.Lsb')
        e('\txorl\t%eax, %eax')
    e('\tjmp\t.Lcore')
    e('\t.size\tSYM(%s), .-SYM(%s)' % (name, name))
    e('')

def tf(regs, discard='r11'):
    """regs[t] = Tf's t-th limb from the top, t < len(regs): the top limbs of X << c (rsi = X, r8 = n, rcx = c;
    rdx = p = 2^c after, `discard` / rax clobbered). One mulx per further X limb: lo = X_t << c, hi = X_t >> (64 - c)."""
    load_top(0, regs[0])
    e('\tmovl\t$1, %edx')
    e('\tshlxq\t%rcx, %rdx, %rdx')                # p
    e(f'\tshlxq\t%rcx, %{regs[0]}, %{regs[0]}')
    for t in range(1, len(regs) + 1):
        load_top(t, 'rax')
        lo = regs[t] if t < len(regs) else discard
        e(f'\tmulxq\t%rax, %{lo}, %rax')
        e(f'\torq\t%rax, %{regs[t - 1]}')

def sat_add():
    """rcx = z0, rax = d (signed, small): rcx = z0 + d clamped into [2^63, 2^64 - 1] (the root lies there)"""
    k = lbl()
    e('\tbtcq\t$63, %rcx')                        # z0 - 2^63, signed
    e('\taddq\t%rax, %rcx')
    e(f'\tjo\t{k}s')
    e(f'{k}:')
    e('\txorl\t%edx, %edx')
    e('\ttestq\t%rcx, %rcx')
    e('\tcmovsq\t%rdx, %rcx')
    e('\tbtcq\t$63, %rcx')
    COLD.append(f'{k}s:')
    COLD.append('\tmovabsq\t$0x7fffffffffffffff, %rcx')
    COLD.append(f'\tjmp\t{k}')

COLD = []

def common_tail():
    # the slow paths: sbn3f_root_core(z, X, n, k, c, s | flag); e_top and k kept across the call
    e('.Lcore:')                                  # rdi z, rsi X, r8 n, r9 e_top, rcx k, r10 c, r11 s, eax flag
    e('\tpushq\t%r9')
    e('\tpushq\t%rcx')
    e('\tsubq\t$8, %rsp')                         # rsp 16-aligned at the call
    e('\torl\t%eax, %r11d')
    e('\tmovq\t%rcx, %rdx')                       # (k into rcx below, n into rdx)
    e('\tmovq\t%r8, %rcx')
    e('\txchgq\t%rdx, %rcx')                      # rdx = n, rcx = k
    e('\tmovl\t%r10d, %r8d')                      # c
    e('\tmovl\t%r11d, %r9d')                      # s | flag
    e('\tcall\tsbn3f_root_core')
    e('\taddq\t$8, %rsp')
    e('\tpopq\t%rdx')                             # ns = k
    e('\tpopq\t%rax')                             # e_top
    e('\tret')
    e('.Lzero:')
    e('\txorl\t%eax, %eax')
    e('\txorl\t%edx, %edx')
    e('\tret')
    e('.Lbad:')
    e('\tud2')

def sqrt1():
    e('.Lq1:')                                    # k = 1: rdi z, rsi X, r8 n, r9 e_top, r10 c, r11 s
    e('\tmovq\t%r9, -8(%rsp)')
    e('\tmovq\t%r11, -16(%rsp)')
    e('\tmovl\t%r10d, %ecx')
    e('\tcmpl\t$14, %r11d')
    e('\tjb\t.Lq1c')
    # s >= 14: z = floor(sqrt(Thi) 2^(32-s)), at least 2^(63-s)
    tf(['r9'])
    e('\tmovq\t-16(%rsp), %r11')
    e('\tvxorpd\t%xmm0, %xmm0, %xmm0')            # (the conversion merges: no dependency on the last call)
    e('\tvcvtusi2sdq\t%r9, %xmm0, %xmm0')
    e('\tvsqrtsd\t%xmm0, %xmm0, %xmm0')
    e('\tmovl\t$1055, %eax')                      # 1023 + 32
    e('\tsubl\t%r11d, %eax')
    e('\tshlq\t$52, %rax')
    e('\tvmovq\t%rax, %xmm1')
    e('\tvmulsd\t%xmm1, %xmm0, %xmm0')
    e('\tvcvttsd2usi\t%xmm0, %rax')
    e('\tmovl\t$63, %ecx')
    e('\tsubl\t%r11d, %ecx')
    e('\tmovl\t$1, %edx')
    e('\tshlxq\t%rcx, %rdx, %rdx')                # 2^(63-s)
    e('\tcmpq\t%rdx, %rax')
    e('\tcmovbq\t%rdx, %rax')
    e('\tmovq\t%rax, (%rdi)')
    e('\tmovq\t-8(%rsp), %rax')
    e('\tmovl\t$1, %edx')
    e('\tret')
    # s < 14: sqrt1b on T = Thi B + Tmid
    e('.Lq1c:')
    tf(['r9', 'r10'])
    e('\tvxorpd\t%xmm0, %xmm0, %xmm0')            # (the conversion merges: no dependency on the last call)
    e('\tvcvtusi2sdq\t%r9, %xmm0, %xmm0')
    e('\tvmulsd\t.Lp64(%rip), %xmm0, %xmm0')     # dt
    e('\tvsqrtsd\t%xmm0, %xmm0, %xmm1')
    e('\tvrsqrt14sd\t%xmm0, %xmm0, %xmm2')        # rd
    e('\tvcvttsd2usi\t%xmm1, %rcx')               # z0 (2^64 - 1 when out of range)
    e('\tmovq\t%rcx, %rdx')
    e('\tmulxq\t%rcx, %rax, %rdx')                # z0^2
    e('\tsubq\t%rax, %r10')
    e('\tsbbq\t%rdx, %r9')                        # R = T - z0^2 (signed, |R| < 2^79)
    e('\tshrdq\t$32, %r9, %r10')                  # R >> 32
    e('\tvcvtsi2sdq\t%r10, %xmm0, %xmm0')
    e('\tvmulsd\t.Lp31(%rip), %xmm2, %xmm2')
    e('\tvmulsd\t%xmm2, %xmm0, %xmm0')
    e('\tvcvttsd2si\t%xmm0, %rax')                # d
    sat_add()                                     # z1
    e('\tmovq\t-16(%rsp), %r11')
    e('\tshrxq\t%r11, %rcx, %rax')
    e('\tmovq\t%rax, (%rdi)')
    e('\tmovq\t-8(%rsp), %rax')
    e('\tmovl\t$1, %edx')
    e('\tret')

def rsqrt1():
    e('.Lr1:')
    e('\tcmpl\t$46, %r11d')
    e('\tja\t.Lr1s')
    e('\tmovq\t%r9, -8(%rsp)')
    e('\tmovq\t%r11, -16(%rsp)')
    e('\tmovl\t%r10d, %ecx')
    tf(['r9'])
    e('\tmovq\t-16(%rsp), %r11')
    e('\tvxorpd\t%xmm0, %xmm0, %xmm0')            # (the conversion merges: no dependency on the last call)
    e('\tvcvtusi2sdq\t%r9, %xmm0, %xmm0')         # a
    e('\tvrsqrt14sd\t%xmm0, %xmm0, %xmm1')        # y (relative 2^-14)
    e('\tvmulsd\t.Lhalf(%rip), %xmm0, %xmm0')    # a/2
    for _ in range(2):                            # y += y (1/2 - (a/2) y^2)
        e('\tvmulsd\t%xmm1, %xmm0, %xmm2')
        e('\tvmovsd\t.Lhalf(%rip), %xmm3')
        e('\tvfnmadd231sd\t%xmm2, %xmm1, %xmm3')
        e('\tvfmadd231sd\t%xmm3, %xmm1, %xmm1')
    e('\tmovl\t$1055, %eax')                      # 2^(32+s)
    e('\taddl\t%r11d, %eax')
    e('\tshlq\t$52, %rax')
    e('\tvmovq\t%rax, %xmm2')
    e('\tvmulsd\t%xmm2, %xmm1, %xmm1')
    e('\tvcvttsd2usi\t%xmm1, %rax')
    e('\tmovl\t$1, %edx')
    e('\tshlxq\t%r11, %rdx, %rdx')                # 2^s
    e('\tcmpq\t%rdx, %rax')
    e('\tcmovbq\t%rdx, %rax')
    e('\tmovq\t%rax, (%rdi)')
    e('\tmovq\t-8(%rsp), %rax')
    e('\tmovl\t$1, %edx')
    e('\tret')
    e('.Lr1s:')                                   # s > 46: Z = rsq2_core(Tf's top two limbs) ~ 2^191 / sqrt(t), z =
    e('\tmovq\t%r9, -8(%rsp)')                   # floor(Z / 2^(127 - s)) = Zh >> (63 - s) in [2^s, B - 1] (Z in
    e('\tmovq\t%r11, -16(%rsp)')                 # [2^127, 2^128)); error < 1 + Z's 2 units / 2^64 + Tf's tail
    e('\tmovq\t%rbp, -32(%rsp)')
    e('\tmovq\t%r12, -40(%rsp)')
    e('\tmovl\t%r10d, %ecx')
    tf(['r9', 'r10'], discard='rcx')
    rsq2_core('.Lr1c', False)
    e('\tmovl\t$63, %ecx')
    e('\tsubl\t-16(%rsp), %ecx')
    e('\tshrxq\t%rcx, %r11, %rax')
    e('\tmovq\t%rax, (%rdi)')
    e('\tmovq\t-32(%rsp), %rbp')
    e('\tmovq\t-40(%rsp), %r12')
    e('\tmovq\t-8(%rsp), %rax')
    e('\tmovl\t$1, %edx')
    e('\tret')

# rsqrt k = 2: R' = B^2 / sqrt(t) in (B^2, 2B^2], z = floor(R' 2^s / B). The prototype's rsq2_core on A = Tf's top two
# limbs with the seed of rsqrt1 (vrsqrt14 and two Newton steps: |e0| < 2^-49, the bound its analysis assumes): z0 ~
# 2^127 / sqrt(A), R'' = floor((2^254 - A z0^2) / 2^120) exact, c1 = floor(z0 R'' / 2^63), c2 = 3/8 e0^2 z0 2^72 in
# double, Z = z0 2^64 + (c1 + c2) >> 8 within 1 + 2^-5 of 2^191 / sqrt(A), clamped into [2^127, 2^128 - 1]; R'(t_A) = 2Z.
# Tf's third limb T0 lowers the target by T0 / (2 B t^1.5) < 4 units (first order; the second order < 2^-60): corr =
# T0 r^3 / 2^65, r = 2^32 y ~ 1 / sqrt(t) from the seed. |R' - B^2 / sqrt(t)| < 2.1; z: error < 1 + 1.05. Clamped into [2^s B,
# B^2 - 1]. Frame: the red zone (e_top, s, rbx, rbp, r12).
def rsq2_core(tag, corr3, z0='rbp', c2='r12'):
    """Z ~ 2^191 / sqrt(A) (rsqrt k = 2's core, the prototype's rsq2_core), A = r9 B + r10 in [2^126, 2^128): Zh in r11,
    Zl in r8, saturated into [2^127, 2^128 - 1]; corr3 (k = 2): rbx = Tf's third limb in, its corr3 out. Clobbers rax,
    rcx, rdx, rsi, z0, c2, xmm0 .. xmm7."""
    # seed: y ~ 1 / sqrt(Thi) (relative 2^-50), z0 = y 2^95 (2^64 - 1 when it reaches 2^64)
    e('\tvxorpd\t%xmm0, %xmm0, %xmm0')
    e('\tvcvtusi2sdq\t%r9, %xmm0, %xmm0')
    e('\tvrsqrt14sd\t%xmm0, %xmm0, %xmm1')
    e('\tvmulsd\t.Lhalf(%rip), %xmm0, %xmm0')
    for _ in range(2):
        e('\tvmulsd\t%xmm1, %xmm0, %xmm2')
        e('\tvmovsd\t.Lhalf(%rip), %xmm3')
        e('\tvfnmadd231sd\t%xmm2, %xmm1, %xmm3')
        e('\tvfmadd231sd\t%xmm3, %xmm1, %xmm1')
    if corr3:
        # corr3 = T0 r^3 / 2^65 with r = 2^32 y ~ 1 / sqrt(t) (relative 2^-50: < 2^-45 units), off the residual's chain
        e('\tvmulsd\t%xmm1, %xmm1, %xmm6')
        e('\tvmulsd\t%xmm1, %xmm6, %xmm6')            # y^3
        e('\tvxorpd\t%xmm7, %xmm7, %xmm7')
        e('\tvcvtusi2sdq\t%rbx, %xmm7, %xmm7')
        e('\tvmulsd\t.Lp31(%rip), %xmm7, %xmm7')     # T0 2^31 (= T0 2^96 / 2^65)
        e('\tvmulsd\t%xmm7, %xmm6, %xmm6')
        e('\tvcvttsd2usi\t%xmm6, %rbx')               # corr3 < 4
    e('\tvmulsd\t.Lp95(%rip), %xmm1, %xmm1')
    e(f'\tvcvttsd2usi\t%xmm1, %{z0}')               # z0
    # P = A z0^2 (4 limbs p3 p2 p1 p0): sq = s1 s0
    e(f'\tmovq\t%{z0}, %rdx')
    e(f'\tmulxq\t%{z0}, %r8, %r11')                 # s0 = r8, s1 = r11
    e('\tmovq\t%r10, %rdx')                       # x0
    e('\tmulxq\t%r8, %rax, %rcx')                 # x0 s0: p0 = rax, rcx
    e('\tmulxq\t%r11, %rdx, %rsi')                # x0 s1: rdx, rsi
    e('\taddq\t%rdx, %rcx')
    e('\tadcq\t$0, %rsi')
    e('\tmovq\t%r9, %rdx')                        # x1
    e(f'\tmulxq\t%r8, %r8, %{c2}')                  # x1 s0: r8, r12
    e('\tmulxq\t%r11, %r11, %rdx')                # x1 s1: r11, rdx
    e('\taddq\t%r8, %rcx')                        # p1
    e(f'\tadcq\t%{c2}, %rsi')
    e('\tadcq\t$0, %rdx')
    e('\taddq\t%r11, %rsi')                       # p2
    e('\tadcq\t$0, %rdx')                         # p3
    # R' = 2^254 - P: limbs r1 r2 r3 of the two's complement (r0 only feeds the borrow)
    e('\tnegq\t%rax')                             # CF = p0 != 0
    e('\tmovl\t$0, %eax')
    e('\tsbbq\t%rcx, %rax')                       # r1
    e('\tmovl\t$0, %ecx')
    e('\tsbbq\t%rsi, %rcx')                       # r2
    e('\tmovabsq\t$0x4000000000000000, %rsi')
    e('\tsbbq\t%rdx, %rsi')                       # r3
    # R'' = R' >> 120: L = (r1 >> 56) | (r2 << 8) (rax), H = (r2 >> 56) | (r3 << 8) (rcx, signed)
    e('\tshrdq\t$56, %rcx, %rax')
    e('\tshrdq\t$56, %rsi, %rcx')
    # e0 = double(R'' >> 32) 2^-102, R'' >> 32 = (H << 32) | (L >> 32) (|R''| < 2^87: fits int64)
    e('\tmovq\t%rcx, %r8')
    e('\tmovq\t%rax, %r8')
    e('\tshrdq\t$32, %rcx, %r8')                  # R'' >> 32
    e('\tvxorpd\t%xmm4, %xmm4, %xmm4')
    e('\tvcvtsi2sdq\t%r8, %xmm4, %xmm4')
    e('\tvmulsd\t.Lm102(%rip), %xmm4, %xmm4')     # e0
    e('\tvmulsd\t%xmm4, %xmm4, %xmm4')            # e0^2
    e('\tvxorpd\t%xmm5, %xmm5, %xmm5')
    e(f'\tvcvtusi2sdq\t%{z0}, %xmm5, %xmm5')
    e('\tvmulsd\t.Lc38(%rip), %xmm5, %xmm5')      # 3/8 2^72 z0
    e('\tvmulsd\t%xmm5, %xmm4, %xmm4')
    e(f'\tvcvttsd2si\t%xmm4, %{c2}')                # c2 (int64, >= 0)
    # c1 = 2 z0 H + ((z0 L) >> 63), signed 128-bit
    e(f'\tmovq\t%{z0}, %rdx')
    e('\tmulxq\t%rax, %r8, %r11')                 # z0 L: r11 r8
    e('\tshrdq\t$63, %r11, %r8')                  # low limb of (z0 L) >> 63
    e('\tshrq\t$63, %r11')                        # its high limb (0 or 1)
    e('\tmovq\t%rcx, %rax')
    e(f'\timulq\t%{z0}')                            # rdx:rax = H z0 signed x signed (z0 read as z0 - 2^64 when >= 2^63)
    e('\tmovq\t%rcx, %rsi')
    e(f'\tmovq\t%{z0}, %rsi')
    e('\tsarq\t$63, %rsi')
    e('\tandq\t%rcx, %rsi')
    e('\taddq\t%rsi, %rdx')                       # + H 2^64 when z0 >= 2^63: H z0 with z0 unsigned
    e('\taddq\t%rax, %rax')                       # 2 H z0
    e('\tadcq\t%rdx, %rdx')
    e('\taddq\t%r8, %rax')
    e('\tadcq\t%r11, %rdx')                       # c1
    e(f'\taddq\t%{c2}, %rax')                       # + c2
    e('\tadcq\t$0, %rdx')
    e('\tshrdq\t$8, %rdx, %rax')                  # corr = (c1 + c2) >> 8 (arithmetic): rdx:rax
    e('\tsarq\t$8, %rdx')
    # Z = z0 2^64 + corr: Zh = z0 + corr_hi + carry(corr_lo), Zl = corr_lo; saturate into [2^127, 2^128 - 1]
    e('\tmovq\t%rax, %r8')                        # Zl
    e(f'\tmovq\t%{z0}, %r11')
    e('\tbtcq\t$63, %r11')                        # z0 - 2^63, signed
    e('\taddq\t%rdx, %r11')                       # + corr_hi (|corr| < 2^80: no signed overflow but at the top)
    e(f'\tjo\t{tag}s')
    e(f'{tag}b:')
    e('\ttestq\t%r11, %r11')
    e(f'\tjs\t{tag}u')                              # Z < 2^127
    e('\tbtcq\t$63, %r11')                        # Zh

    COLD.extend([f'{tag}s:', '\tmovq\t$-1, %r8', '\tmovabsq\t$0x7fffffffffffffff, %r11', f'\tjmp\t{tag}b',
                 f'{tag}u:', '\txorl\t%r8d, %r8d', '\txorl\t%r11d, %r11d', f'\tjmp\t{tag}b'])

def rsqrt2():
    e('.Lr2:')
    e('\tmovq\t%r9, -8(%rsp)')
    e('\tmovq\t%r11, -16(%rsp)')
    e('\tmovq\t%rbx, -24(%rsp)')
    e('\tmovq\t%rbp, -32(%rsp)')
    e('\tmovq\t%r12, -40(%rsp)')
    e('\tmovl\t%r10d, %ecx')
    tf(['r9', 'r10', 'rbx'], discard='rcx')       # A = r9 r10, T0 = rbx
    rsq2_core('.Lr2', True)
    e('\tmovq\t%rbx, %rax')                       # corr3
    # R' (3 limbs) = 2Z - corr3: R0 R1 R2
    e('\tmovq\t%r8, %rcx')
    e('\taddq\t%rcx, %rcx')                       # 2 Zl
    e('\tmovq\t%r11, %rdx')
    e('\tadcq\t%rdx, %rdx')                       # 2 Zh + carry
    e('\tmovl\t$0, %esi')
    e('\tadcq\t$0, %rsi')                         # R2 (1)
    e('\tsubq\t%rax, %rcx')
    e('\tsbbq\t$0, %rdx')
    e('\tsbbq\t$0, %rsi')
    # z = floor(R' 2^s / B) = (R2 R1 R0) >> (64 - s): z1 = shld(R2, R1, s), z0 = shld(R1, R0, s)
    e('.Lr2o:')
    e('\tmovq\t-16(%rsp), %r9')                   # s
    e('\tmovl\t%r9d, %eax')
    e('\tmovq\t%rcx, %r10')
    e('\tmovl\t%eax, %ecx')
    e('\tmovq\t%rsi, %r11')
    e('\tshldq\t%cl, %rdx, %r11')                 # z1 (s = 0: R2, then the clamp below)
    e('\tshldq\t%cl, %r10, %rdx')                 # z0
    # (s = 0: z = R2 B + R1 = (r11 = R2, rdx = R1): shld by 0 leaves them; z >= B^2 needs R2 >= B: never)
    # s > 0: z >= B^2 <=> R2 >> (64 - s) != 0
    e('\tmovl\t$64, %eax')
    e('\tsubl\t%ecx, %eax')
    e('\tshrxq\t%rax, %rsi, %rax')                # R2 >> (64 - s) (count mod 64: s = 0 gives R2, masked below)
    e('\ttestl\t%ecx, %ecx')
    e('\tcmovzq\t%rcx, %rax')                     # s = 0: no overflow (rcx = 0)
    e('\ttestq\t%rax, %rax')
    e('\tjnz\t.Lr2f')
    e('\tmovl\t$1, %eax')
    e('\tshlxq\t%rcx, %rax, %rax')                # 2^s
    e('\tcmpq\t%rax, %r11')
    e('\tjb\t.Lr2l')
    e('\tmovq\t%rdx, (%rdi)')
    e('\tmovq\t%r11, 8(%rdi)')
    e('.Lr2x:')
    e('\tmovq\t-24(%rsp), %rbx')
    e('\tmovq\t-32(%rsp), %rbp')
    e('\tmovq\t-40(%rsp), %r12')
    e('\tmovq\t-8(%rsp), %rax')
    e('\tmovl\t$2, %edx')
    e('\tret')
    e('.Lr2f:')                                   # >= B^2: B^2 - 1
    e('\tmovq\t$-1, (%rdi)')
    e('\tmovq\t$-1, 8(%rdi)')
    e('\tjmp\t.Lr2x')
    e('.Lr2l:')                                   # < 2^s B: 2^s B
    e('\tmovq\t$0, (%rdi)')
    e('\tmovq\t%rax, 8(%rdi)')
    e('\tjmp\t.Lr2x')
def isqrt_exact(tag):
    """rcx = q = floor(sqrt(T)), r9 r10 = T - q^2 (<= 2q) for T = r9 B + r10 in [2^126, 2^128); clobbers rax, rdx, r8,
    r11, xmm0 .. xmm4. z0 = sqrtsd(r9 2^64) - 4096 (the double's error < 2^11.6 with r10 dropped: sqrt T - z0 in
    (1000, 2^12.9)), R = T - z0^2 in (0, 2^78), z1 = z0 + hi((R >> 32) yi) with yi = floor(2^95 (1 - 2^-20) y), y = 1 /
    sqrt T from vrsqrt14 and one Newton step (relative error < 2^-26, so yi < 2^95 / sqrt T): R / (2 sqrt T) undershoots
    sqrt T - z0 by < 2^-38, yi by < 2^-7 and the floors by < 1 + 2^-32, so z1 in (sqrt T - 1.01, sqrt T]: q = z1 or
    z1 + 1 (one exact step)"""
    e('\tvxorpd\t%xmm0, %xmm0, %xmm0')
    e('\tvcvtusi2sdq\t%r9, %xmm0, %xmm0')
    e('\tvmulsd\t.Lp64(%rip), %xmm0, %xmm0')     # dt
    e('\tvsqrtsd\t%xmm0, %xmm0, %xmm1')
    e('\tvrsqrt14sd\t%xmm0, %xmm0, %xmm2')        # y
    e('\tvmulsd\t%xmm2, %xmm0, %xmm3')            # y = y (1.5 - 0.5 dt y^2)
    e('\tvmulsd\t%xmm2, %xmm3, %xmm3')
    e('\tvmovsd\t.L1p5(%rip), %xmm4')
    e('\tvfnmadd231sd\t.Lhalf(%rip), %xmm3, %xmm4')
    e('\tvmulsd\t%xmm4, %xmm2, %xmm2')
    e('\tvmulsd\t.Ly95(%rip), %xmm2, %xmm2')
    e('\tvcvttsd2usi\t%xmm2, %r11')              # yi
    e('\tvcvttsd2usi\t%xmm1, %rcx')
    e('\tsubq\t$4096, %rcx')                      # z0
    e('\tmovq\t%rcx, %rdx')
    e('\tmulxq\t%rcx, %rax, %rdx')
    e('\tmovq\t%r10, %r8')
    e('\tsubq\t%rax, %r8')
    e('\tmovq\t%r9, %rax')
    e('\tsbbq\t%rdx, %rax')                       # R = T - z0^2
    e('\tshrdq\t$32, %rax, %r8')
    e('\tmovq\t%r8, %rdx')
    e('\tmulxq\t%r11, %rax, %rax')                # d
    e('\taddq\t%rax, %rcx')                       # z1
    e('\tmovq\t%rcx, %rdx')
    e('\tmulxq\t%rcx, %rax, %rdx')
    e('\tsubq\t%rax, %r10')
    e('\tsbbq\t%rdx, %r9')                        # rem = T - z1^2 >= 0, < 2^66
    e('\tleaq\t1(%rcx,%rcx), %rax')               # rem >= 2 z1 + 1: z1 + 1, rem -= 2 z1 + 1
    e('\tmovq\t%rcx, %rdx')
    e('\tshrq\t$63, %rdx')
    e('\tmovq\t%r10, %r8')
    e('\tmovq\t%r9, %r11')
    e('\tsubq\t%rax, %r8')
    e('\tsbbq\t%rdx, %r11')
    e(f'\tjb\t{tag}')
    e('\tmovq\t%r8, %r10')
    e('\tmovq\t%r11, %r9')
    e('\tincq\t%rcx')
    e(f'{tag}:')

# ---- sqrt, 3 <= k <= KSB: the short schoolbook root (the digit recurrence; the user's quadratic sqrt, 2026-10-12) with
# overlapped steps (v2). N = floor(t B^(2k)) over positions [cg, 2k), cg = k - 2 - G (t's top k + 2 + G limbs: the top
# limbs of X << c), kept complemented (~R: every update an addition, two carry chains). S_j = the first j root digits,
# D_j = 2 S_j at Db[k-j .. k] (Db[k] = 1: s1 >= B / 2), R = N - S_j^2 B^(2k-2j); z = D_k >> (s + 1) at the end.
#   Digits 1, 2 exact: s1 = isqrt128(N's top two limbs) (rem <= 2 s1); s2 = floor((rem B + N[2k-3]) / 2 s1)
#   (B - 1 when rem = 2 s1), R = rem B^2 + N[2k-3] B + N[2k-4] - s2 (2 s1 B + s2) exactly (at most one add-back). V
#   (the estimate's two-limb reciprocal of E = <s1, s2> + 1: inv_window) from invert_limb(s1) (the one divq; s2 by
#   udiv_qr_2by1 with it) and the 3/2 adjustment for s2 + 1.
#   Step j = 2 .. k-1 (digit j + 1, pt = 2k - j; V = [q, D_j]: V[i] at position 2k - 2j - 2 + i, V[j+1] = Db[k] = 1):
#   state on entry: the window W3 W2 W1 W0 = ~R at pt+1 .. pt-2 (registers; W0 without the pending carry bits delta of
#   step j-1's lower chain), F = ~R at pt-3 (register), ~R below pt-3 in RW. Checks R[pt+1] = 0, R[pt] <= 1; the
#   division prototype's low-biased product-only estimate q from R / 2 at pt-1, pt-2 (W2 W1 W0, without delta: at most
#   2 B^(pt-2) over, so q may exceed the digit; the next step's checks and the fix catch it); W0 += delta; the top part
#   (positions pt-3 .. pt in registers: products with Db[k-1], Db[k-2], hi(q Db[k-3]); the window moves down one); the
#   lower chain (positions s .. pt-4, multiplier q, memory; position pt-4's sum into F, its carry bits the next
#   delta), in groups of G from s = pt - 4 - G ceil(min(j - 2, k - j - 1) / G): V's limbs below V[0] read zero
#   (Db's future digits, G zero limbs below Db[0]) in the first half; below cut = k - 2 (positions >= cg) in the second,
#   where the omitted products (the high half of the one below s included) stay < B^cut per step. The next step's
#   estimate depends only on the top part: its lower chain runs beside it.
#   Cold (sbn3f_sqrt_sb_fix, sqrt.cpp, with the window spilled and delta applied): R < 0 (an estimate over the digit) ->
#   S_j - 1; R >= D_j + 1 at 2k - 2j (an earlier digit low: R[pt] >= 2 or the estimate >= B) -> S_j + 1; until
#   0 <= R <= D_j + ..; an estimate >= B with R in range: the digit is B - 1. delta's carry out of W0: carried through
#   the window, estimate again. After step k-1: delta, the checks once more (fix at j = k).
#   R stays within k B^cut of the exact N - S^2 (the omitted products), so S ends within 1 + 2^-60 of sqrt(N) (2 where
#   a low digit decided by R's limb at cut stays; gate: <= 1.03 units).
#   Work: step j's lower chain covers min(j, k - j) + O(G) limbs: ~k^2 / 4 products.
KSB = 54                                          # the IFMA root (div_qv.cpp's sqrt_qv) wins from ~55 (2026-10-12:
                                                  # schoolbook / IFMA 2.47K / 2.25K cycles at 56, 2.22K / 2.26K at 52)
SBG = 2                                           # the lower chain's group size (even; 2 beat 4 by 2-4% at k 8 .. 64)
EST_BIAS = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # test builds: the estimate's mutation
F_K, F_Z, F_S, F_ET, F_V1, F_V0, F_FS, F_N, F_C = 0, 8, 16, 24, 32, 40, 48, 56, 64
F_RSI, F_RDI, F_KG, F_RW = 72, 80, 88, 96
# frame: RW (k + 2 + G limbs: positions cg .. 2k-1) and the Db area (G zero limbs below Db[0], Db[0 .. k+1], rounded up
# for the two-limb zeroing): F_RW + 8 (k + 2 + G) + 8 (k + 5 + G) = 16k + 152 + 16G (= 8 mod 16: rsp 16-aligned at the
# cold calls)

def pi1_adj(d1, d0, v, t0, t1, tag):
    """v = invert_limb(d1) -> invert_pi1(d1, d0) (GMP's 3/2 adjustment; uses rax, rdx, t0, t1)"""
    e(f'\tmov\t%{d1}, %{t0}')
    e(f'\timul\t%{v}, %{t0}')
    e(f'\tadd\t%{d0}, %{t0}')
    e(f'\tjnc\t.Lpa1_{tag}')
    e(f'\tsub\t$1, %{v}')
    e(f'\tcmp\t%{d1}, %{t0}')
    e(f'\tsbb\t%{t1}, %{t1}')
    e(f'\tnot\t%{t1}')
    e(f'\tsub\t%{d1}, %{t0}')
    e(f'\tadd\t%{t1}, %{v}')
    e(f'\tand\t%{d1}, %{t1}')
    e(f'\tsub\t%{t1}, %{t0}')
    e(f'.Lpa1_{tag}:')
    e(f'\tmov\t%{d0}, %rdx')
    e(f'\tmulx\t%{v}, %{t1}, %rax')
    e(f'\tadd\t%rax, %{t0}')
    e(f'\tjnc\t.Lpa2_{tag}')
    e(f'\tsub\t$1, %{v}')
    e(f'\tcmp\t%{d1}, %{t0}')
    e(f'\tjb\t.Lpa2_{tag}')
    e(f'\tja\t.Lpa3_{tag}')
    e(f'\tcmp\t%{d0}, %{t1}')
    e(f'\tjb\t.Lpa2_{tag}')
    e(f'.Lpa3_{tag}:')
    e(f'\tsub\t$1, %{v}')
    e(f'.Lpa2_{tag}:')

def sb_fix(ovf):
    """cold: spill the window (positions pt-3 .. pt+1 at 8 .. 40(rsi)), sbn3f_sqrt_sb_fix(RW, Db, k, j, ovf, cg),
    reload; delta = 0, rax = the result"""
    G = SBG
    return [f'\tmovq\t%r15, 8(%rsi)', '\tmovq\t%r8, 16(%rsi)', '\tmovq\t%r9, 24(%rsi)', '\tmovq\t%r10, 32(%rsi)',
            '\tmovq\t%r11, 40(%rsi)', f'\tmovq\t%rsi, {F_RSI}(%rsp)', f'\tmovq\t%rdi, {F_RDI}(%rsp)',
            f'\tmovq\t{F_K}(%rsp), %rdx', '\tleaq\t0(,%rdx,8), %rax', '\tmovq\t%rdi, %rsi', '\tsubq\t%rax, %rsi',
            '\taddq\t$24, %rsi', f'\tleaq\t{F_RW}(%rsp), %rdi', '\tmovq\t%r14, %rcx', f'\tmovl\t${ovf}, %r8d',
            f'\tleaq\t{-2 - G}(%rdx), %r9', '\tcall\tsbn3f_sqrt_sb_fix', f'\tmovq\t{F_RSI}(%rsp), %rsi',
            f'\tmovq\t{F_RDI}(%rsp), %rdi', '\tmovq\t8(%rsi), %r15', '\tmovq\t16(%rsi), %r8', '\tmovq\t24(%rsi), %r9',
            '\tmovq\t32(%rsi), %r10', '\tmovq\t40(%rsi), %r11', '\txorl\t%r13d, %r13d']

def core_sb():
    G = SBG
    FR = 152 + 16 * G
    e('.Lsb:')                                    # rdi z, rsi X, r8 n, r9 e_top, rcx k, r10 c, r11 s
    for r in ['rbx', 'rbp', 'r12', 'r13', 'r14', 'r15']:
        e(f'\tpushq\t%{r}')
    e('\tmovq\t%rcx, %r12')
    e('\tshlq\t$4, %rcx')
    e(f'\taddq\t${FR}, %rcx')                     # 16k + FR
    e('\tsubq\t%rcx, %rsp')
    e(f'\tmovq\t%rcx, {F_FS}(%rsp)')
    e(f'\tmovq\t%r12, {F_K}(%rsp)')
    e(f'\tleaq\t{G - 2}(%r12), %rax')
    e(f'\tmovq\t%rax, {F_KG}(%rsp)')              # k + G - 2 (the lower chain's length)
    e(f'\tmovq\t%rdi, {F_Z}(%rsp)')
    e(f'\tmovq\t%r11, {F_S}(%rsp)')
    e(f'\tmovq\t%r9, {F_ET}(%rsp)')
    e(f'\tmovq\t%r8, {F_N}(%rsp)')
    e('\tmovl\t%r10d, %ecx')                      # c
    tf(['r9', 'r10', 'r13', 'r14', 'r15'])         # N[2k-1 .. 2k-5]; rdx = 2^c
    e(f'\tmovq\t%rdx, {F_C}(%rsp)')
    isqrt_exact('.Lsb1q')                          # rcx = s1, r9 r10 = rem (<= 2 s1)
    e('\tmovq\t%rcx, %rbx')                       # s1
    e('\tmovq\t%rbx, %rdx')                       # v = invert_limb(s1): the one division (the divider runs one
    e('\tnotq\t%rdx')                             # divq at a time: s2 from v below)
    e('\tmovq\t$-1, %rax')
    e('\tdivq\t%rbx')
    e('\tmovq\t%rax, %rbp')
    # the fill, dispatched under the division: Db[-G .. k+1] = 0; RW[i] = ~N[cg + i] for positions cg .. 2k-6
    # (k - 3 + G limbs; N's limb at p is the limb jx = n - 2k + p of X << c: lo(X[jx] 2^c) | hi(X[jx-1] 2^c), zero
    # below X)
    e(f'\tmovq\t{F_K}(%rsp), %rcx')
    e(f'\tleaq\t{F_RW + 8 * (2 + G)}(%rsp,%rcx,8), %rdi')   # Db[-G]
    e('\txorl\t%eax, %eax')                       # scalar stores: the steps' 8-byte loads forward from them
    e(f'\tleaq\t{2 + G + 1}(%rcx), %r8')
    e('\tshrq\t$1, %r8')
    e('.Lsbz:')
    e('\tmovq\t%rax, (%rdi)')
    e('\tmovq\t%rax, 8(%rdi)')
    e('\taddq\t$16, %rdi')
    e('\tdecq\t%r8')
    e('\tjnz\t.Lsbz')
    e(f'\tleaq\t{F_RW}(%rsp), %rdi')
    e(f'\tmovq\t{F_N}(%rsp), %r8')
    e('\tsubq\t%rcx, %r8')
    e(f'\tsubq\t${2 + G}, %r8')                    # jx0 = n - k - 2 - G
    e(f'\tleaq\t{G - 3}(%rcx), %r11')             # limbs
    e('.Lsbf0:')                                  # jx < 0: ~0
    e('\ttestq\t%r8, %r8')
    e('\tjns\t.Lsbf1')
    e('\tmovq\t$-1, (%rdi)')
    e('\taddq\t$8, %rdi')
    e('\tincq\t%r8')
    e('\tdecq\t%r11')
    e('\tjnz\t.Lsbf0')
    e('\tjmp\t.Lsbfd')
    e('.Lsbf1:')
    e(f'\tmovq\t{F_C}(%rsp), %rdx')              # 2^c
    e('\txorl\t%ecx, %ecx')                       # hi(X[jx-1] 2^c)
    e('\ttestq\t%r8, %r8')
    e('\tjz\t.Lsbf2')
    e('\tmulxq\t-8(%rsi,%r8,8), %rax, %rcx')
    e('.Lsbf2:')
    e('\taddq\t%r11, %r8')
    e('\tleaq\t(%rsi,%r8,8), %rsi')               # X + 8 (jx + limbs)
    e('\tleaq\t(%rdi,%r11,8), %rdi')
    e('\tmovq\t%r11, %r8')
    e('\tnegq\t%r8')
    e('.Lsbfl:')
    e('\tmulxq\t(%rsi,%r8,8), %rax, %r11')
    e('\torq\t%rcx, %rax')
    e('\tnotq\t%rax')
    e('\tmovq\t%rax, (%rdi,%r8,8)')
    e('\tincq\t%r8')
    e('\tjz\t.Lsbfd')
    e('\tmulxq\t(%rsi,%r8,8), %rax, %rcx')
    e('\torq\t%r11, %rax')
    e('\tnotq\t%rax')
    e('\tmovq\t%rax, (%rdi,%r8,8)')
    e('\tincq\t%r8')
    e('\tjnz\t.Lsbfl')
    e('.Lsbfd:')
    # s2 = floor((rem B + N[2k-3]) / 2 s1) = floor((n1 B + n0) / s1), n1 = rem >> 1 (< s1), n0 = (rem & 1) 2^63 +
    # N[2k-3] >> 1, by GMP's udiv_qr_2by1 with v; B - 1 when rem >= 2 s1 (2 s1 = B + (2 s1 mod B))
    e('\tleaq\t(%rbx,%rbx), %rax')
    e('\tcmpq\t$1, %r9')
    e('\tjb\t.Lsb2d')
    e('\tcmpq\t%rax, %r10')
    e('\tjae\t.Lsb2m')
    e('.Lsb2d:')
    e('\tmovq\t%r13, %rdi')
    e('\tshrq\t$1, %rdi')
    e('\tmovq\t%r10, %rax')
    e('\tshlq\t$63, %rax')
    e('\torq\t%rax, %rdi')                        # n0
    e('\tmovq\t%r10, %rdx')
    e('\tshrdq\t$1, %r9, %rdx')                   # n1
    e('\tleaq\t1(%rdx), %rsi')
    e('\tmulxq\t%rbp, %r8, %r12')
    e('\taddq\t%rdi, %r8')
    e('\tadcq\t%rsi, %r12')                       # (qh, ql) = n1 v + (n1 + 1, n0)
    e('\tmovq\t%r12, %rax')
    e('\timulq\t%rbx, %rax')
    e('\tsubq\t%rax, %rdi')                       # r = n0 - qh s1
    e('\tcmpq\t%rdi, %r8')
    e('\tsbbq\t%rax, %rax')                       # r > ql: qh - 1, r + s1
    e('\taddq\t%rax, %r12')
    e('\tandq\t%rbx, %rax')
    e('\taddq\t%rax, %rdi')
    e('\tcmpq\t%rbx, %rdi')
    e('\tjae\t.Lsb2r')                            # r >= s1 (rare): qh + 1
    e('\tjmp\t.Lsb2u')
    e('.Lsb2m:')
    e('\tmovq\t$-1, %r12')
    e('.Lsb2u:')                                  # s2 = r12
    COLD.extend(['.Lsb2r:', '\tincq\t%r12', '\tjmp\t.Lsb2u'])
    # R = [rem_hi, rem_lo, N3, N4] - s2 (2 s1 B + s2) (r9 r10 r13 r14), P = s2^2 + 2 s1 s2 B
    e('\tmovq\t%r12, %rdx')
    e('\tmulxq\t%r12, %r8, %r11')                 # s2^2
    e('\tmulxq\t%rbx, %rax, %rcx')                # s1 s2
    e('\tmovq\t%rax, %rsi')
    e('\tshlq\t$1, %rsi')
    e('\tmovq\t%rcx, %rdi')
    e('\tshldq\t$1, %rax, %rdi')
    e('\tshrq\t$63, %rcx')
    e('\taddq\t%r11, %rsi')                       # P1
    e('\tadcq\t$0, %rdi')                         # P2
    e('\tadcq\t$0, %rcx')                         # P3
    e('\tsubq\t%r8, %r14')                        # R0
    e('\tsbbq\t%rsi, %r13')                       # R1
    e('\tsbbq\t%rdi, %r10')                       # R2
    e('\tsbbq\t%rcx, %r9')                        # R3
    e('\tjae\t.Lsb2s')
    e('\tdecq\t%r12')                             # s2 - 1, R += 2 (s1 B + s2) + 1
    e('\tleaq\t1(%r12,%r12), %rdx')
    e('\tmovq\t%r12, %rsi')
    e('\tshrq\t$63, %rsi')
    e('\tleaq\t(%rbx,%rbx), %rdi')
    e('\taddq\t%rsi, %rdi')
    e('\taddq\t%rdx, %r14')
    e('\tadcq\t%rdi, %r13')
    e('\tadcq\t$1, %r10')
    e('\tadcq\t$0, %r9')
    e('.Lsb2s:')
    # V = inv_window(s1, s2): E = <s1, s2> + 1 (e1 rdi, e0 rsi), v = invert_pi1(E) from invert_limb(s1); V = 0 at
    # E = B^2
    e('\tmovq\t%r12, %rsi')
    e('\taddq\t$1, %rsi')
    e('\tmovq\t%rbx, %rdi')
    e('\tadcq\t$0, %rdi')
    e('\tjc\t.Lsbv0')
    e('\tcmpq\t%rbx, %rdi')
    e('\tjne\t.Lsbvc')
    e('.Lsbvr:')
    pi1_adj('rdi', 'rsi', 'rbp', 'r8', 'r11', 'sb')
    e('\tmovq\t%rbp, %rdx')
    e('\tmulxq\t%rsi, %rax, %rcx')                # v e0
    e('\tmovq\t%rbp, %r8')
    e('\timulq\t%rdi, %r8')                       # lo(v e1)
    e('\taddq\t%rsi, %rcx')
    e('\taddq\t%r8, %rcx')                        # x1
    e('\tnegq\t%rax')                             # r0 = -x0, CF = x0 != 0
    e('\tmovl\t$0, %r8d')
    e('\tsbbq\t%rcx, %r8')                        # r1
    e('\tmovq\t%r8, %rdx')
    e('\tmulxq\t%rbp, %r11, %rcx')                # r1 v
    e('\tmovq\t%rax, %rdx')
    e('\tmulxq\t%rbp, %rdx, %rdx')                # hi(r0 v)
    e('\taddq\t%r11, %rax')
    e('\tadcq\t$0, %rcx')
    e('\taddq\t%rdx, %rax')
    e('\tadcq\t$0, %rcx')
    e('\taddq\t%r8, %rcx')                        # v0
    e(f'\tmovq\t%rbp, {F_V1}(%rsp)')
    e(f'\tmovq\t%rcx, {F_V0}(%rsp)')
    e('.Lsbvd:')
    COLD.extend(['.Lsbv0:', f'\tmovq\t$0, {F_V1}(%rsp)', f'\tmovq\t$0, {F_V0}(%rsp)', '\tjmp\t.Lsbvd',
                 '.Lsbvc:', '\tmovq\t%rdi, %rdx', '\tnotq\t%rdx', '\tmovq\t$-1, %rax', '\tdivq\t%rdi',
                 '\tmovq\t%rax, %rbp', '\tjmp\t.Lsbvr'])
    # D_2 = 2 (s1 B + s2) at Db[k-2 .. k]; the window for step 2; rdi = &Db[k-3], rsi = &RW[pt-4] (position 2k-6)
    e(f'\tmovq\t{F_K}(%rsp), %rax')
    e('\tleaq\t(%rax,%rax), %rcx')
    e(f'\tleaq\t{F_RW + 8 * (2 + 2 * G) - 24}(%rsp,%rcx,8), %rdi')
    e(f'\tleaq\t{F_RW + 8 * (G - 4)}(%rsp,%rax,8), %rsi')
    e('\tmovq\t$1, 24(%rdi)')
    e('\tleaq\t(%rbx,%rbx), %rax')
    e('\tmovq\t%r12, %rdx')
    e('\tshrq\t$63, %rdx')
    e('\torq\t%rdx, %rax')
    e('\tmovq\t%rax, 16(%rdi)')
    e('\tleaq\t(%r12,%r12), %rax')
    e('\tmovq\t%rax, 8(%rdi)')
    e('\tnotq\t%r9')
    e('\tmovq\t%r9, %r11')                        # W3 = ~R[2k-1]
    e('\tnotq\t%r10')                             # W2
    e('\tmovq\t%r13, %r9')
    e('\tnotq\t%r9')                              # W1
    e('\tmovq\t%r14, %r8')
    e('\tnotq\t%r8')                              # W0
    e('\tnotq\t%r15')                             # F = ~N[2k-5]
    e('\txorl\t%r13d, %r13d')                     # delta
    e('\tmovl\t$2, %r14d')                        # j
    # steps j = 2 .. k-1
    e('\t.p2align\t5')
    e('.Lsbj:')
    e('\tmovq\t%r11, %rax')                       # R[pt+1] = 0, R[pt] <= 1: W3 & W2 >= B - 2 (W3 is B - 1 or, R < 0,
    e('\tandq\t%r10, %rax')                       # 0: R stays within 2 D_j B^(2k-2j) of [0, D_j B^(2k-2j)])
    e('\tcmpq\t$-2, %rax')
    e('\tjb\t.Lsbc')
    e('\tmovq\t%r8, %rcx')
    e('\tshrdq\t$1, %r9, %rcx')
    e('\tnotq\t%rcx')                             # w3 = R / 2 at pt - 2 (~shrd of the complements)
    e('\tmovq\t%r9, %rdx')
    e('\tshrdq\t$1, %r10, %rdx')
    e('\tnotq\t%rdx')                             # w4 = R / 2 at pt - 1
    e(f'\tmulxq\t{F_V1}(%rsp), %rbx, %r12')       # w4 v1
    e(f'\tmulxq\t{F_V0}(%rsp), %rbp, %rbp')       # hi(w4 v0)
    e('\taddq\t%rdx, %r12')                       # w4 + hi(w4 v1)
    e('\tjc\t.Lsbo')
    e('\tmovq\t%rcx, %rdx')
    e(f'\tmulxq\t{F_V1}(%rsp), %rax, %rax')       # hi(w3 v1)
    e('\tcmpq\t$-4, %r12')
    e('\tjae\t.Lsbes')
    e('\taddq\t%rbx, %rdx')                       # column 3: w3 + lo(w4 v1) + hi(w3 v1) + hi(w4 v0)
    e('\tadcq\t$0, %r12')
    e('\taddq\t%rbp, %rax')
    e('\tadcq\t$0, %r12')
    e('\taddq\t%rdx, %rax')
    e('\tadcq\t$0, %r12')
    if EST_BIAS > 0:                              # test builds only: an estimate one over (the fix's S_j - 1)
        e('\taddq\t$1, %r12')
        e('\tjc\t.Lsbo')
    elif EST_BIAS < 0:                            # test builds only: one under where possible (the fix's S_j + 1)
        e('\ttestq\t%r12, %r12')
        e('\tjz\t1f')
        e('\tsubq\t$1, %r12')
        e('1:')
    e('.Lsbq:')                                   # q = r12
    e('\taddq\t%r13, %r8')                        # W0 += delta
    e('\tjc\t.Lsbdc')
    e('\tmovq\t%r12, %rdx')
    e('\tmovq\t%r14, %rax')
    e('\tnegq\t%rax')
    e('\tmovq\t%rdx, 16(%rdi,%rax,8)')            # V[0] = q at Db[k-j-1]
    # top part: positions pt-3 .. pt
    e('\txorl\t%ecx, %ecx')                       # CF = OF = 0
    e('\tmulxq\t(%rdi), %r12, %rax')              # q Db[k-3]: lo for the lower chain's top, hi at pt-3
    e('\tmulxq\t8(%rdi), %rbx, %rbp')             # q Db[k-2]
    e('\tadoxq\t%rax, %rbx')
    e('\tadcxq\t%rbx, %r15')                      # pt-3
    e('\tmulxq\t16(%rdi), %rax, %rbx')            # q Db[k-1]
    e('\tadoxq\t%rbp, %rax')
    e('\tadcxq\t%rax, %r8')                       # pt-2
    e('\tmovq\t%rdx, %rbp')                       # q Db[k] = q
    e('\tadoxq\t%rbx, %rbp')
    e('\tadcxq\t%rbp, %r9')                       # pt-1
    e('\tadoxq\t%rcx, %r10')
    e('\tadcxq\t%rcx, %r10')                      # pt
    e('\tmovq\t%r10, %r11')
    e('\tmovq\t%r9, %r10')
    e('\tmovq\t%r8, %r9')
    e('\tmovq\t%r15, %r8')
    # lower chain: positions pt-4-G m .. pt-4, m = ceil(min(j - 2, k - j - 1) / G)
    e(f'\tleaq\t{G - 3}(%r14), %rax')            # j - 2 + G - 1
    e(f'\tmovq\t{F_KG}(%rsp), %rcx')
    e('\tsubq\t%r14, %rcx')                       # k - j - 1 + G - 1
    e('\tcmpq\t%rcx, %rax')
    e('\tcmovlq\t%rax, %rcx')
    e(f'\tandq\t${-G}, %rcx')
    e('\tnegq\t%rcx')
    e('\txorl\t%ebp, %ebp')                       # hp; CF = OF = 0
    e('\tjrcxz\t.Lsbt')
    e('\t.p2align\t4')
    e('.Lsbg:')
    for j in range(G):
        hi, hp = ('rbx', 'rbp') if j % 2 == 0 else ('rbp', 'rbx')
        e(f'\tmulxq\t{8 * j}(%rdi,%rcx,8), %rax, %{hi}')
        e(f'\tadoxq\t%{hp}, %rax')
        e(f'\tadcxq\t{8 * j}(%rsi,%rcx,8), %rax')
        e(f'\tmovq\t%rax, {8 * j}(%rsi,%rcx,8)')
    e(f'\tleaq\t{G}(%rcx), %rcx')
    e('\tjrcxz\t.Lsbt')
    e('\tjmp\t.Lsbg')
    e('.Lsbt:')                                   # position pt-4 (rcx = 0)
    e('\tadoxq\t%rbp, %r12')
    e('\tadcxq\t(%rsi), %r12')
    e('\tmovq\t%r12, %r15')                       # F
    e('\tmovl\t$0, %r13d')
    e('\tadoxq\t%rcx, %r13')
    e('\tadcxq\t%rcx, %r13')                      # delta
    e('\tmovq\t%r14, %rax')                       # D_(j+1) = D_j B + 2q
    e('\tnegq\t%rax')
    e('\tleaq\t(%rdx,%rdx), %rbx')
    e('\tmovq\t%rbx, 16(%rdi,%rax,8)')
    e('\tshrq\t$63, %rdx')
    e('\taddq\t%rdx, 24(%rdi,%rax,8)')
    e('\tsubq\t$8, %rsi')
    e('\tincq\t%r14')
    e(f'\tcmpq\t{F_K}(%rsp), %r14')
    e('\tjb\t.Lsbj')
    # j = k: delta, the checks
    e('\taddq\t%r13, %r8')
    e('\tadcq\t$0, %r9')
    e('\tadcq\t$0, %r10')
    e('\tadcq\t$0, %r11')
    e('\txorl\t%r13d, %r13d')
    e('\tcmpq\t$-1, %r11')
    e('\tjne\t.Lsbfx')
    e('\tcmpq\t$-2, %r10')
    e('\tjb\t.Lsbfx')
    e('.Lsbx:')                                   # z = D_k >> (s + 1) (rdi = &Db[k-3])
    e(f'\tmovq\t{F_S}(%rsp), %rcx')
    e('\tincl\t%ecx')
    e('\tmovl\t%ecx, %eax')
    e('\tshrl\t$6, %eax')
    e('\tandl\t$63, %ecx')
    e('\tleaq\t24(%rdi,%rax,8), %rsi')            # base + 8k
    e(f'\tmovq\t{F_K}(%rsp), %r8')
    e(f'\tmovq\t{F_Z}(%rsp), %rdx')
    e('\tleaq\t(%rdx,%r8,8), %rdx')
    e('\tnegq\t%r8')
    e('\tmovq\t(%rsi,%r8,8), %rax')
    e('.Lsbs:')
    e('\tmovq\t8(%rsi,%r8,8), %r9')
    e('\tshrdq\t%cl, %r9, %rax')
    e('\tmovq\t%rax, (%rdx,%r8,8)')
    e('\tmovq\t%r9, %rax')
    e('\tincq\t%r8')
    e('\tjnz\t.Lsbs')
    e(f'\tmovq\t{F_ET}(%rsp), %rax')
    e(f'\tmovq\t{F_K}(%rsp), %rdx')
    e(f'\taddq\t{F_FS}(%rsp), %rsp')
    for r in reversed(['rbx', 'rbp', 'r12', 'r13', 'r14', 'r15']):
        e(f'\tpopq\t%{r}')
    e('\tret')
    # cold: the step's checks (delta pending): delta through the window, the fix, again
    COLD.extend(['.Lsbc:', '\taddq\t%r13, %r8', '\tadcq\t$0, %r9', '\tadcq\t$0, %r10', '\tadcq\t$0, %r11'] + sb_fix(0) +
                ['\tjmp\t.Lsbj'])
    # cold: the estimate >= B (delta pending): the fix; unchanged: the digit is B - 1
    COLD.extend(['.Lsbo:', '\taddq\t%r13, %r8', '\tadcq\t$0, %r9', '\tadcq\t$0, %r10', '\tadcq\t$0, %r11'] + sb_fix(1) +
                ['\ttestq\t%rax, %rax', '\tjz\t.Lsbj', '\tmovq\t$-1, %r12', '\tjmp\t.Lsbq'])
    # cold: column 3 near B: its carries with overflow checks
    COLD.extend(['.Lsbes:', '\taddq\t%rbx, %rdx', '\tadcq\t$0, %r12', '\tjc\t.Lsbo', '\taddq\t%rbp, %rax',
                 '\tadcq\t$0, %r12', '\tjc\t.Lsbo', '\taddq\t%rdx, %rax', '\tadcq\t$0, %r12', '\tjc\t.Lsbo',
                 '\tjmp\t.Lsbq'])
    # cold: delta's carry out of W0: through the window, estimate again
    COLD.extend(['.Lsbdc:', '\tadcq\t$0, %r9', '\tadcq\t$0, %r10', '\tadcq\t$0, %r11', '\tmovl\t$0, %r13d',
                 '\tjmp\t.Lsbj'])
    # cold: the final checks
    COLD.extend(['.Lsbfx:'] + sb_fix(0) + ['\tjmp\t.Lsbx'])

def sqrt2():
    e('.Lq2:')                                    # k = 2
    e('\tmovq\t%r9, -8(%rsp)')
    e('\tmovq\t%r11, -16(%rsp)')
    e('\tmovq\t%rbx, -24(%rsp)')
    e('\tmovl\t%r10d, %ecx')
    tf(['r9', 'r10', 'rbx'])                      # Thi, Tmid, Tlo
    isqrt_exact('.Lq2k')                          # q = rcx, rem = r9 r10 (<= 2q)
    e('\tleaq\t(%rcx,%rcx), %rax')                # rem == 2q: (q + 1) B, or B^2 - 1 at q = B - 1
    e('\tmovq\t%rcx, %rdx')
    e('\tshrq\t$63, %rdx')
    e('\tcmpq\t%rax, %r10')
    e('\tjne\t.Lq2d')
    e('\tcmpq\t%rdx, %r9')
    e('\tje\t.Lq2e')
    e('.Lq2d:')                                   # lo = ((rem >> 1) B + (rem & 1) 2^63 + Tlo >> 1) / q
    e('\tmovq\t%rbx, %rax')
    e('\tshrq\t$1, %rax')
    e('\tmovq\t%r10, %rdx')
    e('\tshlq\t$63, %rdx')
    e('\torq\t%rdx, %rax')
    e('\tshrdq\t$1, %r9, %r10')
    e('\tmovq\t%r10, %rdx')
    e('\tdivq\t%rcx')                             # rax = lo
    e('.Lq2o:')                                   # Z' = rcx B + rax; z = Z' >> s
    e('\tmovq\t%rcx, %r9')
    e('\tmovq\t-16(%rsp), %rcx')
    e('\tshrdq\t%cl, %r9, %rax')                  # z0 (s = 0: rax as it is)
    e('\tshrxq\t%rcx, %r9, %r9')                  # z1
    e('\tmovq\t%rax, (%rdi)')
    e('\tmovq\t%r9, 8(%rdi)')
    e('\tmovq\t-24(%rsp), %rbx')
    e('\tmovq\t-8(%rsp), %rax')
    e('\tmovl\t$2, %edx')
    e('\tret')
    e('.Lq2e:')                                   # rem = 2q: Z' = (q + 1) B; q = B - 1: B^2 - 1
    e('\txorl\t%eax, %eax')
    e('\taddq\t$1, %rcx')
    e('\tjnc\t.Lq2o')
    e('\tmovq\t$-1, %rcx')
    e('\tmovq\t$-1, %rax')
    e('\tjmp\t.Lq2o')

def main():
    e('/* Generated by tools/generate_fp_sqrt.py; do not edit. fp kernel sqrt / rsqrt entries, x86-64 (AVX-512F scalar,')
    e(' * BMI2, ADX, LZCNT): the decode, the k <= 2 paths of the research prototype\'s roots, the short schoolbook sqrt;')
    e(' * the rest in sqrt.cpp. invert_pi1 and udiv_qr_2by1 arithmetic of GMP 6.3.0 (dual LGPLv3+ / GPLv2+; Copyright')
    e(' * 1991-2016 Free Software Foundation, Inc.; notice src/core/x86_64/scalar-donor-notice.txt). */')
    e('#ifndef FPK_PFX')
    e('#define FPK_PFX sbn3f_')
    e('#endif')
    e('#define FPK_CAT2(a, b) a##b')
    e('#define FPK_CAT(a, b) FPK_CAT2(a, b)')
    e('#define SYM(x) FPK_CAT(FPK_PFX, x)')
    e('\t.text')
    entry('sqrt', False)
    entry('rsqrt', True)
    sqrt1()
    sqrt2()
    rsqrt1()
    rsqrt2()
    core_sb()
    common_tail()
    out.extend(COLD)
    e('\t.section\t.rodata')
    e('\t.p2align\t3')
    e('.Lp64:\t.quad\t0x43f0000000000000')        # 2^64
    e('.Lp31:\t.quad\t0x41e0000000000000')        # 2^31
    e('.Lhalf:\t.quad\t0x3fe0000000000000')       # 0.5
    e('.L1p5:\t.quad\t0x3ff8000000000000')        # 1.5
    e('.Lbias:\t.quad\t0x3f50000000000000')       # 2^-10
    e('.Ly95:\t.quad\t0x45dffffe00000000')        # 2^95 (1 - 2^-20)
    e('.Lp95:\t.quad\t0x45e0000000000000')        # 2^95
    e('.Lm102:\t.quad\t0x3990000000000000')       # 2^-102
    e('.Lc38:\t.quad\t0x4458000000000000')        # 3/8 2^72

    e('\t.section\t.note.GNU-stack,"",@progbits')
    open(sys.argv[1] if len(sys.argv) > 1 else OUT, 'w').write('\n'.join(x for x in out if x is not None) + '\n')

if __name__ == '__main__':
    main()
