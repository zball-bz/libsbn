"""The two-limb seed X_2 of the scalar reciprocal (src/fp/x86_64/inv.S) without invert_pi1's 3/2 adjustment: w3 of FIX-INV
(Claude, 2026-10-16; experiments/results/fpk_2026-10-12/fix_inv, gen/fix_w3.py there). One instruction list, used by
tools/generate_fp_inv.py (FIX w3: replaces the invert_pi1 port + v0 step + "W - 5"); the research gate's standalone copy of
the sequence (fix_inv/gen/gen_w3_test.py) is built from the same list, so that gate tests exactly the production sequence.
Original code (LGPL-3.0-or-later).

In:  rsi = d1 (top bit set), r9 = d0, r10 = dm' (D'' = d1 B^2 + d0 B + dm', the top three limbs of y B^3 as the kernel's
     normalization forms them: dm' = lo(b[nb-3] 2^ld), without the bits of b[nb-4]).
Out: X_2 = B^2 + r15 B + r14 with 0 < T - X_2 < 8, T = B^5 / D' (D' = floor(y B^3), the model's D'_2), or a jump to
     .Lc_seedclamp when X_2 < B^2 (the clamp X_2 = B^2 is then inside the same bound: T in (B^2, X_2 + 7.0002)).
Clobbers rax, rbx, rcx, rdx, rsi, rdi, r8, r11..r15, flags (the kernel reloads everything it needs at .Lc_seeddone).

Derivation (B = 2^64; every quantity an exact integer unless said otherwise).
  v = floor((B^2 - 1 - d1 B) / d1) by one div (rdx = ~d1 < d1, rax = -1): R = B + v = floor((B^2 - 1) / d1), v in [1, B - 1].
  E = B^4 - D'' R. From R in ((B^2 - 1)/d1 - 1, (B^2 - 1)/d1] and d1 >= B/2: E in (-2 B^3 + B^2, B^3 + B^2).
    D'' R = D'' B + D'' v; limbs 1..3 of D'' v + D'' B: limb1 = dm' + lo(v d0) + hi(v dm'), limb2 = d0 + lo(v d1) + hi(v d0)
    + c1, limb3 = d1 + hi(v d1) + c2 (mod B; limb 0 = lo(v dm') carries nothing). E_used = ~S (4 limbs) = E - 1 mod B^4, exact
    as a signed 4-limb value (|E_used| < B^4 / 2): e3 = its top limb (signed, -2 .. 1), e2 the next (unsigned); e1, e0 dropped.
  T'' = B^5 / D'' = B R / (1 - eps), eps = E / B^4, |eps| < 2/B:
    T'' = B R + R E / B^3 + R E^2 / B^7 + c3,  |c3| < 2 B^2 (2/B)^3 / (1 - 2/B) < 1e-18.
  Linear term: R E / B^3 = E / B^2 + v E / B^3 = [e3 B + e2 + v e3 + hi(v e2)] + dL with
    dL = (e1 B + e0 + 1) / B^2 + lo(v e2) / B + v (e1 B + e0 + 1) / B^3 in (0, 3).
    (v e3 with signed e3: mulx v (e3 mod B), minus v B when e3 < 0.)
  Quadratic term: Qt = (R / B) tau^2, tau = E / B^3 in (-2, 1.01). Q2 = floor(t^2 r16 / 2^48), t = floor(E_used / 2^176)
    (shld of e3:e2 by 16: t / 2^16 in (tau - 2^-16 - B^-3, tau - B^-3]), r16 = 2^16 + floor(v / 2^48) (r16 / 2^16 in
    (R/B - 2^-16, R/B]); |t| < 2^17 + 1, t^2 r16 < 2^52 (exact). Qt - Q2 in (-1.3e-4, 1 + 1.3e-4).
  So T'' - X'' in (-1.3e-4, 4.0002) for X'' = B R + e3 B + e2 + v e3 + hi(v e2) + Q2.
  T vs T'': D' - D'' = the bits of b[nb-4] in dm, in [0, B/2) (ld <= 63); D', D'' >= B^3/2: T'' - T in [0, 2).
  X_2 = X'' - 3: Delta_2 = T - X_2 in (0.9998, 7.0002), inside the model's (0, 8) (newton_inv model_c: X_2 = W - 5, Delta < 8).
  X_2 < T <= 2 B^2: the top limb 1 + sext(Z_hi) + carry is 0 (clamp: X_2 < B^2, then B^2 in (T - 7.0002, T)) or 1.
Latency (M, CPU 12, chain fed back): the invert_pi1 port + v0 + W - 5 61 cycles; this sequence: see FIX-INV RESULTS.md.
"""

W3_LINES = [
    'movq\t%rsi, %rdx',
    'notq\t%rdx',
    'movq\t$-1, %rax',
    'divq\t%rsi',                      # rax = v = floor((B^2 - 1) / d1) - B
    'movq\t%rax, %rdx',
    'mulxq\t%r10, %rcx, %rcx',         # hi(v dm')
    'mulxq\t%r9, %r11, %r12',          # lo / hi(v d0)
    'mulxq\t%rsi, %r13, %rbx',         # lo / hi(v d1)
    'movq\t%rax, %r8',
    'shrq\t$48, %r8',
    'btsq\t$16, %r8',                  # r16 = 2^16 + floor(v / 2^48)
    'xorl\t%edi, %edi',                # CF = OF = 0, rdi = 0
    'adcxq\t%r10, %r11',               # limb1 = lo(v d0) + dm'
    'adoxq\t%rcx, %r11',               #       + hi(v dm')
    'adcxq\t%r9, %r13',                # limb2 = lo(v d1) + d0 + c
    'adoxq\t%r12, %r13',               #       + hi(v d0) + c'
    'adcxq\t%rsi, %rbx',               # limb3 = hi(v d1) + d1 + c
    'adoxq\t%rdi, %rbx',               #       + c'
    'notq\t%r13',                      # e2 (E_used = ~S)
    'notq\t%rbx',                      # e3 (signed, -2 .. 1)
    'movq\t%rbx, %rcx',
    'shldq\t$16, %r13, %rcx',          # t = floor(E_used / 2^176)
    'mulxq\t%rbx, %r14, %r15',         # v (e3 mod B)
    'mulxq\t%r13, %rdx, %rdx',         # hi(v e2)
    'imulq\t%rcx, %rcx',               # t^2
    'movq\t%rbx, %rsi',
    'sarq\t$63, %rsi',
    'andq\t%rax, %rsi',                # v when e3 < 0
    'subq\t%rsi, %r15',                # (r15:r14) = v e3, signed
    'imulq\t%r8, %rcx',                # t^2 r16
    'addq\t%r13, %r14',                # Z' = e3 B + e2 + v e3 + hi(v e2) - 3 (signed, r15:r14)
    'adcq\t%rbx, %r15',
    'addq\t%rdx, %r14',
    'adcq\t$0, %r15',
    'subq\t$3, %r14',
    'sbbq\t$0, %r15',
    'movq\t%r15, %rsi',
    'sarq\t$63, %rsi',                 # sext(Z'_hi)
    'addq\t%rax, %r15',                # x1 = v + Z'_hi
    'adcq\t$1, %rsi',                  # top = 1 + sext + carry
    'shrq\t$48, %rcx',                 # Q2 = floor(t^2 r16 / 2^48) in [0, 8]
    'addq\t%rcx, %r14',
    'adcq\t$0, %r15',
    'adcq\t$0, %rsi',                  # top of X_2 = B^2 + x1 B + x0: 0 (clamp) or 1
]
W3_CLAMP_JUMP = 'jz\t.Lc_seedclamp'
