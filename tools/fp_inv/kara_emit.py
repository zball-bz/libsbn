# KARA512 (Claude, 2026-10-15; experiments/results/fpk_2026-10-12/kara512, then slim_inv and ovh2_inv: gen/kara_emit.py there):
# the one-level Karatsuba middle product of the reciprocal's IFMA levels, emitted into the engine by
# tools/generate_fp_inv_engine.py (exec'd in its namespace: e, lab, z, LO, HI, GM, VR, HR, PHI, CPHI, T1..T3, BADC, red, zb,
# store_v, tst2, the frame offsets) when --kara-policy is given (the shipped engine: tools/fp_inv/kara_policy.txt). SLIM-INV
# (2026-10-16): the inv-mode leaf calls (pair_loop, leftover, the .Ldtab targets) take any segment unroll U.
# The rsqrt mode (RSQ, --kara-rs of a research engine generator) is a research variant; the reciprocal's generator has no
# such option, so it is never emitted here.
#
# Level h -> nn with NV = 2V or 2V - 1 vectors (V' = NV - V), C = 8 V columns per half: the MP window W (columns
# c0 .. c0 + 8 NV - 1 of D X_h, every full product of those diagonals, mod beta^(8 NV)) as lo half (P0 + P1, V vectors)
# and hi half (P2 - P1, V' vectors) over P pairs of C-row chunks a0 (lower rows), a1 (upper rows) aligned at the top row
# i = h (X[n] = 1), the remaining Lr bottom rows plain on both halves, or one more pair padded below X_h[0] (zero rows):
#   S_k = D_k + D_{k+C} (one array for every pair),  t' = A0 - A1 + beta^C per pair (digits t'_0..C-1, rho = 1 - top digit)
#   P0 = sum over a1 rows of x_i S'[c + K - i],  P2 = over a0 rows (c < 8 V'),  P1 = sum_j t'_j D[c + K - i0 - j]
#   (K = n + 2 - 8 NV: row i, lane c reads D[c + K - i]). S' and t' are normalized by one carry / borrow step (exact
#   fallback when a lane is left out of range); the carry polynomials kappa (S = S' + (beta - x) kappa) and b (t' = t +
#   (beta - x) b + rho x^C) leave telescoped boundary terms: lane 0 gets D4 - Cc, lane C gets A - B - D3 - D4, out_lo
#   gets - rho R and out_hi + rho R (R = the D row of T1 at j = C), so W is exactly the plain window (bit-identical
#   canonical digits). A bias 2^11 (beta - 1) per lane (+ 2^11 at lane 0; sum 2^11 beta^(8 NV) = 0 mod beta^(8 NV))
#   keeps every lane in [0, 2^64). Leaves: the engine's own segment loops (.Lseg{V-1}, .Lseg{V'-1}, the duff table).
# Scratch (KS bytes) is taken below the level's frame only on this path; the canonical digits then go through the
# stubs' own canon / G / correction / update (joins .Lst{NV}_m, .Lbig_a): the plain path's bytes are unchanged.
import ksched as KSCH                       # (tools/generate_fp_inv_engine.py puts this directory on sys.path)

_X = 1280 if getattr(a, 'kara_rs', '') else 0   # rsqrt mode: KF holds 40 vectors (two column-block pairs)
KS = 21056 + _X                 # scratch below the level's frame (64-byte multiple)
KF, WHI, RS, SCR, SB, KR, TB = 0, 1280 + _X, 1920 + _X, 2560 + _X, 3072 + _X, 9216 + _X, 15424 + _X   # KF raw window 20 v, WHI out_hi init 10 v, RS rho R
                                                                        # sums 10 v, SCR scalars + rho bytes, SB S' digits,
                                                                        # KR reversed carries (+ zero tail), TB t' rows
SB_DIG, KR_DIG, TB_DIG = (KR - SB) // 8, (TB - KR) // 8 - 8, (KS - TB) // 8
sV, sVP, sC, sP, sLR, sPAD, sNS, sH = 0, 4, 8, 12, 16, 20, 24, 28
sDK, sXR, sSK, sI0 = 32, 40, 48, 56
sTGTA, sTGTB, sTZA, sTZB, sTREDA, sTREDB, sTSTA, sTSTB, sTC1, sTHST = 64, 72, 80, 88, 96, 104, 112, 120, 128, 136
sRBX, sR12, sR13, sR9, sRDX = 144, 152, 160, 168, 176
sHA, sHB, sHCC, sHD3, sHD4, sSEAM, sL0, sLC = 184, 192, 200, 208, 216, 224, 232, 240
RHO_OFF = SCR + 256             # rho mask bytes (0 / 0xff), one per pair, P <= 32
assert SCR + 248 + 8 <= RHO_OFF and RHO_OFF + 32 <= SB and TB + 8 * 704 == KS
KBIAS_K = 11
sDUP = 248                      # duplication ablations (timing only; --kara-dup): the repeat counter
sHD = 288                       # h + delta: the S array starts at kmin' = K - h - delta, a multiple of 8 (aligned D loads)
sXI0, sXPI = 296, 304           # the exact t' pass: i0 and the pair index across its inner loop
sB10, sB11, sB8, sBCX, sMSK = 312, 320, 328, 332, 336   # rsqrt mode: the band setup's r10 / r11 / r8d / ecx (plain fallback), the S tail mask
sKREL, sKFO, sRET, sXB1, sP23 = 340, 344, 352, 360, 364  # rsqrt mode: n + 17 - K'', the pair's KF offset, the core's continuation, B - 1, pair (2,3) on
sQPAD, sQXR, sQDK = 368, 376, 384                        # rsqrt mode: pair (0,1)'s pad, Xr', Dk'' (the block remainders after pair (2,3)'s core)
sTOPX, sS1, sP2H, sE, sXTOP, sTHSTB, sLX = 392, 400, 408, 416, 424, 432, 440   # rsqrt mode: a pair below the window top (pair
# (0, 1) of B >= 3 blocks): its dropped lane-2C terms Y = P2's top hi - P1's top hi + E + D3 (signed) carried into block 2's
# lane 0; every lane above the pair carries the bias 2^11 (beta - 1) too (pair (2, 3) without its lane-0 extra sLX, plain
# blocks by an add), so the whole window's bias is 2^11 beta^(8 NV) = 0 and block 2's lanes absorb a negative Y
assert 448 <= 512
DUP = set(a.kara_dup.split(',')) if a.kara_dup else set()


def dup_begin(tag):
    if tag in DUP: e(f'movl $2, {S(sDUP)}'); lab(f'.Lk_d{tag}')


def dup_end(tag):
    if tag in DUP: e(f'decl {S(sDUP)}'); e(f'jnz .Lk_d{tag}')
S = lambda o: f'{SCR + o}(%rsp)'
OF = lambda o: f'{o + KS}(%rsp)'
XG = [(LO[v, g], HI[v, g]) for v in range(NVMAX) for g in range(1, GM[v])]


def kara_padmax(nv):
    V = (nv + 1) // 2; Vp = nv - V; C = 8 * V
    om = 10 * V + 40                                         # per-pair overhead in vector-row equivalents (MODEL.md)
    lim = 2 * C - (C * (2 * V + Vp) + om) / nv
    return max(0, min(8 * Vp, int(lim)))


RSQ = bool(getattr(a, 'kara_rs', ''))   # rsqrt mode (gen_engine_xk.py --kara-rs NV,..): the square-first band's rectangle
if RSQ:
    # the band rows 0 .. h + 6 over every vector (rows below: the staircase, which reads D_h only where it is zero) as the
    # inv geometry with h' = h + 6 and K' = K + 6; at most one staircase row may join the bottom pair (pad <= 1: its
    # top-vector products read D_h at n + 1 .. n + 8, zero); NV <= 20, even (odd NV: 2C = 8 NV + 8 > h + 8 rows)
    KNV = set() if a.kara_rs == 'none' else set(int(x) for x in a.kara_rs.split(','))   # none: the control (code present, never routed)
    assert all((x % 2 == 0 and 10 <= x <= 20) or 21 <= x <= 40 for x in KNV), KNV   # NV 21..40: column-block pairs (rs_xbig)
    V23MIN = 6                  # pair (2,3) only when the top block has >= V23MIN vectors
    PADMAX = [1] * 21
    KBITS = set()
else:
    PADMAX = [0] * 21
    if a.kara_padmax == 'model':
        for nv in range(3, 21): PADMAX[nv] = kara_padmax(nv)
    elif a.kara_padmax:
        for nv in range(3, 21): PADMAX[nv] = min(8 * (nv // 2), int(a.kara_padmax))
    # the policy: lines "c nn" (c 0: the last level l = 0, c 1: l >= 1) -> bit c * 640 + nn
    KBITS = set()
    for line in open(a.kara_policy):
        line = line.split('#')[0].split()
        if len(line) == 2: KBITS.add(int(line[0]) * 640 + int(line[1]))
    KNV = set()
    _maxp = 0
    for n, u, h, nn, l in KSCH.all_levels():
        if (min(l, 1) * 640 + nn) not in KBITS: continue
        nv = KSCH.nv_of(h, nn)
        assert nv >= 3, f'kara level with NV {nv} (h {h} nn {nn})'
        p = KSCH.plan(h, nv, PADMAX[nv])
        V, Vp, C, P, pad, Lr = p['V'], p['Vp'], p['C'], p['P'], p['pad'], p['Lr']
        nS, kmin_ = KSCH.ns_of(n, h, nn, p)
        K = n + 2 - 8 * nv
        assert P >= 1 and P <= 32, (n, h, nn, p)
        assert 8 * nS <= SB_DIG and 8 * nS + 8 <= KR_DIG + 8 and P * C <= TB_DIG, (n, h, nn, p)
        assert pad <= 8 * Vp and n - h - pad >= -16, (n, h, nn, p)
        assert kmin_ >= -8 and kmin_ + 8 * nS - 1 + C <= n + 8, (n, h, nn, p)      # D reads inside the driver's zeroed D
        assert 5 * (P * C + Lr) + 3 < 2 ** KBIAS_K, (n, h, nn, p)   # lanes in [0, 2^64): bias >= 5 PC beta, bias + 5 (PC + Lr) beta < 2^64
        KNV.add(nv); _maxp = max(_maxp, P)



def k_tgt(tab, reg_idx, disp, slot, scale=4):
    e(f'leaq {tab}(%rip), %rsi'); e(f'movslq {disp}(%rsi,{reg_idx},{scale}), %rdi'); e('addq %rsi, %rdi'); e(f'movq %rdi, {S(slot)}')


def hsum(src, slot):
    zs = z(src)
    e(f'vextracti64x4 $1, {zs}, %ymm8'); e(f'vpaddq %ymm8, {zs.replace("zmm", "ymm")}, %ymm8')
    e('vextracti128 $1, %ymm8, %xmm9'); e('vpaddq %xmm9, %xmm8, %xmm8')
    e('vpshufd $0x4e, %xmm8, %xmm9'); e('vpaddq %xmm9, %xmm8, %xmm8'); e(f'vmovq %xmm8, {S(slot)}')


def leftover(width_slot, dk_extra):
    """the Lr bottom rows (i = Lr - 1 .. 0) on one half by the duff table; dk_extra = 0 (lo) or C (hi)"""
    e('call .Lkzx')
    e(f'movl {S(sLR)}, %eax')
    e("movl %eax, %r8d"); e("negl %r8d"); e(f"andl ${U - 1}, %r8d")                    # r0 = -Lr mod U (SLIM-INV: any U)
    e(f"leal {U - 1}(%rax), %ecx"); e(f"shrl ${LOGU}, %ecx")                             # iterations
    e(f'movq {S(sXR)}, %r10'); e('leaq -8(%r10,%rax,8), %r10'); e('leaq (%r10,%r8,8), %r10')
    e('movl %eax, %edx'); e('negq %rdx')                                           # -Lr
    if dk_extra: e(f'movl {S(sC)}, %esi'); e('addq %rsi, %rdx')                    # + C: the hi half's lanes
    e(f'movq {S(sDK)}, %r11'); e('leaq 8(%r11,%rdx,8), %r11'); e('leaq (,%r8,8), %rdx'); e('subq %rdx, %r11')
    e(f"movl {S(width_slot)}, %edx"); e(f"leal {SN * (UN - U) - U}(%r8,%rdx,{U}), %edx")   # U (width - 1) + r0 (+ the narrow segments' extra entries)
    e('leaq .Ldtab(%rip), %rsi'); e('movslq (%rsi,%rdx,4), %rdi'); e('addq %rsi, %rdi')
    e('movl $1, %eax'); e('call *%rdi')


def pair_loop(tag, ptrs, tgt_slot, first_zeroed=True):
    """leaf calls over the P pairs: ptrs(tag) emits r10 / r11 from rbx = i0 (eax = C)"""
    e(f'movq {S(sI0)}, %rbx'); e('xorl %r8d, %r8d')
    lab(f'.Lk_{tag}')
    e('testl %r8d, %r8d'); e(f'jz .Lk_{tag}a'); e('call .Lkzx')
    lab(f'.Lk_{tag}a')
    e(f'movl {S(sC)}, %eax')
    ptrs()
    if 'C' in DUP:                                                                  # ablation: two calls of V iterations
        e(f'movl {S(sV)}, %ecx'); e('movl $1, %eax'); e(f'call *{S(tgt_slot)}'); e('call .Lkzx')
        e(f'movl {S(sV)}, %ecx'); e('movl $1, %eax'); e(f'call *{S(tgt_slot)}')
    else:
        e(f"movl {S(sV)}, %ecx"); e("addl %ecx, %ecx" if U == 4 else f"shll ${3 - LOGU}, %ecx" if U < 8 else "nop"); e("movl $1, %eax")   # C / U = 8 V / U iterations (SLIM-INV: any U)
        e(f'call *{S(tgt_slot)}')
    e(f'movl {S(sC)}, %eax'); e('addq %rax, %rax'); e('subq %rax, %rbx')           # i0 -= 2 C
    e('incl %r8d'); e(f'cmpl {S(sP)}, %r8d'); e(f'jb .Lk_{tag}')


def tblock(exact):
    """per pair: t' = A0 - A1 + beta^C, the borrows b, the five dots A, B, Cc, D3, D4 (zmm20..24). Fast: one borrow step
    (bit 63 of a t' lane marks a borrow that would ripple: zmm19); exact (the fallback): the lane borrow chain of S4X's
    exact pass (generate d < 0, propagate lo = 0, the 8-lane chain through a GPR, the borrow carried across vectors)"""
    x = 'x' if exact else ''
    for r_ in range(20, 25): e(f'vpxord %zmm{r_}, %zmm{r_}, %zmm{r_}')
    if exact: e('movl $1, %eax'); e('vpbroadcastq %rax, %zmm26')
    else: e('vpxord %zmm19, %zmm19, %zmm19')
    e(f'movq {S(sI0)}, %rbx'); e('xorl %r8d, %r8d'); e(f'leaq {TB}(%rsp), %r12')
    lab(f'.Lk_tp{x}')
    e(f'movl {S(sC)}, %ecx'); e('shlq $3, %rcx')                                     # 8 C bytes
    e(f'movq {S(sXR)}, %rsi'); e('leaq (%rsi,%rbx,8), %rsi'); e('leaq (%rsi,%rcx), %r9')   # a0 = Xr + 8 i0, a1 = a0 + 8 C
    e(f'movl {S(sNS)}, %eax'); e('shll $3, %eax'); e(f'subl {S(sHD)}, %eax'); e('movslq %eax, %rax'); e('addq %rbx, %rax')
    e(f'leaq {KR}(%rsp,%rax,8), %r10'); e('leaq (%r10,%rcx), %rdx')                 # kappa_{K-1-i0-j} = KR[8nS - h - delta + i0 + j], + C
    e('movq %rbx, %rax'); e('negq %rax'); e(f'movq {S(sDK)}, %r11'); e('leaq -64(%r11,%rax,8), %r11'); e('addq %rcx, %r11')   # D + 8(K + C - 8 - i0)
    e('movq %r11, %r13'); e('subq %rcx, %r13')                                       # D + 8(K - 8 - i0)
    e(f'movl {S(sV)}, %ecx'); e('xorl %eax, %eax')
    if exact:
        e(f'movq %rbx, {S(sXI0)}'); e(f'movl %r8d, {S(sXPI)}'); e('xorl %r8d, %r8d')     # r8 = the borrow into the vector
    else: e('vpxord %zmm25, %zmm25, %zmm25'); e('xorl %edi, %edi')
    e('.p2align 4')
    lab(f'.Lk_tl{x}')
    e('vmovdqu64 (%rsi,%rax), %zmm0'); e('vmovdqu64 (%r9,%rax), %zmm1')
    if exact:
        e('vpsubq %zmm1, %zmm0, %zmm2'); e('vpmovq2m %zmm2, %k5'); e('vpandq %zmm16, %zmm2, %zmm2'); e('vptestnmq %zmm2, %zmm2, %k6')
        e('kmovb %k5, %ebx'); e('kmovb %k6, %edi'); e('leal (%r8,%rbx,2), %ebx'); e('addl %edi, %ebx')   # chain = p + ((g << 1) | in)
        e('movl %ebx, %r8d'); e('shrl $8, %r8d'); e('xorl %edi, %ebx'); e('kmovb %ebx, %k7')            # out; the lanes taking a borrow
        e('vpsubq %zmm26, %zmm2, %zmm2{%k7}'); e('vpandq %zmm16, %zmm2, %zmm2')                         # t'
        e('kandb %k6, %k7, %k6'); e('korb %k5, %k6, %k6'); e('vpxord %zmm3, %zmm3, %zmm3'); e('vmovdqa64 %zmm26, %zmm3{%k6}')   # b
        e('vmovdqu64 %zmm2, (%r12,%rax)')
    else:
        e('vpsubq %zmm1, %zmm0, %zmm2'); e('vpsrlq $63, %zmm2, %zmm3'); e('vpandq %zmm16, %zmm2, %zmm2')
        e('valignq $7, %zmm25, %zmm3, %zmm4'); e('vpsubq %zmm4, %zmm2, %zmm2'); e('vporq %zmm2, %zmm19, %zmm19')
        e('vmovdqu64 %zmm2, (%r12,%rax)'); e('vmovdqa64 %zmm3, %zmm25')
    e('vmovdqu64 (%r10,%rax), %zmm5')
    e('vpmadd52luq %zmm5, %zmm1, %zmm20'); e('vpmadd52luq %zmm5, %zmm0, %zmm21'); e('vpmadd52luq (%rdx,%rax), %zmm1, %zmm22')
    if exact:
        e('vpermq (%r11), %zmm17, %zmm6'); e('vpmadd52luq %zmm6, %zmm3, %zmm23')
        e('vpermq (%r13), %zmm17, %zmm7'); e('vpmadd52luq %zmm7, %zmm3, %zmm24')
        e('addq $64, %rax'); e('subq $64, %r11'); e('subq $64, %r13')
    else:
        e('vpermq (%r11,%rdi), %zmm17, %zmm6'); e('vpmadd52luq %zmm6, %zmm3, %zmm23')
        e('vpermq (%r13,%rdi), %zmm17, %zmm7'); e('vpmadd52luq %zmm7, %zmm3, %zmm24')
        e('addq $64, %rax'); e('subq $64, %rdi')
    e('decl %ecx'); e(f'jnz .Lk_tl{x}')
    e('addq %rax, %r12')                                                             # the next pair's t'
    if exact:
        e('movl %r8d, %eax'); e(f'movq {S(sXI0)}, %rbx'); e(f'movl {S(sXPI)}, %r8d')       # rho = the borrow out of digit C - 1
    else: e('vextracti64x2 $3, %zmm25, %xmm8'); e('vpextrq $1, %xmm8, %rax')
    e('negl %eax'); e(f'movb %al, {RHO_OFF}(%rsp,%r8)')
    e(f'movl {S(sC)}, %eax'); e('addq %rax, %rax'); e('subq %rax, %rbx')
    e('incl %r8d'); e(f'cmpl {S(sP)}, %r8d'); e(f'jb .Lk_tp{x}')


def stair(first_seg, nseg, rows, yoff, xr=None, dk=None, pd=None):
    """rsqrt mode: band rows below a pair's rectangle (row i' = -1 - pad of the pair down), entered at segment first_seg with
    `rows` rows in it (Duff entry r0 = pad: rows % 4 = (4 - pad) % 4), then nseg - 1 more segments of 8 rows (eax = nseg);
    y offset yoff bytes from the pair's Dk''; the pair's Xr' / Dk'' / pad slots (default: the last core run's)"""
    xr, dk, pd = xr or sXR, dk or sDK, pd or sPAD
    e(f'movl {S(pd)}, %r8d')                                                       # r0 = pad (0 or 1)
    e(f'movq {S(xr)}, %r10'); e('leaq -8(%r10), %r10')                               # x: row i' = -1 - pad, + 8 r0 = Xr' - 8
    e(f'movq {S(dk)}, %r11'); e(f'leaq {8 + yoff}(%r11), %r11')                      # y: D index K'' + 1 + pad, - 8 r0 = Dk'' + 8
    e(f'movl ${rows}, %ecx') if isinstance(rows, int) else rows()
    e('addl $3, %ecx'); e('subl %r8d, %ecx'); e('shrl $2, %ecx')                     # iterations ceil((rows - pad) / 4)
    e(f'movl {nseg}, %eax') if isinstance(nseg, str) else e(f'movl ${nseg}, %eax')
    e(f'movl ${8 // U}, %r9d')
    e('leaq .Ldtab(%rip), %rsi')
    if isinstance(first_seg, int): e(f'movslq {4 * U * first_seg}(%rsi,%r8,4), %rdi')
    else: first_seg(); e('movslq (%rsi,%rdi,4), %rdi')
    e('addq %rsi, %rdi'); e('call *%rdi')


def rs_front20():
    """rsqrt mode, NV <= 20: the band setup's registers saved (plain fallback), the pair geometry, then the core"""
    e(f'movq %r10, {S(sB10)}'); e(f'movq %r11, {S(sB11)}'); e(f'movl %r8d, {S(sB8)}'); e(f'movl %ecx, {S(sBCX)}')
    e(f'movl {OF(RS_H)}, %r8d'); e('addl $6, %r8d')                                 # h' = h + 6 (rows 0 .. h + 6)
    e(f'movq {OF(PY_OFF)}, %rdi'); e('leaq (%rdi,%r8,8), %rdi')                      # Dk' = D_h + 8 (n + 8 - 8 NV)
    e('movq %r10, %rsi'); e('movl %r8d, %eax'); e('shlq $3, %rax'); e('subq %rax, %rsi')   # Xr' = S's top - 8 h'
    e('leal 9(,%r14,8), %eax'); e(f'movl %eax, {S(sKREL)}')                          # n + 17 - K' = 9 + 8 NV
    e(f'movq $0, {S(sKFO)}'); e('leaq .Lk_rsback(%rip), %rax'); e(f'movq %rax, {S(sRET)}'); e(f'movl $0, {S(sTOPX)}')
    e(f'movq ${2 ** KBIAS_K}, {S(sLX)}')
    e('leal 1(%r14), %eax'); e('shrl $1, %eax'); e('movl %r14d, %edx'); e('shrl $1, %edx')   # V, V'
    e('call .Lk_rgeo')
    e(f'cmpl $0, {S(sP)}'); e('je .Lk_rsbail')                                       # no pair: the plain band


def rs_geo():
    """rsqrt mode: .Lk_rgeo, the pair geometry (called: the slots 8 bytes up). In: r8d h', rdi Dk'', rsi Xr', eax V, edx V'.
    The S array starts at the 64-byte aligned address at or below Dk'' - 8 h' (delta digits); pad <= 1"""
    S8 = lambda o: f'{SCR + o + 8}(%rsp)'
    e('.p2align 4')
    lab('.Lk_rgeo')
    e(f'movl %r8d, {S8(sH)}'); e(f'movq %rdi, {S8(sDK)}'); e(f'movq %rsi, {S8(sXR)}'); e(f'movl %eax, {S8(sV)}'); e(f'movl %edx, {S8(sVP)}')
    e('leal (,%rax,8), %ebx'); e(f'movl %ebx, {S8(sC)}')
    e('movl %r8d, %ecx'); e('shlq $3, %rcx'); e('movq %rdi, %rax'); e('subq %rcx, %rax'); e('shrq $3, %rax'); e('andl $7, %eax')
    e('addl %r8d, %eax'); e(f'movl %eax, {S8(sHD)}')                                 # h' + delta
    e(f'leaq {SB + 8}(%rsp,%rax,8), %rsi'); e(f'movq %rsi, {S8(sSK)}')
    e('leal 1(%r8), %eax'); e('leal (%rbx,%rbx), %ecx'); e('xorl %edx, %edx'); e('divl %ecx')   # P, rem
    e('xorl %r9d, %r9d'); e('movl %edx, %r10d'); e('testl %edx, %edx'); e('jz .Lk_rg1')
    e('movl %ecx, %r11d'); e('subl %edx, %r11d'); e('cmpl $1, %r11d'); e('ja .Lk_rg1')   # one row below the rectangle at most
    e('incl %eax'); e('movl %r11d, %r9d'); e('xorl %r10d, %r10d')
    lab('.Lk_rg1')
    e(f'movl %eax, {S8(sP)}'); e(f'movl %r9d, {S8(sPAD)}'); e(f'movl %r10d, {S8(sLR)}')
    e('leal 1(%r8), %eax'); e('subl %ecx, %eax'); e('movslq %eax, %rax'); e(f'movq %rax, {S8(sI0)}')
    e(f'movl {S8(sVP)}, %eax'); e(f'movl {S8(sHD)}, %edx'); e('leal 7(%rdx,%rax,8), %eax'); e('subl %r10d, %eax'); e('shrl $3, %eax'); e(f'movl %eax, {S8(sNS)}')
    e(f'movl {S8(sV)}, %eax'); e(f'movl {S8(sVP)}, %ecx')
    def tg(tab, reg, disp, slot, scale=4):
        e(f'leaq {tab}(%rip), %rsi'); e(f'movslq {disp}(%rsi,{reg},{scale}), %rdi'); e('addq %rsi, %rdi'); e(f'movq %rdi, {S8(slot)}')
    tg('.Lkz_tab', '%rax', -4, sTZA); tg('.Lkred_tab', '%rax', -4, sTREDA); tg('.Lkst_tab', '%rax', -4, sTSTA)
    tg('.Lkc1_tab', '%rax', -4, sTC1); tg('.Lkhst_tab', '%rax', -4, sTHST); tg('.Lkhst_tab', '%rcx', -4, sTHSTB)
    tg('.Lkred_tab', '%rcx', -4, sTREDB); tg('.Lkst_tab', '%rcx', -4, sTSTB); tg('.Lbig_tab', '%rcx', -4, sTZB)
    e('shll $4, %eax'); e('shll $4, %ecx')
    tg('.Ldtab', '%rax', -16, sTGTA, scale=1); tg('.Ldtab', '%rcx', -16, sTGTB, scale=1)
    e('ret')


def rs_back():
    """rsqrt mode: the rectangle's raw window KF -> the band's staircase rows -> the stubs' joins (.Lst{NV}_m / .Lbig_m)"""
    for r_, s_ in (('rbx', sRBX), ('r12', sR12), ('r13', sR13), ('rdx', sRDX)): e(f'movq {S(s_)}, %{r_}')
    e('cmpl $10, %r14d'); e('ja .Lk_rsb')
    e(f'leaq {KF + 8}(%rsp), %r11')
    e('leaq .Lbig_tab(%rip), %rsi'); e('movslq -4(%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('call *%rdi')   # Z{NV-1}: LO[v,0] = KF[v]
    # segments NV - 2 .. 0: 8 rows each (the first one 8 - pad)
    def fs():
        e('leal -8(,%r14,4), %edi'); e('addl %r8d, %edi')                         # 4 (NV - 2) + r0
    e('movl %r14d, %eax'); e('decl %eax'); e(f'movl %eax, {S(sB8)}')
    stair(fs, f'{S(sB8)}', 8, 0)
    e(f'addq ${KS}, %rsp'); e(f'movl {SQ_I0}(%rsp), %r9d')
    e('leaq .Lkj_tab(%rip), %rsi'); e('movslq (%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('jmpq *%rdi')
    lab('.Lk_rsb')                                                                  # NV 11..20: two column blocks
    e('leaq .Lbig_tab(%rip), %rsi'); e('leal -11(%r14), %edi')
    for i in range(5):
        e(f'movslq {40 * i}(%rsi,%rdi,4), %r8'); e('addq %rsi, %r8'); e(f'movq %r8, {OF(TGT_OFF + 8 * i)}')
    e(f'leaq {KF + 8}(%rsp), %r11'); e('call .Lz9')                                 # block 0: the rectangle's lanes
    def rows0():                                                                    # block 0 beyond the rectangle: 8 (NV - 10) rows on segment 9
        e('leal -80(,%r14,8), %ecx')
    stair(9, 10, rows0, 0)
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    red(9)
    e(f'vmovdqu64 {z(HR[9])}, {OF(H9_OFF)}'); e(f'vpsrlq $52, {z(VR[9])}, {z(T1)}'); e(f'vmovdqu64 {z(T1)}, {OF(C9_OFF)}')   # save_b0
    e('call .Lrc9'); store_v(F_OFF + KS)
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k0')
    e(f'leaq {KF + 648}(%rsp), %r11')
    e('leaq .Lbig_tab(%rip), %rsi'); e('movslq -44(%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('call *%rdi')  # Z{nb-1}: block 1
    e('cmpl $11, %r14d'); e('je .Lk_rsb1')
    def fs1():
        e('leal -48(,%r14,4), %edi'); e('addl %r8d, %edi')                        # 4 (NV - 12) + r0: block 1's local segment NV - 12
    e('movl %r14d, %eax'); e('subl $11, %eax'); e(f'movl %eax, {S(sB8)}')
    stair(fs1, f'{S(sB8)}', 8, 640)
    lab('.Lk_rsb1')
    e(f'addq ${KS}, %rsp'); e(f'movl {SQ_I0}(%rsp), %r9d')
    e('jmp .Lbig_m')
    # no pair: the plain band from the saved setup
    lab('.Lk_rsbail')
    for r_, s_ in (('rbx', sRBX), ('r12', sR12), ('r13', sR13), ('rdx', sRDX)): e(f'movq {S(s_)}, %{r_}')
    e(f'movq {S(sB10)}, %r10'); e(f'movq {S(sB11)}, %r11'); e(f'movl {S(sB8)}, %r8d'); e(f'movl {S(sBCX)}, %ecx')
    e('movl %r14d, %eax'); e(f'movl ${8 // U}, %r9d')
    e(f'addq ${KS}, %rsp')
    e('leaq .Lsf_tab0(%rip), %rsi'); e('movslq (%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('jmpq *%rdi')


def rs_xbig():
    """rsqrt mode, NV 21..40 (B = 3 or 4 column blocks of 10, the plain path .Lxsfbig): the band rectangles of the block pairs
    (0, 1) (rows covering block 1: h + 7 + 8 (NV - 20)) and, for B = 4 with a top block of >= V23MIN vectors, (2, 3) (rows
    0 .. h + 6) through the core (KF + 0 / + 1280); then per block the remaining band rows (the pair's next row down on
    segment 9 or the staircase), reduction, fast round and store at F + 640 b as the plain loop does; the top block in the
    registers; .Lxbig_m with the plain path's slots (rsp = the frame - XFW, its F at [rsp, rsp + XFW))"""
    OF2 = lambda o: f'{o + KS + XFW}(%rsp)'
    def blockfin(off):
        if off == 0: e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}')
        else: e(f'vmovdqu64 {OF2(H9_OFF)}, {z(PHI)}'); e(f'vmovdqu64 {OF2(C9_OFF)}, {z(CPHI)}')
        e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
        red(9)
        e(f'vmovdqu64 {z(HR[9])}, {OF2(H9_OFF)}'); e(f'vpsrlq $52, {z(VR[9])}, {z(T1)}'); e(f'vmovdqu64 {z(T1)}, {OF2(C9_OFF)}')
        e('call .Lrc9')
        for v in range(NVMAX): e(f'vmovdqu64 {z(VR[v])}, {KS + off + 64 * v}(%rsp)')
        e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k6'); e('korb %k6, %k0, %k0')
    def plain_rows(nrows_fn, yoff, seg_fn, nseg_fn):   # rows from row 0 (x = S's top) on segment seg, Duff, as the plain loop
        nrows_fn()                                                                  # r8d = rows
        e('leal 3(%r8), %ecx'); e('shrl $2, %ecx'); e('negl %r8d'); e('andl $3, %r8d')
        e(f'movq {S(sB10)}, %r10'); e(f'movq {OF2(PY_OFF)}, %r11'); e(f'addq ${yoff}, %r11') if isinstance(yoff, int) else yoff()
        e('call .Lzz9')
        e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %r9'); e('subq %r9, %r11')
        seg_fn()                                                                    # r8d = 4 seg + r0
        e('leaq .Ldtab(%rip), %rax'); e('movslq (%rax,%r8,4), %r8'); e('addq %rax, %r8')
        nseg_fn(); e(f'movl ${8 // U}, %r9d')
        e('call *%r8')
    def biasadd():                                                                  # a plain block above pair (0, 1): + 2^11 (beta - 1) per lane
        e('vpbroadcastq .Lkbias(%rip), %zmm31')
        for v in range(NVMAX): e(f'vpaddq %zmm31, {z(VR[v])}, {z(VR[v])}')
    e('.p2align 4')
    lab('.Lksfx')
    e(f'subq ${KS + XFW}, %rsp')
    for r_, s_ in (('rbx', sRBX), ('r12', sR12), ('r13', sR13), ('r9', sR9), ('rdx', sRDX)): e(f'movq %{r_}, {S(s_)}')
    e(f'movq %r10, {S(sB10)}'); e(f'movq %r11, {S(sB11)}'); e(f'movl %r8d, {S(sB8)}'); e(f'movl %ecx, {S(sBCX)}')
    e('leal -1(%r14), %eax'); e('imull $205, %eax, %eax'); e('shrl $11, %eax'); e(f'movl %eax, {S(sXB1)}')   # B - 1
    # pair (0, 1): h' = h + 6 + 8 (NV - 20)
    e(f'movl {OF2(RS_H)}, %r8d'); e('leal -154(%r8,%r14,8), %r8d')
    e(f'movq {OF2(PY_OFF)}, %rdi'); e('leaq (%rdi,%r8,8), %rdi')
    e('movq %r10, %rsi'); e('movl %r8d, %eax'); e('shlq $3, %rax'); e('subq %rax, %rsi')
    e(f'movl $169, {S(sKREL)}'); e(f'movq $0, {S(sKFO)}'); e('leaq .Lk_x1(%rip), %rax'); e(f'movq %rax, {S(sRET)}')
    e(f'movl $1, {S(sTOPX)}'); e(f'movq ${2 ** KBIAS_K}, {S(sLX)}')                # its lane 2C is block 2's lane 0
    e('movl $10, %eax'); e('movl $10, %edx'); e('call .Lk_rgeo')
    e(f'cmpl $0, {S(sP)}'); e('je .Lk_xbail')
    e('jmp .Lkcore')
    lab('.Lk_x1')
    e(f'movq {S(sP2H)}, %rax'); e(f'subq {S(sS1)}, %rax'); e(f'addq {S(sE)}, %rax'); e(f'addq {S(sHD3)}, %rax')
    e(f'movq %rax, {S(sXTOP)}')                                                     # Y: pair (0, 1)'s dropped terms at lane 160
    e(f'movl {S(sPAD)}, %eax'); e(f'movl %eax, {S(sQPAD)}'); e(f'movq {S(sXR)}, %rax'); e(f'movq %rax, {S(sQXR)}')
    e(f'movq {S(sDK)}, %rax'); e(f'movq %rax, {S(sQDK)}'); e(f'movl $0, {S(sP23)}')
    e(f'cmpl $3, {S(sXB1)}'); e('jne .Lk_x2')
    e(f'cmpl ${30 + V23MIN}, %r14d'); e('jb .Lk_x2')
    # pair (2, 3): h' = h + 6, lanes 160 ..
    e(f'movl {OF2(RS_H)}, %r8d'); e('addl $6, %r8d')
    e(f'movq {OF2(PY_OFF)}, %rdi'); e('leaq 1280(%rdi,%r8,8), %rdi')
    e(f'movq {S(sB10)}, %rsi'); e('movl %r8d, %eax'); e('shlq $3, %rax'); e('subq %rax, %rsi')
    e('leal -151(,%r14,8), %eax'); e(f'movl %eax, {S(sKREL)}')                    # n + 17 - K'' = 8 NV - 151
    e(f'movq $1280, {S(sKFO)}'); e('leaq .Lk_x2(%rip), %rax'); e(f'movq %rax, {S(sRET)}'); e(f'movl $0, {S(sTOPX)}')
    e(f'movq $0, {S(sLX)}')                                                         # the bias of pair (0, 1) is cancelled at the window top
    e('movl $10, %eax'); e('leal -30(%r14), %edx'); e('call .Lk_rgeo')
    e(f'cmpl $0, {S(sP)}'); e('je .Lk_x2')
    e(f'movl $1, {S(sP23)}')
    e('jmp .Lkcore')
    lab('.Lk_x2')
    for r_, s_ in (('rbx', sRBX), ('r12', sR12), ('r13', sR13), ('rdx', sRDX)): e(f'movq {S(s_)}, %{r_}')
    e(f'movl {S(sXB1)}, %eax'); e(f'movl %eax, {OF2(X_NF)}'); e('imull $640, %eax, %ecx'); e(f'movq %rcx, {OF2(X_TOFF)}')
    e('imull $-10, %eax, %edi'); e('leal -1(%r14,%rdi), %edi'); e(f'movl %edi, {OF2(X_TV)}')
    e('leaq .Lbig_tab(%rip), %rsi')
    for i in range(5): e(f'movslq {40 * i}(%rsi,%rdi,4), %r8'); e('addq %rsi, %r8'); e(f'movq %r8, {OF2(TGT_OFF + 8 * i)}')
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vmovdqu64 {z(PHI)}, {OF2(H9_OFF)}'); e(f'vmovdqu64 {z(PHI)}, {OF2(C9_OFF)}')
    e('kxorb %k0, %k0, %k0')
    q = dict(xr=sQXR, dk=sQDK, pd=sQPAD)
    e(f'leaq {KF + 8}(%rsp), %r11'); e('call .Lz9')                               # block 0: segment 9 from the pair's next row
    stair(9, 10, 80, 0, **q)
    blockfin(0)
    e(f'leaq {KF + 648}(%rsp), %r11'); e('call .Lz9')                             # block 1: its staircase 8 .. 0
    stair(8, 9, 8, 640, **q)
    blockfin(640)
    e(f'movq {OF2(H9_OFF + 56)}, %rax'); e(f'addq {S(sXTOP)}, %rax'); e(f'movq %rax, {OF2(H9_OFF + 56)}')   # block 2's PHI lane 7 += X
    e(f'cmpl $3, {S(sXB1)}'); e('je .Lk_x4')
    lab('.Lk_xtop')                                                                 # the top block, plain: rows 0 .. h + 6 on its local segment
    def ntop(): e(f'movl {OF2(RS_H)}, %r8d'); e('addl $7, %r8d')
    def ytop(): e(f'addq {OF2(X_TOFF)}, %r11')
    def stop(): e(f'movl {OF2(X_TV)}, %eax'); e('leal (%r8,%rax,4), %r8d')
    def etop(): e(f'movl {OF2(X_TV)}, %eax'); e('incl %eax')
    plain_rows(ntop, ytop, stop, etop)
    biasadd()
    e('jmp .Lk_xend')
    lab('.Lk_x4')                                                                   # B = 4
    e(f'cmpl $0, {S(sP23)}'); e('je .Lk_x4p')
    e(f'leaq {KF + 1288}(%rsp), %r11'); e('call .Lz9')                            # block 2: segment 9 from pair (2, 3)'s next row
    def r2(): e('leal -240(,%r14,8), %ecx')
    stair(9, 10, r2, 0)
    blockfin(1280)
    e(f'leaq {KF + 1928}(%rsp), %r11')                                             # top block 3: Z{TV}, then TV - 1 .. 0
    e(f'movl {OF2(X_TV)}, %edi'); e('leaq .Lbig_tab(%rip), %rsi'); e('movslq (%rsi,%rdi,4), %rdi'); e('addq %rsi, %rdi'); e('call *%rdi')
    e(f'cmpl $0, {OF2(X_TV)}'); e('je .Lk_xend')
    def f3(): e(f'movl {OF2(X_TV)}, %edi'); e('leal -4(%r8,%rdi,4), %edi')
    stair(f3, f'{OF2(X_TV)}', 8, 640)
    e('jmp .Lk_xend')
    lab('.Lk_x4p')                                                                  # B = 4 without pair (2, 3): block 2 plain
    def n2(): e(f'movl {OF2(RS_H)}, %r8d'); e('leal -233(%r8,%r14,8), %r8d')
    def s2(): e('addl $36, %r8d')
    def e2(): e('movl $10, %eax')
    plain_rows(n2, 1280, s2, e2)
    biasadd()
    blockfin(1280)
    e('jmp .Lk_xtop')
    lab('.Lk_xend')
    e(f'movl {OF2(SQ_I0)}, %r9d')
    e(f'addq ${KS}, %rsp')
    e('jmp .Lxbig_m')
    lab('.Lk_xbail')                                                                # no pair: the plain path
    for r_, s_ in (('rbx', sRBX), ('r12', sR12), ('r13', sR13), ('rdx', sRDX)): e(f'movq {S(s_)}, %{r_}')
    e(f'movq {S(sB10)}, %r10'); e(f'movq {S(sB11)}, %r11'); e(f'movl {S(sB8)}, %r8d'); e(f'movl {S(sBCX)}, %ecx')
    e('movl %r14d, %eax'); e(f'movl ${8 // U}, %r9d')
    e(f'addq ${KS + XFW}, %rsp')
    e('leaq .Lsf_tab0(%rip), %rsi'); e('movslq (%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('jmpq *%rdi')
    rs_geo()


def kara_hot():
    # ------------------------------------------------ dispatch check (NV in KNV): the (level class, nn) bit
    if not RSQ:
        e('.p2align 4')
        lab('.Lkchk')
        e(f'movl {NN_OFF}(%rsp), %edi'); e(f'cmpl $0, {CH_L}(%rsp)'); e('je .Lkchk_t'); e('addl $640, %edi')
        lab('.Lkchk_t')
        e('leaq .Lkbits(%rip), %rsi'); e('btl %edi, (%rsi)'); e('jc .Lkara')
        e('leaq .Lst_tab0(%rip), %rsi'); e('movslq (%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('jmpq *%rdi')
    # ------------------------------------------------ the Karatsuba level
    e('.p2align 4')
    lab('.Lksf' if RSQ else '.Lkara')
    e(f'subq ${KS}, %rsp')
    for r_, s_ in (('rbx', sRBX), ('r12', sR12), ('r13', sR13), ('r9', sR9), ('rdx', sRDX)): e(f'movq %{r_}, {S(s_)}')
    if RSQ:
        rs_front20()
    else:
        dup_begin('G')
        if RSQ:                                                                          # the band setup: r10 S's top digit, r11 D_h + 8(n + 2 - 8 NV - h)
            e(f'movq %r10, {S(sB10)}'); e(f'movq %r11, {S(sB11)}'); e(f'movl %r8d, {S(sB8)}'); e(f'movl %ecx, {S(sBCX)}')
            e(f'movl {OF(RS_H)}, %r8d'); e('addl $6, %r8d'); e(f'movl %r8d, {S(sH)}')    # h' = h + 6 (rows 0 .. h + 6)
            e(f'movl {OF(N_OFF)}, %ecx'); e('addl $6, %ecx')                             # K' = (n + 6) + 2 - 8 NV
        else:
            e(f'movq {OF(CH_LV)}, %rsi'); e(f'movl {OF(CH_L)}, %eax'); e('movl 4(%rsi,%rax,4), %r8d')     # h = lv[l + 1]
            e(f'movl %r8d, {S(sH)}')
            e(f'movl {OF(N_OFF)}, %ecx')                                                 # n
        e('leal 1(%r14), %eax'); e('shrl $1, %eax'); e(f'movl %eax, {S(sV)}')            # V
        e('movl %r14d, %edx'); e('shrl $1, %edx'); e(f'movl %edx, {S(sVP)}')             # V'
        e('leal (,%rax,8), %ebx'); e(f'movl %ebx, {S(sC)}')                              # C = 8 V
        if RSQ:
            e(f'movq {OF(PY_OFF)}, %rdi'); e('leaq (%rdi,%r8,8), %rdi'); e(f'movq %rdi, {S(sDK)}')   # Dk' = D_h + 8 K'
            e(f'movq {S(sB10)}, %rsi'); e('movl %r8d, %eax'); e('shlq $3, %rax'); e('subq %rax, %rsi'); e(f'movq %rsi, {S(sXR)}')   # row i' = 0: S's top - h'
        else:
            e('leal 2(%rcx), %esi'); e('leal (,%r14,8), %edi'); e('subl %edi, %esi'); e('movslq %esi, %rsi')
            e(f'movq {OF(CH_D)}, %rdi'); e('leaq (%rdi,%rsi,8), %rdi'); e(f'movq %rdi, {S(sDK)}')   # Dk = D + 8 K
            e('movl %ecx, %esi'); e('subl %r8d, %esi'); e('leaq (%r15,%rsi,8), %rsi'); e(f'movq %rsi, {S(sXR)}')   # row 0
        e('leal 2(%rcx), %eax'); e('leal (,%r14,8), %edx'); e('subl %edx, %eax'); e('subl %r8d, %eax'); e('andl $7, %eax')   # delta = (K - h) mod 8
        e('addl %r8d, %eax'); e(f'movl %eax, {S(sHD)}')                                 # h + delta
        e(f'leaq {SB}(%rsp,%rax,8), %rsi'); e(f'movq %rsi, {S(sSK)}')                   # S'[K] = SB + 8 (h + delta)
        e('leal 1(%r8), %eax'); e('leal (%rbx,%rbx), %ecx'); e('xorl %edx, %edx'); e('divl %ecx')   # P, rem
        e('xorl %r9d, %r9d'); e('movl %edx, %r10d'); e('testl %edx, %edx'); e('jz .Lk_np')
        e('movl %ecx, %r11d'); e('subl %edx, %r11d')                                     # pad candidate 2C - rem
        e('leaq .Lkpad(%rip), %rsi'); e('movzwl (%rsi,%r14,2), %esi'); e('cmpl %esi, %r11d'); e('ja .Lk_np')
        e('incl %eax'); e('movl %r11d, %r9d'); e('xorl %r10d, %r10d')
        lab('.Lk_np')
        e(f'movl %eax, {S(sP)}'); e(f'movl %r9d, {S(sPAD)}'); e(f'movl %r10d, {S(sLR)}')
        if RSQ: e('testl %eax, %eax'); e('jz .Lk_rsbail')                                 # no pair: the plain band
        e('leal 1(%r8), %eax'); e('subl %ecx, %eax'); e('movslq %eax, %rax'); e(f'movq %rax, {S(sI0)}')   # i0 top = R - 2C
        e(f'movl {S(sVP)}, %eax'); e(f'movl {S(sHD)}, %edx'); e('leal 7(%rdx,%rax,8), %eax'); e('subl %r10d, %eax'); e('shrl $3, %eax'); e(f'movl %eax, {S(sNS)}')
        # section / leaf targets
        e(f'movl {S(sV)}, %eax'); e(f'movl {S(sVP)}, %ecx')
        k_tgt('.Lkz_tab', '%rax', -4, sTZA); k_tgt('.Lkred_tab', '%rax', -4, sTREDA); k_tgt('.Lkst_tab', '%rax', -4, sTSTA)
        k_tgt('.Lkc1_tab', '%rax', -4, sTC1); k_tgt('.Lkhst_tab', '%rax', -4, sTHST)
        k_tgt('.Lkred_tab', '%rcx', -4, sTREDB); k_tgt('.Lkst_tab', '%rcx', -4, sTSTB); k_tgt('.Lbig_tab', '%rcx', -4, sTZB)
        e(f"shll ${2 + LOGU}, %eax"); e(f"shll ${2 + LOGU}, %ecx")                          # 4 U (V - 1): segment V - 1, row 0 (SLIM-INV: any U)
        k_tgt(".Ldtab", "%rax", -4 * U + DTAB_EXTRA, sTGTA, scale=1); k_tgt(".Ldtab", "%rcx", -4 * U + DTAB_EXTRA, sTGTB, scale=1)
    dup_end('G')
    # ------------------------------------------------ S pass: S'_k = (D_k + D_k+C) one carry step, kappa reversed into KR
    if RSQ: lab('.Lkcore')                                                          # rsqrt mode: entered with every slot set
    dup_begin('S')
    e(f'movl {S(sHD)}, %eax'); e(f'movq {S(sDK)}, %rsi'); e('shlq $3, %rax'); e('subq %rax, %rsi')      # D + 8 kmin' (aligned)
    e(f'movl {S(sC)}, %eax'); e('leaq (%rsi,%rax,8), %rdi')
    e(f'leaq {SB}(%rsp), %r10')
    e(f'movl {S(sNS)}, %ecx'); e('movl %ecx, %r11d'); e('shlq $6, %r11'); e(f'leaq {KR - 64}(%rsp,%r11), %r11')
    e('vpbroadcastq .Lm52(%rip), %zmm16'); e('vmovdqu64 .Lkrev(%rip), %zmm17')
    e('vpxord %zmm18, %zmm18, %zmm18'); e('vpxord %zmm19, %zmm19, %zmm19')
    if RSQ:                                                                          # the last vector separately, D_h above n + 8 masked
        e('decl %ecx')
        e(f'movl {S(sHD)}, %eax'); e(f'addl {S(sKREL)}, %eax'); e(f'movl {S(sNS)}, %edx'); e('shll $3, %edx'); e('subl %edx, %eax')
        e(f'subl {S(sC)}, %eax'); e('xorl %edx, %edx'); e('testl %eax, %eax'); e('cmovsl %edx, %eax')   # lanes 9 + 8 NV + h + delta - 8 nS - C
        e('movl $255, %edx'); e('bzhil %eax, %edx, %edx'); e(f'movb %dl, {S(sMSK)}')
    e('xorl %eax, %eax'); e('movl %ecx, %r8d'); e('shrl $1, %ecx'); e('jz .Lk_s1')     # pairs of vectors, then an odd one
    def sv(o, cin, c, t0, t1, t2):                                                  # one vector at byte offset o (+ rax)
        e(f'vmovdqu64 {o}(%rsi,%rax), %zmm{t0}'); e(f'vpaddq {o}(%rdi,%rax), %zmm{t0}, %zmm{t0}')
        e(f'vpsrlq $52, %zmm{t0}, %zmm{c}'); e(f'vpandq %zmm16, %zmm{t0}, %zmm{t0}')
        e(f'valignq $7, %zmm{cin}, %zmm{c}, %zmm{t1}'); e(f'vpaddq %zmm{t1}, %zmm{t0}, %zmm{t0}'); e(f'vporq %zmm{t0}, %zmm19, %zmm19')
        e(f'vmovdqu64 %zmm{t0}, {o}(%r10,%rax)'); e(f'vpermq %zmm{c}, %zmm17, %zmm{t2}'); e(f'vmovdqu64 %zmm{t2}, {-o}(%r11)')
    e('.p2align 4')
    lab('.Lk_sl')
    sv(0, 18, 1, 0, 2, 2); sv(64, 1, 4, 3, 5, 5)
    e('vmovdqa64 %zmm4, %zmm18'); e('subq $-128, %rax'); e('addq $-128, %r11')
    e('decl %ecx'); e('jnz .Lk_sl')
    lab('.Lk_s1')
    e('testl $1, %r8d'); e('jz .Lk_s2')
    sv(0, 18, 1, 0, 2, 2)
    if RSQ: e('vmovdqa64 %zmm1, %zmm18'); e('addq $64, %rax'); e('subq $64, %r11')
    lab('.Lk_s2')
    if RSQ:                                                                          # the last vector: D_h[k + C] masked above n + 8
        e(f'kmovb {S(sMSK)}, %k6')
        e('vmovdqu64 (%rsi,%rax), %zmm0'); e('vmovdqu64 (%rdi,%rax), %zmm3{%k6}{z}'); e('vpaddq %zmm3, %zmm0, %zmm0')
        e('vpsrlq $52, %zmm0, %zmm1'); e('vpandq %zmm16, %zmm0, %zmm0')
        e('valignq $7, %zmm18, %zmm1, %zmm2'); e('vpaddq %zmm2, %zmm0, %zmm0'); e('vporq %zmm0, %zmm19, %zmm19')
        e('vmovdqu64 %zmm0, (%r10,%rax)'); e('vpermq %zmm1, %zmm17, %zmm2'); e('vmovdqu64 %zmm2, (%r11)')
    dup_end('S')
    e('vpxord %zmm2, %zmm2, %zmm2'); e(f'movl {S(sNS)}, %ecx'); e('shlq $6, %rcx'); e(f'vmovdqu64 %zmm2, {KR}(%rsp,%rcx)')   # kappa_{kmin-1} = 0
    e('vptestmq .Lb52(%rip){1to8}, %zmm19, %k5'); e('kortestb %k5, %k5')
    e('jnz .Lk_sx')
    lab('.Lk_sb')
    # ------------------------------------------------ per pair: t' = A0 - A1 + beta^C (one borrow step), the five dots
    dup_begin('T')
    tblock(False)
    dup_end('T')
    e('vpmovq2m %zmm19, %k5'); e('kortestb %k5, %k5'); e('jnz .Lk_tx')               # a t' lane left negative: the exact pass
    lab('.Lk_tb')
    dup_begin('H')
    for r_, s_ in ((20, sHA), (21, sHB), (22, sHCC), (23, sHD3), (24, sHD4)): hsum(r_, s_)
    dup_end('H')
    if RSQ:                                                                         # E = sum_j a0_j kappa_(C-1+K-i0-j): P2's top carry dot
        e(f'cmpl $0, {S(sTOPX)}'); e('je .Lk_noe')
        e('vpxord %zmm20, %zmm20, %zmm20'); e(f'movq {S(sI0)}, %rbx'); e('xorl %r8d, %r8d')
        lab('.Lk_ep')
        e(f'movq {S(sXR)}, %rsi'); e('leaq (%rsi,%rbx,8), %rsi')
        e(f'movl {S(sNS)}, %eax'); e('shll $3, %eax'); e(f'subl {S(sHD)}, %eax'); e(f'subl {S(sC)}, %eax'); e('movslq %eax, %rax'); e('addq %rbx, %rax')
        e(f'leaq {KR}(%rsp,%rax,8), %r10'); e(f'movl {S(sV)}, %ecx')
        lab('.Lk_el')
        e('vmovdqu64 (%rsi), %zmm0'); e('vpmadd52luq (%r10), %zmm0, %zmm20'); e('addq $64, %rsi'); e('addq $64, %r10'); e('decl %ecx'); e('jnz .Lk_el')
        e(f'movl {S(sC)}, %eax'); e('addq %rax, %rax'); e('subq %rax, %rbx'); e('incl %r8d'); e(f'cmpl {S(sP)}, %r8d'); e('jb .Lk_ep')
        hsum(20, sE)
        lab('.Lk_noe')
    e(f'movq {S(sHD4)}, %rax'); e(f'subq {S(sHCC)}, %rax'); e(f'movq %rax, {S(sL0)}')
    e(f'movq {S(sHA)}, %rax'); e(f'subq {S(sHB)}, %rax'); e(f'subq {S(sHD3)}, %rax'); e(f'subq {S(sHD4)}, %rax'); e(f'movq %rax, {S(sLC)}')
    # ------------------------------------------------ Rsum[v] = sum_r rho_r R_r[v] (R_r = D + 8(K - i0_r - C) + 64 v)
    e(f'cmpl $1, {S(sP)}'); e('je .Lk_r1')                                          # P = 1: rho R straight from D (.Lkc1)
    dup_begin('R')
    e(f'movq {S(sDK)}, %rsi'); e(f'movq {S(sI0)}, %rax'); e(f'movl {S(sC)}, %ecx'); e('addq %rcx, %rax'); e('negq %rax')
    e('leaq (%rsi,%rax,8), %rsi'); e(f'leaq {RS}(%rsp), %rdi'); e('movl %ecx, %edx'); e('shll $4, %edx'); e(f'movl {S(sV)}, %ecx')
    lab('.Lk_rv')
    e('vpxord %zmm0, %zmm0, %zmm0'); e('movq %rsi, %r10'); e('xorl %r8d, %r8d')
    lab('.Lk_rp')
    e(f'kmovb {RHO_OFF}(%rsp,%r8), %k5'); e('vpaddq (%r10), %zmm0, %zmm0{%k5}'); e('addq %rdx, %r10')
    e('incl %r8d'); e(f'cmpl {S(sP)}, %r8d'); e('jb .Lk_rp')
    e('vmovdqu64 %zmm0, (%rdi)'); e('addq $64, %rsi'); e('addq $64, %rdi'); e('decl %ecx'); e('jnz .Lk_rv')
    dup_end('R')
    lab('.Lk_r1')
    # ------------------------------------------------ P1: rows t'_r over D (lo geometry), V vectors
    e(f'call *{S(sTZA)}')
    e(f'leaq {TB}(%rsp), %r12')
    def p1ptr():
        e('leaq -8(%r12,%rax,8), %r10')                                              # t'_r[C - 1]
        e('movq %rbx, %rcx'); e('addq %rax, %rcx'); e('negq %rcx'); e(f'movq {S(sDK)}, %r11'); e('leaq 8(%r11,%rcx,8), %r11')   # D + 8(K + 1 - i0 - C)
        e('leaq (%r12,%rax,8), %r12')
    pair_loop('p1', p1ptr, sTGTA)
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e('vpbroadcastq .Lkbias(%rip), %zmm31')
    e(f'leaq {WHI}(%rsp), %rdi'); e(f'leaq {RS}(%rsp), %rsi'); e('kxnorb %k5, %k5, %k5')
    e(f'cmpl $1, {S(sP)}'); e('jne .Lk_c1')                                          # P = 1: rsi = R = D + 8(K - i0 - C), k5 = rho
    e(f'movq {S(sDK)}, %rsi'); e(f'movq {S(sI0)}, %rax'); e(f'movl {S(sC)}, %ecx'); e('addq %rcx, %rax'); e('negq %rax')
    e('leaq (%rsi,%rax,8), %rsi'); e(f'kmovb {RHO_OFF}(%rsp), %k5')
    lab('.Lk_c1')
    e(f'call *{S(sTC1)}')                                                            # reduce; WHI = B - P1 + rho R; lo += B - rho R
    e(f'call *{S(sTHST)}'); e(f'movq %rax, {S(sSEAM)}')                              # P1's top hi lane
    if RSQ: e(f'movq %rax, {S(sS1)}')
    e('call .Lkzh')
    # ------------------------------------------------ P0: a1 rows over S' (+ the Lr bottom rows over D), onto P1
    def p0ptr():
        e('leaq (%rbx,%rax,2), %rcx'); e(f'movq {S(sXR)}, %r10'); e('leaq -8(%r10,%rcx,8), %r10')   # Xr + 8(i0 + 2C - 1)
        e('negq %rcx'); e(f'movq {S(sSK)}, %r11'); e('leaq 8(%r11,%rcx,8), %r11')                  # S' + 8(K + 1 - i0 - 2C)
    pair_loop('p0', p0ptr, sTGTA)
    e(f'cmpl $0, {S(sLR)}'); e('je .Lk_p0z')
    leftover(sV, 0)
    lab('.Lk_p0z')
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'call *{S(sTREDA)}')
    e(f'call *{S(sTHST)}'); e(f'addq %rax, {S(sSEAM)}')                             # + P0's top hi lane
    e(f'movq {S(sL0)}, %rax')
    if RSQ: e(f'addq {S(sLX)}, %rax')                                               # rsqrt: 2^11, or 0 for pair (2, 3) above pair (0, 1)
    else: e(f'addq ${2 ** KBIAS_K}, %rax')
    e(f'vmovq %rax, {z(T1).replace("zmm", "xmm")}'); e(f'vpaddq {z(T1)}, {z(VR[0])}, {z(VR[0])}')   # lane 0
    if RSQ: e(f'movq {S(sKFO)}, %rdi'); e(f'leaq {KF}(%rsp,%rdi), %rdi')
    else: e(f'leaq {KF}(%rsp), %rdi')
    e(f'call *{S(sTSTA)}')
    # ------------------------------------------------ P2: a0 rows over S' (+ the bottom rows over D + C), from WHI
    e(f'leaq {WHI + 8}(%rsp), %r11'); e(f'call *{S(sTZB)}')
    def p2ptr():
        e('leaq (%rbx,%rax), %rcx'); e(f'movq {S(sXR)}, %r10'); e('leaq -8(%r10,%rcx,8), %r10')     # Xr + 8(i0 + C - 1)
        e('negq %rcx'); e(f'movq {S(sSK)}, %r11'); e('leaq 8(%r11,%rcx,8), %r11')                   # S' + 8(K + 1 - i0 - C)
    pair_loop('p2', p2ptr, sTGTB)
    e(f'cmpl $0, {S(sLR)}'); e('je .Lk_p2z')
    leftover(sVP, 1)
    lab('.Lk_p2z')
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'call *{S(sTREDB)}')
    if RSQ: e(f'call *{S(sTHSTB)}'); e(f'movq %rax, {S(sP2H)}')                     # P2 - P1's hi half: its top hi lane
    e(f'movq {S(sLC)}, %rax'); e(f'addq {S(sSEAM)}, %rax'); e(f'vmovq %rax, {z(T1).replace("zmm", "xmm")}'); e(f'vpaddq {z(T1)}, {z(VR[0])}, {z(VR[0])}')   # lane C
    e(f'movl {S(sV)}, %eax'); e('shll $6, %eax')
    if RSQ: e(f'addq {S(sKFO)}, %rax')
    e(f'leaq {KF}(%rsp,%rax), %rdi'); e(f'call *{S(sTSTB)}')
    if RSQ: e(f'jmpq *{S(sRET)}'); lab('.Lk_rsback'); rs_back(); rs_xbig()
    else:
        # ------------------------------------------------ join: the raw window KF -> the stub's canon
        for r_, s_ in (('rbx', sRBX), ('r12', sR12), ('r13', sR13), ('r9', sR9), ('rdx', sRDX)): e(f'movq {S(s_)}, %{r_}')
        e('cmpl $10, %r14d'); e('ja .Lk_jb')
        e(f'leaq {KF + 8}(%rsp), %r11')
        e('leaq .Lbig_tab(%rip), %rsi'); e('movslq -4(%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('call *%rdi')   # Z{NV-1}: LO[v,0] = KF[v]
        e(f'addq ${KS}, %rsp')
        e('leaq .Lkj_tab(%rip), %rsi'); e('movslq (%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('jmpq *%rdi')
        lab('.Lk_jb')                                                                   # NV 11..20: the big stub's two blocks
        e(f'leaq {KF + 8}(%rsp), %r11'); e('call .Lz9')
        e(f'vpsrlq $52, {z(VR[9])}, {z(T1)}'); e(f'vmovdqu64 {z(T1)}, {OF(C9_OFF)}')
        e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
        e('call .Lrc9'); store_v(F_OFF + KS)
        e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k0')
        e(f'leaq {KF + 648}(%rsp), %r11')
        e('leaq .Lbig_tab(%rip), %rsi'); e('movslq -44(%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('call *%rdi')  # Z{nb-1}
        e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vmovdqu64 {OF(C9_OFF)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
        e('leaq .Lbig_tab(%rip), %rsi'); e('movslq 36(%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('call *%rdi')   # RCP{nb-1}
        e(f'addq ${KS}, %rsp')
        e('leaq .Lbig_tab(%rip), %rsi'); e('leal -11(%r14), %edi')
        for i in range(5):
            e(f'movslq {40 * i}(%rsi,%rdi,4), %r8'); e('addq %rsi, %r8'); e(f'movq %r8, {TGT_OFF + 8 * i}(%rsp)')
        if a.force_exact: e('movl $640, %r8d'); e('call .Lexact')                     # (forced build: the exact pass always)
        else: tst2(BADC, '.Lbig_xa')
        e('jmp .Lbig_a')
    # ------------------------------------------------ sections (descending entries)
    for v in range(NVMAX - 1, -1, -1): lab(f'.Lkz{v}'); zb(v)
    e('ret')
    lab('.Lkzx')
    for lo_, hi_ in XG: e(f'vpxord {z(lo_)}, {z(lo_)}, {z(lo_)}'); e(f'vpxord {z(hi_)}, {z(hi_)}, {z(hi_)}')
    e('ret')
    lab('.Lkzh')
    for v in range(NVMAX):
        for g in range(GM[v]): e(f'vpxord {z(HI[v, g])}, {z(HI[v, g])}, {z(HI[v, g])}')
    for lo_, hi_ in XG: e(f'vpxord {z(lo_)}, {z(lo_)}, {z(lo_)}')
    e('ret')
    for v in range(NVMAX - 1, -1, -1): lab(f'.Lkred{v}'); red(v)
    e('ret')
    for v in range(NVMAX - 1, -1, -1): lab(f'.Lkst{v}'); e(f'vmovdqu64 {z(VR[v])}, {64 * v}(%rdi)')
    e('ret')
    for v in range(NVMAX - 1, -1, -1):
        lab(f'.Lkc1{v}')
        red(v)
        e(f'vpsubq {z(VR[v])}, %zmm31, {z(T1)}'); e(f'vpaddq {64 * v}(%rsi), {z(T1)}, {z(T1)}{{%k5}}'); e(f'vmovdqu64 {z(T1)}, {64 * v}(%rdi)')
        e(f'vpaddq %zmm31, {z(VR[v])}, {z(VR[v])}'); e(f'vpsubq {64 * v}(%rsi), {z(VR[v])}, {z(VR[v])}{{%k5}}')
    e('ret')
    for v in range(NVMAX):                                                          # rax = lane 7 of HR[v] (the top hi lane)
        lab(f'.Lkhst{v}'); e(f'vextracti64x2 $3, {z(HR[v])}, {z(T2).replace("zmm", "xmm")}'); e(f'vpextrq $1, {z(T2).replace("zmm", "xmm")}, %rax'); e('ret')


def kara_cold():
    # exact S (a fast lane reached beta): the lane carry chain of S4X's exact pass over the whole S range, vectorized:
    # generate g = s >= beta, propagate p = (s mod beta) = M52, chain = p + ((g << 1) | in) through a GPR, carried
    # across vectors; S' = (lo + carry in) mod beta, kappa = g | (p & carry in), reversed into KR as the fast pass does
    lab('.Lk_sx')
    if a.kara_count: e(f'incq {a.kara_count}_s(%rip)')                         # (counting build for the gate only)
    e(f'movl {S(sHD)}, %eax'); e(f'movq {S(sDK)}, %rsi'); e('shlq $3, %rax'); e('subq %rax, %rsi')
    e(f'movl {S(sC)}, %eax'); e('leaq (%rsi,%rax,8), %rdi'); e(f'leaq {SB}(%rsp), %r10')
    e(f'movl {S(sNS)}, %ecx'); e('movl %ecx, %r11d'); e('shlq $6, %r11'); e(f'leaq {KR - 64}(%rsp,%r11), %r11')
    e('movl $1, %eax'); e('vpbroadcastq %rax, %zmm26'); e('xorl %r8d, %r8d')
    lab('.Lk_sxl')
    e('vmovdqu64 (%rsi), %zmm0'); e('vpaddq (%rdi), %zmm0, %zmm0'); e('vpsrlq $52, %zmm0, %zmm1'); e('vpandq %zmm16, %zmm0, %zmm0')
    e('vptestmq %zmm1, %zmm1, %k5'); e('vpcmpeqq %zmm16, %zmm0, %k6')
    e('kmovb %k5, %eax'); e('kmovb %k6, %edx'); e('leal (%r8,%rax,2), %eax'); e('addl %edx, %eax')
    e('movl %eax, %r8d'); e('shrl $8, %r8d'); e('xorl %edx, %eax'); e('kmovb %eax, %k7')
    e('vpaddq %zmm26, %zmm0, %zmm0{%k7}'); e('vpandq %zmm16, %zmm0, %zmm0')
    e('kandb %k6, %k7, %k7'); e('vmovdqa64 %zmm26, %zmm1{%k7}')
    e('vmovdqu64 %zmm0, (%r10)'); e('vpermq %zmm1, %zmm17, %zmm2'); e('vmovdqu64 %zmm2, (%r11)')
    e('addq $64, %rsi'); e('addq $64, %rdi'); e('addq $64, %r10'); e('subq $64, %r11'); e('decl %ecx'); e('jnz .Lk_sxl')
    e('jmp .Lk_sb')
    # exact t' (a fast lane went negative): the whole per-pair block again with the exact borrow chain
    lab('.Lk_tx')
    if a.kara_count: e(f'incq {a.kara_count}_t(%rip)')
    tblock(True)
    e('jmp .Lk_tb')


def kara_rodata():
    e('.p2align 2')
    if RSQ:                                                                          # the band's original targets (plain fallback)
        lab('.Lsf_tab0')
        for N in range(0, NV2 + 1):
            e('.long 0' if N < a.sf_min else (f'.long .Lsf{N} - .Lsf_tab0' if N <= NVMAX else '.long .Lsfbig - .Lsf_tab0' if N <= 20
                                            else '.long .Lxsfbig - .Lsf_tab0'))
    lab('.Lst_tab0')
    e('.long 0')
    for N in range(1, NVMAX + 1): e(f'.long .Lst{N} - .Lst_tab0' if (N <= a.small or not a.tabstub) else '.long .Lsttab - .Lst_tab0')
    for N in range(NVMAX + 1, 21): e('.long .Lstbig - .Lst_tab0')
    lab('.Lkj_tab')
    for N in range(21): e(f'.long .Lst{N}_m - .Lkj_tab' if 3 <= N <= NVMAX and N > a.small else '.long 0')
    for nm in ('kz', 'kred', 'kst', 'kc1', 'khst'):
        lab(f'.L{nm}_tab')
        for v in range(NVMAX): e(f'.long .L{nm}{v} - .L{nm}_tab')
    e('.p2align 1')
    lab('.Lkpad')
    for nv in range(21): e(f'.short {PADMAX[nv]}')
    e('.p2align 3')
    lab('.Lkbits')
    for q in range(20):
        e('.quad 0x%016x' % sum(1 << (b - 64 * q) for b in KBITS if 64 * q <= b < 64 * q + 64))
    e('.p2align 6')
    lab('.Lkrev')
    for r in range(7, -1, -1): e(f'.quad {r}')
    lab('.Lkbias'); e(f'.quad {(2 ** KBIAS_K) * ((1 << 52) - 1)}')
