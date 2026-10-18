# Helper of tools/generate_fp_rsqrt_engine.py (exec'd in its namespace when --kl is given; the shipped engine does): the
# Karatsuba band of the fp reciprocal square root's engine, as the research kernel G's source/gen/kl_part.py
# (experiments/results/fpk_2026-10-12/rsqrt_s4x/joint/x512g_claude), unchanged below.
# KSQ-RSQRT (Claude, 2026-10-16; TEST-ONLY): KARA-L1's kl_part.py with lean node passes (dots fused into the Y / t passes,
# no carry / borrow stores, no dot loop; depth-1 nodes without the P1 buffer: .Lkl_leaf2's dual-destination ACC2).
# KARA-L1 (joint rsqrt, Claude, 2026-10-16; TEST-ONLY): the cubic band D_h x S through one recursive Karatsuba middle product.
# exec'd by gen_engine_kl.py (in its globals) when --kl is given; emits into the sections .lsbn_kl_text / .lsbn_kl_ro / .lsbn_kl_cold
# (orphans: every x512c section keeps its bytes but .Lsf_tab[1], which now points at .Lkl_b). LGPL-3.0-or-later.
#
# Notation: beta = 2^52; P(x, y, R, M) = sum_(L < M) beta^L sum_(i < R) x_i y_(i+L) (x_i = row i, rows descend in memory; y ascending).
# The band is the full J x 8 NV rectangle P with D_h zero-extended (karal1/MODEL.md 2), so its exact integer mod beta^(8 NV) is
# x512c's window: the canonical digits handed to x512c's .Lcu_ba are bit for bit the plain path's.
#  .Lkl_b     entry from .Lsf_tab[1]: plan byte by h (m0 / 16, depth); plain (.Lcu_b) when the byte is 0 or the shape guard fails.
#             Copies (x rows zero from J on; y = D_h zero above n), W = 0, node(rows / lanes [0, 2 m0)), the bottom leaf (rows
#             [2 m0, J)) and the right leaf (lanes [2 m0, 8 NV)), one carry round W -> F (exact serial pass, cold), the top block
#             into the window registers, x512c's cu_sp frame slots, jmp .Lcu_ba.
#  .Lkl_node  rdi x (row 0's digit), rsi y (y_0), rdx out, ecx m (half size, multiple of 8), r8d depth >= 1: adds P(x, y, 2m, 2m)
#             into out[0 .. 2m] (64-bit lanes, signed, exact). Y = y + y(+m) (one carry round, carries c stored), t = XB - XA (one
#             borrow round, borrows stored), dots, P1's buffer = the top-borrow row, children P0 = (xa, Y'), P2 = (xb, Y' + m),
#             P1 = (t, y + m) (node of half m / 2 at depth - 1, else leaf m x m / 8), combine lo += P1, hi -= P1, the boundary terms
#             (lane 0: -A0 + B0; m: A1 - A2 - B1 - B0; 2m: A3 + B1) split into 52-bit digit and carry (tools/nodecheck.py's algebra).
#  .Lkl_leaf  rdi x, rsi y, rdx out, ecx R >= 1, r8d V >= 1: adds P(x, y, R, 8 V) into out[0 .. 8 V]: column blocks of <= 10
#             vectors on the engine's .Lzz / .Lseg (Duff entry) / .Lsx, then .Lkla{t} (per vector: hi shifted in, add, store).
KL_TEXT, KL_RO, KL_COLD = '.lsbn_kl_text', '.lsbn_kl_ro', '.lsbn_kl_cold'
KL_HN = 216                                  # plan table: h < KL_HN
RXMAX, NYMAX, WMAX = 448, 912, 480           # x copy rows, y copy digits, W lanes (m0 <= 224, J <= 431, 8 NV <= 480)
KB_XC = 128
KB_YC = KB_XC + 8 * (RXMAX + 16)
KB_W = KB_YC + 8 * (NYMAX + 16)
KLF = KB_W + 8 * (WMAX + 16)
assert KLF % 64 == 0 and KB_YC % 64 == 0 and KB_W % 64 == 0
KB_M0, KB_D, KB_J, KB_LW, KB_C0, KB_RX, KB_NV, KB_H, KB_XT = 0, 4, 8, 12, 16, 20, 24, 28, 32
KN_FS, KN_X, KN_Y, KN_OUT, KN_M, KN_D, KN_B0, KN_L0, KN_LM, KN_L2 = 0, 8, 16, 24, 32, 36, 40, 48, 56, 64
KN_OCC, KN_OTT, KN_OBB, KN_OP1 = 72, 76, 80, 84
KL_FORCE = a.kl_force


def kl_plan(path=None):
    path = a.kl_plan if path is None else path
    tab = [0] * KL_HN
    if path:
        for ln in open(path):
            p = ln.split('#')[0].split()
            if len(p) != 3: continue
            h, m0, d = int(p[0]), int(p[1]), int(p[2])
            assert 0 <= h < KL_HN and 16 <= m0 <= 224 and m0 % 16 == 0 and 1 <= d <= 3 and m0 % (8 << (d - 1)) == 0, ln
            tab[h] = (m0 // 16) | (d << 5)
    return tab


def kl_zero(base, nvec_reg):
    """zero nvec_reg (>= 1) vectors from address base: .Lkl_zero (rdi, eax; clobbers rdi, eax, zmm16)"""
    if base != '%rdi': e(f'movq {base}, %rdi')
    if nvec_reg != '%eax': e(f'movl {nvec_reg}, %eax')
    e('call .Lkl_zero')


def kl_pro(disp):
    """x512c's cu_sp prologue as a call: X_NF = B - 1, X_TOFF = 640 (B - 1), X_TV, TGT_OFF = .Lbig_tab row X_TV from CU_SPNV, at
    the frame displaced by disp (the F area's: XFWC); .Lkl_pro itself is emitted at disp + 8 (its return address)"""
    e('call .Lkl_pro')


def kl_pro_body():
    sv = cu_disp(XFWC + 8)
    lab('.Lkl_pro')
    e(f'movl {CU_SPNV}(%rsp), %eax'); e('decl %eax'); e('imull $205, %eax, %eax'); e('shrl $11, %eax')
    e(f'movl %eax, {X_NF}(%rsp)')
    e('imull $640, %eax, %ecx'); e(f'movq %rcx, {X_TOFF}(%rsp)')
    e('imull $-10, %eax, %edi'); e(f'addl {CU_SPNV}(%rsp), %edi'); e('decl %edi'); e(f'movl %edi, {X_TV}(%rsp)')
    e('leaq .Lbig_tab(%rip), %rsi')
    for i in range(5):
        e(f'movslq {40 * i}(%rsi,%rdi,4), %r8'); e('addq %rsi, %r8'); e(f'movq %r8, {TGT_OFF + 8 * i}(%rsp)')
    e('ret')
    globals().update(sv)


def kl_norm_body():
    """.Lkl_norm (called at the band frame): W[0 .. LW) -> canonical digits at F (NV vectors): one carry round (signed lanes), the
    exact serial pass (cold) when a lane is left outside [0, M52]"""
    o = 8
    lab('.Lkl_norm')
    e(f'leaq {KB_W + o}(%rsp), %rsi'); e(f'leaq {KLF + o}(%rsp), %rdi'); e(f'movl {KB_NV + o}(%rsp), %ecx')
    e('vpbroadcastq .Lm52(%rip), %zmm16'); e('vpxord %zmm17, %zmm17, %zmm17'); e('vpxord %zmm18, %zmm18, %zmm18')
    lab('.Lkl_bn')
    e('vmovdqu64 (%rsi), %zmm19'); e('vpsraq $52, %zmm19, %zmm20'); e('valignq $7, %zmm17, %zmm20, %zmm21')
    e('vmovdqa64 %zmm20, %zmm17'); e('vpandq %zmm16, %zmm19, %zmm19'); e('vpaddq %zmm21, %zmm19, %zmm19')
    e('vporq %zmm19, %zmm18, %zmm18'); e('vmovdqu64 %zmm19, (%rdi)')
    e('addq $64, %rsi'); e('addq $64, %rdi'); e('decl %ecx'); e('jnz .Lkl_bn')
    if KL_FORCE: e('jmp .Lkl_bnx')
    else: e('vptestmq .Lkl_nm52(%rip){1to8}, %zmm18, %k5'); e('kortestb %k5, %k5'); e('jnz .Lkl_bnx')
    lab('.Lkl_bnb')
    e('ret')
    lab('.Lkl_zero')
    e('vpxord %zmm16, %zmm16, %zmm16')
    lab('.Lkl_zl')
    e('vmovdqu64 %zmm16, (%rdi)'); e('addq $64, %rdi'); e('decl %eax'); e('jg .Lkl_zl')
    e('ret')


def klblock():
    # ------------------------------------------------------------------ band entry
    e(f'.section {KL_TEXT},"ax",@progbits')
    e('.p2align 6')
    lab('.Lkl_b')
    hmin = 0 if a.kl_export else min([h for h, b in enumerate(kl_plan()) if b] or [KL_HN])
    if hmin: e(f'cmpl ${hmin}, {CU_H}(%rsp)'); e('jb .Lcu_b')                                # below the plan: x512c's band at once
    e(f'movl {CU_H}(%rsp), %eax')
    e(f'cmpl ${KL_HN}, %eax'); e('jae .Lcu_b')
    e('leaq .Lkl_plan(%rip), %rcx'); e('movzbl (%rcx,%rax), %ecx')
    e('movl %ecx, %r12d'); e('andl $31, %r12d'); e('jz .Lcu_b'); e('shll $4, %r12d')          # m0
    e('movl %ecx, %r13d'); e('shrl $5, %r13d')                                                 # depth
    e(f'movl {CU_NV}(%rsp), %r14d'); e('leal 16(,%r14,8), %edx'); e('leal (%r12,%r12), %ecx')
    e('cmpl %edx, %ecx'); e('ja .Lcu_b')                                                       # 2 m0 <= 8 NV + 16
    e('leal 25(%rax,%rax), %edx'); e('cmpl %edx, %ecx'); e('ja .Lcu_b')                         # 2 m0 <= J + 24
    # .Lcu_b's own setup (the frame slots x512c's .Lcu_ba reads)
    e(f'movq %r10, {CU_SPX}(%rsp)')
    e('movl %eax, %r8d')
    e('leal 7(%r8), %eax'); e(f'movl %eax, {CU_SPC0}(%rsp)'); e('leal 1(%r8,%r8), %eax'); e(f'movl %eax, {CU_SPJ}(%rsp)')
    e(f'movl %r14d, {CU_SPNV}(%rsp)')
    e(f'movl {N_OFF}(%rsp), %eax'); e('subl %r8d, %eax'); e('addl $2, %eax'); e('leal (,%r14,8), %ecx'); e('subl %ecx, %eax'); e('cltq')
    e(f'movq {CH_D}(%rsp), %rdx'); e('leaq (%rdx,%rax,8), %rax'); e(f'movq %rax, {CU_SPY}(%rsp)')   # D_h + n + 2 - 8 NV - h
    e(f'subq ${XFWC}, %rsp')
    sv = cu_disp(XFWC)
    kl_pro(XFWC)                                                                               # X_NF, X_TOFF, X_TV, TGT_OFF
    globals().update(sv)
    e(f'subq ${KLF}, %rsp')
    sv = cu_disp(XFWC + KLF)
    e(f'movl %r12d, {KB_M0}(%rsp)'); e(f'movl %r13d, {KB_D}(%rsp)'); e(f'movl %r14d, {KB_NV}(%rsp)')
    e(f'movl {CU_H}(%rsp), %ebx'); e(f'movl %ebx, {KB_H}(%rsp)')
    e('leal 1(%rbx,%rbx), %eax'); e(f'movl %eax, {KB_J}(%rsp)')                                 # J = 2h + 1
    e('leal (,%r14,8), %ecx'); e(f'movl %ecx, {KB_LW}(%rsp)')                                   # LW = 8 NV
    e('leal 7(%rbx), %edx'); e(f'movl %edx, {KB_C0}(%rsp)')                                     # c0 = h + 7
    e('leal (%r12,%r12), %edx'); e('cmpl %eax, %edx'); e('cmovbl %eax, %edx')                   # Rx = max(2 m0, J), to 8
    e('addl $7, %edx'); e('andl $-8, %edx'); e(f'movl %edx, {KB_RX}(%rsp)')
    # x copy: XC[Rx - J .. Rx) = S rows J - 1 .. 0 (x_j = S[top - j]), XC[0 .. Rx - J) = 0 (zeroed first)
    e(f'leaq {KB_XC}(%rsp), %rdi'); e('movl %edx, %ecx'); e('subl %eax, %ecx'); e('addl $7, %ecx'); e('shrl $3, %ecx')
    e('testl %ecx, %ecx'); e('jz .Lkl_b1')
    kl_zero('%rdi', '%ecx')
    lab('.Lkl_b1')
    e(f'movl {KB_RX}(%rsp), %edx'); e(f'movl {KB_J}(%rsp), %eax')
    e('movl %edx, %ecx'); e('subl %eax, %ecx'); e(f'leaq {KB_XC}(%rsp,%rcx,8), %rdi')            # dst XC + 8 (Rx - J)
    e(f'movq {CU_SPX}(%rsp), %rsi'); e('leal -1(%rax), %ecx'); e('shll $3, %ecx'); e('subq %rcx, %rsi')   # src S top - 8 (J - 1)
    e('leal 7(%rax), %ecx'); e('shrl $3, %ecx')
    lab('.Lkl_bx')
    e('vmovdqu64 (%rsi), %zmm16'); e('vmovdqu64 %zmm16, (%rdi)'); e('addq $64, %rsi'); e('addq $64, %rdi'); e('decl %ecx'); e('jnz .Lkl_bx')
    e(f'leaq {KB_XC - 8}(%rsp,%rdx,8), %rax'); e(f'movq %rax, {KB_XT}(%rsp)')                  # x top = XC + 8 (Rx - 1)
    # y copy: YC[s] = D_h[y0 + s] for s < nv = 8 NV + h - 1 (D index <= n), 0 above, to NY = max(4 m0, J + LW) + 16
    e(f'movq {CU_SPY}(%rsp), %rsi'); e(f'leaq {KB_YC}(%rsp), %rdi')
    e(f'movl {KB_LW}(%rsp), %ecx'); e('leal 6(%rcx,%rbx), %ecx'); e('shrl $3, %ecx')            # ceil(nv / 8) (reads D to n + 7: zero)
    e('movl %ecx, %r8d')
    lab('.Lkl_by')
    e('vmovdqu64 (%rsi), %zmm16'); e('vmovdqu64 %zmm16, (%rdi)'); e('addq $64, %rsi'); e('addq $64, %rdi'); e('decl %ecx'); e('jnz .Lkl_by')
    e(f'movl {KB_M0}(%rsp), %eax'); e('shll $2, %eax')                                          # 4 m0
    e(f'movl {KB_J}(%rsp), %ecx'); e(f'addl {KB_LW}(%rsp), %ecx'); e('cmpl %eax, %ecx'); e('cmovbl %eax, %ecx')
    e('addl $23, %ecx'); e('shrl $3, %ecx'); e('subl %r8d, %ecx')                               # vectors to zero above the copy
    e('jle .Lkl_b2')
    kl_zero('%rdi', '%ecx')
    lab('.Lkl_b2')
    # W = 0 over max(2 m0, LW) + 16 lanes
    e(f'leaq {KB_W}(%rsp), %rdi'); e(f'movl {KB_M0}(%rsp), %ecx'); e('addl %ecx, %ecx'); e(f'movl {KB_LW}(%rsp), %eax')
    e('cmpl %eax, %ecx'); e('cmovbl %eax, %ecx'); e('shrl $3, %ecx'); e('addl $2, %ecx')
    kl_zero('%rdi', '%ecx')
    # node over rows / lanes [0, 2 m0)
    e(f'movq {KB_XT}(%rsp), %rdi'); e(f'leaq {KB_YC}(%rsp), %rsi'); e(f'leaq {KB_W}(%rsp), %rdx')
    e(f'movl {KB_M0}(%rsp), %ecx'); e(f'movl {KB_D}(%rsp), %r8d')
    e('call .Lkl_node')
    # bottom leaf: rows [2 m0, J) over V = min(NV, ceil((LW + c0 - 2 m0) / 8)) vectors from lane 0
    e(f'movl {KB_J}(%rsp), %ecx'); e(f'movl {KB_M0}(%rsp), %eax'); e('addl %eax, %eax'); e('subl %eax, %ecx'); e('jle .Lkl_b3')
    e(f'movl {KB_LW}(%rsp), %r8d'); e(f'addl {KB_C0}(%rsp), %r8d'); e('subl %eax, %r8d'); e('addl $7, %r8d'); e('shrl $3, %r8d')
    e(f'movl {KB_NV}(%rsp), %edx'); e('cmpl %edx, %r8d'); e('cmoval %edx, %r8d')
    e(f'movq {KB_XT}(%rsp), %rdi'); e('shll $3, %eax'); e('subq %rax, %rdi')                    # x top - 8 (2 m0)
    e(f'leaq {KB_YC}(%rsp,%rax), %rsi'); e(f'leaq {KB_W}(%rsp), %rdx')
    e('call .Lkl_leaf')
    lab('.Lkl_b3')
    # right leaf: lanes [2 m0, LW) over rows [0, min(2 m0, J, LW + c0 - 2 m0))
    e(f'movl {KB_LW}(%rsp), %r8d'); e(f'movl {KB_M0}(%rsp), %eax'); e('addl %eax, %eax'); e('subl %eax, %r8d'); e('jle .Lkl_b4')
    e(f'movl {KB_LW}(%rsp), %ecx'); e(f'addl {KB_C0}(%rsp), %ecx'); e('subl %eax, %ecx')
    e('cmpl %eax, %ecx'); e('cmoval %eax, %ecx'); e(f'movl {KB_J}(%rsp), %edx'); e('cmpl %edx, %ecx'); e('cmoval %edx, %ecx')
    e('shrl $3, %r8d')
    e(f'movq {KB_XT}(%rsp), %rdi'); e('shll $3, %eax'); e(f'leaq {KB_YC}(%rsp,%rax), %rsi'); e(f'leaq {KB_W}(%rsp,%rax), %rdx')
    e('call .Lkl_leaf')
    lab('.Lkl_b4')
    e('call .Lkl_norm')                                                                         # W -> canonical F
    globals().update(sv)
    e(f'addq ${KLF}, %rsp')
    sv = cu_disp(XFWC)
    e(f'movq {X_TOFF}(%rsp), %rsi'); load_vr('%rsi')                                            # the top block in the registers
    globals().update(sv)
    e('jmp .Lcu_ba')

    kl_norm_body()
    kl_pro_body()
    # ------------------------------------------------------------------ leaf (KSQ-RSQRT: single or dual destination)
    # .Lkl_leaf: out += P;  .Lkl_leaf2 (r9 = out_b - out in bytes): out += P and out_b -= P (the P1 child of a depth-1 node:
    # out = the node's lo half, out_b = its hi half; no P1 buffer, no combine). Frame: PHI at (%rsp), delta at 64, ACC table
    # offset at 72 (40: ACC, 80: ACC2).
    e('.p2align 4')
    lab('.Lkl_leaf2')
    if a.kl_export: e(f'.globl {a.name}_kl_leaf2'); lab(f'{a.name}_kl_leaf2')
    e('subq $128, %rsp')
    e('movq %r9, 64(%rsp)'); e('movl $80, 72(%rsp)')
    e('jmp .Lkl_lf')
    e('.p2align 4')
    lab('.Lkl_leaf')
    if a.kl_export: e(f'.globl {a.name}_kl_leaf'); lab(f'{a.name}_kl_leaf')                    # test build: the unit test's entry
    e('subq $128, %rsp')
    e('movq $0, 64(%rsp)'); e('movl $40, 72(%rsp)')
    lab('.Lkl_lf')
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vmovdqu64 {z(PHI)}, (%rsp)')                 # PHI's slot (the segment loop
    e('leal 3(%rcx), %r12d'); e('shrl $2, %r12d')                                               # broadcasts x in zmm30 = PHI)
    e('negl %ecx'); e('andl $3, %ecx'); e('movl %ecx, %r13d')                                    # r0 = -R mod 4
    e('leaq (%rdi,%rcx,8), %rdi'); e('shll $3, %ecx'); e('subq %rcx, %rsi')                      # Duff bases
    e('movl %r8d, %r15d')
    lab('.Lkl_lb')
    e('movl $10, %ebx'); e('cmpl %ebx, %r15d'); e('cmovbl %r15d, %ebx'); e('decl %ebx')          # t = min(10, V left) - 1
    e('leaq .Lkl_tab(%rip), %rax'); e('movslq (%rax,%rbx,4), %r14'); e('addq %rax, %r14'); e('call *%r14')   # ZZ{t}
    e('movq %rdi, %r10'); e('movq %rsi, %r11'); e('movl %r12d, %ecx')
    e(f'leal (%r13,%rbx,{U}), %eax'); e('leaq .Ldtab(%rip), %r9'); e('movslq (%r9,%rax,4), %r14'); e('addq %r9, %r14')
    e('movl $1, %eax'); e(f'movl ${8 // U}, %r9d'); e('call *%r14')                              # rows on segment t, then .Lsx
    e(f'vmovdqu64 (%rsp), {z(PHI)}')
    e('leaq .Lkl_tab(%rip), %rax'); e('movl 72(%rsp), %ecx'); e('addq %rax, %rcx')
    e('movslq (%rcx,%rbx,4), %r14'); e('addq %rax, %r14'); e('movq 64(%rsp), %rax'); e('call *%r14')   # ACC{t} / ACC2{t}
    e(f'vmovdqu64 {z(PHI)}, (%rsp)')
    e('addq $640, %rsi'); e('addq $640, %rdx'); e('subl $10, %r15d'); e('jg .Lkl_lb')
    e('shll $6, %ebx'); e('leaq -576(%rdx,%rbx), %rdx')                                           # out + 64 V: the carry lane
    e(f'vpxord {z(T1)}, {z(T1)}, {z(T1)}'); e(f'valignq $7, {z(PHI)}, {z(T1)}, {z(T1)}')         # lane 0 = top hi lane 7
    e(f'vpaddq (%rdx), {z(T1)}, {z(T2)}'); e(f'vmovdqu64 {z(T2)}, (%rdx)')
    e('movq 64(%rsp), %rax'); e('testq %rax, %rax'); e('jz 1f')
    e(f'vmovdqu64 (%rdx,%rax), {z(T2)}'); e(f'vpsubq {z(T1)}, {z(T2)}, {z(T2)}'); e(f'vmovdqu64 {z(T2)}, (%rdx,%rax)')
    e('1:')
    e('addq $128, %rsp')
    e('ret')
    # ACC{t}: block result into out (rdx): for v = t .. 0: V_v = LO_v + (HI_v up one lane, lane 0 from HI_(v-1) or PHI), out += V_v;
    # then PHI = HI_t (the next block's lane-0 hi). Entries .Lkla{t} save HI_t in zmm31 (TR, free after the segment loop)
    for t in range(NVMAX - 1, -1, -1):
        lab(f'.Lkla{t}'); e(f'vmovdqa64 {z(HR[t])}, {z(TR)}'); e(f'jmp .Lklb{t}')
    for v in range(NVMAX - 1, -1, -1):
        lab(f'.Lklb{v}')
        prev = HR[v - 1] if v >= 1 else PHI
        e(f'valignq $7, {z(prev)}, {z(HR[v])}, {z(T3)}')
        e(f'vpaddq {z(T3)}, {z(VR[v])}, {z(VR[v])}')
        e(f'vpaddq {64 * v}(%rdx), {z(VR[v])}, {z(VR[v])}')
        e(f'vmovdqu64 {z(VR[v])}, {64 * v}(%rdx)')
    e(f'vmovdqa64 {z(TR)}, {z(PHI)}')
    e('ret')
    # ACC2{t} (KSQ-RSQRT): the same V_v, then out[v] += V_v and out_b[v] = (rdx + rax)[v] -= V_v
    for t in range(NVMAX - 1, -1, -1):
        lab(f'.Lkld{t}'); e(f'vmovdqa64 {z(HR[t])}, {z(TR)}'); e(f'jmp .Lkle{t}')
    for v in range(NVMAX - 1, -1, -1):
        lab(f'.Lkle{v}')
        prev = HR[v - 1] if v >= 1 else PHI
        e(f'valignq $7, {z(prev)}, {z(HR[v])}, {z(T3)}')
        e(f'vpaddq {z(T3)}, {z(VR[v])}, {z(VR[v])}')
        e(f'vpaddq {64 * v}(%rdx), {z(VR[v])}, {z(T1)}')
        e(f'vmovdqu64 {z(T1)}, {64 * v}(%rdx)')
        e(f'vmovdqu64 {64 * v}(%rdx,%rax), {z(T2)}')
        e(f'vpsubq {z(VR[v])}, {z(T2)}, {z(T2)}')
        e(f'vmovdqu64 {z(T2)}, {64 * v}(%rdx,%rax)')
    e(f'vmovdqa64 {z(TR)}, {z(PHI)}')
    e('ret')

    # ------------------------------------------------------------------ node (KSQ-RSQRT lean passes)
    # The algebra is KARA-L1's (tools/nodecheck.py); the dots A0..A3 are accumulated inside the Y pass from the carry vector in
    # its register and B0 / B1 inside the t pass from the borrow vector (both passes unrolled x 2, the dots by vpmadd52luq into
    # two accumulators each); no
    # carry / borrow stores and no separate dot loop on the fast path. Depth 1 (leaf children): P1 through .Lkl_leaf2 straight
    # into lo (+) and hi (-), the top-borrow row in place; depth >= 2: KARA-L1's P1 buffer and combine.
    e('.p2align 4')
    lab('.Lkl_node')
    if a.kl_export: e(f'.globl {a.name}_kl_node'); lab(f'{a.name}_kl_node')
    e('leal (%rcx,%rcx,8), %eax'); e('leal 568(,%rax,8), %eax')                                  # frame 72 m + 568 (64-aligned)
    e('subq %rax, %rsp')
    e(f'movq %rax, {KN_FS}(%rsp)'); e(f'movq %rdi, {KN_X}(%rsp)'); e(f'movq %rsi, {KN_Y}(%rsp)'); e(f'movq %rdx, {KN_OUT}(%rsp)')
    e(f'movl %ecx, {KN_M}(%rsp)'); e(f'movl %r8d, {KN_D}(%rsp)')
    e('leal (%rcx,%rcx,2), %eax'); e('leal 192(,%rax,8), %eax'); e(f'movl %eax, {KN_OCC}(%rsp)')  # Cc = 192 + 24 m (cold path)
    e('leal -128(%rax,%rax), %edx')                                                             # Tt = Cc + 24 m + 64 = 256 + 48 m
    e(f'movl %edx, {KN_OTT}(%rsp)')
    e('leal 64(%rdx,%rcx,8), %edx'); e(f'movl %edx, {KN_OBB}(%rsp)')                            # Bb = Tt + 8 m + 64 (cold path)
    e('leal 64(%rdx,%rcx,8), %edx'); e(f'movl %edx, {KN_OP1}(%rsp)')                            # P1 = Bb + 8 m + 64 (depth >= 2)
    e('movl %ecx, %r9d'); e('shll $3, %r9d')                                                     # 8 m (bytes)
    e('vpbroadcastq .Lm52(%rip), %zmm16'); e('vmovdqu64 .Lkl_rev(%rip), %zmm22')
    # Y pass (KSQ-RSQRT, unrolled x 2): Y'_s = (Y_s & M52) + c_s, c_s = Y_(s-1) >> 52, Y_s = y_s + y_(s+m), s < 3m. Per vector the
    # carries reversed times the x rows of the dots (vpmadd52luq c, x: exact for c in {0, 1}; two accumulators a dot, one per
    # slot of the pair, so no IFMA latency chain): third 0 A0 (xa), third 1 A1 (xa) and A2 (xb), third 2 A3 (xb). Slot 0: Y zmm19,
    # Y >> 52 zmm20, c zmm21, c reversed zmm27, previous shift zmm9; slot 1: zmm8, zmm9, zmm10, zmm11, previous zmm20.
    e(f'movq {KN_Y}(%rsp), %rsi'); e('leaq (%rsi,%r9), %r11'); e('leaq 128(%rsp), %rdi'); e('xorl %edx, %edx')
    e(f'movq {KN_X}(%rsp), %r12'); e('subq $56, %r12'); e('movq %r12, %r13'); e('subq %r9, %r13')     # xa / xb rows 0..7 reversed
    for r_ in (0, 1, 2, 3, 4, 5, 6, 7, 9, 18): e(f'vpxord %zmm{r_}, %zmm{r_}, %zmm{r_}')
    YS = {0: (19, 20, 21, 27, 9, 0), 1: (8, 9, 10, 11, 20, 64)}
    def ys(sl, dots):
        Y, S, C, R, P, o = YS[sl]
        e(f'vmovdqu64 {o}(%rsi,%rdx), %zmm{Y}'); e(f'vpaddq {o}(%r11,%rdx), %zmm{Y}, %zmm{Y}')
        e(f'vpsrlq $52, %zmm{Y}, %zmm{S}'); e(f'valignq $7, %zmm{P}, %zmm{S}, %zmm{C}')
        e(f'vpandq %zmm16, %zmm{Y}, %zmm{Y}'); e(f'vpaddq %zmm{C}, %zmm{Y}, %zmm{Y}'); e(f'vporq %zmm{Y}, %zmm18, %zmm18')
        e(f'vmovdqu64 %zmm{Y}, {o}(%rdi,%rdx)')
        e(f'vpermq %zmm{C}, %zmm22, %zmm{R}')
        for ptr, acc in dots: e(f'vpmadd52luq {-o}({ptr}), %zmm{R}, %zmm{acc + sl}')
    def third(t, dots):
        ptrs = sorted({p for p, _ in dots})
        e('movl %ecx, %eax'); e('shrl $4, %eax'); e('jz 2f')                                     # V / 2 pairs
        lab(f'.Lkl_ny{t}')
        ys(0, dots); ys(1, dots)
        for p in ptrs: e(f'subq $128, {p}')
        e('subq $-128, %rdx'); e('decl %eax'); e(f'jnz .Lkl_ny{t}')
        e('2:')
        e('testl $8, %ecx'); e('jz 3f')                                                         # V odd: one more vector
        ys(0, dots); e('vmovdqa64 %zmm20, %zmm9')
        for p in ptrs: e(f'subq $64, {p}')
        e('addq $64, %rdx')
        e('3:')
    third(0, [('%r12', 0)])
    e(f'movq {KN_X}(%rsp), %r12'); e('subq $56, %r12')
    third(1, [('%r12', 2), ('%r13', 4)])
    e(f'movq {KN_X}(%rsp), %r13'); e('subq $56, %r13'); e('subq %r9, %r13')
    third(2, [('%r13', 6)])
    for a_, b_, d_ in ((0, 1, 23), (2, 3, 24), (4, 5, 25), (6, 7, 26)): e(f'vpaddq %zmm{b_}, %zmm{a_}, %zmm{d_}')
    if KL_FORCE: e('jmp .Lkl_nyx')
    else: e('vptestmq .Lb52(%rip){1to8}, %zmm18, %k5'); e('kortestb %k5, %k5'); e('jnz .Lkl_nyx')
    lab('.Lkl_nyb')
    e('leal (%rcx,%rcx,2), %eax'); e('vpxord %zmm17, %zmm17, %zmm17'); e('vmovdqu64 %zmm17, 128(%rsp,%rax,8)')   # Yp[3m .. 3m + 8) = 0
    # t pass (unrolled x 2): t = XB - XA (rows m..2m-1 minus rows 0..m-1 as integers; memory ascending = position ascending),
    # borrows as 0 / 1 (t = (raw & M52) - borrow in); B0 / B1 = sum of the reversed borrows times y (y + 8 (2m - 9) - 64 w, + 8 m),
    # vpmadd52luq, two accumulators each; b_0 = -(lane 7 of the last borrow vector). Slot 0: raw zmm19, borrow zmm20, borrow in
    # zmm21, reversed zmm27, previous zmm9; slot 1: zmm8, zmm9, zmm10, zmm11, previous zmm20.
    e(f'movq {KN_X}(%rsp), %rsi'); e('subq %r9, %rsi'); e('addq $8, %rsi')                       # xa_mem = x - 8 (m - 1)
    e('movq %rsi, %r11'); e('subq %r9, %r11')                                                    # xb_mem = xa_mem - 8 m
    e(f'movl {KN_OTT}(%rsp), %eax'); e('leaq (%rsp,%rax), %rdi')
    e(f'movq {KN_Y}(%rsp), %r14'); e('leaq (%r14,%r9,2), %r14'); e('leaq -72(%r14), %r14'); e('leaq (%r14,%r9), %r15')
    e('xorl %edx, %edx'); e('xorl %ebx, %ebx')
    for r_ in (9, 12, 13, 14, 15, 18): e(f'vpxord %zmm{r_}, %zmm{r_}, %zmm{r_}')
    def ts(sl):
        X, B, I, R, P, o = YS[sl]
        e(f'vmovdqu64 {o}(%r11,%rdx), %zmm{X}'); e(f'vpsubq {o}(%rsi,%rdx), %zmm{X}, %zmm{X}')
        e(f'vpsrlq $63, %zmm{X}, %zmm{B}'); e(f'valignq $7, %zmm{P}, %zmm{B}, %zmm{I}')
        e(f'vpandq %zmm16, %zmm{X}, %zmm{X}'); e(f'vpsubq %zmm{I}, %zmm{X}, %zmm{X}'); e(f'vporq %zmm{X}, %zmm18, %zmm18')
        e(f'vmovdqu64 %zmm{X}, {o}(%rdi,%rdx)')
        e(f'vpermq %zmm{B}, %zmm22, %zmm{R}')
        e(f'vpmadd52luq {-o}(%r14,%rbx), %zmm{R}, %zmm{12 + sl}'); e(f'vpmadd52luq {-o}(%r15,%rbx), %zmm{R}, %zmm{14 + sl}')
    e('movl %ecx, %eax'); e('shrl $4, %eax'); e('jz 2f')
    lab('.Lkl_nt')
    ts(0); ts(1)
    e('subq $-128, %rdx'); e('addq $-128, %rbx'); e('decl %eax'); e('jnz .Lkl_nt')
    e('2:')
    e('testl $8, %ecx'); e('jz 3f')
    ts(0); e('vmovdqa64 %zmm20, %zmm9'); e('addq $64, %rdx'); e('subq $64, %rbx')
    e('3:')
    e('vpaddq %zmm13, %zmm12, %zmm28'); e('vpaddq %zmm15, %zmm14, %zmm29')
    if KL_FORCE: e('jmp .Lkl_ntx')
    else: e('vpmovq2m %zmm18, %k5'); e('kortestb %k5, %k5'); e('jnz .Lkl_ntx')
    e('valignq $7, %zmm9, %zmm9, %zmm19'); e('vmovq %xmm19, %rax'); e('negq %rax'); e(f'movq %rax, {KN_B0}(%rsp)')
    lab('.Lkl_ntb')
    def hsum(zr, gpr):
        e(f'vextracti64x4 $1, %zmm{zr}, %ymm19'); e(f'vpaddq %ymm19, %ymm{zr}, %ymm19')
        e('vextracti64x2 $1, %ymm19, %xmm20'); e('vpaddq %xmm20, %xmm19, %xmm19')
        e('vpshufd $0x4e, %xmm19, %xmm20'); e('vpaddq %xmm20, %xmm19, %xmm19'); e(f'vmovq %xmm19, {gpr}')
    hsum(23, '%r10'); hsum(24, '%r11'); hsum(25, '%r12'); hsum(26, '%r13'); hsum(28, '%r14'); hsum(29, '%r15')
    # b_0's terms out of B0 / B1 (row 0: y_(m-1), y_(2m-1))
    e(f'movq {KN_B0}(%rsp), %rax'); e(f'movq {KN_Y}(%rsp), %rsi')
    e('movq -8(%rsi,%r9), %rdx'); e('andq %rax, %rdx'); e('subq %rdx, %r14')
    e('movq -8(%rsi,%r9,2), %rdx'); e('andq %rax, %rdx'); e('subq %rdx, %r15')
    # the boundary terms: L0 = -A0 + B0, Lm = A1 - A2 - B1 - B0, L2m = A3 + B1
    e('movq %r14, %rax'); e('subq %r10, %rax'); e(f'movq %rax, {KN_L0}(%rsp)')
    e('movq %r11, %rax'); e('subq %r12, %rax'); e('subq %r15, %rax'); e('subq %r14, %rax'); e(f'movq %rax, {KN_LM}(%rsp)')
    e('leaq (%r13,%r15), %rax'); e(f'movq %rax, {KN_L2}(%rsp)')
    def child(setup):
        setup(); e('call .Lkl_child')
    def s_p0():
        e(f'movq {KN_X}(%rsp), %rdi'); e('leaq 128(%rsp), %rsi'); e(f'movq {KN_OUT}(%rsp), %rdx')
    def s_p2():
        e(f'movl {KN_M}(%rsp), %eax'); e('shll $3, %eax')
        e(f'movq {KN_X}(%rsp), %rdi'); e('subq %rax, %rdi'); e('leaq 128(%rsp,%rax), %rsi'); e(f'movq {KN_OUT}(%rsp), %rdx'); e('addq %rax, %rdx')
    def s_p1():
        e(f'movl {KN_M}(%rsp), %eax'); e('shll $3, %eax')
        e(f'movl {KN_OTT}(%rsp), %ecx'); e('leaq -8(%rsp,%rcx), %rdi'); e('addq %rax, %rdi')          # Tt + 8 (m - 1): row 0
        e(f'movq {KN_Y}(%rsp), %rsi'); e('addq %rax, %rsi'); e(f'movl {KN_OP1}(%rsp), %ecx'); e('leaq (%rsp,%rcx), %rdx')
    e(f'cmpl $1, {KN_D}(%rsp)'); e('je .Lkl_n1')
    # depth >= 2: P1's buffer: lane l in 1..m = -b_0 y_(m+l-1) (the top-borrow row), lanes 0 and > m zero
    e(f'movl {KN_OP1}(%rsp), %eax'); e('leaq (%rsp,%rax), %rdi'); e('movl %ecx, %eax'); e('shrl $3, %eax'); e('addl $2, %eax')
    e('movq %rdi, %r8'); kl_zero('%rdi', '%eax'); e('movq %r8, %rdi')
    e(f'cmpq $0, {KN_B0}(%rsp)'); e('je .Lkl_np')
    e(f'movq {KN_Y}(%rsp), %rsi'); e('leaq -8(%rsi,%r9), %rsi')                                  # y_(m-1)
    e('movl %ecx, %eax'); e('shrl $3, %eax'); e('xorl %edx, %edx'); e('vpxord %zmm16, %zmm16, %zmm16')
    lab('.Lkl_nr')
    e('vpsubq (%rsi,%rdx), %zmm16, %zmm17'); e('vmovdqu64 %zmm17, (%rdi,%rdx)')
    e('addq $64, %rdx'); e('decl %eax'); e('jnz .Lkl_nr')
    e('movq $0, (%rdi)'); e('movq (%rsi,%rdx), %rax'); e('negq %rax'); e('movq %rax, (%rdi,%rdx)')   # lane 0 = 0; lane m = -y_(2m-1)
    lab('.Lkl_np')
    child(s_p0); child(s_p2); child(s_p1)
    # combine: out[l] += P1[l], out[m + l] -= P1[l], l in 0 .. m + 7 (P1 zero above m)
    e(f'movl {KN_M}(%rsp), %ecx'); e('movl %ecx, %r9d'); e('shll $3, %r9d')
    e(f'movl {KN_OP1}(%rsp), %eax'); e('leaq (%rsp,%rax), %rsi'); e(f'movq {KN_OUT}(%rsp), %rdi'); e('leaq (%rdi,%r9), %r10')
    e('leal 16(%rcx), %eax'); e('shrl $4, %eax'); e('xorl %edx, %edx')                         # ceil((V + 1) / 2) pairs (P1 zero above m)
    lab('.Lkl_nc')
    e('vmovdqu64 (%rsi,%rdx), %zmm16'); e('vmovdqu64 64(%rsi,%rdx), %zmm19')
    e('vmovdqu64 (%r10,%rdx), %zmm18'); e('vpsubq %zmm16, %zmm18, %zmm18'); e('vmovdqu64 %zmm18, (%r10,%rdx)')
    e('vmovdqu64 64(%r10,%rdx), %zmm20'); e('vpsubq %zmm19, %zmm20, %zmm20'); e('vmovdqu64 %zmm20, 64(%r10,%rdx)')
    e('vpaddq (%rdi,%rdx), %zmm16, %zmm17'); e('vmovdqu64 %zmm17, (%rdi,%rdx)')
    e('vpaddq 64(%rdi,%rdx), %zmm19, %zmm21'); e('vmovdqu64 %zmm21, 64(%rdi,%rdx)')
    e('subq $-128, %rdx'); e('decl %eax'); e('jnz .Lkl_nc')
    e('jmp .Lkl_nb')
    # depth 1: the top-borrow row in place (b_0 = -1: lo[l] -= y_(m+l-1), hi[l] += y_(m+l-1), l in 1..m), then P0, P2 (leaves) and
    # P1 through .Lkl_leaf2 (lo += P1, hi -= P1)
    lab('.Lkl_n1')
    e(f'cmpq $0, {KN_B0}(%rsp)'); e('je .Lkl_n1c')
    e(f'movq {KN_Y}(%rsp), %rsi'); e('leaq -8(%rsi,%r9), %rsi')                                  # y_(m-1)
    e(f'movq {KN_OUT}(%rsp), %rdi'); e('leaq (%rdi,%r9), %r10')                                   # lo, hi
    e('movl %ecx, %eax'); e('shrl $3, %eax'); e('xorl %edx, %edx')
    e('movl $0xfe, %r8d'); e('kmovb %r8d, %k5'); e('vmovdqu64 (%rsi), %zmm17{%k5}{z}')            # lane 0 excluded
    e('jmp .Lkl_n1s')
    lab('.Lkl_n1r')
    e('vmovdqu64 (%rsi,%rdx), %zmm17')
    lab('.Lkl_n1s')
    e('vmovdqu64 (%rdi,%rdx), %zmm18'); e('vpsubq %zmm17, %zmm18, %zmm18'); e('vmovdqu64 %zmm18, (%rdi,%rdx)')
    e('vpaddq (%r10,%rdx), %zmm17, %zmm19'); e('vmovdqu64 %zmm19, (%r10,%rdx)')
    e('addq $64, %rdx'); e('decl %eax'); e('jnz .Lkl_n1r')
    e('movq (%rsi,%rdx), %rax'); e('subq %rax, (%rdi,%rdx)'); e('addq %rax, (%r10,%rdx)')      # lane m: y_(2m-1)
    lab('.Lkl_n1c')
    child(s_p0); child(s_p2)
    e(f'movl {KN_M}(%rsp), %ecx'); e('movl %ecx, %eax'); e('shll $3, %eax')
    e(f'movl {KN_OTT}(%rsp), %edx'); e('leaq -8(%rsp,%rdx), %rdi'); e('addq %rax, %rdi')            # t row 0
    e(f'movq {KN_Y}(%rsp), %rsi'); e('addq %rax, %rsi'); e(f'movq {KN_OUT}(%rsp), %rdx')
    e('movq %rax, %r9'); e('movl %ecx, %r8d'); e('shrl $3, %r8d'); e('call .Lkl_leaf2')
    e(f'movl {KN_M}(%rsp), %ecx'); e('movl %ecx, %r9d'); e('shll $3, %r9d'); e(f'movq {KN_OUT}(%rsp), %rdi')
    lab('.Lkl_nb')
    # the boundary terms, each split into its 52-bit digit and its (signed) carry: lanes 0 / 1, m / m + 1, 2m / 2m + 1
    e('movabsq $0xfffffffffffff, %r11')
    for off, slot in ((0, KN_L0), (1, KN_LM), (2, KN_L2)):
        e(f'movq {slot}(%rsp), %rax'); e('movq %rax, %rdx'); e('andq %r11, %rax'); e('sarq $52, %rdx')
        if off == 0: e('addq %rax, (%rdi)'); e('addq %rdx, 8(%rdi)')
        else: e(f'addq %rax, (%rdi,%r9,{off})'); e(f'addq %rdx, 8(%rdi,%r9,{off})')
    e(f'addq {KN_FS}(%rsp), %rsp')
    e('ret')
    # a child of the node whose frame is at rsp + 8: node of half m / 2 at depth - 1 (tail jump), or leaf m rows x m / 8 vectors
    lab('.Lkl_child')
    e(f'movl {KN_M + 8}(%rsp), %ecx'); e(f'movl {KN_D + 8}(%rsp), %r8d'); e('decl %r8d'); e('jz 1f')
    e('shrl $1, %ecx'); e('jmp .Lkl_node')
    e('1:')
    e('movl %ecx, %r8d'); e('shrl $3, %r8d'); e('jmp .Lkl_leaf')



def klcorr():
    """KARA-L1 correction product (--kl-cplan): C = sum_(q <= h) X[n - q] H[q + L], lanes 0 .. 8 NV - 1 (H zero above). Entered from
    the patched instruction of .Lcu_hz (x512c's 'movl $8, CU_SPC0(%rsp)', now jmp .Lkl_c + nop; .Lcu_c0 = the instruction after it).
    The right part (lanes [2 m0, 8 NV), every row, x512c's staircase from c0 = 8) runs first on the F area through a copy of cu_sp
    (init: row 0 = H itself); its blocks and its top block (PS) are left at F. Then the band frame: YC = H zero-extended, W = 0,
    node over rows / lanes [0, 2 m0) (x = X + n: row 0 = X[n] = 1, rows past h zero by the engine's precondition), the bottom leaf
    (rows [2 m0, h + 1)), W[2 m0 ..] += the right part, one carry round W -> F (exact pass, cold), x512c's slots for NV, the top
    block into the registers, r14 = NV, r15 = X, jmp .Lcu_ca (x512c's update)."""
    if not a.kl_cplan: return
    ctab = kl_plan(a.kl_cplan)
    hminc = 0 if a.kl_export else min([h for h, b in enumerate(ctab) if b] or [KL_HN])
    e(f'.section {KL_TEXT},"ax",@progbits')
    e('.p2align 4')
    lab('.Lkl_c')
    e(f'movl $8, {CU_SPC0}(%rsp)')                                                              # the replaced instruction
    e(f'cmpl ${hminc}, {CU_H}(%rsp)'); e('jb .Lcu_c0')
    e(f'movl {CU_H}(%rsp), %eax'); e(f'cmpl ${KL_HN}, %eax'); e('jae .Lcu_c0')
    e('leaq .Lkl_cplan(%rip), %rcx'); e('movzbl (%rcx,%rax), %ecx')
    e('movl %ecx, %r12d'); e('andl $31, %r12d'); e('jz .Lcu_c0'); e('shll $4, %r12d')          # m0
    e('movl %ecx, %r13d'); e('shrl $5, %r13d')                                                 # depth
    e('leal (%r12,%r12), %ecx'); e('leal 25(%rax), %edx'); e('cmpl %edx, %ecx'); e('ja .Lcu_c0')   # 2 m0 <= J + 24 (J = h + 1)
    e('leal -8(,%r14,8), %edx'); e('cmpl %edx, %ecx'); e('ja .Lcu_c0')                          # 2 m0 <= 8 NV - 8 (a right part)
    # the right part through cu_sp: NVo = NV - 2 m0 / 8, y0 = H + 2 m0, x0 = X + n, c0 = 8, J = h + 1
    e('incl %eax'); e(f'movl %eax, {CU_SPJ}(%rsp)')
    e('movl %r12d, %ecx'); e('shrl $2, %ecx'); e('movl %r14d, %eax'); e('subl %ecx, %eax'); e(f'movl %eax, {CU_SPNV}(%rsp)')
    e(f'movl {N_OFF}(%rsp), %eax'); e('leaq (%r15,%rax,8), %rax'); e(f'movq %rax, {CU_SPX}(%rsp)')
    e(f'movq {CH_G}(%rsp), %rax'); e('leaq (%rax,%r12,8), %rax'); e('leaq (%rax,%r12,8), %rax'); e(f'movq %rax, {CU_SPY}(%rsp)')
    m0s, ds = CU_SPNV + 8, CU_SPNV + 12                                                         # free base-frame slots (< FRAME)
    assert ds + 4 <= FRAME
    e(f'movl %r12d, {m0s}(%rsp)'); e(f'movl %r13d, {ds}(%rsp)')                                # m0, depth
    e(f'subq ${XFWC}, %rsp')
    sv = cu_disp(XFWC)
    cu_sp('kc', True)                                                                          # top block: RCP done (registers)
    e(f'movl {X_TV}(%rsp), %ecx'); e(f'movq {X_TOFF}(%rsp), %rdx'); e('leaq (%rsp,%rdx), %rdx')
    e('leaq .Lps_tab(%rip), %rax'); e('movslq (%rax,%rcx,4), %rdi'); e('addq %rax, %rdi'); e('call *%rdi')   # PS{X_TV}: F + X_TOFF
    globals().update(sv)
    e(f'subq ${KLF}, %rsp')
    sv = cu_disp(XFWC + KLF)
    m0s += XFWC + KLF; ds += XFWC + KLF
    e(f'movl {m0s}(%rsp), %eax'); e(f'movl %eax, {KB_M0}(%rsp)'); e(f'movl {ds}(%rsp), %eax'); e(f'movl %eax, {KB_D}(%rsp)')
    e(f'movl {CU_NV}(%rsp), %r14d'); e(f'movl %r14d, {KB_NV}(%rsp)'); e('leal (,%r14,8), %ecx'); e(f'movl %ecx, {KB_LW}(%rsp)')
    e(f'movl {CU_H}(%rsp), %ebx'); e('leal 1(%rbx), %eax'); e(f'movl %eax, {KB_J}(%rsp)')
    # YC = H[0 .. 8 NV), zero to max(4 m0, J + LW) + 16 lanes
    e(f'movq {CH_G}(%rsp), %rsi'); e(f'leaq {KB_YC}(%rsp), %rdi'); e('movl %r14d, %ecx')
    lab('.Lkl_cy')
    e('vmovdqu64 (%rsi), %zmm16'); e('vmovdqu64 %zmm16, (%rdi)'); e('addq $64, %rsi'); e('addq $64, %rdi'); e('decl %ecx'); e('jnz .Lkl_cy')
    e(f'movl {KB_M0}(%rsp), %eax'); e('shll $2, %eax'); e(f'movl {KB_J}(%rsp), %ecx'); e(f'addl {KB_LW}(%rsp), %ecx')
    e('cmpl %eax, %ecx'); e('cmovbl %eax, %ecx'); e('addl $23, %ecx'); e('shrl $3, %ecx'); e('subl %r14d, %ecx')
    kl_zero('%rdi', '%ecx')
    # W = 0 over LW + 16 lanes (2 m0 < LW)
    e(f'leaq {KB_W}(%rsp), %rdi'); e('leal 2(%r14), %ecx')
    kl_zero('%rdi', '%ecx')
    # node: x = X + n (row 0), y = YC, out = W
    e(f'movl {N_OFF}(%rsp), %eax'); e(f'movq {CU_X}(%rsp), %rdi'); e('leaq (%rdi,%rax,8), %rdi'); e(f'movq %rdi, {KB_XT}(%rsp)')
    e(f'leaq {KB_YC}(%rsp), %rsi'); e(f'leaq {KB_W}(%rsp), %rdx'); e(f'movl {KB_M0}(%rsp), %ecx'); e(f'movl {KB_D}(%rsp), %r8d')
    e('call .Lkl_node')
    # bottom leaf: rows [2 m0, J) over lanes [0, 2 m0) (2 m0 / 8 vectors; the right part has every row of lanes >= 2 m0)
    e(f'movl {KB_J}(%rsp), %ecx'); e(f'movl {KB_M0}(%rsp), %eax'); e('addl %eax, %eax'); e('subl %eax, %ecx'); e('jle .Lkl_c3')
    e('movl %eax, %r8d'); e('shrl $3, %r8d')
    e(f'movq {KB_XT}(%rsp), %rdi'); e('shll $3, %eax'); e('subq %rax, %rdi'); e(f'leaq {KB_YC}(%rsp,%rax), %rsi'); e(f'leaq {KB_W}(%rsp), %rdx')
    e('call .Lkl_leaf')
    lab('.Lkl_c3')
    # W[2 m0 ..] += the right part (F at rsp + KLF, NV - 2 m0 / 8 vectors)
    e(f'movl {KB_M0}(%rsp), %eax'); e(f'leaq {KB_W}(%rsp,%rax,8), %rdi'); e('leaq (%rdi,%rax,8), %rdi')
    e(f'leaq {KLF}(%rsp), %rsi'); e(f'movl {KB_NV}(%rsp), %ecx'); e('shrl $2, %eax'); e('subl %eax, %ecx')
    lab('.Lkl_ca')
    e('vmovdqu64 (%rsi), %zmm16'); e('vpaddq (%rdi), %zmm16, %zmm16'); e('vmovdqu64 %zmm16, (%rdi)')
    e('addq $64, %rsi'); e('addq $64, %rdi'); e('decl %ecx'); e('jnz .Lkl_ca')
    e('call .Lkl_norm')                                                                         # W -> canonical F
    globals().update(sv)
    e(f'addq ${KLF}, %rsp')
    sv = cu_disp(XFWC)
    # x512c's slots for the full window (NV): cu_sp's prologue with CU_SPNV = NV, then the top block into the registers
    e(f'movl {CU_NV}(%rsp), %r14d'); e(f'movl %r14d, {CU_SPNV}(%rsp)')
    kl_pro(XFWC)
    e(f'movq {X_TOFF}(%rsp), %rsi'); load_vr('%rsi')
    e(f'movq {CU_X}(%rsp), %r15')
    globals().update(sv)
    e('jmp .Lcu_ca')


def klcold():
    """exact serial passes (cold): the node's Y and t passes (KSQ-RSQRT: each followed by its dots from the stored carries /
    borrows, KARA-L1's dot loop), the band's carry round"""
    e(f'.section {KL_COLD},"ax",@progbits')
    # Y: rsi = y, r11 = y + 8 m, 3 m digits; Y'_s, c_s exactly (c_s = carry into s)
    lab('.Lkl_nyx')
    e(f'movq {KN_Y}(%rsp), %rsi'); e('leaq (%rsi,%r9), %r11'); e('leaq 128(%rsp), %rdi')
    e(f'movl {KN_OCC}(%rsp), %eax'); e('leaq (%rsp,%rax), %r10')
    e('leal (%rcx,%rcx,2), %eax'); e('xorl %edx, %edx'); e('xorl %r8d, %r8d'); e('movabsq $0xfffffffffffff, %rbx')
    lab('.Lkl_nyx1')
    e('movq %r8, (%r10,%rdx,8)'); e('addq (%rsi,%rdx,8), %r8'); e('addq (%r11,%rdx,8), %r8')
    e('movq %r8, %r12'); e('andq %rbx, %r12'); e('movq %r12, (%rdi,%rdx,8)'); e('shrq $52, %r8')
    e('incq %rdx'); e('decl %eax'); e('jnz .Lkl_nyx1')
    # A0..A3 from the stored carries: xa / xb reversed, the carry vectors at offsets 0, m, 2m as masks
    e(f'movq {KN_X}(%rsp), %rsi'); e('leaq -56(%rsi), %rsi'); e('movq %rsi, %r11'); e('subq %r9, %r11')
    e(f'movl {KN_OCC}(%rsp), %eax'); e('leaq (%rsp,%rax), %r10'); e('leaq (%r10,%r9), %r12'); e('leaq (%r12,%r9), %r13')
    e('vmovdqu64 .Lkl_rev(%rip), %zmm22'); e('vpternlogq $0xff, %zmm29, %zmm29, %zmm29')
    for r_ in (23, 24, 25, 26): e(f'vpxord %zmm{r_}, %zmm{r_}, %zmm{r_}')
    e('movl %ecx, %eax'); e('shrl $3, %eax'); e('xorl %edx, %edx'); e('xorl %ebx, %ebx')
    lab('.Lkl_nyx2')
    e('vpermq (%rsi,%rbx), %zmm22, %zmm19'); e('vpermq (%r11,%rbx), %zmm22, %zmm20')
    e('vptestmq (%r10,%rdx), %zmm29, %k5'); e('vptestmq (%r12,%rdx), %zmm29, %k6'); e('vptestmq (%r13,%rdx), %zmm29, %k7')
    e('vpaddq %zmm19, %zmm23, %zmm23{%k5}'); e('vpaddq %zmm19, %zmm24, %zmm24{%k6}')
    e('vpaddq %zmm20, %zmm25, %zmm25{%k6}'); e('vpaddq %zmm20, %zmm26, %zmm26{%k7}')
    e('addq $64, %rdx'); e('subq $64, %rbx'); e('decl %eax'); e('jnz .Lkl_nyx2')
    e('jmp .Lkl_nyb')
    # t: memory ascending p = 0 .. m - 1: v = xb_p - xa_p - borrow; t_p = v mod beta; Bb[p] = -(v < 0)
    lab('.Lkl_ntx')
    e(f'movq {KN_X}(%rsp), %rsi'); e('subq %r9, %rsi'); e('addq $8, %rsi'); e('movq %rsi, %r11'); e('subq %r9, %r11')
    e(f'movl {KN_OTT}(%rsp), %eax'); e('leaq (%rsp,%rax), %rdi'); e(f'movl {KN_OBB}(%rsp), %eax'); e('leaq (%rsp,%rax), %r10')
    e('movl %ecx, %eax'); e('xorl %edx, %edx'); e('xorl %r8d, %r8d'); e('movabsq $0xfffffffffffff, %rbx')
    lab('.Lkl_ntx1')
    e('movq (%r11,%rdx,8), %r12'); e('subq (%rsi,%rdx,8), %r12'); e('addq %r8, %r12')            # r8 = -borrow (0 / -1)
    e('movq %r12, %r8'); e('sarq $63, %r8'); e('andq %rbx, %r12')
    e('movq %r12, (%rdi,%rdx,8)'); e('movq %r8, (%r10,%rdx,8)')
    e('incq %rdx'); e('decl %eax'); e('jnz .Lkl_ntx1')
    # b_0 = Bb[m - 1]; B0 / B1 from the stored borrows (reversed, ANDed with y)
    e('movq -8(%r10,%rcx,8), %rax'); e(f'movq %rax, {KN_B0}(%rsp)')
    e(f'movq {KN_Y}(%rsp), %r14'); e('leaq (%r14,%r9,2), %r14'); e('leaq -72(%r14), %r14'); e('leaq (%r14,%r9), %r15')
    e('vpxord %zmm28, %zmm28, %zmm28'); e('vpxord %zmm29, %zmm29, %zmm29')
    e('movl %ecx, %eax'); e('shrl $3, %eax'); e('xorl %edx, %edx'); e('xorl %ebx, %ebx')
    lab('.Lkl_ntx2')
    e('vpermq (%r10,%rdx), %zmm22, %zmm19')
    e('vpandq (%r14,%rbx), %zmm19, %zmm20'); e('vpaddq %zmm20, %zmm28, %zmm28')
    e('vpandq (%r15,%rbx), %zmm19, %zmm20'); e('vpaddq %zmm20, %zmm29, %zmm29')
    e('addq $64, %rdx'); e('subq $64, %rbx'); e('decl %eax'); e('jnz .Lkl_ntx2')
    e('jmp .Lkl_ntb')
    # W[0 .. LW) -> F, serial signed carry (inside .Lkl_norm: its return address at rsp)
    sv = cu_disp(XFWC + KLF)
    lab('.Lkl_bnx')
    e(f'leaq {KB_W + 8}(%rsp), %rsi'); e(f'leaq {KLF + 8}(%rsp), %rdi'); e(f'movl {KB_LW + 8}(%rsp), %ecx')
    e('xorl %edx, %edx'); e('xorl %r8d, %r8d'); e('movabsq $0xfffffffffffff, %rbx')
    lab('.Lkl_bnx1')
    e('movq (%rsi,%rdx,8), %rax'); e('addq %r8, %rax'); e('movq %rax, %r8'); e('sarq $52, %r8'); e('andq %rbx, %rax')
    e('movq %rax, (%rdi,%rdx,8)'); e('incq %rdx'); e('decl %ecx'); e('jnz .Lkl_bnx1')
    e('jmp .Lkl_bnb')
    globals().update(sv)



def klro():
    e(f'.section {KL_RO},"a",@progbits')
    e('.p2align 6')
    lab('.Lkl_rev')
    for v in range(7, -1, -1): e(f'.quad {v}')
    lab('.Lkl_nm52'); e('.quad 0xfff0000000000000')
    e('.p2align 2')
    lab('.Lkl_tab')                                    # ZZ{t}, ACC{t} entries
    for t in range(NVMAX): e(f'.long .Lzz{t} - .Lkl_tab')
    for t in range(NVMAX): e(f'.long .Lkla{t} - .Lkl_tab')
    for t in range(NVMAX): e(f'.long .Lkld{t} - .Lkl_tab')          # KSQ-RSQRT: ACC2{t} (dual destination)
    if a.kl_export:                                    # test build: the plan table writable and global (tools/planbench.cpp)
        e('.data'); e(f'.globl {a.name}_kl_plan'); lab(f'{a.name}_kl_plan')
    lab('.Lkl_plan')
    tab = kl_plan()
    for i in range(0, KL_HN, 24): e('.byte ' + ', '.join(str(x) for x in tab[i:i + 24]))
    if a.kl_cplan:
        if a.kl_export: e('.data'); e(f'.globl {a.name}_kl_cplan'); lab(f'{a.name}_kl_cplan')   # test build: writable
        else: e(f'.section {KL_RO},"a",@progbits')
        lab('.Lkl_cplan')
        tab = kl_plan(a.kl_cplan)
        for i in range(0, KL_HN, 24): e('.byte ' + ', '.join(str(x) for x in tab[i:i + 24]))
