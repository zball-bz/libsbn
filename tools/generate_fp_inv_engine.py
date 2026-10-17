#!/usr/bin/env python3
"""Generate src/fp/x86_64/inv_engine.S: the IFMA level engine of the fp reciprocal (sbn3f_inv_chain, called by
sbn3f_inv_qv in src/fp/x86_64/inv_qv.cpp). usage: generate_fp_inv_engine.py [OUT.S] [options]; without options it
writes the shipped engine (PROD below: S4X of experiments/results/fpk_2026-10-12, frozen/inv_s4x_v1).

One shared register level body for the Newton levels of the reciprocal (research origin: newton_inv/bitprec/claude_impl,
designs compact/des_shared, s4/s4 and s4x, 2026-10-14; the option notes below are that history).

Generates an AT&T .S file with one function LEVEL(X, D, G, n, h, nn, mid) that does exactly what v1.1's level()
(f4_sched/qv/inv_c_qv_q.cpp) does, for every window width NV = ceil((nn - h + 3) / 8) = 1..20, with the same partial
products in every lane sum (so the same canonical digits, the same G, C and update):

  MP          rows q = 0..h of X_h from the top (q = 0 is the leading digit X[n] = 1: v1.1's init row D'[c - h]), each row
              over the window's NV vectors (columns nn + 2 - 8 NV .. nn + 1); padded to a multiple of U rows by rows
              of X below X_h[0] (zero in the production array: X is zero below n - h until a later level)
  canon       fast round (lane >> 52 shifted up one lane), the exact pass from the fast-round digits only when a lane
              is left above M52 (cold, through memory)
  G           G = canonical MP xor M52, lanes below e zeroed (k1 / k2), stored at G[0 .. 8 NV)
  correction  rows t = 8 NV - 1 .. 0 (x = G lane t: zero for t < e, so the rows v1.1 skips add nothing), row t over the
              vectors 0 .. floor(t / 8): a staircase of segments s = NV - 1 .. 0 (8 rows each) on vectors 0 .. s
  update      X[n - h + 1 - 8 NV + 8 v + lane] += C lane (lanes >= e), -2 at lane e when mid; fix() when a digit is left
              out of range

One shared body. The row code exists once per segment width s + 1 (s = 0..9): segment s is a loop of U rows over vectors
0..s, and segment s falls through into segment s - 1 (the correction's staircase); the MP enters segment NV - 1 and
leaves after it. The straight-line sections (zeroing, reduction + canon, G, update) are written once for 10 vectors in
descending order and entered at vector NV - 1 (each ends in ret). A per-NV stub (NV <= 10) is a list of direct calls into
these entries; NV 11..20 (the last level of k >= 140) runs the same sections in two column blocks (vectors 0..9 and
10..NV-1) from one stub with indirect calls through a 10-row table.

Accumulators stay in zmm registers: vector v has GM[v] row groups (row r of an iteration adds into group r mod GM[v]),
each a lo / hi pair; the groups of a vector are summed once at the staircase exit.

usage: gen_engine.py OUT.S --name SYM [--U 4] [--gmap 4,2,2] [--bench] [--align N] [--force-exact] [--fixsym SYM]

BUILD-S4 (s4/s4, 2026-10-14): production copy of compact/des_shared/tools/gen_engine.py. Without the three options below
it writes byte for byte what the original writes (checked by s4/s4/tools/build_s4.sh, phase gen). The options change no
instruction of .text:
  --hidden          .hidden on the entry and on the fix() symbol it calls (internal to the library)
  --prod-header     the production header comment (ABI, precondition, data areas) instead of the one-line one
  --debug-hook SYM  (--chain only; debug builds) call SYM(X, n, lv, l, h) before every level, at the 64-byte aligned
                    frame, after vzeroupper; SYM checks the precondition X[i] = 0 for i < n - h. It adds code to .text, so
                    it is never used for the production object

S4X (s4x, 2026-10-14): copy of s4/s4/tools/gen_engine_s4.py; without the options below it writes byte for byte what that
file writes (checked by s4x/tools/build_s4x.sh, phase gen). They change only the exact carry pass of the NV 1..10 stubs
(cold); the stubs' trampolines keep their bytes (xorl r8d; call; jmp), so .text is byte-identical to S4's; NV 11..20 (the
big stub) keep .Lexact / .Lcanon_mem.
  --xs     (production) v1.1's exact pass in the window registers: canon() of f4_sched/qv/inv_c_qv_q.cpp (B's carry
           masks) per vector, ascending from vector 0: second round W = (V & M52) + (V >> 52 shifted up one lane) (every
           lane <= M52 + 2^12 - 1, so a lane above M52 carries exactly 1), g = W > M52 and p = W == M52, the 8-lane chain
           p + ((g << 1) | sig) in eax (sig = the carry into the next vector), the carry lanes W - M52, then W & M52. The
           same arithmetic as .Lcanon_mem without the frame: 10 spills, the loop and 10 reloads go. One entry .Lxs (the
           trampolines call it); after vector v it returns when NV = v + 1 (cmp / je on r14 = NV). Uses rax, rcx, rdi, zmm1,
           zmm2, zmm16 (M52), k5..k7; leaves the vector registers above NV - 1 and everything else unchanged. With
           --test-exports also the global label NAME_xs (no code change).
  Exploration options, measured and not used (s4x/meas/xbench: per-pass latency / throughput against --xs):
  --xreg   one 80-bit add for the whole carry chain: per-vector g / p masks to GPRs (kmovb, shl 8 / or, descending entry
           .Lxe{NV}), chain = P + (G << 1) by add / adc, carries = P xor chain back to k (kmovq, kshiftrq). Slower than
           --xs: two k <-> GPR crossings (about 9 cycles on this core) and two GPR ops per vector
  --xpair  --xreg with the masks of vector pairs joined by kunpckbw
  --xk     --xreg with masks and chain in k registers (kshiftlq / korq, kaddq; k1 / k2 for vectors 8, 9, restored from
           the frame); best latency at NV 1, but the k-ops issue one per cycle (throughput 2..3x --xs)
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--name', required=True)
ap.add_argument('--U', type=int, default=4)
ap.add_argument('--gmap', default='4,2,2')           # row groups of vectors 0, 1, 2, ...; the rest 1
ap.add_argument('--bench', action='store_true')       # no fix() call (as QV_BENCH)
ap.add_argument('--align', type=int, default=0)       # .p2align of each segment loop head (0: none)
ap.add_argument('--force-exact', action='store_true') # test build: the exact canon pass after every canon
ap.add_argument('--fixsym', default='qvs_fix')
ap.add_argument('--force-fix', action='store_true')   # test build: fix() after every update (the call path; fix of canonical digits changes nothing)
ap.add_argument('--tabstub', action='store_true')     # NV SMALL+1..10: one shared stub, its section entries from a 10-row table (indirect calls)
ap.add_argument('--chain', action='store_true')
ap.add_argument('--inline', action='store_true')      # diagnostic: the straight-line sections inline in every stub (no calls)
ap.add_argument('--inline-upto', type=int, default=0)  # the sections inline in the stubs of NV <= this (calls above)
ap.add_argument('--small', type=int, default=0)        # NV <= SMALL (0..2): the correction from G in registers (lane permutes), rows from e
ap.add_argument('--small-mp', action='store_true')    # small stubs: their own MP loop, 4 rows into 4 row groups per vector (spare registers)
ap.add_argument('--small-cg4', action='store_true')   # small stubs: the correction's rows into 4 row groups per vector (index per row from memory)
ap.add_argument('--duff', action='store_true')        # the MP enters its segment at row r0 = -(rows) mod U (no padding rows)
ap.add_argument('--mpinit', action='store_true')      # MP: group 0's lo accumulators start at the leading digit's row D'[c - h] (v1.1's init), rows of X_h[h-1..0] only
ap.add_argument('--test-exports', action='store_true')  # global symbol for the exact pass (unit test)
ap.add_argument('--maplabels', action='store_true')    # labels as local symbols Lq_* (footprint maps), not .L*
ap.add_argument('--hidden', action='store_true')       # BUILD-S4: .hidden entry and fix symbol
ap.add_argument('--prod-header', action='store_true')  # BUILD-S4: production header comment
ap.add_argument('--debug-hook', default='')           # BUILD-S4: per-level precondition hook (debug builds only)
ap.add_argument('--xreg', action='store_true')        # S4X exploration: the NV 1..10 exact pass as one 80-bit add (.Lxe{NV})
ap.add_argument('--xpair', action='store_true')       # S4X exploration: --xreg with the masks of vector pairs joined by kunpckbw (fewer GPR ops)
ap.add_argument('--xk', action='store_true')          # S4X exploration: --xreg with the masks and the carry chain in k registers (kaddq)
ap.add_argument('--xs', action='store_true')          # S4X production: the exact pass as v1.1's serial per-vector chain in registers (.Lxs, one entry, exit after vector NV - 1)
ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
PROD = ['--name', 'sbn3f_inv_chain', '--fixsym', 'sbn3f_inv_fix', '--chain', '--U', '4', '--gmap', '4,2,2', '--duff',
        '--small', '2', '--mpinit', '--hidden', '--prod-header', '--xs']   # the shipped engine
argv = __import__('sys').argv[1:]
if not any(x.startswith('-') for x in argv):
    argv = (argv or [str(ROOT / 'src/fp/x86_64/inv_engine.S')]) + PROD
a = ap.parse_args(argv)
assert not (a.mpinit and a.small_mp), '--mpinit with the small stubs own MP loop: not built'
assert not (a.xreg and a.tabstub), '--xreg with --tabstub: not built'
if a.xpair or a.xk or a.xs: a.xreg = True
assert a.xpair + a.xk + a.xs <= 1

NVMAX = 10
U = a.U
assert U in (1, 2, 4, 8)
LOGU = {1: 0, 2: 1, 4: 2, 8: 3}[U]
gm = [int(x) for x in a.gmap.split(',')] if a.gmap else []
GM = [gm[v] if v < len(gm) else 1 for v in range(NVMAX)]
for g in GM: assert g >= 1 and U % g == 0, 'groups must divide U'

# registers: lo groups, then hi groups; zmm30 = X (row broadcast) / PHI, zmm31 = T (row load) / CPHI
LO, HI = {}, {}
r = 0
for v in range(NVMAX):
    for g in range(GM[v]): LO[v, g] = r; r += 1
for v in range(NVMAX):
    for g in range(GM[v]): HI[v, g] = r; r += 1
assert r <= 30, f'{r} accumulator registers'
XR, TR = 30, 31
PHI, CPHI = 30, 31
VR = {v: LO[v, 0] for v in range(NVMAX)}
HR = {v: HI[v, 0] for v in range(NVMAX)}
free = [i for i in range(30) if i not in VR.values() and i not in HR.values()]
assert len(free) >= 5, free
T1, T2, T3, BADC, BADU = free[:5]
z = lambda i: f'%zmm{i}'

FRAME = 2048
F_OFF = 0             # F[0 .. 20) vectors: the exact pass / blocked levels (1280 B)
H9_OFF = 1280         # block 0's summed hi of vector 9 (block 1's PHI)
C9_OFF = 1344         # block 0's vector 9 >> 52 before canon (block 1's CPHI)
N_OFF, NN_OFF, PY_OFF, IM_OFF = 1408, 1412, 1416, 1424
K_OFF = 1432          # k1 (16 bits), k3 (16 bits) as words
TGT_OFF = 1440        # big stub: 5 call targets (8 B each)
CH_D, CH_G, CH_LV, CH_L, CH_H = 1488, 1496, 1504, 1512, 1516   # chain entry: D, G, lv, l, h
H_OFF = 1520          # h (small stubs' own MP loop)
I0_OFF = 1524         # i0 (table stub)

L = []
def e(s=''): L.append(('\t' + s) if s and not s.endswith(':') and not s.startswith('.') else s)
def lab(s): L.append(s + ':')

def ifma_row(rr, s):
    e(f'vpbroadcastq {-8 * rr}(%r10), {z(XR)}')
    for v in range(s, -1, -1):
        e(f'vmovdqu64 {8 * rr + 64 * v}(%r11), {z(TR)}')
        g = rr % GM[v]
        e(f'vpmadd52luq {z(TR)}, {z(XR)}, {z(LO[v, g])}')
        e(f'vpmadd52huq {z(TR)}, {z(XR)}, {z(HI[v, g])}')

def red(v):
    """V_v = L_v + (H_v shifted up one lane, lane 0 from H_{v-1} lane 7)"""
    prev = HR[v - 1] if v >= 1 else PHI
    e(f'valignq $7, {z(prev)}, {z(HR[v])}, {z(T3)}')
    e(f'vpaddq {z(T3)}, {z(VR[v])}, {z(VR[v])}')

def groupsum():
    for v in range(NVMAX):
        for reg in (LO, HI):
            gs = [reg[v, g] for g in range(GM[v])]
            d = 1
            while d < len(gs):
                for i in range(0, len(gs) - d, 2 * d):
                    e(f'vpaddq {z(gs[i + d])}, {z(gs[i])}, {z(gs[i])}')
                d *= 2

e('.text')
e('.p2align 6')
e(f'.globl {a.name}')
if a.hidden:
    e(f'.hidden {a.name}')
    if not a.bench: e(f'.hidden {a.fixsym}')
    if a.debug_hook: e(f'.hidden {a.debug_hook}')
e(f'.type {a.name}, @function')
lab(a.name)
e('pushq %rbp')
e('movq %rsp, %rbp')
e('pushq %rbx'); e('pushq %r12'); e('pushq %r13'); e('pushq %r14'); e('pushq %r15')
e('andq $-64, %rsp')
e(f'subq ${FRAME}, %rsp')
if a.chain:
    # LEVEL(X, D, G, n, lv, L): the whole chain lv[L-1] = m0 -> ... -> lv[0] = n (inv_c_qv_q.cpp's loop), one call per inv
    e(f'movq %rsi, {CH_D}(%rsp)'); e(f'movq %rdx, {CH_G}(%rsp)'); e(f'movq %r8, {CH_LV}(%rsp)')
    e('movl %ecx, %ecx'); e(f'movl %ecx, {N_OFF}(%rsp)')
    e('movq %rdi, %r15')
    e('leal -1(%r9), %eax'); e(f'movl %eax, {CH_L}(%rsp)')            # l = L - 1
    e('movl (%r8,%rax,4), %eax'); e(f'movl %eax, {CH_H}(%rsp)')       # h = lv[L - 1] = m0
    lab('.Ldone')
    e(f'movl {CH_L}(%rsp), %eax'); e('testl %eax, %eax'); e('jz .Lepi')
    e('decl %eax'); e(f'movl %eax, {CH_L}(%rsp)')
    if a.debug_hook:                                                    # debug: SYM(X, n, lv, l, h) checks X = 0 below n - h
        e('movq %r15, %rdi'); e(f'movl {N_OFF}(%rsp), %esi'); e(f'movq {CH_LV}(%rsp), %rdx'); e('movl %eax, %ecx')
        e(f'movl {CH_H}(%rsp), %r8d'); e('vzeroupper'); e(f'call {a.debug_hook}'); e(f'movl {CH_L}(%rsp), %eax')
    e(f'movq {CH_LV}(%rsp), %rsi'); e('movl (%rsi,%rax,4), %r9d')      # nn = lv[l]
    e(f'movl {CH_H}(%rsp), %r8d'); e(f'movl %r9d, {CH_H}(%rsp)')      # h, and the next level's h = nn
    e('xorl %ebx, %ebx'); e('testl %eax, %eax'); e('setnz %bl')        # mid = l != 0
    e('movq %r15, %rdi'); e(f'movq {CH_D}(%rsp), %rsi'); e(f'movq {CH_G}(%rsp), %rdx'); e(f'movl {N_OFF}(%rsp), %ecx')
else:
    # LEVEL(X, D, G, n, h, nn, mid): one level (the level bench's form)
    e('movq %rdi, %r15')
    e('movl %ecx, %ecx'); e('movl %r8d, %r8d'); e('movl %r9d, %r9d')
    e(f'movl %ecx, {N_OFF}(%rsp)')
    e('movzbl 16(%rbp), %ebx')
# ---------------------------------------------------------------- per-level setup: rdi X, rsi D, rdx G, ecx n, r8d h, r9d nn, ebx mid
e(f'movl %r9d, {NN_OFF}(%rsp)')
if a.small_mp: e(f'movl %r8d, {H_OFF}(%rsp)')
e('movl %r9d, %eax'); e('subl %r8d, %eax')    # w = nn - h
e('leal 10(%rax), %r14d'); e('shrl $3, %r14d')  # NV = (w + 10) >> 3
e('leal (,%r14,8), %r10d')
e('subl %eax, %r10d'); e('decl %r10d')        # e = 8 NV - w - 1 (2..9)
e('movl $0xffff, %eax'); e('shlxl %r10d, %eax, %eax'); e('kmovw %eax, %k1'); e('kshiftrw $8, %k1, %k2')   # lanes >= e
e('negl %ebx'); e('movl $1, %eax'); e('shlxl %r10d, %eax, %eax'); e('andl %ebx, %eax')
e('kmovw %eax, %k3'); e('kshiftrw $8, %k3, %k4')                       # lane e when mid
e(f'kmovw %k1, {K_OFF}(%rsp)'); e(f'kmovw %k3, {K_OFF + 2}(%rsp)')
# i0: iterations of segment 0 (its rows 7..0, U per iteration; rows below e add zero): max(1, (7 - e) / U + 1)
e('movl $7, %r9d'); e('subl %r10d, %r9d'); e(f'sarl ${LOGU}, %r9d'); e('incl %r9d')
e('movl $1, %eax'); e('cmpl %eax, %r9d'); e('cmovll %eax, %r9d')
if a.tabstub: e(f'movl %r9d, {I0_OFF}(%rsp)')
# pointers (bytes)
MPI = 1 if a.mpinit else 0                    # --mpinit: the MP's rows start one row lower (the init row is a load)
e(f'leaq {"-8" if MPI else ""}(%rdi,%rcx,8), %r10')     # MP x: X + n (- 1 with --mpinit), descending
e('movl %ecx, %eax'); e('subl %r8d, %eax'); e(f'addl ${2 + MPI}, %eax'); e('leal (,%r14,8), %ebx'); e('subl %ebx, %eax')
e('movslq %eax, %rax'); e('leaq (%rsi,%rax,8), %r11')   # MP y: D + n + 2 - 8 NV - h (+ 1 with --mpinit)
e(f'movq %r11, {PY_OFF}(%rsp)')
e('movl %ecx, %eax'); e('subl %ebx, %eax'); e('incl %eax'); e('movslq %eax, %rax')
e('leaq (%rdi,%rax,8), %r12')                 # correction y: X + n - 8 NV + 1
e('movl %ecx, %eax'); e('subl %r8d, %eax'); e('incl %eax'); e('subl %ebx, %eax'); e('movslq %eax, %rax')
e('leaq (%rdi,%rax,8), %r13')                 # update: X + n - h + 1 - 8 NV
e('leaq -8(%rdx,%rbx,8), %rbx')               # correction x: G + 8 NV - 1, descending
e(f'leal {U - MPI}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx')    # MP iterations ceil(rows / U), rows = h + 1 (h with --mpinit)
e(f'movl %ecx, {IM_OFF}(%rsp)')
if a.duff:                                    # r0 = -rows mod U
    if not MPI: e('leal 1(%r8), %r8d')
    e('negl %r8d'); e(f'andl ${U - 1}, %r8d')
e('movl $1, %eax')                            # MP: one segment
e('leaq .Lst_tab(%rip), %rsi'); e('movslq (%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('jmpq *%rdi')

def icall(i): e(f'call *{TGT_OFF + 8 * i}(%rsp)')       # 0 Z, 1 SEG, 2 RCP, 3 GZ, 4 UPD (row: block 1's nb - 1, or NV - 1)
# ---------------------------------------------------------------- per-vector blocks of the straight-line sections
def zib(v):                     # the MP's start of vector v: zero, or (--mpinit) group 0's lo = the init row (r11 - 8: the
    if not a.mpinit: zb(v); return  # leading digit's row D'[c - h])
    e(f'vmovdqu64 {64 * v - 8}(%r11), {z(LO[v, 0])}'); e(f'vpxord {z(HI[v, 0])}, {z(HI[v, 0])}, {z(HI[v, 0])}')
    for g in range(1, GM[v]):
        e(f'vpxord {z(LO[v, g])}, {z(LO[v, g])}, {z(LO[v, g])}'); e(f'vpxord {z(HI[v, g])}, {z(HI[v, g])}, {z(HI[v, g])}')
def zb(v):                      # zero the accumulators of vector v
    for g in range(GM[v]):
        e(f'vpxord {z(LO[v, g])}, {z(LO[v, g])}, {z(LO[v, g])}'); e(f'vpxord {z(HI[v, g])}, {z(HI[v, g])}, {z(HI[v, g])}')
def rcb(v):                     # reduction of vector v - 1, fast canon round of vector v
    if v >= 1:
        red(v - 1)
        e(f'vpsrlq $52, {z(VR[v])}, {z(T1)}')
        e(f'vpsrlq $52, {z(VR[v - 1])}, {z(T2)}')
        e(f'valignq $7, {z(T2)}, {z(T1)}, {z(T1)}')
    else:
        e(f'vpsrlq $52, {z(VR[0])}, {z(T1)}')
        e(f'valignq $7, {z(CPHI)}, {z(T1)}, {z(T1)}')
    e(f'vpandq .Lm52(%rip){{1to8}}, {z(VR[v])}, {z(VR[v])}')
    e(f'vpaddq {z(T1)}, {z(VR[v])}, {z(VR[v])}')
    e(f'vporq {z(VR[v])}, {z(BADC)}, {z(BADC)}')
def gzb(v):                     # G of vector v stored, its accumulators zeroed
    msk = '{%k1}{z}' if v == 0 else '{%k2}{z}' if v == 1 else ''
    e(f'vpxorq .Lm52(%rip){{1to8}}, {z(VR[v])}, {z(T1)}{msk}')
    e(f'vmovdqu64 {z(T1)}, {64 * v}(%rdx)')
    zb(v)
def upb(v):                     # update of vector v
    if v <= 1:
        k, km = ('%k1', '%k3') if v == 0 else ('%k2', '%k4')
        e(f'vmovdqa64 {z(VR[v])}, {z(T1)}{{{k}}}{{z}}')
        e(f'vpsubq .Ltwo(%rip){{1to8}}, {z(T1)}, {z(T1)}{{{km}}}')
        e(f'vpaddq {64 * v}(%r13), {z(T1)}, {z(T1)}')
    else:
        e(f'vpaddq {64 * v}(%r13), {z(VR[v])}, {z(T1)}')
    e(f'vmovdqu64 {z(T1)}, {64 * v}(%r13)')
    e(f'vporq {z(T1)}, {z(BADU)}, {z(BADU)}')
def sec(name, s):               # a straight-line section from vector s down: a call, or inline (--inline: diagnostic)
    if a.inline or s < a.inline_upto:
        fn = {'z': zib, 'rc': rcb, 'gz': gzb, 'upd': upb}[name]
        for v in range(s, -1, -1): fn(v)
    else: e(f'call .L{name}{s}')
def mpcall(s):                  # the MP: segment s, padded to whole iterations, or entered at row r0 (--duff)
    if a.duff:
        e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %rdi'); e('subq %rdi, %r11')
        e('leaq .Ldtab(%rip), %rsi'); e(f'movslq {4 * U * s}(%rsi,%r8,4), %rdi'); e('addq %rsi, %rdi'); e('call *%rdi')
    else: e(f'call .Lseg{s}')

# ---------------------------------------------------------------- stubs NV = 1..10
def tst(reg, lbl):
    if a.force_fix and lbl == '.Lfix': e('jmp .Lfix'); return
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(reg)}, %k6')
    e('kortestb %k6, %k6')
    e(f'jnz {lbl}')

def tst2(reg, lbl):                 # big stub: this block's test or-ed with block 0's (k0)
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(reg)}, %k6')
    e('korb %k0, %k6, %k6')
    e('kortestb %k6, %k6')
    e(f'jnz {lbl}')

def xcall(N):                       # the exact pass of an NV = N stub (N <= 10)
    e('xorl %r8d, %r8d'); e('call .Lxs' if a.xs else f'call .Lxe{N}' if a.xreg else 'call .Lexact')

def fexact(lbl_back, N):
    xcall(N); e(f'jmp {lbl_back}')

for N in range(1, NVMAX + 1):
    if N <= a.small or a.tabstub: continue
    s = N - 1
    e('.p2align 4')
    lab(f'.Lst{N}')
    sec('z', s)
    mpcall(s)
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    red(s)
    sec('rc', s)
    if a.force_exact: xcall(N)
    else: tst(BADC, f'.Lst{N}_xa')
    lab(f'.Lst{N}_a')
    sec('gz', s)
    e('movq %rbx, %r10'); e('movq %r12, %r11'); e(f'movl ${8 // U}, %ecx'); e(f'movl ${N}, %eax')
    e(f'call .Lseg{s}')
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    red(s)
    sec('rc', s)
    if a.force_exact: xcall(N)
    else: tst(BADC, f'.Lst{N}_xb')
    lab(f'.Lst{N}_b')
    e(f'vpxord {z(BADU)}, {z(BADU)}, {z(BADU)}')
    sec('upd', s)
    tst(BADU, '.Lfix')
    e('jmp .Ldone')
    if not a.force_exact:
        lab(f'.Lst{N}_xa'); fexact(f'.Lst{N}_a', N)
        lab(f'.Lst{N}_xb'); fexact(f'.Lst{N}_b', N)

# ---------------------------------------------------------------- small stubs NV <= SMALL (option): G kept in registers,
# the correction's row t multiplier = lane t % 8 of G vector t / 8 by one vpermq (as v1.1's corr_small for NV <= 2), rows
# t = e .. 8 NV - 1 fully unrolled in ascending order, entered at row e through a table (rows 0, 1 never exist: e >= 2)
SMALLREG = {}
if a.small:
    used = {LO[v, g] for v in range(2) for g in range(GM[v])} | {HI[v, g] for v in range(2) for g in range(GM[v])}
    used |= {T1, T2, T3, BADC, BADU, PHI, CPHI}
    spare = [i for i in range(30) if i not in used]
    assert len(spare) >= 10, spare
    SMALLREG = {'G0': spare[0], 'G1': spare[1], 'I': spare[2:10], 'rest': spare[2:]}
def small_stub(N):
    G = [SMALLREG['G0'], SMALLREG['G1']]; I = SMALLREG['I']
    s_ = N - 1
    e('.p2align 4')
    lab(f'.Lst{N}')
    sec('z', s_)
    if a.small_mp:
        # MP: rows of X_h from the top, 4 per iteration, row r into group r of every vector (groups beyond the shared map
        # in spare registers), padded to whole iterations (zero rows below X_h[0]); then the extra groups into the shared
        # ones and the shared group sums
        sp = [SMALLREG['G0'], SMALLREG['G1']] + list(SMALLREG['I'])
        EX = {}
        for v in range(N):
            for g in range(GM[v], 4): EX[v, g] = (sp.pop(0), sp.pop(0))
        for (lo_, hi_) in EX.values(): e(f'vpxord {z(lo_)}, {z(lo_)}, {z(lo_)}'); e(f'vpxord {z(hi_)}, {z(hi_)}, {z(hi_)}')
        e(f'movl {H_OFF}(%rsp), %ecx'); e('addl $4, %ecx'); e('shrl $2, %ecx')     # ceil((h + 1) / 4) iterations
        e('.p2align 4')
        lab(f'.Lsmp{N}')
        for rr in range(4):
            e(f'vpbroadcastq {-8 * rr}(%r10), {z(XR)}')
            for v in range(s_, -1, -1):
                e(f'vmovdqu64 {8 * rr + 64 * v}(%r11), {z(TR)}')
                lo_, hi_ = (LO[v, rr], HI[v, rr]) if rr < GM[v] else EX[v, rr]
                e(f'vpmadd52luq {z(TR)}, {z(XR)}, {z(lo_)}'); e(f'vpmadd52huq {z(TR)}, {z(XR)}, {z(hi_)}')
        e('subq $32, %r10'); e('addq $32, %r11'); e('decl %ecx'); e(f'jnz .Lsmp{N}')
        for (v, g), (lo_, hi_) in EX.items():
            e(f'vpaddq {z(lo_)}, {z(LO[v, g % GM[v]])}, {z(LO[v, g % GM[v]])}'); e(f'vpaddq {z(hi_)}, {z(HI[v, g % GM[v]])}, {z(HI[v, g % GM[v]])}')
        e('call .Lsx')
    else:
        mpcall(s_)
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    red(s_)
    for v in range(s_, -1, -1): rcb(v)
    if a.force_exact: xcall(N)
    else: tst(BADC, f'.Lst{N}_xa')
    lab(f'.Lst{N}_a')
    for v in range(s_, -1, -1):                                   # G in registers, lanes below e zeroed
        msk = '{%k1}{z}' if v == 0 else '{%k2}{z}'
        e(f'vpxorq .Lm52(%rip){{1to8}}, {z(VR[v])}, {z(G[v])}{msk}')
        zb(v)
    if a.small_cg4:
        cs = list(SMALLREG['rest'])
        IT = cs.pop(0)
        EXC = {}
        for v in range(N):
            for g in range(GM[v], 4): EXC[v, g] = (cs.pop(0), cs.pop(0)) if len(cs) >= 2 else None
        assert all(x is not None for x in EXC.values()), 'not enough spare registers for 4 correction groups'
        for (lo_, hi_) in EXC.values(): e(f'vpxord {z(lo_)}, {z(lo_)}, {z(lo_)}'); e(f'vpxord {z(hi_)}, {z(hi_)}, {z(hi_)}')
    else:
        for r in range(2, 8): e(f'vpbroadcastq .Lidx+{8 * r}(%rip), {z(I[r])}')
        if N == 2: e(f'vpbroadcastq .Lidx(%rip), {z(I[0])}'); e(f'vpbroadcastq .Lidx+8(%rip), {z(I[1])}')
    # e = 8 N - w - 1 was in r10d during setup; recompute from the masks: e = number of zero bits of k1 (16-bit)
    e('kmovw %k1, %eax'); e('notl %eax'); e('movzwl %ax, %eax'); e('popcntl %eax, %eax')
    e(f'leaq .Lsm{N}_tab(%rip), %rsi'); e('movslq -8(%rsi,%rax,4), %rdi'); e('addq %rsi, %rdi'); e('jmpq *%rdi')
    for t in range(2, 8 * N):
        lab(f'.Lsm{N}_t{t}')
        if a.small_cg4:
            e(f'vpbroadcastq .Lidx+{8 * (t % 8)}(%rip), {z(IT)}')
            e(f'vpermq {z(G[t // 8])}, {z(IT)}, {z(XR)}')
        else:
            e(f'vpermq {z(G[t // 8])}, {z(I[t % 8])}, {z(XR)}')
        for v in range(t // 8, -1, -1):
            e(f'vmovdqu64 {8 * (8 * N - 1 - t) + 64 * v}(%r12), {z(TR)}')
            if a.small_cg4:
                g = t % 4
                lo_, hi_ = (LO[v, g], HI[v, g]) if g < GM[v] else EXC[v, g]
            else:
                g = t % GM[v]; lo_, hi_ = LO[v, g], HI[v, g]
            e(f'vpmadd52luq {z(TR)}, {z(XR)}, {z(lo_)}')
            e(f'vpmadd52huq {z(TR)}, {z(XR)}, {z(hi_)}')
    lab(f'.Lsm{N}_t{8 * N}')
    if a.small_cg4:
        for (v, g), (lo_, hi_) in EXC.items():
            e(f'vpaddq {z(lo_)}, {z(LO[v, g % GM[v]])}, {z(LO[v, g % GM[v]])}'); e(f'vpaddq {z(hi_)}, {z(HI[v, g % GM[v]])}, {z(HI[v, g % GM[v]])}')
    e('call .Lsx')                                                # group sums
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    red(s_)
    for v in range(s_, -1, -1): rcb(v)
    if a.force_exact: xcall(N)
    else: tst(BADC, f'.Lst{N}_xb')
    lab(f'.Lst{N}_b')
    e(f'vpxord {z(BADU)}, {z(BADU)}, {z(BADU)}')
    for v in range(s_, -1, -1): upb(v)
    tst(BADU, '.Lfix')
    e('jmp .Ldone')
    if not a.force_exact:
        lab(f'.Lst{N}_xa'); fexact(f'.Lst{N}_a', N)
        lab(f'.Lst{N}_xb'); fexact(f'.Lst{N}_b', N)

for N in range(1, a.small + 1): small_stub(N)

# ---------------------------------------------------------------- table stub (option): every NV SMALL+1..10 through one stub,
# its five section entries (Z, SEG, RCP = reduction of the entry vector + RC, GZ, UPD of vector NV - 1) from .Lbig_tab's
# row NV - 1 into frame slots, then indirect calls; the MP enters its segment at row r0 (--duff) through .Ldtab
if a.tabstub:
    e('.p2align 4')
    lab('.Lsttab')
    e('leaq .Lbig_tab(%rip), %rsi'); e('leal -1(%r14), %edi')
    for i in range(5):
        e(f'movslq {40 * i}(%rsi,%rdi,4), %r9'); e('addq %rsi, %r9'); e(f'movq %r9, {TGT_OFF + 8 * i}(%rsp)')
    e(f'movl {I0_OFF}(%rsp), %r9d')                 # i0 back (r9 was the table scratch)
    icall(0)
    if a.duff:
        e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %rsi'); e('subq %rsi, %r11')
        e(f'leal (%r8,%rdi,{U}), %esi') if U in (1, 2, 4, 8) else None
        e('leaq .Ldtab(%rip), %rdi'); e('movslq (%rdi,%rsi,4), %rsi'); e('addq %rdi, %rsi'); e('call *%rsi')
    else:
        icall(1)
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    icall(2)
    if a.force_exact: e('xorl %r8d, %r8d'); e('call .Lexact')
    else: tst(BADC, '.Lsttab_xa')
    lab('.Lsttab_a')
    icall(3)
    e('movq %rbx, %r10'); e('movq %r12, %r11'); e(f'movl ${8 // U}, %ecx'); e('movl %r14d, %eax')
    icall(1)
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    icall(2)
    if a.force_exact: e('xorl %r8d, %r8d'); e('call .Lexact')
    else: tst(BADC, '.Lsttab_xb')
    lab('.Lsttab_b')
    e(f'vpxord {z(BADU)}, {z(BADU)}, {z(BADU)}')
    icall(4)
    tst(BADU, '.Lfix')
    e('jmp .Ldone')
    if not a.force_exact:
        lab('.Lsttab_xa'); fexact('.Lsttab_a', 0)
        lab('.Lsttab_xb'); fexact('.Lsttab_b', 0)

# ---------------------------------------------------------------- big stub NV = 11..20: two column blocks
def store_v(off):
    for v in range(NVMAX): e(f'vmovdqu64 {z(VR[v])}, {off + 64 * v}(%rsp)')
def load_v(off):
    for v in range(NVMAX): e(f'vmovdqu64 {off + 64 * v}(%rsp), {z(VR[v])}')
def save_b0():                     # block 1's PHI (summed hi of vector 9) and CPHI (vector 9 >> 52 before canon)
    e(f'vmovdqu64 {z(HR[9])}, {H9_OFF}(%rsp)')
    e(f'vpsrlq $52, {z(VR[9])}, {z(T1)}'); e(f'vmovdqu64 {z(T1)}, {C9_OFF}(%rsp)')
e('.p2align 4')
lab('.Lstbig')
e('leaq .Lbig_tab(%rip), %rsi'); e('leal -11(%r14), %edi')      # nb - 1
for i in range(5):
    e(f'movslq {40 * i}(%rsi,%rdi,4), %r8'); e('addq %rsi, %r8'); e(f'movq %r8, {TGT_OFF + 8 * i}(%rsp)')
# MP block 0
e('call .Lz9'); e('call .Lseg9')
e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
red(9); save_b0(); e('call .Lrc9'); store_v(F_OFF)
e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k0')         # block 0's lanes left above M52 (the loops clobber BADC)
# MP block 1
e(f'movl {N_OFF}(%rsp), %ecx'); e(f'leaq {"-8" if MPI else ""}(%r15,%rcx,8), %r10'); e(f'movq {PY_OFF}(%rsp), %r11'); e('addq $640, %r11')
e(f'movl {IM_OFF}(%rsp), %ecx'); e('movl $1, %eax')
icall(0); icall(1)
e(f'vmovdqu64 {H9_OFF}(%rsp), {z(PHI)}'); e(f'vmovdqu64 {C9_OFF}(%rsp), {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
icall(2)
if a.force_exact: e('movl $640, %r8d'); e('call .Lexact')
else: tst2(BADC, '.Lbig_xa')
lab('.Lbig_a')
e('kxnorw %k0, %k0, %k1'); e('kxnorw %k0, %k0, %k2')           # block 1: no lane below e
e('addq $640, %rdx'); icall(3); e('subq $640, %rdx')
load_v(F_OFF); e(f'kmovw {K_OFF}(%rsp), %k1'); e('kshiftrw $8, %k1, %k2')
e('call .Lgz9')
# correction block 0: rows 8 NV - 1 .. 0 on vectors 0..9 (segment 9 takes every row t >= 72)
e('movq %rbx, %r10'); e('movq %r12, %r11'); e(f'leal -9(%r14), %ecx'); e(f'shll ${3 - LOGU}, %ecx'); e('movl $10, %eax')
e('call .Lseg9')
e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
red(9); save_b0(); e('call .Lrc9'); store_v(F_OFF)
e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k0')
# correction block 1: rows 8 NV - 1 .. 80 on vectors 10 .. NV - 1 (local staircase, every row real)
e('movq %rbx, %r10'); e('leaq 640(%r12), %r11'); e(f'movl ${8 // U}, %ecx'); e('leal -10(%r14), %eax'); e(f'movl ${8 // U}, %r9d')
if a.mpinit:                    # Z is the MP's init with --mpinit: the correction's accumulators are zeroed here instead
    for v in range(NVMAX - 1, -1, -1): zb(v)
else: icall(0)
icall(1)
e(f'vmovdqu64 {H9_OFF}(%rsp), {z(PHI)}'); e(f'vmovdqu64 {C9_OFF}(%rsp), {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
icall(2)
if a.force_exact: e('movl $640, %r8d'); e('call .Lexact')
else: tst2(BADC, '.Lbig_xb')
lab('.Lbig_b')
e('kxnorw %k0, %k0, %k1'); e('kxnorw %k0, %k0, %k2'); e('kxorw %k3, %k3, %k3'); e('kxorw %k4, %k4, %k4')
e(f'vpxord {z(BADU)}, {z(BADU)}, {z(BADU)}')
e('addq $640, %r13'); icall(4); e('subq $640, %r13')
load_v(F_OFF)
e(f'kmovw {K_OFF}(%rsp), %k1'); e('kshiftrw $8, %k1, %k2'); e(f'kmovw {K_OFF + 2}(%rsp), %k3'); e('kshiftrw $8, %k3, %k4')
e('call .Lupd9')
tst(BADU, '.Lfix')
e('jmp .Ldone')
if not a.force_exact:
    lab('.Lbig_xa'); e('movl $640, %r8d'); e('call .Lexact'); e('jmp .Lbig_a')
    lab('.Lbig_xb'); e('movl $640, %r8d'); e('call .Lexact'); e('jmp .Lbig_b')

# ---------------------------------------------------------------- epilogue, fix
lab('.Lfix')
if not a.bench:
    e('movq %r15, %rdi'); e(f'movl {N_OFF}(%rsp), %edx'); e('movl %edx, %esi'); e(f'subl {NN_OFF}(%rsp), %esi')
    e('vzeroupper')
    e(f'call {a.fixsym}')
if not a.chain: lab('.Ldone')
else: e('jmp .Ldone')
lab('.Lepi')
e('leaq -40(%rbp), %rsp')
e('popq %r15'); e('popq %r14'); e('popq %r13'); e('popq %r12'); e('popq %rbx'); e('popq %rbp')
e('vzeroupper')
e('ret')

# ---------------------------------------------------------------- sections (descending vector entries, each ends in ret)
# Z: zero the accumulators of vectors v .. 0
for v in range(NVMAX - 1, -1, -1): lab(f'.Lz{v}'); zib(v)
e('ret')

# SEG: the staircase. Segment s: U rows per iteration over vectors 0..s; r10 x pointer (down 8 B a row), r11 y pointer
# (up 8 B a row), ecx iterations, eax segments still to run; the next segment runs 8 / U iterations (segment 0: r9d).
# Row r of segment s also has a label (--duff: the MP enters there, rows r0 .. U-1 of its first iteration)
for s_ in range(NVMAX - 1, -1, -1):
    if a.align: e(f'.p2align {a.align}')
    lab(f'.Lseg{s_}')
    for rr in range(U):
        if rr: lab(f'.Lseg{s_}_r{rr}')
        ifma_row(rr, s_)
    e(f'subq ${8 * U}, %r10'); e(f'addq ${8 * U}, %r11')
    e('decl %ecx'); e(f'jnz .Lseg{s_}')
    if s_ >= 1:
        e('decl %eax'); e('jz .Lsx')
        e('movl %r9d, %ecx' if s_ == 1 else f'movl ${8 // U}, %ecx')
lab('.Lsx')
groupsum()
e('ret')

# RCP (big stub, block 1): the reduction of the entry vector, then RC
for v in range(NVMAX - 1, -1, -1):
    lab(f'.Lrcp{v}'); red(v); e(f'jmp .Lrc{v}')

# RC: reduction of vector v - 1 and the fast canon round of vector v (needs V_v and V_{v-1} before canon); the entry
# vector's reduction comes before the call. Lanes left above M52 are or-ed into BADC (bit 52)
for v in range(NVMAX - 1, -1, -1): lab(f'.Lrc{v}'); rcb(v)
e('ret')

# GZ: G = V xor M52 (lanes below e zeroed: k1 for vector 0, k2 for vector 1), stored at rdx + 64 v; then zero the
# accumulators of vector v (the correction's)
for v in range(NVMAX - 1, -1, -1): lab(f'.Lgz{v}'); gzb(v)
e('ret')

# UPD: X[r13 + 64 v] += C (lanes >= e: k1 / k2; -2 at lane e when mid: k3 / k4); bit 52 of any result -> BADU
for v in range(NVMAX - 1, -1, -1): lab(f'.Lupd{v}'); upb(v)
e('ret')
lab('.Lend')

# ---------------------------------------------------------------- cold: the exact canon pass
# .Lexact: r8 = byte offset of the block in F (0 or 640); r14 = NV. The registers V_0..V_9 go to F + r8, B's exact
# carry pass runs over F[0 .. NV) (from the fast-round digits: same value mod beta^(8 NV), so the same canonical digits
# as v1.1's pass from the raw sums), V_0..V_9 come back from F + r8. Uses rax, rcx, rsi, rdi, zmm T1..T3, k5..k7.
e('.section .text.unlikely,"ax",@progbits')
lab('.Lexact')
for v in range(NVMAX): e(f'vmovdqu64 {z(VR[v])}, {8 + F_OFF + 64 * v}(%rsp,%r8)')   # +8: the return address
e(f'leaq {8 + F_OFF}(%rsp), %rdi'); e('movl %r14d, %esi')
e('call .Lcanon_mem')
for v in range(NVMAX): e(f'vmovdqu64 {8 + F_OFF + 64 * v}(%rsp,%r8), {z(VR[v])}')
e('ret')
if a.test_exports:
    e(f'.globl {a.name}_canon_mem')
    lab(f'{a.name}_canon_mem')
lab('.Lcanon_mem')                             # rdi: vectors, esi: count (>= 1); B's canon() per vector:
e(f'vpxord {z(T3)}, {z(T3)}, {z(T3)}')       # carry vector (the previous vector's lane >> 52)
e('xorl %eax, %eax')                           # sig: the carry into lane 0
e('pushq %r10'); e('pushq %r11')
e('movl %esi, %r10d')
lab('.Lcm_loop')
e(f'vmovdqu64 (%rdi), {z(T1)}')
e(f'vpsrlq $52, {z(T1)}, {z(T2)}')
e(f'vpandq .Lm52(%rip){{1to8}}, {z(T1)}, {z(T1)}')
e(f'valignq $7, {z(T3)}, {z(T2)}, {z(T3)}')   # lane 0: previous hi lane 7; lanes 1..7: this hi 0..6
e(f'vpaddq {z(T3)}, {z(T1)}, {z(T1)}')
e(f'vmovdqa64 {z(T2)}, {z(T3)}')
e(f'vpcmpuq $6, .Lm52(%rip){{1to8}}, {z(T1)}, %k5')   # g: lane > M52
e(f'vpcmpeqq .Lm52(%rip){{1to8}}, {z(T1)}, %k6')      # p: lane == M52
e('kmovb %k5, %ecx'); e('kmovb %k6, %r11d')
e('leal (%rax,%rcx,2), %ecx')                  # (g << 1) | sig  (sig in {0, 1}; bit 0 of g << 1 is 0)
e('addl %r11d, %ecx')                          # chain = p + ((g << 1) | sig)
e('movl %ecx, %eax'); e('shrl $8, %eax')       # sig = chain >> 8
e('xorl %r11d, %ecx')                          # p ^ chain: the lanes that take a carry
e('kmovb %ecx, %k7')
e(f'vpsubq .Lm52(%rip){{1to8}}, {z(T1)}, {z(T1)}{{%k7}}')
e(f'vpandq .Lm52(%rip){{1to8}}, {z(T1)}, {z(T1)}')
e(f'vmovdqu64 {z(T1)}, (%rdi)')
e('addq $64, %rdi')
e('decl %r10d')
e('jnz .Lcm_loop')
e('popq %r11'); e('popq %r10')
e('ret')
if a.xreg:
    # S4X: the exact pass of the NV 1..10 stubs in registers (--xs; exploration: --xreg / --xpair / --xk). Free in every
    # caller: the trampolines run right after a fast round, where only the window's vector registers and the stub's
    # pointers (rbx, rdx, r9, r12..r15, r14 = NV) are live; the row-group registers (zmm16 among them) and k5..k7 are not.
    XM = 16
    assert XM not in VR.values() and XM not in (T1, T2), XM
if a.xs:
    # --xs: v1.1's exact pass (canon() of f4_sched/qv/inv_c_qv_q.cpp, B's carry masks) per vector, ascending, in the
    # window registers: second round W = (V & M52) + (V >> 52 shifted up one lane), g / p masks, the 8-lane chain
    # p + ((g << 1) | sig) through eax (sig = the carry into the next vector), carry lanes W - M52, W & M52. One entry;
    # after vector v the section returns when NV = v + 1 (r14 = NV). The hi vectors alternate between T1 and T2.
    e('.p2align 4')
    lab('.Lxs')
    if a.test_exports:
        e(f'.globl {a.name}_xs')
        lab(f'{a.name}_xs')
    e(f'vpbroadcastq .Lm52(%rip), {z(XM)}')
    H = [T1, T2]
    e(f'vpxord {z(H[1])}, {z(H[1])}, {z(H[1])}')                    # no hi below vector 0
    e('xorl %eax, %eax')                                            # sig: no carry into vector 0
    for v in range(NVMAX):
        cur, prv = H[v % 2], H[(v + 1) % 2]
        V = z(VR[v])
        e(f'vpsrlq $52, {V}, {z(cur)}')                             # this vector's lane >> 52
        e(f'vpandq {z(XM)}, {V}, {V}')
        e(f'valignq $7, {z(prv)}, {z(cur)}, {z(prv)}')              # lane 0: the previous vector's lane 7
        e(f'vpaddq {z(prv)}, {V}, {V}')                             # W <= M52 + 2^12 - 1
        e(f'vpcmpuq $6, {z(XM)}, {V}, %k5')                         # g: W > M52
        e(f'vpcmpeqq {z(XM)}, {V}, %k6')                            # p: W == M52
        e('kmovb %k5, %ecx'); e('kmovb %k6, %edi')
        e('leal (%rax,%rcx,2), %ecx')                               # (g << 1) | sig
        e('addl %edi, %ecx')                                        # chain = p + ((g << 1) | sig)
        e('movl %ecx, %eax'); e('shrl $8, %eax')                    # sig = chain >> 8
        e('xorl %edi, %ecx')                                        # p ^ chain: the carry lanes
        e('kmovb %ecx, %k7')
        e(f'vpsubq {z(XM)}, {V}, {V}{{%k7}}')
        e(f'vpandq {z(XM)}, {V}, {V}')
        if v < NVMAX - 1: e(f'cmpl ${v + 1}, %r14d'); e('je .Lxs_ret')
    lab('.Lxs_ret')
    e('ret')
elif a.xreg:
    # exploration (--xreg / --xpair / --xk): .Lxe{N}: M52 to XM, the masks section from vector N - 1 down (it ends in the
    # combine), then the apply section from vector N - 1 down (its ret returns to the stub's trampoline). --xreg: G_lo /
    # P_lo = rax / rsi (vectors 0..7, 8 bits a vector, vector 0 lowest), G_hi / P_hi = r10 / r11 (vectors 8, 9); rcx, rdi
    # the mask bytes; carries to k7 (lo) and k6 (hi), k5 the shifted mask
    for N in range(1, NVMAX + 1):
        lab(f'.Lxe{N}')
        if a.test_exports:
            e(f'.globl {a.name}_xe{N}')
            lab(f'{a.name}_xe{N}')
        e(f'vpbroadcastq .Lm52(%rip), {z(XM)}')
        e(f'call .Lxm{N - 1}')
        if a.xk and N >= 9:                                         # --xk: vectors 8, 9 used k1 / k2: the stub's
            e(f'kmovw {8 + K_OFF}(%rsp), %k1'); e('kshiftrw $8, %k1, %k2')   # lanes >= e masks back from the frame
        e(f'jmp .Lxa{N - 1}')
    for v in range(NVMAX - 1, -1, -1):
        lab(f'.Lxm{v}')
        e(f'vpsrlq $52, {z(VR[v])}, {z(T1)}')                       # this vector's lane >> 52 (0 .. 2^12 - 1)
        if v >= 1: e(f'vpsrlq $52, {z(VR[v - 1])}, {z(T2)}')        # the vector below (not yet rewritten)
        else: e(f'vpxord {z(T2)}, {z(T2)}, {z(T2)}')                # nothing below vector 0
        e(f'valignq $7, {z(T2)}, {z(T1)}, {z(T1)}')                 # lane 0: the lower vector's lane 7
        e(f'vpandq {z(XM)}, {z(VR[v])}, {z(VR[v])}')
        e(f'vpaddq {z(T1)}, {z(VR[v])}, {z(VR[v])}')                # W <= M52 + 2^12 - 1
        if a.xk:                                                    # --xk: masks accumulated in k registers
            e(f'vpcmpuq $6, {z(XM)}, {z(VR[v])}, %k5')              # g: W > M52 (carries 1 out)
            e(f'vpcmpeqq {z(XM)}, {z(VR[v])}, %k6')                 # p: W == M52 (passes a carry on)
            kg, kp = ('%k1', '%k2') if v >= 8 else ('%k7', '%k0')
            e(f'kshiftlq $8, {kg}, {kg}'); e(f'korq %k5, {kg}, {kg}')
            e(f'kshiftlq $8, {kp}, {kp}'); e(f'korq %k6, {kp}, {kp}')
            continue
        G_, P_ = ('%r10', '%r11') if v >= 8 else ('%rax', '%rsi')
        if not a.xpair:
            e(f'vpcmpuq $6, {z(XM)}, {z(VR[v])}, %k5')              # g: W > M52 (carries 1 out)
            e(f'vpcmpeqq {z(XM)}, {z(VR[v])}, %k6')                 # p: W == M52 (passes a carry on)
            e('kmovb %k5, %ecx'); e('kmovb %k6, %edi')
            e(f'shlq $8, {G_}'); e(f'orq %rcx, {G_}')
            e(f'shlq $8, {P_}'); e(f'orq %rdi, {P_}')
        elif v % 2:                                                 # --xpair: odd vector of the pair (v, v - 1): k5 / k6
            e(f'vpcmpuq $6, {z(XM)}, {z(VR[v])}, %k5')
            e(f'vpcmpeqq {z(XM)}, {z(VR[v])}, %k6')
        else:                                                       # even vector: k7 / k0, then the pair's 16 bits (entered
            e(f'vpcmpuq $6, {z(XM)}, {z(VR[v])}, %k7')              # here, k5 / k6 are stale: bits above 8 NV)
            e(f'vpcmpeqq {z(XM)}, {z(VR[v])}, %k0')
            e('kunpckbw %k7, %k5, %k5'); e('kunpckbw %k0, %k6, %k6')
            e('kmovw %k5, %ecx'); e('kmovw %k6, %edi')
            e(f'shlq $16, {G_}'); e(f'orq %rcx, {G_}')
            e(f'shlq $16, {P_}'); e(f'orq %rdi, {P_}')
    if a.xk:
        # combine in k registers: lo (vectors 0..7) chain = P + (G << 1), carries C = P xor chain (k5); the carry out of
        # lane 63 is g63 | (p63 & c63); hi (vectors 8, 9) chain = P_hi + ((G_hi << 1) | that carry), carries to k6
        e('kshiftlq $1, %k7, %k5'); e('kaddq %k0, %k5, %k5'); e('kxorq %k0, %k5, %k5')
        e('kandq %k0, %k5, %k6'); e('korq %k7, %k6, %k6'); e('kshiftrq $63, %k6, %k6')
        e('kshiftlq $1, %k1, %k7'); e('korq %k6, %k7, %k7'); e('kaddq %k2, %k7, %k7'); e('kxorq %k2, %k7, %k6')
        e('ret')
        for v in range(NVMAX - 1, -1, -1):
            lab(f'.Lxa{v}')
            src, sh = ('%k6', 8 * (v - 8)) if v >= 8 else ('%k5', 8 * v)
            if sh: e(f'kshiftrq ${sh}, {src}, %k7')
            k_ = '%k7' if sh else src
            e(f'vpsubq {z(XM)}, {z(VR[v])}, {z(VR[v])}{{{k_}}}')    # carry in: W + 1 - 2^52 (mod 2^52 after the and)
            e(f'vpandq {z(XM)}, {z(VR[v])}, {z(VR[v])}')
        e('ret')
    else:
        # combine: chain = P + (G << 1) over 128 bits (only the low 8 NV are used), carries = P xor chain
        e('addq %rax, %rax'); e('adcq %r10, %r10')
        e('addq %rsi, %rax'); e('adcq %r11, %r10')
        e('xorq %rsi, %rax'); e('xorq %r11, %r10')
        e('kmovq %rax, %k7'); e('kmovq %r10, %k6')
        e('ret')
        for v in range(NVMAX - 1, -1, -1):
            lab(f'.Lxa{v}')
            src, sh = ('%k6', 8 * (v - 8)) if v >= 8 else ('%k7', 8 * v)
            if sh: e(f'kshiftrq ${sh}, {src}, %k5')
            k_ = '%k5' if sh else src
            e(f'vpsubq {z(XM)}, {z(VR[v])}, {z(VR[v])}{{{k_}}}')    # carry in: W + 1 - 2^52 (mod 2^52 after the and)
            e(f'vpandq {z(XM)}, {z(VR[v])}, {z(VR[v])}')
        e('ret')
e('.text')

# ---------------------------------------------------------------- tables, constants
e('.section .rodata')
e('.p2align 2')
lab('.Lst_tab')
e('.long 0')
for N in range(1, NVMAX + 1): e(f'.long .Lst{N} - .Lst_tab' if (N <= a.small or not a.tabstub) else '.long .Lsttab - .Lst_tab')
for N in range(NVMAX + 1, 21): e('.long .Lstbig - .Lst_tab')
if a.duff:
    lab('.Ldtab')
    for s_ in range(NVMAX):
        for rr in range(U): e(f'.long .Lseg{s_}' + (f'_r{rr}' if rr else '') + ' - .Ldtab')
for N in range(1, a.small + 1):
    lab(f'.Lsm{N}_tab')
    for ee in range(2, 10): e(f'.long .Lsm{N}_t{min(ee, 8 * N)} - .Lsm{N}_tab')
lab('.Lbig_tab')
for sec in ('z', 'seg', 'rcp', 'gz', 'upd'):
    for nb in range(1, 11): e(f'.long .L{sec}{nb - 1} - .Lbig_tab')
e('.section .rodata.cst8,"aM",@progbits,8')
e('.p2align 3')
lab('.Lm52'); e('.quad 0xfffffffffffff')
lab('.Lb52'); e('.quad 0x10000000000000')
lab('.Ltwo'); e('.quad 2')
if a.small:                     # an array (addressed as .Lidx + 8 r): not in the mergeable .rodata.cst8, whose pieces the linker
    e('.section .rodata')       # may deduplicate and move one by one
    e('.p2align 3')
    lab('.Lidx')
    for r in range(8): e(f'.quad {r}')
e('.text')
e(f'.size {a.name}, .Lend - {a.name}')
e('.section .note.GNU-stack,"",@progbits')

src = '\n'.join(L) + '\n'
if a.maplabels:
    import re
    src = re.sub(r'\.L(?=[a-z_])', 'Lq_', src)
hdr = ('/* Generated by gen_engine.py (compact/des_shared); do not edit.\n'
       f' * {a.name}: U {U}, groups {GM}, bench {a.bench}, align {a.align}, force_exact {a.force_exact} */\n')
if a.prod_header:
    assert a.chain and not a.bench
    hdr = (f'/* {a.name}: the S4 IFMA level engine of the fp reciprocal (fpks4_inv_qv). Generated by\n'
           ' * s4/s4/tools/gen_engine_s4.py (BUILD-S4, from compact/des_shared/tools/gen_engine.py); do not edit.\n'
           f' * Options: U {U}, row groups {GM}, duff {a.duff}, small {a.small}, mpinit {a.mpinit}, hidden {a.hidden},\n'
           f' * debug hook {a.debug_hook or "none"}.\n'
           ' *\n'
           f' * void {a.name}(uint64_t *X, const uint64_t *D, uint64_t *G, unsigned n, const unsigned *lv, unsigned L):\n'
           ' * runs every level lv[L-1] = m0 -> lv[L-2] -> .. -> lv[0] = n of one reciprocal on the digit array X (beta = 2^52,\n'
           ' * X_m has its digit 0 at X[n - m] and its leading digit at X[n]), reading D (y in n + 1 digits) and using G as\n'
           ' * scratch. Each level h -> nn is v1.1\'s level() (f4_sched/qv/inv_c_qv_q.cpp) with the same partial products in\n'
           ' * every lane sum: MP of X_h by D over NV = (nn - h + 10) >> 3 vectors, canon, G = complement, correction X_h G_hi,\n'
           ' * X[n - nn ..] += C (-2 at the bottom kept digit below the last level), fix() when a digit leaves [0, 2^52).\n'
           ' *\n'
           ' * Precondition (every level): X[i] = 0 for every i < n - h. The level reads up to 9 digits below X_h[0]\n'
           ' * (the correction\'s rows read X[n - t + 8 v + lane] down to X[n - 8 NV + 1]; the NV 11..20 path pads its MP\n'
           ' * rows to a multiple of 4 with X[n - h - 1 .. n - h - 3]); those products are only zero, as the error proof\n'
           ' * assumes, when the digits are zero. The driver establishes it (s4/s4/src/inv_qv_s4.cpp lines 331..333: the zero\n'
           ' * stores of X[-16 .. 8 ceil((n + 32) / 8) - 16); line 340: the seed digits, from X[n - m0] up; line 347: this\n'
           ' * call; FPKS4_DEBUG builds check it at line 345 and before every level); every level h -> nn then changes\n'
           ' * only X[n - nn .. n] (its update rewrites the lanes below e unchanged; fix() works on X[n - nn .. n]), so the\n'
           ' * next level, h\' = nn, finds zeros below n - nn. Debug builds check it per level (--debug-hook).\n'
           ' *\n'
           ' * Memory (s4/s4/model, every reachable chain of the table; s4/s4 guard-page test): reads D[-7 .. n + 4],\n'
           ' * X[6 .. n + 7] and G[0 .. 8 NV); writes X[-8 .. n] (a level h -> nn changes only X[n - nn .. n]) and\n'
           ' * G[0 .. 8 NV) (NV >= 3); nothing outside these and its own stack frame. Never in place: X, D, G are disjoint.\n'
           ' * ABI (SysV x86-64): saves and restores rbx, rbp, r12..r15; 2048 + 64 B frame, 64-byte aligned, no red zone;\n'
           ' * the internal sections are entered by call/ret inside the frame; the only external call (fix) is made at the\n'
           ' * 64-byte aligned frame after vzeroupper; vzeroupper before the return; zmm0..31 and k0..k7 are clobbered\n'
           ' * (caller-saved). The exact carry pass is in .text.unlikely.\n'
           ' */\n')
if a.xreg:                      # S4X: header names this generator and the register exact pass (only with --xreg)
    if a.prod_header:
        hdr = hdr.replace('the S4 IFMA level engine of the fp reciprocal (fpks4_inv_qv). Generated by\n'
                          ' * s4/s4/tools/gen_engine_s4.py (BUILD-S4, from compact/des_shared/tools/gen_engine.py); do not edit.\n',
                          'the S4X IFMA level engine of the fp reciprocal (fpks4x_inv_qv). Generated by\n'
                          ' * s4x/tools/gen_engine_s4x.py --xreg (S4X, from s4/s4/tools/gen_engine_s4.py); do not edit.\n', 1)
        hdr = hdr.replace(' * (caller-saved). The exact carry pass is in .text.unlikely.\n',
                          ' * (caller-saved). The exact carry passes are in .text.unlikely: NV 1..10 in the window registers (%s),\n'
                          ' * NV 11..20 through the frame (.Lexact).\n' % ('.Lxs, v1.1\'s per-vector chain' if a.xs else '.Lxe{NV}'), 1)
        hdr = hdr.replace('s4x/tools/gen_engine_s4x.py --xreg (S4X,', 's4x/tools/gen_engine_s4x.py %s (S4X,' % ('--xs' if a.xs else '--xk' if a.xk else '--xpair' if a.xpair else '--xreg'), 1)
        assert 'S4X' in hdr and 'NV 11..20 through the frame' in hdr
    else:
        hdr = hdr.replace(' */\n', f', xreg {a.xreg} */\n', 1)
if a.prod_header and a.xs and a.name == 'sbn3f_inv_chain':     # the library's engine: its own header
    hdr = (f'/* Generated by tools/generate_fp_inv_engine.py; do not edit. {a.name}: the IFMA level engine of the fp\n'
           ' * reciprocal (sbn3f_inv_qv, src/fp/x86_64/inv_qv.cpp).\n'
           f' * Options: U {U}, row groups {GM}, duff {a.duff}, small {a.small}, mpinit {a.mpinit}, exact pass in registers (--xs).\n'
           ' *\n'
           f' * void {a.name}(uint64_t *X, const uint64_t *D, uint64_t *G, unsigned n, const unsigned *lv, unsigned L):\n'
           ' * runs every level lv[L-1] = m0 -> lv[L-2] -> .. -> lv[0] = n of one reciprocal on the digit array X (beta = 2^52,\n'
           ' * X_m has its digit 0 at X[n - m] and its leading digit at X[n]), reading D (y in n + 1 digits) and using G as\n'
           ' * scratch. Level h -> nn: MP of X_h by D over NV = (nn - h + 10) >> 3 vectors, canon, G = complement, correction\n'
           ' * X_h G_hi, X[n - nn ..] += C (-2 at the bottom kept digit below the last level), fix() when a digit leaves\n'
           ' * [0, 2^52); the partial products of every lane sum are those of the error proof in inv_qv.cpp.\n'
           ' *\n'
           ' * Preconditions (every level h -> nn), established by sbn3f_inv_qv (its zero stores of X and the seed digits) and\n'
           ' * kept by every level (which changes only X[n - nn .. n]): X[i] = 0 for i < n - h (the level reads up to 9 digits\n'
           ' * below X_h[0]: the correction\'s lowest rows and the NV 11..20 path\'s MP pad rows, whose products the proof\n'
           ' * takes as zero); X[n] = 1 (the MP\'s init row is the leading digit times 1); X[n - h .. n) canonical (IFMA reads\n'
           ' * 52 bits); X = 0 on (n, n + 15] (read by the correction and the encode).\n'
           ' *\n'
           ' * Memory: reads D[-7 .. n + 4], X[-8 .. n + 7] and G[0 .. 8 NV); writes X[-8 .. n] (lanes below the update\n'
           ' * written back unchanged; a level h -> nn changes only X[n - nn .. n]) and G[0 .. 8 NV) (NV >= 3); nothing\n'
           ' * outside these and its own stack frame. Never in place: X, D, G are disjoint.\n'
           ' * ABI (SysV x86-64): saves and restores rbx, rbp, r12..r15; 2048 + 64 B frame, 64-byte aligned, no red zone;\n'
           ' * the internal sections are entered by call/ret inside the frame; the only external call (fix) is made at the\n'
           ' * 64-byte aligned frame after vzeroupper; vzeroupper before the return; zmm0..31 and k0..k7 are clobbered\n'
           ' * (caller-saved). The exact carry passes are in .text.unlikely: NV 1..10 in the window registers (.Lxs),\n'
           ' * NV 11..20 through the frame (.Lexact).\n'
           ' */\n')
open(a.out, 'w').write(hdr + src)
