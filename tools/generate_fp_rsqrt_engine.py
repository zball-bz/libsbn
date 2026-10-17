#!/usr/bin/env python3
"""Generate src/fp/x86_64/rsqrt_engine.S: the IFMA level engine of the fp reciprocal square root (sbn3f_rsqrt_chain_cu, the
chain entry called by sbn3f_rsqrt_qv in src/fp/x86_64/rsqrt_qv.S; contract: src/fp/kernel.hpp).
  usage: generate_fp_rsqrt_engine.py [--check] [OUT.S] [options]; without options it writes the shipped engine (PROD below)
  to OUT (default src/fp/x86_64/rsqrt_engine.S); --check generates in memory and compares with OUT (exit status 1 if they
  differ).

The shipped engine is the research kernel G's (experiments/results/fpk_2026-10-12/rsqrt_s4x/joint/x512g_claude, OVH2-RSQRT,
accepted by the owner 2026-10-16): its engine.S with the library's symbol names, so its machine code is engine.o's. This
file is that snapshot's source/gen/gen_engine_sl.py with the library's defaults (PROD, the output path, the header below,
the helpers in tools/fp_rsqrt: kl_part.py, the Karatsuba band plan kl_plan.txt and correction plan kl_cplan.txt); every
other line, and the research notes below (the rsqrt engine grew out of the reciprocal's engine generator, so the notes start
with that history), are the snapshot's. The options other than PROD are the research history's switches (test builds:
--force-*, --kl-force, --kl-export; variants), not built by the library.

SLIM-RSQRT (joint rsqrt, Claude, 2026-10-16; TEST-ONLY): copy of ksq_rsqrt/gen/gen_engine_kl.py (= x512cs_claude/source/gen) with
one option, --nv-keep LIST (default: every NV, the output is the original's byte for byte): the level stubs / band entries of
the window widths NV not in LIST are not emitted (P-first stubs .Lst{NV}, band entries .Lsf{NV}; the two-block stub .Lstbig /
.Lsfbig when no NV 11..20 is kept, the NV 21..40 block path .Lxsfbig / .Lxbig_m when no NV 21..40 is kept); their .Lst_tab /
.Lsf_tab entries point to .Lnv_dead (ud2, cold). The schedule must reach no other NV (slim_rsqrt/tools/nvclass.py checks the
tables). Every other instruction is the original's; the kept paths run the same code.
KARA-L1 (joint rsqrt, Claude, 2026-10-16; TEST-ONLY): copy of cubic512/gen/gen_engine_c.py with --kl / --kl-plan / --kl-force
(the Karatsuba band of gen/kl_part.py); without --kl the output is gen_engine_c.py's byte for byte (tools/gencheck_kl.sh).
EXT512-RSQRT (joint rsqrt, Claude, 2026-10-15; TEST-ONLY): copy of the j14 kit's / j15c's gen_engine_v3.py (0af3472b..)
with one option, --nvmax2 N (default 20: the output is the original's byte for byte; checked by tools/gencheck.sh). With
--nvmax2 40 (needs --sqfirst --sf-min <= 11 --sq-local; the rsqrt engine) windows of NV 21..40 run in B = ceil(NV / 10) = 3
or 4 column blocks of 10 vectors; NV <= 20 keeps the original code (the two-block path of NV 11..20 included) at the
original addresses (the new code is appended after the sections, before .Lend, so no original instruction moves):
  - band (.Lxsfbig): the B - 1 full blocks in a loop (block b: segment 9 takes 8 NV + h - 73 - 80 b rows from S's top digit
    over D_h + 640 b, then the staircase 8..0; reduction with the previous block's PHI / CPHI from the frame, fast round,
    stored at F + 640 b, lanes above M52 or-ed into k0), then the top block (local segment NV - 1 - 10 (B - 1), h + 7 rows,
    through .Lbig_tab's entries) in the registers, as block 1 of the two-block path;
  - .Lxbig_m: canon of the top block (exact pass over F[0 .. NV) when any block left a lane above M52), G of the top block
    from the registers and of blocks B - 2 .. 0 from F; correction of the full blocks in a loop (block b: rows 8 NV - 1 ..
    80 b + 72 on segment 9, then 8..0; block 0's segment 0 runs i0 iterations), chained canon, stored at F + 640 b; the top
    block's local staircase, canon (exact pass over F[0 .. NV)); update of the top block, then blocks B - 2 .. 0 from F;
  - the same sections and the same partial products in every lane sum as the two-block path;
  - frame: F grows to 40 vectors in place (offset 0); every slot above it moves up by 1280 B (displacements only: they are
    all disp32 already, so no instruction changes size); the block loop's slots follow; FRAME 3328 B;
  - .Lsf_tab runs to NV 40 (NV 21..40: .Lxsfbig) and moves behind every original .rodata item (its old place: 84 zero
    bytes), so no original table moves either; .Lst_tab stays at NV 20 (only P-first levels, NV < sf_min, index it).
Generate src/fp/x86_64/inv_engine.S: the IFMA level engine of the fp reciprocal (sbn3f_inv_chain, called by
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

RSQRT (joint rsqrt, Claude IFMA side, task IFMA-BENCH, 2026-10-14; experiments/results/fpk_2026-10-12/rsqrt_s4x/joint/
claude_ifma/bench): copy of tools/generate_fp_inv_engine.py (SHA-256 f7905bde..). Without --rsqrt / --force-pexact it writes
byte for byte what the original writes (checked: PROD regenerates src/fp/x86_64/inv_engine.S). --rsqrt adds the P-first
rsqrt level of rsqrt_s4x/claude/DESIGN.md section 3 (ifma/IFMA.md sections 1.3, 2.1): before inv's per-level setup, the
P stage forms P' = D_h X_h (D_h = floor(t beta^(n+1) / 2), the halved radicand in D's layout), every product
D_h[i] X[n - q] (rows q = 0..h, X[n] = 1) with P' index j = i - q >= ilo = n - 8 NV - h kept whole, the rest dropped
whole; canonical (fast round per block, the exact pass over Pbuf when a lane is left above M52), stored at
Pbuf[ilo .. ilo + 8 NVP), NVP = ((8 NV + h) >> 3) + 1. It runs on the shared segment body: per column block of 10
vectors (deepest first) Z (row 0 = D_h itself), the staircase through .Ldtab from segment min(9, (rem - 1) >> 3) down to
max(0, NV - 10 b) (rem = 8 NV + h - 80 b; every segment ends on its own rows: no padding row; rem = 0: row 0 only), RC / RCP with the
previous block's PHI / CPHI (the big stub's save_b0), the store section PS. Then inv's per-level setup and stub run
unchanged with D -> Pbuf: the window holds -e/2 (beta^(nn+h+1)/2 = 0 mod beta^(nn+2) for h >= 2), so the update is
x (1 + e/2), with inv's -2 below the last level and fix().
  entries: --chain:  NAME(X, D_h, G, n, lv, L, Pbuf)   every level of one rsqrt
           else:     NAME(X, D_h, G, n, h, nn, mid, Pbuf)  one level (the level bench's form)
  extra preconditions: D_h zero on [-16, 0) and (n, n + 8]; Pbuf[n + 1 .. n + 8] = 0 (the caller, once per call);
  Pbuf writable on [n - 8 NV - h, n + 8). Reads D_h[ilo .. n + 7], X[n - h .. n - 1]; writes Pbuf[ilo .. ilo + 8 NVP).

INTEG (joint rsqrt, Claude IFMA side, 2026-10-15; joint/claude_ifma/r2/integ): the ifma-v3 engine generator = SMALL-NV's
gen_engine_sn.py (36d82368..) + SQUARE-FORM's changes to the same base (gen_engine_sf.py 28400faf.., applied as its diff
against IFMA-BUILD's gen_engine_rs.py c39eefb8..). The two option sets are independent (SMALL-NV: NV <= 3 dispatch at the top
of the P stage; SQUARE-FORM: NV >= --sf-min test after the NV computation); the only shared emission point, the stubs'
join label .Lst{N}_m after their MP, is emitted once (for N <= PFUSE or with --sqfirst); SQUARE-FORM's rodata follows
SMALL-NV's. Byte-for-byte checks (r2/integ/tools/gencheck.sh; receipts r2/integ/receipts/gencheck_1.txt, gencheck_2.txt):
no new option = gen_engine_rs.py's output; --pfuse alone = gen_engine_sn.py's; --sqfirst alone = gen_engine_sf.py's; both:
the INTEG header, and the .S equals neither (checked to assemble, then gated). ifma-v3 production (joint/j09_claude,
j10_claude): --pfuse 3 --sqfirst --sq-groups 8 --sq-win --sf-min 9 (--sf-min 7 was the undelivered j08_claude).

STACK (joint rsqrt, Claude IFMA side, 2026-10-15; joint/claude_ifma/r2/stack): copy of INTEG's gen_engine_v3.py (2d54eaa2..)
with one option, --sq-local N (with --sqfirst --sq-win): the square-first level allocates the window array itself, N windows
of 64 B, instead of every engine frame carrying 192 of them. The engine frame is then 2048 B again (ifma-v1's); the S stage
does subq $64N, %rsp after its last frame access before the windows (the SQ_DH store) and addq $64N, %rsp right after its
vector loop (the last window read), so the windows lie at [rsp, rsp + 64 N) during the S stage only and the base frame at
rsp + 64 N (the three frame accesses in between, SQ_DS / RS_P / SQ_SB, carry + 64 N in their displacements). rsp stays
64-byte aligned (64 N is a multiple of 64; the vmovdqa64 window stores need it). No call happens between the two (the S
stage's exact pass .Lsq_xa runs after the addq on the base frame). Added: 2 instructions (subq / addq, 7 B each) per
square-first level, no branch; the two window base computations lose their displacement (leaq 2048(%rsp) -> movq %rsp,
leaq 2048(%rsp,%r9) -> leaq (%rsp,%r9)). The P-first path and every other level are unchanged apart from addresses. N must
cover the stored windows 4 ((h + 14) >> 2) of every square-first level the table reaches (ckbound.py --win-cap N; the INTEG
table: at most 176). Not combined with --sq-down (its first finalization writes the frame's F area at rsp). No stack probing
is added (the same unprobed-subq class exists in the driver and in the library's sbn3f_inv / sbn3f_rsqrt: a separate
library decision). Without --sq-local this file writes byte for byte what INTEG's gen_engine_v3.py writes (checked by
r2/stack/tools/gencheck_stack.sh: j10_claude's engine.S / engine_fx.S and INTEG's 15 check forms). Production (ifma-v3.1):
ifma-v3's options + --sq-local 176.

SMALL-NV (joint rsqrt, Claude IFMA side, 2026-10-15; joint/claude_ifma/r2/small): copy of IFMA-BUILD's gen_engine_rs.py
(c39eefb8..) with --pfuse P [--pfuse-hmax H1,H2,H3] [--pfuse-gmp G0,G1,G2]: levels with NV <= P and h <= H_NV take a fused
path .Lsl{NV} (see the comment at its emitter): the P stage of ifma-v1 on the shared body, P' left in the registers (no
Pbuf store / reload: the measured 21..31-cycle hand-over of claude/ifma/IFMA.md 2.4), an MP of straight-line valignq rows
entered at row h + 2, then inv's stub from after its MP (labels .Lst{NV}_m, label only). Same partial products in every
lane sum as ifma-v1, so the same canonical digits. Without --pfuse this file writes byte for byte what IFMA-BUILD's
generator writes (checked in r2/small/tools/build.sh, phase gen).

IFMA-BUILD (joint rsqrt, Claude IFMA side, 2026-10-14; joint/claude_ifma/build): copy of bench/gen/gen_engine_rs.py with one
option that changes no instruction: --labelprefix P renames every local label .L<x> to .L<P><x> (the joint convention:
.Lre_ for the engine). The production engine of the joint rsqrt is
  --name fpkrs_pchain --fixsym fpkrs_pfix --chain --rsqrt --hidden --labelprefix re_ --U 4 --gmap 4,2,2 --duff --small 2
  --mpinit --xs
(fpkrs_pfix: the driver's asm fix(), in its .text.unlikely). Without --labelprefix this file writes byte for byte what
bench/gen/gen_engine_rs.py writes (checked by tools/build.sh, phase gen).

SQUARE-FORM (joint rsqrt, Claude IFMA side, round 2, 2026-10-15; joint/claude_ifma/r2/square): copy of IFMA-BUILD's
build/gen/gen_engine_rs.py (c39eefb8..). Without the options below it writes byte for byte what that file writes (checked by
tools/build.sh, phase gen). --sqfirst (with --rsqrt) adds the square-first level for NV >= --sf-min (default 4): instead
of the P stage, an S stage forms S = X_h^2 to depth DS = (8 NV + h) & ~1 by a dedicated symmetric square (one
product-scanning loop per S vector: each cross product once, the group sums doubled, the diagonal by one vpermq; the
rows of a vector run from its first nonzero row a_lo, padded down to whole iterations, to the 3 diagonal rows with lane
masks; canonical by the fast round, the exact pass over Sbuf when a lane is left above M52), stored at Sbuf = Pbuf + ilo
(P' 's range); then inv's per-level setup with D = D_h and the band D_h x S on inv's window: rows r = 0 .. h + 8 NV - 2 of
S (broadcast, descending) over D_h's vectors, entering segment NV - 1 at row r0 (Duff) and falling through the
staircase to segment 0 (8 rows a segment), per column block for NV 11..20; it joins inv's stubs after their MP
(labels .Lst{N}_m, .Lbig_m; labels only, no code change there). The math and the exact model: tools/model_sf.py.
--force-sexact (test build) runs the S stage's exact pass after every S stage. The P-first path gains one compare and
branch per level (cmp $sf_min / jae .Lsq). Entries and preconditions as --rsqrt (Pbuf doubles as Sbuf: written on
[ilo, ilo + 8 NVS), NVS = DS / 8 + 1 <= NVP); the S stage reads X[n - h - 10 .. n + 3] (zero outside X_h).
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
ap.add_argument('--small', type=int, default=0)        # NV <= SMALL (0..2; SMALL-NV: 3): the correction from G in registers (lane permutes), rows from e
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
ap.add_argument('--rsqrt', action='store_true')       # joint rsqrt (IFMA-BENCH): the P-first rsqrt level, see RSQRT below
ap.add_argument('--force-pexact', action='store_true')  # test build (--rsqrt): the P stage's exact carry pass after every P stage
ap.add_argument('--labelprefix', default='')           # IFMA-BUILD: local labels .L<x> -> .L<prefix><x> (no code change)
ap.add_argument('--pfuse', type=int, default=0)        # SMALL-NV: NV 1..PFUSE levels with h <= HMAX: P' in registers, valignq MP (see SMALL-NV)
ap.add_argument('--pfuse-hmax', default='15,23,31')    # SMALL-NV: HMAX per NV 1..3 (NVPMAX = NV + (HMAX >> 3) + 1 <= 7)
ap.add_argument('--pfuse-gmp', default='4,2,2')        # SMALL-NV: MP row groups of vectors 0..2 in the fused path
ap.add_argument('--sqfirst', action='store_true')      # SQUARE-FORM: the square-first level for NV >= --sf-min (see SQUARE-FORM above)
ap.add_argument('--sf-min', type=int, default=4)       # SQUARE-FORM: smallest NV that takes the square-first level
ap.add_argument('--force-sexact', action='store_true')  # SQUARE-FORM test build: the S stage's exact carry pass after every S stage
ap.add_argument('--sq-groups', type=int, default=4)    # SQUARE-FORM: row groups per S vector (4 or 8)
ap.add_argument('--sq-win', action='store_true')      # SQUARE-FORM: the S rows load aligned windows X[s .. s + 7] stored once per level
ap.add_argument('--sq-down', action='store_true')     # SQUARE-FORM: S vectors top-down (shallowest first), each finalized after the next
ap.add_argument('--sq-only', action='store_true')     # SQUARE-FORM probe build (level entry): return after the S stage
ap.add_argument('--sq-local', type=int, default=0)     # STACK: the square-first level allocates N windows itself (subq / addq around the S stage)
ap.add_argument('--nvmax2', type=int, default=20)      # EXT512: the widest window (20: two column blocks, the original; 40: up to 4 blocks)
ap.add_argument('--frame-pad', type=int, default=0)    # EXT512 (with --nvmax2 > 20): extra engine frame bytes (multiple of 64)
ap.add_argument('--fw-dyn', action='store_true')       # EXT512: the NV 21..NV2 path allocates its own F (subq / addq); frame and slots unchanged
ap.add_argument('--cubic', default='')                 # CUBIC512: NAME of an appended chain entry with 1 or 2 cubic top levels (see CUBIC512)
ap.add_argument('--cu-gxg', default='sq')              # CUBIC512: the G x G stage: sq (the S stage on G's top digits, symmetric),
                                                       # stair (rows of G over G's vectors on the block staircase), auto:N (stair for R <= N)
ap.add_argument('--kl', action='store_true')          # KARA-L1: the cubic band through the recursive Karatsuba MP (gen/kl_part.py)
ap.add_argument('--kl-plan', default='')              # KARA-L1: plan file, lines 'h m0 depth' (no line: x512c's plain band)
ap.add_argument('--kl-force', action='store_true')    # KARA-L1 test build: every exact serial pass of the Karatsuba path runs
ap.add_argument('--kl-cplan', default='')             # KARA-L1: correction plan file 'h m0 depth' (empty: x512c's cubic code byte for byte)
ap.add_argument('--kl-export', action='store_true')   # KARA-L1 test build: global entries NAME_kl_node / NAME_kl_leaf (unit test)
ap.add_argument('--xs-mem', action='store_true')       # FIX-RSQRT: .Lxs as one memory pass (spill, .Lcanon_mem, reload) instead of the unrolled register chain
ap.add_argument('--nv-keep', default='')              # SLIM-RSQRT: window widths NV whose stub / band entry is emitted (e.g. 1-7,9); '' = all
ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
FPRS = ROOT / 'tools/fp_rsqrt'
PROD = ['--name', 'sbn3f_rsqrt_chain', '--fixsym', 'sbn3f_rsqrt_fix', '--chain', '--rsqrt', '--hidden', '--labelprefix', 're_',
        '--U', '4', '--gmap', '4,2,2', '--duff', '--small', '2', '--mpinit', '--xs',
        '--pfuse', '3', '--sqfirst', '--sq-groups', '8', '--sq-win', '--sf-min', '9', '--sq-local', '332', '--nvmax2', '40', '--fw-dyn',
        '--cubic', 'sbn3f_rsqrt_chain_cu', '--kl', '--kl-plan', str(FPRS / 'kl_plan.txt'), '--kl-cplan', str(FPRS / 'kl_cplan.txt'),
        '--nv-keep', '1-7,9', '--xs-mem']   # the shipped engine (G: gensrc_fx.sh's PROD, V3, --cubic, KL and ENGX)
argv = __import__('sys').argv[1:]
CHECK = '--check' in argv
argv = [x for x in argv if x != '--check']
LIB = not any(x.startswith('-') for x in argv)
if LIB:
    argv = (argv or [str(ROOT / 'src/fp/x86_64/rsqrt_engine.S')]) + PROD
a = ap.parse_args(argv)
assert not (a.mpinit and a.small_mp), '--mpinit with the small stubs own MP loop: not built'
assert not (a.xreg and a.tabstub), '--xreg with --tabstub: not built'
if a.xpair or a.xk or a.xs: a.xreg = True
assert a.xpair + a.xk + a.xs <= 1

NVMAX = 10
KEEP = set(range(0, 41))      # SLIM-RSQRT --nv-keep
if a.nv_keep:
    KEEP = {0}
    for part in a.nv_keep.split(','):
        lo_, _, hi_ = part.partition('-')
        KEEP |= set(range(int(lo_), int(hi_ or lo_) + 1))
KEEP_BIG = any(N in KEEP for N in range(11, 21))
KEEP_X = any(N in KEEP for N in range(21, 41))
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

assert a.sq_local == 0 or (a.sqfirst and a.sq_win and not a.sq_down and 1 <= a.sq_local <= 1024), '--sq-local: --sqfirst --sq-win, not --sq-down'
FRAME = 2048 + (64 * 192 if a.sqfirst and a.sq_win and not a.sq_local else 0)   # --sq-win: 192 windows of 64 B at WIN_OFF
WIN_OFF = 2048
SQW = 64 * a.sq_local   # STACK (--sq-local N): the S stage's own window area at [rsp, rsp + SQW); the base frame at rsp + SQW
if SQW: WIN_OFF = 0
F_OFF = 0             # F[0 .. 20) vectors: the exact pass / blocked levels (1280 B)
NV2 = a.nvmax2        # EXT512
assert NV2 == 20 or (NV2 in (30, 40) and a.sqfirst and a.sf_min <= 11 and a.sq_local), '--nvmax2: 20, or 30 / 40 with --sqfirst --sf-min <= 11 --sq-local'
SH = 0                # EXT512: F grows in place to 10 vectors per block; the slots above it move up by SH
XFW = 0               # EXT512 --fw-dyn: bytes of the NV 21..NV2 path's own F (at rsp while it runs; frame slots at rsp + XFW)
if NV2 > 20 and a.fw_dyn:
    XFW = 640 * ((NV2 + 9) // 10)
    assert not a.frame_pad
elif NV2 > 20:
    SH = 640 * ((NV2 + 9) // 10) - 1280
    FRAME = 2048 + SH + a.frame_pad
    assert a.frame_pad % 64 == 0 and a.frame_pad >= 0
H9_OFF = 1280 + SH    # block 0's summed hi of vector 9 (block 1's PHI)
C9_OFF = 1344 + SH    # block 0's vector 9 >> 52 before canon (block 1's CPHI)
N_OFF, NN_OFF, PY_OFF, IM_OFF = 1408 + SH, 1412 + SH, 1416 + SH, 1424 + SH
K_OFF = 1432 + SH     # k1 (16 bits), k3 (16 bits) as words
TGT_OFF = 1440 + SH   # big stub: 5 call targets (8 B each)
CH_D, CH_G, CH_LV, CH_L, CH_H = 1488 + SH, 1496 + SH, 1504 + SH, 1512 + SH, 1516 + SH   # chain entry: D, G, lv, l, h
H_OFF = 1520 + SH     # h (small stubs' own MP loop)
I0_OFF = 1524 + SH    # i0 (table stub)
RS_XR, RS_P = 1536 + SH, 1544 + SH                     # --rsqrt: X + n - 1 (the P stage's row 1), Pbuf
RS_H, RS_NN, RS_MID = 1552 + SH, 1556 + SH, 1560 + SH  # --rsqrt: h, nn, mid across the P stage
SQ_DH, SQ_SB, SQ_DS, SQ_I0 = 1568 + SH, 1576 + SH, 1584 + SH, 1588 + SH   # --sqfirst: D_h, Sbuf = Pbuf + ilo, DS, the correction's i0
X_NF, X_TOFF, X_TV, X_ST = 1600 + SH, 1608 + SH, 1616 + SH, 1624 + SH   # EXT512: full blocks B - 1, 640 (B - 1), the top block's local top vector, S's top
assert X_ST + 8 <= FRAME and all(x >= 128 for x in (H9_OFF, SQ_I0))
# CUBIC512 (--cubic): the cubic entry's slots (base frame, above X_ST), its column-block F area (XFWC B at rsp while it runs)
CU_X, CU_LV = 1632 + SH, 1640 + SH
CU_L, CU_H, CU_NN, CU_MID = 1648 + SH, 1652 + SH, 1656 + SH, 1660 + SH
CU_NV, CU_R, CU_E, CU_DSQ = 1664 + SH, 1668 + SH, 1672 + SH, 1676 + SH
CU_SPX, CU_SPY = 1680 + SH, 1688 + SH
CU_SPC0, CU_SPJ, CU_SPNV = 1696 + SH, 1700 + SH, 1704 + SH
NVC = 60                                                # the widest cubic window (n <= 631: NV <= 53)
CU_TEXT, CU_RO = '.lsbn_cu_text', '.lsbn_cu_ro'         # CUBIC512's own sections (orphans: placed after .text / .rodata)
XFWC = 640 * ((NVC + 9) // 10)
if a.cubic:
    assert CU_SPNV + 4 <= FRAME and NV2 == 40 and a.fw_dyn and a.rsqrt and a.chain and a.sqfirst and a.sq_win and a.sq_local \
        and a.mpinit and a.duff and U == 4 and not a.sq_down and a.sf_min <= 9 and (a.cu_gxg in ('sq', 'stair') or a.cu_gxg.startswith('auto:'))
CU_RSTAIR = 0 if a.cu_gxg == 'sq' else 10 ** 6 if a.cu_gxg == 'stair' else int(a.cu_gxg[5:]) if a.cubic else 0

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

def pstage():
    """--rsqrt: the P stage. In: r15 X, rsi D_h, ecx n, r8d h, r9d nn, ebx mid, Pbuf at RS_P, G at CH_G. Out: Pbuf[ilo ..
    ilo + 8 NVP) = P' (canonical), then the registers of inv's per-level setup with D = Pbuf. Registers in the block
    loop: r12d rem = 8 NV + h - 80 b, r13d NV - 10 b, r14 D_h + ilo + 1 + 80 b, rbx Pbuf + ilo + 80 b, edi h, edx top;
    the sections keep them (Z: r11; SEG: r10, r11, ecx, eax, r9d; RC / PS: rdx). k0 = P' lanes left above M52."""
    if a.pfuse:                                                                # SMALL-NV dispatch: NV <= PFUSE, h <= HMAX[NV]
        e('movl %r9d, %eax'); e('subl %r8d, %eax'); e(f'cmpl ${8 * a.pfuse - 3}, %eax'); e('ja .Lp_gen')   # w > 8 PFUSE - 3: NV > PFUSE
        e('addl $10, %eax'); e('shrl $3, %eax')                                # NV 1..PFUSE
        e('leaq .Lsl_hmax(%rip), %rdi'); e('cmpl -4(%rdi,%rax,4), %r8d'); e('ja .Lp_gen')
        if a.pfuse == 1: e('jmp .Lsl1')
        else:
            e('cmpl $2, %eax'); e('jb .Lsl1')
            if a.pfuse == 2: e('jmp .Lsl2')
            else: e('je .Lsl2'); e('jmp .Lsl3')
        lab('.Lp_gen')
    e(f'movl %r8d, {RS_H}(%rsp)'); e(f'movl %r9d, {RS_NN}(%rsp)'); e(f'movl %ebx, {RS_MID}(%rsp)')
    e('leaq -8(%r15,%rcx,8), %rax'); e(f'movq %rax, {RS_XR}(%rsp)')            # X + n - 1: row 1 (rows descend)
    e('movl %r9d, %eax'); e('subl %r8d, %eax'); e('addl $10, %eax'); e('shrl $3, %eax')   # NV = (w + 10) >> 3
    if a.sqfirst: e(f'cmpl ${a.sf_min}, %eax'); e('jae .Lsq')                 # SQUARE-FORM: NV >= sf_min -> square-first
    e('movl %eax, %r13d')                                                      # NV - 10 b
    e('leal (%r8,%rax,8), %r12d')                                              # rem = t0 = 8 NV + h (b = 0)
    e('movl %ecx, %eax'); e('subl %r12d, %eax'); e('cltq')                     # ilo = n - t0 (>= -10)
    e(f'movq {RS_P}(%rsp), %rbx'); e('leaq (%rbx,%rax,8), %rbx')               # Pbuf + ilo
    e('leaq 8(%rsi,%rax,8), %r14')                                             # D_h + ilo + 1 (Z's row 0 at -8)
    e('movl %r8d, %edi')                                                       # h
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}')
    e(f'vmovdqu64 {z(PHI)}, {H9_OFF}(%rsp)'); e(f'vmovdqu64 {z(PHI)}, {C9_OFF}(%rsp)')   # block 0: PHI = CPHI = 0
    e('kxorb %k0, %k0, %k0')
    e('cmpl $79, %r12d'); e('jbe .Lp_last')                                    # one block: rem >> 3 <= 9
    # blocks below the top one: vectors 0..9 all live; first segment 9 (rem - 1 >= 79), its rows 1..min(h, rem - 72)
    e('.p2align 4')
    lab('.Lp_blk')
    e('movq %r14, %r11'); e('call .Lz9')
    e('leal -72(%r12), %r8d'); e('cmpl %edi, %r8d'); e('cmoval %edi, %r8d')   # c_f
    e('movl %r13d, %ecx'); e('xorl %eax, %eax'); e('testl %ecx, %ecx'); e('cmovsl %eax, %ecx')
    e('movl $9, %eax'); e('cmpl %eax, %ecx'); e('cmovgl %eax, %ecx')          # s_last = max(0, min(9, NV - 10 b))
    e('movl $10, %eax'); e('subl %ecx, %eax')                                  # segments 9 .. s_last
    e(f'leal {U - 1}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx')                     # iterations of the first segment
    e('negl %r8d'); e(f'andl ${U - 1}, %r8d')                                  # r0 = -c_f mod U
    e(f'movq {RS_XR}(%rsp), %r10'); e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %r9'); e('subq %r9, %r11')
    e('leaq .Ldtab(%rip), %r9'); e(f'movslq {4 * U * 9}(%r9,%r8,4), %rsi'); e('addq %r9, %rsi')
    e(f'movl ${8 // U}, %r9d'); e('call *%rsi')
    e(f'vmovdqu64 {H9_OFF}(%rsp), {z(PHI)}'); e(f'vmovdqu64 {C9_OFF}(%rsp), {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    red(9)
    e(f'vmovdqu64 {z(HR[9])}, {H9_OFF}(%rsp)')                                 # the next block's PHI and CPHI
    e(f'vpsrlq $52, {z(VR[9])}, {z(T1)}'); e(f'vmovdqu64 {z(T1)}, {C9_OFF}(%rsp)')
    e('call .Lrc9')
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k6'); e('korb %k6, %k0, %k0')
    e('movq %rbx, %rdx'); e('call .Lps9')
    e('addq $640, %r14'); e('addq $640, %rbx'); e('subl $80, %r12d'); e('subl $10, %r13d')
    e('cmpl $79, %r12d'); e('ja .Lp_blk')
    # the top block: vectors 0..top, top = rem >> 3; first segment s_f = (rem - 1) >> 3, rows 1..min(h, rem - 8 s_f); the
    # staircase ends at segment max(0, NV - 10 b) (row h on its last row, or row rem when the block is above the window)
    lab('.Lp_last')
    e('movq %r14, %r11')
    e('movl %r12d, %edx'); e('shrl $3, %edx')                                  # top
    e('leaq .Lbig_tab(%rip), %rsi'); e('movslq (%rsi,%rdx,4), %rax'); e('addq %rsi, %rax'); e('call *%rax')   # Z{top}
    e('testl %r12d, %r12d'); e('jz .Lp_rows_done')                             # rem = 0 (t0 = 80 b): row 0 only, P'[n ..]
    e('leal -1(%r12), %esi'); e('shrl $3, %esi')                               # s_f
    e('leal (,%rsi,8), %ecx'); e('movl %r12d, %r8d'); e('subl %ecx, %r8d'); e('cmpl %edi, %r8d'); e('cmoval %edi, %r8d')   # c_f
    e('xorl %ecx, %ecx'); e('testl %r13d, %r13d'); e('cmovgl %r13d, %ecx')    # s_last = max(0, NV - 10 b) (<= 8 here)
    e('movl %esi, %eax'); e('subl %ecx, %eax'); e('incl %eax')                 # segments s_f .. s_last
    e(f'leal {U - 1}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx')
    e('negl %r8d'); e(f'andl ${U - 1}, %r8d')
    e(f'movq {RS_XR}(%rsp), %r10'); e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %r9'); e('subq %r9, %r11')
    e(f'leal (%r8,%rsi,{U}), %esi')                                            # Duff entry U s_f + r0
    e('leaq .Ldtab(%rip), %r9'); e('movslq (%r9,%rsi,4), %rsi'); e('addq %r9, %rsi')
    e(f'movl ${8 // U}, %r9d'); e('call *%rsi')
    lab('.Lp_rows_done')
    e(f'vmovdqu64 {H9_OFF}(%rsp), {z(PHI)}'); e(f'vmovdqu64 {C9_OFF}(%rsp), {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    e('leaq .Lbig_tab(%rip), %rsi'); e('movslq 80(%rsi,%rdx,4), %rax'); e('addq %rsi, %rax'); e('call *%rax')   # RCP{top}
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k6'); e('korb %k6, %k0, %k0')
    e('leaq .Lps_tab(%rip), %rsi'); e('movslq (%rsi,%rdx,4), %rax'); e('addq %rsi, %rax'); e('movq %rbx, %rdx'); e('call *%rax')   # PS{top}
    if a.force_pexact: e('jmp .Lp_xa')
    else: e('kortestb %k0, %k0'); e('jnz .Lp_xa')
    lab('.Lp_xa_back')
    # inv's per-level setup with D = Pbuf
    e('movq %r15, %rdi'); e(f'movq {RS_P}(%rsp), %rsi'); e(f'movq {CH_G}(%rsp), %rdx'); e(f'movl {N_OFF}(%rsp), %ecx')
    e(f'movl {RS_H}(%rsp), %r8d'); e(f'movl {RS_NN}(%rsp), %r9d'); e(f'movl {RS_MID}(%rsp), %ebx')

if a.rsqrt: assert a.xs and a.mpinit and a.duff and U == 4 and not a.tabstub and not a.small_mp and not a.debug_hook and not a.inline
if a.sqfirst or a.force_sexact: assert a.rsqrt and a.sqfirst and 1 <= a.sf_min <= 20 and a.gmap == '4,2,2'
assert a.sq_groups in (4, 8) and not (a.sq_only and (a.chain or not a.sqfirst))
if a.chain:
    # LEVEL(X, D, G, n, lv, L): the whole chain lv[L-1] = m0 -> ... -> lv[0] = n (inv_c_qv_q.cpp's loop), one call per inv
    if a.rsqrt: e('movq 16(%rbp), %rax'); e(f'movq %rax, {RS_P}(%rsp)')       # --rsqrt: Pbuf (7th argument)
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
    if a.rsqrt: pstage()
else:
    # LEVEL(X, D, G, n, h, nn, mid): one level (the level bench's form)
    e('movq %rdi, %r15')
    e('movl %ecx, %ecx'); e('movl %r8d, %r8d'); e('movl %r9d, %r9d')
    e(f'movl %ecx, {N_OFF}(%rsp)')
    e('movzbl 16(%rbp), %ebx')
    if a.rsqrt:                                                                 # --rsqrt: Pbuf (8th argument)
        e('movq 24(%rbp), %rax'); e(f'movq %rax, {RS_P}(%rsp)'); e(f'movq %rdx, {CH_G}(%rsp)')
        pstage()
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
    if N <= a.small or a.tabstub or N not in KEEP: continue      # SLIM-RSQRT --nv-keep
    s = N - 1
    e('.p2align 4')
    lab(f'.Lst{N}')
    sec('z', s)
    mpcall(s)
    if a.sqfirst or N <= a.pfuse: lab(f'.Lst{N}_m')    # SMALL-NV fused path / SQUARE-FORM band join here (INTEG: once)
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
    NS_ = max(2, a.small)                                         # SMALL-NV: --small 3 keeps vector 2's registers too
    used = {LO[v, g] for v in range(NS_) for g in range(GM[v])} | {HI[v, g] for v in range(NS_) for g in range(GM[v])}
    used |= {T1, T2, T3, BADC, BADU, PHI, CPHI}
    spare = [i for i in range(30) if i not in used]
    assert len(spare) >= 8 + NS_, spare
    if a.small <= 2: SMALLREG = {'G0': spare[0], 'G1': spare[1], 'I': spare[2:10], 'rest': spare[2:]}
    else: SMALLREG = {'G0': spare[0], 'G1': spare[1], 'G2': spare[2], 'I': spare[3:11], 'rest': spare[3:]}
def small_stub(N):
    G = [SMALLREG['G0'], SMALLREG['G1']] + ([SMALLREG['G2']] if 'G2' in SMALLREG else []); I = SMALLREG['I']
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
    if a.sqfirst or N <= a.pfuse: lab(f'.Lst{N}_m')    # SMALL-NV fused path / SQUARE-FORM band join here (INTEG: once)
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    red(s_)
    for v in range(s_, -1, -1): rcb(v)
    if a.force_exact: xcall(N)
    else: tst(BADC, f'.Lst{N}_xa')
    lab(f'.Lst{N}_a')
    for v in range(s_, -1, -1):                                   # G in registers, lanes below e zeroed
        msk = '{%k1}{z}' if v == 0 else '{%k2}{z}' if v == 1 else ''    # (v 2: --small 3, no lane below e)
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
        if N >= 2: e(f'vpbroadcastq .Lidx(%rip), {z(I[0])}'); e(f'vpbroadcastq .Lidx+8(%rip), {z(I[1])}')
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
def load_vr(reg):                   # EXT512: block at F + reg (bytes)
    for v in range(NVMAX): e(f'vmovdqu64 {F_OFF + 64 * v}(%rsp,{reg}), {z(VR[v])}')
def store_vr(reg):
    for v in range(NVMAX): e(f'vmovdqu64 {z(VR[v])}, {F_OFF + 64 * v}(%rsp,{reg})')
def zero_pcb():
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
def load_pcb():                    # the previous block's PHI / CPHI from the frame, BADC = 0
    e(f'vmovdqu64 {H9_OFF}(%rsp), {z(PHI)}'); e(f'vmovdqu64 {C9_OFF}(%rsp), {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
_slim_mark = len(L)                 # SLIM-RSQRT: the two-block stub (NV 11..20), dropped below unless kept
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
if a.sqfirst: lab('.Lbig_m')                        # SQUARE-FORM: the band's block 1 joins here
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
if not KEEP_BIG: del L[_slim_mark:]  # SLIM-RSQRT --nv-keep: no NV 11..20 kept

# ---------------------------------------------------------------- SMALL-NV: fused small levels (--pfuse)
# .Lsl{N} (NV = N, h <= HMAX[N]), entered from the P stage's dispatch with r15 X, rsi D_h, rdx G, ecx n, r8d h, r9d nn,
# ebx mid. (1) inv's per-level setup for NV = N (masks k1..k4, r12, r13, r14 = N; i0 / rbx for the generic NV 3 stub).
# (2) The P stage of ifma-v1 on the shared body: Z{N + (h >> 3)} and the staircase from segment N + ((h - 1) >> 3) down
# to N (the same rows and vectors, so the same lane sums), then the reduction and fast round of vectors NVPMAX - 1 .. 0
# (vectors above N + (h >> 3) are zero: their accumulators are cleared first) by a direct call of .Lrcp{NVPMAX - 1}; a
# lane left above M52 takes the register exact pass .Lxs over NVPMAX vectors. P' stays in VR[0 .. NVPMAX - 1] and
# VR[NVPMAX] = 0 (Pbuf's zeros above index n). (3) The MP from registers: row q = R - 2 (R = 3 .. h + 2) of vector v reads
# P'[ilo + R + 8 v + lane] = valignq $(R & 7) of P[(R >> 3) + v] (low) and P[(R >> 3) + v + 1] (high); the init row
# (q = 0, X[n] = 1) is valignq $2 into the lo group 0 (--mpinit's load). Straight-line rows R = HMAX + 2 .. 3, entered at
# R = h + 2 through .Lslm{N}_tab; the groups are summed into VR[v] / HR[v] and the path joins inv's stub at .Lst{N}_m
# (after its MP), so canon, G, correction, update and fix() are ifma-v1's code. Pbuf is neither written nor read.
SLHM = [int(x) for x in a.pfuse_hmax.split(',')]
SLGMP = [int(x) for x in a.pfuse_gmp.split(',')]
if a.pfuse:
    assert a.rsqrt and 1 <= a.pfuse <= 3 and len(SLHM) >= a.pfuse and len(SLGMP) >= a.pfuse
for N in range(1, a.pfuse + 1):
    HMAX = SLHM[N - 1]
    NVPMAX = N + (HMAX >> 3) + 1
    assert 1 <= HMAX and NVPMAX <= 7, (N, HMAX)
    P = [VR[u] for u in range(NVPMAX + 1)]
    gmp = SLGMP[:N]
    reserved = set(P) | {XR, TR} | {HR[v] for v in range(N)}
    pool = [r for r in (1, 2, 3, 5, 7, 16, 17, 18, 20, 22, 13, 14, 23, 24, 25, 26, 27, 28, 29) if r not in reserved]
    LA = [[pool.pop(0) for g in range(gmp[v])] for v in range(N)]
    HA = [[HR[v]] + [pool.pop(0) for g in range(1, gmp[v])] for v in range(N)]
    e('.p2align 4')
    lab(f'.Lsl{N}')
    e(f'movl %r9d, {NN_OFF}(%rsp)')
    e('movl %r9d, %edi'); e('subl %r8d, %edi')                                 # w
    e(f'movl ${8 * N - 1}, %r10d'); e('subl %edi, %r10d')                      # e = 8 N - w - 1
    e('movl $0xffff, %eax'); e('shlxl %r10d, %eax, %eax'); e('kmovw %eax, %k1'); e('kshiftrw $8, %k1, %k2')   # lanes >= e
    e('negl %ebx'); e('movl $1, %eax'); e('shlxl %r10d, %eax, %eax'); e('andl %ebx, %eax')
    e('kmovw %eax, %k3'); e('kshiftrw $8, %k3, %k4')                           # lane e when mid
    if N > a.small:                                                            # generic stub: i0 (its correction's segment 0)
        e('movl $7, %r9d'); e('subl %r10d, %r9d'); e(f'sarl ${LOGU}, %r9d'); e('incl %r9d')
        e('movl $1, %eax'); e('cmpl %eax, %r9d'); e('cmovll %eax, %r9d')
    e(f'movl ${N}, %r14d')
    e(f'leaq {8 - 64 * N}(%r15,%rcx,8), %r12')                                 # correction y: X + n + 1 - 8 N
    e('movl %ecx, %eax'); e('subl %r8d, %eax')                                 # n - h
    e(f'leaq {8 - 64 * N}(%r15,%rax,8), %r13')                                 # update: X + n - h + 1 - 8 N
    e(f'leaq {8 - 64 * N}(%rsi,%rax,8), %r11')                                 # D_h + ilo + 1 (Z's row 0 at -8)
    e('leaq -8(%r15,%rcx,8), %rsi')                                            # X + n - 1: row q = 1 (P stage and MP)
    for u in range(N + 1, NVPMAX):                                             # vectors that may lie above N + (h >> 3)
        zb(u)
    e(f'vpxord {z(P[NVPMAX])}, {z(P[NVPMAX])}, {z(P[NVPMAX])}')                # P[NVPMAX] = 0
    e('movl %r8d, %eax'); e('shrl $3, %eax')
    e('leaq .Lbig_tab(%rip), %rdi'); e(f'movslq {4 * N}(%rdi,%rax,4), %rax'); e('addq %rdi, %rax'); e('call *%rax')   # Z{N + (h >> 3)}
    e('leal -1(%r8), %eax'); e('shrl $3, %eax')                                # j = (h - 1) >> 3
    e('leal (,%rax,8), %ecx'); e('movl %r8d, %edi'); e('subl %ecx, %edi')      # c_f = h - 8 j (rows of segment N + j)
    e(f'leal {U - 1}(%rdi), %ecx'); e(f'shrl ${LOGU}, %ecx')                   # its iterations
    e('negl %edi'); e(f'andl ${U - 1}, %edi')                                  # r0 = -c_f mod U
    e('leaq (%rsi,%rdi,8), %r10')                                              # X + n - 1 + 8 r0
    e(f'leal {U * N}(%rdi,%rax,{U}), %ebx')                                    # Duff entry U (N + j) + r0
    e('negq %rdi'); e('leaq (%r11,%rdi,8), %r11')                              # D_h + ilo + 1 - 8 r0
    e('incl %eax')                                                             # segments N + j .. N
    e('leaq .Ldtab(%rip), %rdi'); e('movslq (%rdi,%rbx,4), %rbx'); e('addq %rdi, %rbx'); e('call *%rbx')
    if N > a.small: e(f'leaq {64 * N - 8}(%rdx), %rbx')                        # generic stub: correction x = G + 8 N - 1
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    e(f'call .Lrcp{NVPMAX - 1}')                                               # P' fast round, vectors NVPMAX - 1 .. 0
    if a.force_pexact: e(f'jmp .Lsl{N}_xp')
    else:
        e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k6'); e('kortestb %k6, %k6'); e(f'jnz .Lsl{N}_xp')
    lab(f'.Lsl{N}_xb')
    for v in range(N):                                                         # MP accumulators; init row (q = 0, X[n] = 1)
        for g in range(1, gmp[v]): e(f'vpxord {z(LA[v][g])}, {z(LA[v][g])}, {z(LA[v][g])}')
        for g in range(gmp[v]): e(f'vpxord {z(HA[v][g])}, {z(HA[v][g])}, {z(HA[v][g])}')
        e(f'valignq $2, {z(P[v])}, {z(P[v + 1])}, {z(LA[v][0])}')
    e(f'leaq .Lslm{N}_tab(%rip), %rax'); e('movslq (%rax,%r8,4), %rdi'); e('addq %rax, %rdi'); e('jmpq *%rdi')
    for R in range(HMAX + 2, 2, -1):                                           # row q = R - 2 reads X[n + 2 - R]
        lab(f'.Lslm{N}_r{R}')
        e(f'vpbroadcastq {-8 * (R - 3)}(%rsi), {z(XR)}')
        s_, u0 = R & 7, R >> 3
        for v in range(N - 1, -1, -1):
            if s_:
                e(f'valignq ${s_}, {z(P[u0 + v])}, {z(P[u0 + v + 1])}, {z(TR)}'); src = TR
            else: src = P[u0 + v]
            g = R % gmp[v]
            e(f'vpmadd52luq {z(src)}, {z(XR)}, {z(LA[v][g])}')
            e(f'vpmadd52huq {z(src)}, {z(XR)}, {z(HA[v][g])}')
    lab(f'.Lslm{N}_r2')
    for v in range(N):                                                         # group sums into VR[v] / HR[v]
        for regs, dst in ((LA[v], VR[v]), (HA[v], HR[v])):
            gs = list(regs)
            while len(gs) > 2:
                nxt = []
                for i in range(0, len(gs) - 1, 2):
                    e(f'vpaddq {z(gs[i + 1])}, {z(gs[i])}, {z(gs[i])}'); nxt.append(gs[i])
                if len(gs) % 2: nxt.append(gs[-1])
                gs = nxt
            if len(gs) == 2: e(f'vpaddq {z(gs[1])}, {z(gs[0])}, {z(dst)}')
            elif gs[0] != dst: e(f'vmovdqa64 {z(gs[0])}, {z(dst)}')
    e(f'jmp .Lst{N}_m')
    lab(f'.Lsl{N}_xp')                                                         # cold: P' exact pass in the registers
    e(f'movl ${NVPMAX}, %r14d'); e('call .Lxs'); e(f'movl ${N}, %r14d'); e(f'jmp .Lsl{N}_xb')

def sqblock():
    """SQUARE-FORM (--sqfirst): the square-first level for NV >= sf_min (header SQUARE-FORM). Entered from the P stage's
    NV test: eax NV, r8d h, r9d nn, ecx n, rsi D_h, r15 X; the frame holds h, nn, mid (RS_*), Pbuf (RS_P), G (CH_G), n.
    S stage: per S vector v = 0 .. NVS - 1 (deepest first; lane l = depth DS - 8 v - l) one product-scanning loop: rows a
    (broadcast x_a = X[n - a]) over the vector X[n - DS + 8 v + a ..], 4 rows per iteration into 4 row groups, rows a_s ..
    a_hi - 4 with a_hi = DS / 2 - 4 v and a_s = a_hi - 3 - 4 iters <= a_lo = max(0, DS - h - 7 - 8 v) (the up to 3 rows below
    a_lo read only zeros: X below n - h, or X above n); then rows a_hi - 3 .. a_hi - 1 with the lane masks of the diagonal
    (off-diagonal lanes l < 2 (a_hi - a)), the group sums doubled, the diagonal x_{a_hi - j}^2 at lane 2 j (one vpermq of
    X[n - a_hi ..]), V = lo + (hi up one lane, lane 0 from the previous vector), the fast round with the previous vector's
    lane 7, stored at Sbuf + 64 v (Sbuf = Pbuf + ilo). A lane left above M52 anywhere: the exact pass over Sbuf (cold).
    Registers: zmm0..3 lo groups, 4..7 hi groups, 8..11 rows, 12..15 row vectors, 16 diagonal, 17 index, 18 M52, 19 / 20
    previous hi / previous >> 52, 21 lanes above M52, 22..24 temporaries; r12 X + n - a_hi, r10 / r11 row pointers,
    ecx iterations, eax / r13d the two counts (count = min), ebx vectors left, rdi store pointer; k1..k4 masks.
    Then inv's per-level setup with D = D_h (no init row) and the band: rows r = 0 .. h + 8 NV - 2 of S (x = S's top digit,
    descending) over D_h + n + 2 - 8 NV - h: segment NV - 1 takes rows 0 .. h + 6 (Duff entry), then 8 rows per segment
    down to 0 (as the correction's staircase); NV 11..20: per column block (block 0: segment 9 takes rows 0 .. 8 NV + h -
    74; block 1: local segment NV - 11 takes rows 0 .. h + 6). It joins inv's stub after the MP (.Lst{N}_m, .Lbig_m) with
    r9d = i0 restored."""
    GS = a.sq_groups                                    # row groups per S vector (4: a 4-row loop; 8: an 8-row loop)
    if GS == 4:
        SL, SH = [0, 1, 2, 3], [4, 5, 6, 7]
        BR, TL = [8, 9, 10, 11], [12, 13, 14, 15]
        DV, IDX, M, PH, PC, BADS, TT, TT2, CC = 16, 17, 18, 19, 20, 21, 22, 23, 24
    else:
        SL, SH = list(range(8)), list(range(8, 16))
        BR, TL = [16, 17, 16, 17, 16, 17, 16, 17], [18, 19, 18, 19, 18, 19, 18, 19]
        DV, IDX, M, PH, PC, BADS, TT, TT2, CC = 20, 21, 22, 23, 24, 25, 26, 27, 28
    MK = {3: '%k1', 2: '%k2', 1: '%k3'}                 # masked row a_hi - j: off-diagonal lanes l < 2 j
    WIN = a.sq_win                                      # the row operands from the window array (aligned) or from X (split)
    VST, VLD, VB = (64, 'vmovdqa64', '%rsi') if WIN else (8, 'vmovdqu64', '%r12')
    e('.p2align 4')
    lab('.Lsq')
    e(f'movq %rsi, {SQ_DH}(%rsp)')
    FO = SQW                                            # STACK: the base frame's offset while the S stage's windows are allocated
    if WIN:                                             # windows W_s = X[s .. s + 7], s = n - h - 10 .., at WIN_OFF + 64 (s - n + h + 10)
        e('movl %ecx, %r9d'); e('subl %r8d, %r9d'); e('subl $10, %r9d'); e('movslq %r9d, %r9'); e('leaq (%r15,%r9,8), %r10')
        if SQW: e(f'subq ${SQW}, %rsp'); e('movq %rsp, %r11')   # STACK: the windows at [rsp, rsp + SQW) until the addq below
        else: e(f'leaq {WIN_OFF}(%rsp), %r11')
        e('leal 14(%r8), %r9d'); e('shrl $2, %r9d')    # (h + 14) >> 2 loops of 4: at least the h + 11 windows the rows read
        lab('.Lsq_w')
        for i in range(4): e(f'vmovdqu64 {8 * i}(%r10), {z(TL[i])}'); e(f'vmovdqa64 {z(TL[i])}, {64 * i}(%r11)')
        e('addq $32, %r10'); e('addq $256, %r11'); e('decl %r9d'); e('jnz .Lsq_w')
    e('leal (%r8,%rax,8), %edx')                        # t0 = 8 NV + h
    e('movl %ecx, %r10d'); e('subl %edx, %r10d'); e('movslq %r10d, %r10')    # ilo = n - t0
    e('andl $-2, %edx'); e(f'movl %edx, {SQ_DS + FO}(%rsp)')                 # DS = t0 & ~1
    e(f'movq {RS_P + FO}(%rsp), %rdi'); e('leaq (%rdi,%r10,8), %rdi'); e(f'movq %rdi, {SQ_SB + FO}(%rsp)')   # Sbuf = Pbuf + ilo
    e('movl %edx, %r10d'); e('shrl %r10d')              # DS / 2 = a_hi of vector 0
    e('movl %ecx, %r11d'); e('subl %r10d, %r11d'); e('movslq %r11d, %r11'); e('leaq (%r15,%r11,8), %r12')   # X + n - DS / 2
    if WIN:                                             # the window of the diagonal of vector 0: s = n - DS / 2
        e('leal 10(%r8), %r9d'); e('subl %r10d, %r9d'); e('shll $6, %r9d'); e('movslq %r9d, %r9')
        e('leaq (%rsp,%r9), %rsi' if SQW else f'leaq {WIN_OFF}(%rsp,%r9), %rsi')
    e('leal 4(%r8), %eax'); e('subl %r10d, %eax')       # count while a_lo > 0: h + 4 - DS / 2 + 4 v
    e('leal -3(%r10), %r13d')                           # count once a_lo = 0: DS / 2 - 3 - 4 v
    e('movl %edx, %ebx'); e('shrl $3, %ebx'); e('incl %ebx')    # NVS = DS / 8 + 1
    if a.sq_down:                                       # top-down: start at vector NVS - 1 (a_hi = DS / 2 - 4 (NVS - 1))
        e('leal -1(%rbx), %r9d')
        e('leal (,%r9,4), %r11d'); e('addl %r11d, %eax'); e('subl %r11d, %r13d')     # counts at v = NVS - 1
        e('shll $5, %r9d'); e('addq %r9, %r12')         # X + n - a_hi
        if WIN: e('leaq (%rsi,%r9,8), %rsi')           # its diagonal window
        e('leaq (%rdi,%r9,2), %rdi')                    # Sbuf + 64 (NVS - 1)
        e('movq %rsp, %rdx')                            # the first finalization goes to the frame (F area), discarded
        e('kxorb %k6, %k6, %k6')                        # and is masked out of the lanes-above-M52 test
    e(f'vpbroadcastq .Lm52(%rip), {z(M)}')
    e(f'vmovdqu64 .Lsq_idx(%rip), {z(IDX)}')
    for kk, mv in (('%k1', 0x3f), ('%k2', 0x0f), ('%k3', 0x03), ('%k4', 0x55)):
        e(f'movl ${mv}, %r9d'); e(f'kmovb %r9d, {kk}')
    for r_ in (PH, PC, BADS): e(f'vpxord {z(r_)}, {z(r_)}, {z(r_)}')
    e('xorl %r14d, %r14d')
    e('.p2align 4')
    lab('.Lsq_v')
    e('movl %eax, %ecx'); e('cmpl %r13d, %ecx'); e('cmovgl %r13d, %ecx')     # count = a_hi - 3 - a_lo
    e('addl $3, %ecx'); e('sarl $2, %ecx'); e('cmovsl %r14d, %ecx')           # iterations max(0, ceil(count / 4))
    e('movl %ecx, %r9d'); e('shll $5, %r9d')
    e('leaq 24(%r12,%r9), %r10')                                              # row a_s: x at X + n - a_s
    if WIN: e('shll $3, %r9d'); e('leaq -192(%rsi), %r11'); e('subq %r9, %r11')   # row a_s: window n - DS + 8 v + a_s
    else: e('leaq -24(%r12), %r11'); e('subq %r9, %r11')                      # row a_s: X + n - DS + 8 v + a_s
    for g in range(GS): e(f'vpxord {z(SL[g])}, {z(SL[g])}, {z(SL[g])}'); e(f'vpxord {z(SH[g])}, {z(SH[g])}, {z(SH[g])}')
    e('testl %ecx, %ecx'); e('jz .Lsq_m')
    if GS == 8:                                         # 8 rows an iteration: an odd count of 4-row blocks enters at row 4
        e('incl %ecx'); e('shrl %ecx'); e('jc .Lsq_l')
        e('addq $32, %r10'); e(f'subq ${4 * VST}, %r11'); e('jmp .Lsq_l4')
    e('.p2align 4')
    lab('.Lsq_l')
    for rr in range(GS):
        if rr == 4: lab('.Lsq_l4')
        e(f'vpbroadcastq {-8 * rr}(%r10), {z(BR[rr])}')
        e(f'{VLD} {VST * rr}(%r11), {z(TL[rr])}')
        e(f'vpmadd52luq {z(TL[rr])}, {z(BR[rr])}, {z(SL[rr])}')
        e(f'vpmadd52huq {z(TL[rr])}, {z(BR[rr])}, {z(SH[rr])}')
    e(f'subq ${8 * GS}, %r10'); e(f'addq ${VST * GS}, %r11'); e('decl %ecx'); e('jnz .Lsq_l')
    lab('.Lsq_m')
    for g, j in enumerate((3, 2, 1)):                   # rows a_hi - 3 .. a_hi - 1
        e(f'vpbroadcastq {8 * j}(%r12), {z(BR[g])}')
        e(f'{VLD} {-VST * j}({VB}), {z(TL[g])}')
        e(f'vpmadd52luq {z(TL[g])}, {z(BR[g])}, {z(SL[g])}{{{MK[j]}}}')
        e(f'vpmadd52huq {z(TL[g])}, {z(BR[g])}, {z(SH[g])}{{{MK[j]}}}')
    e(f'{VLD} ({VB}), {z(DV)}'); e(f'vpermq {z(DV)}, {z(IDX)}, {z(DV)}{{%k4}}{{z}}')    # x_{a_hi - j} at lane 2 j
    for X_ in (SL, SH):                                 # group sums
        d_ = 1
        while d_ < GS:
            for i_ in range(0, GS, 2 * d_): e(f'vpaddq {z(X_[i_ + d_])}, {z(X_[i_])}, {z(X_[i_])}')
            d_ *= 2
    e(f'vpaddq {z(SL[0])}, {z(SL[0])}, {z(SL[0])}'); e(f'vpaddq {z(SH[0])}, {z(SH[0])}, {z(SH[0])}')     # cross products twice
    e(f'vpmadd52luq {z(DV)}, {z(DV)}, {z(SL[0])}'); e(f'vpmadd52huq {z(DV)}, {z(DV)}, {z(SH[0])}')       # + the diagonal
    if not a.sq_down:                                   # bottom-up: finalize vector v now (the previous one's hi and >> 52 known)
        e(f'valignq $7, {z(PH)}, {z(SH[0])}, {z(TT)}'); e(f'vmovdqa64 {z(SH[0])}, {z(PH)}')              # V = lo + hi up one lane
        e(f'vpaddq {z(TT)}, {z(SL[0])}, {z(SL[0])}')
        e(f'vpsrlq $52, {z(SL[0])}, {z(CC)}'); e(f'valignq $7, {z(PC)}, {z(CC)}, {z(TT2)}'); e(f'vmovdqa64 {z(CC)}, {z(PC)}')
        e(f'vpandq {z(M)}, {z(SL[0])}, {z(SL[0])}'); e(f'vpaddq {z(TT2)}, {z(SL[0])}, {z(SL[0])}')     # fast round
        e(f'vporq {z(SL[0])}, {z(BADS)}, {z(BADS)}')
        e(f'vmovdqu64 {z(SL[0])}, (%rdi)')
        e('addq $64, %rdi'); e('addq $32, %r12'); e('addl $4, %eax'); e('subl $4, %r13d')
        if WIN: e('addq $256, %rsi')
        e('decl %ebx'); e('jnz .Lsq_v')
        if SQW: e(f'addq ${SQW}, %rsp')                 # STACK: the windows are dead (last read in the loop): the base frame again
    else:                                              # top-down: finalize vector v + 1 (raw sums in PC / PH) with vector v's
        PL = PC                                         # hi lane 7 and its V lane 7 = lo[7] + hi[6]
        e(f'valignq $7, {z(SH[0])}, {z(PH)}, {z(TT)}'); e(f'vpaddq {z(TT)}, {z(PL)}, {z(TT)}')          # V_{v+1}
        e(f'vpsrlq $52, {z(TT)}, {z(CC)}')
        e(f'valignq $7, {z(SH[0])}, {z(SH[0])}, {z(TT2)}'); e(f'vpaddq {z(TT2)}, {z(SL[0])}, {z(TT2)}')   # lanes 1..7: V_v
        e(f'vpsrlq $52, {z(TT2)}, {z(TT2)}'); e(f'valignq $7, {z(TT2)}, {z(CC)}, {z(TT2)}')
        e(f'vpandq {z(M)}, {z(TT)}, {z(TT)}'); e(f'vpaddq {z(TT2)}, {z(TT)}, {z(TT)}')                   # fast round
        e(f'vporq {z(TT)}, {z(BADS)}, {z(BADS)}{{%k6}}')
        e(f'vmovdqu64 {z(TT)}, (%rdx)')
        e(f'vmovdqa64 {z(SL[0])}, {z(PL)}'); e(f'vmovdqa64 {z(SH[0])}, {z(PH)}'); e('kxnorb %k6, %k6, %k6')
        e('movq %rdi, %rdx'); e('subq $64, %rdi'); e('subq $32, %r12'); e('subl $4, %eax'); e('addl $4, %r13d')
        if WIN: e('subq $256, %rsi')
        e('decl %ebx'); e('jnz .Lsq_v')
        e(f'vpxord {z(CC)}, {z(CC)}, {z(CC)}')          # vector 0: nothing below it
        e(f'valignq $7, {z(CC)}, {z(PH)}, {z(TT)}'); e(f'vpaddq {z(TT)}, {z(PL)}, {z(TT)}')
        e(f'vpsrlq $52, {z(TT)}, {z(TT2)}'); e(f'valignq $7, {z(CC)}, {z(TT2)}, {z(TT2)}')
        e(f'vpandq {z(M)}, {z(TT)}, {z(TT)}'); e(f'vpaddq {z(TT2)}, {z(TT)}, {z(TT)}')
        e(f'vporq {z(TT)}, {z(BADS)}, {z(BADS)}')
        e(f'vmovdqu64 {z(TT)}, (%rdx)')
    if a.force_sexact: e('jmp .Lsq_xa')
    else: e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADS)}, %k5'); e('kortestb %k5, %k5'); e('jnz .Lsq_xa')
    lab('.Lsq_xa_back')
    if a.sq_only: e('jmp .Ldone')                     # probe build: the S stage alone
    # inv's per-level setup with D = D_h and the band's x = S: rdi X, rsi D_h, rdx G, ecx n, r8d h, r9d nn, ebx mid
    e('movq %r15, %rdi'); e(f'movq {SQ_DH}(%rsp), %rsi'); e(f'movq {CH_G}(%rsp), %rdx'); e(f'movl {N_OFF}(%rsp), %ecx')
    e(f'movl {RS_H}(%rsp), %r8d'); e(f'movl {RS_NN}(%rsp), %r9d'); e(f'movl {RS_MID}(%rsp), %ebx')
    e(f'movl %r9d, {NN_OFF}(%rsp)')
    e('movl %r9d, %eax'); e('subl %r8d, %eax')          # w
    e('leal 10(%rax), %r14d'); e('shrl $3, %r14d')      # NV
    e('leal (,%r14,8), %r10d'); e('subl %eax, %r10d'); e('decl %r10d')        # e = 8 NV - w - 1
    e('movl $0xffff, %eax'); e('shlxl %r10d, %eax, %eax'); e('kmovw %eax, %k1'); e('kshiftrw $8, %k1, %k2')
    e('negl %ebx'); e('movl $1, %eax'); e('shlxl %r10d, %eax, %eax'); e('andl %ebx, %eax')
    e('kmovw %eax, %k3'); e('kshiftrw $8, %k3, %k4')
    e(f'kmovw %k1, {K_OFF}(%rsp)'); e(f'kmovw %k3, {K_OFF + 2}(%rsp)')
    e('movl $7, %r9d'); e('subl %r10d, %r9d'); e(f'sarl ${LOGU}, %r9d'); e('incl %r9d')
    e('movl $1, %eax'); e('cmpl %eax, %r9d'); e('cmovll %eax, %r9d')
    e(f'movl %r9d, {SQ_I0}(%rsp)')                      # i0: the correction's segment-0 iterations
    e('movl %ecx, %eax'); e('subl %r8d, %eax'); e('addl $2, %eax'); e('leal (,%r14,8), %ebx'); e('subl %ebx, %eax')
    e('movslq %eax, %rax'); e('leaq (%rsi,%rax,8), %r11')    # band y: D_h + n + 2 - 8 NV - h
    e(f'movq %r11, {PY_OFF}(%rsp)')
    e('movl %ecx, %eax'); e('subl %ebx, %eax'); e('incl %eax'); e('movslq %eax, %rax')
    e('leaq (%rdi,%rax,8), %r12')                       # correction y: X + n - 8 NV + 1
    e('movl %ecx, %eax'); e('subl %r8d, %eax'); e('incl %eax'); e('subl %ebx, %eax'); e('movslq %eax, %rax')
    e('leaq (%rdi,%rax,8), %r13')                       # update: X + n - h + 1 - 8 NV
    e('leaq -8(%rdx,%rbx,8), %rbx')                     # correction x: G + 8 NV - 1, descending
    e(f'movq {SQ_SB}(%rsp), %r10'); e(f'movl {SQ_DS}(%rsp), %eax'); e('leaq (%r10,%rax,8), %r10')   # band x: S's top digit
    e('leal 7(%r8), %r8d'); e(f'leal {U - 1}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx'); e('negl %r8d'); e(f'andl ${U - 1}, %r8d')
    e('movl %r14d, %eax'); e(f'movl ${8 // U}, %r9d')   # NV segments; segment 0 has 8 rows
    e('leaq .Lsf_tab(%rip), %rsi'); e('movslq (%rsi,%r14,4), %rdi'); e('addq %rsi, %rdi'); e('jmpq *%rdi')
    for N in range(a.sf_min, NVMAX + 1):
        if N not in KEEP: continue                      # SLIM-RSQRT --nv-keep
        s = N - 1
        e('.p2align 4')
        lab(f'.Lsf{N}')
        e(f'call .Lzz{s}')
        e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %rdi'); e('subq %rdi, %r11')
        e('leaq .Ldtab(%rip), %rsi'); e(f'movslq {4 * U * s}(%rsi,%r8,4), %rdi'); e('addq %rsi, %rdi'); e('call *%rdi')
        e(f'movl {SQ_I0}(%rsp), %r9d')
        e(f'jmp .Lst{N}_m')
    _slim_mark = len(L)                                 # SLIM-RSQRT: the two-block band entry, dropped below unless kept
    e('.p2align 4')
    lab('.Lsfbig')
    e('leaq .Lbig_tab(%rip), %rsi'); e('leal -11(%r14), %edi')      # nb - 1
    for i in range(5):
        e(f'movslq {40 * i}(%rsi,%rdi,4), %r8'); e('addq %rsi, %r8'); e(f'movq %r8, {TGT_OFF + 8 * i}(%rsp)')
    e(f'movl {RS_H}(%rsp), %r8d'); e('leal -73(%r8,%r14,8), %r8d')            # block 0: segment 9 takes 8 NV + h - 73 rows
    e(f'leal {U - 1}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx'); e('negl %r8d'); e(f'andl ${U - 1}, %r8d')
    e('movl $10, %eax'); e(f'movl ${8 // U}, %r9d')
    e('call .Lzz9')
    e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %rdi'); e('subq %rdi, %r11')
    e('leaq .Ldtab(%rip), %rsi'); e(f'movslq {4 * U * 9}(%rsi,%r8,4), %rdi'); e('addq %rsi, %rdi'); e('call *%rdi')
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vpxord {z(CPHI)}, {z(CPHI)}, {z(CPHI)}'); e(f'vpxord {z(BADC)}, {z(BADC)}, {z(BADC)}')
    red(9); save_b0(); e('call .Lrc9'); store_v(F_OFF)
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k0')
    e(f'movq {SQ_SB}(%rsp), %r10'); e(f'movl {SQ_DS}(%rsp), %eax'); e('leaq (%r10,%rax,8), %r10')
    e(f'movq {PY_OFF}(%rsp), %r11'); e('addq $640, %r11')
    e(f'movl {RS_H}(%rsp), %r8d'); e('addl $7, %r8d')                         # block 1: local segment NV - 11 takes h + 7 rows
    e(f'leal {U - 1}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx'); e('negl %r8d'); e(f'andl ${U - 1}, %r8d')
    e('leal -10(%r14), %eax'); e(f'movl ${8 // U}, %r9d')
    e('call .Lzz9')
    e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %rdi'); e('subq %rdi, %r11')
    e('leal -11(%r14), %esi'); e(f'leal (%r8,%rsi,{U}), %esi')
    e('leaq .Ldtab(%rip), %rdi'); e('movslq (%rdi,%rsi,4), %rsi'); e('addq %rdi, %rsi'); e('call *%rsi')
    e(f'movl {SQ_I0}(%rsp), %r9d')
    e('jmp .Lbig_m')
    if not KEEP_BIG: del L[_slim_mark:]                 # SLIM-RSQRT --nv-keep: no NV 11..20 kept

def xbig_post():
    """EXT512: canon / G / correction / update of NV 21..NV2 (see the header); appended before .Lend"""
    # EXT512: NV 21..NV2 in B = ceil(NV / 10) column blocks; the band (.Lxsfbig) leaves the top block in the registers, the
    # full blocks 0 .. B - 2 at F + 640 b (fast round), their lanes above M52 in k0, the top block's section entries in
    # TGT_OFF, X_NF = B - 1, X_TOFF = 640 (B - 1), X_TV = NV - 1 - 10 (B - 1) (rsi / rdi: block offset / count; the
    # sections Z, ZZ, SEG, RC, RCP, GZ, UPD do not touch them)
    e('.p2align 4')
    lab('.Lxbig_m')
    load_pcb()
    icall(2)                                        # RCP{top}: reduction + fast round of the top block
    if a.force_exact: e(f'movq {X_TOFF}(%rsp), %r8'); e('call .Lexact')
    else: tst2(BADC, '.Lxbig_xa')
    lab('.Lxbig_a')
    e('kxnorw %k0, %k0, %k1'); e('kxnorw %k0, %k0, %k2')           # blocks >= 1: no lane below e
    e(f'movq {X_TOFF}(%rsp), %rsi'); e('addq %rsi, %rdx'); icall(3); e('subq %rsi, %rdx')
    e(f'movl {X_NF}(%rsp), %edi')                    # G of the full blocks B - 2 .. 1 from F, then block 0
    lab('.Lxbg_g')
    e('subq $640, %rsi'); e('decl %edi'); e('jz .Lxbg_g0')
    load_vr('%rsi'); e('addq %rsi, %rdx'); e('call .Lgz9'); e('subq %rsi, %rdx'); e('jmp .Lxbg_g')
    lab('.Lxbg_g0')
    load_v(F_OFF); e(f'kmovw {K_OFF}(%rsp), %k1'); e('kshiftrw $8, %k1, %k2')
    e('call .Lgz9')
    # correction block 0: rows 8 NV - 1 .. 0 on vectors 0..9 (segment 9 takes every row t >= 72; segment 0 i0 iterations)
    e('movq %rbx, %r10'); e('movq %r12, %r11'); e('leal -9(%r14), %ecx'); e(f'shll ${3 - LOGU}, %ecx'); e('movl $10, %eax')
    e(f'movl {SQ_I0}(%rsp), %r9d')
    e('call .Lseg9')
    zero_pcb()
    red(9); save_b0(); e('call .Lrc9'); store_v(F_OFF)
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k0')
    # correction blocks b = 1 .. B - 2: rows 8 NV - 1 .. 80 b on vectors 10 b .. 10 b + 9 (segment 9: rows >= 80 b + 72)
    e('xorl %esi, %esi'); e(f'movl {X_NF}(%rsp), %edi')
    lab('.Lxbc_blk')
    e('addq $640, %rsi'); e('decl %edi'); e('jz .Lxbc_top')
    e('movq %rbx, %r10'); e('leaq (%r12,%rsi), %r11')
    e('movl %esi, %ecx'); e('shrl $6, %ecx'); e('negl %ecx'); e('leal -9(%r14,%rcx), %ecx'); e(f'shll ${3 - LOGU}, %ecx')   # NV - 10 b - 9
    e('movl $10, %eax'); e(f'movl ${8 // U}, %r9d')
    e('call .Lzz9'); e('call .Lseg9')
    load_pcb()
    red(9); save_b0(); e('call .Lrc9'); store_vr('%rsi')
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k6'); e('korb %k6, %k0, %k0')
    e('jmp .Lxbc_blk')
    # correction of the top block: rows 8 NV - 1 .. 80 (B - 1) on its local staircase (every row real)
    lab('.Lxbc_top')
    e('movq %rbx, %r10'); e('leaq (%r12,%rsi), %r11'); e(f'movl ${8 // U}, %ecx'); e(f'movl {X_TV}(%rsp), %eax'); e('incl %eax')
    e(f'movl ${8 // U}, %r9d')
    e('call .Lzz9'); icall(1)
    load_pcb()
    icall(2)
    if a.force_exact: e(f'movq {X_TOFF}(%rsp), %r8'); e('call .Lexact')
    else: tst2(BADC, '.Lxbig_xb')
    lab('.Lxbig_b')
    e('kxnorw %k0, %k0, %k1'); e('kxnorw %k0, %k0, %k2'); e('kxorw %k3, %k3, %k3'); e('kxorw %k4, %k4, %k4')
    e(f'vpxord {z(BADU)}, {z(BADU)}, {z(BADU)}')
    e(f'movq {X_TOFF}(%rsp), %rsi'); e('addq %rsi, %r13'); icall(4); e('subq %rsi, %r13')
    e(f'movl {X_NF}(%rsp), %edi')                    # update of the full blocks B - 2 .. 1 from F, then block 0
    lab('.Lxbu_blk')
    e('subq $640, %rsi'); e('decl %edi'); e('jz .Lxbu_0')
    load_vr('%rsi'); e('addq %rsi, %r13'); e('call .Lupd9'); e('subq %rsi, %r13'); e('jmp .Lxbu_blk')
    lab('.Lxbu_0')
    load_v(F_OFF)
    e(f'kmovw {K_OFF}(%rsp), %k1'); e('kshiftrw $8, %k1, %k2'); e(f'kmovw {K_OFF + 2}(%rsp), %k3'); e('kshiftrw $8, %k3, %k4')
    e('call .Lupd9')
    if XFW: e(f'addq ${XFW}, %rsp')                   # --fw-dyn: the base frame again (.Lfix, .Ldone)
    tst(BADU, '.Lfix')
    e('jmp .Ldone')
    if not a.force_exact:
        lab('.Lxbig_xa'); e(f'movq {X_TOFF}(%rsp), %r8'); e('call .Lexact'); e('jmp .Lxbig_a')
        lab('.Lxbig_xb'); e(f'movq {X_TOFF}(%rsp), %r8'); e('call .Lexact'); e('jmp .Lxbig_b')


def xbig_band():
    """EXT512: the band of NV 21..NV2 (see the header); appended before .Lend"""
    # EXT512: the band of NV 21..NV2 in B = ceil(NV / 10) column blocks (see the header); block b's rows start at S's top
    # digit (r10 at entry, kept at X_ST) over D_h + n + 2 - 8 NV - h + 640 b (PY_OFF + 640 b)
    lab('.Lxsfbig')
    if XFW: e(f'subq ${XFW}, %rsp')                   # --fw-dyn: this path's F at [rsp, rsp + XFW); the frame at rsp + XFW
    e('leal -1(%r14), %eax'); e('imull $205, %eax, %eax'); e('shrl $11, %eax')    # B - 1 = (NV - 1) / 10 (exact for NV <= 1029)
    e(f'movl %eax, {X_NF}(%rsp)')
    e('imull $640, %eax, %ecx'); e(f'movq %rcx, {X_TOFF}(%rsp)')
    e('imull $-10, %eax, %edi'); e('leal -1(%r14,%rdi), %edi'); e(f'movl %edi, {X_TV}(%rsp)')   # the top block's local top vector
    e('leaq .Lbig_tab(%rip), %rsi')
    for i in range(5):
        e(f'movslq {40 * i}(%rsi,%rdi,4), %r8'); e('addq %rsi, %r8'); e(f'movq %r8, {TGT_OFF + 8 * i}(%rsp)')
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vmovdqu64 {z(PHI)}, {H9_OFF}(%rsp)'); e(f'vmovdqu64 {z(PHI)}, {C9_OFF}(%rsp)')  # block 0
    e('kxorb %k0, %k0, %k0')
    e(f'movq %r10, {X_ST}(%rsp)')
    e('xorl %esi, %esi'); e(f'movl {X_NF}(%rsp), %edi')
    e('.p2align 4')
    lab('.Lxsb_blk')                                 # full block b (rsi = 640 b): segment 9 takes 8 NV + h - 73 - 80 b rows
    e(f'movl {RS_H}(%rsp), %r8d'); e('leal -73(%r8,%r14,8), %r8d'); e('movl %esi, %ecx'); e('shrl $3, %ecx'); e('subl %ecx, %r8d')
    e(f'leal {U - 1}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx'); e('negl %r8d'); e(f'andl ${U - 1}, %r8d')
    e(f'movq {X_ST}(%rsp), %r10'); e(f'movq {PY_OFF}(%rsp), %r11'); e('addq %rsi, %r11')
    e('call .Lzz9')
    e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %r9'); e('subq %r9, %r11')
    e('leaq .Ldtab(%rip), %rax'); e(f'movslq {4 * U * 9}(%rax,%r8,4), %r8'); e('addq %rax, %r8')
    e('movl $10, %eax'); e(f'movl ${8 // U}, %r9d')
    e('call *%r8')
    load_pcb()
    red(9); save_b0(); e('call .Lrc9'); store_vr('%rsi')
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k6'); e('korb %k6, %k0, %k0')
    e('addq $640, %rsi'); e('decl %edi'); e('jnz .Lxsb_blk')
    # the top block (rsi = 640 (B - 1)): local segment X_TV takes h + 7 rows
    e(f'movl {RS_H}(%rsp), %r8d'); e('addl $7, %r8d')
    e(f'leal {U - 1}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx'); e('negl %r8d'); e(f'andl ${U - 1}, %r8d')
    e(f'movq {X_ST}(%rsp), %r10'); e(f'movq {PY_OFF}(%rsp), %r11'); e('addq %rsi, %r11')
    e('call .Lzz9')
    e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %r9'); e('subq %r9, %r11')
    e(f'movl {X_TV}(%rsp), %eax'); e(f'leal (%r8,%rax,{U}), %r8d')
    e('leaq .Ldtab(%rip), %rax'); e('movslq (%rax,%r8,4), %r8'); e('addq %rax, %r8')
    e(f'movl {X_TV}(%rsp), %eax'); e('incl %eax'); e(f'movl ${8 // U}, %r9d')
    e('call *%r8')
    e(f'movl {SQ_I0}(%rsp), %r9d')
    e('jmp .Lxbig_m')

if a.sqfirst: sqblock()                             # SQUARE-FORM: S stage, band setup, band stubs

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
if a.sqfirst:
    # ZZ (--sqfirst): zero the accumulators of vectors v .. 0 (the band has no init row; .Lz is --mpinit's init form)
    for v in range(NVMAX - 1, -1, -1): lab(f'.Lzz{v}'); zb(v)
    e('ret')
if a.rsqrt:
    # PS (--rsqrt): the P stage's canonical-by-fast-round digits of vectors v .. 0 stored at rdx + 64 v
    for v in range(NVMAX - 1, -1, -1): lab(f'.Lps{v}'); e(f'vmovdqu64 {z(VR[v])}, {64 * v}(%rdx)')
    e('ret')
def with_fw(fn):                # EXT512 --fw-dyn: emit fn with every frame slot displaced by XFW (its own F at rsp)
    g = globals()
    names = ['H9_OFF', 'C9_OFF', 'N_OFF', 'NN_OFF', 'PY_OFF', 'IM_OFF', 'K_OFF', 'TGT_OFF', 'CH_D', 'CH_G', 'CH_LV', 'CH_L', 'CH_H',
             'H_OFF', 'I0_OFF', 'RS_XR', 'RS_P', 'RS_H', 'RS_NN', 'RS_MID', 'SQ_DH', 'SQ_SB', 'SQ_DS', 'SQ_I0', 'X_NF', 'X_TOFF', 'X_TV', 'X_ST']
    saved = {n: g[n] for n in names}
    for n in names: g[n] += XFW
    try: fn()
    finally: g.update(saved)
if NV2 > 20 and KEEP_X:         # EXT512: the NV 21..NV2 path after every original instruction (none of them moves); SLIM-RSQRT: if kept
    e('.p2align 4'); with_fw(xbig_band)
    e('.p2align 4'); with_fw(xbig_post)
lab('.Lend')


# ---------------------------------------------------------------- CUBIC512 (--cubic NAME): cubic (third-order) top levels
# All of it lives in the sections .lsbn_cu_text / .lsbn_cu_ro (orphans: the linker places them after .text / .rodata), so every
# original section of this object keeps its size, its bytes (but .Lsf_tab entries 1, 2: unused below sf_min) and its offsets.
# NAME(X, D_h, G, n, lv, L, Pbuf): the chain of NAME's quadratic twin (lv[L - 1] = m0 -> .. -> lv[0] = n), where the top level
# (lv[1] -> lv[0]) and, when lv[1] > 2 lv[2] too, the one below it are cubic: a level h -> nn is cubic when nn > 2h (guarded:
# nn <= 3h - 1). Without a cubic top level it jumps to the quadratic entry (same arguments). Otherwise: the quadratic part
# lv[c .. L - 1] through a nested call of the quadratic entry (its last level with mid = 0), the -2 of that level (X[n - h] -= 2,
# fix() on a borrow), then the c cubic levels (mid = 1 below the top). One cubic level h -> nn (w = nn - h, NV = (w + 10) >> 3,
# e = 8 NV - w - 1, R = 8 NV - h; cubic512/MODEL.md):
#   S1    the square-first S stage (.Lsq, unchanged) entered with "NV" ceil(h / 8): S = X_h^2 exactly to depth DS1 in
#         [2h, 2h + 7], at Sbuf = Pbuf + n - t0 (t0 = 8 ceil(h / 8) + h; Pbuf[n - t0 - 8 .. n - t0) zeroed first); RS_NN is set
#         to h (w 0), so the quadratic band setup that follows computes "NV" 1 and .Lsf_tab[1] jumps back (.Lcu_b)
#   band  D_h x S, rows j = 0 .. 2h (x = S's top digit, descending; y = D_h + n + 2 - 8 NV - h + j): the block staircase .Lcu_sp
#         (c0 = h + 7 rows over all NV vectors, then one vector less every 8 rows; column blocks of 10 vectors); canon;
#         G = beta^(8 NV) - 1 - W stored with every lane (no lane below e zeroed: G_full)
#   GxG   the S stage again with X := G, n := 8 NV - 1, h := R, "NV" := 0 (t0 = R, DS_q = R & ~1): every product of G's digits
#         at depth <= DS_q (each cross product once), stored at Pbuf + h - 1 (Pbuf[h + DS_q .. + 16) zeroed first);
#         RS_NN = R + 8: "NV" 2, .Lsf_tab[2] jumps back (.Lcu_q). Q lane m = Pbuf[h + DS_q - R + m] (zero from lane R)
#   H     per vector u = 0 .. s0 + 1 (s0 = (R - 1) >> 3): (3 Q_u >> 1) + ((Q one lane up) & 1) << 51 + G_u, one fast round
#         (the previous vector's lane 7 >> 52), stored over G; a lane above M52 or a carry out of vector s0 + 1: the carry into
#         G[8 (s0 + 2)] and the exact pass over G[0 .. 8 NV) (cold); then the lanes below e zeroed (H_hi)
#   corr  rows q = 0 .. h of X_h (x = X[n - q], descending; row 0 = H itself by Z's init load) over H's vectors (y = H + q):
#         .Lcu_sp with c0 = 8, J = h + 1; canon; update X[n - h + 1 - 8 NV + ..] += C (lanes >= e; -2 at lane e when mid); fix()
def cu_disp(d):
    g = globals()
    names = ['H9_OFF', 'C9_OFF', 'N_OFF', 'NN_OFF', 'PY_OFF', 'IM_OFF', 'K_OFF', 'TGT_OFF', 'CH_D', 'CH_G', 'CH_LV', 'CH_L', 'CH_H',
             'H_OFF', 'I0_OFF', 'RS_XR', 'RS_P', 'RS_H', 'RS_NN', 'RS_MID', 'SQ_DH', 'SQ_SB', 'SQ_DS', 'SQ_I0', 'X_NF', 'X_TOFF', 'X_TV',
             'X_ST', 'CU_X', 'CU_LV', 'CU_L', 'CU_H', 'CU_NN', 'CU_MID', 'CU_NV', 'CU_R', 'CU_E', 'CU_DSQ', 'CU_SPX', 'CU_SPY', 'CU_SPC0',
             'CU_SPJ', 'CU_SPNV']
    saved = {n: g[n] for n in names}
    for n in names: g[n] += d
    return saved


def cu_sp(tag, init):
    """the block staircase product (F area at rsp: the frame slots displaced). In: CU_SPX x0 (row 0's x digit; rows descend 8 B),
    CU_SPY y0 (row 0's lane 0; rows ascend 8 B), CU_SPC0 c0, CU_SPJ J, CU_SPNV NVo. Row j < J runs over vectors 0 .. top(j),
    top = NVo - 1 for j < c0, then one vector less every 8 rows (rows past J inside the last segment read x digits that are zero);
    init: row 0 is the y operand itself (x0's digit is 1; Z's init load). Column blocks of 10 vectors: full block b (rsi = 640 b)
    takes min(J, c0 + 8 (NVo - 10) - 80 b) rows on segment 9 (Duff entry), then up to 9 segments of 8 rows; the top block
    (local top vector X_TV) min(J, c0) rows, then up to X_TV segments. Out: full blocks at F + 640 b (fast round), k0 = their
    lanes above M52; the top block in the registers after RCP (BADC); X_NF = B - 1, X_TOFF = 640 (B - 1), X_TV, TGT_OFF =
    .Lbig_tab row X_TV. Uses rax, rcx, rsi, rdi, r8..r12; sections keep rsi, rdi, r12."""
    e(f'movl {CU_SPNV}(%rsp), %eax'); e('decl %eax'); e('imull $205, %eax, %eax'); e('shrl $11, %eax')      # B - 1
    e(f'movl %eax, {X_NF}(%rsp)')
    e('imull $640, %eax, %ecx'); e(f'movq %rcx, {X_TOFF}(%rsp)')
    e('imull $-10, %eax, %edi'); e(f'addl {CU_SPNV}(%rsp), %edi'); e('decl %edi'); e(f'movl %edi, {X_TV}(%rsp)')
    e('leaq .Lbig_tab(%rip), %rsi')
    for i in range(5):
        e(f'movslq {40 * i}(%rsi,%rdi,4), %r8'); e('addq %rsi, %r8'); e(f'movq %r8, {TGT_OFF + 8 * i}(%rsp)')
    e(f'vpxord {z(PHI)}, {z(PHI)}, {z(PHI)}'); e(f'vmovdqu64 {z(PHI)}, {H9_OFF}(%rsp)'); e(f'vmovdqu64 {z(PHI)}, {C9_OFF}(%rsp)')
    e('kxorb %k0, %k0, %k0')
    e('xorl %esi, %esi'); e(f'movl {X_NF}(%rsp), %edi')
    e('testl %edi, %edi'); e(f'jz .Lcu_{tag}_top')
    e('.p2align 4')
    lab(f'.Lcu_{tag}_blk')
    e(f'movl {CU_SPC0}(%rsp), %r8d'); e(f'movl {CU_SPNV}(%rsp), %eax'); e('leal -80(%r8,%rax,8), %r8d')
    e('movl %esi, %ecx'); e('shrl $3, %ecx'); e('subl %ecx, %r8d')                       # c0 + 8 (NVo - 10) - 80 b
    e(f'movl {CU_SPJ}(%rsp), %ecx'); e('cmpl %ecx, %r8d'); e('cmovgl %ecx, %r8d')         # c9 = min(J, ..)
    e('subl %r8d, %ecx'); e('addl $7, %ecx'); e('shrl $3, %ecx')                          # ceil((J - c9) / 8)
    e('movl $9, %eax'); e('cmpl %eax, %ecx'); e('cmoval %eax, %ecx'); e('leal 1(%rcx), %r12d')   # segments: 1 + min(9, ..)
    e(f'movq {CU_SPY}(%rsp), %r11'); e('addq %rsi, %r11'); e(f'movq {CU_SPX}(%rsp), %r10')
    if init: e('addq $8, %r11'); e('subq $8, %r10'); e('decl %r8d'); e('call .Lz9')    # row 0: Z's init load
    else: e('call .Lzz9')
    e(f'leal {U - 1}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx'); e('negl %r8d'); e(f'andl ${U - 1}, %r8d')   # iterations, r0
    e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %r9'); e('subq %r9, %r11')
    e('leaq .Ldtab(%rip), %rax'); e(f'movslq {4 * U * 9}(%rax,%r8,4), %r8'); e('addq %rax, %r8')
    e('movl %r12d, %eax'); e(f'movl ${8 // U}, %r9d')
    e('call *%r8')
    load_pcb(); red(9); save_b0(); e('call .Lrc9'); store_vr('%rsi')
    e(f'vptestmq .Lb52(%rip){{1to8}}, {z(BADC)}, %k6'); e('korb %k6, %k0, %k0')
    e('addq $640, %rsi'); e('decl %edi'); e(f'jnz .Lcu_{tag}_blk')
    lab(f'.Lcu_{tag}_top')
    e(f'movl {CU_SPC0}(%rsp), %r8d'); e(f'movl {CU_SPJ}(%rsp), %ecx'); e('cmpl %ecx, %r8d'); e('cmovgl %ecx, %r8d')   # min(J, c0)
    e('subl %r8d, %ecx'); e('addl $7, %ecx'); e('shrl $3, %ecx')
    e(f'movl {X_TV}(%rsp), %eax'); e('cmpl %eax, %ecx'); e('cmoval %eax, %ecx'); e('leal 1(%rcx), %r12d')
    e(f'movq {CU_SPY}(%rsp), %r11'); e('addq %rsi, %r11'); e(f'movq {CU_SPX}(%rsp), %r10')
    if init: e('addq $8, %r11'); e('subq $8, %r10'); e('decl %r8d'); icall(0)           # Z{X_TV}: the init load
    else: e('call .Lzz9')
    e(f'leal {U - 1}(%r8), %ecx'); e(f'shrl ${LOGU}, %ecx'); e('negl %r8d'); e(f'andl ${U - 1}, %r8d')
    e('leaq (%r10,%r8,8), %r10'); e('leaq (,%r8,8), %r9'); e('subq %r9, %r11')
    e(f'movl {X_TV}(%rsp), %eax'); e(f'leal (%r8,%rax,{U}), %r8d')                         # Duff entry U X_TV + r0
    e('leaq .Ldtab(%rip), %rax'); e('movslq (%rax,%r8,4), %r8'); e('addq %rax, %r8')
    e('movl %r12d, %eax'); e(f'movl ${8 // U}, %r9d')
    e('call *%r8')
    load_pcb(); icall(2)                                                                   # RCP{X_TV}


def cublock():
    CUN = a.cubic
    e(f'.section {CU_TEXT},"ax",@progbits')               # CUBIC512: own section: every original section keeps x512's size
    e('.p2align 6')
    e(f'.globl {CUN}')
    if a.hidden: e(f'.hidden {CUN}')
    e(f'.type {CUN}, @function')
    lab(CUN)
    e('cmpl $2, %r9d'); e(f'jb {a.name}')                                                  # no level
    e('movl 4(%r8), %r10d'); e('leal (%r10,%r10), %r11d'); e('cmpl %r11d, (%r8)'); e(f'jbe {a.name}')   # lv[0] <= 2 lv[1]: quadratic
    e('pushq %rbp'); e('movq %rsp, %rbp')
    e('pushq %rbx'); e('pushq %r12'); e('pushq %r13'); e('pushq %r14'); e('pushq %r15')
    e('andq $-64, %rsp'); e(f'subq ${FRAME}, %rsp')
    e('movq 16(%rbp), %rax'); e(f'movq %rax, {RS_P}(%rsp)')
    e(f'movq %rsi, {CH_D}(%rsp)'); e(f'movq %rdx, {CH_G}(%rsp)'); e(f'movq %r8, {CU_LV}(%rsp)')
    e('movl %ecx, %ecx'); e(f'movl %ecx, {N_OFF}(%rsp)')
    e('movq %rdi, %r15'); e(f'movq %rdi, {CU_X}(%rsp)')
    e('movl $1, %eax')                                                                     # c: 1, or 2 when lv[1] > 2 lv[2]
    e('cmpl $3, %r9d'); e('jb .Lcu_c')
    e('movl 8(%r8), %r11d'); e('addl %r11d, %r11d'); e('cmpl %r11d, %r10d'); e('jbe .Lcu_c')
    e('movl $2, %eax')
    lab('.Lcu_c')
    e(f'movl %eax, {CU_L}(%rsp)')
    e('movl %r9d, %r11d'); e('subl %eax, %r11d')                                           # entries of lv[c .. L - 1]
    e('cmpl $2, %r11d'); e('jb .Lcu_next')                                                 # no quadratic level below
    e('movq %r15, %rdi'); e(f'movq {CH_D}(%rsp), %rsi'); e(f'movq {CH_G}(%rsp), %rdx'); e(f'movl {N_OFF}(%rsp), %ecx')
    e('leaq (%r8,%rax,4), %r8'); e('movl %r11d, %r9d')
    e(f'movq {RS_P}(%rsp), %rax'); e('subq $16, %rsp'); e('movq %rax, (%rsp)')
    e(f'call {a.name}')                                                                     # the quadratic part
    e('addq $16, %rsp')
    e(f'movq {CU_LV}(%rsp), %r8'); e(f'movl {CU_L}(%rsp), %eax'); e('movl (%r8,%rax,4), %eax')   # h = lv[c]
    e(f'movl {N_OFF}(%rsp), %edx'); e('movl %edx, %esi'); e('subl %eax, %esi')
    e('subq $2, (%r15,%rsi,8)')                                                            # its -2 (mid); a borrow: fix()
    if not a.force_fix: e('jae .Lcu_next')
    e('movq %r15, %rdi'); e('vzeroupper'); e(f'call {a.fixsym}')
    e('.p2align 4')
    lab('.Lcu_next')
    e(f'movl {CU_L}(%rsp), %eax'); e('testl %eax, %eax'); e('jz .Lcu_epi')
    e('decl %eax'); e(f'movl %eax, {CU_L}(%rsp)')
    e(f'movq {CU_LV}(%rsp), %rsi'); e('movl (%rsi,%rax,4), %r9d'); e('movl 4(%rsi,%rax,4), %r8d')   # nn, h
    e('xorl %ebx, %ebx'); e('testl %eax, %eax'); e('setnz %bl')                                     # mid
    e(f'movl %r8d, {CU_H}(%rsp)'); e(f'movl %r9d, {CU_NN}(%rsp)'); e(f'movl %ebx, {CU_MID}(%rsp)')
    e('movl %r9d, %eax'); e('subl %r8d, %eax')                                             # w
    e('leal 10(%rax), %r14d'); e('shrl $3, %r14d'); e(f'movl %r14d, {CU_NV}(%rsp)')        # NV
    e('leal (,%r14,8), %edx'); e('subl %eax, %edx'); e('decl %edx'); e(f'movl %edx, {CU_E}(%rsp)')    # e
    e('leal (,%r14,8), %edx'); e('subl %r8d, %edx'); e(f'movl %edx, {CU_R}(%rsp)')         # R
    e('vpxord %zmm0, %zmm0, %zmm0')
    e(f'movq {CH_G}(%rsp), %rdx'); e('movl %r14d, %eax'); e('shll $6, %eax'); e('vmovdqu64 %zmm0, (%rdx,%rax)')   # G[8 NV ..] = 0
    e('leal 7(%r8), %eax'); e('shrl $3, %eax')                                             # S1's "NV": ceil(h / 8)
    e('leal (%r8,%rax,8), %edi'); e(f'movl {N_OFF}(%rsp), %ecx'); e('movl %ecx, %r10d'); e('subl %edi, %r10d'); e('movslq %r10d, %r10')
    e(f'movq {RS_P}(%rsp), %r11'); e('vmovdqu64 %zmm0, -64(%r11,%r10,8)')                  # Pbuf[n - t0 - 8 .. n - t0) = 0
    e(f'movl %r8d, {RS_H}(%rsp)'); e(f'movl %r8d, {RS_NN}(%rsp)'); e(f'movl $0, {RS_MID}(%rsp)')   # fake w 0: "NV" 1
    e(f'movq {CH_D}(%rsp), %rsi')
    e('jmp .Lsq')                                                                          # S1, back at .Lcu_b
    # ---- band, G
    e('.p2align 4')
    lab('.Lcu_b')
    e(f'movq %r10, {CU_SPX}(%rsp)')                                                        # S's top digit
    e(f'movl {CU_NV}(%rsp), %r14d'); e(f'movl {CU_H}(%rsp), %r8d')
    e('leal 7(%r8), %eax'); e(f'movl %eax, {CU_SPC0}(%rsp)'); e('leal 1(%r8,%r8), %eax'); e(f'movl %eax, {CU_SPJ}(%rsp)')
    e(f'movl %r14d, {CU_SPNV}(%rsp)')
    e(f'movl {N_OFF}(%rsp), %eax'); e('subl %r8d, %eax'); e('addl $2, %eax'); e('leal (,%r14,8), %ecx'); e('subl %ecx, %eax'); e('cltq')
    e(f'movq {CH_D}(%rsp), %rdx'); e('leaq (%rdx,%rax,8), %rax'); e(f'movq %rax, {CU_SPY}(%rsp)')   # D_h + n + 2 - 8 NV - h
    e(f'subq ${XFWC}, %rsp')
    sv = cu_disp(XFWC)
    cu_sp('b', False)
    if a.force_exact: e('jmp .Lcu_bxa')
    else: tst2(BADC, '.Lcu_bxa')
    lab('.Lcu_ba')
    e('kxnorw %k0, %k0, %k1'); e('kxnorw %k0, %k0, %k2')                                  # G_full: no lane zeroed
    e(f'movq {X_TOFF}(%rsp), %rsi'); e(f'movq {CH_G}(%rsp), %rdx'); e('addq %rsi, %rdx'); icall(3)   # GZ{X_TV}
    e(f'movl {X_NF}(%rsp), %edi')
    lab('.Lcu_bg')
    e('testl %edi, %edi'); e('jz .Lcu_bgd')
    e('subq $640, %rsi'); e('subq $640, %rdx'); e('decl %edi')
    load_vr('%rsi'); e('call .Lgz9')
    e('jmp .Lcu_bg')
    lab('.Lcu_bgd')
    if CU_RSTAIR:                                                                          # ---- G x G on the staircase
        if CU_RSTAIR < 10 ** 6: e(f'cmpl ${CU_RSTAIR}, {CU_R}(%rsp)'); e('ja .Lcu_gsq')
        # rows r = 0 .. R - 1 of G[8 NV - 1 - r] (x, descending) over G + h + r (y): s0 + 1 vectors, cf = R - 8 s0 rows on segment s0,
        # then 8 a segment: the last row is R - 1 exactly (no product of a digit below G[h]); products below lane 0 dropped whole
        e(f'movl {CU_R}(%rsp), %eax'); e('leal -1(%rax), %ecx'); e('shrl $3, %ecx'); e('leal 1(%rcx), %edx'); e(f'movl %edx, {CU_SPNV}(%rsp)')
        e('shll $3, %ecx'); e('subl %ecx, %eax'); e(f'movl %eax, {CU_SPC0}(%rsp)'); e(f'movl {CU_R}(%rsp), %eax'); e(f'movl %eax, {CU_SPJ}(%rsp)')
        e(f'movq {CH_G}(%rsp), %rax'); e('leal -1(,%r14,8), %ecx'); e('leaq (%rax,%rcx,8), %rcx'); e(f'movq %rcx, {CU_SPX}(%rsp)')
        e(f'movl {CU_H}(%rsp), %ecx'); e('leaq (%rax,%rcx,8), %rcx'); e(f'movq %rcx, {CU_SPY}(%rsp)')
        cu_sp('q', False)
        e(f'movl {CU_SPNV}(%rsp), %r14d')                                                  # the exact pass's vector count
        if a.force_exact: e('jmp .Lcu_qxa')
        else: tst2(BADC, '.Lcu_qxa')
        lab('.Lcu_qa')
        # Q lane m stored at Pbuf[h + m] (CU_DSQ = R: the H loop's lane 0 = Pbuf[h + DS_q - R]), two zero vectors above it
        e(f'movq {RS_P}(%rsp), %rdx'); e(f'movl {CU_H}(%rsp), %eax'); e('leaq (%rdx,%rax,8), %rdx')
        e(f'movq {X_TOFF}(%rsp), %rsi'); e('addq %rsi, %rdx')
        e(f'movl {X_TV}(%rsp), %ecx'); e('leaq .Lps_tab(%rip), %rax'); e('movslq (%rax,%rcx,4), %rdi'); e('addq %rax, %rdi'); e('call *%rdi')   # PS{X_TV}
        e('shll $6, %ecx'); e('vpxord %zmm0, %zmm0, %zmm0'); e('vmovdqu64 %zmm0, 64(%rdx,%rcx)'); e('vmovdqu64 %zmm0, 128(%rdx,%rcx)')
        e(f'movl {X_NF}(%rsp), %edi')
        lab('.Lcu_qs')
        e('testl %edi, %edi'); e('jz .Lcu_qsd')
        e('subq $640, %rsi'); e('subq $640, %rdx'); e('decl %edi')
        load_vr('%rsi'); e('call .Lps9')
        e('jmp .Lcu_qs')
        lab('.Lcu_qsd')
        e(f'movl {CU_R}(%rsp), %eax'); e(f'movl %eax, {CU_DSQ}(%rsp)')
        globals().update(sv)
        e(f'addq ${XFWC}, %rsp')
        e(f'movl {CU_NV}(%rsp), %r14d')
        e('jmp .Lcu_q')
        sv = cu_disp(XFWC)
        lab('.Lcu_gsq')
    globals().update(sv)
    e(f'addq ${XFWC}, %rsp')
    # ---- G x G: the S stage on G's top digits
    e(f'movl {CU_NV}(%rsp), %r14d')
    e(f'movl {CU_R}(%rsp), %r8d'); e('movl %r8d, %eax'); e('andl $-2, %eax'); e(f'movl %eax, {CU_DSQ}(%rsp)')
    e(f'addl {CU_H}(%rsp), %eax'); e('cltq'); e(f'movq {RS_P}(%rsp), %rdx'); e('vpxord %zmm0, %zmm0, %zmm0')
    e('vmovdqu64 %zmm0, (%rdx,%rax,8)'); e('vmovdqu64 %zmm0, 64(%rdx,%rax,8)')             # Pbuf[h + DS_q .. + 16) = 0
    e(f'movl %r8d, {RS_H}(%rsp)'); e('leal 8(%r8), %r9d'); e(f'movl %r9d, {RS_NN}(%rsp)')              # fake w 8: "NV" 2
    e(f'movq {CH_G}(%rsp), %r15'); e('leal -1(,%r14,8), %ecx'); e('xorl %eax, %eax')       # X := G, n := 8 NV - 1, "NV" 0
    e('jmp .Lsq')                                                                          # back at .Lcu_q
    # ---- H = G + floor(3 Q / 2)
    e('.p2align 4')
    lab('.Lcu_q')
    e(f'movq {CU_X}(%rsp), %r15'); e(f'movl {CU_NV}(%rsp), %r14d')
    e(f'movl {CU_E}(%rsp), %r10d'); e('movl $0xffff, %eax'); e('shlxl %r10d, %eax, %eax'); e('kmovw %eax, %k1'); e('kshiftrw $8, %k1, %k2')
    e(f'movl {CU_MID}(%rsp), %ebx'); e('negl %ebx'); e('movl $1, %eax'); e('shlxl %r10d, %eax, %eax'); e('andl %ebx, %eax')
    e('kmovw %eax, %k3'); e('kshiftrw $8, %k3, %k4')
    e(f'kmovw %k1, {K_OFF}(%rsp)'); e(f'kmovw %k3, {K_OFF + 2}(%rsp)')
    e(f'movl {CU_H}(%rsp), %eax'); e(f'addl {CU_DSQ}(%rsp), %eax'); e(f'subl {CU_R}(%rsp), %eax'); e('cltq')
    e(f'movq {RS_P}(%rsp), %rsi'); e('leaq (%rsi,%rax,8), %rsi')                           # Q lane 0
    e(f'movq {CH_G}(%rsp), %rdx')
    e(f'movl {CU_R}(%rsp), %ecx'); e('decl %ecx'); e('shrl $3, %ecx'); e('addl $2, %ecx')  # s0 + 2 vectors
    e('vpbroadcastq .Lm52(%rip), %zmm22'); e('vpbroadcastq .Lcu_one(%rip), %zmm23')
    e('vpxord %zmm21, %zmm21, %zmm21'); e('vpxord %zmm24, %zmm24, %zmm24')
    e('.p2align 4')
    lab('.Lcu_h')
    e('vmovdqu64 (%rsi), %zmm16'); e('vmovdqu64 8(%rsi), %zmm17')
    e('vpaddq %zmm16, %zmm16, %zmm18'); e('vpaddq %zmm16, %zmm18, %zmm18'); e('vpsrlq $1, %zmm18, %zmm18')   # floor(3 Q / 2) a lane
    e('vpandq %zmm23, %zmm17, %zmm17'); e('vpsllq $51, %zmm17, %zmm17'); e('vpaddq %zmm17, %zmm18, %zmm18')   # + the lane above's half
    e('vpaddq (%rdx), %zmm18, %zmm18')                                                     # + G
    e('vpsrlq $52, %zmm18, %zmm19'); e('valignq $7, %zmm21, %zmm19, %zmm20'); e('vmovdqa64 %zmm19, %zmm21')
    e('vpandq %zmm22, %zmm18, %zmm18'); e('vpaddq %zmm20, %zmm18, %zmm18')                # fast round
    e('vporq %zmm18, %zmm24, %zmm24')
    e('vmovdqu64 %zmm18, (%rdx)')
    e('addq $64, %rsi'); e('addq $64, %rdx'); e('decl %ecx'); e('jnz .Lcu_h')
    if a.force_exact: e('jmp .Lcu_hx')
    e('vptestmq %zmm21, %zmm21, %k5'); e('kmovb %k5, %eax'); e('testb $-128, %al'); e('jnz .Lcu_hx')   # carry out of the last
    e('vptestmq .Lb52(%rip){1to8}, %zmm24, %k5'); e('kortestb %k5, %k5'); e('jnz .Lcu_hx')            # a lane above M52
    lab('.Lcu_hz')
    e(f'movq {CH_G}(%rsp), %rdx'); e('vpxord %zmm0, %zmm0, %zmm0'); e('knotw %k1, %k5')
    e('vmovdqu64 %zmm0, (%rdx){%k5}'); e('kshiftrw $8, %k5, %k5'); e('vmovdqu64 %zmm0, 64(%rdx){%k5}')   # H_hi
    # ---- correction, update
    if a.kl_cplan:                                                                         # KARA-L1: the same 11 bytes as the
        e('jmp .Lkl_c'); e('.byte 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00')                     # movl $8, CU_SPC0(%rsp) (c7 84 24 disp32
        lab('.Lcu_c0')                                                                     # imm32): jmp rel32 + a 6-byte nop
    else: e(f'movl $8, {CU_SPC0}(%rsp)')
    e(f'movl {CU_H}(%rsp), %eax'); e('incl %eax'); e(f'movl %eax, {CU_SPJ}(%rsp)')
    e(f'movl %r14d, {CU_SPNV}(%rsp)')
    e(f'movl {N_OFF}(%rsp), %eax'); e('leaq (%r15,%rax,8), %rax'); e(f'movq %rax, {CU_SPX}(%rsp)')   # X + n: row 0
    e(f'movq {CH_G}(%rsp), %rax'); e(f'movq %rax, {CU_SPY}(%rsp)')
    e(f'subq ${XFWC}, %rsp')
    sv = cu_disp(XFWC)
    cu_sp('c', True)
    if a.force_exact: e('jmp .Lcu_cxa')
    else: tst2(BADC, '.Lcu_cxa')
    lab('.Lcu_ca')
    e(f'movl {N_OFF}(%rsp), %eax'); e(f'subl {CU_H}(%rsp), %eax'); e('incl %eax'); e('leal (,%r14,8), %ecx'); e('subl %ecx, %eax'); e('cltq')
    e('leaq (%r15,%rax,8), %r13')                                                          # X + n - h + 1 - 8 NV
    e(f'vpxord {z(BADU)}, {z(BADU)}, {z(BADU)}')
    e(f'movl {X_NF}(%rsp), %edi')
    e(f'kmovw {K_OFF}(%rsp), %k1'); e('kshiftrw $8, %k1, %k2'); e(f'kmovw {K_OFF + 2}(%rsp), %k3'); e('kshiftrw $8, %k3, %k4')
    e('testl %edi, %edi'); e('jz .Lcu_u1')                                                 # one block: the top block is block 0
    e('kxnorw %k0, %k0, %k1'); e('kxnorw %k0, %k0, %k2'); e('kxorw %k3, %k3, %k3'); e('kxorw %k4, %k4, %k4')
    lab('.Lcu_u1')
    e(f'movq {X_TOFF}(%rsp), %rsi'); e('addq %rsi, %r13'); icall(4); e('subq %rsi, %r13')  # UPD{X_TV}
    lab('.Lcu_ub')
    e('testl %edi, %edi'); e('jz .Lcu_ud')
    e('subq $640, %rsi'); e('decl %edi'); e('jnz .Lcu_um')
    e(f'kmovw {K_OFF}(%rsp), %k1'); e('kshiftrw $8, %k1, %k2'); e(f'kmovw {K_OFF + 2}(%rsp), %k3'); e('kshiftrw $8, %k3, %k4')
    lab('.Lcu_um')
    load_vr('%rsi'); e('addq %rsi, %r13'); e('call .Lupd9'); e('subq %rsi, %r13')
    e('jmp .Lcu_ub')
    lab('.Lcu_ud')
    globals().update(sv)
    e(f'addq ${XFWC}, %rsp')
    if a.force_fix: e('jmp .Lcu_fix')
    else: tst(BADU, '.Lcu_fix')
    e('jmp .Lcu_next')
    lab('.Lcu_fix')
    e('movq %r15, %rdi'); e(f'movl {N_OFF}(%rsp), %edx'); e('movl %edx, %esi'); e(f'subl {CU_NN}(%rsp), %esi')
    e('vzeroupper'); e(f'call {a.fixsym}')
    e('jmp .Lcu_next')
    lab('.Lcu_epi')
    e('leaq -40(%rbp), %rsp')
    e('popq %r15'); e('popq %r14'); e('popq %r13'); e('popq %r12'); e('popq %rbx'); e('popq %rbp')
    e('vzeroupper')
    e('ret')
    cucold()
    lab('.Lcu_end')


def cucold():
    """CUBIC512 cold paths (.text.unlikely): the exact passes of the band and the correction (F area allocated), of H"""
    sv = cu_disp(XFWC)
    lab('.Lcu_bxa'); e(f'movq {X_TOFF}(%rsp), %r8'); e('call .Lexact'); e('jmp .Lcu_ba')
    lab('.Lcu_cxa'); e(f'movq {X_TOFF}(%rsp), %r8'); e('call .Lexact'); e('jmp .Lcu_ca')
    if CU_RSTAIR: lab('.Lcu_qxa'); e(f'movq {X_TOFF}(%rsp), %r8'); e('call .Lexact'); e('jmp .Lcu_qa')
    globals().update(sv)
    lab('.Lcu_hx')                                                 # rdx = G + 64 (s0 + 2): the carry out into that digit (if
    e('vextracti64x2 $3, %zmm21, %xmm0'); e('vpextrq $1, %xmm0, %rax')   # inside the window), then the exact pass over G[0 .. 8 NV)
    e(f'movq {CH_G}(%rsp), %rdi'); e('movq %rdx, %rcx'); e('subq %rdi, %rcx'); e('shrq $6, %rcx')
    e('cmpl %r14d, %ecx'); e('jae .Lcu_hx1')
    e('addq %rax, (%rdx)')
    lab('.Lcu_hx1')
    e('movl %r14d, %esi'); e('call .Lcanon_mem')
    e('jmp .Lcu_hz')


if a.cubic: cublock()
if a.kl:                        # KARA-L1: the Karatsuba band (own sections; .Lsf_tab[1] points at .Lkl_b)
    assert a.cubic
    exec(open(FPRS / 'kl_part.py').read())
    klblock(); klcorr(); klcold(); klro()

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
if a.rsqrt:
    # --rsqrt, cold: a P' lane left above M52 after the fast rounds: the exact pass over Pbuf[ilo .. ilo + 8 NVP) (the stored
    # fast-round digits of all blocks are one fast round of the whole P'; no carry enters lane ilo), then back
    lab('.Lp_xa')
    e(f'movl {RS_NN}(%rsp), %esi'); e(f'subl {RS_H}(%rsp), %esi'); e('addl $10, %esi'); e('shrl $3, %esi')
    e(f'shll $3, %esi'); e(f'addl {RS_H}(%rsp), %esi')                          # t0 = 8 NV + h
    e(f'movl {N_OFF}(%rsp), %eax'); e('subl %esi, %eax'); e('cltq')             # ilo
    e(f'movq {RS_P}(%rsp), %rdi'); e('leaq (%rdi,%rax,8), %rdi')
    e('shrl $3, %esi'); e('incl %esi')                                          # NVP = (t0 >> 3) + 1
    e('call .Lcanon_mem')
    e('jmp .Lp_xa_back')
if a.sqfirst:
    # --sqfirst, cold: an S lane left above M52 after the fast rounds: the exact pass over Sbuf[0 .. 8 NVS) (the stored
    # fast-round digits are one fast round of the whole S; no carry enters its lane 0), then back
    lab('.Lsq_xa')
    e(f'movq {SQ_SB}(%rsp), %rdi'); e(f'movl {SQ_DS}(%rsp), %esi'); e('shrl $3, %esi'); e('incl %esi')   # NVS
    e('call .Lcanon_mem')
    e('jmp .Lsq_xa_back')
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
if a.xs and a.xs_mem:
    # FIX-RSQRT --xs-mem: the same contract (vectors 0 .. NV - 1 of the window registers canonical, NV = r14d; the vectors above
    # unchanged; zmm16 = M52 on return; rax, rcx, rdi, zmm1, zmm2, k5..k7 clobbered; everything else kept) through the one memory
    # pass .Lcanon_mem (same arithmetic): spill the NVMAX window registers and zmm3 (.Lcanon_mem's third temporary) below rsp,
    # .Lcanon_mem over r14d vectors (rsi kept), reload. Cold code: never executed on generic inputs; the forced build runs it.
    SP = 64 * (NVMAX + 1)
    e(f'subq ${SP}, %rsp')
    for v in range(NVMAX):
        e(f'vmovdqu64 {z(VR[v])}, {64 * v}(%rsp)')
    e(f'vmovdqu64 %zmm3, {64 * NVMAX}(%rsp)')
    e('pushq %rsi')
    e('leaq 8(%rsp), %rdi'); e('movl %r14d, %esi')
    e('call .Lcanon_mem')
    e('popq %rsi')
    for v in range(NVMAX):
        e(f'vmovdqu64 {64 * v}(%rsp), {z(VR[v])}')
    e(f'vmovdqu64 {64 * NVMAX}(%rsp), %zmm3')
    e(f'vpbroadcastq .Lm52(%rip), {z(XM)}')
    e(f'addq ${SP}, %rsp')
    e('ret')
elif a.xs:
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
if a.nv_keep:                   # SLIM-RSQRT --nv-keep: the target of every table entry of a width that is not emitted
    e('.section .text.unlikely,"ax",@progbits')
    lab('.Lnv_dead'); e('ud2')
e('.section .rodata')
e('.p2align 2')
lab('.Lst_tab')
e('.long 0')
for N in range(1, NVMAX + 1): e(f'.long .Lst{N} - .Lst_tab' if (N <= a.small or (not a.tabstub and N in KEEP)) else
                                ('.long .Lnv_dead - .Lst_tab' if N not in KEEP else '.long .Lsttab - .Lst_tab'))
for N in range(NVMAX + 1, 21): e('.long .Lstbig - .Lst_tab' if KEEP_BIG else '.long .Lnv_dead - .Lst_tab')   # EXT512: not extended (only NV < sf_min <= 9 indexes it)
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
if a.rsqrt:
    lab('.Lps_tab')
    for v in range(NVMAX): e(f'.long .Lps{v} - .Lps_tab')
if a.pfuse:                     # SMALL-NV: HMAX per NV; the fused MP's row entries by h (h = 0: no row)
    lab('.Lsl_hmax')
    for N in range(1, a.pfuse + 1): e(f'.long {SLHM[N - 1]}')
    for N in range(1, a.pfuse + 1):
        lab(f'.Lslm{N}_tab')
        for h in range(SLHM[N - 1] + 1): e(f'.long .Lslm{N}_r{h + 2} - .Lslm{N}_tab')
if a.sqfirst:
    def sf_tab():
        lab('.Lsf_tab')
        for N in range(0, NV2 + 1):
            if a.cubic and N in (1, 2):         # CUBIC512: "NV" 1 / 2 (below sf_min: never a square-first level) return from the cubic
                e(f'.long {(".Lkl_b" if a.kl else ".Lcu_b") if N == 1 else ".Lcu_q"} - .Lsf_tab'); continue   # level's two S stages (its fake nn)
            if N >= a.sf_min and N not in KEEP: e('.long .Lnv_dead - .Lsf_tab'); continue   # SLIM-RSQRT --nv-keep
            e('.long 0' if N < a.sf_min else (f'.long .Lsf{N} - .Lsf_tab' if N <= NVMAX else '.long .Lsfbig - .Lsf_tab' if N <= 20
                                            else '.long .Lxsfbig - .Lsf_tab'))
    if NV2 == 20: sf_tab()
    else: e('.skip 84')                             # EXT512: the 21-entry table's bytes; the 41-entry table follows .Lidx
    e('.p2align 6')
    lab('.Lsq_idx')                                 # the diagonal: lane 2 j <- lane j
    for v in (0, 0, 1, 1, 2, 2, 3, 3): e(f'.quad {v}')
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
if a.sqfirst and NV2 > 20:      # EXT512: .Lsf_tab (NV 0..NV2) after every original .rodata item (none of them moves)
    e('.section .rodata'); e('.p2align 2'); sf_tab()
if a.cubic:                     # CUBIC512: its constant in its own read-only section
    e(f'.section {CU_RO},"a",@progbits'); e('.p2align 3'); lab('.Lcu_one'); e('.quad 1')
e('.text')
e(f'.size {a.name}, .Lend - {a.name}')
if a.cubic: e(f'.size {a.cubic}, .Lcu_end - {a.cubic}')
e('.section .note.GNU-stack,"",@progbits')

src = '\n'.join(L) + '\n'
if a.maplabels:
    import re
    src = re.sub(r'\.L(?=[a-z_])', 'Lq_', src)
if a.labelprefix:                # IFMA-BUILD: the joint label convention (.Lre_ engine); every label starts .L<lowercase>
    import re
    assert not a.maplabels and not re.search(r'\.L(?![a-z_])', src.replace('.LONG', ''))
    src = re.sub(r'\.L(?=[a-z_])', '.L' + a.labelprefix, src)
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
if a.rsqrt:                     # joint rsqrt (IFMA-BENCH): its own header
    sig = '(X, D_h, G, n, lv, L, Pbuf)' if a.chain else '(X, D_h, G, n, h, nn, mid, Pbuf)'
    hdr = (f'/* Generated by joint/claude_ifma/{"build" if a.labelprefix else "bench"}/gen/gen_engine_rs.py --rsqrt (copy of tools/generate_fp_inv_engine.py);\n'
           f' * do not edit. {a.name}{sig}: the P-first rsqrt level on inv\'s engine (rsqrt_s4x/claude/DESIGN.md section 3).\n'
           f' * Options: U {U}, row groups {GM}, duff {a.duff}, small {a.small}, mpinit {a.mpinit}, chain {a.chain}, force_exact\n'
           f' * {a.force_exact}, force_pexact {a.force_pexact}, force_fix {a.force_fix}, bench {a.bench}, fix {a.fixsym}. */\n')
    if a.pfuse and a.sqfirst:   # INTEG (ifma-v3): both stages' lines under this generator's name
        hdr = hdr.replace('/* Generated by joint/claude_ifma/', '/* Generated by joint/claude_ifma/r2/integ/gen/gen_engine_v3.py (INTEG: SMALL-NV + SQUARE-FORM;\n * from joint/claude_ifma/', 1)
        hdr = hdr.replace(' */\n', f'\n * SMALL-NV: --pfuse {a.pfuse} (NV 1..{a.pfuse}, h <= HMAX {SLHM[:a.pfuse]}: P\' in registers, valignq MP; MP groups {SLGMP[:a.pfuse]}).'
                                    f'\n * Square-first level (S = X_h^2 by a symmetric square, then D_h x S) for NV >= {a.sf_min}; force_sexact {a.force_sexact}. */\n', 1)
    elif a.pfuse:               # SMALL-NV: own header line (without --pfuse the file is IFMA-BUILD's byte for byte)
        hdr = hdr.replace('/* Generated by joint/claude_ifma/', '/* Generated by joint/claude_ifma/r2/small/gen/gen_engine_sn.py (SMALL-NV, from joint/claude_ifma/', 1)
        hdr = hdr.replace(' */\n', f'\n * SMALL-NV: --pfuse {a.pfuse} (NV 1..{a.pfuse}, h <= HMAX {SLHM[:a.pfuse]}: P\' in registers, valignq MP; MP groups {SLGMP[:a.pfuse]}). */\n', 1)
    elif a.sqfirst:
        hdr = hdr.replace('Generated by joint/claude_ifma/', 'Generated by joint/claude_ifma/r2/square/gen/gen_engine_sf.py --sqfirst (SQUARE-FORM;\n * from ', 1)
        hdr = hdr.replace(' */\n', f'\n * Square-first level (S = X_h^2 by a symmetric square, then D_h x S) for NV >= {a.sf_min}; force_sexact {a.force_sexact}. */\n', 1)
    if SQW:                     # STACK (--sq-local): this generator's name and the window allocation
        hdr = hdr.replace('r2/integ/gen/gen_engine_v3.py (INTEG:', 'r2/stack/gen/gen_engine_v3.py --sq-local (STACK, from INTEG:', 1)
        hdr = hdr.replace(' */\n', f'\n * Square-first windows: {a.sq_local} x 64 B, allocated by the square-first level only (subq / addq around its S stage);'
                                    f'\n * engine frame {FRAME} B. */\n', 1)
if a.cubic:                     # CUBIC512: own header line (only with --cubic)
    hdr = hdr.replace(' */\n', f'\n * CUBIC512 (cubic512/gen/gen_engine_c.py --cubic {a.cubic}): appended entry {a.cubic}(X, D_h, G, n, lv, L, Pbuf): the chain'
                                f'\n * with 1 or 2 cubic (third-order) top levels (nn > 2h); .Lsf_tab entries 1 / 2 (unused below sf_min) return from its two'
                                f'\n * S stages; its code and constant are in the sections {CU_TEXT} / {CU_RO} (every original section keeps its size). */\n', 1)
if a.kl:                        # KARA-L1: own header line (only with --kl)
    hdr = hdr.replace(' */\n', f'\n * KARA-L1 (karal1/gen/gen_engine_kl.py --kl, kl_part.py): the cubic band through a recursive Karatsuba middle'
                                f'\n * product in .lsbn_kl_text / .lsbn_kl_ro / .lsbn_kl_cold; .Lsf_tab[1] -> .Lkl_b (plain .Lcu_b below the plan); plan '
                                f'{__import__("os").path.basename(a.kl_plan) or "none"}, correction plan {__import__("os").path.basename(a.kl_cplan) or "none"}, force {a.kl_force}. */\n', 1)
if NV2 > 20:                    # EXT512: own header line
    hdr = hdr.replace(' */\n', f'\n * EXT512 (ext512/gen/gen_engine_x.py --nvmax2 {NV2}): NV 21..{NV2} in up to {(NV2 + 9) // 10} column blocks of 10 vectors'
                                f'\n * (.Lxsfbig / .Lxbig_m, after the sections); NV <= 20 unchanged; '
                                + (f'that path allocates its own F ({XFW} B) at rsp; engine frame unchanged. */\n' if XFW else
                                   f'F {(NV2 + 9) // 10 * 10} vectors at 0, slots +{SH} B. */\n'), 1)
    assert all((x * 205) >> 11 == x // 10 for x in range(0, NV2))
if LIB:                         # the library's engine: its own first lines, then the research header
    hdr = ('/* Generated by tools/generate_fp_rsqrt_engine.py; do not edit. The IFMA level engine of the fp reciprocal square\n'
           ' * root: sbn3f_rsqrt_chain_cu (the chain with 1 or 2 cubic top levels; the only entry the driver sbn3f_rsqrt_qv,\n'
           ' * src/fp/x86_64/rsqrt_qv.S, calls) and sbn3f_rsqrt_chain (the quadratic chain entry; no caller in the library, the\n'
           ' * level sections that follow it in .text serve both entries);\n'
           ' * fix: sbn3f_rsqrt_fix (rsqrt_qv.S). Research kernel G (experiments/results/fpk_2026-10-12/rsqrt_s4x/joint/\n'
           ' * x512g_claude/engine.S, OVH2-RSQRT; this file is that engine.S with the library\'s symbol names and these lines).\n'
           ' * The research generator\'s header (its paths relative to experiments/results/fpk_2026-10-12/rsqrt_s4x/joint):\n'
           ' *\n' + hdr.replace('/* ', ' * ', 1))
if CHECK:
    if open(a.out).read() != hdr + src:
        raise SystemExit(f'{a.out} differs from tools/generate_fp_rsqrt_engine.py\'s output')
    print(f'OK: {a.out} is tools/generate_fp_rsqrt_engine.py\'s output')
else:
    open(a.out, 'w').write(hdr + src)
