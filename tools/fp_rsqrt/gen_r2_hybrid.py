#!/usr/bin/env python3
"""Helper of tools/generate_fp_rsqrt_rsp.py (imported; its __main__ block is a research probe only): the hybrid R2 core of
the fp reciprocal square root's scalar paths. Codex's (joint/j12_codex, scalar-v3), as the research kernel G's
source/gen/gen_r2_hybrid.py (experiments/results/fpk_2026-10-12/rsqrt_s4x/joint/x512g_claude), unchanged below.
LGPL-3.0-or-later. FIX-RSQRT copy (Claude, 2026-10-16): emit_core / emit_constants unchanged; added emit_core_v2 /
emit_constants_v2 (R2 v2) and emit_fp_v3 / emit_core_v3 (R2 v3: the FP part hoisted), see their docstrings. F1 hybrid: FP initial approximation, integer residual and cheap integer c2.
New symbol/source, never changes the negative integer-R2 a1 evidence.
"""
import argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent

def emit_core(e,tag='.Lr2hy',scratch0='0(%rsp)',scratch1='8(%rsp)'):
    """Same embedding ABI as gen_r2.emit_core: rbx out, r15 top3, rbp untouched, 2 scratch words."""
    def block(s):
        for x in s.strip().splitlines():e(x.strip())
    block(f'''
# Initial FP approximation is the existing donor's vrsqrt14 + two FMA Newton steps.
movq 16(%r15), %r14
vxorpd %xmm0, %xmm0, %xmm0
vcvtusi2sdq %r14, %xmm0, %xmm0
vrsqrt14sd %xmm0, %xmm0, %xmm1
vmulsd {tag}_half(%rip), %xmm0, %xmm0
vmulsd %xmm1, %xmm0, %xmm2
vmovsd {tag}_half(%rip), %xmm3
vfnmadd231sd %xmm2, %xmm1, %xmm3
vfmadd231sd %xmm3, %xmm1, %xmm1
vmulsd %xmm1, %xmm0, %xmm2
vmovsd {tag}_half(%rip), %xmm3
vfnmadd231sd %xmm2, %xmm1, %xmm3
vfmadd231sd %xmm3, %xmm1, %xmm1
vmulsd {tag}_p95(%rip), %xmm1, %xmm1
vcvttsd2usi %xmm1, %r12
movabsq $0x8000000000000000, %rax
cmpq %rax, %r12
cmovbq %rax, %r12
# y^2 and the retained top3 product. Omitted terms total <3 B^2 (R2 effect <3*2^-62).
movq %r12, %rdx
mulxq %r12, %r13, %r14
movq 0(%r15), %rdx
mulxq %r14, %rax, %r8
movq 8(%r15), %rdx
mulxq %r13, %rax, %r9
xorl %r10d, %r10d
xorl %edi, %edi
addq %r9, %r8
adcq $0, %r10
mulxq %r14, %rax, %rcx
addq %rax, %r8
adcq %rcx, %r10
adcq $0, %rdi
movq 16(%r15), %rdx
mulxq %r13, %rax, %rcx
addq %rax, %r8
adcq %rcx, %r10
adcq $0, %rdi
mulxq %r14, %rax, %rcx
addq %rax, %r10
adcq %rcx, %rdi
# Rpp=floor((2^318-P_retained)/2^176), approximate e*2^142.
xorl %ecx, %ecx
subq %r8, %rcx
movl $0, %r8d
sbbq %r10, %r8
movabsq $0x4000000000000000, %r9
sbbq %rdi, %r9
shrdq $48, %r8, %rcx
shrdq $48, %r9, %r8
# v=abs(round(e*2^80)); exact half-up integer rounding, so v fits 32 bits.
movq %rcx, %r13
movq %r8, %r14
movabsq $0x2000000000000000, %rax
addq %rax, %r13
adcq $0, %r14
shrdq $62, %r14, %r13
movq %r13, %r14
sarq $63, %r14
xorq %r14, %r13
subq %r14, %r13
# c1_G=floor(y*Rpp/2^62), G=16.
movq %r12, %rdx
mulxq %rcx, %r10, %r11
shrdq $62, %r11, %r10
shrq $62, %r11
movq %r8, %rax
imulq %r12
addq %r8, %rdx
addq %rax, %rax
adcq %rdx, %rdx
addq %rax, %rax
adcq %rdx, %rdx
addq %r10, %rax
adcq %r11, %rdx
movq %rax, {scratch0}
movq %rdx, {scratch1}
# c2_G=3*floor(y*v^2/2^82); only one 64-bit square and one 128-bit product.
imulq %r13, %r13
movq %r12, %rdx
mulxq %r13, %rax, %rcx
shrq $18, %rcx
leaq (%rcx,%rcx,2), %rax
movq {scratch0}, %r8
movq {scratch1}, %r9
addq %rax, %r8
adcq $0, %r9
shrdq $16, %r9, %r8
sarq $16, %r9
# Lower result: y*2^65+correction-1; endpoint clamps establish leading one.
movq %r12, %r10
addq %r10, %r10
movq %r9, %r11
sarq $63, %r11
addq %r9, %r10
adcq $1, %r11
movq %r8, %rax
subq $1, %rax
sbbq $0, %r10
sbbq $0, %r11
cmpq $1, %r11
je {tag}_store
jb {tag}_low
movq $-1, %rax
movq $-1, %r10
movl $1, %r11d
jmp {tag}_store
{tag}_low:
xorl %eax, %eax
xorl %r10d, %r10d
movl $1, %r11d
{tag}_store:
movq %rax, 0(%rbx)
movq %r10, 8(%rbx)
movq %r11, 16(%rbx)
''')

def emit_core_v2(e,tag='.Lr2hy',top=None):
    """FIX-RSQRT (Claude, 2026-10-16) R2 v2, same embedding ABI as emit_core (rbx out, r15 top3, rbp untouched; outputs rax, r10,
    r11 at {tag}_store, then the three stores), no scratch words. Changes against emit_core:
      * top: the register holding top3[2] (the top limb) when the core starts (no store-forward wait on 16(%r15)); 'r14': no move;
      * FP initial value: vrsqrt14 y0 (|rel| < 2^-14), e = 1 - d y0^2 (one FMA), then one step of order 4 with the 2^95 scale
        folded in: Y = Y0 + Y0 e / 2 + Y0 e^2 / 2 (3/4 + 5/8 e), Y0 = y0 2^95 (Estrin form: e + 10 cycles instead of two Newton
        steps and the scale multiply); |Y / Y* - 1| < 3.56 u, |e0| < 7.2 u < (9/8) 2^-49, the precondition of the integer part
        (proof: fix_rsqrt/proof/r2v2_bound.py);
      * integer part bit for bit emit_core's for a given Y, with two removals: the abs of v (only v^2 is used: imul v, v of the
        signed v has the same low 64 bits) and the c1 scratch store / reload (c1 stays in r8:r9)."""
    def block(s):
        for x in s.strip().splitlines():e(x.strip())
    first = '' if top == 'r14' else (f'movq %{top}, %r14' if top else 'movq 16(%r15), %r14')
    block('# FIX-RSQRT R2 v2: vrsqrt14 + one order-4 step (scale 2^95 folded in); integer part as v1 without abs(v) and the c1 spill.\n'
          + first + '\n' + f'''
vxorpd %xmm0, %xmm0, %xmm0
vcvtusi2sdq %r14, %xmm0, %xmm0
vrsqrt14sd %xmm0, %xmm0, %xmm1
vmulsd %xmm1, %xmm1, %xmm2
vmovsd {tag}_one(%rip), %xmm3
vfnmadd231sd %xmm2, %xmm0, %xmm3
vmulsd {tag}_p94(%rip), %xmm1, %xmm4
vmulsd {tag}_p95(%rip), %xmm1, %xmm1
vmovsd {tag}_c34(%rip), %xmm5
vfmadd231sd {tag}_c58(%rip), %xmm3, %xmm5
vmulsd %xmm3, %xmm4, %xmm4
vaddsd %xmm4, %xmm1, %xmm1
vmulsd %xmm3, %xmm4, %xmm4
vfmadd231sd %xmm5, %xmm4, %xmm1
vcvttsd2usi %xmm1, %r12
movabsq $0x8000000000000000, %rax
cmpq %rax, %r12
cmovbq %rax, %r12
# y^2 and the retained top3 product. Omitted terms total <3 B^2 (R2 effect <3*2^-62).
movq %r12, %rdx
mulxq %r12, %r13, %r14
movq 0(%r15), %rdx
mulxq %r14, %rax, %r8
movq 8(%r15), %rdx
mulxq %r13, %rax, %r9
xorl %r10d, %r10d
xorl %edi, %edi
addq %r9, %r8
adcq $0, %r10
mulxq %r14, %rax, %rcx
addq %rax, %r8
adcq %rcx, %r10
adcq $0, %rdi
movq 16(%r15), %rdx
mulxq %r13, %rax, %rcx
addq %rax, %r8
adcq %rcx, %r10
adcq $0, %rdi
mulxq %r14, %rax, %rcx
addq %rax, %r10
adcq %rcx, %rdi
# Rpp=floor((2^318-P_retained)/2^176), approximate e*2^142.
xorl %ecx, %ecx
subq %r8, %rcx
movl $0, %r8d
sbbq %r10, %r8
movabsq $0x4000000000000000, %r9
sbbq %rdi, %r9
shrdq $48, %r8, %rcx
shrdq $48, %r9, %r8
# v=round(e*2^80), signed (exact half-up rounding, |v| < 2^32); only v^2 is used, so no abs.
movq %rcx, %r13
movq %r8, %r14
movabsq $0x2000000000000000, %rax
addq %rax, %r13
adcq $0, %r14
shrdq $62, %r14, %r13
# c1_G=floor(y*Rpp/2^62), G=16; kept in r8:r9.
movq %r12, %rdx
mulxq %rcx, %r10, %r11
shrdq $62, %r11, %r10
shrq $62, %r11
movq %r8, %rax
imulq %r12
addq %r8, %rdx
addq %rax, %rax
adcq %rdx, %rdx
addq %rax, %rax
adcq %rdx, %rdx
addq %r10, %rax
adcq %r11, %rdx
movq %rax, %r8
movq %rdx, %r9
# c2_G=3*floor(y*v^2/2^82); only one 64-bit square and one 128-bit product.
imulq %r13, %r13
movq %r12, %rdx
mulxq %r13, %rax, %rcx
shrq $18, %rcx
leaq (%rcx,%rcx,2), %rax
addq %rax, %r8
adcq $0, %r9
shrdq $16, %r9, %r8
sarq $16, %r9
# Lower result: y*2^65+correction-1; endpoint clamps establish leading one.
movq %r12, %r10
addq %r10, %r10
movq %r9, %r11
sarq $63, %r11
addq %r9, %r10
adcq $1, %r11
movq %r8, %rax
subq $1, %rax
sbbq $0, %r10
sbbq $0, %r11
cmpq $1, %r11
je {tag}_store
jb {tag}_low
movq $-1, %rax
movq $-1, %r10
movl $1, %r11d
jmp {tag}_store
{tag}_low:
xorl %eax, %eax
xorl %r10d, %r10d
movl $1, %r11d
{tag}_store:
movq %rax, 0(%rbx)
movq %r10, 8(%rbx)
movq %r11, 16(%rbx)
''')


def emit_constants_v2(e,tag='.Lr2hy'):
    """FIX-RSQRT: the constants of emit_core_v2 (1, 2^95, 2^94, 3/4, 5/8)"""
    e('.section .rodata\n.p2align 3\n'+tag+'_one:\n.quad 0x3ff0000000000000\n'+tag+'_p95:\n.quad 0x45e0000000000000\n'+tag+'_p94:\n.quad 0x45d0000000000000\n'
      +tag+'_c34:\n.quad 0x3fe8000000000000\n'+tag+'_c58:\n.quad 0x3fe4000000000000')


def emit_fp_v3(e,tag='.Lr2hy'):
    """FIX-RSQRT (Claude, 2026-10-16) R2 v3, part 1: the FP initial value of emit_core_v2, hoisted to an entry point. In: rsi the
    radicand's limbs, r8 nb (>= 1), r10 c (only its low 6 bits are read: shlx counts), the top limb x[nb-1] with c leading zeros
    after the shift. Out: xmm1 = y1, the double R2's integer part converts (vcvttsd2usi); H = top3[2] = (x[nb-1] << c) |
    (x[nb-2] >> (64 - c)) (the decode's top limb; mulx by 2^c: 0 when c = 0; nb = 1: x[nb-1] << c) is computed here exactly as
    the decode does. Clobbers rax, rdx, xmm0, xmm2..xmm5, flags. Same FP instructions and rounding as emit_core_v2, so its bound
    (proof/r2v2_bound.py) holds unchanged. Between this part and the core (emit_core_v3) no instruction may write xmm1."""
    def block(s):
        for x in s.strip().splitlines():e(x.strip())
    block(f'''
# FIX-RSQRT R2 v3 FP part begin (hoisted: xmm1 = y1 for the R2 core)
shlxq %r10, -8(%rsi,%r8,8), %rax
cmpq $1, %r8
je {tag}_h1
movl $1, %edx
shlxq %r10, %rdx, %rdx
mulxq -16(%rsi,%r8,8), %rdx, %rdx
orq %rdx, %rax
{tag}_h1:
vxorpd %xmm0, %xmm0, %xmm0
vcvtusi2sdq %rax, %xmm0, %xmm0
vrsqrt14sd %xmm0, %xmm0, %xmm1
vmulsd %xmm1, %xmm1, %xmm2
vmovsd {tag}_one(%rip), %xmm3
vfnmadd231sd %xmm2, %xmm0, %xmm3
vmulsd {tag}_p94(%rip), %xmm1, %xmm4
vmulsd {tag}_p95(%rip), %xmm1, %xmm1
vmovsd {tag}_c34(%rip), %xmm5
vfmadd231sd {tag}_c58(%rip), %xmm3, %xmm5
vmulsd %xmm3, %xmm4, %xmm4
vaddsd %xmm4, %xmm1, %xmm1
vmulsd %xmm3, %xmm4, %xmm4
vfmadd231sd %xmm5, %xmm4, %xmm1
# FIX-RSQRT R2 v3 FP part end
''')


def emit_core_v3(e,tag='.Lr2hy'):
    """FIX-RSQRT R2 v3, part 2: emit_core_v2 without its FP part (from vcvttsd2usi on): xmm1 = y1 comes from emit_fp_v3 at the
    entry point. Same embedding ABI and outputs as emit_core_v2."""
    a = []
    emit_core_v2(a.append, tag, top='r14')
    i = a.index('vcvttsd2usi %xmm1, %r12')
    assert a[i - 1] == 'vfmadd231sd %xmm5, %xmm4, %xmm1' and a.count('vrsqrt14sd %xmm0, %xmm0, %xmm1') == 1 and a.index('vrsqrt14sd %xmm0, %xmm0, %xmm1') < i
    e('# FIX-RSQRT R2 v3 core: xmm1 = y1 from the entry point (emit_fp_v3); integer part as R2 v2.')
    for x in a[i:]:
        e(x)


def emit_constants(e,tag='.Lr2hy'):
    e('.section .rodata\n.p2align 3\n'+tag+'_half:\n.quad 0x3fe0000000000000\n'+tag+'_p95:\n.quad 0x45e0000000000000')

def generate(entry='fpkrs2_r2_hybrid'):
    assert entry.replace('_','').isalnum();tag='.L'+entry;lines=[];e=lines.append
    e('/* F1 hybrid generated by gen_r2_hybrid.py. LGPL-3.0-or-later. FP initial approximation follows read-only tools/generate_fp_sqrt.py; donor notice src/core/x86_64/scalar-donor-notice.txt. */')
    e('.text\n.p2align 5\n.globl '+entry+'\n.hidden '+entry+'\n.type '+entry+',@function\n'+entry+':')
    for r in ('rbx','r12','r13','r14','r15'):e('pushq %'+r)
    e('subq $16,%rsp\nmovq %rdi,%rbx\nmovq %rsi,%r15');emit_core(e,tag)
    e('addq $16,%rsp')
    for r in ('r15','r14','r13','r12','rbx'):e('popq %'+r)
    e('ret\n.size '+entry+',.-'+entry);emit_constants(e,tag);e('.section .note.GNU-stack,"",@progbits')
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',nargs='?',type=Path,default=HERE/'r2_hybrid.S');ap.add_argument('--entry',default='fpkrs2_r2_hybrid');a=ap.parse_args();a.output.write_text(generate(a.entry));print(a.output)
if __name__=='__main__':main()
