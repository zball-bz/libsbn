"""The shared IFMA kit of the fp reciprocal and reciprocal square root engines (MULSHARE T1, fp_mulshare_2026-10-17).

tools/generate_fp_inv_engine.py and tools/generate_fp_rsqrt_engine.py emit the same text for these level sections (the
rsqrt generator is a copy of the reciprocal's): Z (.Lz9..0), the segment loops SEG (.Lseg9..0, their Duff row entries, the
group sums .Lsx), RCP (.Lrcp9..0), RC (.Lrc9..0), GZ (.Lgz9..0), UPD (.Lupd9..0) and the cold exact carry pass (.Lexact,
.Lcanon_mem). Each generator records where it emitted those lines (spans of its line list L, its own emission unchanged)
and:
  --kit        leaves them out and refers to the kit's hidden global entries sbn3f_qvk_<name> instead of .L<name>: the same
               instructions (call rel32), now relocated against src/fp/x86_64/qv_kit.S. The tables that point into the
               sections (.Ldtab: the Duff row entries; .Lbig_tab: the section entries) stay in each engine's .rodata,
               next to .Lst_tab, which a level reads right before them (a kit holding its own copies measured no better);
  --kit-emit   prints only the kit (body()); tools/generate_fp_qv_kit.py writes src/fp/x86_64/qv_kit.S from the reciprocal
               generator's output after checking that the rsqrt generator prints the same text.
Inside the kit every reference stays a local label (.L<name>: short jumps and loop branches keep their encodings); each
exported entry gets its global label at the same address."""
import re

PREFIX = 'sbn3f_qvk_'
NAMES = ([f'{s}{v}' for s in ('z', 'seg', 'rcp', 'rc', 'gz', 'upd') for v in range(9, -1, -1)]
         + [f'seg{v}_r{r}' for v in range(9, -1, -1) for r in (1, 2, 3)] + ['sx', 'exact', 'canon_mem'])
_REF = re.compile(r'\.L((?:z|seg|rcp|rc|gz|upd)[0-9]|seg[0-9]_r[1-3]|sx|exact|canon_mem)\b')
CONSTS = [('m52', '0xfffffffffffff'), ('b52', '0x10000000000000'), ('two', '2')]   # the engines' .rodata.cst8 values


HOT_PAD = 4         # the hot sections' size mod 64 (6,788 B): int3 bytes left in their place in each engine, so the engine code
                    # after them (inv's Karatsuba path, rsqrt's ZZ / PS) keeps its placement in the 64-byte lines


def strip(L, hot, cold):
    """remove the kit's lines (hot, cold = (start, end) index pairs into L, hot before cold) from an engine's line list"""
    del L[cold[0]:cold[1]]
    L[hot[0]:hot[1]] = [f'.skip {HOT_PAD}, 0xcc']


def rename(src):
    """an engine's references to the kit: .L<name> -> sbn3f_qvk_<name>; returns the text and the names it uses"""
    used = sorted(set(_REF.findall(src)), key=NAMES.index)
    return _REF.sub(lambda m: PREFIX + m.group(1), src), used


def hidden(used):
    """the .hidden declarations of the kit entries an engine uses (resolved inside the library, no PLT / GOT)"""
    return ''.join(f'.hidden {PREFIX}{n}\n' for n in used)


def body(L, hot, cold):
    """the kit's text from an engine generator's lines: hot = the span of Z .. UPD (.text), cold = .Lexact .. .Lcanon_mem
    (.text.unlikely)"""
    out = []
    def put(span):
        for x in L[span[0]:span[1]]:
            if x.startswith('.L') and x.endswith(':') and x[2:-1] in NAMES: out.append(PREFIX + x[2:-1] + ':')
            out.append(x)
    for n in NAMES:
        out += [f'.globl {PREFIX}{n}', f'.hidden {PREFIX}{n}', f'.type {PREFIX}{n}, @function']
    out += ['.text', '.p2align 6']
    put(hot)
    out += ['.section .text.unlikely,"ax",@progbits']
    put(cold)
    txt = '\n'.join(out) + '\n'
    out = ['.section .rodata.cst8,"aM",@progbits,8', '.p2align 3']
    for n, v in CONSTS:
        if f'.L{n}(' in txt: out += [f'.L{n}:', f'.quad {v}']
    out += ['.section .note.GNU-stack,"",@progbits']
    txt += '\n'.join(out) + '\n'
    labels = set(re.findall(r'^(\.L\w+):', txt, re.M))
    assert set(re.findall(r'\.L\w+', txt)) <= labels, 'the kit refers to a label it does not hold'
    return txt
