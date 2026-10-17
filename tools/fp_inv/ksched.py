"""The reciprocal's schedule enumeration and the Karatsuba level plan (KARA512, Claude, 2026-10-15; used by
tools/fp_inv/kara_emit.py for tools/generate_fp_inv_engine.py). The chains as sbn3f_inv_qv decodes them: n <= 316
qv_chain[n][u], n > 316 qv_top above qv_chain[t][1] (both in src/fp/x86_64/inv_sched.hpp).
plan(h, NV, padmax) is the runtime plan of the engine's Karatsuba path (MODEL.md section 2): V = ceil(NV/2), V' = NV - V,
C = 8 V, P pairs of 2C rows aligned at the top row i = h, then either Lr plain rows at the bottom or one more pair padded
with pad <= padmax zero rows below X_h[0]."""
import re
from pathlib import Path

SCHED = Path(__file__).resolve().parents[2] / 'src/fp/x86_64/inv_sched.hpp'


def read_chain(path=SCHED):
    txt = Path(path).read_text()
    body = txt[txt.index('qv_chain[321][2] = {'):]
    body = body[:body.index('};')]
    rows = re.findall(r'\{0x([0-9a-f]+)ull, 0x([0-9a-f]+)ull\}', body)
    assert len(rows) == 321
    return [(int(a, 16), int(b, 16)) for a, b in rows]


def read_top(path=SCHED):
    txt = Path(path).read_text()
    body = txt[txt.index('qv_top[315] = {'):]
    body = body[:body.index('};')]
    v = [int(x, 16) for x in re.findall(r'0x([0-9a-f]+)ull', body)]
    assert len(v) == 315
    return v


QC, QT = read_chain(), read_top()


def chain_of(n, u):
    """levels lv[0] = n .. lv[L-1] = m0 as the drivers decode them"""
    lv, t = [], n
    if n > 316:
        x = QT[n - 317]
        while True:
            lv.append(t); t = x & 1023; x >>= 10
            if t <= 316: break
        u = 1
    c = QC[t][u]
    lv.append(t)
    for i in range(c & 15):
        t = (t >> 1) + ((c >> (4 + 4 * i)) & 15)
        lv.append(t)
    return lv


def levels(n, u):
    """[(h, nn, l)] from the top (l = 0: the last level, nn = n)"""
    lv = chain_of(n, u)
    return [(lv[i + 1], lv[i], i) for i in range(len(lv) - 1)]


def all_levels(nmin=20, nmax=631):
    """every reachable (n, u, h, nn, l)"""
    for n in range(nmin, nmax + 1):
        for u in (0, 1):
            for h, nn, l in levels(n, u):
                yield n, u, h, nn, l


def nv_of(h, nn):
    return (nn - h + 10) >> 3


def plan(h, nv, padmax):
    V = (nv + 1) // 2; Vp = nv - V; C = 8 * V
    R = h + 1
    P, rem = divmod(R, 2 * C)
    pad, Lr = 0, rem
    if rem and 2 * C - rem <= padmax: P, pad, Lr = P + 1, 2 * C - rem, 0
    return dict(V=V, Vp=Vp, C=C, P=P, pad=pad, Lr=Lr, i0top=R - 2 * C)


def ns_of(n, h, nn, p):
    """S vectors from kmin' = (K - h) rounded down to a multiple of 8 (K = n + 2 - 8 NV) up to K + 8 V' - 1 - Lr"""
    K = n + 2 - 8 * nv_of(h, nn)
    d = (K - h) % 8
    return (h + d + 8 * p['Vp'] - p['Lr'] + 7) // 8, K - h - d
