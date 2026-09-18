# u52 scratch envelope

Let D=A+B be the actual canonical radix-2^52 input digit count, with A>=B. Every allocation is vector rounded (8 digits). The conversion buffers are exactly `128*(va+vb)` bytes, including one slack vector per input and their combined output allocation.

For recursive temporary storage use S(D)<=5D+64 digits. This is a worst-case bound, not a claim that every input consumes it. The current normal dispatcher recurses at B>=112. The proof uses the weaker B>=96, D>=192 condition, so the raised threshold preserves the same envelope.

| Node | Local temporary bound (digits) | Largest recursive input total |
|---|---:|---:|
| Karatsuba, B>A/2 | 2D/3+8 | 2D/3+1 |
| Toom32, D>4n | 3D/4+16 | D/2+2 |
| Toom33, D>4n | 5D/4+33 | D/2+2 |
| Toom42, D>4n | 2D+57 | D/2+2 |

Each row preserves S(D)<=5D+64 for D>=192. Forced tiny Toom/Karatsuba roots have nonrecursive children; the +64 covers their vector rounding.

Strip-mining reserves round8(4B). Its direct 2B-by-B Toom42 child can be combined with that reservation: <=8B+68 local digits, then an evaluation child of <=B+3 digits. Since the automatic strip condition has A>=2.5B, this also fits 5D+64. Its final remainder has total <=2D/3; with the strip buffer <=8D/7+7 the bound is preserved. Equalized sliver blocks satisfy D>=5B and child total <=3B+4, also within the same envelope. Forced strip mode requires A>=3B.

Actual bit trimming only decreases D. The query uses maximum input digit counts, so it remains safe for high zero limbs. When the maximum short side is below the basecase threshold, automatic mode has no recursive allocation. Private forced-root callers supply the root to `scratch_bytes` so the query retains its appropriate temporary bound.

Native/ASan/TSan guard and allocation tests measure Frame peaks against this bound; they supplement these structural inequalities rather than replace them.
