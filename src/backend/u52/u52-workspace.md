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

Rectangular automatic products with a side above 4096 limbs and a nonzero
short side at most 1024 limbs convert the long side in strips of 4096 limbs.
The same envelope is evaluated at `(4096, short_limbs)`; one additional,
64-byte-rounded `short_limbs`-word span retains the preceding product's
overlap. Each strip writes its result directly to the caller output, then
adds that overlap. Its carry cannot exceed the strip: the accumulated prefix
is a product of an integer below `B^(offset+strip)` with one below `B^short`.
This gives scratch independent of the long side (supported through 2^31
limbs). Forced roots keep the original envelope and 2^20-limb capacity.

Streaming does not consume all inputs before its first output write. Public
bindings already require disjoint values. Product programs with a planned
side above 4096 consequently do not advertise `program_consume_inputs`, even
when only a shortened runtime input would enter the streaming path. Bounded
runtime shapes remain within the planned scratch envelope.
