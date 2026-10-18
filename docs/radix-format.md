# Radix formatting

The public interface remains `sbn3_format_query/bind/execute/unbind`.
`options.repeated` is a cost hint: a binding is reusable with either setting.
Fresh single-worker conversion uses a request-scaled dyadic ladder once its
penultimate power reaches the cached FFT band, so that preparation and
conversion can share that transform. Smaller trees use the immutable prefix
and local middle products. This regime test uses the base, requested length
and the backend's common/fresh crossover bounds. Repeatedly
halve the requested fragment count, rounding upward, until the seed fits a
leaf group; internal split powers are that seed times powers of two. Full
subtrees share one power per level and the rightmost tail skips unused levels.
The immutable power bank supplies constants without constraining the splits.
Fresh parallel conversion uses the scaled ladder once nodes reach the team
band; repeated conversion retains the regular grid. A short tail after a
published power can use that prefix and a native thin middle product. There
is no second public API and no table of preferred input lengths.

Each large balanced level uses one left power and one cached product envelope
for the full nodes and ragged tail in the serial frontier. Live input support
and the earliest requested window jointly determine the envelope. A cache
whose linear square is supported produces the next power, keeping its forward
representation for conversion. This coupling currently uses single-worker
squares; a parallel square retains its original worker recipe. Construction
powers live in setup scratch; only powers requested by the tree survive bind.
Short direct formatting has no tree plan. Below 32 input limbs it uses the
existing multiply-by-u64 path. Otherwise it compares a bound on radix-2^51
vector iterations with `4000 + 15 * ceil(digits/64)`, including the input's
known low zeros. The bound uses only shape and base. The direct IFMA path
extracts words in the largest radix power below 2^50 and postpones carries
until one final pass; SRT handles larger work volumes.

The IFMA basecase stores redundant binary digits below 2*B, B=2^51. With
D<=B/2, each step writes `lo(x_i*D) + hi(x_(i-1)*D)`, still below 2*B.
The identity `D*V = q*B^N + V_next` holds exactly. After the final step, a
binary carry pass supplies the remaining integer carry, followed by reverse
normalization of the base-D output words. This is exact integer arithmetic.

SRT leaf groups contain at most sixteen 64-digit fragments. Equal-shaped
groups are batched across eight IFMA lanes and split approximately in half,
using published short powers. Four independent accumulation chains hide
multiply latency. Two low guard columns bound an omitted carry to at most
one retained unit; sixteen precision guard bits keep the accumulated tree
error below one final overlap unit. Internal numeric-word seams are resolved
before byte emission, leaving one outer seam per group.

Groups use at most 32 input limbs until the request provides two vector
batches of maximal groups per worker; then the cap is 64 limbs. This keeps
small calls from paying for inactive lanes. Very short batches emit across
the eight words of a fragment instead of transposing inactive group lanes.

## Parallel storage and carry repair

Child capacities partition one fraction slab, including the actual leaf guard
words. A parent finishes all source reads before publishing its right child
into the overlapping low slot. Its left child is a surviving high view.
Serial subtrees visit right first, then left, in the same partitioned slab.
They reuse one product episode and no longer retain a separate right-fraction
array at each recursive depth. Worker subtrees own disjoint scratch slices.
Space does not accumulate one full parent array per parallel recursion depth.
Cache-sized staged windows keep the admitted serial FFT kernel and distribute
independent nodes across workers. Assigning two workers to such a node must
not silently replace its cached window with an ordinary full product.

The work reservation is the maximum of setup, exact terminal repair, frontier
work, and each stage's live prefix plus its episode and pool. Stage extents
are never summed across disjoint lifetimes. A stage may borrow the unused
digit output for its numeric result when its backing span and alignment fit.
Ordinary product stages keep only their workspace in the arena; ring stages
also include any output-backed transpose in the loan. All numeric reads finish before any
frontier emits digits. Deep NTT stages use a single spare-prime batch;
smaller stages retain the backend's calibrated batch policy.

A wrapped split computes `P = L + beta^r H`, with `beta = 2^64`, using a cyclic
product modulo `beta^r - 1`. The selected period leaves a gap below the wanted
window. A nonzero residue in that gap proves that no wrapped carry reached
the window. A zero gap has two meanings, so an exact low-product witness must
settle it. This witness is part of the exact contract, not a probability check.

For staged products, an ambiguous node leaves its source intact. After the
stage joins, the cyclic bindings release their pool. Exact witnesses then use
that same pool, one at a time. The stage regenerates only ambiguous residues,
applies the saved carry decisions, and finally allows dependent work to run.
Thus the pool reserves the maximum of cyclic and witness storage. The ordinary
path neither recomputes residues nor enters the witness phase. Serial frontier
products continue to reuse their local cyclic/witness scratch directly.

The wide CT FFT can emit a guarded high window without first closing a
canonical residue. At word `origin-1` its signed modular error is less than
`E=2^40`. If the observed guard lies in `[E, 2^64-1-E)`, the error cannot change
word `origin` and the exact guard is nonzero. The wrapped high part ends below
that guard, so its carry cannot have reached the window either. The extracted
window is then exact. Otherwise the complete residue and witness settle it;
this is a deterministic guard test, not a probabilistic omission of repair.
The fallback's full reservation remains in the scratch quote. This path uses
the established minus-ring numerical domain; it does not reinterpret that
spectrum as a negacyclic one or expand the floating-point envelope.

An exact carry witness computes one product word by streaming bounded
operand blocks. It does not materialize the full low product. The same
primitive serves serial windows, staged rings and the final EXACT digit tie.
Bounded witnesses and small power seeds use the existing local integer
or FFT multiplication recipe. A witness inside a parallel tree still uses
that serial recipe when it fits the local capability; dormant parallel
workspace is not reserved merely because the enclosing team is wide. Their full scratch is queried without creating
dormant service plans or preparing tables on the normal conversion path.

EXACT fraction formatting also reserves a final deterministic digit-boundary
repair. Its lifetime follows the entire tree and overlaps the retired ring
pool. Removing the per-stage witness sum does not remove this separate bound.

## Certified interval output

ENCLOSED accepts an unknown real in `[X, X + 2^exponent2)`. It returns a certified
prefix, which need not be the longest prefix mathematically determined by that
interval. It does not reserve the final full exact digit-boundary product.

Let `K = 64 F + 8`, and let the reconciled tree output represent integer `S`.
The tree guarantees `S <= floor(X b^K) <= S + 1`. For `D` requested digits,
write `O = 64 F - D` and `U = floor(S / b^O)`. The existing precision cap makes
the input interval width less than one quarter of a unit at `D + 8` digits.
The possible integer there is consequently in `U..U+1` when `O > 0`, or
`U..U+2` when `O = 0`. If the eight-digit guard cannot overflow under that
uncertainty, all `D` digits are certified. Otherwise the output stops at the
common prefix of the produced digits and their successor. Known terminating
dyadics are settled by their trailing-zero count before this test.

This preserves the interval contract without a large exact repair. EXACT
still returns every requested digit and uses its original deterministic
boundary resolution. Storage comparisons must identify which contract was
queried; a smaller ENCLOSED quote is not an EXACT quote.

## Plan construction and shared data

Only live classes, stages and powers are packed into bound control storage.
The larger query-time directory stays on the control stack and never becomes
a per-operation arena reservation. Candidate ring shapes are priced from
backend geometry/resource quotes. Those quotes carry no scale constants,
identities or bindable plans. The winner is materialized through the ordinary
producer, future-cache descriptor and consumer queries, and bind replays it.
The geometry gate compares all resource and cost fields against that full
sequence. This changes planning work, not arithmetic admission or the chosen
candidate set.

Published FFT roots, p48 root prefixes and radix constants are counted once
per process alongside private operation storage. Fresh timing still includes
query, power construction, spectrum preparation, execution and unbind. No
input-dependent state survives a fresh call.


Short exact word extraction can reduce its working precision as output digits
are emitted. It computes eight extra radix guard digits. At each step it retains
an upper bound for the bits needed by every remaining digit, plus
`ceil(log2(rounds)) + 1` binary guard bits. Subtracting
`floor(log2(base^K))` per emitted K-digit word preserves that upper bound.
Each discarded tail changes the final guard integer by less than
`2^(-guard_bits)`; the sum is below one. A guard other than eight maximal digits
therefore certifies the requested prefix. An ambiguous guard reruns exact word
extraction from the preserved input. Long output expansions and the smallest
inputs keep ordinary exact extraction. Both choices are recorded in the plan.
