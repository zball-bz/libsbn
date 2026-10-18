# Stored Newton and constant finishing

The private external providers implement normalized reciprocal, quotient,
small-radicand rsqrt and fused multiply-rsqrt over immutable stored integers.
They reuse the numerical recipes in `algorithms/refinement.hpp`, `rsqrt.cpp`
and `mul_rsqrt.cpp`, including existing native seeds. They are part of the
[OOC implementation](ooc-implementation.md). Small simulated constant chains
now reach certified binary and [fractional radix output](ooc-radix.md).
The [generic job compiler](ooc-job.md) composes these providers, with
[durable algorithm-phase publication](ooc-checkpoint.md) and a
[file-backed reference application](ooc-runner.md).

The provider now uses [certified cyclic residuals](ooc-cyclic-residual.md) for
admitted reciprocal rungs, quotient cancellation and the final multiply-rsqrt
square. Unit or stored-numerator subtraction is fused into emission; negative
windows use a complemented source view. Correction remains LIN/TFT with shared
spectra or raw-operand streaming. General middle windows still need efficient
lowering. Actual 5B-digit pi now
passes independent tail verification on tmpfs; fresh y-cruncher parity remains
unachieved. The small simulated tests below remain correctness evidence, not
performance measurements.

## Input views and numerical contract

Let `B = 2^64`. For a normalized n-word denominator D, the result U has n+1
words, `B^n <= U < 2 B^n`, and `|U - B^(2n)/D| < 3`. It is the reciprocal
of the selected normalized prefix, not an automatically exponent-adjusted
reciprocal of the original `StoredInteger` value.

`SourceWindow` keeps the original Region and checksum identity. Its fields
describe backing length, lower/upper magnitude bounds, virtual low zero words,
a bit shift in -63..63,
and the first/count of virtual words. Bounds are applied before shifting;
discarded neighboring words cannot enter the result. The reader needs two
guard words for shifted input and clears both after use. A source stripe may
read one neighboring physical word; the integrity-chunk overlap is included
in the product's exact read quote.

`WordExtent` records the highest nonzero word while output is finalized.
`normalized_source` uses that metadata to form a shifted high prefix, so
normalization requires neither a separate scan nor a temporary stored value.
The helper requires a positive operand with known extent. Short magnitudes
receive virtual low zero words; equivalent prefixes have a canonical view
identity. Source IO counts exclude entirely virtual stripes and charge only
actual shared physical integrity chunks. No padded file is constructed.

## One reciprocal rung

The steps below describe the retained LIN fallback. The cyclic route performs
the same refinement while publishing the checked residual window directly;
it removes steps 2–3's intermediate residual and absolute-value files. Both
routes preserve the same normalized reciprocal and final error contract.

The rung accepts `3 <= m < n <= 2m-1` and the existing predecessor error bound.
It compiles the following fixed recipe:

1. Select the common m+1-word approximation U's representation for both uses:
   cached outer/completed rows, or a raw source transformed prime by prime.
2. Compute U D, retaining its low n+2 words. Since m >= 3, the unit term
   `B^(n+m)` vanishes modulo `B^(n+2)`. The residual
   `E = U D - B^(n+m)` is therefore recovered by a signed low-residue lift.
3. Check `|E| < 16 B^n` and stream its absolute value. Retain the window
   beginning at m-2 with n-m+3 words; call it rho.
4. Multiply the same U by rho. Retain n-m+1 correction words beginning
   at m+2. This implements the existing two-guard-word correction recipe.
5. Stream `U B^(n-m) - sign(E) * correction`, then enforce the normalized
   output interval. An endpoint clamp, if required, writes a separately
   admitted constant-pattern generation.

The signed lift is a generic bounded-window operation. Its supplied magnitude
bound must make the signed representative unique. The full lifted magnitude
is checked before omitted words are discarded; m61 checks its relationship
to the original residue. Product emission likewise retains full carry/m61
checks and a full logical magnitude certificate even when only an output
window is physically written.

The streaming linear provider accepts stored views or a virtual scalar.
Projected output still resolves the complete signed arithmetic and its
magnitude bound. It does not allocate a stored shift or constant input.

## Ladder, storage and ownership

`query_stored_inverse` selects the largest complete native seed that fits on
the mathematical ladder `n -> floor(n/2)+1`. It compiles all remaining stored
rungs before operand arithmetic. `maximum_native_seed` can cap that choice for
policy or correctness tests; it is not a grid-search control.

Two uses alone do not justify publishing a spectrum. After quoting a reusable
cache, the compiler compares its build plus both applications with two raw
source replays on the same arithmetic shapes. It admits the alternative only
inside the caller's grant after fixed live state, with no extra writes, lower total transfers
and acceptable complete estimated cost. There is no NP/T/geometry search in
this comparison. Source backing envelopes survive the replay independently of
the narrower current windows, including aligned residual seams. Each ordinary
consumer retains its own prime streaming buffer; this is not a hidden raw cache
between calls.

The released cache lifetime is available to those raw consumers. A smaller
realized cached footprint must not become an artificial RAM cap for its
replacement; the final recipe is requoted against the actual caller budget.
Stored-numerator cancellation also supports the joined overlapped final pass.
Its subtractand rows have one buffer per live bank, and widening a stripe
recomputes the exact source-edge read count as well as workspace.

The query includes published engine banks once, plan metadata, the caller's
declared other live state, seed input/output/IO and binding, and the largest
external phase. Each rung quotes its cache lifetime, displaced B retention,
private workspace, immutable-value backing, plane backing and output writes.
Ordinary writes and a separate worst-case bound include the possible clamp.
The whole ladder admits the sum of worst-case writes and the maximum live
backing. The common phase allowance is rounded down to the native seed's arena
alignment before a product spends spare RAM on retained planes. Final alignment
cannot then overrun a previously admitted grant. These aggregate bounds do not
yet prove first-fit physical placement
under arbitrary preexisting fragmentation.

Execution borrows an aligned, prepared, initially unleased arena interval.
The caller assigns it exclusively through return, including gaps between
native leases. Seed input/output/IO occupy disjoint leases; the native service
binds its own separate range. After it unbinds, external rungs reuse the same
region. When selected, the cache survives the residual and correction products; intermediate
values and planes retire at their last use. No arithmetic heap allocation or
arena expansion occurs.

The denominator and supplied rung approximation are preserved. Successful
output owns an immutable stored handle. IO errors/cancellation publish no
returned result and release private values, caches and arena leases. Numerical
certificate/invariant violations remain fatal. Native arithmetic runs to
completion; cancellation is observed at IO/safe boundaries. A failed complete
reciprocal currently restarts from its input, not from a durable internal rung.

## Current validation and next integration

The focused tests cover bidirectional source shifts, bounded neighbors,
multi-stripe guard traffic, perturbed predecessor approximations, non-power-of-
two inverse rungs through 8,191 words and complete ladders with up to seven
external rungs. Independent integer references establish the strict 3-ulp
bound. The native-only seed path, normalized endpoints, allocation watches,
write-cap refusal, cancellation cleanup and lease restoration are checked.
Existing complete series and stored-value regressions also pass.

All operand IO in these tests uses the in-memory test Device. There are no
real SSD operand/swap writes or performance measurements. CYC/TMP and further
linear fusion can replace product lowering without replacing the refinement
mathematics.

## Quotient and modular cancellation

`query_stored_quotient` uses a half-precision reciprocal at `m=floor(n/2)+1`.
Its three products follow the existing quotient recipe: the high numerator
prefix times U forms a coarse quotient C, `C D - A B^m` forms the residual,
and U times the residual window forms the final correction. U's representation
is selected jointly for the first and third products, with cache construction
and all IO charged. The residual product
has its own selected geometry; the recipe does not enlarge every product to
one common basis.

The compiler compares complete LIN and cyclic cancellation recipes before
execution. The cyclic route streams A*B^m into the residual's CRT emission,
checks its unique signed lift and m61, and stores only the required correction
window. It creates neither a full product file nor an absolute-value file.
The third source remains immutable; its index, segment buffers, exact reads
and lifetime are included in the query. The next product consumes an explicit
complemented view when needed.

In the retained LIN fallback, the signed-lift reader subtracts a nonnegative stored view or virtual scalar
at a whole-word offset modulo its guard power. It resolves the sign before
streaming the absolute window. When a lower borrow can change the sign bit,
a high-to-low comparison resolves it; exact equality can require a full scan.
The subsequent stream checks the subtraction/complement identity with m61.
There is no separate full cancellation file. Normal writes are deterministic;
read counts may depend on cancellation depth.

For normalized n-word A,D the raw quotient error is strictly below 3 ulps.
Independently truncating longer normalized source prefixes adds less than
4 ulps; the stored N/D wrapper therefore advertises a strict 7-ulp bound
relative to the original positive ratio. It restores the binary exponent from
both input views, including virtual word shifts for short exact values.
This is approximate quotient finishing, not exact divrem.

## Rsqrt and integer emission fusion

`query_stored_rsqrt(a,n,...)` supports positive u64 a. Its integer result R has
n+1 words with a zero top word and `|R-B^n/sqrt(a)|<3`. A rung uses the existing
`2<=m<n<=2m-1` recurrence. It squares the m-word predecessor, obtains
`E=a R^2-B^(2m)` by a signed lift modulo `B^(m+2)`, and applies
`floor(R*|E|/(2 B^(3m-n)))`. The residual bound is `|E|<2^36 B^m`.
The final division by two is a right-shifted source view. Endpoint clamping
has a separate worst-case output/write allowance.

The approximation's immutable outer Rows now serve both cached SQR and its
correction MUL. Cached SQR reads each source row once and writes fresh product
rows; it does not create a B forward plane. Result-prefix retention saves one
read and one write per retained byte in this mode, rather than the two of an
ordinary three-pass product. Exact byte gates cover both padded short rows
and multi-stripe rows, followed by reuse of the unchanged cache.

A scalar can be folded into integer CRT emission without changing the residue
capacity proof: multiply each zero-incoming-carry segment and its outgoing
carry, then resolve the existing global carry chain. The m61 expectation is
scaled too. The caller must prove complete integer output support. This avoids
a separate stored scalar pass for a and, in the final consumer, its multiplier.
The scalar loop currently uses portable u128 arithmetic in each parallel row;
no performance claim is made for that kernel.

## Fused final multiply-rsqrt and certified output

`query_stored_mul_rsqrt` computes `scalar*X/sqrt(a)` using only a half-precision
stored rsqrt. The scaled raw input must be below `B^(n+1)`. Its final three
products have independent bounded selections, matching the existing RAM recipe:
low square, `X*R`, and `high(floor(X*R/B^m))*|E|`. A scalar on X is folded into
the main product's integer emission. The raw error is below `2^38` ulps;
the returned binary exponent equals X's. No full-precision rsqrt value is
materialized just to multiply it once. This contract excludes input uncertainty,
which the enclosing constant recipe must propagate separately.

`query_fixed_output` / `run_fixed_output` take a nonnegative approximation and
an explicit absolute error in its raw integer units. They check both discarded
guard margins before streaming the fixed-point floor. Whole-word and bit-shifted
guards use the same reader. An ambiguous output cell is a fatal mathematical
certificate failure. The known nonnegative zero-cell exception is retained.
The final value owns `fractional+1` words with exponent `-64*fractional`.
An optional exact nonnegative integer addend is fused into the final word,
with a checked u64 integer-part bound. The Euler terminal uses this for +1
because its Hyperdescent formula produces e-1; no extra stored addition pass
is needed. Older finishing receipts checked that ratio under the label e;
the corrected gate now checks the omitted integer term too.
This operation does not establish a formula's truncation/tail bound; that is a
required input certificate.

Focused gates connect stored Hyperdescent, CommonP2B3 and BinaryBBP results to
quotient finishing. Stored e reaches 17 fractional binary limbs; stored pi
reaches 17 and 65 limbs through fused finishing and fixed output, matching an
independent Machin reference. All stages reuse one explicitly prepared arena
interval. Their test controller sums write allowances and preserves the input
ownership contracts. The [production job runner](ooc-job.md) now compiles these
phases before operands exist and propagates caller-supplied formula/tail bounds.
A unified physical placement proof and durable execution remain work to do.

At 259,525,640 working words (5B decimal-digit scale), allocation-free metadata
queries with a 4 GiB grant and 64 MiB other-live allowance give:

|Finishing phase|Native seed / external seed-refinement rungs|Total RAM bytes|Peak backing bytes|Worst write bytes|
|---|---:|---:|---:|---:|
|Quotient|32,440,706 / 2|4,293,660,232|26,841,251,840|40,425,213,952|
|Fused multiply-rsqrt|32,440,706 / 2|4,293,658,200|19,929,497,600|37,352,898,560|

These are phase queries, with their named stored inputs, not an executed 5B
job or a complete series/radix/IO total. The hypothetical values have metadata
only. No operand allocation, computation or Store IO occurs at this scale.
