# Executable spilled integer product

Implementation status, 2026-10-04. `external/spilled_product` runs complete
MUL/SQR through simulated and RAM-backed FileStore storage. Standalone native
OOC multiplication is the current performance target; no full constant run or
SSD operand traffic is needed for its development measurements.

## Query and execution

The query accepts explicit NP/T and Bailey geometry, worker/prime batch and
stripe parameters. It checks the existing exact coefficient bound
`min(input_trunks)*(2^T-1)^2 < product(primes)`, derives the rounded live-row
prefix and quotes all private storage. It performs no search or benchmarking.

Private storage includes control objects, pinned plane indexes and coverage,
output metadata, IO buffers, carry fragments, and a phase pool sized by the
maximum of forwards, middle, and simultaneous inverse/emission requirements.
The published root-bank requirement is returned separately and charged once.
The owning context must add its Team/stacks, caller-held and other fixed state
before applying a total RAM limit. Backing extents are supplied and admitted
by the caller. The `owned_product` adapter can reserve these extents from the
preallocated `ValueStore`, returning all scratch after the operation joins.
Its output becomes an immutable stored handle only on successful sealing.

`planning/product` now selects this external path from a total resident budget.
It charges the p48 shared bank once, plus caller-declared Team/stacks, held
values and other live state. One ordinary backend query owns NP/T selection.
The default bounds the active row by cache capacity, balances prime batches
over a bounded number of IO workers, and derives stripe width from phase
storage. A compact raw operand uses the larger stripe fitting its existing
prime phase. There are at most two cache-derived shapes and one serial lifetime
replay; constrained requests may use four legacy shapes, at most seven complete
quotes in total. No operands, timing search or heap allocations are needed.
One additional forward-stage replay chooses a wider IO stripe from phase-dead
workspace. The forward width is explicit in the plan and independent of the
inverse/CRT stripe; the plane's row-major address mapping is unchanged. Admission
rejects that replay if it increases total storage or read/write traffic.
The chosen plan includes RAM, conservative backing capacity and normal writes;
`run_selected_product` also checks the remaining transport write allowance.
This selector is for operations already lowered to external execution. Resident
promotion and whole-job pending-suffix admission remain separate work.

Spare admitted RAM retains a chunk-aligned prefix of the B-plane family
(the sole family for SQR), in prime order, through forward/middle/inverse.
The prefix is authoritative in RAM and is discarded after its final read.
Neither a forward write nor a middle overwrite reaches disk for these bytes.
Backing reservations are currently conservative: they still include the full
planes, even where payload bytes stay resident.

Execution is:

1. Read verified stored limbs and produce forward planes once; hand their
   current indexes directly to scratch ownership.
2. Run the stored middle pass, replacing B. SQR has one source/forward plane
   family and one row array. Dead A planes are discarded.
3. Read packed inverse stripes, complete inverse/CRT in cache-sized tiles, and
   join exact tile carries into output rows. Separate readers, compute and a
   writer overlap where the three-bank grant and provider permit it.
4. Finalize ordered segment carries early, resolve remaining boundaries, check the product's m61 identity
   and seal the result. Temporary planes are discarded without index writeback.

## Reusable common rows

The cache records its completion state: outer Rows or completed inner Columns.
Both use canonical packed48 and the same per-prime row layout. Consumers must
match that state explicitly; matching byte dimensions alone is insufficient.
The completed state includes the row's global coset and inner Fourier order,
so its consumer skips that operand's inner forward transform.

Completed construction reads a guarded normalized raw source once, produces
one prime of outer rows, performs its inner forwards in RAM and only then
writes that prime's cache plane. It does not write/read an intermediate Rows
family. `build_bytes` quotes raw source/index, one prime and the maximum of
direct outer-forward and unary row work. It is separate from persistent cache
metadata. Cached square can consume the same completed spectrum twice without
running another forward.

The whole-recipe helper can promote an existing cache and all its consumers
together. It admits construction within the recipe's already quoted workspace,
requires no greater cache IO, and compares the builder plus consumers' work.
It changes neither transform geometry nor the output/carry contract. A wider
cache may still supply a compatible consumer's row prefix. Cached data remains
ephemeral and is not an algorithm checkpoint.

Series sharing requires the two already selected products to agree on prime
basis, codec and transform axes. Each consumer keeps its own support, output
capacity, stripe and retention quote; the cache supplies the larger required
Fourier prefix. It does not enlarge the smaller product into the larger one's
geometry. The cache build plus both consumers must not increase total predicted
read/write traffic. Write allowance, backing admission and whole-recipe cost
checks remain in force.

### Compact source with a transient prime

The fixed-geometry selector also replays one compact-source layout for ordinary
MUL: keep the smaller input as normalized raw limbs and materialize only the
other operand's plane family. The raw window is verified and loaded once,
including its original index, shifts and complements. It is zero-padded through
the final codec cell and a 64-byte readable guard. For each prime, the forward
kernel reads those limbs directly and writes a private canonical packed48
plane; the middle pass consumes that plane immediately. There is no copied raw
stripe bank, packed output bank or Store allocation for this transient prime.

The complete query adds raw limbs/indexes and quotes the phase maximum of
the other forward, one transient prime plus direct-forward/middle work, and
inverse/CRT emission. Only then does it assign the remaining RAM to the retained
family prefix. Operand roles may swap; original value handles remain immutable
and the product/carry/scale contracts are unchanged. No arithmetic basis search
is added. The two representations are priced on the same quoted geometry.

For this layout, normal MUL plane traffic becomes `2P-2R` in each direction,
plus source reads, output writes and the existing uncached patch guards.
The compact source's read count has no repeated stripe-boundary chunks. Raw
limbs are currently retained within one product, not shared across subsequent
products. An explicit cached-row recipe converts back to its separate Rows
contract and accounts for that complete footprint. SQR retains its existing
single-family path. Direct resident sources and ordinary/cached paths share
the same outer-transform and modular-row arithmetic.

If the full raw value stops fitting, the same recipe can keep a normalized
prefix and read the missing suffix for each prime. `raw_cache_bytes` is explicit
in the plan; cached cells feed the decoder directly, and only missing input rows
need stripe staging. Source windows, shifts, complements and virtual zeros retain
their original checksum domain. The source index stays pinned, while its IO
buffer is reused only across joined phases. No payload cache survives those
buffer reuses.

The prefix quote includes two suffix staging slots, one transient prime and the
existing middle/final phase maxima. A zero-prefix bound followed by two direct
capacity fills chooses the cache size without searching. Remaining RAM can
retain B. Queries count the prefix once, the suffix once per prime, and exact
physical chunk overlaps at source/stripe edges. The selector compares reads plus
writes and independently enforces the write cap. Cached-row conversion discards
this private representation and reconstructs its ordinary Rows contract.

### Immutable stored outer rows

`RowCachePlan` quotes an ephemeral common-operand cache: canonical packed48
outer-transform rows before the cross twist, pinned checksum expectations,
coverage/dirty bits, control objects and a 1 MiB IO buffer. Its forward work
grant is separate and can share the consumers' phase workspace. Construction
reads the original operand once and records its m61 residue. The cache writes
one plane family, with no extra column-plane or index-file publication.

`use_cached_a` removes A construction from an otherwise identical MUL plan.
The apply grant contains only fresh B planes/indexes and their retention;
the cache remains a separately admitted live object. Middle rows read A
without modifying its backing. An outer-Rows cache repeats A's inner transform
on each apply; a completed-Columns cache skips it.
Full-product validation uses the saved A residue and fresh B residue, so the
raw A value may retire after a successful cache build. External pins still
preserve any required original generation.

The cache instance identifies its operand. Compatibility checks NP/T, C/M,
input capacity, row support/stride and the actual source window under the
current canonical-row/scale contract. Worker count, stripe width and prime
batching are execution choices. Cache metadata and apply workspace must be
disjoint. IO failure cannot publish a partial product; failed cache construction
releases all reservations. Disposal writes no dirty indexes. No durable cache
recovery is implied.

For two consumers at the same geometry and retention, reuse removes one
original-operand read (including its source index) and one common-plane-family
write. The controlled unequal-product gate saves 53,248 read bytes and 589,824
write bytes, with 1,085,440 bytes of persistent cache RAM and 3,051,520 bytes of
apply workspace. It frees and overwrites the old raw operand before both
consumers, proving that they use only the cache. These figures are component
IO evidence, not a timing or constant-computation result.

The current layout rounds the last inner row, retaining the outer TFT. Full
integer inputs and outputs need not fit the arithmetic grant. No complete
coefficient plane or result correction pass is written.

`SourceWindow` now describes a limb slice of an original immutable Region.
The input index remains in its original checksum domain; no rebased temporary
file is written. The selected transform can zero-extend a shorter actual view
within its prequoted operand capacity. A binding overlay changes only source
offsets/lengths and exact IO counters, never geometry, NP/T, roots or workspace.
Original-region/index capacities are enforced by the overlay.

The forward query counts the physical source index plus the covered payload
chunks. If a nonaligned window crosses multiple aligned raw stripes, each
interior boundary rereads one integrity chunk; those extra reads are explicit.
The complete-window product gate reconciles these counts exactly. Finalized
output writes can simultaneously collect a `WordExtent` for later normalization
and truncation, without a separate output scan.

## CRT and segment output

The provider also has an explicitly typed
[cyclic cancellation mode](ooc-cyclic-residual.md), with a required signed-result
descriptor. It validates the recovered residual's magnitude and m61 rather
than applying the full-product fingerprint to a ring representative. Ordinary
MUL/SQR retain the contracts below.

Outer transforms now process four adjacent columns together. Four packed48
cells form exactly three cache lines, emitted with complete non-temporal stores
when aligned. Inverse columns similarly stream complete 64-byte cells after
gathering their group. Store fences complete the synchronous kernel boundary.
Temporary columns have an odd vector stride; all additional scratch is included
in the query and resource model. The packed plane format and IO alignment remain
unchanged. Short/unaligned forward output retains a supported temporal path.

Each existing p48 instance exposes a private CRT adapter, reusing its vector
composition, lag chain and Pack8 code. It follows the existing default mode:
Garner for NP4 and direct CRT for NP5–10. Preparation records whether normalized
coefficients already include each prime's direct-CRT cofactor inverse. Complete
products fold that factor into the middle kernel's existing normalization
multiply, including SQR; cached inputs remain unscaled. The ordinary CRT entry
still accepts unscaled coefficients. Both representations have independent
NP4–10 endpoint/range tests, including T80 and four-word trunks.

A coefficient segment emits `cells*T` bytes with zero incoming carry and an
exact bounded carry-out. Its descriptor retains a 64-byte guard, the following
`0xff` run length, its first non-`0xff` endpoint, and at most two arbitrary
partial chunks. Complete stable-body chunks are immediately final.

Carry resolution adds the preceding carry to the guard, writes a symbolic run
as ones or zeros and adjusts the endpoint. If no endpoint exists, the carry
propagates into the next segment's small carry-out. Long runs are not retained
or reread. Arbitrary arrival needs one fragment slot per chunk-aligned visible
segment, plus a possible final partial-output slot; unaligned segments retain
two. The ordinary product promises increasing segment order within each row,
while rows may interleave. A non-`0xff` endpoint fixes the outgoing carry
independently of the incoming one, allowing subsequent segments to be finalized
and written as complete runs. Only one arbitrary fragment per row can remain
unresolved; earlier all-ones segments need only symbolic descriptors. Cyclic
closure still resolves the remaining boundaries exactly. All storage is
resident and included in the query.
The collector resolves every logical product chunk regardless of segment arrival
order and verifies that bytes beyond the declared arithmetic capacity are zero.
The writer publishes each retained output chunk once.

An m61 accumulator uses eight independent word chains and position weighting.
Input residues are accumulated from already loaded raw stripes; output residues
are accumulated when final chunks are published. There is no extra IO scan.
The arithmetic fingerprint is independent of XXH3 transport integrity.

## Projected output

`ChunkWriter::project` maps one chunk-aligned-start interval of the full logical
stream into a smaller value region. Excluded chunks generate no payload or
index IO. Kept coverage, checksum indexes and `WordExtent` use the new region's
coordinates. The mapping becomes immutable at first dispatch, including a
discarded chunk. Cancellation is still observed on discarded ranges.

`CarryOutput` continues to resolve and fingerprint the complete product before
the projected writer seals. A dropped low prefix can therefore propagate carry
through a long all-ones run into the retained window. Projection does not replace
this carry work with an assumed zero carry. Input/root/transform choices are
unchanged, and no result is reread or rewritten.

For high-normalized nonzero input windows of n and m limbs, the product has H
or H-1 limbs, where H=n+m. To retain its high P limbs, start at
`floor(max(0,H-P-1)/Qw)*Qw`, end at H, with Qw=2048 limbs per integrity chunk.
This stores at most `min(H,P+Qw)` limbs. The one-word allowance covers unknown
top normalization; chunk alignment avoids a second output assembly scheme.
Empty operands produce an empty stored window. The private API also supports
an explicitly specified chunk-aligned-start interval.

The selector can quote a stored-result capacity envelope. Arithmetic and
workspace are quoted once; the owning recipe binds the actual output interval
without another transform query. Output write counts and backing costs use the
stored interval. Current carry fragments and metadata grants remain conservatively
sized for the full stream; this increment saves IO/backing, not private RAM.

## Traffic and failure scope

Let P be the total packed payload of one prime-plane family, I the input
payload/index reads and O the stored result payload/index writes. The ordinary full-
spill route reports:

|Operation|Reads|Writes|
|---|---:|---:|
|MUL|I + 3P + quoted edge guards|3P + O|
|SQR|I + 2P + quoted edge guards|2P + O|

Retaining R payload bytes removes 2R from both columns. Small padded rows
require guard read/modify/write; the query charges only uncached guard reads.
Normal-path totals reconcile exactly with the simulated transport. Failed
submitted writes still consume the transport cap. IO failure leaves immutable
inputs unchanged, invalidates unpublished output and discards private planes.
Kernel math/capacity violations retain the fatal diagnostic contract.

The gate covers CRT endpoints for NP4–10, independent integer carry oracles,
long all-ones runs, reverse segment arrival, output clipping and final carries.
Complete products include non-power-of-two and unequal operand lengths,
all-ones MUL/SQR, multiple stripes and failure cleanup. Allocation watches and
the production audit pass. Small products use exact reference arithmetic;
larger ones use the existing independent product-verification protocol.
These gates are not fresh-call performance sweeps.

Cached SQR can now read immutable outer Rows and write a new product plane
directly. It does not create or transform a second input plane and leaves the
cache available for the next consumer. Excluding cache construction, its reads
are `2P-R` and its writes `P-R+O`; complete padded rows avoid patch guard reads.
R is the retained result-plane prefix. The query and ownership runner distinguish
this mode from ordinary SQR and cached MUL; their retention savings differ.

Integer emission can multiply the exact CRT output by a u64 scalar before
carry resolution. Each row segment and its outgoing carry are scaled together,
and the expected m61 residue changes accordingly. This introduces no new
modular coefficient-capacity requirement, but the enclosing recipe must admit
the scaled integer support. It removes separate stored scaling passes from
rsqrt and fused constant finishing. Tests cover UINT64_MAX scalar carries,
short padded rows, multiple stripes and reuse of the unchanged cache.

The budget-selection gate queries four limb shapes (including unequal lengths
and a point above 2^28) at 4/6/8/12 GiB, with explicit 64 MiB other-live scenario
overhead; it does not allocate their operands. A separate small selected product
runs under an allocation watch and reconciles exact traffic. Retained MUL/SQR,
cross-boundary patches, explicit release, cached corruption and release failures
also pass. The cost profile uses declared analytic rates, not measured device
performance; exact resource admission does not depend on their accuracy.

## Remaining integration

Runtime stored-value ownership, complete series traversal and stored numerical
finishing now execute; see [series](ooc-series.md) and [Newton](ooc-newton.md).
The implementation has [whole-job composition](ooc-job.md) and still needs further linear/CYC/TMP
fusion, efficient SRT lowering, file/async IO and durable commit/recovery. These consumers then
need the 5B/YC acceptance run under declared write
limits. Simulated selected-product success does not close those requirements.
