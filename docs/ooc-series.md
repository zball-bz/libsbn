# Stored values and generic finite-series merges

Implementation status, 2026-10-02. The private runner now executes complete
series reductions through resident and stored nodes for Hyperdescent,
CommonP2B3 and BinaryBBP. This is not yet a public constant-computation API or
a performance-qualified OOC pi pipeline. The [generic job runner](ooc-job.md)
now composes private finishing and fractional radix providers; durable recovery
and large-run performance qualification remain.

## Value ownership

`ValueStore` binds a caller-prepared metadata span and one admitted backing
interval. It creates no files, maps no pages and performs no ordinary allocation.
The metadata query reserves K stable region descriptors/reference counters and
K+1 sorted free intervals. First-fit reservation and adjacent coalescing cost
O(K), where K counts live recipe objects, not limbs or IO chunks. Release cannot
run out of metadata. Empty values own a descriptor but no backing bytes.

`RegionLease` is a move-only unpublished reservation. A writer or scratch owner
may use its region while the lease stays live. Successful value sealing allows
explicit transfer to a `StoredValue`; this is runtime publication, not durable
commit. Abandoning a reservation returns its extent without destructor IO.
Every new reservation has a new checksum incarnation, even at a reused offset.

`StoredValue` is an immutable reference. Explicit sharing creates a read,
checkpoint or future original-request IO pin. The controlling thread owns
reference changes; a worker/async IO request must join before its pin is dropped
on that controller. Its final reference returns the extent. A reset/moved
handle cannot retain a stale reference accidentally. Generic readers also reject
a descriptor whose incarnation/revision changed after they were opened.

Current backing reservations are contiguous. A live-byte peak is not a proof
that a fragmented free list contains a sufficiently large interval. Reservation
reports the largest available interval on failure; it does not silently grow
the backing or move immutable data. Whole-job planned slot placement/fragmentation
admission is still required. Imported persistent generations and durable pins
are not implemented by these runtime references.

## Exact stored integers and merge recipes

`StoredInteger` denotes an immutable region, a suffix window of its limbs, a
sign and a signed binary exponent. Exact calls can still retain high zero
padding. Optional `WordExtent` metadata records the first/last nonzero limbs
during finalized writes, including carry-resolved chunks arriving out of order.
It is valid only after sealing; collecting it performs no normalization IO.

`limit_stored_integer` uses that metadata to trim known outer zero words and
discard low whole words toward zero. It updates only offsets/count/exponent.
Interior low zero words exposed by truncation may remain in the view; this
changes the representation, not its value, high extent or rounding error.
The original backing and checksum index stay immutable and live until the last
reference retires. The view does not reclaim its discarded physical prefix.

The streaming adder aligns exponents as virtual bit shifts, including shifts
not divisible by 64. It performs exact signed addition/subtraction using bounded
buffers, with carry/borrow and an m61 check accumulated during the data scan.
Output chunks are written once. Equal signs need one input scan. Opposite signs
first compare high windows; the worst cancellation case needs an extra input
scan. The query gives complete private storage and exact output write bytes.
No hidden whole-value allocation or output correction scan is used.

Limited addition uses the resident rule: align at the larger of the lower
input exponent and `high-(P+1)*64`, discard lower whole words toward zero, add,
then keep P words. The limited path requires the existing word-aligned exponent
contract. Exact addition continues to support arbitrary bit shifts.

For left/right values (T,D,U), the stored recipe is:

|Family|Numerator|Denominator|Optional contraction|
|---|---|---|---|
|Hyperdescent|T_L D_R + T_R|D_L D_R|None|
|CommonP2B3|T_L D_R + U_L T_R|D_L D_R|U_L U_R|
|BinaryBBP|T_L D_R + D_L T_R, with explicit binary exponents|D_L D_R|None|

`query_stored_merge` compiles at most five stages, deriving arithmetic capacity
and resource bounds before any multiplication. Each product uses the existing bounded external
selector; there is no new formula-specific split or NP/grid policy. One shared
workspace is the maximum of product and linear stages, with the shared root
bank and caller-declared context state charged throughout. Last-reader counts
derive live backing and a cumulative normal-write bound. Declared input
pins keep their backing charged after the last local read. An undeclared shared
input is rejected before consumption.

The compiler now nominates one common-denominator cache proposal for
`T_L*D_R` and `D_L*D_R`. It uses the ordinary plan for the larger fresh operand
as a shared arithmetic envelope; the smaller consumer zero-extends inside it.
This costs at most one additional complete product quote and no NP/geometry
search. Cached applies, cache construction and the intervening sum have one
explicit lifetime. Cache metadata/IO stay live; phase workspace is reused.
The common raw value can retire after the build, while declared pins stay charged.

The proposal is recomposed at episode scope, including source retirement,
cache backing, live outputs, transient B planes and displaced B retention.
It is chosen only if admitted and no more write-heavy than an admitted separate
recipe, with a modeled time improvement or lower writes within the existing 5%
slowdown allowance. These times remain analytic estimates. Disk/write caps are
enforced after shared producers are composed; otherwise a separate-product
quote could incorrectly veto an admitted cached recipe. This bounded policy
does not prove that all possible geometry/placement combinations are covered.

Execution explicitly takes the two child triples after binding and write-cap
checks. Every owned product reserves its output and scratch extents before IO.
Dead child/intermediate references retire after synchronous readers join.
Successful execution publishes the requested parent triple together. On an IO
error or cancellation, no parent is returned and all taken private state is
discarded; caller-held pins remain readable. A caller requiring restart must
keep its committed inputs/frontier pinned separately. This is not pass recovery.

## Precision tail merge

A nonzero precision selects the resident PSR tail-merge recipe, with
numerator/denominator output (`need==3`). The block T copy keeps P words, its
numerator products keep P+1, and the final numerator/denominator keep P. The
policy caller supplies `precision_block.precision + precision_guard_words`.
Cuts, exact-suffix decisions and finite-subtree rounding continue to come from
`series/precision_policy`; this adapter introduces no alternative policy.
Per-node finite-prefix merging is a separate `StoredRounding::finite` recipe:
all its products and final sums keep P words, and it supports the required U
output. It does not truncate block T using the tail recipe. Both modes can
bind shorter actual inputs inside precompiled capacity/source envelopes.

Product arithmetic is selected before execution at bounded operand capacities.
After a previous stage returns, a pure binding overlay supplies the actual
source window and zero-extends it within that capacity. It changes no roots,
NP/T, transform geometry or workspace offsets and performs no backend search.
The signed sum instantiates only its scalar length/exponent bounds from the
result headers. Its actual output can be smaller than the reserved P+2 words,
so the limited recipe's `normal_write_bytes` is an upper bound; the exact recipe
retains exact normal-write accounting. Both account for the complete physical
backing they actually publish, including chunk/extent padding.

Precision products now publish only a high output window. The selected plan
reserves at most `min(arithmetic_capacity, P+2048)` stored limbs (P+1 is used for
the numerator intermediate). Runtime source lengths instantiate the chunk-aligned
window; complete carry resolution and the full-product m61 check still run.
The stored integer's exponent incorporates the discarded prefix, and final
metadata limiting removes the alignment/normalization allowance. This lowers
temporary backing and writes without an output-copy pass. Private carry and
metadata RAM remain conservatively quoted at full-stream sizes.

## Current evidence and remaining work

The focused gate runs four unequal native subtree ranges and a two-level stored
merge tree for e, exp(-1), Chudnovsky pi and BinaryBBP binary log. Final exact
values/exponents match an independent sequential formula oracle. Queried writes
match the simulated execution and live-backing peaks stay within their bounds.
Separate tests cover
arbitrary exponent shifts and sign combinations across multiple IO buffers,
zero products, empty values, stale readers, metadata/backing exhaustion,
coalescing, surviving pins, pre-take write refusal and in-flight IO/cancellation
cleanup. Allocation watches pass. These are correctness gates without SSD
operand writes, not timing or whole-job qualification.

The precision gate also executes 1/19/21/41/17-block chains for exact e,
limited e, exp(-1), Chudnovsky pi and BinaryBBP log. Both N and D are numerically
identical to resident PSR after reconciling binary exponents, a stronger check
than comparing only their quotient. A sparse-window test covers exposed low
zero words and cancellation. An unaligned source-window product verifies zero
extension, unchanged transform/storage, exact observed read/write counts and
normalization metadata from final carry output.

Projected-output gates cover reverse arrival, an omitted prefix feeding carry
through long all-ones runs, nonaligned segment seams, empty output, kept-range
coverage and cancellation/IO failure. One product stores 1,120 rather than
296,192 logical bytes and submits 294,912 fewer write bytes. A signed CommonP2B3
tail merge at P=5001 matches an independent exact-integer rounding reference;
all three product-result extents shrink, and the write bound drops by 49,152
bytes. Those figures describe the checked fixtures, not a whole-job speed claim.

Cached exact pi/BinaryBBP trees and the P5001 precision merge also pass their
independent references and write/backing checks. The small gate's tighter RAM
grant selects separate products; its larger grant selects reuse. A write cap
below the separate recipe's quote still admits the cached recipe. Cache gates
cover source backing reuse, failed construction, failed fresh-plane writes,
safe reuse after an unaffected-input failure, and incompatible/released handles.

## Complete series runner

`query_series_metadata` sizes caller-owned, stable-address block/node/memo
storage. `compile_series` uses the existing precision cuts, coarse split
selector and native resource summaries. It materializes only external nodes
and their resident frontier, not a graph of all terms. Metadata capacity and
recursion depth are bounded; compile failure invalidates its output and leaves
the exclusively owned metadata discardable.

The compiler selects a complete resident suffix with the existing role-envelope
policy. Its tail pair remains in one RAM region through all those episodes.
Before each external block, a finite tree promotes a complete native subtree
whenever it fits; otherwise it uses the formula's existing logical split and
an admitted stored finite merge. Two lifetime equations choose physical L/R
execution order by peak backing, without changing the ordered formula. Stored
tail merges then connect the blocks. All product plans exist before arithmetic;
runtime headers instantiate only bounded views and scalar alignment.

RAM accounting includes both published engine banks, supplied metadata, a
controller-stack allowance, IO and the maximum phase workspace. Native outputs,
preparation and scratch come from the same retained arena region. Complete
normal-write and live-backing bounds are accumulated over the chosen traversal
before execution; the transport still enforces its cumulative write cap.
Store metadata/free-byte admission and metadata/work/IO separation are checked
before computation. Contiguous-extent fragmentation remains a separate physical
placement limitation: an aggregate bound alone cannot guarantee every allocation
in an already fragmented Store.

`run_series` returns borrowed resident N/D when the entire computation fits,
unless stored output was explicitly requested. An external prefix returns
owning stored values. A resident suffix is published only at the crossing, and
native frontier values can now use the
[native parent handoff](ooc-resident-merges.md#native-subtree-input-handoff).
The last native child may keep its tuple, or one valuable component, in
last-use-ordered arena slots. Native parent products read those windows directly
without copied input arrays; unsupported or inadmissible parents keep stored
values. Loans to genuinely spilled multiplication are still future work. Cancellation
occurs between native operations or at IO boundaries. Failure returns no result
and discards private stored state; durable restart is not implemented here.

The small integration gates force a prefix only to exercise placement cheaply;
the production default has no forced depth. All-resident e/pi have zero Store
IO. Complete mixed runs for e, exp(-1), pi and BinaryBBP match resident N/D and
their write/backing bounds, including cancellation and cap rejection.

Natural query-only pi plans (no operands or arithmetic) currently give:

|Scale / RAM|Blocks before resident suffix|Coarse finite nodes|
|---|---:|---:|
|2B / 4 GiB|3|7|
|2B / 6 GiB|2|2|
|2B / 8 GiB|1|1|
|2B / 12 GiB|0|0|
|5B / 4 GiB|6|28|

The 5B query takes about 0.14 s in the recorded run and quotes 21,371,617,280
live backing bytes and 189,232,185,344 write bytes for the **series phase**.
Finishing and radix are excluded. The frozen YC 2B series transitions were
3/2/1/0 with 21 blocks, while these native plans use 24 blocks, so counts alone
do not establish matching cuts or memory behavior even though the frontier
counts now agree. Bounded merge rebatching reduced the native 12 GiB case's
total quote from 11,944,570,880 to 9,632,346,112 bytes. Canonical growth/cuts,
finishing and end-to-end resource behavior still need comparison.

Each native episode now derives merge execution batches from the workspace
already required by its finite phase. NP/T, transform geometry, value capacities
and consume contracts stay fixed. The compiler probes batch one, uses the
verified per-plane pitch to nominate the largest fitting batch, and otherwise
keeps the verified compact form. It does not rerank prime counts or scan a grid.
The ordinary/compact finite candidates have separate merge descriptors. After
the suffix envelope is fixed, smaller episodes may regain faster batches inside
that same envelope; no new storage or arithmetic identity is introduced.

This implementation still materializes retained product windows before addition.
Common outer rows are now shared; completed columns and arbitrary cache groups
are not. Product/add fusion, physical slot/fragmentation admission,
real file/async IO and durable
recovery remain. Small fixtures deliberately exercise the external provider;
ordinary small integer/series APIs continue using their native resident paths.

The [stored Newton providers](ooc-newton.md) now include quotient, rsqrt and
fused multiply-rsqrt. Small simulated chains produce e and independently
verified binary and [decimal pi](ooc-radix.md) while reusing one arena interval.
The production job compiler uses exposed result backing envelopes to precompile
finishing, then rebinds normalization views without backend reselection.
Efficient CYC/TMP lowering remains pending.
