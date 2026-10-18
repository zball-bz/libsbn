# OOC implementation objective and milestones

2026-10-02. This implementation objective is now active in the thread goal.
The user replaced the earlier research objective; its historical receipts do
not constitute a universal near-optimality result.

## Goal text

Implement and optimize generic out-of-core constant computation in libsbn_v3,
using the recursive budget planner in `docs/ooc-planner.md`. Support the existing
Hyperdescent, CommonP2B3 and BinaryBBP formula abstractions, the necessary Newton
finishing operations, and radix output through one bounded-memory executor.
Preserve caller ownership and allocation-free resident arithmetic. Implement
Store integrity, failure cleanup and value/algorithm-boundary recovery.

Deliver a correctly verified 5B-digit pi computation on the current machine
with at least 4 GiB total resident RAM, aiming for y-cruncher-comparable complete
fresh time and IO efficiency. Use the recorded 2B / 4,6,8,12 GiB matrix as a
behavioral reference; compare native resource transitions using libsbn's own
kernel footprints. Report actual memory, reads, writes and temporary disk
alongside time. Prepared timing or query-only predictions do not establish
performance completion.

Prioritize reducing materializations and transferred bytes. Use modeled,
simulated and RAM-backed IO first; any real-device test needs a named question,
predicted traffic and a separately declared cumulative write allowance.
Production selection uses bounded heuristics, not exhaustive search. Research
is limited to concrete gaps exposed by executable integration. Do not wait for
a universal near-optimality certificate across arbitrary resource ratios.

## Delivery order

|Milestone|Deliverable|Completion evidence|State|
|---|---|---|---|
|M1|Native resource-summary compiler and complete generic subtree/precision-block grants|Deterministic cuts and selected tokens; full fixed/shared/value/workspace admission; resident replay|Generic job compilation/object placement and actual 2B/5B resource admission pass; broader RAM-transition qualification remains|
|M2|One external product path and bounded coordinator|Existing numerical kernels execute through simulated Store; modeled and observed pass bytes agree; consume/join and carry contracts hold|Complete spilled MUL/SQR, bounded retention and promoted episodes execute in the full job; CYC/TMP and further fusion remain efficiency work|
|M3|Generic series, division/rsqrt and radix composition|End-to-end constants traverse the same executor at resident/external boundaries; no per-constant allocator|Generic composition, physical files and independently verified 5B pi pass; optimization remains|
|M4|Durable Store and recovery integration|XXH3-64, commit/resolve, cancellation and fault injection; restart from valid value/algorithm frontiers|Protected imports, atomic phase manifests, UNKNOWN resolution and tmpfs process-crash recovery pass; VM/power-loss and large-device qualification remain|
|M5|Whole-job optimization and qualification|Fresh comparisons against YC, non-power-of-two coverage and RAM transitions; bounded 5B acceptance|First verified 5B/4GiB tmpfs execution passes; YC parity, broader coverage and device qualification remain|

Each slice should produce runnable code and one focused gate. Expand testing
only for a changed contract, a failure, or an integration milestone. Keep
experiments/results in the private research checkout, with no production build
dependency on it. Do not publish unused future API declarations.

The first resource test executes 36 native subtree runs (three families, two
worker widths, three ranges, two decompositions), compares exact results across
decompositions and checks independent modular residues. Query, preparation and
execution pass allocation watches. These are correctness gates, not performance
measurements or external-execution acceptance.

The second increment shares the existing precision policy and consuming merger
between resident PSR and `planning/precision`. Complete episode quotes include
the persistent tail, block/merge alias lanes, metadata and phase-shared kernel
workspace. A single exact root avoids intermediate value lanes. The focused
gate runs 24 complete resident chains (760 block episodes) across exact/limited
Hyperdescent, CommonP2B3 and BinaryBBP cases, including signed terms, with
independent ratio-error checks and allocation watches. The existing precision
planner gate also passes after extraction. The iterator itself exceeds the old
128-block controller limit; external execution and its complete resource/IO
admission are still pending.

The third increment implements the [value Store boundary](ooc-store.md), pinned
upstream XXH3-64, batched integrity-checked transfers and exact submitted/completed
IO ledgers. Fault injection covers unpublished output, retries, identity/range,
short IO, cancellation and write quotas. Six stored-boundary native products
cover three non-power-of-two shapes at W1/W4. This is resident promotion, not
the still-pending spilled multiplication or durable checkpoint implementation.

The fourth increment implements [scratch planes and native modular rows](ooc-modular-rows.md).
Its production middle driver reads verified packed rows and overwrites B in
place; SQR needs one row array. Local/global-root, scaling, parallel inverse,
codec and ownership/failure gates pass. The complete integer multiplication
still needs its outer passes and CRT/carry output.

The fifth increment adds out-of-order finalized-chunk construction, pinned input
indexes, shared-codec forward production and the inverse-stripe coordinator.
The complete modular sequence agrees with independent convolution references
and its source/plane byte counts reconcile. Integer CRT/carry and whole-job
qualification remain; no OOC performance claim follows from these gates.

The sixth increment completes the [spilled integer product](ooc-integer-product.md)
in simulation. Existing vector CRT/lag/Pack8 kernels feed bounded carry
descriptors and write-once output. MUL/SQR numerical, byte-ledger and failure
gates pass; the query includes the actual private phase maxima and persistent
metadata. Budget-driven selection, stored consumers, durable IO and 5B/YC
qualification are still pending.

The seventh increment adds the executable bounded external selector and
retention. It performs at most four complete summaries and one ordinary basis
query, includes declared fixed/shared/private RAM, and admits disk/write costs.
The selected B-family prefix remains resident through the three passes, removing
two reads and two writes per retained byte. Numerical results and exact traffic
pass for retained MUL/SQR; explicit-release and failure checks verify ownership.
Large 4/6/8/12 GiB cases are allocation-free planning checks, not timings or
large-operand runs. The cost profile remains an analytic scenario. Next is
stored value/extent ownership and its connection to generic precision episodes.

The eighth increment implements [stored ownership and exact series merges](ooc-series.md).
Preallocated metadata manages unique unpublished reservations, immutable handles,
read/checkpoint pins and contiguous extent reuse. Products publish only sealed
results and return their scratch after joins. A bounded streaming signed adder
handles binary exponents without stored shift buffers. The same at-most-five-
stage exact merger handles all three series families; last-reader counts release
consumed children and account for declared pins. Native subtree -> stored merge
trees, exact writes/peak backing, independent formula references and failure
cleanup pass. Precision-window integration and removal of these initial
intermediate materializations remain necessary before performance qualification.

The ninth increment adds immutable operand windows, normalization metadata from
finalized output and the resident PSR tail-merge rounding recipe. Window changes
replay selected transforms with bounded zero extension; original checksums and
backing identity are preserved. Nonaligned stripe guards are included in exact
product IO counts. Five complete stored precision chains (99 blocks total)
match both resident N and D, including an exact suffix, signed terms and
BinaryBBP. Sparse low-word/cancellation and unequal source-window products pass.
This still writes complete product payloads and keeps full backing allocations;
fusion/output cropping and paired-spectrum reuse remain necessary IO work.

The tenth increment implements projected product output and enables it for
precision merges. Full carries and m61 still resolve, but excluded chunks have
no Store writes. A bounded high window accounts for top normalization and
integrity alignment; its actual backing and writes replace full-result costs.
Adversarial carry seams, projected failure/coverage checks, a larger signed
precision merge and all existing precision chains pass. Example gates save
294,912 product write bytes and reduce a P5001 merge's write bound by 49,152
bytes. Arithmetic and private RAM are unchanged; linear fusion, shared spectra
and whole-job/terminal integration remain.

The eleventh increment implements reusable canonical outer rows and connects
them to the series numerator/denominator pair. A single extra shared-envelope
quote accounts for cache lifetime, metadata/IO, displaced retention, early raw
source retirement and whole-episode caps. Numerical and exact IO gates pass,
including reuse after the raw operand's backing is overwritten. The controlled
pair saves one source read and one plane-family write (53,248/589,824 bytes in
the gate). Cached exact and precision series, tight-RAM fallback, write-cap
admission and cache failure cleanup pass. Next is the remaining finite-prefix
adapter and complete bounded series traversal; fusion, Newton/SRT and durable
IO/qualification remain part of the same objective.

The twelfth increment compiles and executes complete generic series reductions.
Finite-node and precision-tail rounding are separate; product capacities are
preselected and bind shorter actual inputs. The existing splitter is shared
with the native scheduler. Complete resident suffixes remain in one region;
external finite prefixes use promoted native children and stored merges in an
order chosen by two live-backing formulas. Whole selected-series writes and
live backing are preflighted. Resident e/pi emit no Store IO; complete mixed
reductions match resident N/D with allocation, cap and cancellation checks.
Natural 5B/4GiB planning yields 27 blocks, resident suffix starting at 7 and
29 coarse nodes in about 0.14 s, without operands. Its series-only write/live
backing bounds exclude finishing/radix. The 2B frontier is still one block above
the recorded YC frontier at 4/6/8 GiB; native merge rebatching needs review.

The thirteenth increment removes full-batch merge overreservation from native
precision episodes. Each finite candidate derives a compact floor and a fitting
execution batch without changing arithmetic or searching NP; suffix realization
can spend already admitted workspace on faster batches. Numerical gates pass.
The native 2B frontiers become 3/2/1/0 at 4/6/8/12 GiB and the 12 GiB case's
RAM quote falls by 2,312,224,768 bytes. At 5B/4 GiB the external prefix loses one
block and the series write bound falls by 4,067,643,392 bytes. These are resource
queries, not measured arithmetic speedups; cuts still differ from YC.

The fourteenth increment implements [stored reciprocal refinement](ooc-newton.md).
Bounded shifted input views remove a normalization materialization. Signed
low-residue lifting and projected linear assembly reuse the existing guarded
Newton recipe. A cached approximation supplies both products; a complete
ladder chooses its largest fitting native seed and compiles all remaining
stored rungs before execution. Independent strict 3-ulp checks, exact shifted
source traffic, allocation watches, write/backing bounds and cancellation/lease
cleanup pass. This first provider uses full LIN products; efficient CYC/TMP
lowering, quotient/rsqrt, SRT and durable IO remain. No benchmark or real SSD
operand/swap write was performed.

The fifteenth increment completes the first stored numerical finishing chain.
Half-reciprocal quotient shares U between coarse/correction products and streams
the shifted numerator subtraction into a retained residual. Stored rsqrt adds
immutable cached square rows and scalar CRT/carry fusion. Fused multiply-rsqrt
uses only a half-precision root; its three final products retain independent
geometries. Certified fixed output checks arbitrary bit/word guard margins.
All three series families feed quotient successfully; e and small pi reach
binary output, with pi independently matching Machin at17/65 fractional limbs.
The5B/4GiB query caught and fixed alignment admission after retained-plane
filling. Both finishing phases now admit without a search. These remain
simulation/query gates, not large-run or performance evidence. Whole-job
terminal compilation/error propagation, SRT, CYC/TMP, physical placement and
durable IO remain before verified5B/YC qualification.

The sixteenth increment adds [stored fractional SRT](ooc-radix.md). A fixed
request-scaled or regular dyadic rail and memoized size classes select complete
native frontiers; stored parents expose left views and produce only right
windows. Right-first traversal resolves carry seams before one reverse chunk
assembler publishes digits. Base factors of two are removed from the products;
power-of-two bases use views only. Exact traffic, seven bases, multi-chunk and
boundary gates pass. Stored e now includes its missing +1 in final emission;
pi reaches200/1151 certified decimal digits with independent references. The
5B/4GiB metadata-only radix query admits7 frontiers and quotes28493787136 write
bytes including output. This is not an actual5B run or performance acceptance.
Whole-job compilation, more efficient external windows/frontiers, durable IO,
physical placement and full fresh YC qualification remain.

The seventeenth increment adds the [production generic job runner](ooc-job.md).
Series result envelopes and shape-only finishing queries fix all backend choices
before operands exist; normalization views rebind without changing geometry or
caps. Virtual low zeros support short exact inputs without padding files. The
job derives its precision/error allowance from formula bounds and tail error,
shares one arena grant across phases, and exposes owned safe points. IO failures
preserve the previous state and charge failed writes. All three series families
pass independent binary/decimal tests, including retry and checkpoint pins.
The full5B/4GiB metadata query quotes4293475808 resident bytes,26841251840 peak
backing bytes and297881231360 total writes. Actual large execution, physical
placement, durable IO/recovery and fresh YC qualification remain.

The eighteenth increment adds [Linux object-file storage](ooc-files.md). Fixed
virtual slots separate address space from a sum-of-live-bytes quota; compact
per-Region files eliminate contiguous-bank fragmentation without moving data.
Creation is fallible, publication enforces read-only value access, and release
defers IO to explicit collection or the next creation. Memfd and tmpfs DIRECT
tests pass, including all three generic job families, exact byte/peak claims,
corruption, retry and cleanup. No SSD operand writes ran. Durable manifests,
restore/commit resolution, async IO and large/fresh YC qualification remain.

The nineteenth increment implements [durable phase checkpoints](ooc-checkpoint.md).
Immutable imports preserve checksum identities while remapping slots; resume
checks all payload/index chunks and write-time numeric m61 summaries. Publication
syncs data and its directory before the manifest, then uses atomic rename and a
final directory sync. UNKNOWN keeps both roots and suspends creation. Persisted
attempt reservations keep crashed numerical writes charged; actual returns
reconcile them. Fresh-plan pi phase restarts and eight SIGKILL points, including
a torn in-progress data write, pass independent references. Corrupt/incompatible
roots refuse non-destructively. All tests use tmpfs; real power-loss, large
execution, performance improvements and verified5B/fresh YC qualification remain.

The twentieth increment adds the [bounded reference application](ooc-runner.md),
with pure-query default, host/backing/MEMLOCK admission, explicit write quotas,
optional checkpoints, phase ledgers and independent BBP verification. The first
actual 2B and 5B pi runs pass with a 4 GiB computation grant on tmpfs; all
requested decimal digits are certified and the independent 128-bit tail agrees.
Compilation and execution remain serialized. These are initial functional and
profiling anchors, not a completed scaling study or SSD performance evidence.
Fresh time/IO comparison and structural optimization remain part of the active
goal; the completion criteria have not been reduced to producing digits.

The twenty-first increment removes avoidable bounce-buffer copies for aligned
complete Store chunks and aligns the receiving middle/inverse banks. Integrity
checks/retries and partial-edge behavior are preserved. A private native bulk
m61 kernel performs the same fingerprint using eight vector lanes, with a
scalar reference and independent reduction gates. Store, product, generic-file
job and checkpoint recovery gates pass. The2B fresh anchor improves with the
same queried RAM and exact IO counts, but remains substantially slower than YC.
The larger implementation work is still CYC/TMP window lowering, series merge
fusion and shared spectra; this increment does not establish performance parity.

The twenty-second increment implements
[certified cyclic residuals](ooc-cyclic-residual.md) in reciprocal refinement
and the fused terminal square. Carry closes around the ring before signed
magnitude/m61 validation, and complemented views remove residual conversion
files. The ring basis is selected by a bounded volume rule. Four-column outer
transfers emit full cache lines using fully quoted scratch. Updated2B and5B pi
pass independent BBP validation at4GiB, with lower fresh time and IO. Series,
stored-subtractand cancellation and general middle windows still require work;
the performance objective remains open.

The twenty-third increment adds
[native products and scratch loans inside stored merges](ooc-resident-merges.md).
Complete product episodes may execute natively even when their operands are
stored; an immediately consumed numerator result can stay in planned scratch
through the sum. Source views, high projection, exact rounding, full m61 and
failure/ownership contracts remain intact. Updated2B and5B runs are verified
at4GiB with lower time and IO. This is promotion/lifetime reuse; general fused
external MAC emission and performance qualification are still outstanding.

The current [middle-pass executor](ooc-modular-rows.md) can overlap checked IO
with arithmetic through two explicitly quoted packed slots, using one existing
Team lane as the sole IO owner. It borrows only phase-dead workspace already
required by another pass and displaces no retained data. Cached spectra retain
their completion-state contracts and immutable-square input ownership. Native
reference/failure gates, ThreadSanitizer stress and verified 2B/5B jobs pass.
Isolated file-backed passes improve, but complete tmpfs jobs do not demonstrate
a meaningful speedup: most raw-A middle rows are already resident. Remaining
outer-forward/inverse/CRT scheduling and fusion, arithmetic-window lowering,
and full fresh performance qualification are still required.

Outer forward production now shares that bounded handoff, reusing each raw
stripe across all prime batches. It can use one or two raw stripes according
to the already reserved phase workspace; no extra source pass is required.
Its raw grant omits mathematically known zero rows and unused final-row cells.
Address/thread sanitizers, exact IO and product/failure gates pass. Updated
unprofiled 2B/5B jobs improve with identical IO counts and the same working
buffer/backing sizes, while remaining slower than YC. Inverse/CRT scheduling,
remaining arithmetic-window lowering and broader qualification remain open.

Inverse prime batches now overlap read/compute inside the existing coefficient
bank. CRT and scalar/cyclic row arithmetic stream through existing output slots
while one owner performs subtraction reads, carry submission and writes.
Partial batches, two-thread execution, independent carry/scaling, exact IO,
in-flight failures and sanitizer gates pass. The verified 5B fresh anchor improves
modestly again with no additional data storage or IO. General middle-product
lowering and the remaining series gap are still required for YC parity.
