# Runtime and ownership

The planned external execution layer is described in
[the OOC planner design](ooc-planner.md). It preserves the resident contracts
below; OOC entry points are not yet declared by that design document.

The private [constant-job runner](ooc-job.md) now composes stored series,
Newton and radix phases under one complete grant. Backend choices are compiled
before arithmetic; actual normalization views are bounded rebindings. One
prepared arena interval is reused through owned algorithm safe points. IO
failure retains the previous state, releases private work, and charges submitted
writes; no runtime plan search or implicit scratch growth occurs. The
[checkpoint layer](ooc-checkpoint.md) now persists these phase boundaries using
verified immutable imports and atomic manifest publication. UNKNOWN retains
both roots and blocks new file creation until resolved.

The [bounded constant example](ooc-runner.md) joins these providers without a
public ABI addition. It defaults to a query, admits computation plus backing
against the host, and executes on tmpfs by default with an explicit write cap.
Real-device execution requires a separate opt-in and finite cumulative cap. Fresh
output timing, verification and final context/output disposal are distinct.

Coarse series boundaries may lend a last native child's private magnitudes to
an all-native parent merge. Last-use-ordered slots occupy the existing arena
tail, and each joined stage can reuse the dead prefix. Bounded-input products
omit copied arrays for those resident sources. The compiler prices compaction,
intermediate result loans, stored writes and backing before execution. These
are consuming private loans; committed/caller-preserved values remain immutable,
and completed parent/phase results retain ordinary Store ownership.

After those handoffs, resident parent merges may keep a common transform across
two products. A joined continuation checks and rounds the first product,
finishes its numerator sum, then reuses the same result array for the second
product. The continuation's IO workspace is separate from the live spectrum;
IO failure stops the episode before more arithmetic begins. Admission preserves
the existing handoff and complete RAM/write/backing bounds.

Stored linear arithmetic can overlap its ordered carry chain with checked IO.
Two operand banks belong to one IO lane and one compute lane of the existing
Team; an active product episode lends a joined sub-scope. No extra thread is
created. The query charges both banks, and the serial and pipelined schedules
share the same chunk arithmetic. Input residues, output residue and magnitude
summary are computed once by the compute owner; the writer retains transport
integrity and publication. Failure stops slot admission and joins both owners
before private buffers or output are released.

Stored SRT splits can use a smaller cyclic transform with an exact middle-window
carry-gap check. An ambiguous gap uses the precompiled LIN witness; no runtime
search or allocation occurs. Normal traffic and worst-case witness traffic are
quoted separately. See [stored radix output](ooc-radix.md) for the ring-specific
arithmetic checksum and the unchanged transport integrity contract.

The [file-backed Store](ooc-files.md) maps bounded virtual object slots to
compact Linux files. Its byte quota counts live object allocations, not virtual
address spans. Create/sync/collect/close are explicit fallible IO operations;
reference release only marks retirement. Reclamation precedes every subsequent
creation, so pending cleanup does not increase the historical live-byte peak.
Published values are read-only to the transport. File and Store registries are
caller-owned metadata; no owner destructor performs IO. Checkpoint shutdown
explicitly preserves committed files; failed imports preserve preexisting data.

FileStore can explicitly prepare separate bounded io_uring read/write rings.
Their mapping allowance is queried separately and charged to the context;
arithmetic storage and retained planes do not change. Joined batch callbacks
hold buffers through every original completion, including IO failure and
cancellation, before the usual cleanup proceeds. No user heap allocation or
additional user worker is introduced. Kernel fallback workers are separately
bounded at preparation. The scalar provider remains the default; see
[queued IO](ooc-files.md#optional-queued-io).

Values and output arrays are caller-owned. Query functions report support,
alignment and complete workspace requirements before execution. Prepare arena
storage explicitly, then bind an operation to its plan, storage and team.

An external product may retain one operand as normalized raw limbs and produce
its packed outer rows one prime at a time, immediately before the middle pass.
Only the other plane family has Store backing. The raw source, one transient
prime and overlapping forward/middle scratch are included in the complete
phase maximum before B-prefix retention is assigned. Input shifts, complements
and windows are resolved once on the immutable source. The transient prime is
private RAM work, not a published cache or checkpoint. Inverse/CRT arithmetic
and result integrity remain unchanged; IO failure retires private work normally.
Cached-row and square recipes retain their separate explicit contracts.

Raw retention can be partial. The plan quotes a normalized, cell-aligned
prefix and two staging buffers covering only the missing source rows. Forward
decoding reads cached cells directly; an IO lane supplies the remaining suffix
for each prime while the other lanes compute. The verified original-value index
stays pinned across these phases. ValueReader keeps no payload cache, so its
joined IO buffers may be reused by the intervening middle pass. Original values
remain immutable and every operation still uses one transient A prime.

The external p48 kernels preserve those layouts while eliminating redundant
RAM passes. Forward decoding shares four columns' byte slices and each prime's
constants; its piece/window specialization follows the codec format. Bounded
source prefetch and the blocked TFT's exact zero-tail requirement add no
staging storage. Deep modular rows load packed inputs in their first upper
butterflies and emit packed results in their final inverse butterflies. The
unpacked row arrays are scratch, not additional outputs. Already transformed
inputs and unary paths keep the conversions they require. Leaf tasks use
contiguous balanced ranges to reduce cache transfers between workers. These
choices do not change arithmetic/storage queries, retained-plane capacity,
source ownership, integrity coverage or submitted IO volume.

Final inverse reads have an independent stripe width. The planner solves the
linear bank-storage plus reciprocal carry-descriptor bound inside the existing
final-phase grant, then admits one exact quote. The raw prefix and its input
index are dead after all middle passes join, so their range joins the final
scratch loan. Retained B planes remain outside that range. Wider checked reads do not
enlarge arithmetic tiles, displace raw/plane retention, increase the complete
grant or change submitted byte totals. Input staging keeps its original width;
final banks remain exclusive through read, compute and emission and join on
failure before storage is released.

A finite raw prefix also has an independent source-read width. When the byte
transfer target exceeds its current staging width, the planner performs one
bounded prefix-lifetime replay. The existing byte/request cost must justify any
additional rereads, and writes may not increase. The replay remains inside the
same complete grant; final-bank widening is admitted afterward. This separates
source, packed-forward and packed-inverse transfers without a width grid search.

The prefix fitter starts from a conservative zero-prefix workspace quote,
fills the admitted cache capacity, then refills capacity released by removed
staged rows once. It does not search prefix sizes. Any spare bytes retain B as
before, and hard write caps still apply. For q primes, replacing one resident
B-plane byte by one raw-input byte saves q−5 total transferred bytes, apart from
explicitly charged chunk edges. The selector compares total reads plus writes
because partial retention makes the two directions asymmetric.

When middle-pass sources are already resident, the executor processes the
complete retained row prefix in one joined Team region. Verification and
checksum refresh run in parallel around a synchronous exclusive RAM loan;
workers unpack, transform and pack directly rather than bouncing each row
through controller copies. The same single-row workspace is reused. File-backed
rows retain checked IO, and a row crossing the retention boundary still uses
that checked path. The callback cannot escape its span or re-enter the borrowed
plane.

The external middle pass can overlap checked IO and arithmetic through two
packed row slots. Its controller owns Transport, plane indexes and all reads/
writes; one existing Team lane runs that controller while the remaining lanes
compute and unpack/pack. The arithmetic arrays, worker scratch and root tables
are shared sequentially between rows. Separate packed slots are admitted only when they
fit phase-dead workspace already required by another product pass: total
workspace, plane retention and submitted byte counts do not grow. One remaining
row uses the synchronous path. There is no helper thread, implicit allocation,
or IO surviving the call. Read/checksum completion publishes a ready slot;
compute completion publishes its result; write completion precedes reuse.
IO failure wakes the waiting consumer, joins any current arithmetic, then
retires unpublished output. Cached squares retain their immutable input.

When A is a disposable prime generated from a retained raw operand, its first
two packed rows can become the middle pipeline's B staging slots after their
last compute read. The executor finishes those rows first (or uses the already
finished resident prefix), then streams the remaining rows through those dead
bytes. This consuming adapter is distinct from the immutable memory/cache
adapter, which rejects the destructive plan. It adds no workspace and displaces
no retained plane bytes; input values remain immutable. A failure may destroy
that private prime, joins all workers, and retires the unpublished output.

For fully spilled operands whose middle pass would otherwise remain serial,
the planner can instead reserve two expanded row banks. Each bank also holds
its packed IO blocks: expansion runs backward inside disjoint blocks, arithmetic
uses the contiguous expanded row, and compaction runs forward in those blocks.
The persistent plane layout is unchanged. The banks are reserved inside the
already admitted grant before assigning the remaining bytes to B retention.
Any displaced retention appears in the exact read/write quote; a hard write
limit can keep the earlier placement. This is one storage replay on the chosen
basis, with no further geometry search or runtime allocation.

The optional three-stage middle schedule separates readers, compute and the
writer. Three explicit slots prevent a reader from overwriting output awaiting
write completion. A disposable A prime can lend three consumed rows; a fully
spilled plan quotes a third expanded bank and any displaced retention. One
writer owns mutable plane metadata. Borrowed full-chunk readers copy immutable
descriptor values and access only chunks disjoint from concurrent writes; their
read ledger is joined back before returning. A provider must advertise duplex
and per-direction concurrency. Unsupported providers use the two-stage schedule.

Independent prime writers in the forward pass have separate index/coverage
state. Only aligned complete fragments use concurrent writes, so their shared
bounce buffers remain unused. Transport admission/completion counters use short
per-direction locks where concurrency is advertised; no lock spans a callback.
FileStore statistics support the declared data concurrency, while registry,
publication, cleanup and snapshots still require the joined controlling owner.

Forward and inverse stripe widths are independent execution parameters. The
planner widens forward transfers when their two packed staging banks fit the
already admitted phase pool. This does not change arithmetic, plane addresses,
retention or total IO bytes. The final inverse keeps its smaller all-prime/CRT
stripe. Reader counts balance complete prime streams within the provider and
Team limits; no per-size timing search or extra thread is introduced.

The final schedule quotes two or three inverse/output stripe banks. With three
banks and a duplex provider, independent readers, arithmetic and one output
writer overlap; a two-role fallback handles providers without duplex support.
These roles use the existing Team, with no helper threads or hot allocations.
Additional banks can displace retained planes; the complete RAM and transfer
quote includes that cost before admission. Stored subtractands have separately
priced row buffers and use the same joined schedule.

The fused final kernel reads compact packed48 banks and completes inverse
transforms across all primes in cache-sized tiles. CRT consumes each tile there;
no expanded full-stripe bank is written and reread. Exact tile carry summaries
are joined into each output row before submission. The direct CRT cofactor
preconditioning is explicit in the plan and folds into the middle pass's existing
normalization multiply. Cached operand spectra remain unscaled.

Ordered, chunk-aligned output rows retain at most one arbitrary carry fragment
per row. A non-0xff endpoint makes that segment's outgoing carry independent of
its incoming carry, so subsequent segments can be finalized and written whole.
Earlier all-0xff segments need only symbolic descriptors. Arbitrary arrival and
unaligned edges keep the general fragment contract; cyclic carry closure and
the final m61 identity remain exact checks before publication.

Outer forward production similarly overlaps input/checksum and output/write
work with the transform on the other Team scope. Two packed prime-batch slots
reuse one compute workspace. With two raw buffers it prefetches another stripe;
with one raw buffer it waits for that stripe's final prime-batch reader before
overwriting it. Each raw stripe is loaded once across all prime batches in
either schedule. Both placements are admitted only inside the existing phase
maximum. Raw staging backs only input cells reachable under the TFT's known-
zero bounds, plus the codec guards; virtual zero rows and a final unused row
tail need no physical buffer. Arithmetic geometry and packed plane layout are
unchanged, and no extra Store pass is introduced.

Inverse stripes can read the next prime batch while computing the preceding
batch in disjoint spans of their already reserved coefficient bank. No second
bank is allocated; all primes join before the borrowed stripe reaches CRT.
CRT/output similarly streams existing row buffers to one carry/IO owner.
Compute lanes own disjoint slots; a single compute lane alternates two slots.
The IO owner loads any stored subtractand, submits the finalized row, then
releases that slot. Numeric scaling, carry journals, checksum rules and final
publication remain unchanged. A failed read/write stops admission and joins all
compute lanes before their storage or unpublished output is released.

The private [external series runner](ooc-series.md) compiles into caller-owned
stable metadata and uses an explicitly prepared work region plus separate IO
storage. A wholly resident result borrows that region; a stored result owns
its immutable handles. The resident suffix is never materialized between its
episodes. Compile failure invalidates the plan/metadata under exclusive ownership;
IO failure returns no result and discards uncommitted private values. Published
engine banks, metadata, controller allowance and complete phase storage are
included in its query; the caller supplies other live context/worker costs.

The private [stored Newton runners](ooc-newton.md) also borrow an explicitly
prepared, initially unleased arena interval through return. Its native seed
owns disjoint leases; external rungs reuse the interval after those leases end.
The query accounts for plan metadata, published banks, seed input/output and
the maximal cache/product/linear phase. Inputs are immutable stored handles;
normalization is a bounded source view. IO failure returns no result and
releases private handles and leases. Arithmetic certificate failures remain
fatal. Reciprocal and quotient cancellation can use bounded cyclic products;
remaining projected windows still have full-LIN fallbacks. Quotient preserves both stored
inputs and restores their binary scales. Rsqrt shares immutable outer Rows
between square and correction; the fused final consumer only materializes a
half-precision root. Scalar CRT/carry emission removes separate scaling passes.
The fixed-output stage requires a caller-supplied mathematical error bound and
checks guard separation before publishing a floor. Sequential phases round
their workspace grant down to the common arena alignment before retaining
planes, so final alignment cannot silently exceed the total RAM quote.

The private [stored fractional formatter](ooc-radix.md) shares one prepared
phase range across native power construction, external right-child products
and native format frontiers. Parent values and the odd-power rail remain
immutable stored handles; left children are views. Right-first traversal fixes
each frontier's carry seam before a reverse chunk assembler writes it once.
Only the returned certified prefix may be exposed, even though the private
output has the requested physical width. IO failure releases every unpublished
value and lease. The current contract is enclosed fractional output with a
u64 integer part, not the complete stored integer/float radix API.

Small/middle GCD uses caller-owned scratch directly, with no plan, arena or team.
It writes only the normalized result and preserves both inputs; exceptional
remainder steps discard quotient digits. Word and long-by-one/two-word GCD need no caller
scratch. See [gcd.md](gcd.md) and `sbn3/gcd.h` for the bounded domain and alias rules.
Jacobi uses the same scratch quote and preserves signed numerator/positive odd
denominator views; it returns only a small symbol. Quotient observers retain
scalar state, with transactional rollback at rejected middle-product attempts.
The leaf producer owns symbol state; no state is shared with the matrix
consumer. See [jacobi.md](jacobi.md) and `sbn3/jacobi.h`.
General GCD and signed extended GCD use the same preserving value model with a
pure complete-storage query. Recursive arithmetic borrows that scratch and an
optional existing team; a serial call constructs only a stack execution scope.
Common whole zero limbs are cancelled with input views; only the GCD output
restores the shared radix power. Coefficients remain valid for the original
inputs. This reduces live arithmetic without changing the length-only query.
Matrix-only reductions may destroy their private working prefixes, but all
public inputs remain unchanged. Cofactor outputs may be omitted independently.
The ordinary-GCD and extended-GCD middle route may retain one
immutable matrix's packed spectra across its second child. Its complete lifetime and child bound must fit
the existing scratch quote. It returns carried matrix values to its parent;
no operation cache survives the call. Dead copied prefixes are reclaimed before
the next exact matrix application. Parallel product batches reserve disjoint
worker Frames and immutable source combinations before dispatch. All input
reads join before aliased matrix outputs are overwritten. Fused matrix emission
reuses dead transform storage; concurrent outputs have distinct carry journals.

Extended GCD may pack outer stage matrices into its existing scratch and defer
coefficient reconstruction until the GCD is emitted. At most 64 descriptors
occupy fixed stack storage; matrix entries stay in the caller's Frame. Each
retention admits the complete eager suffix and subsequent reconstruction
bounds. If they do not fit, the current suffix finishes eagerly. The two
private operand arrays become the reconstructed column only after their last
residue read. History is released in reverse order. Optional coefficient
outputs retain their smaller query contract; no buffer is borrowed from an
omitted or unused output. No history or operation cache survives return.
An equal-length seed quotient has at most one word; its separate fixed stack
matrix avoids consuming history space before any whole limb has been removed.
It is used only when both coefficient components are available for reconstruction.

A matrix-bearing HGCD leaf may use two existing workers. Its producer owns
the residuals and arithmetic Frame; the consumer owns the reconstruction
matrix. Equal prefixes exit without dispatch; an initial synchronous reduction
also keeps completed one-step cases from recruiting a helper. Certified word
matrices travel through a bounded 64-entry stream,
published four at a time. Exact fallback drains the stream before touching
the matrix, and the leaf joins before returning it. The stream occupies about
2.3 KiB of fixed stack storage, with no heap allocation or larger scratch quote.
Recursive children remain dependent; this pipeline does not execute them
concurrently.

Native FFT matrix caches contain executable arrays and remain owned by one
recursive node. Their transient workspace excludes a redundant unpacked copy.
Pointwise tasks own complete mirror-block and inverse-leaf pairs; all source
reads join before integer outputs can consume old matrix values. The FFT uses
a team sized for its independent inverse/output streams. Flat NTT residual
pairs retain two source spectra while reusing two coefficient buffers across
rows. The signed coefficient bias is included in the MAC2 capacity and physical
plane-scale contracts, then removed by cyclic folding. Matrix-only programs
advertise their matrix capability explicitly; they have no ordinary MUL entry.

Small decimal primitives in `sbn3/decimal.h` need no runtime object or caller
workspace. Their dimensions are bounded by u64/u128 or 1–8 limbs, for both
integers and fixed-point fractions; their constants are immutable.
Caller-owned input/output spans and register/stack temporaries
suffice; no arena or team is consulted. These direct primitives are distinct
from the general radix service. See [decimal.md](decimal.md).

Radix `*_once` calls share the same value and output contracts. Zero-workspace
leaves access no arena or team. Other local recipes borrow the supplied range
for the duration of the call; general recipes run the existing lifecycle inside
it. Checked builds register borrowed-workspace leases and audit overlap. No
per-operation state survives return. See [radix-once.md](radix-once.md).

Small one-shot services may instead accept a caller-owned scratch span with a
pure size query. An external Frame only subdivides that exclusive span; it does
not retain an arena lease, allocate memory, or change page residency. The caller
keeps the span live through the call, including all nested views.

The arithmetic path does not perform ordinary heap allocation or automatic
scratch expansion. Scratch comes from reserved arena regions and is divided
among recursive tasks. Exhausting a correctly planned region or violating a
mathematical invariant is a fatal diagnostic; it is not a request to silently
switch algorithms. Resource preparation reports failure before execution.

A binding retains exclusive ownership of its declared storage until unbind.
A private serial Flat cyclic capability may execute fresh operands while
retaining its cached operand as a root-table owner. This does not mutate the
sealed plan or cache state; its support query proves that the original worker
workspace suffices. The output planes hold only the fresh temporary spectrum.
For compact exact division and small non-FFT approximate division the caller
grants that ownership by lending a prepared
range for the whole binding lifetime. Normal builds do not register a second
arena lease for this borrowed span; checked builds do. The caller must keep
its pages resident and its bytes exclusive, even while no execute is active.
Internally, planned phases may release and reacquire leases over the same bytes
after all users of the previous phase have joined. The caller must not reuse or
trim that storage between phases. In-place result views remain valid only
for the lifetime declared by that service. Inputs, output and workspace must obey
the overlap rules in the public header. Cache compatibility includes arithmetic
basis, completion state, scaling and support, not merely transform size. A
consumer may read a wider or rescaled spectrum, but only a binding whose own
representation equals the reservation may build it; a service that builds
through its consumer selects that one representation at query time.

A team has one controlling thread. Epoch waits normally spin for at most 1 ms
before parking (capped at 2 µs in the thread-sanitizer build). An external product
temporarily sets its Team's idle-worker/join polling budget to 10 µs, then restores
the prior value on success or IO failure. Arithmetic region barriers keep their
existing policy. This avoids prolonged idle polling across IO stages while
retaining the normal policy for resident operations. Binding and top-level operations must obey
its idle/owner rules; internal worker scopes may split into subteams. Never share
an exclusive workspace between concurrent bindings. Read-only prepared tables
may be shared only through their explicit lifetime contract.

A compact Newton binding runs its local reciprocal seed before activating
any product in the stage pool. Seed scratch may span the still-empty product
pool and its adjacent integer-work region, whose leases remain owned by that
binding. The seed result is retained outside this union. Once the seed frame
has ended, products may initialize their tables and work in those same bytes.
The resource query reserves the maximum of these sequential phase requirements.
Reciprocal ladders keep intermediate approximations in caller output, whose
final capacity is already sufficient. At the normal precision-doubling step,
only two or three source words overlap the shifted destination; they are
saved before the remaining disjoint copy, so large rungs retain parallel copy
bandwidth. No separate reciprocal ping-pong values are reserved.
Automatic serial ladders use the largest supported local reciprocal prefix
on their fixed precision-doubling sequence. This is the same product capability
used by a standalone reciprocal; only the remaining rungs own general product
bindings. Explicit prime selections and parallel ladders retain their stated
seed policy. The prefix's complete scratch is included in the phase maximum.

A cached product may advertise that all fresh-input reads finish before any
integer output write. The private live-input entry can then consume its B
span in place. Reciprocal correction uses this guarantee to retain rho in
the product output buffer; it does not reserve a second full integer buffer.
Bindings that borrow output for an earlier transpose do not grant this guarantee.

The serial Flat backend can also lend its dead result planes to a synchronous
scratch callback between products. The loan excludes control objects, tables,
worker Frames and cached spectra. A borrowed child Frame protects the parent
lifetime and restores ASan access on return; product execution is marked active
for the duration. The callback must not re-enter, prepare or unbind that binding,
and no pointer or child Frame from the loan may escape. Query reports the full
available span before selecting a recipe that depends on it.

Consult include/sbn3/arena.h, team.h, product.h and the specific service header.
Some bindings are single-use; radix bindings explicitly support repeated execute.

The native FFT also uses 7,771,648 bytes of published immutable root tables in the
library's read-only image. It has no runtime initializer and is shared by all
bindings; operation storage queries exclude it, just as they exclude code and
fixed scalar constants. Both power-of-two stage kinds through branch 131072 and the small RightAngle twists
are instantiated once in a
data-only translation unit; `constinit` rules out hidden runtime construction.
Stages 65536 and 131072 use the compact layout consumed by the kernels, rather
than retaining unused full-layout arrays and rebuilding compact tables per call.
The next power-of-two product additionally shares its branch-262144 radix-4
stage and a PQ prefix through that branch. Its radix-8 children are already
covered, so no unused radix-8 stage at 262144 is published. These two additions
cost 1.5 MiB shared and eliminate 2 MiB of private tables per affected binding;
the corresponding single-call shared-plus-private bound falls by 512 KiB.
Operations not using that level still account for the larger shared bank.
The CT cross-root prefix at radix 3/5/7 and branches 128–4096 is shared
by all digit widths and accounts for 1,548,288 of those bytes.
For larger CT branches through 131072, cross roots factor against that prefix
and a 13,824-byte immutable fine-rotation bank. They do not create full cross-root
planes inside each call. Both factors are independently rounded constants;
there is no recurrence across blocks. Beyond the published branch domain,
additional roots still use planned arena storage.
Memory comparisons report the complete root bank once per process beside the
queried private storage. No operation-sized cache survives unbind.

The p48 workspace uses the alignment required by its actual plane pool:
64 bytes for Flat, 4 KiB for small Bailey planes and 2 MiB for large ones.
Flat primes are statically assigned to workers. Ranks with no transform keep
only their CRT-emission scratch, and ranks with neither task keep a minimal
Frame. Prime spreading permutes distinct frame pointers while keeping its CPU
assignment. Paired tails retain their required transform frames. An uncached
Flat product can keep its live inverse-transform operand in an enlarged
result plane, then compact it to 48-bit slots in ascending order. Query enables
this only when the exact sum of plane and worker storage decreases. `per_worker_bytes` is the maximum worker reservation;
`workspace_bytes` accounts for the individual reservations.

The p48 backend also publishes a 1,235,920-byte immutable integer-root bank.
Its compressed tower covers orders through 2^14 for all ten primes. Column
factors cover 256 columns and row levels 0–13. The tables depend on the
prime and root index, independently of NP, codec, operand size and scaling.
Covered bindings borrow those read-only prefixes; larger requests construct
their remaining tables in the quoted arena storage. The table-preparation
model prices only the entries that need construction. There is no runtime
initializer or operation-sized cache retained after unbind.

Radix conversion has a 3,047,976-byte immutable power/reciprocal bank. The
626,680-byte integer-power prefix covers all 31 odd bases from 3 through 63 and
the first nine powers `odd^(64 * 2^k)`, including readable zero padding. The bank
also contains 8,320 bytes of parser leaf descriptors and 2,412,976 bytes of exact
normalized integer-format reciprocals and their directory. Reciprocal prefixes
are shared by bases with the same odd part; the precision cap includes the
guards needed by the 256-limb direct-format domain. Larger shapes construct
their constants in planned storage. There is no runtime initializer. Report
the whole shared bank once per process, separately from operation workspace.

The radix digit and leaf codecs publish 474,808 bytes of per-base numeric
constants, short exact powers and byte maps. This includes IFMA polynomial
coefficients, identity-alphabet descriptors and single-limb digit extraction
constants, a standard decimal ASCII descriptor and four aligned u52-to-u64
packing maps. Their resource query additionally counts an 8,192-byte bound for
the shared direct-decimal powers, word reciprocals and SIMD constants, including
alignment: 483,000 bytes in total. These are compile-time constants without a runtime initializer.
Small integer calls recognize the ten standard decimal alphabet bytes during
query and borrow that descriptor; custom alphabets still validate and prepare
their mappings in each fresh call. No operation-sized conversion state is
retained between calls.

Direct radix bit/leaf bindings borrow the supplied resident, exclusive range,
with lease registration and full plan-integrity checks enabled in checked
builds. This has the same lifetime contract as compact division. Direct formatting
and small integer parsing check sizes, alignment and aliasing in checked builds
and otherwise trust those caller preconditions. Parsing always validates digits.
Small bindings retain only execution inputs; the general query/bind/execute
bodies are outlined so small calls do not reserve their large planning frames.
Tree bindings retain lease registration, execution checks
and replayed product plans in all builds. Parsing parallel child
values occupy disjoint parts of one slab; merges overwrite a child only after
its last read. Sequential tree/scaling/repair phases reserve their maximum
workspace, not a sum. A repeated scaling reciprocal's cached FFT spectrum
belongs to that binding and is included in its prepared-storage query.

Small Newton services keep the size query's selected shared/cancellation/
peeling product descriptors in the opaque plan. Bind copies the packed records
into its caller-owned control region; execute reuses them instead of repeating
codec selection. The plan seal covers every record. This is per-plan state,
not a hidden process cache; full fresh timing includes its construction and
cleanup. Below the FFT crossover approximate division instead stores just
the size, scratch bound, arithmetic recipe and timing flag in its opaque
plan. It neither initializes nor reads the unused ABI envelope. Automatic
selection shares exact division's word/IFMA/DC work thresholds and the
existing local-product capability boundary; it does not search candidates.
This small binding has 128 bytes of control and borrows its resident span
as described above. Checked builds retain lifetime/alias diagnostics;
ordinary calls keep the single-use and mathematical input checks.

Native local division uses scalar work thresholds and direct quotient-block
arithmetic. Its former 49,792-byte cost table is retired. Division shares
dynamically prepared FFT roots between equal geometries within a call; it
retains no operation-sized root cache after that call.

Radix formatting separates sequential pool phases. A staged cyclic product
joins and unbinds before an ambiguous carry witness borrows its pool; only
ambiguous residues are regenerated afterward. Source limbs remain live until
the witness is settled, including when the child output aliases their slot.
Fresh trees retain requested powers and retire construction-only
powers after setup. ENCLOSED formatting certifies its prefix from the combined
tree/input error bound and does not reserve the final EXACT tie-repair product.
See [radix-format.md](radix-format.md) for the split and interval contracts.

Radix stage reservations use the largest simultaneous requirement. A large
ordinary-product episode may overlap another stage's ring pool. After every
task and binding in a stage joins, the controller reissues the work lease at
the same address with the next stage's extent; live fraction data stay in a
fixed prefix. Eligible stages temporarily use the caller's still-empty digit area for
integer results. Ring stages can also use its backing span for transpose
scratch; ordinary stages keep their separate workspace in the arena. They finish all reads
and copy child fractions before digit emission begins. The public output
capacity is unchanged and remains part of total memory accounting.
