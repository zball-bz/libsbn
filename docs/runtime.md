# Runtime and ownership

Values and output arrays are caller-owned. Query functions report support,
alignment and complete workspace requirements before execution. Prepare arena
storage explicitly, then bind an operation to its plan, storage and team.

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

A team has one controlling thread. Binding and top-level operations must obey
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

Radix conversion also has a 626,680-byte immutable integer-power prefix for all
31 odd bases from 3 through 63. It contains the first nine powers
`odd^(64 * 2^k)`, including readable zero padding, and has no runtime initializer.
Plans reference these published constants directly; larger powers are generated
inside each operation's planned storage. Report this shared bank once per process.

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
