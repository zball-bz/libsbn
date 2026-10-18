# Scratch planes and the native middle pass

Implementation status, 2026-10-02. These are private executable components of
the [OOC objective](ooc-implementation.md). The complete integer coordinator
also runs in simulation; durable and whole-job execution remain separate.

## Plane ownership

`ScratchPlane` consumes a region explicitly created with `RegionKind::scratch`.
It rejects value-kind regions. The caller grants an aligned checksum-index
image, one dirty bit per index page and bounded IO buffers. Index pages are
verified before ownership changes. There is no hidden region-sized allocation.

A read handle contains its owning plane and an epoch. Beginning a consuming
update invalidates old handles; reads during that update require its exclusive
token. The coordinator must join all old backing consumers before starting it.
Current transport callbacks complete synchronously, so no original IO retains a
buffer after return. Ordinary reads return verified copies. The synchronous
resident callback below is the only mutable-alias exception.

Patches preserve verified bytes outside their logical range. Full-chunk
replacements skip the old-data read; partial edges perform charged read/modify/
write. After a successful data write, its RAM checksum becomes authoritative.
Readers, including corruption retries, never reload stale disk expectations.
Dirty index pages are written once when persistence is requested. Consumed
scratch can be discarded without index writeback. Any data/index IO failure
invalidates the plane; there is no resume from its partly mutated state.

Generic value readers cannot access a scratch-owned region. Taking ownership
also advances its runtime revision: a reader opened before an update cannot
silently observe the replacement after release. Region incarnation remains the
stable checksum domain; revision changes do not invalidate untouched chunks.
Views cannot outlive their owner and buffers remain exclusive throughout use.

This implements the pinned-index policy. `ChunkWriter` now initializes complete
chunks in arbitrary order, using an admitted coverage bitmap independent of
checksums. It rejects duplicate/missing chunks and can hand its index image
directly to `ScratchPlane`, without an index write/read or a zero-plane pass.
Paged mutable indexes and persistent value/version ownership remain separate
work. The construction bitmap can become the smaller dirty-page bitmap after
coverage is complete.

An optional aligned payload prefix stays resident from construction through
plane consumption. It uses the same current index and update epochs, with
explicit reads/patches split between RAM and backing. Release materializes the
prefix before its index; discard skips both. The admitted product retains B
through the middle overwrite and inverse read, saving two writes and two reads
per retained byte. It still reserves conservative complete backing extents.

`edit_resident` can lend a range entirely inside that retained prefix during an
existing exclusive update. The Team first verifies every covered integrity
chunk, including unchanged edge guards. It then invokes one joined, RAM-only
callback and refreshes disjoint checksum records in parallel. The controller
updates dirty-page bits after the workers join. The callback cannot retain its
span or re-enter read/patch/finish on that plane. Corruption and pre-operation
cancellation prevent callback execution and invalidate private scratch.
The original update token still owns final finish/release; no IO or additional
arena storage occurs inside this loan.

Inverse input staging uses the already allocated all-prime coefficient bank as
its pipeline storage. Independent prime batches have disjoint spans. One Team
lane owns checked reads while at most the remaining lanes expand and invert a
ready batch. A stripe is exposed only after every prime has joined. A single
prime batch or a one-worker plan keeps synchronous execution. Read volume,
coefficient layout and the complete storage quote are unchanged. Failure returns
an empty stripe after joining work; the caller never receives a partly inverted
bank.

## Native row contract

Outer forwards have the same single-IO-owner handoff, using two packed batch
slots and one or two raw stripes. A raw stripe stays immutable until all its
prime batches have consumed it. The compact one-stripe schedule joins its last
reader before reload; the two-stripe schedule can also overlap the next read.
`enable_forward_pipeline` chooses the larger schedule if it fits, otherwise
the compact schedule if it fits, otherwise leaves synchronous execution. These
are fixed lifetime quotes, not a geometry search; no retained plane is displaced.
Source reads and packed output bytes are identical to the synchronous pass.

The raw staging quote is the largest reachable raw-cell endpoint over the
input prefix, plus 64 readable guard bytes. For the last input cell `d`,
`row = d / inner_cells`, `columns = min(stripe_cells, d % inner_cells + 1)`;
the quote is `row * raw_stride + columns * trunk_bits + 64`. Empty input needs
only the guard. Here one cell holds eight T-bit trunks, hence T bytes. The
existing TFT live-input test produces virtual zero rows without loading them.
This also removes the unused tail of a partial final row. Output Fourier support
and its canonical packed48 address mapping do not change.

The middle adapter optionally uses two packed row slots for IO/compute overlap.
`query_middle_rows` reports the serial requirement and the additional complete
pipeline requirement; `enable_middle_pipeline` admits it within an explicit
grant. The product coordinator enables it only from unused space in its existing
phase maximum, so it displaces no retained planes and changes no transfer count.
There is one IO controller and at most `workers - 1` arithmetic workers within
the existing Team, with no new thread or stack. Each slot follows checked read,
parallel unpack/compute/pack, checked write, reuse. One owner serializes all
plane metadata and transport counters; the compute scope only touches borrowed
buffers and its disjoint work. Original IO still completes synchronously on its
owner, but runs concurrently with arithmetic on the other slot. Error and
cancellation join both scopes before invalidating output or freeing storage.
The same protocol handles MUL, raw resident A, SQR, and immutable cached SQR;
the latter writes a new plane and preserves its cached source on output failure.
Retained prefixes keep their direct bulk entry; a single remaining file row
uses the serial path. Optional private counters time the file-backed section
for diagnostics without adding clocks to normal execution.

`product/modular_row.hpp` has no ISA types. Its implementation in
`kernels/x86v4ifma/modular_row.cpp` reuses the existing p48 butterflies, radix-8
DFS, degree-8 convolution and packed48 codec.

One cell contains eight polynomial coefficients. Local Fourier evaluation uses
`local_tau * z_cells^k`, with bit-reversed spectral order. Polynomial leaves
multiply modulo `x^8 - z_global^(prefix + stride*k)`, where
`stride = global_N/cells`. Local twist and global leaf wrap are distinct;
already-twisted inputs must not acquire the global prefix a second time.

Product operands can explicitly declare completion of that inner Fourier
basis. Upper and leaf forward stages skip only the named completed operands;
pointwise scaling and inverse normalization still occur once. If A and B alias
for SQR, their completion states must agree. Unary forward/inverse calls do not
accept these product-state flags. The packed-row batch also supports unary
forward construction of completed cache planes using the same arithmetic.

Supported operations are full forward, normalized full inverse and product.
Exact A==B gives SQR. Product consumes its private arrays and returns B;
the IFMA REDC compensation and inverse-length normalization are folded into
the pointwise constant. `product_scale` multiplies the product of the supplied
inputs once, including SQR. Working arrays are lazy `[0,4p)`; packing produces
canonical 48-bit residues.

Upper layers use disjoint butterfly ranges and groups of up to three radix
levels per pass, with Team joins. Cache-sized leaves use the existing native
DFS/convolution kernels. One A/B array pair is shared by workers; SQR needs only
one array. Each worker reuses one local twiddle heap for forward and inverse.
The published root bank is shared. Remaining root/coset constants have bounded
control state, without a full external-sized tower.

At leaf size 4096, worker scratch is 32,832 bytes; 16 workers need 525,312 bytes.
The query supports the mathematical order through 2^32 cells while keeping
control/twiddle space bounded. Its row arrays are still explicitly charged and
must fit the caller's grant; this does not promise a resident 2^32-cell row.

Codec expansion can run backward and compaction forward at the same base, so a
second packed copy is unnecessary. Packed reads require 16 readable padding
bytes; disjoint leaf subspans do not inherit a neighbor's padding implicitly.
The ordinary in-place codec adapter is serial. `p48_rows_packed` additionally
keeps a Team region across a sequence of independent packed rows: workers
unpack directly from the resident sources, run the same row arithmetic and
pack directly into B. It reuses the existing single row pair and worker heaps,
without reserving storage per row or per worker. Preparation and row boundaries
have joined barriers. Each store covers exactly48bytes; the final load is masked
to48bytes, so this interface requires no source overrun. Padding between rows
is preserved. Partial/mixed transforms are not added by this interface.

## Stored middle-pass driver

`external/middle_rows` is the production connection between scratch planes and
the row kernel. Its first wire adapter handles contiguous packed48 rows with
explicit offsets/strides. Rows are outer Fourier outputs **before** the cross
twist, ordered by bit-reversed outer frequency. The driver derives each natural
frequency/coset, reads verified A/B, expands in place, executes a full row
product, compacts B and overwrites its original backing. Its updated checksum
index may stay in RAM for the next pass.

MUL preserves A backing. SQR uses one plane and one array. The quote covers row
arrays, prepared control and worker scratch; plane indexes, IO buffers and the
shared context/Team are separate named grants. No third complete product plane
is introduced. The entire input plane need not fit in the row grant.

## Evidence and next connection

The focused gate uses independent polynomial multiplication for small rows,
sparse full-output references for large rows through 131072 cells, scaled/SQR
and distinct local/global-root cases, parallel forward/inverse round trips,
and in-place codec round trips. It also runs the production stored-row MUL/SQR
driver with cross-chunk guards, checks A preservation, tests authoritative dirty
indexes and stale readers, and injects data/index write failures. Allocation
watches and the production ISA/import audit are part of the gate. This is
correctness evidence, not measured performance parity with YC.

## Integer forward producer and inverse stripes

`external/forward_planes` now composes verified integer reads, the existing
wide/T80 decoder and outer TFTs. A raw stripe is read once and retained across
prime batches. Wide-codec byte funnels are shared within each batch. The
kernel writes canonical packed outer Fourier prefixes into row-major planes;
the chunk writer writes each final chunk exactly once in producer order.
No complete-plane transpose or preliminary zero-fill pass is inserted.

Stripes simultaneously align raw input and packed output to the integrity
quantum. For the usual T divisible by16, the minimum is1024 cells; for the
other supported byte-aligned T values it is2048. Larger explicit stripe sizes
are allowed. The initial per-component default is an alignment minimum, not
a measured optimum or a substitute for whole-budget IO-run selection. Small
one-stripe fixtures read the source contiguously and charge row padding.

The immutable input checksum index is verified and pinned once, preventing
metadata rereads during strided access. Output index images remain in RAM for
the middle pass. Source/index/coverage/IO grants are explicit in their respective
queries; grouping primes changes working memory without rereading the raw stripe.

`external/inverse_planes` gathers product-plane stripes into their final64-bit
coefficient slots, expands packed residues in place and applies normalized
outer ITFTs with a proven zero time tail. It returns a borrowed
`[prime][row][column][lane]` coefficient stripe for CRT. Prepared transform state
is reused across stripe calls. It reads each product payload once and performs
no coefficient-plane write. General mixed-time tails and a partial last inner
row remain separate capabilities; the present recipe rounds that row up.

The production forward → stored middle → inverse-stripe sequence passes
independent modular convolution checks at T64/72/80/128/224, prime counts1/4/6/10,
ragged prime batches, truncated outer prefixes, one/four workers and non-power-
of-two input lengths. Large cases use polynomial evaluation checks; smaller
cases compare every coefficient. Byte counters confirm one complete source
read, one write of each forward plane and one product-plane read for inversion.
This is modular-convolution evidence, not yet a carried integer product.

The [complete integer coordinator and CRT/carry output](ooc-integer-product.md)
now pass simulated MUL/SQR and whole-pass byte checks. Bounded product selection
and retention also pass with caller-declared total-context costs. Stored
consumers, whole-job admission, durable IO and 5B qualification remain; component
success is not a performance-parity claim.
