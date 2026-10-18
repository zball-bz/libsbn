# Object-file placement and Linux IO

`ValueStore::open_slots` and `FileStore` provide the first file-backed Store.
They preserve the integer/plane checksum formats and numerical executors.
It supports ephemeral files and protected immutable imports. The
[checkpoint layer](ooc-checkpoint.md) provides durable job-root publication and
restore. Provider callbacks complete synchronously, but run concurrently with
arithmetic on dedicated lanes of the existing Team. FileStore advertises duplex
operation and up to four independent reads/writes per direction. Mutable spans
must be disjoint; lifecycle/registry changes and snapshots remain joined.
Per-direction ledger locks cover admission/completion, never the syscall.
The multiplication pipeline needs neither extra user threads nor per-request
user allocations. An optional Linux io_uring provider supplies joined batches;
ordinary pread/pwrite remains the default.

## Optional queued IO

After opening a quiescent FileStore, call `enable_uring(depth, workers)` and
then obtain a new `transport()`. Depth is a power of two from 1 through 64;
the default fallback-worker limit is one. Preparation can fail and never
silently substitutes another provider. Scalar callbacks remain available.
The caller admits `uring_memory_bound(depth)` in its context allowance in
addition to registry/control storage. This bounds the two userspace ring
mappings, not all kernel accounting (request objects, io-wq stacks or filesystem
metadata). At QD16 the mapping bound is 32 KiB. Kernel resources and fallback
workers belong in the host resource allowance; they are not free Team lanes.

There is one read ring and one write ring. A reader batch and writer batch can
overlap; competing owners of one ring serialize. A batch holds at most 64
requests and uses fixed stack descriptors. SQPOLL, IOPOLL, fixed-buffer/file
registration and a separate user IO thread are not enabled. The raw-ring owner
is private to the Linux runtime; numerical kernels do not see ring operations.

`read_batch_exact`/`write_batch_exact` validate the entire group before
admission. Read destinations and write extents must be disjoint. Write caps
charge all admitted requests conservatively, even if a provider failure leaves
some unsent. The provider joins every original request before returning,
including errors or late cancellation. Only then can checksums run, a ready
slot be published, or buffers/objects be recycled. No early return relies on a
cancel CQE as proof that the original request stopped. Closing/reconfiguring
the Store while IO owns buffers is a caller lifetime violation.

Inverse gathers and multi-MiB middle rows use the batch capability. Large
aligned complete output chunks also use it. Retained chunks issue no requests;
partial/projected chunks retain scalar handling. Parallel prime writers keep
their existing Team lanes because checksum/copy work also needs parallelism.
Transport wrappers which replace `context` must forward or clear both batch
callbacks. A pread/pwrite interposer alone cannot simulate this provider.

## Placement without compaction

`query_slot_store(slots, live_byte_limit)` chooses a power-of-two virtual
address span at least as large as the byte limit. Each metadata slot owns
one such interval. A Region uses only its actual data/index allocation inside
that interval; the unused virtual addresses reserve no RAM or file bytes.
The file backend maps the slot to a separate compact file and translates the
address to a file-local offset. A file is resized to its actual Region
allocation, never to its virtual span.

Two checks admit an allocation: a free descriptor exists, and its allocation
fits the remaining sum-of-live-bytes allowance. Since the virtual span is at
least the entire allowance, any admitted object fits its slot and cannot
overlap another object. Variable object sizes and release order cannot create
an artificial contiguous-bank fragmentation failure. No data relocation or
packing search is needed. The existing contiguous Store mode remains for
flat-bank clients and older simulation tests.

This establishes the physical *object placement* part of the aggregate byte
proof. It does not guarantee that a filesystem has enough free blocks or that
file creation/IO succeeds. The caller still admits available host/filesystem
resources; OS errors remain fallible. Slot-registry and ValueStore metadata
are caller-owned, queried separately, and belong in the context's other-live
RAM allowance. Virtual address capacity is distinct from physical byte budget.

## Lifecycle and dispatch

A Region lifecycle creates its backing before a reservation is returned. A
failed creation consumes its incarnation but leaves no live Store handle.
Publication marks value files read-only to this transport; scratch files stay
mutable for the existing in-place B-plane pass. Descriptor lookups use a
shift/mask and a fixed registry, with no allocation or filename lookup per IO.

Last-reference release only marks the file retired. It performs no IO. Before
any new creation, `collect()` reclaims all retired files; it is also callable
explicitly. Therefore delayed retirement cannot inflate the historical claimed
byte peak: no additional object is created before reclaiming those bytes.
A collection failure blocks new creation and is reported. Files retained by
input/checkpoint references are not retired.

`FileStore::close()` requires no live values, collects remaining retired files,
then closes its owned directory descriptor. The destructor performs no IO and
requires explicit close. Linux close errors are reported without retrying the
descriptor. The registry records nominal claims, file counts and actual syscall
bytes separately from the transport's submitted/completed request ledger.

## Backends

- `open_memory` creates Linux memfds. These use real file descriptors and
  pread/pwrite/ftruncate/fsync calls without SSD operand writes.
- `open_directory` duplicates an exclusive caller-supplied directory descriptor.
  Each owned filename encodes the Region identity. Creation uses exclusive
  create, no-follow and close-on-exec flags. DIRECT is explicit and never silently
  falls back to buffered IO. Only filenames created by this registry are removed.

Both adapters reject stale or retired slots, cross-object ranges and writes to
published values. Short IO is returned to the existing transport contract.
No implicit synchronization or root publication occurs on value sealing.
`sync(region)` is explicit and fallible. Directory ownership and object identity
remain stable even if the caller closes its original directory descriptor.

## Evidence and remaining work

Memfd and named-tmpfs round trips reconcile bytes, check compact file lengths,
reject writes to published values, detect injected corruption, execute sync,
and check pins and deferred cleanup. A deliberately fragmented allocation
sequence frees two disjoint 3Q objects, then admits a 5Q object under the same
12Q byte cap. The fresh incarnation prevents stale identity reuse. Creation
refusal leaves metadata unowned and a subsequent reservation succeeds.

The same production generic job tests pass with these backends: e200 and
BinaryBBP200 on memfds, pi1151 through named tmpfs files with DIRECT requested.
Independent numeric references, injected write failure/retry, cumulative write
charging, allocation watches and phase cleanup remain active. All115/298/151
created objects are explicitly reclaimed. Actual file claims stay within the
whole-job backing quote and syscall write bytes equal completed transport bytes.
The contiguous-value regression also passes.

Tests verify a tmpfs filesystem before creating named operand files; they never
fall back to an SSD path. DIRECT acceptance on tmpfs does not qualify NVMe IO
behavior or bandwidth. No real SSD operand/swap writes or performance benchmark
ran. One fixture initially forgot that data and index extents are independently
rounded to64KiB; the test sizes were corrected, without changing the allocator.

Protected Region import now remaps slots without changing checksums. Explicit
adoption permits ordinary retirement after verification; close-preservation
retains committed files and forbids further creation. The checkpoint layer adds
manifests, publication resolution, scoped orphan GC and process-crash gates.
The io_uring gate adds mixed successful/error/EOF completions, ring reuse,
budget/alias/readonly/cancellation checks, retained-plane corruption rereads,
multi-MiB writes and complete same-plan products. Native allocation watches and
ThreadSanitizer cover the user-side ownership paths. Kernel DMA/io-wq accesses
are not instrumented by ThreadSanitizer; the explicit join contract remains
necessary. Broad performance qualification and verified5B/fresh YC qualification
are separate from these correctness gates.
