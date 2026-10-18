# Store execution: implemented boundary

2026-10-02. Private implementation in `src/external`; no new public OOC API is
declared yet. This is the first executable storage boundary for the
[implementation goal](ooc-implementation.md).

## Transport and value IO

Aligned complete chunks now transfer directly between the transport and the
caller's stable source or discardable destination. Value and scratch-plane
readers verify those destination bytes in place, with the same one-retry rule.
Scratch updates likewise write complete replacements directly; a retained RAM
prefix needs one copy rather than a bounce-buffer round trip. Unaligned edges
and a short final chunk still use the quoted IO buffer. Failure leaves read
destinations discardable and never publishes a partially updated value.
Middle row arrays and inverse coefficient banks use page-aligned starts to
enable this path. Transport limits, checksum domains and submitted-byte budgets
are unchanged; any layout padding remains part of the complete workspace query.

Numeric m61 integrity uses a private AVX-512 bulk kernel in the native bundle.
Eight independent word chains multiply by `2^24 mod (2^61-1)` per vector; the
kernel rotates/folds 61-bit residues and combines the lanes with their original
word weights. The scalar version remains the short-input path and reference.
This changes neither the fingerprint nor the stored representation. Vector
types remain private, and the bulk TU's manifest explicitly requires AVX512F.

`Transport` has synchronous completion callbacks for bounded reads/writes.
Returning, including on an error, means no original request still owns the
buffer. A future asynchronous provider must provide the corresponding explicit
join. The current gate uses a memory-backed device; there is no real-device
benchmark or durability claim.

All buffers and descriptors are caller-owned. Store uses neither ordinary heap
allocation nor a per-chunk object table. Data transfers batch multiple integrity
chunks; checksum granularity does not force one syscall per chunk. Readers and
writers each borrow a data buffer and one index page. These remain exclusive
while their cached state is in use; reopen a reader after another phase has
reused its buffers. Source/destination values are disjoint from those buffers.

The implemented value layout is:

- Little-endian byte/limb payload; 64 KiB allocation extents.
- 16 KiB integrity chunks with explicit final covered length and zero physical
  padding. Unused allocation padding is not written unnecessarily.
- One XXH3-64 record per chunk; 4 KiB index pages with the specified 32-byte
  magic/version, index-instance, first-record and checksum header.
- Chunk hashing binds region UUID/incarnation, logical chunk index and covered
  bytes. Metadata checksums also validate the expected identity and range.

The unmodified upstream xxHash 0.8.3 header and license are pinned under
`third_party/xxhash`. The native checksum translation unit uses AVX-512 with
`XXH_NO_STDLIB`; production audit verifies the vendor hashes and absence of
allocator/page imports. Chunk digests remain exactly `XXH3_64bits(header ||
payload)`. The fixed 48-byte header and first 16 payload bytes form one local
stripe; subsequent long-input stripes are read directly from the payload using
upstream accumulation, scrambling and finalization. Short inputs use a bounded
240-byte local concatenation. No checksum-format/version, padding, coverage or
identity change is involved. Metadata remains upstream one-shot XXH3. The ISA
is private to the TU and its explicit native build manifest; public interfaces
contain no vector types. This implements the selected XXH3-64 integrity scheme,
without a content-addressed value hash.

Value/Chunk writers may collect optional first/last nonzero-limb metadata from
their already resident finalized payload. This metadata is valid after seal;
it adds no disk index records or normalization IO. Precision consumers retain
the same immutable Region and index while describing a shorter source window.
Future durable manifests must preserve the signed exponent/window metadata
alongside their backing reference; runtime window handles alone are not a
checkpoint format.

Finalized value output may project a chunk-aligned-start interval of a larger
logical producer stream into its own smaller Region. Integrity records cover
only retained chunks in that Region's coordinates. The producer remains
responsible for full-stream carry and arithmetic validation; it cannot use
skipped storage writes as a reason to skip required arithmetic checks. Kept
coverage and failure invalidation use the ordinary writer state machine.

## State and failures

The sequential finalized-value writer transitions:

```text
empty/invalid -- fresh identity --> building -- open --> writing -- seal --> sealed
                                                       \-- failure/abandon --> invalid
```

Opening a second writer is rejected. Sealing requires complete byte coverage
and successfully written checksum pages. Zero is a valid checksum value;
initialization is tracked by the sequential coverage cursor instead.
`sealed` means initialized and integrity-indexed. It does **not** mean durable
commit or an arithmetic certificate. There is no destructor IO or implicit
root publication. Reusing failed storage requires a fresh incarnation.

The reader verifies pages/chunks before returning success. It makes at most
one immediate retry per failing metadata-page load or data-chunk verification.
Short IO returns an error rather than looping indefinitely. On any read error,
the destination is discardable; source versions stay unchanged. Valid checksums
on a wrong index identity or page range are rejected.

The transport ledger distinguishes submitted and completed bytes, includes
metadata, and charges failed submitted writes against the write allowance.
Cancellation prevents further dispatch at bounded IO boundaries. A failed
writer cannot subsequently seal or expose a successful result.

## Stored-boundary resident multiplication

`ResidentProductPlan` quotes both complete operands, result capacity, native
tables/scratch and the IO buffers. Its owner adds context/shared/Team costs to
the total grant. `run_resident_product` loads verified inputs, runs the existing
allocation-free arithmetic program, checks the m61 product identity and writes
the result through the finalized-value writer. Input allocations remain sealed
and unchanged; IO failure invalidates the unpublished output.

This is **resident promotion with stored boundaries**. It is useful inside the
external subtree frontier, but does not execute a transform larger than its
RAM grant. No hidden allocation, mmap-based substitute for explicit OOC, or
unquoted low-RAM fallback is used.

For a successful whole-value read/write, normal-path transferred bytes are
exactly padded payload bytes plus index-page bytes. The focused gate checks
those totals, odd-size ranges, index-page crossings, corruption/reread, short
IO, cancellation, write caps, source preservation, native results and allocation
watches. These are correctness tests, not performance qualification.

## Next implementation boundaries

Mutable scratch ownership, pinned current indexes and a native full-row middle
pass are now implemented; see [the row/plane contract](ooc-modular-rows.md).
`ChunkWriter` adds write-once out-of-order initialization with separately quoted
coverage and index images. It hands scratch indexes directly to their next
owner; `ValueReader::open_pinned` retains verified immutable source indexes
across strided reads. Neither path allocates hidden storage.

Construction may retain a chunk-aligned scratch prefix in caller-owned RAM.
The writer and then the exclusive plane own those bytes through all accesses;
disk contains no usable copy until materialization. Reads/patches split at this
boundary and verify the authoritative RAM checksum image. A cached corruption
retry must not fall back to unwritten disk. Explicit persistence/release writes
retained payload before dirty index pages; discard writes neither. Failures
invalidate the scratch version before generic value readers may use it.

The [complete spilled integer coordinator](ooc-integer-product.md), CRT/carry
and budget-driven retention now run in simulation. `ValueStore` now provides
preallocated region ownership, explicit immutable references and contiguous
extent reuse; [stored finite-series merges](ooc-series.md) consume those handles.
Original-request/checkpoint pins prevent premature reuse. The
[object-file adapter](ooc-files.md) now provides fixed virtual slots with a
separate live-byte allowance, Linux memfd/directory IO and explicit collection.
This removes contiguous-bank fragmentation without copying data. The
[generic job](ooc-job.md) composes series/terminal/SRT. The
[checkpoint layer](ooc-checkpoint.md) now adds protected import, namespace
barriers, commit/resolve and phase restore. Async transport and large-device
qualification remain; plain sealing still does not imply durable publication.
