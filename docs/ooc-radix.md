# Stored fractional radix output

`external/format` implements the first bounded stored radix path. It uses the
existing SRT split arithmetic and resident format service, with stored odd
powers and projected right-child products above the resident frontier. It is
part of the [OOC implementation](ooc-implementation.md); this is a private
provider, not a new public C ABI.

Once a resident leaf's carry seam is resolved, its complete interior chunks
are submitted as one contiguous range through the existing bounded writer.
Only chunks crossing leaves use the seam buffer. This preserves reverse leaf
order, write-once coverage, checksums and the exact byte ledger while allowing
full transfer-sized writes. No larger output staging buffer is reserved.

## Value and output contract

The input is a nonnegative, limb-aligned dyadic interval
`[X, X + B^-P)`, where `B=2^64`, represented by a stored magnitude with binary
exponent `-64P`. Its integer part must fit one u64. Bases 2 through 64 and the
ordinary optional digit alphabet are supported. The caller supplies enough
binary fraction words for `radix::node_limbs(base,ceil(D/64))`.

The result contains the integer word and an immutable requested-width fraction
digit value. Only `certified_digits` bytes at its beginning may be exposed as
certified output. Near a radix boundary this can be shorter, including empty;
it need not be the longest common prefix. Bytes beyond it remain private
provisional digits. This is the existing ENCLOSED interpretation, not general
EXACT formatting or an implementation of all four stored radix flavors.

Root normalization is a high-word view of the binary fraction. Integer words
and unused low fraction words are excluded without a temporary file. The
original value and checksum domain are preserved. Non-word-aligned exponents,
multiword integer parts and stored radix parsing remain future consumers.

## Bounded planning

`FormatOptions.radix.repeated` selects a regular power-of-two grid. A fresh
request repeatedly halves its fragment count upward to a seed of at most
16 fragments, then uses that seed times powers of two for the large split
powers. Thus one dyadic odd-power rail serves the request. It does not search
alternative split points, RAM ratios or NP grids. The current runner rebuilds
its private powers on each execution; retained cross-call prepared state is
not implemented by this option.

Classes are memoized by fragment count in a bounded directory. A resident
frontier must admit its complete input copy, digit output, and native service
storage. If it does not fit, the fixed split produces two child classes.
The external multiplication uses the existing bounded selector once per live
class. Powers use the existing resident multiplication when its complete
episode fits, otherwise a stored square. Construction-only high zero words
can be omitted only after a full arithmetic/residue check.

The common arena alignment is applied before filling spare RAM with retained
planes. The total RAM quote includes the complete plan, control allowance,
published engine/radix constants, output integrity/coverage metadata, bounded
IO, seam assembly and the largest phase. Other context, team and Store metadata
are supplied by the caller in `other_live_bytes`.

The write quote sums the rail construction, all selected right-child products
and one finalized digit output. Backing includes the preserved input, private
rail, output, live right-descent values and temporary planes. Retaining a left
view does not allocate a second value. These are aggregate lifetime bounds;
arbitrary first-fit fragmentation still needs the whole-job placement layer.

## Traversal and seams

A node holds a fraction with its binary point above its top word. The left
child is a high view. For `base=2^s*odd`, multiplying the low
`parent_words-s*left_fragments` words by `odd^(64*left_fragments)` produces the
right child as one exact word window. The same rule removes the trailing
binary zeros of bases such as 10. Power-of-two bases need only views.

Traversal visits right, then left, using one shared phase region. Stored
parents remain immutable while their right child executes. At a native
frontier the existing service produces its digits plus an eight-digit overlap.
The next frontier's corrected head is already known. The current overlap must
equal that head or precede it by one modulo `base^8`; a wrap increments the
current digit buffer before publication. The corrected head then feeds the
next frontier to the left. Native frontiers may use the entire team; external
siblings are not concurrently materialized in this initial executor.

The reverse output assembler holds one integrity chunk. It combines any seam
fragments in RAM and submits complete finalized chunks in descending address
order. It does not read or patch a previously emitted digit chunk. No full
digit array or per-depth fraction array stays resident across frontiers.

The existing SRT precision/overlap invariant encloses the desired extended
integer in the reconciled result and its successor. A conservative two-unit
bound after aligning the final eight-digit guard also covers the supplied
input interval. A guard unable to overflow certifies every requested digit;
otherwise the result stops at the common prefix with the successor. Exact
terminating-boundary repairs are deliberately not implied by this contract.

## Execution and failure

`run_stored_format` borrows a prepared, initially unleased aligned arena range
exclusively through return. Fixed IO/metadata has one lease; native frontiers
bind in disjoint phase partitions after all earlier arithmetic has joined.
No ordinary allocation or implicit page-residency change occurs during query
or execution. The caller may explicitly prepare/trim storage between algorithms.

Output is published only after complete coverage and integrity sealing. IO
failure or cancellation releases all private powers, children and output;
the input survives. Mathematical seam/carry/capacity violations remain fatal.
The output and rail are currently ephemeral Store objects, not durable format
checkpoints. Actual file/async IO and recovery remain separate work.

## Current evidence and limitations

Simulation gates cover bases 3, 7, 10, 15, 16, 63 and 64, both grids, resident
and forced stored paths, 35,011-digit multi-chunk output, all-ones input and
terminating/crossing radix boundaries. Independent word-by-word exact fraction
conversion checks the certified prefix and bounds the entire provisional
string to the exact string or its predecessor. Observed writes equal the
quote; allocation, cap and cancellation cleanup checks pass.

The stored constant test chain now reaches 200 decimal digits of e and
200/1,151 digits of pi. Pi's binary value matches an independent Machin formula;
an independent exact word conversion checks the decimal output. The e terminal
adds the omitted k=0 term: Hyperdescent's existing Euler formula starts at 1
and returns e-1. This exact +1 is fused into fixed binary output. Earlier
finishing receipts labeled that unadjusted ratio as e; their frozen artifacts
are preserved, and the current test checks the integer part as well.

The 5B-digit/4 GiB metadata-only decimal query selects 6 classes, 23 rail
powers and 7 resident frontiers, with seed 10. It quotes 4,293,713,504 total
resident bytes, 20,883,767,296 peak backing bytes and 28,493,787,136 writes,
including final digits. No operands, arithmetic or Store IO ran at this scale.
No real SSD operand/swap writes or performance benchmarks were performed.

Those figures describe the initial bring-up, not the current execution plan.
The external split now compares LIN with a shorter cyclic middle window, as
specified below. Shared power spectra and fused child emission remain open.
Resident frontiers currently use the complete EXACT native formatter for their
local dyadic value, including its boundary-repair reservation. A direct SRT
frontier adapter can remove that redundant reservation. Power construction's
resident path now uses the square program with one source array/read and one
forward transform. Frame-owned square and cyclic programs dispatch through
their complete recipe, including cyclic carry folding. The remaining limitations
must be addressed or measured before claiming YC-level efficiency. The
[whole-job compiler](ooc-job.md) now admits combined phase resources and error
bounds. Physical placement, durable IO and the verified 5B fresh comparison
are tracked by the current experimental whole-job receipts.

## Exact middle windows with wraparound

Let the source lengths be n and m words, and request product words `[s,e)`.
Choose a native cyclic period `K >= max(n,m,e,n+m-s+1)`, and compute modulo
`B^K-1`. If `K < n+m`, the wrapped high part is below `B^h`, where
`h=n+m-K`. It can affect `[s,e)` only by carrying through the complete gap
`[h,s)`. If any finalized word in that gap is nonzero, the window is exact.
A gap of zero words is ambiguous: it can be an original zero gap or the
aftermath of a carry through all-ones. The test therefore never accepts it
without the precompiled original LIN product. If the complete product fits
in K, no gap witness is necessary.

CarryOutput observes the gap while finalizing its existing chunks, including
chunks outside the stored projection. The common path adds no output reread,
repair pass or payload buffer. An ambiguous private child is retired, and the
already admitted LIN recipe produces its exact replacement. This is a
deterministic arithmetic branch, with no runtime planning or precision change.
Tests construct both a wrapped carry that changes the first requested word
and zero-valued gaps, and compare complete converted digits independently.

Each class compares the complete analytic cost and normal writes of one LIN
recipe and one ring recipe. Ring selection uses one legal period per compiled
prime basis, without a T/geometry grid. The selected ring may not increase
normal writes. The workspace and backing grant cover the larger of both
recipes; `write_bytes` includes ring **plus** witness writes in the worst case.
`normal_write_bytes` reports the no-witness path separately. Runtime results
report attempted middle products and actual witness count. Never interpret the
larger worst-case write allowance as traffic that was actually submitted.

General ring products cannot use the full-integer m61 identity: reducing by
`B^K-1` changes that residue. This middle provider checks the complete carried
ring product modulo `2^64-1`, which divides every supported limb-aligned ring.
Inputs are checked during existing source reads and output during existing
carry finalization, using a private vector word-sum kernel. LIN and signed
cyclic-residual paths retain m61. Transport stripes retain XXH3-64. The ring
check detects arithmetic corruption but is not an order-sensitive hash or a
universal probability bound; independent exact tests cover the permutation,
projection and carry rules. No hash is used as a mathematical correctness proof.

## Native frontier overlap

A private native formatter entry point returns exactly `64 F` digits together
with the SRT's existing eight-digit overlap. It avoids requesting `64 F + 8`
digits through the public API, which would construct another 64-digit fragment.
Small recipes keep the public entry point. The ordinary exact-prefix witness
remains planned: the emitted prefix is exact for the supplied dyadic, while
the overlap word may be one unit low. No public radix ABI changes.

The external binary cut contributes at most one additional unit. Therefore
the next frontier's leading word differs by zero, one or two modulo `b^8`;
crossing that modulus increments the preceding prefix once. At the global
right edge the certified-prefix guard increases from two to three units.
Independent conversion tests cover random inputs, long carry chains and
terminating dyadics across native/external seams. The extra uncertainty is
accounted for in certification, never hidden by relaxing an exact-output claim.
