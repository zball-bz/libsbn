# Radix conversion

For a u64/u128 decimal integer, use the direct primitives in
[decimal.h](../include/sbn3/decimal.h): they bypass the general query/bind
lifecycle and have no scratch requirement. [Usage and native inlining](decimal.md).

include/sbn3/radix.h supports bases 2..64. The input is a nonnegative dyadic
value X = M * 2^exponent2, with little-endian uint64 mantissa limbs; the caller
carries the sign. This covers integers and finite binary fractions.

Formatting returns most-significant-first digit bytes, with separate integer
and fractional regions. The decimal point is not stored. By default bytes are
digit values; a caller-provided alphabet maps them to characters.

- EXACT treats X as exact and returns certified truncated fractional digits.
- ENCLOSED treats X as the lower endpoint of [X, X + 2^exponent2) and returns
  only digits determined by the whole interval. It requires exponent2 <= 0.
  The certified prefix can be shorter than the mathematically longest one;
  callers must use the returned digit count.

Parsing returns M = floor(value * 2^fraction_bits), exactly. Invalid input reports
the first invalid position; it is not a successful arithmetic value. The header
specifies alignment, padding and output capacities.

The lifecycle is query, prepare arena, bind, execute, unbind. Large bindings prepare powers
and product programs. A radix binding can execute repeatedly for its declared
shape. The repeated option lets planning distinguish one-shot and repeated use.
Both use the same public API. Eligible single-worker formatting uses a balanced
ladder, including reused fixed-shape conversion. Parallel stages use their
existing balanced eligibility rule too; smaller trees retain the regular grid.
Integer reciprocal caching is independent of that geometry. Fresh formatting retains only
the powers its tree needs. Staged cyclic products and their rare carry witnesses
share a pool in separate phases; ENCLOSED omits the global EXACT tie product.
The storage and interval arguments are in [radix-format.md](radix-format.md).

Small integer calls use compact descriptors and the same lifecycle. Parsing up
to 256 output limbs uses a fixed storage bound and local recursion over published
powers, without a product-program planner. Plain integer formatting uses a word
kernel or a direct normalized fractional leaf; powers of two use bit packing.
The general planner is a separate function body, keeping its large stack frame
out of these entries. Bindings retain only the fields execution needs.

Decimal numeric bytes and the standard ASCII alphabet have specialized leaves.
Up to 38 decimal digits use scalar packed-byte reductions and a u128 combine;
wider leaves use IFMA with a fixed SIMD map back to u64 limbs. Digit validation
is retained, and partial input loads never read outside the input span. The
tiny entry is separate from the wider leaf's stack frame. Standard decimal
ASCII uses a published descriptor; no per-call decode table is constructed.
Decimal emission uses exact range-specific reciprocal products and packs
quotients and remainders together. The same emitter serves integer and SRT
leaves, with arbitrary alphabets retaining their final byte mapping.
For small power-of-two formatting, a padded leading zero is resolved from the
input bit length instead of scanning the padded digit prefix.

One tree conversion plans once. The tree plan is far larger than a plan value, so bind
assembles it again; the plan value (4096 bytes) carries the sealed layout, the
Newton plan of the scaling reciprocal and a transcript with the winning candidate
of every product search the query ran. Bind repeats each winner's single backend
query instead of the search, checks every entry against its request and the
identity it stands for, and compares the identity of the whole assembly with the
sealed one; an entry that does not reproduce, or a transcript without room, is
planned by the search again. Program preparation, the rail squarings and the power
chain use the same recorded winners. Nothing is kept after unbind: the transcript
is part of the caller's plan value. Direct bit/leaf bindings borrow their
caller-owned resident range, as compact division does. Checked builds retain
range, lease and full plan-integrity checks. Unused opaque plan bytes are
unspecified; they are neither initialized nor read by the library.

Small integer formatting can normalize with one local product by a published
exact reciprocal, then use the existing exact fractional leaf. The +3/4 bias
separates the integer from either digit boundary. The word-work crossover is
160 input-limb × maximal-radix-word operations; repeated calls retain the
existing tree crossover from 32 input limbs. The bounded reciprocal bank covers
the direct domain and its precision guards. Its memory is reported as shared
constants, not hidden as zero-cost workspace.

Larger fresh integer formatting normalizes its root with fused approximate
division; fresh fractional parsing uses the same scaling episode after parsing. Repeated
bindings retain the reciprocal in aligned private storage, and cache its spectrum in the serial FFT band.
For small retained trees, preparation copies the exact published reciprocal into
that storage; one-shot normalized leaves borrow the published value directly.
Short reciprocals share division's local inverse kernel. Integer formatting's
direct path divides by the largest whole radix power below 2^64 using a prepared
word inverse. Its fresh crossover is 2600 input-limb × radix-word operations;
repeated trees have a lower crossover. Short fractional formatting compares
direct word/IFMA work with tree setup and emission work, with separate fresh
and retained-tree costs.

Parsing assembles 64/128-digit blocks with IFMA polynomial dot products and exact
carry propagation. Independent accumulators hide multiply-add latency. The same
blocks feed the bounded leaf and the local or general integer trees. Small local
merges trim high zero limbs and use the existing word/U52 work crossover.
Small trees carry a compact split plan and call the same local products directly;
their scratch bound follows the live child/product phases. Fractional results
use the shared prepared word/U52 divisor, after cancelling the radix's known
power of two. The compact domain is bounded by published powers and a small
value/precision limit; larger division recipes keep the general service.
Reused calls retain the general tree where a power or reciprocal spectrum is
useful. Larger mixed
values have two roots sharing powers and programs: the integer part goes
directly to output, and only the fraction is scaled. Parallel child values
partition one slab and merges overwrite consumed inputs; work is sized by the
maximum simultaneous phase. Bind stores only the plan's live classes.

All power-of-two bases share bit-pack/unpack kernels; alphabets use the same
vectorized byte mapping and validation. These paths and the common parser/SRT
leaves serve both integer and finite-dyadic APIs.
tests/integration/radix_plan_replay_test.cpp is the gate of these statements.

See examples/radix.c for an executable caller. The conventional alphabet is
0-9 a-z through base 36, and 0-9 A-Z a-z + / for larger bases.

For individual conversions, the [one-shot entries](radix-once.md) dispatch small
requests directly and retain the existing planned path for larger work. They
return the output layout/width, so consuming the result needs no separate query.
