# Small decimal primitives

`include/sbn3/decimal.h` provides unsigned ASCII decimal conversion for u64,
u128, and small values of 1–8 little-endian u64 limbs. Both integer and pure
fractional conversion are available. There is no context, plan, binding, heap
allocation or caller scratch. The multiword leaves use bounded stack storage;
the original u64/u128 integer entries remain register-sized primitives.
The general radix service handles arbitrary lengths, bases, alphabets and
binary exponents, and automatically uses the small decimal algorithms where
they are faster.

```cpp
#include <sbn3/decimal.h>

char text[39];
sbn3_uint128 value{123, 1}; // value = lo + hi * 2^64
char *end = sbn3_decimal_write_u128(text, value);

sbn3_uint128 back;
auto status = sbn3_decimal_read_u128(&back, text, size_t(end - text));
```

Formatting writes exactly the result characters, returns the end pointer and
does not append a NUL. Zero produces `"0"`. Buffers of 20 and 39 bytes suffice
for every u64 and u128 respectively. Smaller buffers are sufficient when the
caller knows the result fits. Input and output spans are caller-owned and
disjoint; no SIMD alignment or readable padding is required.

Parsing consumes the complete span, permits leading zeros, and leaves output
unchanged on error. Empty input, signs, whitespace and non-ASCII digits are
invalid. `SBN3_DECIMAL_OVERFLOW` reports a valid decimal integer outside the
destination type. Invalid text takes precedence if a span is both too large
and contains a nondigit. There is no runtime inverse or table preparation.

For a bounded multiword integer use `sbn3_decimal_write_small` and
`sbn3_decimal_read_small`. The `limbs` parameter is both the input width and
the parsing destination capacity, between 1 and `SBN3_DECIMAL_SMALL_LIMBS`
(8). Leading zero limbs are allowed. Parsing writes all destination limbs
on success, including high zeros. An integer formatting buffer of `20*limbs`
bytes always suffices. No input or output SIMD padding is required.

The fractional entries interpret the words as `value / 2^(64*limbs)`:

```cpp
uint64_t half[2]{0, uint64_t(1) << 63};
char fraction[5];
sbn3_decimal_write_fraction(fraction, half, 2, 5); // "50000", no "0." or NUL

uint64_t eighth[2];
sbn3_decimal_read_fraction(eighth, 2, "125", 3); // exactly 1/8 at 128-bit scale
```

Writing emits exactly the requested count, including leading/trailing zeros,
and truncates toward zero. Reading returns exactly
`floor(0.digits * 2^(64*limbs))`; it does not round to nearest. Empty fractional
text represents zero. Invalid text leaves the complete destination unchanged.
These are fixed-point primitives, not shortest-format IEEE floating-point
converters. Use `radix.h` for a mixed value `M*2^exponent2`, partial-word binary
precision, other alphabets, or sizes above the leaf API's limit.

## Ordinary and inline linkage

The normal `libsbn_v3.a` contains these C entry points. An ordinary caller
makes one call into a leaf primitive, with no internal service dispatch.

For a compatible native Clang caller, a small optional ThinLTO archive makes
the implementation available to the optimizer while keeping intrinsics out of
public headers:

```sh
python3 tools/build.py decimal-inline
clang++ -O3 -march=native -flto=thin -fuse-ld=lld -Iinclude app.cpp \
    build/native/libsbn_v3_decimal_inline.a -o app
```

This archive can be used alone for these primitives. If the application also
uses other library operations, link it before `build/native/libsbn_v3.a`.
Use the pinned Clang toolchain and a caller target compatible with the native
kernel's features. A baseline caller can instead use the ordinary archive;
the public C header itself has no ISA requirement. An opaque binary call
cannot be promised zero overhead. The native ThinLTO test checks the stronger
case: public-header calls become arithmetic directly in the caller.

The library remains native-only in this release. Alternate architectures can
implement the same scalar signatures without exposing vector types.

## Kernel contracts

Formatting uses SIMD lanes for decimal digit positions. For each eight-digit
chunk, biased reciprocal multiplication and IFMA high multiplication produce
the eight characters, which are narrowed and stored with exact byte masks.
The u128 split is fixed: seven high digits and two sixteen-digit blocks. If
`B=2^64`, `D=10^32`, and `m=floor(B^2/D)=3402823`, then

```
q0 = high64(high64(value) * m)
r  = value - q0 * D
if r >= D: r -= D; q0 += 1
```

is exact after at most one correction. The constant inequality
`(B^2 mod D) + B < D` proves the estimate is at most one low. The remaining
sixteen-digit split uses a constant normalized word reciprocal. No general
division loop is involved.

Parsing reduces sixteen-character blocks using byte/word SIMD multiply-adds.
The u64 tail uses one 256-bit reduction for 17–20 characters. The u128 path
combines at most three sixteen-character blocks, with an exact limit check
at 39 characters. Both paths retain digit validation and safe partial loads.

The u64/u128 integer helpers inline into the primitive. Shuffling biases are applied
to constant vectors to avoid partial-byte GPR dependencies when Clang inlines
the functions into loops. Production does not retain experimental dispatch
switches or value-distribution sampling.

The multiword integer formatter repeatedly splits off `10^19` with a constant
normalized reciprocal, finishes with the existing u64/u128 formatter, and
emits the saved blocks through the same SIMD digit-position kernel. Integer
parsing uses the existing sixteen-digit reductions and multiply-add carry
chains. The general service keeps its existing integer parser and switches
to scaled integer formatting above eleven limbs. Kernel choices account
for the common service overhead; a direct wrapper win alone does not justify
replacing a faster generic kernel.
Its one-limb formatter also keeps the existing kernel that emits the whole
padded block directly, with zero arithmetic workspace. The new split-by-word
formatter is selected for 2–11 limb ASCII integers.

Fractional formatting keeps an exact binary remainder and multiplies it by
`10^19`, emitting the carry as a fixed-width decimal block. Known low zero bits
allow completed low words to drop out. Fractional parsing visits the decimal
blocks in reverse. For `B=2^(64*limbs)`, the recurrence is

```
x = 0
for block a, from last to first:
    x = floor((a*B + x) / 10^block_length)
```

This is exact: for integer `A` and positive integer `D`,
`floor((floor(y)+A)/D) = floor((y+A)/D)`. Induction therefore gives precisely
`floor(B*0.digits)`, without an error budget, guard precision, a large power
of ten, or a runtime reciprocal. The division uses the same fixed word
reciprocals as formatting. The last partial decimal block has its own exact
constant; full blocks all use `10^19`.

The u64/u128 fraction leaves keep their remainder in registers. Their larger
stack frames live in separate functions, so ordinary C calls do not pay the
multiword frame cost. The 1–8 limb interface has at most 192 bytes of explicit
stack value storage, plus compiler spills/call frames. The ordinary service
uses the fractional recurrence through 26 formatting limbs and nine parsing
limbs (also bounded by `limbs*ceil(digits/19) <= 108` for parsing). Larger
fractional format leaves reuse the existing carry-deferred IFMA kernel through
the same compact descriptor, then hand off to SRT under its existing work
rule. There is no input-dependent search. These shortcuts currently select
the standard decimal ASCII alphabet; other representations retain their
existing generic kernels.

Correctness gates: `decimal_tiny_test`, `decimal_small_test`, `decimal_c_test`,
and `decimal_inline_test`. The format/parse service tests also accept
`--decimal-small` for dense 1–40 limb boundaries and generic handoffs. They
cover integer and decimal boundaries, exact dyadic ties, invalid bytes,
overflow, leading zeros, guard pages, allocation freedom, scalar C callers,
and repeated use after invalid input.

Algorithmic credit: the parallel IFMA digit method is described by
[Gareau and Lemire](https://lemire.me/blog/2026/05/18/simd-accelerated-integer-to-string-conversion/)
and credited by [xjb](https://github.com/xjb714/xjb#8-acknowledgements).
The normalized word-reciprocal division retains the GMP donor attribution in
the implementation. No floating-point shortest-roundtrip algorithm is used.

The general [one-shot interface](radix-once.md) avoids bindings for these small
paths, and can use the fraction parser through ten limbs under its work bound.
It returns output layout/width metadata directly.
