# One-shot radix conversion

`sbn3_format_once` and `sbn3_parse_once` perform a complete conversion with one
call. They support the same bases, alphabets, dyadic values and exact/enclosed
semantics as the planned radix services. Values, output buffers and any required
workspace remain caller-owned. Arithmetic does not allocate or expand storage.

Use these entries for individual conversions. Use query/bind/execute/unbind
when retaining a plan or prepared powers for several values is useful.

```cpp
#include <sbn3/radix.h>
#include <cstring>

sbn3_radix_options options{};
options.workers = 1;
options.use_alphabet = 1;
std::memcpy(options.alphabet, "0123456789", 10);

uint64_t input = 12345;
unsigned char text[64]; // the integer area is padded to 64 bytes
sbn3_format_spec spec{10, 1, 0, 0, SBN3_RADIX_EXACT};
sbn3_format_once_result result;
sbn3_format_once(&spec, &options, &input, text, nullptr, &result);

// Integer digits occupy [integer_first, integer_digits).
// An empty span represents zero, as in the planned format service.
auto *begin = text + result.integer_first;
auto length = result.integer_digits - result.integer_first;
```

The fractional area starts at `result.integer_digits`, and has
`result.fraction_digits` meaningful bytes. No sign, point or terminating NUL is
written. All integer-area bytes are valid digits, including its leading zeros.
The unpadded, fixed-width decimal primitives in `decimal.h` remain available.

Parsing returns validity, the first invalid byte position and `result.limbs`,
the complete output width on success. High zero limbs are possible; the result
is not implicitly normalized. On invalid text, `limbs` is zero and the value
buffer is unspecified. Digit validation is always performed.

For a pure fraction, set `integer_digits=0` and request `fraction_bits`; the
result is exactly `floor(0.digits * 2^fraction_bits)`. Formatting interprets the
mantissa as `M*2^exponent2`, with the existing EXACT/ENCLOSED distinction.

## Storage

`sbn3_format_once_query` and `sbn3_parse_once_query` provide output-capacity and
workspace bounds without returning a plan. They are optional when the caller
already knows sufficient bounds. The execution call repeats any necessary
selection or preparation; it does not require a previously queried plan.
The result includes enough layout information to consume a small conversion
without a separate sizing query.

When `storage_bytes == 0`, pass a null workspace: no arena or team is accessed.
Otherwise provide an `sbn3_radix_workspace` containing the arena, idle
caller-owned team, prepared offset, and available bytes. The range must be
resident, unleased, aligned as queried, exclusive throughout the call, and
disjoint from the input/output. It can be reused immediately after return.
An existing planned-service query also provides sufficient capacities.

Checked builds validate small-call alignment, overlap, and borrowed-range
ownership. As with the other arithmetic APIs, insufficient capacity or an
unsupported request violates a caller precondition. These calls do not perform
implicit reserve, page locking, heap allocation, or asynchronous work.

`plan_id == 0` denotes a direct/local recipe without a materialized public plan.
Small leaves use bounded stack temporaries. Local larger leaves use the supplied
workspace directly, without a binding/control region. General conversions
perform the existing query, preparation, execution and cleanup inside the call.

## Selection and code boundaries

The common decimal ASCII integer/pure-fraction cases use a small shape dispatch.
It does not hash a plan, build an alphabet map, copy a binding, or register an
arena lease. Compiler-visible constants and fixed thresholds choose the leaf.
General parsing and format kernels remain shared with the planned services.

The local formatter shares its recipe selection and arithmetic with the
existing service. Its scaled integer and carry-deferred fractional leaves use
raw planned workspace. Integer parsing similarly reuses the existing local
recursion once the bounded block leaf is exceeded. That recursion reuses the
whole optimized leaf whenever a child fits, avoiding a separate slower path
for the same child just above the public leaf boundary. These handoffs do not require
an opaque control object merely because the input grew by one limb.

The large query/preparation bodies and the block parser's temporary-array frame
are explicitly outlined. Their stack frames do not enter the u64/u128 path.
ISA intrinsics stay in private native kernels; the public header is scalar C.

Mixed values, custom alphabets, partial binary exponents, and ENCLOSED formatting
remain supported through the applicable local or general recipes. The near-leaf
overhead target is measured for the specialized decimal ASCII paths, not for
work such as constructing a custom alphabet or computing a large reciprocal.
