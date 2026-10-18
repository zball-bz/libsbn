# C++ integers

The resident C++ integer core is implemented, including owning values, strict
destination operations and explicit prepared expressions. The API requires
C++20; production arithmetic keeps the library's no-exceptions contract.
Root extraction and modular powering are the next integer algorithm families;
they are deliberately outside this implementation milestone. General floating
arithmetic and out-of-core execution remain separate work. This is not a
complete replacement for mpz or MPFR, and the API is not yet declared frozen.

Use `<sbn3/sbn.hpp>` for the current integer surface, or the individual headers
below. A working example is `examples/cpp_integer.cpp`.

## Choosing an interface

| Use | Interface | Storage and execution |
|---|---|---|
| Quick calculations | `integer` and eager operators | Owns results; provisions capacity at the outer operation boundary |
| Allocation-sensitive code | `integer_ref`, `*_requirements`, `*_into` | Caller provisions output and quoted workspace; execution does not grow them |
| Repeated operand or shape | `prepared_divisor`, `context::prepare` | Explicit retained state and lifetime; divisor preparation and shape preparation are different capabilities |
| Future external-memory work | Explicit mathematical requests | Resident lowering exists now; a Store and IO lowering must be implemented separately |

Small eager/strict arithmetic does not go through the expression compiler.
Use the compiler when intermediate-lifetime planning or repeated execution is
useful, rather than as a prerequisite for ordinary number operations.

## Integer operations

- `integer`: normalized signed values, two inline limbs, deep copy, move,
  checked native integral conversions (including Clang's 128-bit integers).
- Eager `+`, `-`, `*`, `square`, ordinary `pow`, unary signs, increment/decrement, comparisons, shifts and
  infinite-two's-complement `&`, `|`, `^`, `~`.
- Arithmetic/floor `>>`; `trunc_shift` explicitly rounds toward zero.
- Bit test/edit/scan and magnitude/finite-range population count.
- Caller-owned `integer_buffer`, non-owning `integer_view`/`integer_ref`,
  and `fixed_integer<N>` with bounded capacity.
- Portable canonical byte import/export (`<sbn3/bytes.hpp>`), big/little endian,
  signed two's complement by default or explicit magnitude encoding.
- Allocation-free destination operations, including multiplication/square
  with a queried raw resident workspace and optional existing Team/scope.
- Truncating `/` and `%`, paired `divrem`, explicit floor/ceiling/Euclidean
  modes, exact division, divisibility/congruence, and retained divisors.
- GCD, extended GCD, LCM, modular inverse and Jacobi through the existing
  optimized kernels. These retain the C kernels' declared size limits.
- Base 2–64 parsing/formatting, custom alphabets and owning text; decimal
  leaves use the existing direct primitives.
- Explicit `memory_resource`, a default pooled value provider, and a
  coalescing `buffer_resource` over caller-supplied storage.

```cpp
#include <sbn3/integer.hpp>
using sbn::integer;
integer a = 123, b = 456;
auto c = sbn::square(a) + b;  // owning, eager
integer d = (c << 100) ^ -1;
auto small = c.try_to<std::uint64_t>();
```

For controlled storage:

```cpp
sbn::integer_buffer out{output_words};
auto need = sbn::multiply_requirements(out.ref(), a.view(), b.view());
if (!need.supported()) { /* provision output or handle unsupported shape */ }
sbn::workspace work;
work.reserve(need.scratch_bytes, need.scratch_alignment); // preparation boundary
sbn::mul_into(out.ref(), a.view(), b.view(), work.on(existing_team));
```

`workspace::on_scope` accepts the current leader's existing subteam. Do not
re-enter a root Team from its worker callback. No worker is created by these
interfaces. Zero/small multiplication uses direct leaves and needs no plan;
nontrivial aliases use fixed small stack storage or explicitly quoted staging.
Large Frame programs share the existing product policy and table/work kernels.

## Ownership and allocation

A supplied memory resource is borrowed and must outlive every value/workspace
referring to it, including an inline value which might grow later. The default
resource has process lifetime. Value storage and operation scratch have separate
owners; a returned integer never points inside a destroyed temporary workspace.

Copy assignment keeps the destination resource. Cross-resource move assignment
may copy, then clears the source. ADL `swap` exchanges storage and resource
provenance together. A moved-from value is a valid zero. Reserve failure preserves
the old value/capacity. `clear` keeps capacity; `reset` releases it.

The default resource uses prefaulted (not locked) mappings and fixed-size slabs;
no malloc/new is used for allocation. It retains at most one empty 128 KiB slab
per size class (64 through 32768 bytes), releasable by `release_unused`.
Larger or specially aligned allocations are individually mapped. Reported
resident bytes are committed mapping/storage accounting, not a live OS RSS
measurement. For pinned execution, supply a resource over an explicitly prepared
resident arena region. The allocator does not raise MEMLOCK.

`buffer_resource` is single-owner, allocates within one fixed span, and coalesces
arbitrary frees without OS calls. Its backing storage must remain live. Values
must be destroyed before the resource. The native pool is thread-safe; this does
not make concurrent mutation of one integer safe. Integer writers still require
external exclusivity, like ordinary C++ containers.

Views are read-only borrows. A temporary owner cannot export a `view()` or limb
span; raw views supplied by the caller retain ordinary C++ lifetime rules.
Writable refs point to the original length/sign descriptor, rather than copying
output metadata. Interop users of `begin_write` must call `finish_write` after
establishing a normalized result; ordinary code should use `_into` functions.

## Bitwise contract

The C bridge is `<sbn3/bitwise.h>`. Signed logical operations use infinite
two's-complement semantics over sign/magnitude storage. Negative inputs are
handled with finite `M-1` identities; the output's possible increment is handled
at its carry prefix. The regular SIMD loop therefore avoids materializing full
two's-complement copies. Exact output alias with either input is supported.

`magnitude_popcount()` counts absolute-value bits. `count_ones(first,count)`
uses signed infinite bit semantics over a finite window, including sign
extension. `scan_bit` returns `SIZE_MAX` when there is no representable matching
index. Neither range operation performs work proportional to an arbitrarily
long sign extension.

Independent native and sanitizer tests cover signed values, zero, leading zeros,
long carry prefixes, aliases, scans/ranges/edits, arithmetic shift and guard-page
tails. Native SIMD handles short vectors and regular streams directly, with
constant auxiliary space. Whole-call measurements cover dense small lengths,
non-power-of-two scaling, all sign pairs and unequal lengths. Some tiny and
highly unequal cases remain slower than GMP; this is not a blanket fastest-
implementation or all-sizes smoothness claim.

Byte import accepts empty input as zero and redundant sign/zero prefixes. Byte
export produces a minimum-length representation, with zero as one zero byte.
`export_bytes` checks capacity before any write; `import_bytes_into` and export
require disjoint input/output. The `magnitude` encoding explicitly ignores sign.
Native-independent byte order is selected by `byte_format`, not inferred from
the machine or the limb representation.

## Division and number theory

```cpp
#include <sbn3/division.hpp>
#include <sbn3/prepared_divisor.hpp>
#include <sbn3/number_theory.hpp>
auto [q, r] = sbn::divrem(a, b);          // truncation toward zero
integer residue = sbn::mod(a, b);       // 0 <= residue < abs(b)
auto rounded = sbn::divrem(a, b, sbn::division_rounding::floor);
sbn::prepared_divisor divisor(b, maximum_numerator_limbs, 100);
auto next = divisor.divrem(a);          // immutable divisor snapshot

auto [g, x, y] = sbn::extended_gcd(a, b); // x*a + y*b == g >= 0
auto inverse = sbn::inverse_mod(a, b);   // optional; noninvertibility is data
```

`divrem_into` accepts optional Q/R destinations. Its query includes internal residual/estimate work and alias staging.
Remainder-only calls do not reserve a full quotient output: word division
discards digits, IFMA retires quotient digits into dead numerator storage,
and FFT/NTT division keeps at most the estimate block needed by its residual. Division by zero and a false
`exact_divide` precondition are fatal diagnostics. Divisibility by zero is true
only for zero; congruence modulo zero means equality. Negative modular-inverse
moduli are accepted through their magnitude; modulus zero is invalid.

A prepared divisor is move-only and single-owner/sequential. It retains the
backend's normalized divisor, inverse and workspace, not a reference to the
original integer. Signed rounding reads those retained words; it neither makes
an extra full divisor copy nor checks the old value on each execution.
`storage_requirements` reports its complete persistent-plus-work reservation;
`bind` uses a caller-supplied span, and the span/Team must outlive the handle.
`query` reports per-execution alias/omitted-output staging. Disjoint Q/R
execution needs no extra scratch. Moving the handle does not move its binding.

GCD and extended GCD have `*_requirements` and `*_into` calls, including
independently omitted Bezout coefficients. Aliased input destinations use
quoted staging. Their current backend domain remains at most 2^20 limbs per
input; support is queried, not silently emulated with a quadratic fallback.
Jacobi requires a positive odd denominator. These are variable-time operations.

General division/radix services use a private resident-region adapter over
already-provisioned caller scratch. It creates byte leases, not mappings,
locked pages or workers. It can use an existing idle Team whose own storage
and stacks are disjoint. It rejects prepare/trim/destroy attempts on the
borrowed mapping. Small division uses direct kernels and compact preparation
uses a raw constructor, avoiding the large adapter. Local radix recipes
also use raw scratch/Frame entries; they share the same leaf arithmetic as
the existing C services and do not construct a resident-region descriptor. Division/radix/GCD root
services currently reject nested worker scopes; multiplication retains its
separate scope-aware entry. This limitation is explicit rather than silently
re-entering a Team.

## Text conversion

```cpp
#include <sbn3/radix.hpp>
auto parsed = sbn::parse_integer("-123456789012345678901234567890");
if (!parsed) { /* inspect parsed.error.code and .index */ }
auto text = sbn::to_string(parsed.value); // resource-owned integer_text
std::string_view borrow = text.view();
auto hex = sbn::to_string(parsed.value, 16);
```

`integer_text` is copyable/movable, keeps small text inline, and uses the same
resource lifetime rules as integer. It is not std::string; `.view()` provides
ordinary string interoperability without another allocation. Its data/view
exports are lvalue-only. Its logical content is NUL-terminated, and may start
inside its backing allocation to avoid copying a large padded digit array.

`format_requirements` includes output padding and NUL. `format_into` returns
the valid string view within that output span; it need not start at byte zero.
`parse_integer_into` preserves the destination on invalid input. Decimal
leaves already provide transactional parsing; other recipes perform a SIMD
validation pass before writing, avoiding a second full integer allocation.
Unaligned destinations may need separately quoted alignment staging.

The whole input must be valid: no skipped whitespace or radix prefix. Default
alphabets are `0-9 a-z` through base 36 and `0-9 A-Z a-z + /` above it. Parsing
uses the selected alphabet exactly. A custom alphabet is an explicit borrowed
`radix{string_view}`; it must have 2–64 unique bytes and exclude the minus sign.
A leading plus is a sign only when plus is not a digit of that alphabet.
Leading zeros and negative zero normalize normally. Empty or sign-only input
is an ordinary conversion error.


## Ordinary powers and managed readers

`pow(a,e)` in `<sbn3/power.hpp>` accepts a nonnegative integral exponent;
`0^0` is 1. Signed negative or wider-than-size_t counts are rejected before
conversion, including for a zero base. Shift and bit-index overloads follow
the same rule. `power_requirements`/`power_into` use a fixed binary ladder
with the existing square/product kernels. Power-of-two bases become shifts;
small results use registers. The general quote includes two padded integer
work buffers and the maximum stage workspace, reclaimed between stages.
This is a complete storage bound, not a claim of optimal power memory usage.
The final output is published only after all source reads, so aliases are safe.

`<sbn3/borrow.hpp>` adds `a.borrow()`, a managed immutable reader intended for
explicit requests. Copying/moving readers retains their owner's lifetime
obligation. Mutation, movement or destruction of an integer with active
readers is diagnosed. Ordinary `integer_view` still follows caller-managed
C++ lifetime rules. These guards are not a read/write lock: concurrent mutation
still requires external exclusion. The explicit expression/context layer below uses these readers.


## Contexts and explicit requests

Ordinary integer operators remain eager. Use `expr::...` only when deliberately
constructing a deferred request. Named factories and operators on an expression
compose arithmetic, division modes, GCD, bit operations, shifts and ordinary
powers. A context supplies value/scratch resources and an optional existing
idle Team; it creates no workers.

```cpp
#include <sbn3/sbn.hpp>
sbn::context ctx; // serial; explicit resources/Team may be supplied
sbn::integer a = 123, b = 456;
auto value = ctx.evaluate(sbn::expr::square(sbn::expr::add(a, b)) - 7);

auto request = sbn::expr::mul(a, b) + 13;
auto need = ctx.query(request);
sbn::integer out;
out.reserve(need.output_limbs);
ctx.reserve(need);                       // preparation boundary
ctx.evaluate_into(out.ref(), request);   // no allocation or scratch growth
```

`context(values_resource, scratch_resource, team)` separates result allocation
from temporary storage. `context(span_of_bytes, values_resource, team)` borrows
a fixed workspace; `try_reserve` reports insufficient capacity and never grows
that span. Returned integers own their independent value storage. Resources
and an externally supplied Team must outlive the objects referring to them.

Capturing an integer lvalue retains a managed immutable reader. Capturing an
integer rvalue owns the value, including small literals. Capturing an explicit
`integer_view` uses its ordinary caller-managed lifetime. A temporary buffer
cannot be captured as a dangling raw view. A retained request prevents mutation
of its managed inputs; discard or move the request when finished. Temporary
requests consumed by `evaluate` release their guards before assignment, so
`a = ctx.evaluate(expr::mul(a,b))` works. Strict evaluation can alias raw input
views: publication happens only after the final source read.

Compiled programs also support changing operands through bounded slots:

```cpp
auto program = ctx.prepare(sbn::expr::input<0>(1024) *
                           sbn::expr::input<1>(1024) + 7);
auto z = program(a, b);                    // each input at most 1024 limbs
out.reserve(program.requirements().output_limbs);
program.execute_into(out.ref(), a, b);     // out needs program's quoted capacity
```

`input_bits<I>(bits)` gives a tighter bit bound. Slots admit every normalized
length up to that bound, including zero; insufficient bounds are precondition
violations. Prepared requests are move-only and execute exclusively. Their
compiled state contains offsets and resolved recipes, not stale pointers into
moved inline operands. Input addresses are rebound for each execution. Shape
preparation does not imply cached operand spectra or a retained reciprocal;
use `prepare_divisor` when divisor preparation itself should be reused.

The resident compiler admits at most 1024 instructions. It is a fixed operation
graph, not a numerical grid search. It quotes control, live intermediate values
and maximum kernel workspace separately. Temporary slots are reused after their
last read, including shared nodes in an expanded power ladder. Kernels run in
joined Team episodes; the compiler does not nest public root Team operations.
No ordinary allocation occurs during compile-into or execution.

Products use proved readable zero padding in internal slots, charged to the
quote. A computed divisor can shrink drastically through cancellation. If its
planned width is `dmax` and its actual width is `d`, the general division recipe
uses `D' = D * B^(dmax-d)` and `N' = N * B^(dmax-d)`, with `B=2^64`.
The quotient is unchanged and the remainder is restored by the same whole-limb
shift. This makes the transform/storage shape known before evaluating the
children. One-/two-limb divisors use the existing direct word kernels without
building a giant opaque plan. General computed-divisor padding is an explicit
memory/performance tradeoff; splitting a request at a value boundary lets a
fresh operation plan the realized smaller shape.

This is a resident lowering. No stored-value handle is disguised as a pointer,
and no operator hides IO. The mathematical request vocabulary, bounded slots
and explicit resource/publication boundary are the seams for a future OOC
executor; this implementation does not claim external-memory execution.


## Standard character and IO adapters

`<sbn3/chars.hpp>` exposes `sbn::to_chars`/`sbn::from_chars` with the standard
result structs. They emit no NUL and parse a numeric prefix without skipping
whitespace or treating plus as an optional sign. Bases 2–36 accept either
letter case with the default alphabet; custom alphabets and bases 37–64 retain
case-sensitive digits. Query/`integer_ref` overloads use provided workspace
without allocation. Owning overloads provision temporary storage and report
resource exhaustion as `std::errc::not_enough_memory`.

Decimal leaves use fixed stack temporaries for tight output spans. For large
non-leaf text, the standard contiguous `to_chars` result requires an output
staging region and copy. Its query charges those bytes. `format_into` returns
a view inside a padded caller buffer and avoids that additional staging when
maximum throughput or output memory is the priority.

`<sbn3/io.hpp>` provides resident stream operators (also included by the
umbrella header), including decimal/octal/hex, signs, alternate prefixes,
width/fill/alignment and input auto-base mode. Invalid extraction preserves the
old value. The adapters use ASCII digits, not locale grouping. Formatting
materializes one resident text value; a stream operator is not an OOC executor.

`<sbn3/format.hpp>` is an optional C++20 `<format>` adapter. It supports
`d`, `x/X`, `o`, `b/B`, sign, alternate prefixes, fill/alignment, zero padding
and fixed/dynamic width. Locale grouping and floating presentations are
rejected. `integer_text` formats as a string view.

```cpp
#include <sbn3/sbn.hpp>
#include <sbn3/format.hpp>
sbn::integer n;
const char input[] = "-aBcD!";
auto r = sbn::from_chars(input, input + 6, n, 16); // n=-43981; r.ptr points at '!'
auto display = std::format("{:+#012x}", n);
```

## Scope before out-of-core execution

General `sqrtrem`/nth roots and modular powering are not implemented by this
layer. Random/primality and integer-sequence utilities are also not supplied.
Ordinary `pow`, LCM and modular inverse currently compose existing kernels;
they have correctness/resource tests, not separate full-domain tuning claims.

The existing dyadic/radix/Newton services are not a correctly rounded general
float type. Such a type needs explicit precision and rounding, bounded exponent
alignment, and residual/sticky/tie handling. It should share resources and
execution machinery without borrowing integer semantics for these decisions.

OOC will keep resident `integer_view` a real pointer/span. Add immutable stored
value handles, admitted block grants, failable IO execution and explicit durable
publication around the mathematical request layer. A custom allocator or a
large mmap alone is not an OOC implementation. The initial external multiplier
can restart the entire operation from committed inputs; retaining intermediate
pass artifacts is a separately priced capability.
