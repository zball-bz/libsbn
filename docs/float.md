# Resident bigfloat arithmetic and functions

Include [`sbn3/float.hpp`](../include/sbn3/float.hpp), or the C++ umbrella
`sbn3/sbn.hpp`. Values are exact dyadic points; controlled operations return
local absolute-error reports. This guide describes the currently declared
resident API. G1/G2's implementation and correctness closeout, including shared
constants and wide-leaf bit burst, is recorded in the
[implementation plan](bigfloat-implementation-plan.md). Tuning and performance
acceptance remain separate.

| Capability | Present contract |
|---|---|
| `bigfloat`, `float_view`, `float_ref`, `float_buffer` | Owning or caller-owned sign/magnitude dyadics, independent bit precision/capacity, signed zero/infinity/quiet NaN |
| `round_into`, `scale2_into`, `add_into`, `subtract_into`, `compare` | Exact point alignment/rounding in the normal range; exponent gaps stay virtual; partial-order comparison |
| `mul_into`, `square_into` | At most p+128 significant input bits retained; omitted-tail error included; native multiplication/square backend |
| `fma_into` | Exact full dyadic product plus addend, rounded only after the sum; low product bits preserved through cancellation |
| `divide_into`, `reciprocal_into`, `sqrt_into`, `rsqrt_into` | Bounded general real division and roots; source-prefix and iteration errors included |
| `powi_into`, `rootn_into`, `pow_into` | Signed 64-bit integer powers, unsigned integer root degree, and real powers; degree zero is invalid and negative radicands require an odd degree |
| `exp_into`, `expm1_into`, `exp2_into`; `log_into`, `log1p_into`, `log2_into`, `log10_into` | Real exponential/logarithm family with direct cancellation-safe increment forms |
| `sin_into`, `cos_into`, `sincos_into`; `sinpi_into`, `cospi_into`, `sincospi_into` | Radian or π-scaled arguments; paired reports keep the two absolute errors separate |
| `atan_into`, `atan2_into`, `asin_into`, `acos_into` | Real inverse trigonometric functions; `atan2` argument order is `(y,x)` |
| `sinh_into`, `cosh_into`, `tanh_into`, `asinh_into`, `acosh_into`, `atanh_into` | Real hyperbolic and inverse functions, with endpoint and large-argument formulas |
| `agm_into`, `elliptic_ke_into` | Nonnegative-real AGM; paired complete elliptic integrals using the parameter or its direct complement |
| `to_integer_into` | Rounds the exact stored dyadic to an integer; nonfinite input is unsupported |
| Eager arithmetic and named functions | Return owning values (`bigfloat`, paired results, or `integer`); local reports are discarded |
| `error_bound`, `float_report` | Exact, finite absolute power-of-two bound, or unbounded; numerical flags separate from resource failures |

Controlled calls have matching `*_requirements` queries. The naming exceptions
are `multiply_requirements` for `mul_into`, and `add_requirements` for both
addition and subtraction; query subtraction with the second view negated.
Round, scale and compare need no operation workspace. FMA currently has a
controlled `fma_into` interface, rather than an eager `fma` overload.

The eager surface includes `+`, `-`, `*`, `/`, their compound assignments,
`add`, `subtract`, `multiply`, `divide`, square/roots/powers, and the
elementary, AGM and elliptic functions listed above. `to_integer` returns an
owning integer. Unary bigfloat results preserve input precision; binary results
use the larger input precision. `sqrt`, `rsqrt`, `reciprocal`, `powi` and
`rootn` also have explicit-result-precision overloads. Unary results use the
input resource; binary results use the left input's resource. Eager calls
reserve their own values/workspace and use serial execution. Use `_into` for
reports, a different result precision, or an admitted Team/scope.

## Values and precision

A finite value means `(-1)^negative * magnitude * 2^exponent2`. The stored
dyadic is exact as an input to the next call. Earlier computational error is
not implicitly attached to the value: retain and combine operation reports
when a computation needs an end-to-end guarantee.

`bits{p}` counts significant bits, not limbs or storage bytes. `reserve_bits`
changes only capacity; `set_precision` rounds the current point and returns
the error of that conversion. Raising precision cannot restore lost digits.
Copy preserves value, precision and sufficient writable capacity. Move leaves
the source as a valid default-precision positive zero.

```cpp
#include <sbn3/float.hpp>

auto x = sbn::bigfloat::from(2);
auto convenient = sbn::sqrt(x, sbn::bits{1024});
// For explicit storage, execution and the local error report:
sbn::bigfloat result(sbn::bits{1024});
const auto need = sbn::sqrt_requirements(result.ref(), x.view());
// Check need.supported() at a resource/plan boundary.
sbn::workspace work;
work.reserve(need.scratch_bytes, need.scratch_alignment);
const auto report = sbn::sqrt_into(result.ref(), x.view(), work.on());
```

For fixed caller storage, construct `float_buffer` over a limb span instead.
The destination's precision is the authoritative output specification; there
is no conflicting precision argument to `_into`. Views/references from owning
temporaries are prohibited. The caller keeps raw views alive and preserves
their sources while an operation is running.

`bigfloat::from(double, bits{p})` treats its argument as the exact binary64
point, including subnormals and signed zero, then rounds to p bits. Thus a
small p can lose input bits. It does not reinterpret binary64 `0.1` as the
rational 1/10. Integer imports likewise begin with an exact point and round
to the requested precision. Use `assign` and retain its report when that
conversion error matters.

The complete [C++ example](../examples/cpp_float.cpp) shows eager evaluation
and a query/reserve/execute call with stack-owned output limbs and a retained
error report. No floating-point decimal formatter is assumed by that example.

## Paired outputs and argument conventions

`sincos(x)` and `sincospi(x)` return `sincos_result{ sine, cosine }`.
Their controlled forms take two output references and return
`sincos_report{ sine, cosine }`, whose members are individual `float_report`s.
The outputs must be disjoint and may request different precisions.
`sinpi(x)` means `sin(πx)`, reducing the exact dyadic argument before introducing
π. Integer sine zeros have the input sign; half-integer cosine zeros are +0.
Radian trig reports include absolute range-reduction error: p-bit storage
does not promise p relative bits in an arbitrarily tiny component. Huge
radian arguments can require a constant bank much wider than the output;
its cost is included in the requirements query.

`elliptic_ke(x, elliptic_argument::parameter)` returns
`elliptic_result{ k, e }` for complete **K(m), E(m), with parameter m=x**,
not modulus `k=sqrt(m)`. Finite real parameters satisfy `m<=1`, including
negative m. With `elliptic_argument::complement`, the supplied point is
`c=1-m>=0`; the result is still K(m), E(m). Supplying c directly preserves a
small complement that would be lost by first rounding m to one. At `m=1`
(`c=0`), K is infinite and E is one.

`elliptic_ke_into` requires two disjoint outputs with the **same** precision
and returns `elliptic_report{ k, e }`. Eager paired results both use the input
precision and resource. General comparison is the partial ordering of stored
points, not comparison of their unretained computational error intervals.

## Accuracy and special values

For finite exact inputs, a finite report promises
`abs(result - mathematical_operation(inputs)) <= 2^error.exponent2`.
The exponent belongs to the error itself; it remains meaningful when the
result is zero or has a different normalization. An exact tag promises zero
error. Nonfinite results carry no finite-real absolute bound.

The rounding enum specifies how to quantize a computed mantissa. Round, scale,
add and subtract determine the exact dyadic result before that quantization;
integer conversion also rounds its exact stored input. Products, division,
roots and transcendental functions may introduce approximation before final
rounding. Their modes do **not** promise correctly rounded or directed
exact-function results. An `upward` center alone is not an upper endpoint of
the mathematical result; use its error report to obtain an enclosure.

The general product/root/division kernels bound normal finite nonzero results
by one p-bit output ulp in nearest-even mode, or two in the other modes, and
may report a tighter or exact bound. An ulp here means
`2^(exponent2 + bit_length(magnitude) - p)`, not a fixed limb unit.
Do not infer a uniform relative-error promise for every elementary function
from storage precision; retain the returned bound, particularly after phase
reduction or cancellation. Correctly rounded transcendental wrappers are not
part of the declared API.

FMA keeps the full product because cancellation can expose any of its bits.
It rounds the exact finite dyadic sum once, including when the intermediate
product lies outside the public exponent range but the final sum returns to it.
Its work therefore depends on the actual multiplicand widths; it does not have
the output-precision-only prefix bound of ordinary multiplication.

Flags can include `rounded`, `invalid`, `divide_by_zero`, `overflow` and
`underflow`. They are not a substitute for the error bound: a conservative
finite radius can be present even when `rounded` is clear. Invalid real-domain
operations produce quiet NaN;
reciprocal square root of signed zero produces signed infinity and
`divide_by_zero`. Square root preserves signed zero. NaNs compare unordered;
the two zeros compare equal.

The normal top exponent is bounded by ±2^60. Range handling follows quantization
of the computed point: an upward carry can bring a boundary result back into
the normal range. Overflow produces signed infinity and an unbounded error.
Underflow flushes to signed zero with an absolute bound; subnormal directed
rounding is not promised. For approximate operations these flags describe the
computed representation, not a separate proof of the true result's side of a
range boundary. The finite absolute-error report remains valid for the true
operation on the exact inputs. Precision ranges from 1 to 2^40
bits subject to the selected operation/backend and available storage.

## Storage and execution

Result limbs belong to the caller. `_into` never allocates or grows scratch,
creates workers, or changes page residency. A requirements query reports full
scratch and alignment. Existing `execution.team` or `execution.scope` supplies
parallelism; the caller owns their lifetimes. Exact-base value aliases are
supported by read-before-write staging. Partial overlaps and value/scratch
overlaps violate the contract.

Some small or exact cases need zero scratch; use the query rather than
assuming that from precision alone. Wide exponent gaps stay virtual instead
of allocating padding. Resource reservation is an explicit boundary:
`try_reserve`/`try_reserve_bits` return resource errors, while their non-try
counterparts and violated arithmetic preconditions use the fatal-error policy.

Private prepared arithmetic/function programs keep shape compilation outside
repeated numerical execution and can retain explicitly owned constants. They
are not a frozen public prepared-float ABI. The public boundary remains the
requirements query, caller-owned values/workspace and `_into` call.

## Explicit shared constants

[`sbn3/float_constants.hpp`](../include/sbn3/float_constants.hpp) adds optional
immutable π/ln2 banks. A `context` can own a bank and reuse its construction
scratch for subsequent arithmetic. Bank integration has passed native and
sanitizer gates; these establish correctness and resource behavior, not speed.

```cpp
#include <sbn3/context.hpp>
#include <sbn3/float_constants.hpp>

const sbn::bits precision{257};
auto x = sbn::bigfloat::from(2, precision);
sbn::bigfloat result(precision);
sbn::context ctx;
ctx.prepare_constants(sbn::log_constants(precision));
const auto bank = ctx.constants();

const auto need = sbn::log_requirements(
    result.ref(), x.view(), ctx.workers(), bank);
ctx.reserve(need); // Checks support and admits the operation's scratch.
const auto report = sbn::log_into(
    result.ref(), x.view(), ctx.execution(),
    sbn::rounding::nearest_even, bank);
```

Use the function's demand helper: internal guards can require more constant
bits than the output precision. Combine demands with `merge_constants` before
preparing one bank for several operations. Radian `sincos_constants` also
takes the input view because its exponent affects reduction precision.
The bank argument is last, so existing calls keep their meaning. A supplied
compatible bank removes duplicate constant construction from those calls;
the usual calls remain available without a bank.

`ctx.reserve` can reuse or replace arithmetic scratch without invalidating
the constant view. A bank rebuild or destruction **does** invalidate its
borrowed views and prepared consumers; reacquire `ctx.constants()` after a
rebuild. Moving the owner preserves borrows into the transferred allocation.
Its resource and allocation must outlive every borrower. Constant preparation
is an explicit allocation boundary; `try_prepare_constants` and `try_reserve`
provide recoverable resource errors.

For separate ownership, use `constant_bank`; for fully caller-owned storage,
`constants_requirements` quotes distinct retained-value and construction-work
spans for `constants_into`. `bank.get` returns an immutable point and its
error bound, while `copy_constant_into` makes a nearest-even prefix copy with
the constant and discarded-tail errors included. Neither operation silently
attaches that error to a later `float_view` computation.

## Scope

There is **no public OOC float execution contract**: a float view describes
resident limbs, and an owning call does not spill to disk implicitly. The
existing integer/radix OOC facilities do not extend this API automatically.
Correctness gates and the remaining implementation milestones are tracked
separately; this guide makes no speed, SoTA or global-completion claim.
