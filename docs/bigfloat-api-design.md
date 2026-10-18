# Minimal bounded-error bigfloat API design

Date: 2026-10-07. Independent design review for the first resident bigfloat vertical slice. This is a design note, not a declaration that any API below is already implemented. It intentionally covers point values, arithmetic, roots, elementary real functions and real AGM; complex functions, expression graphs and stored/OOC values can wait.

This initial proposal is retained for design rationale. The implemented API,
shared constants and current scope are documented in the [float guide](float.md)
and [implementation/evidence plan](bigfloat-implementation-plan.md).

Reviewed: [integer.hpp](../include/sbn3/integer.hpp), [execution.hpp](../include/sbn3/execution.hpp), [context.hpp](../include/sbn3/context.hpp), [memory.hpp](../include/sbn3/memory.hpp), [runtime.md](runtime.md), and [bigfloat roadmap](bigfloat-roadmap-2026-10-05.md). No code or header was edited and no compilation or timing was performed during this review.

## 1. Recommendation

Use the proposed public header include/sbn3/float.hpp with:

- typed bits;
- float_class finite/infinity/quiet_nan, with sign separate so signed zero survives;
- immutable float_view, writable float_ref, caller-backed float_buffer and owning bigfloat;
- per-value precision in significant bits, independent of allocation capacity;
- float_report returned by arithmetic, containing an absolute binary error bound and numerical flags;
- strict, allocation-free *_into operations using existing execution;
- pure requirements queries plus existing workspace/context reservation.

The stored value is a point. Error information is a separate result of an operation, never an implicit radius attached to every bigfloat. A view does not carry uncertainty; passing the result of a previous approximation into another operation passes its stored dyadic point exactly. Applications that need a cumulative error bound must compose the reports themselves or use a later explicitly named bounded-expression facility.

Keep correct rounding distinct from mantissa rounding. Exact dyadic round/add/subtract can provide stronger guarantees where implemented cheaply; that does not imply correctly rounded division, roots, transcendental functions or AGM.

## 2. Minimal semantic types

The first slice uses these names and field meanings:

    struct bits { std::size_t count; };

    enum class float_class { finite, infinity, quiet_nan };

    enum class error_kind { exact, finite, unbounded };
    struct error_bound {
        error_kind kind;
        std::int64_t exponent2;  // for finite: abs(error) <= 2^exponent2
    };

    struct float_report {
        error_bound error;
        float_flag flags;
    };

Explicit tags are preferable to special exponent sentinel values. In particular, INT64_MIN must remain an ordinary exponent if the documented range allows it; it cannot also mean exact. The default zero bound is exact. An unbounded report means no useful finite absolute bound has been promised.

For finite exact point inputs a,b, a returned point y and target mathematical operation f, the report means

\[
 |y-f(a,b)|\le E,\qquad
 E=0,\ 2^{e_{\rm err}},\ \text{or }+\infty.
\]

The error exponent is independent of y's exponent and requested precision. This matters after cancellation: y may be zero while the error is nonzero. A generic “three ulps of the output” is not adequate unless output spacing and scale are specified independently, especially at zero or a normalization boundary.

For nonfinite inputs/results, the special-value behavior table governs; the finite-real inequality above is not meaningful for NaN or infinity. Avoid presenting a finite error as a guarantee that infinity approximates a finite target. A finite-target overflow must return an unbounded error together with overflow.

### Precision is not accuracy

A value with precision p stores at most p significant bits. This says nothing by itself about how accurately it approximates an external intended value. Increasing p preserves the existing point and cannot restore discarded information.

For finite values, use the existing dyadic interpretation

\[
 x=(-1)^s M\,2^{e},\qquad M\ge0.
\]

Leading zero limbs should be trimmed logically; zero has no significant payload. Do not require removal of all trailing zero bits if doing so would impose an avoidable whole-payload move. Numerically equal dyadics can have different internal scales; comparison must compare their values, not descriptors. A later canonical serialization can normalize at its explicit materialization boundary.

Capacity is a storage quantity. reserve_bits changes capacity only; it must not change precision, value, sign or error history. set_precision rounds the stored point, changes its precision metadata and returns the error of that conversion. Increasing precision is exact relative to the stored point.

Check p>=1 and a documented supported maximum. Compute limb counts without overflowing p+63. Exponent-plus-bit-length, exponent differences, doubled root exponents and multiplication exponents need wider checked intermediates. Publish an exponent domain with enough headroom for the supported precision. Do not let host floating-point overflow/underflow select the library's range.

## 3. Owners, views and strict output references

Follow integer.hpp's established roles:

| Type | Owns limbs? | Allowed role |
|---|---|---|
| bigfloat | Yes, via memory_resource, with a small inline case if useful | Value lifetime, copy/move, explicit reserve/conversion |
| float_view | No | Immutable exact dyadic/special descriptor; valid only while source lives and is not mutated |
| float_ref | No | Writable descriptor and existing payload; no growth |
| float_buffer | No | Descriptor over caller-provided limb span |
| fixed_float<Bits> | Optional later | Stack/static backing; not required to ship the foundation |

Delete view/ref access on rvalues, as integer does. A raw view is not a managed borrow. If managed borrow support is added, it should use the same generation/reader rules as integer rather than introduce a second ownership system.

A float_ref must update class, sign, exponent, used length and precision coherently, not just mutate an embedded integer magnitude. Capture aliased source metadata before publishing output metadata. The owner sign must remain separate from integer's canonical sign so negative zero is representable.

Copy preserves the point and declared precision. Move transfers storage/resource preference without changing the numerical value. The memory_resource must outlive all associated values, including inline values retaining that preference. Reuse integer storage only through legal lifetime/ownership interfaces; avoid exposing a mutable integer whose normalization could silently erase negative zero or desynchronize metadata.

Exact same-object aliases can be supported where proved and quoted. Reject partial overlaps unless a specific operation documents them. Output payload, output descriptor and scratch must obey existing checked-build overlap rules. A disjoint public contract is acceptable for the first expensive root/elementary operation; do not hide an unquoted temporary to simulate in-place support.

## 4. The smallest useful operation shape

The current output-owned precision interface is:

    requirements add_requirements(float_ref out, float_view a, float_view b);

    float_report add_into(float_ref out, float_view a, float_view b,
                          execution ex = {},
                          rounding mode = rounding::nearest_even);

    float_report round_into(float_ref out, float_view a,
                            rounding mode = rounding::nearest_even);

Use the same pattern for subtract, multiply/square, divide/reciprocal, sqrt/rsqrt, then elementary functions and AGM. Names should match existing integer spellings where practical.

The precision explicitly selected when constructing the output owner/buffer, or by set_precision, is authoritative for *_into. A call uses out.precision(); it has no second precision argument. Output capacity is checked independently. The operations do not silently promote precision from their inputs. A float_view is an exact point descriptor and does not supply an implicit result-precision policy.

Small operations can have zero scratch and require no context or plan object. Larger operations use existing requirements, workspace and execution. There is no reason to make one-limb addition instantiate a generic graph.

The ordinary controlled flow is:

1. Query the complete output/scratch requirement.
2. Reserve output and scratch explicitly, handling resource_error.
3. Execute over the admitted spans and already-existing Team/scope.
4. Receive numerical metadata without allocation or automatic retries.

requirements already separates supported/unsupported/capacity from resource allocation failures. Numeric flags must not be used for resource failures. A correctly admitted computation exhausting scratch remains an invariant failure, as in runtime.md. Unsupported recipes or insufficient admitted accuracy must be resolved before arithmetic, or through an explicitly designed outer boundary; do not add a hidden allocation/replan loop.

For an input-sensitive query, state what must remain unchanged between query and execute: class, sign, exponent, precision/shape and, if used in planning, value. Do not promise that a query for one point covers every equal-sized point unless it supplies an envelope proof.

A convenience constructor such as bigfloat::from(2, bits{256}) is intuitive. Eager arithmetic wrappers may allocate at a documented outer boundary, use maximum operand precision by default, and omit the report only under a documented operation guarantee. Keep the report-preserving *_into route available. Delay a large operator/conversion overload set until these rules are stable; there is no process-global precision or inherited host rounding environment.

## 5. Error accumulation inside a recipe

A public power-of-two bound is sufficient for a local report. It is not an ideal internal accumulator. Repeatedly rounding every positive error addition up to the next power of two can be catastrophically pessimistic:

- a bound 1 followed by adding a tiny positive error rounds up to 2;
- adding another tiny error rounds up to 4;
- repeated steps can inflate the bound exponentially even when true accumulated error grows slowly.

Internally, keep a small upward-rounded significand/exponent magnitude accumulator, an exact count at a common binary scale, or sum a proved error budget once. Round to a power-of-two bound only when publishing the report. This does not require public interval semantics or full ball arithmetic on every operation.

Error-bound exponent overflow must saturate outward to unbounded, never wrap. Underflow of a positive error representation must round upward to a supported positive bound, not become exact zero. Zero has an explicit exact tag.

For a chain of approximate operations, distinguish:

- local quantization of a computed dyadic;
- approximation error in reciprocal/Newton/series/range reduction;
- coefficient/constant error;
- propagated upstream uncertainty, if an explicitly bounded recipe takes responsibility for it.

For example, composition of multiplication errors requires terms such as \(|a|E_b+|b|E_a+E_aE_b\), not just the sum of reports. The point arithmetic API does not perform this propagation implicitly. Root and elementary implementations can use proved recipe-level bounds instead of expensive independent recomputation.

### Numerical flags

A positive error bound does not prove the result is mathematically inexact. Define flags precisely:

| Flag/concept | Safe meaning |
|---|---|
| invalid | Operation is undefined in the real/special-value contract |
| divide_by_zero | A defined pole/infinite result from the zero divisor/input case specified by the API |
| overflow | The rounded computed point exceeded the representation range and was replaced by infinity; approximate operations do not separately certify the true value's side of the boundary |
| underflow | A nonzero computed point was flushed by the documented low-range policy; the absolute-error report still bounds the true operation |
| rounded, if exposed | Known nonzero bits of a computed dyadic were discarded |
| inexact, if exposed | Use only with an explicitly stated meaning; do not infer it merely from a positive conservative error bound |

For the first slice it is simpler to omit a global mathematical-inexactness claim. The exact error tag is sufficient to promise an exact finite result; a nonzero bound leaves exactness unresolved. For exact dyadic round/add/subtract, discarded-bit observations can provide stronger information.

## 6. Special values and domain behavior

Expected numerical domain cases belong in the float frontend, before invoking positive/normalized integer/Newton kernels. They are not fatal kernel failures.

Recommended initial nearest-even point conventions:

| Operation/case | Result / behavior |
|---|---|
| Exact cancellation of finite nonzero values | +0 under nearest-even; -0 for downward rounding |
| Addition of two -0 values | -0; other opposite-sign zero sums +0 |
| Subtraction of zeros | Apply the documented sign-negation/addition rule |
| Quiet NaN input | Quiet NaN propagation; no payload/signaling-NaN promise |
| +infinity plus -infinity | NaN, invalid |
| sqrt(-0), sqrt(+infinity) | -0, +infinity respectively |
| sqrt(negative finite nonzero) | NaN, invalid |
| rsqrt(+0), rsqrt(-0) | Signed infinity if adopting reciprocal-of-sqrt semantics; document divide_by_zero |
| log(+0), log(-0) | -infinity, divide_by_zero |
| log(negative finite nonzero) | NaN, invalid |
| exp(-infinity), exp(+infinity) | +0, +infinity respectively |
| AGM of finite nonnegative a,b | Symmetric nonnegative result; zero if either is zero |
| AGM with negative finite nonzero input | NaN, invalid |
| AGM(0,+infinity) | Define explicitly; NaN/invalid avoids claiming a unique two-parameter limit |

Do not promise all IEEE-754 details accidentally. Signaling NaNs, payload propagation, traps, subnormals, dynamic rounding modes and totalOrder can be postponed. Publish the actual exponent-range underflow policy before shipping. Comparison should treat +0 and -0 as numerically equal and NaN as unordered; C++ partial_ordering is appropriate, unlike integer's strong_ordering.

## 7. Foundation/root/elementary priorities

1. Exact point construction, classify/signbit, comparisons, scale-by-power-of-two, precision changes and report semantics.
2. Correct bounded add/subtract with virtual exponent gaps, cancellation and guard/round/sticky observations.
3. Multiply/square and signed normalized adapters for reciprocal/division, retaining existing kernel error bounds at the correct binary scale.
4. General-radicand sqrt/rsqrt with exponent parity and exact square cases; share normalization and result emission.
5. A small vertical set of real elementary operations with complete error budgets; real AGM as a bounded iteration on positive inputs.
6. Convenience text/conversion/operators after their numerical and resource policies are clear.

Do not call a multiply-add correctly rounded FMA merely because only the final approximate dyadic is quantized. Prove the mathematical operation guarantee or use a distinct bounded approximation name.

For AGM, exact iterates bound the answer between geometric and arithmetic sequences. Rounded approximate iterates require their own propagated errors; the visible iterate gap alone is not a proof. A small internal error accumulator and explicit power-of-two homogeneity are sufficient building blocks, with no requirement to expose a ball owner.

## 8. Available correctness oracles and tests

Read-only environment inspection found:

- MPFR headers report version 4.2.2 at /usr/include/mpfr.h; libmpfr.so and libgmp.so are present.
- Python mpmath is installed; gmpy2 and python-flint are not.
- tests/oracle/oracle.h supplies independent signed-integer arithmetic and square root.
- tools/check_oracle.py already cross-checks integer arithmetic against CPython integers and math.isqrt.
- tools/build.py deliberately adds tests/no_gmp to maintained tests/examples. That guard rejects GMP headers. Its explicit reference_gmp escape is for research benchmarks, not an implicit maintained-test exception.

Therefore use independent exact integer/dyadic properties in maintained tests. An optional external MPFR verifier can compare directed high-precision enclosures outside timed regions, without making core functionality or normal tests depend on the research checkout. Its process/adapter/build policy should be explicit. mpmath is useful for discovery and a second numerical comparison, but agreement at increased precision alone is not proof of the claimed bound.

High-value test cases, without writing tests that merely duplicate the implementation:

- Precisions 1,2,3,31,32,53,63,64,65,127,128,129 and non-limb widths.
- Exact halfway values and values one bit to either side; carries across all-one mantissas.
- Cancellation to zero or a very small surviving tail; large exponent gaps with memory use bounded by input/output precision rather than the gap.
- Mixed source/output precisions; signed zeros; NaN/infinity/domain matrix.
- Same-object aliases where supported; partial-overlap rejection; preservation of disjoint inputs.
- Independent exact dyadic error comparisons: convert both result and reference to a common integer scale and prove \(|y-r|\le2^e\).
- sqrt/rsqrt: perfect squares, near-square radicands, odd/even exponents and extreme power-of-two scales. Check root enclosures via monotone squared endpoints, using an independent exact integer oracle.
- Elementary functions: exact special values, range-reduction boundaries and cancellation-sensitive log1p/expm1 inputs; optional directed MPFR reference enclosures.
- AGM: symmetry, homogeneity, equality case, zero boundary and independent arithmetic/geometric brackets with certified oracle rounding.
- Reservation failure leaves the original owner intact; admitted *_into operations do not change resource allocation statistics or create an implicit Team.
- Error metadata remains meaningful at zero and extreme scale; bound exponent overflow/underflow never wraps or silently becomes exact.

No oracle program, numerical probe, test or benchmark was run as part of this design review.
