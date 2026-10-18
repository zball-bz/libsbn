# Integer powers of bounded-error bigfloats

Design and implementation, 2026-10-07, integrated and validated 2026-10-08. `src/float/power.hpp` exposes a private reusable prepare/execute recipe for a fixed result precision, signed 64-bit exponent and worker budget. `power.cpp` defines the public `powi_requirements`/`powi_into` wrappers, with owning C++ overloads in `src/cpp/float.cpp`. See the [float guide](float.md) and [implementation evidence](bigfloat-implementation-plan.md). No performance acceptance claim is made here.

## Arithmetic and range

Let `N=|n|`, formed with unsigned subtraction so `INT64_MIN` becomes exactly `2^63`. For nontrivial powers use

\[
q=p+\operatorname{bit\_width}(N)+16
\]

working bits. Keep an exactly q-bit integer mantissa M and a separate signed 128-bit top exponent E, representing `M*2^(E-q)`, with normalized mantissa in `[1/2,1)`. Only a bounded high prefix of the exact input is read; omitted low input bits receive an analytic error allowance and are not scanned.

For a product P of two mantissa integers, retain its leading q bits. If L is P's bit length, the new top exponent is `E_a+E_b+L-2q`. Since `L` is either `2q-1` or `2q`, normalization is a one-bit decision. Exact integer MUL/SQR recipes and all tables are prepared before numerical execution. The binary ladder reuses these two fixed shapes; it does not query new plans or allocate memory.

Negative n reciprocates the **normalized base first** using the prepared reciprocal service at q bits. Its normalized argument stays in `[1/2,1)`, so that service never sees an extreme original exponent. The reciprocal's small normalization exponent is combined with `-E` in the private wide exponent. The same positive-N binary ladder follows. Computing a possibly overflowing positive power before taking its reciprocal is deliberately avoided.

Checked signed 128-bit arithmetic suffices: the original exponent magnitude is bounded near `2^60`, N is at most `2^63`, and all normalization adjustments are bounded by the same multiplication count. This stays well below the signed 128-bit limit. Only the final result is mapped into the public exponent range. Extremely small results report an outward absolute bound even when its natural exponent is below the error type's range; the bound saturates upward to its minimum positive power of two, never to exact zero.

## Error bound

Put `u=2^-q`. A truncated positive input mantissa has relative error below `2u`. For a negative power, prefix error transported through reciprocal on `[1/2,1)` is below `4u` in absolute value. The q-bit reciprocal result's local bound contributes at most another `4u`; its exact normalization adds no error. A common conservative base-relative bound `16u` therefore covers both signs of n.

Each exact mantissa product truncated to q bits has relative error below `2u`. For the left-to-right binary ladder, initial base error is weighted N times and the sum of all multiplication-error weights is N−1. With `|log(1+d)|<=2|d|` on the relevant tiny-error domain,

\[
\left|\log\frac{\widehat{x^n}}{x^n}\right|
<32Nu+4(N-1)u<36Nu.
\]

Even at p=1, `36Nu < 36*2^(-p-16) < 1/4`, so the logarithmic-to-relative conversion below is valid. Relative error measured against the computed center is at most twice this logarithmic bound. For a center with top exponent E,

\[
|\widehat{x^n}-x^n|<2^{E+7+\operatorname{bit\_width}(N)-q}.
\]

With the 16-bit guard, the last exponent is `E-p-9`: the algorithm contributes less than 1/512 of the p-bit ulp at the unrounded center. The final report adds this analytic bound to `round_magnitude`'s local mantissa-rounding error through `ErrorMagnitude`. Nearest rounding contributes at most half an ulp, so their sum is below one; the other modes contribute less than one ulp, so their sum is below two. The reported power-of-two radius respects those same bounds (a carry into the next binade only enlarges the result ulp). These modes do not promise directed mathematical powers or correct rounding. A nonzero analytic bound does not by itself set the `rounded` flag.

## Special values and ownership

Exponent zero returns 1, including zero/NaN/infinite bases. Exponent one rounds the original point directly. Other NaNs propagate; zero to a negative exponent returns signed infinity with divide-by-zero, and infinity to a negative exponent returns signed zero. Negative bases use exponent parity. Single-word powers of two, including ±1, use an exact checked exponent calculation and need no scratch.

The prepared state owns only an exclusive caller span: immutable product tables, optional prepared reciprocal, base/current mantissas, one raw-product bank and one maximum work region. Execution stages input before writing caller output, so exact-base aliases remain safe. A guard rejects concurrent reuse of the same mutable prepared span. Independent spans can run on existing Team subscopes. No threads, ordinary allocations or product queries are created during execution.

The arithmetic cost is `O(M(q)(1+log N))` for N>=1, including the reciprocal-only case N=1, and the live storage is `O(q)` plus multiplier/reciprocal workspace and immutable tables. Since N is a signed 64-bit input, `q=p+O(1)` here. This is the integer-power foundation; general nth roots and real-exponent powers require separate contracts and algorithms.

## Integration gates

Before claiming completion, test positive/negative exponents, `INT64_MIN`, signed zero/NaN/infinity, ±1 and powers of two, exact-base aliases, low output precisions, non-limb precisions, near-one inputs with large N, huge original exponents whose reciprocal power remains representable, final range boundaries, prepared reuse and active subscopes. Check the reported bounds using independent exact integer/rational powers for bounded exponents; do not use the production-backed large-product witness in the test oracle. Allocation and query instrumentation should cover the complete numerical loop. Compilation and timing remain serialized under the repository benchmark lock.
