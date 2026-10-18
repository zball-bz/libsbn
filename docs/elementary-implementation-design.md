# Elementary-function implementation design

Design date: 2026-10-07; implementation status updated 2026-10-08. This document
describes the resident exp/log/sincos algorithms over bounded-error dyadics.
Algorithmic scaling and complete storage bounds come before threshold tuning.

The private [bit-burst evaluator](../src/float/bit_burst.hpp) now implements
reduced exp/expm1/sincos with the wide-leaf Frame adapter described below.
Its native gate passed 856 independent interval certificates, 36 prepared
replays, 24 executions with leaves wider than 64 limbs, and 40 extreme analytic
leaves. The sanitizer gate, including the additional tiny-prefix scratch case,
passed 866 certificates with the same prepared, wide-leaf and extreme-leaf
counts. These are correctness and resource checks, not performance or
crossover evidence.

The bit-burst exp/sincos recipes below cost O(M(p) log² p) on a fixed
bounded argument domain. They deliberately avoid applying exact binary
splitting to a long Taylor series whose argument already has p significant
bits. The public high-precision path uses an independent AGM logarithm and its
Newton inverse for O(M(p) log p) log/exp, including its cold constants. Reduced
sincos uses restricted complex AGM atan followed by real Newton inversion,
also O(M(p) log p); positive-real AGM alone would not suffice. Bit burst is an
independent private method, not a newly selected public crossover policy.

## 1. Existing components and exact reuse boundaries

The repository contracts remain authoritative: [runtime](runtime.md),
[series.h](../include/sbn3/series.h), [formula.h](../include/sbn3/formula.h),
[formula_sum.h](../include/sbn3/formula_sum.h),
[newton.h](../include/sbn3/newton.h), [constants.h](../include/sbn3/constants.h)
and [log.h](../include/sbn3/log.h).

| Existing component | Reuse | Required boundary / missing piece |
|---|---|---|
| `sbn3_series_query/plan_init/visit/execute` | Ordered finite trees, exact capacity oracles, worker schedule, immutable plan | Consumer owns tails, numeric parameters and error analysis |
| Private `series::FiniteFormula` and `finite_frame_*` | Existing CommonP2B3 merges, frozen product plans, table pools and child lifetimes | New Frame adapter admits checked wide leaves; legacy public constant-schema limits remain unchanged |
| `series::PsrSpec`, `PrecisionPolicy`, limited merges | Later precision-tapered splitting and existing constant recipes | Its contraction and exact-suffix proofs must not be transferred blindly to new dyadic leaves |
| `series::RatioTerminalPlan`, `RatioOutput::Approximate` | Existing normalized Newton division and bounded fixed-point conversion | Bit burst converts a positive magnitude and restores its separately retained sign |
| `formula_execute_approximate` | Bounded-error evaluation of existing nonnegative constant definitions | Public `formula_execute` requests a certified floor; that is a different contract |
| Formula sum | Sequential component execution, coefficient guards, consumed component storage | Its certified-floor terminal is not used by bounded-error elementary evaluation |
| Small-integer log / rational atanh catalog | Alternative independent constant recipes | Not the active float log2 producer; retain its fixed schema and supported argument bounds |
| Chudnovsky pi | Alternative independent pi bootstrap using existing u64 rsqrt(10005) | Not the active float pi producer; public pi retains exact-floor certification |
| Product, Newton divide, Arena, Team | All long arithmetic and resource execution | No separate allocator, product selector or hidden per-function thread pool |

The relevant private implementations are
[arithmetic.hpp](../src/series/arithmetic.hpp),
[arithmetic.cpp](../src/series/arithmetic.cpp),
[psr.hpp](../src/series/psr.hpp),
[precision_policy.cpp](../src/series/precision_policy.cpp),
[terminal.hpp](../src/series/terminal.hpp),
[scaled_ratio.hpp](../src/series/scaled_ratio.hpp),
[formula_api.hpp](../src/series/formula_api.hpp) and
[formula_sum.cpp](../src/series/formula_sum.cpp).

Three concrete boundaries are explicit in the implementation:

1. The legacy finite adapter's 64-limb limit does not apply to the new private
   `finite_frame_*` path. Burst numerators can have Θ(p) bits in queried Frame
   slots. The consumer uses `leaf_terms=1`; a wide parameterized multi-term
   leaf without a complete fast batch is rejected, preventing a hidden scalar
   quadratic accumulation path. Existing public constant limits are preserved
   and the original CommonP2B3 merge programs are reused.
2. The PSR exact-suffix argument explicitly uses a magnitude condition on U.
   In the representation below U has a negative binary exponent and can be
   smaller than 1. The first burst implementation uses exact finite trees with
   proved O(p)-bit capacities, without PSR truncation. Later PSR use needs a
   proof for this representation, including its exact-suffix predicate.
3. `guard_separated` can fail for an otherwise useful bounded approximation.
   Ordinary elementary evaluation must not enter a constant's certified-floor
   terminal. Retain its approximate ratio and certified absolute error, then
   round the point result and add that rounding error. Existing strict constant
   entry points retain their current behavior.

## 2. Shared bounded-error foundation

Names in this section denote implemented capabilities, not public declarations.

| Primitive | Required numerical behavior |
|---|---|
| Exact dyadic point view | Sign, integer magnitude and checked binary exponent; precision/capacity separate; no invented input uncertainty |
| Absolute error descriptor | Nonnegative upward bound such as `units * 2^scale`, independently scaled from the value; exact zero representable |
| Normalize / scale2 / classify / compare | Metadata-first exact operations, exponent overflow checked |
| Prefix and bit-slice observation | Extract a dyadic chunk and bound omitted low bits without allocating virtual exponent gaps |
| Add/subtract | Exact on a common admitted fixed-point grid; otherwise return alignment/rounding error; severe cancellation observes sufficient input |
| Multiply/square | Exact integer product followed by bounded truncation, or an already proved short-product capability; error ≤ one target-grid unit for one truncation |
| Bounded divide / ratio conversion | Signed wrapper over normalized divide; return center and absolute error, without a floor-separation test |
| Exact small-integer conversion and linear combination | Reduction quotient, quadrant bits, coefficient products and exact power-of-two scaling |
| Output emitter | Round/normalize a dyadic center to requested significant bits and add the center-rounding error to the analytic error |
| General sqrt/rsqrt | Positive normalized dyadic core with an explicit error; used by the active AGM path |

Direct two-series bit-burst sincos does not require general sqrt. It remains
independent of the AGM path's roots; a future sine-plus-sqrt optimization can
reuse the shared root core.

An internal evaluator may accept an input-error descriptor, but the public
point input remains exact by default. Do not put full interval objects or
correct-rounding decisions inside every limb operation. Common-grid fixed-point
arithmetic plus a compact outward error bound suffices here.

### Bounded float multiplication adapter

The first float product adapter retains at most k=p+128 significant bits of
each exact input, where p is output precision. For positive magnitudes with
`2^(Ea-1) <= a < 2^Ea` and similarly for b, write a=a0+α and b=b0+β after
downward prefix truncation. Then α<2^(Ea-k) when a was truncated (otherwise
α=0), and analogously for β. Hence

\[
 0\le ab-a_0b_0=\alpha b+a_0\beta
   <2^{E_a+E_b-p-127}.
\]

The lower prefixes retain their leading bits. A nonzero finite rounded product
has top exponent at least Ea+Eb−1, so this omitted-product error is less than
2^-126 of its output ulp. Multiplying the prefixes exactly and applying the
common mantissa rounder gives:

| Input treatment | Nearest-even mantissa | Other mantissa modes |
|---|---|---|
| Both inputs fully retained | Common rounder's exact/tighter bound | Common rounder's exact/tighter bound |
| Either prefix truncated | At most one result ulp | At most two result ulps |

The second row remains a finite bound even if all unobserved low bits happen
to be zero or the prefix product rounds exactly. No scan of the omitted range
is required. Other modes round the computed approximate mantissa; they do not
promise a directed bound on the full mathematical product. Their two-ulp bound
is necessary because a discarded input tail can move the exact product across
a target-grid boundary. There is no per-mode perturbation of the product.

The output exponent limit is checked separately. If output flushes to zero,
use the full-input bound `|ab| < 2^(Ea+Eb)`, not the possibly smaller top
exponent of a truncated prefix product. Infinite overflow has an unbounded
absolute error descriptor. Exact zero, NaN and infinity cases are handled
before ordinary products.

With at most O(p) retained input bits, arithmetic is O(M(p)) and staging is
O(p), independent of the omitted exact-input length. Up to 256 result bits,
two at-most-six-limb prefixes and a twelve-limb product fit bounded local
storage. Larger calls compile a private `FloatProductPlan`, separately prepare
its existing product program, and execute without new queries. The scratch
quote includes both prefix slots, the backend's complete padded output, tables,
workspace and alignment. Staging both inputs first permits exact-base aliases.

## 3. Private evaluator shape and lifecycle

Keep reduction independent of reduced-series evaluation. The private concepts
are:

```text
ReducedRequest:
    function = exp / expm1 / sincos
    exact reduced point, or point plus separately supplied reduction error
    absolute target 2^-t
    requested outputs

DyadicSliceFacts:
    family = expm1 / sinc_even / cos_even
    argument z = A * 2^-b, with |A| < 2^(b-a), |z| < 2^-a
    range [begin,end), term count N
    exact capacity and factor bounds; output grid 2^-W

DyadicSliceRun:
    borrowed read-only A view and sign
    admitted output and scratch slots; pre-existing Team/scope

BoundedValue:
    dyadic center + absolute error descriptor

ReducedPlan:
    reduction depth, slice descriptors, finite-tree and ratio plans
    arithmetic programs, lifetime layout, complete peak/storage/error bounds
```

`DyadicSliceFacts` and the compiled tree are immutable. Runtime A belongs in an
executor/run context, not mutable coefficient fields inside a sealed plan.
The implemented `finite_frame_execute` takes a separate runtime leaf context;
the retained `FiniteFormula` context supplies immutable bounds/work facts.
Existing constant consumers keep the old path. A plan therefore covers every
A within its proved bit bound, including zero,
without querying a new product when the numerical residual becomes available.

Query prepares all slice geometries from W, a and b, not from the individual
digits of A. During execute, extract one A into its already admitted slot,
evaluate it, consume its result into the accumulator, and reuse that slot for
the next slice. A zero slice may take a preplanned zero-work branch. The original
input, constant bank and reduced argument remain read-only for their declared
lifetimes. No execution-time plan search, heap allocation or arena growth is
needed.

The returned error is always meaningful. Distinguish domain/exponent outcomes,
resource rejection, and an achieved-bound result that does not meet an optional
relative target. Insufficient relative accuracy is handled by the outer owner
at a preparation boundary; it is not a failed carry invariant.

## 4. Implemented reduced bit-burst exp/sincos algorithm

Assume the exact reduced argument r satisfies |r|≤1. The broad domain is useful:
coarse reduction need not determine the uniquely nearest mathematical multiple
of log2 or pi/2. It only needs to choose an exact integer and certify this
residual domain plus its reduction error.

Let W be the working fractional bits, rounded upward to a limb boundary, and
U=2^-W. Choose

\[
 s=\lceil\log_2 W\rceil+3,\qquad y=r/2^s.
\]

Thus |y|≤2^-s. Define B0=s−1 and B(j+1)=min(2 Bj,W), giving the strict
first-window bound |y|<2^-B0 even at r=±1. Extract from |y| the bit windows

\[
 \delta_j=\operatorname{sign}(y) A_j2^{-B_{j+1}},\qquad
 0\le A_j<2^{B_{j+1}-B_j}.
\]

Here a=Bj, b=B(j+1), |δj|<2^-a, b≤2a and the discarded remainder is smaller
than U. There are K=O(log W) slices. The last interval need not have a
power-of-two width. Extracting the disjoint bit windows costs O(W) total data
movement, apart from bounded descriptor work.

Use exact finite binary splitting for each slice, then approximate ratio
conversion to the W-bit fixed grid. Accumulate exp factors or sincos pairs on
that grid. Finally undo s halvings with squaring / angle doubling. These extra
O(log W) full-precision steps are intentional: they give simple term-count and
operand-height proofs for this implementation. Later tuning can
reduce s without changing the interface.

### 4.1. Expm1 slice as the existing CommonP2B3 recurrence

For z=A 2^-b, retain

\[
 E_N(z)=\sum_{k=1}^{N}\frac{z^k}{k!}.
\]

On k=1,…,N, use

\[
 F_k(v)=\frac{z+zv}{k},\qquad
 T_k=U_k=A2^{-b},\quad D_k=k.
\]

Then `F_1 o ... o F_N (0)` is exactly E_N. This matches the existing
CommonP2B3 representation `F(v)=(T+U v)/D`. The denominator is positive;
the numerator may be signed. Do not expand the binary denominator 2^b into a
stored integer: its exponent belongs in `sbn3_series_value.exponent2`.

The absolute omitted tail is at most 2|z|^(N+1), since |z|≤1/2 and the
factorials only improve convergence. The simple sufficient choice

\[
 N=\left\lceil\frac{W+2}{a}\right\rceil
\]

therefore leaves a tail below U/2. This conservative integer rule avoids
depending on libm rounding for certification. Later factorial-aware bounds can
reduce N without altering the proof structure.

The exact finite sum has the sign of z on this domain. Convert its magnitude
through the existing positive approximate-ratio terminal and restore the sign.
With its documented <2U conversion error, the total slice error is <3U.
Adding 1 on the fixed grid is exact. For expm1 accumulation, retain increments
instead of repeatedly subtracting 1: `u <- u + d + u*d`, where d=E_N(δ).

### 4.2. Sincos slices with two real trees

Use

\[
 S_N(z)=z\sum_{k=0}^{N-1}\frac{(-z^2)^k}{(2k+1)!},\qquad
 C_N(z)=\sum_{k=0}^{N-1}\frac{(-z^2)^k}{(2k)!}.
\]

Precompute the exact A² once per slice and share it. Both even polynomials
fit CommonP2B3 with

\[
 F_k(v)=1-\frac{z^2}{q_k}v,
 \quad T_k=D_k=q_k,\quad U_k=-A^2 2^{-2b}.
\]

For `sinc_even`, qk=(2k+2)(2k+3); for `cos_even`, qk=(2k+1)(2k+2).
Compose k=0,…,N−1 at zero. The last denominator cancels mathematically but
is harmless in the uniform representation. Multiply the sinc result by z to
obtain sine. All qk fit a checked 128-bit scalar when the supported index
range is bounded consistently with the existing scheduler.

Choose

\[
 N=\max\!\left(1,\left\lceil\frac{W+2}{2a}\right\rceil\right).
\]

The tails of both even polynomials are bounded by 2|z|^(2N)≤U/2.
Their values and finite sums are positive and bounded away from zero on the
slice domain, so the positive approximate terminal applies directly. Including
the final multiply by z, a conservative sum of coordinate errors is 8U.

Combine pairs with four real products:

\[
 (S,C)\circ(s,c)=(Sc+Cs,\ Cc-Ss).
\]

Same-grid sums/subtractions are exact; four products truncated to the common
grid contribute at most 4U to the sum of coordinate errors. Begin with (0,1).
Undo halvings by `(S,C) <- (2SC, C²-S²)`; its two-coordinate rounding budget
is again at most 4U. A three-product identity or cosine recovered by sqrt is
a later optimization, not a prerequisite for scalable behavior.

### 4.3. Exact tree capacity proof

For an exp interval of n indices, D is a product of scalar indices. Use the
existing integer bit-length bucket machinery to bound its bits by F; a loose
alternative is n ceil(log2(N+1)). In a common exponent representation,

\[
 \operatorname{bits}(T)\le nb+F+2,\qquad
 \operatorname{bits}(D)\le F+1,\qquad
 \operatorname{bits}(U)\le n\,\operatorname{bits}(A)+1.
\]

The T bound follows from the absolute local sum being less than 1 and the
common denominator D 2^(nb). A sinc/cos interval similarly has bounds
`bits(T) <= 2nb + F + 2`, `bits(D) <= F+1`,
`bits(U) <= 2n*bits(A)+1`; its local polynomial has magnitude below 2.
Exponent products, byte sizes and scalar factors use checked arithmetic.

For scheduler envelopes, the oracle must bound every contiguous subrange with
at most `max_terms`, not just the actual split. Use the worst permitted index
range / factor-bit sums and reserve sign/alignment/carry room for intermediate
merges. The numeric magnitude bound does not replace those merge-layout bounds.

Since a≥ceil(log2 W), b≤2a and N=O(W/a), both nb and N log N are O(W).
Each exact tree therefore has O(W)-bit operands, even for the last slice with
a Θ(W)-bit numerator. It costs O(M(W) log W); O(log W) slices give
O(M(W) log² W). Reconstruction and A² precomputations do not exceed this.
With W=p+O(log p) on a fixed well-conditioned domain, this is the promised
O(M(p) log² p) bound. No 4096-bit cap or full-height long Taylor tree is hidden
in the argument.

## 5. Implemented error and working-precision schedule

The current conservative envelope supersedes the initial draft's smaller
constants. It uses the slice tails above, the existing <2U approximate-ratio
result, one-grid-unit product truncations, and exact fixed-point additions.

For exp, maintain the increment u=expm1(y) and write v=1+u. Slice composition
is `u <- u+d+u*d`. Since the sum of slice magnitudes is at most 2^-s, the
product of exact amplification factors is below two. Slice errors, product
errors and the discarded input tail give initial absolute error at most
`16(K+1)U`.

Increment reconstruction `u <- u(2+u)` is equivalently `v_next=v²+delta`,
with |delta|<U. In the multiplicative error coordinate h=log(v)−r_j,
the small-error induction gives

\[
 |h_{j+1}|\le 2|h_j|+8U.
\]

The initial coordinate error is at most four times the initial absolute error;
converting back at |r|≤1 costs less than six. The final exp or expm1 absolute
error is below `512*2^s*(K+1)U`. This also accounts for increment-product
rounding; cancellation avoidance does not remove that budget.

For sincos, use z=C+iS. A slice pair has norm error at most 8U. Four-product
composition and doubling each have local Euclidean error below 3U (the
coordinate-sum budget in §4 is 4U). Initial norm error is at most
`16(K+1)U`. A complex logarithm error coordinate near the unit circle gives
`|h_next| <= 2|h|+24U`, and final component errors below
`128*2^s*(K+1)U`.

The implementation publishes the common envelope

\[
 2048\,2^s\,2^{\lceil\log_2(K+1)\rceil}U,
\]

and explicitly checks

\[
 W\ge t+s+\lceil\log_2(K+1)\rceil+16.
\]

Here `W=round_up_to_64(t+64+4*ceil_log2(t+2))`. All ceiling logs, capacity
checks and admission tests are integer planning operations. This condition
makes the small-error inductions self-consistent and leaves separate reduction
and final-emission budgets. Output quantization is added to the analytic error;
the bound does not imply correct rounding. The independent
[Frame audit](../experiments/docs/bit-burst-frame-audit-2026-10-08.md)
records the implemented recurrence, capacity and lifetime checks.

## 6. Constants and coarse reduction without dependency cycles

### Implemented bootstrap and alternative constant recipes

The active float dependency graph is:

```text
integer products / bounded divide / general sqrt
    -> Gauss-Legendre pi -> independent power-of-two AGM log2
    -> immutable pi/log2 constant bank

bank + independent real AGM log -> Newton exp
bank + restricted complex AGM atan -> Newton tan -> reduced sincos

finite Frame trees + normalized ratios -> independent reduced bit-burst
                                        exp / expm1 / sincos
```

The cold producers are in [constants.cpp](../src/float/constants.cpp); shared
values and construction-work lifetimes are in
[constant_bank.hpp](../src/float/constant_bank.hpp). They do not invoke generic
log or exp to initialize themselves. π/log2 banks have passed native and
sanitizer integration gates. Prime-log and angle catalogs are not implemented
entries in this float bank.

The following remain alternative producer designs, not the active bootstrap.
For log2, existing rational-atanh formula definitions could use their private
approximate execution and a bounded linear-combination finish with
the same coefficient amplification bound, instead of `formula_sum`'s final
guard separation. The currently exposed <2-unit component bound and outer
coefficient/divisor scaling give the constant error. A fixed catalog identity
for 2 suffices; no online relation discovery is necessary on every exp call.

For pi, [pi.cpp](../src/series/pi.cpp) already computes a bounded result before
its final `guard_separated` call. An alternative private bounded finish must preserve and
return the actual finish error, including the fused or separate rsqrt/product
recipe, then add any dropped-word error. Do not silently reinterpret the public
exact-floor pi routine. Retained constants are immutable caller/context-owned
dyadics with precision/error and mathematical identity; charge generation and
complete retained bytes in fresh calls.

The constant definitions fit existing small-parameter infrastructure. This
does not mean the public `sbn3_formula_def` can represent arbitrary burst
numerators: its 128-bit coefficient fields remain deliberately bounded.

### Coarse reduction

Choose an exact integer q from a low-precision quotient estimate and compute
the residual at admitted precision:

\[
 r=x-qC,\quad C=\log2\ \text{or}\ \pi/2.
\]

Use |q| times the constant error plus the subtraction error to enclose r.
The chosen q need not be the uniquely nearest quotient; validate |r|≤1.
That avoids requiring a difficult exact quotient-boundary decision merely to
enter a larger safe core domain. For exp the result is `2^q exp(r)`; for sincos
only q mod4 is required for final signs/swaps.

When |x|≈2^E, reduction requires about t+max(E,0) constant bits plus guards.
The implemented [general trig reducer](../src/float/trig.cpp) uses a window of
the fixed-point 2/pi coefficient and retains only the phase and q mod4. For
x=M2^e it forms the relevant low J+2 bits of M times that coefficient window,
where J=L−e and L≈max(E,0)+t+guards. Input-prefix staging and modular product
work depend on the retained input height plus t rather than a full quotient.
The constant/inverse bank still costs O(max(E,0)+t) bits and is fully quoted;
the window does not make arbitrary huge-argument reduction O(t) storage.
Excessive requests are rejected before execution.

For exp, q must fit the bounded public exponent range; classify overflow or
underflow before creating a huge residual plan. For sincos, finite huge x has
bounded output but potentially expensive reduction. A capacity result is
appropriate when its requested reduction exceeds admitted resources.

Multiprime reduction is a later candidate over the same reduced evaluator.
Its prime logs must come from independent atanh/Machin recipes, and Gaussian
angles from independent rational atan recipes. They must not call generic log
or atan implementations that recurse into their own uninitialized exp/sincos
constant caches.

## 7. Log and cancellation-safe variants

### Alternative log recipe: inverse of an independent exp kernel

The active public log uses local atanh or independent AGM, and public exp
inverts that log. The reverse dependency described here remains a valid
alternative when explicitly bound to the independent bit-burst exp evaluator;
it is not the current public dispatch and must not call generic exp recursively.

Normalize a positive x by an exact power of two. Keep x near 1 unscaled where
possible so a needless subtraction of log2 is avoided. Evaluate the normalized
log using a recursively more accurate seed and the identity

\[
 \log x=y+\log(1+\delta),\qquad \delta=x\exp(-y)-1.
\]

The seed y is a stored exact dyadic; its known error is used to bound δ, not
added to the final result as if the residual correction did not cancel it.
For a simple quadratic correction, if the seed error is at most η≤1/4, then
|δ|≤2η. For |δ|≤1/2,

\[
 |\log(1+\delta)-\delta|\le\frac{|\delta|^2}{2(1-|\delta|)}
 \le |\delta|^2.
\]

A seed with about `(t+6)/2` correct absolute bits therefore leaves a correction
remainder at most 2^(-t-4). Bound the evaluated exp, multiplication, subtraction
and final addition separately. Validate a bound on the actual residual; a host
libm seed can be a guess but is not the certificate. A short normalized atanh
series supplies an independent certified base seed.

Precision grows geometrically, so lower-precision calls do not add another
log p factor to an O(M(p)log²p) exp. Higher-order short log1p corrections can
replace the quadratic step later for constants, with the same private interface.
Reconstruct `k*log2 + log(m)` with its propagated absolute error.

### expm1 and log1p

For tiny exact x, return a small polynomial with a scaled error instead of
forming 1+x on a p-bit grid. For |x|≤1/2,

\[
 |\operatorname{expm1}(x)-x|\le |x|^2,
\qquad
 |\log(1+x)-x|\le |x|^2,
\]

and the quadratic log1p approximation has tail at most
`|x|³ / (3(1-|x|))`. Select the tiny branch when its bound meets the requested
absolute/relative goal, not solely from a machine-double comparison.

Otherwise, if |x|≈2^-c and the requested result has p significant bits, use an
absolute working target around 2^(-p-c) plus guards. When c is very large the
tiny polynomial has already become sufficient; do not allocate c leading zero
bits. In the remaining domain c=O(p), this extra precision preserves the
asymptotic bound. expm1 uses increment products/doubling; log1p forms 1+x with
the required additional precision or uses its atanh reduction. log near 1
should preserve the exact difference x−1 and reuse this path.

### Sincos accuracy near zeros

The reduced kernel provides an absolute pair bound. To promise relative
accuracy for a tiny sine/cosine, request enough residual accuracy and constant
precision. Exact x=0 and exact sinpi/cospi integer/half-integer cases are
handled directly. For ordinary trig at a nonzero dyadic near a multiple of pi,
the required extra bits depend on the residual, not just the input exponent.

Return the achieved absolute bound in every case. An optional significant-bit
goal is accepted only when a lower magnitude bound proves it; otherwise the
outer owner can refine at a safe query/reserve boundary. Do not silently call
an absolute 2^-p error “p correct relative bits”.

sinpi/cospi first reduce the exact dyadic x modulo small integers, then multiply
only the small residual by pi. This preserves exact roots and avoids huge
ordinary-trig reduction for large integer parts.

Hyperbolic and inverse wrappers reuse expm1/log1p and the shared root core:
for u=expm1(a), a=|x|, use sinh(a)=u(2+u)/(2(1+u)); tanh near zero uses
expm1(2a)/(expm1(2a)+2), while large a uses exp(−2a). Near x=1, acosh uses
`log1p(d+sqrt(d*(d+2)))`, d=x−1. These compositions need ordinary error
propagation and domain checks, not new large series engines.

## 8. AGM route and the best-known asymptotic bound

The implemented real AGM/root core supplies the independent logarithm recipe:

\[
 \log x\approx\frac{\pi}{2\operatorname{AGM}(1,4/z)}-m\log2,
 \qquad z=2^m x.
\]

Choose the expansion and precision from a rigorous elliptic remainder, not
merely an iteration difference. Ordinary forward AGM carries essentially full
precision, takes O(log p) steps and costs O(M(p)log p) with O(M(p)) roots.
Guard cancellation in the final subtraction, or use the better theta-based
AGM log identity described by the AGM design. Small log1p cases remain separate.

Exp solves `log(y)=x` by Newton, `y <- y*(1+x-log(y))`, with a
geometric precision schedule and certified residual. This yields the same
O(M(p)log p) complexity. The call graph selects a concrete independent log
recipe; it cannot invoke the generic log dispatcher that might invert exp.

**Cold constants are part of this claim.** The active independent
Gauss–Legendre/Brent–Salamin pi recipe uses general roots/AGM plus its coupled
correction sum. log2 is bootstrapped independently from a large exact power of
two:

\[
 \log2\approx\frac{\pi}
 {2m\operatorname{AGM}(1,2^{2-m})}.
\]

Taking m on the order of p/2 makes the elliptic approximation error
exponentially small; the exact m must be chosen from the proved bound including
division by m and AGM/constant errors. This formula avoids calling generic log
to initialize log2. A coupled pi implementation is a distinct recipe, not just
the value-only positive AGM routine with a different output label.

The active [restricted complex AGM path](../src/float/trig_agm.cpp) closes the
trig dependency explicitly. For |t|≤2, atan(t) is the imaginary part of the
principal log(1+it), evaluated inside a fixed right-half-plane cone. For a
reduced |r|≤1, real Newton solves atan(t)=r with |tan(r)|<2, then computes
`(sin(r),cos(r))=(t,1)/sqrt(1+t²)`. Certified seeds, geometric precision,
branch-preserving complex roots and norm-error transport are specified in the
[trig design](../experiments/docs/high-precision-trig-research-design-2026-10-07.md)
and [primitive contracts](../experiments/docs/complex-agm-primitive-contracts-2026-10-07.md).
Thus reduced trig has the O(M(p)log p) AGM route as well as the independent
O(M(p)log²p) bit-burst route. Neither bound establishes a practical crossover;
general argument reduction has the additional exponent-dependent cost in §6.

## 9. Storage, parallel execution and validation status

Bit burst evaluates slices sequentially through one quoted phase pool. Live
numeric state is the reduced argument, accumulator, one slice/A², one exact
tree root and temporary arithmetic values. Every slice retains its immutable
schedule and product plans. A bounded shared directory retains mandatory fixed
tables plus at most the largest slice's table allowance; a missing table is
rebuilt from its frozen plan within the phase, without a geometry query.
That work still counts in any execution measurement.

Query sums simultaneously live values and metadata, then takes the maximum
of mutually exclusive finite-tree, square, ratio and reconstruction phases.
The principal numeric payload is O(W), conditional on linear-space product
and Newton providers; retained schedule records are separately quoted at
approximately O(workers log² W). Full-span prepared children in the other
AGM/composite paths are summed while simultaneously live, not incorrectly
treated as interchangeable workspaces. Constant banks distinguish immutable
values from reusable construction work. No operation-sized cache survives
its declared owner.

Parallelize admitted tree subproblems and backend operations with the existing
Team first. Later independent slices may use disjoint subteams if the complete
live-memory grant allows it; a full near-p-bit private state per slice is not
free. Both schedules produce the same mathematical slice/error descriptors.
OOC lowering and device experiments are outside this initial resident design.

The bounded-error foundation, independent cold constants/shared bank, private
wide-leaf Frame trees, bit-burst controller, AGM log/Newton exp, and complex
AGM atan/Newton sincos are implemented. Cancellation-safe and general reduction
frontends are present. The alternative exp-inverted log in §7 and alternative
constant producers in §6 are not required by the active call graph.

Parent-owned correctness gates include 25 combined sanitizer targets, 15 native
and 21 sanitizer bank-integration targets, and the native/sanitizer bit-burst
counts reported at the top. The bit-burst oracle uses independent scalar-limb
integer operations and rational interval bounds; actual >64-limb execution is
checked rather than inferred from a query. Prepared reuse, non-limb precisions,
signed/tiny cases, active scopes, aliases and watched allocation are included.

Tuning direct/rectangular leaves, reduction depth, paired sine-plus-root,
precision tapering, prime/angle tables and parallel chunks remains separate.
Future performance acceptance needs non-power-of-two shape/scaling and
smoothness coverage, and complete cold calls including query, constants,
preparation, execution, emission and cleanup under the benchmark lock. No
timing or performance-completion claim is made here.

## Primary algorithm references

- Brent and Zimmermann, [Modern Computer Arithmetic, chapter 4](https://members.loria.fr/PZimmermann/mca/mca-cup-0.5.9.pdf): binary splitting, bit burst, inverse iteration, AGM and error/guard issues.
- Johansson, [medium-precision elementary functions](https://arxiv.org/html/1410.7176v2): fixed-point arithmetic, rectangular splitting, denominator blocks and local error contracts; later optimization material.
- Johansson, [multi-prime argument reduction](https://arxiv.org/pdf/2207.02501): separate reduced evaluator and reduction policy, independent constant precomputation, high-order inverse corrections.
- [GNU MPFR sin_cos.c](https://gitlab.inria.fr/mpfr/mpfr/-/raw/master/src/sin_cos.c): real binary-split dyadic slices and angle reconstruction; implementation reference, not code to import without preserving notices.

The conservative halving, leaf formulas, capacity bounds and error schedule
above are repository derivations encoded with checked integer bounds and
reviewed against the implemented primitives. Future changes need the same
capacity/error and lifecycle review; historical crossover thresholds are not
part of this design.
