# Positive-real bounded-error AGM design

Original scalar design: 2026-10-07; implementation-status update: 2026-10-08. This document specifies the value-only AGM program. Coupled complete K/E, its weighted-square correction, direct complementary-parameter input and certified coupled closure are now implemented separately in [elliptic.cpp](../src/float/elliptic.cpp), with a reusable interface in [elliptic.hpp](../src/float/elliptic.hpp). A value-only termination certificate must not be reused as a moment/E-tail certificate.

The implementation is [agm.cpp](../src/float/agm.cpp), with a private reusable interface in [agm.hpp](../src/float/agm.hpp). It uses the existing finite-sum, prepared float-product, prepared root and fixed-size error-magnitude services. It adds no allocator, worker pool, public ball type or runtime precision-refinement loop.

## 1. Numerical contract

For exact positive finite input points a,b, compute a p-bit dyadic y and a local absolute error report for M=AGM(a,b). For finite nonzero normal output, the published bound is at most

\[
 2^{T_y-p}\quad\text{for nearest-even},\qquad
 2^{T_y-p+1}\quad\text{for other mantissa-rounding modes},
\]

where \(2^{T_y-1}\le y<2^{T_y}\). These are one/two units of the requested p-bit grid, not a claim of correctly rounding the exact AGM. The error exponent is independent of y's stored low-bit exponent.

The final point conversion retains the core's exponent-range semantics. A final overflowing approximation returns infinity with overflow and an unbounded error report. This differs from avoidable overflow of an intermediate a*b or a+b, which this implementation prevents.

Domain behavior:

| Input case | Result |
|---|---|
| Quiet NaN input | Quiet NaN |
| Negative nonzero finite input or negative infinity | NaN, invalid |
| Two finite nonnegative inputs, either zero | +0, exact |
| Positive infinity with zero | NaN, invalid |
| Positive infinity with a positive finite input, or two positive infinities | Positive infinity |
| Equal positive finite values | Round that exact point directly to output precision |

Negative zero is accepted as zero and AGM's zero result is canonical +0. Exact-base output/input aliases are supported; inputs are copied into private state before any nontrivial caller-output write. Partial overlaps and overlap with prepared storage are rejected.

## 2. Prepared program and storage

Public entry points are `agm_requirements(out,a,b,workers)` and `agm_into(out,a,b,execution,rounding)`, declared in [float.hpp](../include/sbn3/float.hpp). The [API guide](float.md) describes their exact-point and local-error conventions.

Private interfaces:

- agm_program_requirements(bits,workers): full general-purpose program storage;
- agm_program_prepare(bits,workers,storage): compile and prepare the reusable program;
- agm_program_execute(prepared,out,a,b,execution,rounding): serial reuse with fresh exact points.

The prepared program reserves:

1. Metadata, immutable product plan and prepared handles.
2. Four q-bit buffers: current a/b, next arithmetic mean (also gap scratch), and geometric product/root.
3. One prepared q-bit multiplication program and its exclusive storage.
4. One prepared q-bit general square-root program, including every Newton rung and its tables.
5. A reusable finite-sum window for working precision above the stack-leaf cutoff.

All product/root plans are compiled before the first AGM iterate. Neither prepared product nor prepared root execution queries a backend. The root's internal precision-doubling ladder is legitimate Newton refinement inside an individual root call; the forward AGM common state stays at q bits throughout.

One outer Team epoch covers the nontrivial standalone iteration chain. An
existing active scope is used directly; prepared product/root calls receive
that scope instead of creating an epoch for each iterate. Special/equal-input
paths can finish without an arithmetic epoch.

Descriptors for the four buffers swap after each iteration. No full iteration trace is stored, and the product buffer is reused as the root output after its input is staged. Private program storage is exclusive; an active-use guard rejects concurrent reuse. Caller inputs/output remain disjoint from the entire program span.

The quote uses checked size additions and the maximum required alignment. Product/root storage remains disjoint because their prepared tables and control objects have declared lifetimes; this stage does not assume that their dead workspace may be borrowed without an explicit capability.

Numeric rolling storage is O(p). Complete storage additionally includes the existing root/product prepared tables and metadata; those services' storage bounds are charged rather than replaced by an unproved flat O(p) byte estimate.

## 3. Working precision and finite iteration budget

The public exponent domain bounds each input's one-past-top exponent by ±L, with \(L=2^{60}\). Define

\[
 N=72+\operatorname{bit\_width}(p+8),\qquad
 q=p+\operatorname{bit\_width}(N+1)+16,\qquad
 \eta=2^{8-q}.
\]

bit_width(N+1) is ceil(log2(N+2)). Let s denote that quantity. Since s>=7 and p>=1, q>=24 and eta<=2^-16. The 16 extra bits follow from the error/convergence inequalities below; they are not a performance-tuned threshold. The query rejects a shape if q or a backend requirement exceeds its admitted domain; execution does not discover new storage needs.

N is a conservative universal limit for this public exponent domain, not the number of iterations normally executed. Each iteration first tests the certified gap, allowing ordinary early termination. The ratio transient depends on \(\log(1+\log(\max(a,b)/\min(a,b)))\); bounded-ratio inputs require O(log p) iterations.

### Ratio transient

Let \(r=\max(a,b)/\min(a,b)\ge1\). For exact arithmetic,

\[
 r_{\rm next}=\frac{r+1}{2\sqrt r}\le\sqrt r.
\]

If each output coordinate has relative error at most eta, sorting the approximate pair gives

\[
 \log_2\widehat r_{\rm next}
 \le\tfrac12\log_2\widehat r+8\eta.
\]

For eta<=1/4, the last term follows from
log2((1+eta)/(1-eta))<=8 eta. Initially, log2(r)<2L+1=2^61+1. After 66 steps,
\[
 \log_2 r_{66}
 <\frac{2^{61}+1}{2^{66}}+16\eta
 <\frac1{16}+16\eta<1.
\]
Thus the ratio is below 2. The positive 2^-66 contribution from the initial +1 is retained in the first inequality; dropping it would make the sharper-looking 1/32+16 eta expression unjustified.

### Quadratic phase with rounding

For \(d=1-\min(a,b)/\max(a,b)\),

\[
 d_{\rm next}
 =\frac{(1-\sqrt{1-d})^2}{2-d}\le d^2.
\]

The approximate update satisfies \(d_{\rm next}\le d^2+2\eta\). Starting from d<=1/2, one step gives d<=1/4+2 eta. Thereafter induction gives

\[
 d_j\le 2^{-2^{j+1}}+8\eta.
\]

Indeed, for t<=1/4 and eta<=1/32, squaring t+8 eta and adding 2 eta is at most t²+8 eta. Moreover,
\[
 8\eta=2^{-p-s-5}\le2^{-p-12}.
\]
Thus O(log p) further steps reduce the gap below the stopping threshold even with the rounding floor. After enough steps to make the first term at most 2^(-p-8), the relative gap is at most (1/256+1/4096)2^-p. The computed gap bound is less than four times the exact gap at q>=24 (also when the gap underflows); this is still below the 2^(-p-4) stopping allowance. N includes the initial quadratic step and several additional steps. A failure to stop within N is an invariant/certificate failure, not permission to allocate or change precision.

## 4. Initial values and range-safe iteration

### Prefix initialization

Copy at most q significant bits of each positive exact input, truncating downward. If the source has w>q bits and scale e, the omitted tail is less than \(2^{e+w-q}\), while the source is at least \(2^{e+w-1}\). Therefore its relative error is less than \(2^{1-q}<\eta\).

Discarded tails are bounded analytically without scanning them. No claim that a nonzero tail was observed is made. This preserves the point-input contract, bounds initialization work by the retained prefix, and avoids rounding the largest admissible input upward to infinity.

The equal-input shortcut may require a full exact comparison for differently encoded equal points; input view construction/validation can also have an input-height cost. This cost must not be hidden in a precision-only complexity claim.

### Arithmetic mean

Construct half-scaled views a/2 and b/2 first, then call the private finite-sum path. The half-scaled views may temporarily fall one bit outside the public exponent domain, but the mean lies between the current valid positive endpoints. Computing public a+b and halving afterward would wrongly introduce avoidable overflow.

Let y be the rounded mean with absolute local bound E and one-past-top exponent T. Since y>=2^(T-1), if the computed bound

\[
 E\,2^{2-T}\le\eta
\]

holds, then E/y<=eta/2 and E/(y-E)<=eta. Thus the mean's relative error against the exact update is at most eta. The implementation checks this inexpensive report-derived bound.

### Geometric mean

Write

\[
 a=m_a2^{E_a},\quad b=m_b2^{E_b},\quad
 m_a,m_b\in[1/2,1),\quad E_a+E_b=2h+r,\ r\in\{0,1\}.
\]

Floor division is used for negative exponent sums. The reusable multiplication computes P approximating \(m_am_b\); the reusable root computes R approximating \(\sqrt{P2^r}\). The exponent h is kept separate until afterward.

Let E_P and E_R be the existing local absolute reports. The true normalized geometric mean G is at least 1/2. Therefore

\[
 |\sqrt{P2^r}-G|
 \le 2^{r+1}E_P,
 \qquad
 \frac{|R-G|}{G}
 \le 2(E_R+2^{r+1}E_P).
\]

The program checks that this upper bound is at most eta. Both operations run at q significant bits; current inputs have at most q bits, so multiplication does not discard an unobserved long-input prefix here. The existing contracts have ample margin at q>=24: conservatively E_P<=2^-q and E_R<=2^(1-q), giving a relative geometric bound at most 12*2^-q before magnitude accumulation, versus eta=256*2^-q. It remains well below eta after outward accumulation and power-of-two publication. The mean's analogous relative bound is at most 2^(1-q).

Restore the exponent h in metadata, then project the candidate onto the previous [min(a,b),max(a,b)] interval. The exact geometric mean belongs to this interval, so projection cannot increase its absolute error. Top-exponent comparisons are performed before public comparison, allowing an out-of-public-range candidate to be clamped without exposing it to a public input validator. The ordinary path changes only exponent metadata; a clamped endpoint is copied exactly.

This prevents avoidable intermediate overflow/underflow, including inputs close to either exponent boundary and inputs separated by nearly the entire exponent domain.

## 5. Common-error propagation

AGM is monotone in each positive input, homogeneous of degree one, and invariant under an exact AGM update. Thus if both coordinates at each step are perturbed by relative factors in [1-eta,1+eta], then after n updates, including one initialization factor,

\[
 (1-\eta)^{n+1}M_{\rm original}
 \le M_{\rm current}
 \le(1+\eta)^{n+1}M_{\rm original}.
\]

Writing k=n+1, the chosen q gives
\[
 2k\eta\le2(N+1)2^{-p-s-8}<2^{-p-7}.
\]
In particular k eta<1/2. Bernoulli's inequality gives \((1-\eta)^k\ge1-k\eta\), hence

\[
 |M_{\rm original}-M_{\rm current}|
 \le2k\eta M_{\rm current}
 \le2k\eta U,
 \quad U=\max(a_n,b_n).
\]

The implementation uses the easy magnitude upper bound U<2^T, giving

\[
E_{\rm propagated}=2(n+1)\,2^{T+8-q}.
\]

This is strictly less than 2^(T-p-7), or 1/128 of the requested p-grid unit at U.

This is a recipe-level proof with small metadata arithmetic. It does not need full interval operations on every limb or an independent recomputation of the limit. It also explains why early low-precision forward AGM steps are invalid: their common error is preserved, even though the difference mode contracts.

## 6. Gap stopping and final report

Compute the nonnegative difference D=U-L using the exact finite-sum path at q bits, with report E_D. For a nonzero returned gap center g, bound |g| by \(2^{T_g}\); for a zero center use zero. Define

\[
 E_{\rm gap}=\operatorname{mag\_upper}(g)+E_D.
\]

This remains valid when a tiny positive difference underflows the public point range: the core's absolute error report still bounds it. The error bound itself has an independent exponent.

Stop when

\[
 E_{\rm gap}\le2^{T-p-4}.
\]

The exact current AGM lies between L and U, so returning U incurs error at most this gap. Round U once to the requested p bits and combine

\[
 E_{\rm total}=E_{\rm final\ round}+E_{\rm gap}+E_{\rm propagated}.
\]

Writing Q=2^(T-p), the component bounds give
\[
 E_{\rm total}<\frac{73}{128}Q
 \quad\text{(nearest-even)},\qquad
 E_{\rm total}<\frac{137}{128}Q
 \quad\text{(other modes)}
\]
before the fixed-size error accumulator's rounding. Its two upward additions introduce only 64-bit-significand rounding, far below the remaining margins: the totals remain below Q and 2Q respectively.

The private ErrorMagnitude accumulator adds these bounds outward without repeated power-of-two inflation. Publishing a power of two therefore gives at most one p-grid unit for nearest-even and two for other modes, for normal outputs. The rounded positive U cannot enter a lower binade, so using the final output's grid scale preserves this bound.

The final report's rounded flag comes from observed final mantissa rounding. A positive conservative total bound does not assert that the mathematical result is inexact.

## 7. Complexity, separate elliptic consumer and validation

Under the existing O(M(q)) general-root/product contracts, the arithmetic cost is

\[
 O\!\left(M(p)\left[
   \log p+\log(1+\log(\max(a,b)/\min(a,b)))
 \right]\right),
\]

with q=p+O(log N), and with the input-height costs stated separately. On a bounded-ratio domain this is O(M(p) log p). This is an algorithmic bound, not measured performance acceptance. All planning, table construction and disposal still belong in future fresh-call timing.

A later division-based rationalized-gap or short-series closure may remove expensive final root steps. Its remainder and arithmetic error must be proved independently. The current baseline deliberately has no division dependency.

K/E requires an additional weighted-square moment and an independent tail bound, especially near the complementary-parameter singularity. The separate elliptic consumer owns those corrections and its closure ledger; the scalar program remains value-only. Its real parameter API includes `m<=1` and direct `c=1-m>=0`, with the negative-parameter transform and the `m=1` endpoint handled explicitly. It can borrow the immutable pi bank described in [float_constants.hpp](../include/sbn3/float_constants.hpp).

Maintained validation is in [float_agm_test.cpp](../tests/integration/float_agm_test.cpp), [float_elliptic_test.cpp](../tests/integration/float_elliptic_test.cpp), [float_constant_binding_test.cpp](../tests/integration/float_constant_binding_test.cpp), and [float_program_replay_test.cpp](../tests/integration/float_program_replay_test.cpp). The suites exercise independent enclosures, zero/equality cases, symmetry/homogeneity, non-limb precisions, rounding modes, extreme scales/ratios, aliases, prepared reuse, Team/scopes, allocations and the explicitly covered query entries. Their receipts belong to the implementation's correctness gates; this document update runs no compilation/tests and makes no performance-acceptance claim.
