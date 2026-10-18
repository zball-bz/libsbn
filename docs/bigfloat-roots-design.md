# General multiprecision roots: bounded-error design

Design, 2026-10-07, followed by the implementation in `src/float/roots.hpp` and `roots.cpp`. General sqrt/rsqrt now have public owning and controlled entry points; see the [float guide](float.md) and [implementation evidence](bigfloat-implementation-plan.md). No performance result is claimed here. Follow [runtime.md](runtime.md), [product-windows.md](product-windows.md) and the bounded-error direction in [bigfloat-roadmap-2026-10-05.md](bigfloat-roadmap-2026-10-05.md). General sqrt is also used by real AGM. Neither root uses digit-by-digit extraction or a large exact division as its main algorithm.

## Existing facilities and limits

| Facility | Reuse | Limit that matters |
| --- | --- | --- |
| `algorithms/rsqrt.hpp`, `rsqrt.cpp` | The certified, constant-size `rsqrt_seed`; ladder/stage ownership pattern | Radicand is u64. Each rung squares R and scales by `mul_1`; its low-square residual proof is specific to that scalar. Widening the argument type does not generalize the proof. |
| `algorithms/reciprocal.hpp`, `refinement.hpp` | Mathematical precision versus backend geometry separation, signed residual/window contracts | Existing formulas and magnitude constants are reciprocal/quotient formulas, not general rsqrt formulas. |
| `product/program.hpp` | Precompiled ordinary products; preparation in caller frames; execution on an existing scope | Actual output capacity can exceed the mathematical product width and must be quoted. Preserve backend alias contracts. |
| `product/window.hpp` | Exact full-product fallback, signed cancellation, bounded mathematical windows | A low modular window is not an approximate high product; sign/unique lift needs a magnitude proof. |
| `product/local_program.hpp`, `product/stage.hpp` | Replay metadata and phase lifetime pattern | Existing local window providers have shape/reuse assumptions tailored to current two/three-product groups. Check them before admitting a new root group. |
| `algorithms/newton_service.cpp` | Complete queries, precompiled phase choices, pooled tables/work/spectra | Public `newton.h` currently promises u64-radicand rsqrt and special sqrt(2), not a general root. Keep those contracts unchanged. |
| `runtime/scratch.hpp`, `team.h` | Bounded Frame subdivisions, single Team and joined scopes | Do not create workers or grow an arena inside root execution. |

One important trap: the reference `ProgramMultiply` adapter in `product/program_windows.hpp` calls `product_program_query` in its execution operator. Do not use it unchanged as the production general-root executor. Compile all rung product descriptors before arithmetic, then replay them through `ProductProgram` or an admitted stage provider.

## 1. Mathematical contract and normalization

The input finite dyadic is exact unless its caller separately supplies an uncertainty bound. For positive x, write

\[
x=a\,2^{2e},\qquad 1\le a<4.
\]

Given `E=floor(log2(x))`, choose `e=floor(E/2)` and obtain a through a bounded limb/bit view. For a negative odd E, C++ division toward zero is wrong: use mathematical floor division. Exponent addition, negation, parity and `64*n` calculations use checked wider intermediates. No array proportional to an exponent gap is allowed.

The normalized kernels return nonnegative fixed-point integers with n fractional limbs, `B=2^64`, such that

\[
|y-1/\sqrt a|<2B^{-n},\qquad |s-\sqrt a|<2B^{-n}.
\]

Here y and s denote the represented real fixed-point values, not their integer limb arrays. Store one integer limb in addition to the n fractional limbs. Restore the scale by multiplying y by `2^-e` or s by `2^e`. The resulting absolute bounds are respectively `2^(1-64n-e)` and `2^(1-64n+e)`.

The float frontend handles zero, signs, infinities, NaNs and exponent overflow according to the agreed value contract before entering this positive finite kernel. It also performs final p-significant-bit mantissa rounding and adds that rounding error to the kernel bound. A simple initial choice is `n=ceil((p+3)/64)`. Rounding the returned point to nearest does not imply correctly rounded sqrt/rsqrt of x. Strict residual/tie decisions are a separate later capability.

The initial implementation deliberately uses the more conservative `n=ceil(p/64)+2` (two complete guard limbs), then combines the kernel bound with `round_magnitude`'s local report. Nearest mantissa rounding normally gives a bound of at most one result ulp; the other modes conservatively give at most two. These modes round the approximate mantissa, not a guaranteed lower/upper bound for the mathematical root.

If an input prefix has q fractional limbs,

\[
a_q=\lfloor a B^q\rfloor B^{-q},\qquad 0\le a-a_q<B^{-q}.
\]

This bound needs no scan of all omitted low limbs. Read only the prefix required by the current rung from the immutable normalized source. An exactness flag or a strict rounding proof might need a separate sticky witness; bounded-error roots do not. If source normalization itself requires scanning leading zeros, charge that input work once rather than describing it as O(output precision).

## 2. General reciprocal-root ladder with an explicit proof

Use

\[
F_a(y)=y+\frac y2(1-a y^2).
\]

For `r=1/sqrt(a)` and `d=y-r`, exact algebra gives

\[
F_a(y)-r=-\tfrac32\sqrt a\,d^2-\tfrac12 a d^3.
\]

For `1<=a<4` and `|d|<=1/2`, its magnitude is at most `4d^2`. This supplies quadratic convergence with constants independent of the original exponent.

Maintain the rung invariant `|y_m-r|<2B^-m`. For a target n, take `m=floor(n/2)+1`; thus `m<n<=2m-1` when `n>2`. Use `q=n+2` fractional working limbs. Let `T_q` truncate a signed value toward zero to the q-fractional-limb grid. Evaluate:

1. `z = T_q(y_m*y_m)`.
2. `t = T_q(a_q*z)`.
3. `rho = 1-t`, exactly on the working grid, with a separate sign.
4. `c = T_q(y_m*rho/2)`.
5. `v = y_m+c`, exactly on that grid; `y_n = T_n(v)`.

No general division occurs. The factor one half is a binary shift included in the last product's truncation. Cancellation in `1-t` must preserve the signed difference; it is not rounded relative to t.

Here is a conservative arithmetic bound for a full-product implementation. Put `u=B^-q`. The invariant gives `0<y_m<2`. Squaring incurs less than u error. Multiplication by the truncated input contributes at most `4u` from the input prefix, `4u` from the square error and u from its own truncation, hence `|t-a*y_m^2|<9u`. The final correction contributes less than `9u+u=10u`; the addition is exact. Therefore

\[
|y_n-r|<16B^{-2m}+10B^{-(n+2)}+B^{-n}
=\left(1+16B^{n-2m}+10B^{-2}\right)B^{-n}<2B^{-n}.
\]

The last inequality uses `n<=2m-1` and B=2^64. These are truncation-error bounds for exact integer products. A future approximate window provider must replace the appropriate u term with its certified signed-window error, then recheck the invariant. A generic float multiplication's undocumented ulp error cannot simply be substituted.

Compile the ladder backward using `floor(n/2)+1` until a seed of at most two fractional limbs, then execute forward. This directly matches the existing ladder shape, including non-power-of-two targets. All radicand prefixes, precision lengths and resource requirements are known before execution; no iterative convergence search is required inside this kernel.

## 3. Constant-size seed reusing the current u64 seed

Extract `A=floor(a*2^62)`, so `2^62<=A<2^64`. The existing `rsqrt_seed(R,A,2)` guarantees

\[
0\le B^2/\sqrt A-R\le1.
\]

Use the exact dyadic `y_0=2^31 R/B^2`. It approximates the reciprocal root of `A/2^62` within `2^-97`. Since the derivative magnitude of `a^-1/2` is at most 1/2 on `[1,4)`, omission of the remaining radicand costs less than `2^-63`. Thus `|y_0-r|<4B^-1`. The seed conversion needs only shifts and a few limbs; y_0 is exactly representable on the q=4 working grid.

Perform two constant-size Newton steps at q=4 fractional limbs using the four-limb radicand prefix, retaining four fractional limbs between those steps. The first error is less than `65B^-2`; the second is less than `16910B^-4`, using the preceding quadratic and arithmetic bounds. Truncating the result to two fractional limbs therefore gives `<2B^-2`. For a one-limb target, truncate that seed and return the corresponding bound. The exact `a=1` case can return 1 immediately.

This seed inherits the existing u64 seed's floating-estimate-plus-exact-certificate mechanism. It does not trust host sqrt as the final high-precision answer, and it performs no variable-length digit root or long division. A separate tiny seed could be developed later, but is unnecessary for the first implementation.

## 4. Sqrt finishing: simple reference and fused main candidate

The simplest reference finish computes a reciprocal root at `n+1` fractional limbs, multiplies it by a matching radicand prefix and truncates to n. The root error contributes at most `8B^-(n+1)`; prefix and product truncation fit below another `3B^-(n+1)`. Including final truncation leaves `<2B^-n`. This is already O(M(n)), useful as a reference route, and needs no exact multiply-back floor certification.

For the main sqrt path, avoid building a full-precision reciprocal-root result. Compute `y_m` at `m=floor(n/2)+1`, then use three products at `q=n+2` fractional working limbs:

\[
h=T_q(a_q y_m),\qquad
\rho=1-T_q(h y_m),\qquad
s_n=T_n\left(h+T_q(h\rho/2)\right).
\]

In exact arithmetic this equals `a F_a(y_m)`, so the mathematical error is at most `64B^-2m`. The computed h differs from `a*y_m` by less than `3u`. The residual differs from `1-a*y_m^2` by less than `7u`; h is below 3 and the exact residual is below 1/2 on the certified predecessor domain. Bounding all three arithmetic effects by `20u`, then including the n-limb output truncation, gives

\[
|s_n-\sqrt a|<64B^{-2m}+20B^{-(n+2)}+B^{-n}<2B^{-n}.
\]

The positive seed/invariant bounds must be checked in the implementation proof; they are not facts about an arbitrary initial y. Tiny targets can use the simple reference finish. The fused path is an algorithmic reduction in work, not a measured crossover claim. It is different from the current u64 `mul_rsqrt.cpp` recipe and cannot inherit that recipe's error constant unchanged.

## 5. Product requirements and staged window improvements

Start with exact ordinary products followed by known shifts. This preserves O(M(n)) scaling at every precision and works beyond the current public `LOW/HIGH` native limit of 256 input limbs. Do not block general roots on a new short-product implementation, and do not fall back to quadratic large multiplication or repeated exact division.

| Step | Mathematical demand | First lowering | Later bounded-window opportunity |
| --- | --- | --- | --- |
| Rsqrt square | `y_m^2` at q fractional limbs | Exact m-by-m square, then bounded slice | Most or all square precision already fits since q is near 2m; a square specialization suffices |
| Rsqrt residual | `1-a_q*T_q(y_m^2)` at q fractional limbs | Exact full product and signed subtraction | Middle/cancellation band after removing known high unity and unused low digits; derive a long-radicand residual bound |
| Rsqrt correction | `y_m*rho/2` | Exact product, binary shift and truncation | rho has about m leading fractional zero limbs; retain its short significant support with an absolute error certificate |
| Sqrt coarse value | `a_q*y_m` | Unbalanced full product | Rounded high product with a proved discarded-tail bound |
| Sqrt residual | `1-h*y_m` | Unbalanced full product, signed cancellation | Same guarded cancellation machinery with new arithmetic bounds |
| Sqrt correction | `h*rho/2` | Product of h and short significant rho | A short correction window; h is common to residual and correction |

For a uniform q-fractional grid, a product of integer arrays scaled by `B^-q` must be shifted by q fractional limbs on emission; y_m itself may retain its m-limb scale to avoid padding. Record each operand's scale explicitly in the private descriptor. Include the half-bit in correction shifts and all top words in the result support. Normalize a residual's significant window only after determining its exact sign or carrying a certified signed ambiguity; a checksum is not an error certificate.

The existing `WindowGroupShape` can describe at most three products but some providers assume their first and last products share a particular operand. The general rsqrt square/residual/correction does not automatically satisfy the reciprocal provider's pattern. Lower it first as independent precompiled products. For fused sqrt, the coarse h product is followed by a natural two-product group sharing h; that is a reasonable later spectrum-reuse opportunity after compatibility and peak storage are proved. Do not retain a full radicand spectrum across the whole ladder: its demanded prefix/geometry changes, and most rungs use that prefix only once.

## 6. Complete storage, preparation and Team contract

The plan contains a finite ladder, input/output scales, per-rung product descriptors, exact output capacities, errors and phase lifetimes. A simple first allocation retains two private approximation banks. For maximum internal target N and `q=N+2`, a conservative value layout is:

- radicand prefix: `q+1` words;
- two approximation banks: `2(N+1)` words;
- square/correction slot: `q+1` words;
- signed residual: `q+1` words plus scalar sign;
- raw product slot: at least `2q+2` words, enlarged to the actual maximum backend output capacity.

Without extra backend output padding this totals `7N+17` words. It is an explicit conservative layout, not the final minimal quote. Add aligned plan/control storage, per-worker requirements and the maximum simultaneous product tables/workspace/spectra. Caller input/output and Team storage/stacks must be counted in full job admission even when excluded from the operation's own scratch field. The fused sqrt terminal can use the same maximum-value scheme, with a half-size old-root bank and h retained across its products.

Sequential product scratch is a **maximum**, not a sum. The implemented reusable
program retains each rung's immutable preparation and shares one maximum work
buffer. The sum of preparation sizes across the geometric ladder remains O(p)
under the backend's linear-space table contract. Preparation occurs before
the first numerical execution, and repeated execution performs no shape queries
or table reconstruction. A future memory-clipped plan may regenerate tables at
declared joined safe points, but must price that choice explicitly. Source
inputs remain immutable; private value banks and caller output are separate
until the final emitter. No caller-input alias is implied merely by slot reuse.

Use one existing Team. A root invoked by the controlling owner can follow the current Newton service's joined product episodes. A scope-based internal form should execute precompiled `ProductProgram`s on the caller's scope, never invoke a second top-level `team_run` from a worker callback. Preparation and scratch subdivision must remain inside preadmitted caller Frames; public bind/idle rules still apply. Products along one root are dependent; initially give their kernels the admitted scope rather than launching competing full teams. Batch roots or an AGM mean/correction may use disjoint subscopes only with separate quoted workspaces. No new worker threads are part of root execution.

Full products at geometrically increasing n have total cost `sum M(n_j)=O(M(N))` under the normal multiplication regularity assumption. Reading/copying only geometrically growing prefixes costs O(N), aside from initial input normalization. Holding a constant number of value banks costs O(N), plus the selected multiplier's workspace. Product planning/preparation, root checks and disposal are part of the complete fresh-call cost; the arithmetic asymptotic is not an acceptance result.

## 7. Minimal foundation interface needed

The root algorithm does not need exp/log, general division, a symbolic recurrence engine or a public ball type. It needs these small private capabilities, with names left provisional:

1. **Exact finite dyadic view and checked scale:** immutable limbs, sign, logical nonzero length/bit length and binary exponent; normalize to `[1,4)` without allocating the exponent gap.
2. **Prefix extraction with an absolute tail bound:** obtain up to q fractional limbs at a selected binary scale, inserting only bounded virtual zeros; return `<B^-q` when clipped.
3. **Planned exact MUL/SQR:** caller outputs, full backend capacity/work quote and existing Team/scope execution. Add bounded signed windows only through explicit error contracts.
4. **Fixed-point signed add/subtract and dyadic shifts:** align already bounded grids, form `1-product` without losing cancellation, and truncate a signed correction toward zero.
5. **Result emitter and absolute-error metadata:** normalize/round to requested significant bits, add the known rounding error and scale the root bound safely. Distinguish resource failure, numeric domain status and a violated internal proof invariant.

A consumer that carries input uncertainty `delta` can propagate it separately using a positive lower bound `a_min`: sqrt adds at most `delta/(2sqrt(a_min))`, while rsqrt adds at most `delta/(2a_min^(3/2))`. These follow from the mean-value theorem over a positive input enclosure. If that enclosure touches zero, the derivative bound is invalid and a different enclosure/domain policy is needed; increasing precision of the same uncertain point does not fix it.

## 8. Implementation and validation sequence

First implement the scaled prefix and fixed-point product/truncation adapters, then the general seed, full-product rsqrt ladder and simple sqrt finish. Add the three-product half-rsqrt sqrt terminal once those identities and bounds are exercised. Only afterward lower residual/correction windows and compatible common spectra; keep the full-product route as a correctness oracle/fallback with the same published error guarantee. Preserve existing u64-radicand fast paths for constants.

The initial root source now implements the certified seed, full-product rsqrt
ladder and three-product sqrt terminal. One same-size exact multiplication
descriptor per rung is compiled and prepared in caller scratch before the seed
and reused for that rung's three products; square operands currently use this
same MUL descriptor. A bounded 6-word leaf uses the basecase kernel. Four value
banks plus a raw-product bank, retained rung tables and one maximum workspace
replace the conservative two-approximation-bank sketch above. The current query
includes RootState, all compiled descriptors/capacities and alignment padding.
The complete demanded input prefix is staged first, so an exact-base
output/input alias is safe. `root_program_prepare` / `root_program_execute`
separate this preparation from repeated arithmetic; the public fresh entry
composes them. Shorter operand descriptors, SQR specialization and residual
windows remain later constant-factor improvements, not changes to the O(M(p))
ladder or its error contract.

Correctness coverage must include odd negative exponents, `[1,2)`/`[2,4)` normalization, exact one and powers of four, squares and adjacent dyadics, all-ones prefixes, source precision both below and far above output precision, non-power-of-two limb ladders, non-limb-aligned output precision, and signed residuals near zero. Compare against an independent MPFR/high-precision oracle where available and verify exact rational inequalities for returned root bounds; the reference library remains test-only. Check input preservation, complete queried storage and no allocations/plan searches during arithmetic. Any compilation/timing must use the pinned toolchain/manifests and serialized `build/benchmark.lock`; no tests or builds were run to create this document.
