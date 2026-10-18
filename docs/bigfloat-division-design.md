# Bounded bigfloat division using the existing Newton recipes

Original design inspection, 2026-10-07; implementation-status update, 2026-10-08. The normalized service now reuses the optimized reciprocal ladder and half-precision quotient terminal at every admitted size, including calls made by an active Team sub-scope. It does not replace them with a full-width inverse-plus-multiply route.

## Implemented status

The private contract is [normalized_newton.hpp](../src/algorithms/normalized_newton.hpp), implemented in [newton_service.cpp](../src/algorithms/newton_service.cpp). Its local provider is [normalized_local.cpp](../src/algorithms/normalized_local.cpp); the float adapter is [division.cpp](../src/float/division.cpp). Preparation freezes all phase bundles, the complete local seed program, and low-lift witness plans before arithmetic. Native replay uses [ScopedResidentSession](../src/runtime/resident.hpp), scoped product ownership and the existing refinement kernels; it neither repeats geometry selection nor creates a nested Team epoch. The public C Newton bindings retain their idle-Team contracts.

The historical diagnosis below records why that extraction needed more than an outer scope parameter. Those diagnosed gaps are not current limitations of the normalized interface. Maintained gates are [normalized_local_test.cpp](../tests/integration/normalized_local_test.cpp), [normalized_newton_test.cpp](../tests/integration/normalized_newton_test.cpp), [float_division_test.cpp](../tests/integration/float_division_test.cpp), [float_short_division_test.cpp](../tests/integration/float_short_division_test.cpp), and [float_program_replay_test.cpp](../tests/integration/float_program_replay_test.cpp). They cover exact endpoint certificates, prepared reuse, nonzero-first-worker and concurrent disjoint scopes, allocation watching, and explicitly enumerated query hooks. Hook coverage is not a universal interposition proof. No tuning or performance result is claimed here.

## Numerical wrapper

Write positive input magnitudes as `x=a*2^ex`, `y=b*2^ey`, with `a,b in [1/2,1)`. For `n=ceil(p/64)+1`, stage the *floors* `A=floor(a B^n)`, `D=floor(b B^n)`, `B=2^64`. Both have n normalized limbs; the quotient service's numerator has an extra zero limb, `A[n]=0`. Its result Q satisfies

\[
|Q/B^n-(A/D)|<3B^{-n}.
\]

Put `u=B^-n`, `a_t=A/B^n`, `b_t=D/B^n`, with nonnegative prefix losses `da,db<u`. Then

\[
\frac a b-\frac{a_t}{b_t}=\frac{da}{b}-\frac{a_t\,db}{b b_t}.
\]

The two terms on the right are nonnegative before subtraction and bounded by `2u` and `4u`; their difference has magnitude below `4u`. Adding the kernel error gives `<7u`, conservatively bounded by `8u`. Reciprocal prefix loss is also below `4u`, so the existing inverse plus prefix truncation has the same `<8u` bound. This proof relies on floor prefixes; arbitrary nearest-rounded prefixes do not have the same signed-error argument.

Restore quotient scale `2^(ex-ey)` or reciprocal scale `2^-ey`, and give `round_magnitude` the staged integer result. Combine its local error with the scaled `8B^-n` bound. One full guard limb suffices: since a normalized quotient is at least 1/2, the approximation contributes less than `2^-60` result ulps. Nearest mantissa rounding therefore fits a one-result-ulp reported bound, and the other modes fit two. This does not promise directed mathematical division or correct rounding. The frontend owns signs, zero, infinities, NaNs and checked exponent arithmetic. Stage both prefixes before touching an aliased caller output.

An exact one-word divisor has a separate O(p) route. Normalize its word once,
reuse the existing word reciprocal for every quotient digit, and feed virtual
zero padding directly. Do not construct an n-word inverse for a fixed-height
word. The fresh <=256-bit path uses stack output and no planner/workspace.
Two-word divisors now use a streaming 3-by-2 reciprocal route. Longer short
divisors use the compiled [short-division program](../src/float/short_division.hpp)
when `3<=k<n/2`, with exact bounded quotient blocks and one reusable reciprocal.
The balanced normalized route remains available for the other shapes. This
preserves short-divisor structure without padding every denominator into a
full-width reciprocal problem.

## What is already reusable

| Existing component | Preserve |
| --- | --- |
| `algorithms/reciprocal.hpp` | The in-place precision ladder, caller-owned approximation and half-precision quotient construction |
| `algorithms/refinement.hpp` | The same `bounded_refinement<Inverse/Quotient>` equations and `<3` output-unit proof, including two guard words and signed residual error |
| `algorithms/local_inverse.cpp` | Tiny word/u52 quotient choices, local reciprocal prefix, fused quotient terminal |
| `product/local_program.hpp` | Packed preselected local/middle/shared/peeled FFT records and allocation-free replay |
| `product/compact_windows.hpp` | Smaller FFT ring, exact low-word witness, retained/rebuilt cache policy |
| `product/window_group.hpp`, `algorithms/newton_planner_impl.hpp` | The existing arithmetic-family, worker, ring and complete-memory decisions, including smaller NTT rings and fresh/cached capabilities |
| `InverseRung`, `DivideTerminal`, `CyclicWindowProducts` | Native cached residual/correction execution, live short operands, consuming residuals, low-lift witnesses and fused cache build/apply |
| `newton_service.cpp` | Current local/general split and exact phase pooling, including the union of seed scratch with later product scratch |

For division, continue to compute the inverse only at `m=floor(n/2)+1` and apply `quotient_terminal`. The standalone reciprocal uses the full ladder only because that full inverse is itself the requested result. The existing u64 rsqrt and unrelated constant service need no numerical change.

## Historical scope and replay diagnosis (2026-10-07)

The following observations describe the pre-extraction implementation. The
normalized/scoped interface linked above closes these gaps while preserving
the old public entry points and their checks.

The Frame-based local arithmetic can run on a valid scope leader without starting a Team episode, but the complete existing service cannot yet be called that way:

- `ResidentSession` requires an idle Team owned by the creating thread, and its destructor requires that Team to be idle. An active worker/sub-scope fails these conditions.
- Public Newton bind/execute/unbind and both FFT/NTT product bind/unbind enforce the same idle-owner condition. Compact stages can rebind products between arithmetic episodes, so changing only the outer execute entry would not suffice.
- `CyclicWindowProducts` passes null scope to ordinary/live/fresh cached products. Ordinary cached execution already accepts a scope internally, but this adapter does not forward one.
- The fused `spectrum_compute_multiply` path and its backend callbacks have no scope parameter. Both FFT and NTT implementations start `sbn3_team_run`; spectrum begin also enforces idle ownership. Cache state must still progress reserved → computing → ready around the joined episode.
- The low-witness `with_product_scratch` callback in the native backend enforces idle ownership. Its exclusivity is about the binding's dead scratch, not necessarily an idle whole Team; that distinction needs an explicit scoped entry.
- `parallel_limbs` has some scope helpers (`copy_scope`, `add_to_scope`, `each_scope`) but most calls used by refinement/CYC still start a top-level Team episode. Passing the busy parent Team there would recurse illegally, even if products themselves used the right scope.

There are also remaining compile/replay gaps. `LocalWindowProgram` has capacity 10 and captures only groups above its FFT eligibility threshold. Replay delegates smaller groups to `LocalWindowFactory`. Some of those are trivial leaf layout calculations; others can call `local_windows_query`. `local_refinement_compile` is currently restricted to the local inverse ceiling. Compact FFT's `with_group` calls its query, and a general `InverseProgram`'s large local seed uses `local_inverse_approximate` rather than a precompiled prefix replay. None of these should silently remain after a new interface promises that all product choices are fixed before the inverse seed starts.

## Extraction rationale retained from the design

The following design boundaries are now implemented by the normalized local
and native providers; the imperative wording records the original rationale.

Keep the mathematical programs and selected geometries unchanged. Add a **private normalized Newton program entry**, with three boundaries:

1. **Compile:** normalized operation (`inverse` or `quotient`), n and maximum workers → caller-owned metadata and complete value/preparation/work quote. Reuse the current query policy rather than copy its thresholds into the float wrapper.
2. **Prepare:** compiled metadata plus an exclusive resident region and an execution owner → concrete selected phase resources. This may occur on the controlling thread or on a validated active scope leader, using the same already-admitted bytes.
3. **Execute/close:** normalized immutable views and caller-owned result; no algorithm query after the seed begins; close only after every child product and scratch loan has joined.

An execution owner is either the existing idle Team owner or a borrowed active scope. It is not a cloned Team. The private scope path verifies `require_scope_leader`, actual Team identity, scope width, disjoint value/work spans and the binding's inactive state. Existing public C entry points retain their idle-owner checks and call the shared implementation with no borrowed scope.

### A. Complete the existing local replay first

Use a caller-sized sequence of tagged compiled group records: leaf, ordinary/middle/shared/peeled local, compact FFT, or native CYC. This is a finite arithmetic program, not a general interpreter. Existing `LocalWindowStep` already carries most local cases; add the compact plan and the small leaf/layout record so execution never performs selection. Keep the existing inlined small kernels.

Compile the entire local prefix, including a large seed used by a native ladder, before any call to `inverse_seed`. The current fixed-capacity public local record can remain for its current contract; the private program should size records from the actual halving ladder and optional quotient terminal. A W1 local group can execute serially on the active scope's leader, retaining all its current arithmetic optimizations; giving that group extra workers is not automatically an improvement.

### B. Extend native bindings with explicit scoped ownership

For the whole range, the least disruptive route is an explicit private scoped variant of the existing borrowed-resident/binding path. It reuses `WindowGroupPlans`, `Assembly` and the current compact lifetime controller instead of implementing another cached FFT/NTT provider from scratch. Required narrow extensions are:

- a private borrowed-resident context that may live under an active scope and checks only its own joined resources on destruction;
- scope-aware private bind/close helpers for selected FFT/NTT product bindings; keep public wrappers' idle checks unchanged;
- optional scope forwarding for cache begin/compute/build-apply and dead-product-scratch loans;
- a scope field in the private rung/terminal/group execution context, plus complete scope-aware value passes in refinement and cyclic reconstruction.

Use the same backend actions after validation: the NTT `prepare_action`/`product_action` and FFT product action already operate on scopes internally. Replace the top-level `team_run` only when an explicit borrowed scope exists; otherwise preserve today's execution. Live/fresh product callbacks already take scopes. Retain all existing spectrum representation checks, scratch exclusion, atomic binding ownership and ready publication.

The borrowed-resident adapter must also handle `serial_team_run`, whose temporary one-worker Team has a null arena. Do not globally change `Arena::accepts_team(nullptr)` or mutate that Team's arena pointer. Validate the actual borrowed Team/scope identity in the new private binding context. Likewise, do not clear `team->busy`, forge creator identity, destroy a context while its bindings live, or call public idle-only APIs from a scope while hoping their guards are absent in release builds.

This touches several execution boundaries, but leaves the numerical kernels and optimized group policy intact. A wholly Frame-native persistent-spectrum API could eventually remove the lease adapter; doing that first would be a larger backend extraction and is not necessary to deliver bounded float division. The plain `ProductProgram` interface alone is insufficient to preserve all current cached CYC behavior: the standalone frame-owned `CyclicProgramPlan` is uncached, while pair-continuation APIs have a different control structure from the existing sequential refinement provider.

### C. Preserve phase reuse and compile everything before arithmetic

The existing public bind already materializes all general bundles during `Assembly` replay. Reuse that preparation order in the private program. Stage transitions may construct/destroy concrete objects in the quoted pool; they may not invoke shape selection. Precompile local-prefix, compact-FFT and low-witness choices as well, so there is no hidden exception for the first or last rung.

Pass the admitted **scope width**, not the parent Team's total width, into the existing planner. Keep worker-local scratch indexed relative to the supplied scope. Under concurrent subteams each normalized Newton call owns an entirely separate region; only genuinely immutable published tables may be shared under their existing contracts. Low-witness loans, retained reciprocal spectra and consuming residuals keep their exact current lifetime rules.

## Current float-facing private interface

- `normalized_newton_query(kind, n, workers, info)` returns support, the actual worker grant, and separate metadata/workspace/full-storage byte bounds.
- `normalized_newton_prepare(kind, n, workers, metadata_frame)` freezes the existing method/geometry choices in immutable caller-owned metadata, including local-prefix and low-witness records.
- `normalized_newton_execute(prepared, A_or_empty, D, Q, work_frame, scope)` binds, runs and closes its transient resources within the supplied active scope. There is no separately escaping binding or close call.

Metadata may be reused with independent exclusive work Frames. The normalized
input/output shapes, alias rules and `<3` output-unit error contract are stated
in [the service header](../src/algorithms/normalized_newton.hpp).

The float wrapper stages A/D, restores signs/scales, rounds and composes the explicit bound. The normalized layer does not need float classes, exp/log or a generic ball type. A convenience `*_into` can compile/prepare/execute/close inside its one prequoted scratch span. No public prepared-object ABI is needed before real consumers exercise the lifetime.

## Verification gates for the extraction

Compare the selected local/native/compact recipe identities and normalized outputs with the existing public service across non-power-of-two n, including every crossover and the largest local-prefix cases. Check exact inequality certificates for reciprocal/quotient and the separate float prefix bound. Run an active **nonzero-first-worker** child scope as well as root W1/W4 calls; a root-scope-only test will miss worker-offset and creator assumptions. Run two simultaneous disjoint child calls to expose accidental shared ownership or parent dispatch.

Watch allocations and instrument product/window query calls to assert that none occur after inverse seed entry. Allocation watching alone cannot detect a hot geometry search. Check quoted high-water storage, cache ready state, input preservation, exact-base float aliases after prefix staging, unbind/destructor behavior while the parent Team remains busy, and both low-witness/cached-residual variants. Existing idle C callers must retain their contracts and pass unchanged tests. All compilation/timing remains serialized through the repository lock; this design inspection performed neither.
