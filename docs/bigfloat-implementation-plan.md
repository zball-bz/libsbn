# Bigfloat and elementary-function implementation plan

Started 2026-10-07; implementation/evidence update 2026-10-08. Scope: the
bounded-error real-arithmetic/root foundation and the elementary-function/real-AGM
goals from the special-function research. Algorithmic scaling, numerical
contracts and complete storage precede tuning. The evidence below distinguishes
implemented numerical capability from later performance acceptance.

## 1. Completion requirements

| Goal | Required end state | Evidence required |
|---|---|---|
| G1 values | Owning C++ bigfloat, immutable exact-dyadic views, caller-owned result views, bit precision independent of capacity, signed zero/infinity/NaN, checked wide exponents | Maintained tests for copy/move/resources, aliases, exponent/bit boundaries and specials; no ISA types in public headers |
| G1 accuracy | Local absolute error independent of mantissa exponent; explicit numerical flags and rounding of stored points; no implicit claim of correctly rounded transcendental results | Exact dyadic/rational inequalities, independent references, analytic per-kernel bounds |
| G1 arithmetic | Add/subtract, multiply/square, division/reciprocal and fused multiply-add; power-of-two scaling and integer conversion | Mixed input/output precisions and signs, severe cancellation, virtual exponent gaps, allocation-free `_into` gates |
| G1 roots | General multiprecision sqrt/rsqrt and integer powers/roots, with valid seeds and geometric precision ladders | Perfect powers and adjacent values, root residual bounds, exponent parity, non-power-of-two precision and shape/storage tests |
| G2 elementary | exp/expm1, log/log1p, exp2/log2/log10; paired sincos, sinpi/cospi, inverse trig and hyperbolic families with cancellation-safe formulas | Certified reduction, tails and arithmetic budgets; zeros/poles/domain edges and large reductions; independent comparisons |
| G2 AGM | General nonnegative-real AGM and first complete elliptic K/E consumers, complementary-parameter inputs and optional correction accumulation | All requested output tails certified independently; O(p) rolling state; zero/equal/extreme-ratio inputs |
| G2 scaling | Fixed-point small/medium evaluation and scalable full-height high-precision methods; no full-height exact Taylor tree or quadratic universal fallback | Per-recipe operation/operand-height/space derivations; tests of query schedules and actual non-power-of-two executions |
| Architecture | Constants have independent acyclic producers; complete plans precede large arithmetic; one Team grant and caller-owned scratch; external support explicit | Plan/binding and allocation tests; cold preparation recorded separately from reuse; docs/examples match actual API |

Gamma, arbitrary-complex public APIs, full special-function coverage, correctly
rounded transcendental wrappers, shortest formatting and new OOC implementations
remain later projects. Private complex arithmetic needed for an asymptotically
fast real elementary recipe is not excluded by deferring the public complex API.
The existing general radix service is reused later through a float policy layer;
this goal does not replace or retune its conversion kernels.

“Algorithmic optimality” here means using the best applicable established
algorithm families and preserving their input-dependent complexity. A universal
matching lower bound for arbitrary special functions is not known. In particular,
O(M(p) log² p) bit-burst is a useful independent scalable route, but cannot be
advertised as the same asymptotic result as O(M(p) log p) AGM-based elementary
evaluation. Real AGM alone does not supply the optimal high-precision trig route.
No performance-completion or speedup claim is made without the existing full
fresh-call, non-power-of-two and smoothness protocol.

### Progress after the integrated correctness pass (2026-10-08)

The G1 implementations now include value ownership/views, rounding/error
reports, arithmetic and exact-product FMA, prepared division/reciprocal, general
sqrt/rsqrt and integer powers/roots, and integer conversion. Short divisors use
word/two-word streaming or the compiled short-division program. Existing Team
subscopes are supported. The native division suite currently covers 2514
independent endpoint certificates, including prepared and native large cases.

The declared G2 surface now has real AGM, paired K/E, log/log1p, exp/expm1,
exp2/log2/log10, general and pi-scaled paired/unary sine/cosine, inverse trig,
the six hyperbolic functions, and real powers. Fixed-point blocked exp/sincos
cover small and medium reduced arguments; the general high-precision recipes
use real/complex AGM and geometric-precision inversion.

The combined native pass passed the C++ owner, division, general trig, reduced
trig/exp series, exponential, inverse-trig, derived-exp/log and real-power
suites. The hyperbolic suite exposed a test-oracle precision mismatch for a
tightly bounded saturated result; its analytic reference was corrected without
weakening the implementation bound. All 25 combined AddressSanitizer suites
then passed, including roots, AGM, complex arithmetic, K/E, conversion and the
calibrated prepared-query replay. These are correctness gates, not performance
acceptance.

The subsequent shared-constant stage passed 15 native integration gates and
21 AddressSanitizer/checked gates, including existing series/formula regressions.
Its immutable pi/log2 values are independent of construction scratch; an
explicit C++ owner or context can lend a bank to the elementary consumers.
Internal compositions merge demands before preparing child programs, and
standalone AGM uses one outer Team epoch. Tests cover one-bit constants,
independent prefix bounds, destroyed producer work, growth/move/reuse, large
trig phase and nonleaf high-precision trig with borrowed constants.
The receipts are `results/bigfloat_constant_bank_2026-10-08/` in the research
checkout. These results establish correctness and lifetime behavior, not a
measured speedup.

The wide-leaf bit-burst implementation subsequently passed 856 native and 866
sanitizer interval certificates. Its 36 prepared replays include 24 executions
with actual leaves wider than 64 limbs. Extreme analytic leaves, mixed output
precision, aliases, caller-scratch prefixes and nonzero-worker scopes are
covered. Frozen metadata and bounded table/phase storage are shared with the
existing finite-series executor. This provides the separate B5 algorithm;
the public default's high-precision AGM route remains unchanged until crossover
tuning is authorized.

Fresh logarithm specials now bypass AGM planning. The focused checked gate
adds 75 special cases with zero scratch/allocations and calibrated query
observation. The baseline-target C++ example has built and run successfully.

Closure qualifications:

1. Replay observes seven calibrated query entry points, including nonleaf
   integer power/root cases. Static source review complements the explicitly
   limited interception coverage; the gate is not a universal query interposer.
2. B4's original exploratory list included log-prime and angle catalogs for
   alternate reduction methods. The selected complete G1/G2 recipes require
   pi/log2, so those are the implemented bank entries. Multiprime/angle-table
   reducers and their catalogs are deferred portfolio/tuning work, not missing
   dependencies of the delivered functions or implicitly implemented features.
3. Bit-burst is available as a private prepared/point evaluator. Selecting its
   practical crossover against AGM is tuning; the asymptotically faster public
   high-precision route is already present. Neither route is labelled fastest
   without fresh-call, non-power-of-two measurements.
4. Gamma, public complex/OOC float APIs, strict rounding and float formatting
   remain outside this G1/G2 scope, as specified in the requirements above.

### Verified closeout

The final unchanged-source build passed **38/38 AddressSanitizer gates** across
all float suites and their directly affected normalized-product, context and
series providers. Build receipts and a dependency/object/binary hash check bind
every final executable to the same current ASAN archive. The final no-GMP audit
passed for 613 TU records and 41 binaries; the native boundary audit passed for
168 TUs and 173 public C symbols. C++ API coverage is exercised by baseline-target
callers, including the built/runnable example; it is not inferred from the C
symbol count.

`check_plan.py` also passes after updating its provenance validator to distinguish
the historical migration snapshot, explicitly hashed adapted donors and pinned
vendored files. Original reference paths/hashes and license metadata were not
rewritten. Malformed hash coverage, escaping vendor paths and missing historical
snapshot coverage are rejected. Original-donor byte verification remains the
separate optional `--verify-donors` mode.

G1/G2's resident implementation, numerical-contract and algorithmic-scaling
requirements are therefore complete. Tuning, fresh-call performance acceptance
and the deferred capabilities above remain separate work. Final evidence is in
`results/bigfloat_completion_2026-10-08/` in the optional research checkout;
the library and maintained tests do not depend on that checkout.

The experiments checkout retains the raw test output and build identities in
`results/bigfloat_expansion_2026-10-07/`. No timing thresholds or performance
completion claims are inferred from these correctness runs. The newer research
dossier refines the later tuning candidates without replacing these exit gates.

## 2. Implementation order and dependencies

| Stage | Concrete work | Depends on | Exit evidence / current state |
|---|---|---|---|
| B0 | Value, precision, local error and exponent contracts; direct round/add/sub/compare | Existing integer storage/resource | Implemented; exact integer witnesses and native/sanitizer gates |
| B1 | Product/square adapter and bounded ratio; plan records for every large arithmetic shape | B0, existing multiplication/Newton | Implemented; exact endpoint/FMA witnesses, complete quoted storage and replay gates |
| B2 | General roots with normalized mantissa/parity and precision ladder; powers/rootn | B1 | Implemented; recurrence contracts and independent mixed-precision/root residual gates |
| B3 | Positive-real AGM, output-specific stop/closure; K/E and complementary input | B2 | Implemented; limit/moment bounds, complementary inputs and one-epoch Team gate |
| B4 | Independent constants required by selected recipes (pi/log2), shared context prefix bank | B1–B3 | Implemented; acyclic bootstrap/TU graph, value/work separation and borrowing gates; catalog qualification above |
| B5 | Fixed-point reduced series, denominator batching, wide-leaf exact bit-burst trees and paired reconstruction | B1, B4 | Implemented; actual >64-limb leaves, O(p)-height slices, bounded phase and independent certificates |
| B6 | Complete exp/log and cancellation-safe forms; range reduction and Newton inversions | B2–B5 | Implemented; fixed inversion dependencies, shared constants and fresh special dispatch |
| B7 | sincos/sinpi/cospi and full trig reduction; high-precision complex AGM | B2–B6 | Implemented; exponent-aware exact phase windows, independent large-phase/borrowed-bank gates |
| B8 | Stable inverse/hyperbolic/power compositions and public conveniences | B6–B7 | Implemented; endpoint/cancellation/range certificates, owner/controlled APIs and runnable example |
| B9 | Requirement-by-requirement completion audit | All | Complete: independent source reviews, 38/38 current-snapshot sanitizer gates, native boundaries, no-GMP, metadata and example checks |

Stages are dependency milestones, not human-day estimates. Independent design
and review can proceed concurrently; only one owner edits each source and only
one build/test activity holds `build/benchmark.lock`. Tuning is a separate later
work item. Each vertical slice adds only the public declarations it implements.

## 3. Public numerical model

`include/sbn3/float.hpp` starts the implemented surface. `bits` denotes requested
significant bits. A finite `float_view` is an exact sign-magnitude dyadic
`(-1)^negative * magnitude * 2^exponent2`; the precision at which an input was
previously computed does not make its stored point uncertain. `float_buffer`
borrows value storage; `bigfloat` reuses integer ownership and `memory_resource`.
Increasing precision preserves the existing value, not discarded information.

An operation returns a `float_report`. A finite `error_bound` means absolute
error at most a power of two, independent of output normalization. Exact and
unbounded are explicit tags. The bound is local to the supplied exact points;
earlier reports are composed deliberately by an algorithm, not silently attached
to every number. A nonzero conservative error bound does not itself prove the
exact mathematical result was inexact. `rounded` reports observed discarded
mantissa information where established.

Internal multi-stage evaluation needs an error accumulator with a significand
or a preallocated analytic budget. Repeatedly rounding the sum of one large and
many tiny errors to a power of two could increase the bound exponentially and
is prohibited as the generic propagation algorithm.

Stored-point rounding supports nearest-even, toward-zero, upward, downward and
away-from-zero. Normal-range round/add/sub can resolve their exact dyadic result
without expensive numerical refinement. This does not impose correct rounding
on later approximate division/root/function evaluators.

The normalized top exponent of finite nonzero inputs/results is between
`-2^60` and `2^60`; admitted precision is 1 through `2^40` bits, subject to the
actual backend/resource support. Metadata arithmetic uses checked wider
intermediates. Overflow returns signed infinity with an unbounded error and an
overflow flag. Underflow flushes to signed zero with an absolute bound and an
underflow flag; it is not a directed subnormal result. Specials do not carry
finite real-valued error claims. Numerical domain results are distinct from
invalid descriptors or a violated arithmetic invariant.

## 4. Private algorithm boundaries

```
typed mathematical request + exact input facts + accuracy goal
  -> reduction/recipe + constant dependencies
  -> precision ladder + concrete product/root shapes + storage lifetimes
  -> bind caller outputs, scratch, shared constants and Team
  -> allocation-free typed kernel execution
  -> point result + error bound + numerical flags
```

Small arithmetic uses direct kernels and bounded stack scratch. Large operations
compile a finite schedule before execution. Product algorithms and transform
formats stay behind the multiplication interface. No polynomial/twiddle/prime
selection details enter a division, root or elementary public API.

The legacy `ProgramMultiply::operator()` queries products when called; admitted
float/root programs instead use precompiled product records. One joined Team
scope is borrowed. Persistent prepared child spans are honestly charged as a
sum unless an interface explicitly separates their mutually exclusive work.
The bit-burst Frame adapter does provide that separation and uses a maximum
phase plus retained values, metadata and tables. `_into` entry points never
grow the arena or result.

The legacy finite-series frontend retains its 64-limb leaf contract. The
private Frame-backed adapter now admits parameterized wide leaves, separate
immutable bounds/runtime leaf contexts, frozen stage products, and execution
on an existing scope. Bit-burst uses this adapter with one term per genuinely
wide leaf, so scalar batch multiplication cannot hide quadratic work there.
It keeps exact finite trees with explicit exponent metadata; it does not reuse
PSR's contraction proof for dyadic leaves with different magnitude assumptions.

## 5. First slice: exact alignment with bounded storage

The round/add/sub implementation represents each input as a word stream at its
absolute binary-word position. Addition/subtraction visits actual operand spans,
skips zero gaps, and represents a borrow across an empty gap as a run of ones.
A rolling high window keeps `ceil(p/64)+2` words plus a discarded-tail flag.
It therefore uses O(p) value/scratch and O(input limbs + p) work independent of
the exponent gap; cancellation cannot erase an earlier retained nonzero result
because the window advances only on nonzero output words.

Tests must distinguish this exact alignment from merely discarding a small
operand: `1 - 2^-huge` under downward rounding differs from one. Same-base
aliases are supported by completing the read/stream phase before output writes;
partial overlaps and scratch aliases are rejected. The stack path handles up
to 256 result bits without planning or arena calls.

### AGM must not overflow an avoidable intermediate

For future general AGM, do not compute the public point product `a*b` and then
take its root: that product can overflow/underflow while the geometric mean
lies between the valid inputs. Write `a=ma*2^ea`, `b=mb*2^eb`, with normalized
mantissas in [1/2,1), and split `ea+eb=2h+r`, r in {0,1}, using floor division.
Compute `sqrt(ma*mb*2^r)*2^h` from bounded mantissas, preserving the exponent
separately until output. This changes neither the quadratic AGM nor its O(M(p))
per-step cost. Product/root local errors and the scale enter the shared budget.

Similarly, form the arithmetic mean from half-scaled inputs in the internal
finite-sum path; an overflowing `a+b` is not a legitimate reason for their
mean to overflow. Private scaled intermediates may use the wider checked
exponent domain already required by FMA. Public range handling occurs at
explicit output boundaries. Full common-state precision, ratio transient,
limit versus K/E-tail stopping, and compiled root reuse remain required.

## 6. Validation and evidence

Maintained tests use the independent integer oracle in `tests/oracle`, exact
dyadic/rational witnesses and analytic residual inequalities. External MPFR and
FLINT comparisons may be separate research verifiers; production code and
maintained tests retain the no-GMP dependency boundary. Numerical tests cover
non-limb and non-power-of-two precisions, different input heights, signs and
scales, severe cancellation, range/domain boundaries and aliases. Allocation
watching covers admitted execution, not just the innermost multiply.

Builds use pinned Clang and explicit manifests. There are no tuning sweeps or
SSD tests in this implementation phase. Memory/schedule tests should demonstrate
the claimed growth before performance work, and reports must distinguish
analytic complexity, observed correctness and measured performance.
