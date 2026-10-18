# Bigfloat completion roadmap

Proposal, 2026-10-05, based on the public headers and resident/external
implementations at that date. This document describes the broader roadmap.
Implementation began on 2026-10-07 under the
[bounded-error arithmetic and elementary plan](bigfloat-implementation-plan.md).
Its completion matrix, actual public declarations and test receipts distinguish
delivered slices from the remaining proposal below.
The [OOC cleanup proposal](ooc-architecture-cleanup-2026-10-05.md) supplies the
shared value, resource, arithmetic-program and recovery boundaries.

Updated direction, 2026-10-07: **bounded-error arithmetic first**. Correct
rounding is not a prerequisite for the first bigfloat release. Specify useful
accuracy bounds and retain rounding/error metadata now; add distinctly named
strict capabilities later where valuable. Rounding an approximate mantissa
to nearest does not imply correct rounding of the mathematical operation.
Do not change an advertised result contract according to size or backend.
The literature and evaluator design are documented in the private research
checkout's `experiments/docs/special-function-architecture-2026-10-07.md`;
production code has no dependency on that checkout.

## 1. Actual starting point

| Area | Available now | Missing for a general bigfloat |
|---|---|---|
| Values | `sbn3_int`, C++ integer ownership/views/resources; `sbn3_series_value = mantissa * 2^exponent2`; private stored/resident dyadics | Public floating precision, class/sign semantics, rounding, flags and C++ owners/views |
| Add/subtract | Integer kernels, limited series arithmetic and stored scaled linear operations | Bit-precision floating alignment, cancellation, sticky information and rounding in all directions |
| Multiply/square | Fast exact integer products, TMP/CYC, cached spectra, MAC2 and external products | Floating result normalization/rounding, certified short-result selection and true floating FMA semantics |
| Reciprocal/division | Normalized Newton services; exact integer divrem; stored positive quotient with an explicit error bound | General signed float frontend, arbitrary input/output precision combinations and rounding resolution |
| Roots | Accurate reciprocal square root for a **positive u64 radicand**, special exact sqrt(2) finishes, stored equivalents | General multiprecision-radicand sqrt/rsqrt; general nth roots |
| Radix | Resident base 2–64 dyadic formatting/parsing, once/reuse paths and fast small decimal kernels | Floating lexical/formatting policy and directed/nearest rounding; shortest roundtrip as a separate feature |
| Stored radix | Positive limb-aligned fractional enclosure, integer part bounded by u64 | General stored integer/float formatting and parsing; exact/directed float rounding |
| Constants / functions | Generic binary-splitting families, formula sums, small-integer log and rational ArcCoth formulas | General real-argument exp/log, trigonometric/hyperbolic functions and a public floating math layer |

Important numerical distinctions:

- `newton.h` promises errors below three output units for its approximate
  reciprocal/division/u64-rsqrt entries. Stored quotient describes a bound below
  seven output ulps after source-prefix truncation. Neither is automatically a
  correctly rounded or faithfully rounded result at that same precision.
- `radix.h` EXACT means exact truncation of an exact dyadic into the requested
  digits. Parsing returns a fixed binary floor. These are useful primitives,
  not arbitrary floating rounding policies.
- `FixedOutputPlan` certifies a nonnegative floor supplied with a known error
  budget, and treats ambiguous certification as fatal. Ordinary float halfway
  cases need a defined rounding decision; they must not use that fatal path as
  normal control flow.

Therefore this is more than putting a class around Newton, but the expensive
arithmetic kernels and much of the resource machinery can be reused.

## 2. Define completion before writing a large header

Use two release checkpoints:

**Arithmetic-complete bigfloat:** values/ownership, explicit bit precision,
classification/comparison, conversions, add/subtract, multiply/square, division/
reciprocal, general sqrt/rsqrt, integer powers/roots and FMA, plus their specified
rounding/status behavior and usable C++/text interfaces.

**Practical mathematical bigfloat:** the above plus general exp/log and their
small-argument variants, real powers, trigonometric and hyperbolic families,
constants, and reductions such as sum/dot/hypot with explicitly defined
rounding. This is not a promise to duplicate every MPFR special function.
Gamma/zeta/Bessel/elliptic families, complex numbers, a public ball-arithmetic
library and MPFR ABI compatibility are separate extensions.

External execution has an independent capability matrix. New resident float
functions may ship while their external lowering is unsupported. That honors
the current pause in OOC development without making their future OOC model an
afterthought.

## 3. Numerical and representation contract (F0)

The stored finite value is a dyadic point. Its relation to an intended real
answer is a separate operation/error contract. Increasing precision does not
recover information discarded by an earlier operation.

| Property | Proposed rule |
|---|---|
| Precision | Significant **bits**, per value/result; independent of allocation capacity and limb alignment |
| Finite storage | Sign plus bounded u64 magnitude view and binary exponent; share the existing dyadic law |
| Exponents | Documented wide bounded range; checked wider intermediates for exponent/bit-length arithmetic; leave headroom for the supported precision range |
| Normalization | Canonical logical value without mandatory movement of a large payload; trim/view metadata where legal, with materialization only at declared boundaries |
| Zero/specials | Signed zero, infinities and a quiet NaN class; no initial NaN-payload or signaling-NaN ABI promise |
| Rounding | Define mantissa rounding and the operation's error guarantee separately; strict nearest/directed modes are later optional capabilities, not promises made by an approximate result |
| Results | Numerical flags and rounding direction/inexactness where promised, distinct from resource and IO outcomes |
| Exceptions | Defined numerical exceptional results handled by the float frontend; invalid low-level preconditions, failed arithmetic invariants or certificates remain fatal |
| Precision changes | `reserve_bits` changes capacity only; `set_precision` rounds the current value; a view does not mutate either |
| Expressions | Preserve specified intermediate precision/rounding; FMA is explicit and rounds once |

Do not inherit the host floating environment as hidden numerical state. A
context may hold explicit defaults, but there is no process-global precision
that changes unrelated computations. Eager binary operators can use the maximum
operand precision and the documented bounded-error contract; `_into` operations
use an explicit result specification. Internal nearest-even mantissa rounding
does not strengthen that contract. Define and test these rules before adding
many convenience overloads.

Correct rounding and a bounded approximation answer different questions. MPFR's
[manual](https://www.mpfr.org/mpfr-current/mpfr.html#Rounding) is a useful
semantic reference for explicit precision and rounding. FLINT's
[ball representation](https://flintlib.org/doc/arb.html) is a useful contrast:
an enclosure is not a point value. We should share internal error-bound machinery
without implicitly turning every bigfloat into a public interval object.

Expected domain results such as a NaN from a real-domain violation are not
evidence of a broken Newton certificate. Handle them before entering existing
positive/normalized kernels. The existing low-level contracts remain intact.

## 4. Foundation that determines both speed and OOC feasibility

Build a small common rounding/observation layer before adding many functions:

- Checked exponent alignment and virtual zero ranges. `1 + 2^-1000000000000`
  at modest precision must not allocate the exponent gap.
- Leading/cancellation analysis, magnitude windows, guard/round/sticky bits,
  carry/borrow and sign-aware final rounding, including halfway cases.
- Exact facts about discarded ranges. `WordExtent` is useful but is not a
  general range-sticky index for every clipped/complemented view. Add typed
  summaries or explicitly quoted reads where required. A checksum or m61
  residue must not be treated as a proof that a discarded range is zero.
- A result emitter shared by resident and stored lowerings: receive finalized
  words, normalize, decide rounding, account for retained/discarded bits, then
  expose the value. Keep its observation fused with existing carry/emission
  passes where possible.
- Explicit `AccuracyRequest` / `ErrorBound` descriptions, e.g. an absolute
  bound expressed at a binary scale, instead of an ambiguous generic `ulp`
  count detached from normalization and requested precision.

There are real worst cases. Severe cancellation can require examining many
input words. Rounding may require a discarded-tail witness. Large trig arguments
may require reduction precision related to the argument exponent as well as
the requested output precision. Quote that work honestly instead of reserving
only `O(p)` unconditionally or creating hidden normalization files.

For strict basic operations, use the existing fast approximation plus an exact
residual/rounding decision where possible. All fallback storage and IO belong
in the query; distinguish ordinary from worst-case traffic. General correctly
rounded transcendental evaluation can require precision refinement. Refinement
is a mathematical algorithm at safe points, not a runtime backend search or a
failed kernel invariant. Its preparation/reserve boundary must be explicit;
do not promise universal strict rounding with an unexplained fixed guard count
and a small fixed scratch cap. MPFR's [algorithm references](https://www.mpfr.org/algo.html)
are a useful starting point when implementing this layer.

## 5. Programming interface shape

Follow the existing integer API's levels. Names below are provisional, not
declarations to add wholesale before implementation.

1. Owning `bigfloat` plus real resident `float_view`, writable `float_ref`,
   caller-supplied `float_buffer`, and optionally `fixed_float<Bits>`.
2. Direct small kernels and `*_into` operations over admitted output/scratch,
   with an existing Team/scope. No generic graph, file handle or plan allocation
   for a one/two-limb operation.
3. Explicit reusable plans/contexts for larger operations and expressions.
4. A distinct immutable `stored_bigfloat` and explicit external preparation,
   fallible execution and materialization through the shared request layer.

Illustrative convenience usage:

```cpp
// Proposed spelling; typed precision avoids confusing a value with a bit count.
auto x = sbn::bigfloat::from(2, sbn::bits{256});
auto y = sbn::sqrt(x);
auto z = sbn::fma(x, y, x);       // explicitly one final rounding
```

The controlled interface must let a caller query the complete requirement,
reserve value/work storage separately and execute repeatedly without growth.
The external interface lowers the same mathematical request but returns stored
owners/IO outcomes, rather than disguising disk access behind `float_view`.
Moving between resident and stored values is explicit and budgeted.

Use templates for typed expressions, static precision and fused leaf policies;
keep implementation-heavy algorithms private. A coarse opaque arithmetic
program is enough for backend substitution. Retain Clang and the existing
explicit manifests; no ISA type appears in public headers.

## 6. Delivery sequence

These are dependency milestones, not calendar or person-day estimates.

| Milestone | Deliverable / reuse | Exit gate |
|---|---|---|
| F0: semantics and common metadata | Precision/class/rounding/status contract; dyadic and accuracy descriptors shared with cleanup C1 | A written behavior matrix, including zero/specials, aliases, exponent limits and exact vs approximate results |
| F1: usable value and rounder | Owners/views/resources; assignment/conversion; compare/classify; scaling, precision changes, add/subtract and basic binary serialization | Exact integer/rational reference plus MPFR checks over bit boundaries, cancellation, huge exponent gaps and all modes; small calls bypass planning |
| F2: multiplicative arithmetic | Float adapters for MUL/SQR, reciprocal/division and multiply-add with specified error; cache reuse and fused emit; strict residual/tie finishing later | Mixed input/output precision, signs and aliases; complete fresh performance and full scratch bounds; approximate multiply-add is not advertised as correctly rounded FMA |
| F3: general roots and powers | Multiprecision-radicand sqrt/rsqrt, integer powers and nth roots; reuse Newton/middle/CYC infrastructure | Perfect powers, near-rounding boundaries, exponent parity, negative/zero domains and extreme scales; share root finishing primitives with missing integer-root work |
| F4: complete conversion/interface | Base 2–64 floating parsing/formatting, fixed/scientific/significant-digit styles, precision-aware integer/native conversions, C++ conveniences | Declared roundtrip/error guarantees; directed modes only when actually supplied; reuse current SRT/parser/basecase implementations; minimal wrappers at small precision |
| Core release | F0–F4 complete for the declared resident domain | Arithmetic-complete scope above, documented capability/size limits and accuracy modes; no claim of full MPFR or OOC coverage |
| F5: elementary real math | General log/exp, log1p/expm1, exp2/log2/log10, real powers, trig/sincos/atan2 and hyperbolic families; sum/dot/hypot with explicit rounding | Range-reduction/error budgets, hard cancellation/pole cases, domain flags, high-precision references and complete-API measurements |
| F6: stored adapters and wider math | Connect approved float recipes to generic job/checkpoint APIs; later special-function/complex modules | Each operation declares external support; real non-constant jobs reuse infrastructure without new Store/allocator implementations |

F1 and F4's lexical/formatting work can overlap. F2 and the OOC arithmetic-adapter
extraction can share the same operation/result contracts. F5 can develop
explicitly bounded-error estimates before strict wrappers are ready, but a
strict entry point is not exported with silently weaker semantics. F6 is a
future extension, not a reason to restart OOC performance development now.

### F3 is substantial new arithmetic

The current rsqrt's radicand is one u64. Generalizing to a long mantissa requires
a normalized mantissa/exponent-parity treatment and a real input-product
residual, not merely widening the C argument type. Plan the doubling ladder,
reusable input spectra, demanded middle/cyclic windows and the final fused
multiply together. Preserve the current u64 specialization for constants.

### F4 reuses kernels, not their current floor contract

Binary-to-radix can keep the fractional conversion machinery with explicit
scaling; radix-to-binary can keep integer parsing followed by its scale/divide
recipe. Add a float rounding wrapper rather than retuning four independent
conversion engines. A fixed/scientific roundtrip format can ship before a
shortest-roundtrip implementation; the latter is an additional interval-boundary
problem, especially at arbitrary precision.

For arbitrary bases/alphabets, `e` or `p` can be an ordinary digit. Keep a
structured digit/sign/point/exponent API, and define an unambiguous textual
grammar separately. General parsing must not silently inherit decimal lexical
rules. Existing tiny decimal primitives remain the small-number foundation.

### F5 starts with reusable reduction and precision machinery

Small-integer log and rational ArcCoth are assets, not general real log. Build
argument reduction and working-precision rules first. Use explicit small-input
paths for `log1p`/`expm1`, share sin/cos reduction, and budget difficult large
arguments. Binary splitting and existing constant caches are reusable parts;
bit-burst/AGM alternatives need complete-operation experiments before selection.
Do not add a per-input parameter grid as an outer math-function planner.

## 7. OOC support is a matrix, not a property of the class name

| Operation family | First bigfloat delivery | Later stored delivery |
|---|---|---|
| Metadata/sign/scale/classification | Direct and cheap | Usually metadata-only; integrity/lifetime checks still apply |
| Compare/add/sub/round | Resident kernels with virtual gaps | Existing streaming linear/view machinery plus exact rounding observations |
| MUL/SQR/FMA | Existing products plus float adapter | Arithmetic provider capabilities and typed emit; no duplicate p48 planner |
| Division/root | General resident normalized recipes | Reuse the same mathematical ladder with external products; preserve appropriate certified fast paths |
| Parse/format | Existing resident conversion families plus float policy | Extend current restricted fractional formatter and add stored parsing explicitly |
| Elementary/special functions | Resident support first, explicit accuracy | Opt in per recipe only after storage, cancellation and recovery are accounted for |

An unsupported stored operation reports that at preparation. It does not load
the whole input into RAM as a hidden fallback. Checkpoints persist canonical
value metadata (including floating precision/class and any job-owned error
certificate) separately from backend-private spectra and execution choices.

## 8. Acceptance and implementation discipline

- Test precisions including 1/2/3/31/32/53/63/64/65/127/128/129 bits and non-limb
  multiples. Cover exact ties, adjacent values, carry-out renormalization,
  cancellation, exponent extremes, signed zero and exceptional classes.
- MPFR is a test/reference dependency only; use exact integer/rational
  witnesses where available. Keep core functionality independent of the
  research checkout and any reference library.
- Compare numerical contracts fairly. YC approximate division is not an
  interchangeable timing reference for correctly rounded float division; use
  a complete adapter or report the different contracts. Check rounded text and
  certified-prefix output separately.
- Use the established fresh-call timing and non-power-of-two scaling protocol.
  Count plan/table construction and required cleanup; reuse results are a
  separate capability. Small cases must have a thin direct wrapper, not an
  opaque-plan construction tax.
- Query complete value, table, scratch and IO storage. Preserve low-memory
  aliases only where proved; an uncertain rounding case cannot cause an
  unquoted full-value allocation or disk pass.
- Implement vertical slices: one value/rounding contract plus working add,
  then MUL/division, then general roots. Do not write a full speculative public
  header and fill it afterward. Freeze the ABI after real consumers exercise
  its lifetimes and numerical contracts.

The immediate combined sequence is **C0/C1 + F0, then F1, then C2 + F2**.
Extract the generic journal in C3 independently; do not put all bigfloat work
behind a complete OOC reorganization. Public OOC APIs and special functions
can follow without burdening small resident arithmetic.
