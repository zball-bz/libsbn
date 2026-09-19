# Exact division

`include/sbn3/divrem.h` provides exact integer division, B=2^64:

- unsigned: N = Q·D + R with 0 <= R < D;
- signed (`sbn3_int_*`): truncation toward zero, |R| < |D|, R carries the sign
  of N, outputs normalized without negative zero.

A zero divisor is a fatal `SBN3_FATAL_ARGUMENT` on the value boundary
(`prepare` or the basecase entry). Inputs may carry leading zero limbs; the
quotient span is always written completely (nn-dn+1 limbs, high zeros
included) and the remainder span has exactly dn limbs. The result structure
reports the normalized lengths.

## Entries

| Entry | Use | Work |
|---|---|---|
| `sbn3_divrem_basecase`, `sbn3_int_divrem_basecase` | one-shot, no plan/arena/team; caller scratch of nn+dn+1 limbs | O(nn·dn) words: GMP-derived divrem_1/divrem_2/3-by-2 schoolbook |
| `sbn3_divrem_query/bind/prepare/execute/unbind` | prepared divisor, repeated numerators, large shapes | word/schoolbook below the shape thresholds; block Barrett otherwise |

The service chooses at query time by the complete cost of using a plan.
Schoolbook serves divisors of up to 14 limbs and requests whose
`reuse_hint * max(quotient_limbs - 5, 0) * divisor_limbs` stays below the
measured fixed planning cost (`divrem_tuning.hpp`). Repeated uses amortize one
extra preparation pass when the live spans fit a CCD cache or the value
passes run in parallel; serial memory-bound short quotients retain their
per-execution cost. An explicitly requested
algorithm or block size bypasses this policy. Long rectangular u52 products
stream a bounded conversion buffer: their scratch does not grow with the
long operand, and crossing 2^20 divisor limbs does not force a transform.

Only the normalized divisor persists for Barrett. The residual product is
replaced in place by the signed remainder; its low shifted numerator block
reuses the already-consumed quotient-estimate operand buffer. The inverse
binding and subsequent product bindings/scratch occupy a shared lifetime
pool. Storage is the maximum of these phases, not their sum. Components in
`info` describe their phases and must not be summed to infer peak storage;
`storage_bytes` is the complete bound.

The planner prices one use of every recipe, preparation included: the FFT
family builds its tables at each binding (priced per table byte; the NTT
roots have their own model), and the short product (u52: no tables, no
spectrum) is a recipe of both block products next to the policy's transform
product. A search is made only where it can pay: the policy product search
when the short product in hand costs more than the least a transform could
plus the search itself, and not when it costs no more than the transform
recipe found for a larger block size of the same request; the cyclic ring
lattice from the divisor length at which the ring family pays on the block
product's team for the expected executions (50000 limbs on one worker and one
execution, times workers^0.8, over executions^0.7) and where a ring can be
taken: against a linear recipe that keeps a spectrum when the lattice's least
ring passes the family rule's fraction; against a linear transform without a
spectrum also when the share of its output a ring drops is worth the
lattice's price; against a short product from 48-limb blocks, when that
product costs more than the least transform over the ring plus the search and
the lattice's least ring, queried alone, is modelled cheaper. Block sizes are
ordered by their linear recipes; when the first is left with an FFT linear
recipe where rings pay, the next sizes within the measured model bias are
asked for their ring, and the first that takes one goes first.

Block Barrett computes the block inverse U of the top `block_limbs` limbs of
the normalized divisor, |U - B^(2in)/Dtop| < 3: up to 3072 limbs by a local u52 Newton recurrence with exact correction
(no product search or root tables), above by the
Newton ladder (`SBN3_NEWTON_INVERSE`). Block sizes are ordered by modelled
cost: one to three blocks (or the fewest blocks of at most dn limbs and one
more), and, where the modelled optimum under a local inverse lies below
those, the whole-block size next to it. The query performs each recipe
search once: the order's searches are the taken size's searches, and the
cached and plain recipes of one product share one search. It estimates each
quotient block as
floor(rtop·U/B^in), forms qhat·D' either as a cyclic product modulo
B^ring-1 (ring >= dn+2, divisor spectrum cached) or as a linear short
product, and recovers the exact residual by a signed lift. The estimate is
within [-5, +7] of the true block quotient, so at most eight corrections per
block are needed; more is a fatal `SBN3_FATAL_MATH`. The derivation is in
the research repository (`docs/divrem-design-2026-09-18.md`).

## Resources and lifetime

`sbn3_divrem_info` reports the complete storage of a binding and its
components. The normalized divisor, block inverse and cached spectra persist
until the next `prepare` or `unbind`. The shared region first holds the
inverse's scratch, then the product tables/workspaces and block scratch.
When both products use caches and sharing saves a large aligned workspace,
U and T reuse one workspace serially. Their immutable roots remain in their
spectra; switching rebinds their already compiled plans and codec constants.
Small products remain bound throughout. Execution performs no allocation,
page operation, plan search
or algorithm switch. Extra workspace grows with the divisor, block and ring
lengths, not with the total quotient length: numerator blocks are read
through an in-flight shift.

Under a `memory_budget` the policy passes over every plan that does not fit:
its block sizes in cost order, each with its taken residual family and then
the other one (a ring holds less than a linear product's workspace), then the
least size halved, twice at most. When no block plan fits, the schoolbook
(the least storage of all) serves the request where its complete cost is
bounded (divisors up to 64 limbs, quotients up to 8, or at most 360000 limb
steps beyond those 8: `divrem_tuning.hpp`); otherwise the query returns
`SBN3_QUERY_CAPACITY` and `info` reports the least requirement it found. A
requested algorithm or block size is never replaced.

`prepare` may be repeated with another divisor of the same length; the plan
is reusable and a range may be rebound after `unbind`. `reuse_hint` enters
the recipe cost as prepare + k·apply and the algorithm choice as above;
`block_limbs` pins the block size for experiments. Numerator, quotient, remainder, binding storage and team
storage must be mutually disjoint.
