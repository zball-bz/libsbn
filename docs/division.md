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

The service chooses at query time, by the complete cost of one use of the
plan: the schoolbook pays no planning, so besides the execution-level
thresholds (divisors up to 64 limbs, quotients up to 8) it serves every
request whose remaining schoolbook work, `reuse_hint` x (quotient limbs - 8)
x divisor limbs, stays below the measured cost of planning and preparing the
block algorithm (`divrem_tuning.hpp`). A requested `block_limbs` is a
request for blocks and is not subject to that rule.

Block Barrett computes the block inverse U of the top `block_limbs` limbs of
the normalized divisor, |U - B^(2in)/Dtop| < 3: up to 864 limbs by one
schoolbook division (no Newton plan and no Newton storage), above by the
Newton ladder (`SBN3_NEWTON_INVERSE`). Block sizes are ordered by modelled
cost: one to three blocks (or the fewest blocks of at most dn limbs and one
more), and, where the modelled optimum under a schoolbook inverse lies below
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
components. Persistent divisor state (normalized divisor, block inverse,
cached spectra, product tables/workspaces) lives from `prepare` until the
next `prepare` or `unbind`. The shared region holds the block inverse's
working storage (the schoolbook division's scratch or the Newton binding)
during `prepare` and the block scratch during `execute`; the two never
coexist. Execution performs no allocation, page operation, plan search
or algorithm switch. Extra workspace grows with the divisor, block and ring
lengths, not with the total quotient length: numerator blocks are read
through an in-flight shift.

`prepare` may be repeated with another divisor of the same length; the plan
is reusable and a range may be rebound after `unbind`. `reuse_hint` enters
the recipe cost as prepare + k·apply and the algorithm choice as above;
`block_limbs` pins the block size for experiments. Numerator, quotient, remainder, binding storage and team
storage must be mutually disjoint.
