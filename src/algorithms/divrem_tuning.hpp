#pragma once
#include <stddef.h>
#include "algorithms/inverse_tuning.hpp"
namespace sbn::v3::divrem_tuning {
// Native one-use crossover, Zen5 AI Max 395 / Clang 21.1.8. The rule prices
// fixed planning against repeated schoolbook limb steps. Rectangular u52
// products stream a bounded working set; the block remainder overwrites the
// product and only the normalized divisor persists. Thus both algorithms
// hold about 24 bytes per divisor limb for short quotients, without a
// transform-capacity cliff at 2^20 or a separate page-commit penalty.
// Complete-call pool/grow comparisons, W1/W16, divisors 65537..2097169,
// quotients 4/5/6/8/12/16: blocks are consistently worthwhile from six
// quotient limbs; four/five straddle equality. Preserve schoolbook there.
// Evidence: experiments/results/codex_divrem_radix_2026-09-19/struct2/crossover-*.
// The 18000-limb-step fixed-work term retains the smaller-divisor calibration
// of stage C-2. Requested algorithms/block sizes bypass this policy.
inline constexpr size_t schoolbook_level_quotient = 5;
inline constexpr double schoolbook_use_work = 18000.0;
// Short-quotient live spans: D', residual, product and caller N/R, ~40 B
// per divisor limb against a 32-MiB CCD cache. REUSE_K=8 measurements keep
// the preparation credit in this regime, and when value passes are parallel;
// serial DRAM at 2M limbs still favors schoolbook for a five-limb quotient.
inline constexpr size_t short_cache_limbs = (size_t(32)<<20)/40;
// Under a memory budget that no block size meets, the schoolbook (it holds the least storage of all) serves the
// request where its complete cost is bounded: within a factor of two of the blocks (divisors up to 64 limbs at any
// quotient length, quotients up to 8 limbs at any divisor length: same measurement) or at most this many limb steps
// beyond those 8 limbs (about 0.12 ms per use). That is the domain it served by default before this rule was
// re-measured, so no request that was planned under a budget is refused now; outside it the query reports the least
// block requirement as before.
// Before that, a budget none of the ordered block sizes meets is tried on smaller blocks (each holds less): the least
// ordered size halved this many times, so a plan taken under a budget has at most four times the blocks of the
// smallest size the order considered.
inline constexpr unsigned budget_halvings = 2;
inline constexpr size_t schoolbook_budget_divisor = 64;
inline constexpr size_t schoolbook_budget_quotient = 8;
inline constexpr double schoolbook_budget_work = 360000.0;
// A block holds at most dn quotient limbs, and its fixed work (two product calls, the passes over residual and
// divisor) is shared by that many: on divisors this short the blocks stay behind the schoolbook at any quotient
// length (same measurement, quotients up to 21000 limbs: schoolbook / blocks 0.22 at three limbs, 0.63 at eight,
// 0.71 at ten, 0.85-0.97 at twelve, 0.94-1.10 at thirteen, 1.08-1.34 at sixteen; complete fresh calls, stage C-2
// review: blocks / schoolbook 1.12 at thirteen limbs, one use and eight, 0.86 from fifteen; the forced curve 1.065,
// 1.125, 1.047 at thirteen to fifteen). Stage C-1 and before: 64, from an execution-only sweep of a planner that
// priced no short product.
inline constexpr size_t schoolbook_max_divisor = 14;
// One-use reciprocal route, shared with radix. The cutoff tracks the complete
// Newton path, including planning (inverse_tuning.hpp).
inline constexpr size_t inverse_basecase_limbs = inverse_tuning::basecase_limbs;
// Planning/binding seed after geometry-only search; ~80 us of a 110-140 us
// complete 513..1219-limb reciprocal. Same price for all Newton candidates.
inline constexpr double inverse_newton_planning_ns = 80000.0;
// Besides quotient-driven block sizes, rank one candidate near the minimum
// of local_inverse_cost(in) + executions*ceil(qn/in)*pair. Its own product
// prices decide; a larger block's pair price is not a pruning bound. The
// local inverse's measured exponent is 1.5 (inverse_tuning.hpp).
// One use of a recipe, complete. The FFT family builds its tables at every binding (its own, or those of the spectrum
// a cached recipe reads), and root_prepare_cost prices the NTT families' roots only, so the order priced them at
// zero: measured 56 us at dn=3001, 112 us at 6428, 113 us at 10120, 125 us at 45000, 1063 us at 100003 (seven-way
// radix), 90 us for 4096 x 513 = 0.27-0.42 ns per table byte whatever the other operand (experiments
// results/divrem_fresh_c2_2026-09-19/recipe-base, one worker); the NTT families measured 19-25 us against 24-32
// modelled at 2^18. The least measured price, so a transform is never priced out by its tables.
inline constexpr double fft_table_ns_per_byte = 0.27;
// The short product (u52: no tables, no spectrum) is a recipe of both block products next to the policy's, which
// compares executions only. One application of D' x qhat, tables + spectrum + product (same measurement): 3001 x 130
// 16 us short against 70; 6428 x 121 34 against 144; 6428 x 287 57 against 143; 10120 x 593 131 against 172;
// 45000 x 501 541 against 436; 16384 x 5462 677 against 205: a transform pays once it is applied often or to long
// blocks, and the order prices both with their preparation.
// What a search costs the query that makes it (same measurement, query column): one policy product search 5.0-7.8
// us for a longer operand of 2^10..2^15 limbs, 51-79 us from 2^15 (the small-NTT domain enumerates its candidates);
// the cyclic ring lattice 60-135 us (about a hundred pinned product and spectrum queries). A search returns a
// transform recipe, and none applies for less than transform_floor_ns_per_limb per output limb on one worker (least
// measured complete product: 3.9 ns at 514 x 513, 4.8-8.6 for 3001..100003-limb divisors, 10.6 for the NTT at 2^18;
// a kept spectrum saves a third per application). The short product in hand is taken without the search when its
// complete modelled cost (preparation and all expected applications) is not above that floor by more than the
// search's price: the search could not save what it costs.
inline constexpr double product_search_short_ns = 2800.0; // longer operand below 2^10 limbs: 2.8-3.4 us
inline constexpr double product_search_ns = 6000.0;
inline constexpr double product_search_small_domain_ns = 55000.0;
inline constexpr double lattice_search_ns = 100000.0;
inline constexpr double transform_floor_ns_per_limb = 3.5;
// Below this many limbs of the shorter operand no search is made whatever the model says: the short product
// measured 0.45-0.65 ns per limb of the longer operand plus 0.04-0.06 ns per limb pair (3001 x 130 .. 300007 x 10),
// under the floor above up to about 48 limbs for any longer operand, while its one-worker model adds memory-pressure
// terms calibrated on saturated batches (1.6 ms modelled for 300007 x 6, 0.2 ms measured) that would send every
// long divisor into three 50-80 us searches for nothing.
inline constexpr size_t transform_min_limbs = 48;
// Planning seed for the block inverse: a Newton ladder to precision n costs
// about this many linear n x n products (rungs sum to ~2 products of the
// final size plus the smaller rungs). Only orders block-size candidates.
inline constexpr double inverse_cost_ratio = 3.0;
// Arithmetic guarantees (docs/divrem-design): |qhat-q|<=7, |E|<7D.
inline constexpr unsigned correction_limit = 8;
inline constexpr size_t ring_guard_words = 2; // ring >= dn + guard for the signed residual lift
// Residual recipe: the cyclic product is taken when its ring is at most this
// fraction of the linear product's dn+in output; otherwise the linear recipe.
// Measured 2026-09-18 (results/divrem_2026-09-18/family, W16, prepare+execute):
// 65536: cyclic 73728/87382=0.84 wins (1.51-1.63 vs 1.71 ms); 262144:
// 294912/349526=0.84 wins (6.86 vs 7.88); 1048576: 1310720/1398102=0.94 loses
// (25.8 vs 22.8). The cost model ranked both cases the other way, so this
// structural rule replaces its linear-versus-cyclic comparison only.
// Controlled experiment 2026-09-19 (experiments results/divrem_fresh_c2_2026-09-19/ring, complete calls of 2d/d with
// the linear family only, the cyclic only and the policy, 1/8-octave divisor lengths, one and eight executions, one
// and sixteen workers, 228 shapes): among rings of 0.67-0.89 of the linear output the fraction does not separate
// wins from losses; the divisor length and the executions do. One use: the ring family is behind below about 40000
// limbs on one worker (2-16 %, the lattice search included) and about 150000 on sixteen (5-33 %), ahead above
// (10-25 % and 2-14 %). Eight executions: ahead nearly everywhere on one worker (geometric mean 0.93, behind 3-16 %
// at 32768-46341), behind 5-29 % at 8192-30048 limbs on sixteen workers (eight-worker products), ahead above.
// The fraction stays as the family rule between two recipes that keep a spectrum. What the experiment separates is
// applied first: the ring family is searched from cyclic_ring_min_limbs divisor limbs on one worker and one
// execution per divisor, times workers^0.8 of the block product's team, over executions^0.7. Forced cyclic / forced
// linear in that experiment: one worker, one use 1.02-1.20 at 19484..46341 limbs and 0.78-0.92 from 50535; eight
// executions 1.01-1.08 up to 10624 and 0.85-0.99 from 11585 (1.01-1.08 again at 35734-42495); sixteen workers
// (eight on the product up to 262160 limbs), one use 1.06-1.58 up to 131072, 0.94-1.14 at 142935..440872 and
// 0.90-0.99 from 480774; eight executions 1.03-1.37 up to 30048, 0.83-1.12 to 120194, 0.83-0.99 from 131072. The
// policy of stage C-2 and before took rings the fraction admits from 18658 limbs on one worker (2d/d), 10-29 %
// behind the linear family up to 36000. Against a short product (no tables, no spectrum) no ring is searched below
// transform_min_limbs of the block: that recipe was taken because no transform pays for so short a block, and a
// ring is a transform of the divisor's length (stage C-2 review: rings in its place 1.1-2.3 times slower at sixteen
// workers, quotients of 11-39 limbs). From that length on a ring may win by its kept spectrum, a ring longer than
// the linear output included (301412 / 913-limb quotient, one worker, 5.2 against 6.6 ms; 359409 / 749 with eight
// executions 30 against 51 ms; same review), but a lattice searched for nothing is 100-170 us, 4-20 % of a complete
// call for 1000-limb quotients on 131101..370727 limbs (results/divrem_fresh_c3_2026-09-19/ab-c2, ring3-w1): when
// the short product costs more than the least transform over the ring and the lattice's price, the lattice's least
// ring is queried alone, and the lattice is searched when that ring is modelled cheaper; the cost model decides.
// Against a linear transform that keeps no spectrum the cost model decides as well, and the lattice is searched when
// its least ring passes the fraction (a ring of 0.899, 131072 for 129567 + 16196 limbs, saved 545 us of a 4 ms call
// where the stage C-2 bound below declined the search) or when the share of the linear output a ring drops, times
// the linear recipe's cost, exceeds the lattice's price (stage C-2's bound: it declines the rings of 0.956-0.99 on
// long divisors at sixteen workers, where the linear recipes measured 0.84-1.03 of them, and keeps the ring of
// 0.9035 for 92683 + 11586 limbs on one worker, 2.8 against 4.1 ms without it).
inline constexpr double cyclic_ring_fraction = 0.9;
inline constexpr double cyclic_ring_min_limbs = 50000.0;
inline constexpr double cyclic_ring_worker_scaling = 0.8;
inline constexpr double cyclic_ring_execution_scaling = 0.7;
inline constexpr unsigned cyclic_ring_execution_limit = 8; // the most executions per divisor that were measured
// Block sizes are ordered by their linear recipes. Where rings pay, the cost model has the FFT family's linear
// recipe at 0.9-1.0 of the ring recipe of a neighbouring size (210472 / 26310-limb quotient, one worker: two linear
// blocks modelled 5.27 ms against one block under the ring 212992 at 5.74) while complete calls measured the ring
// plan 1.11-1.35 times faster (8.09 against 5.99 ms there; 20 of 20 such shapes from 58961 limbs, stage C-2
// review): that family's model is calibrated inside the cache. When the first size of the order is left with an FFT
// linear recipe, the following sizes whose linear totals are within this multiple of its total are asked for their
// ring in order, and the first that takes one goes first. Asking every size at ordering time measured 6-9 % of a
// complete call at 59000-103000 limbs (two to three lattices of about 85 us each on plans that did not change).
// The same bias applies where the cost model decides between such a linear transform and a ring of the same size
// (the ring of 0.899 above: the linear recipe without a spectrum is modelled cheaper and measured 16 % slower).
inline constexpr double fft_linear_order_bias = 1.25;
// Short head block (quotient limbs modulo the block size) by word division:
// one multiply-subtract pass over the divisor per limb, against one padded
// block's product pair. Planning seeds (ns) for that crossover, measured
// 2026-09-18 (results/divrem_stageA_2026-09-18/headcost, forced full block,
// numerators 2dn-1..2dn+3): 0.25-0.5 ns/limb single-threaded (dn 4096 W1,
// 65536 W16), 0.13 ns/limb with team passes inside the shared cache (2^20,
// W16), 0.5-0.6 ns/limb once divisor and residual exceed it (2^22, W16).
inline constexpr double head_step_ns = 40.0;
inline constexpr double head_ns_per_limb = 0.45;
inline constexpr double head_parallel_ns_per_limb = 0.13;
inline constexpr double head_memory_ns_per_limb = 0.6;
inline constexpr size_t head_cache_limbs = size_t(1) << 21; // divisor + residual within the 64 MiB shared cache
// The head limit takes this fraction of the modelled crossover. Measured
// crossovers (results/divrem_stageA_2026-09-18/pre-margin-7b1fd45/headcross
// and results/divrem_stageA_2026-09-18/headcost, forced full block): 42 limbs
// at dn=1000 W1, 17 at 20000 W16, 19 at 300000
// W16, ~55 at 2^20 W16, ~11 at 2^22 W16, against unscaled model limits of
// 24, 24, 8, 25 and 6: the product-pair estimate is up to 1.4x high, and a
// head past the true crossover would cost more than the padded block.
inline constexpr double head_cost_margin = 0.6;
// Complete dn-limb blocks (a full inverse for 2d/d, the quotient limbs above
// them by word division) are served on request (options.block_limbs) and are
// not a policy candidate. As one they measured (stage A, 2026-09-18,
// arithmetic diagnostics of prepare + 8 executions, not fresh-call timing)
// 0.49-0.91 of the near-equal blocks up to dn~49k and 0.54-0.92 above
// dn~262k, but level or behind in between (0.92-1.15 at 16 workers, 1.03-1.06
// at 8, 1.09 for a one-shot 8d/d), where the cost order's modelled ratio does
// not predict the measured one, and they hold 1.24-3.29 times the storage for
// 2d/d (experiments results/divrem_stageA_2026-09-18/final,
// divrem_stageA2_2026-09-18/final/ab-zone*, divrem_stageA3_2026-09-18).
// Promotion needs fresh-call cost, boundary and equal-budget evidence.
// The block-size order prices whole blocks and gives the word-division head
// no credit: the head serves what the taken size leaves over, within the
// limit above. An order that credited it changed the policy's size for
// quotients of 9-23 limbs above dn=2^20 (experiments
// docs/divrem-stageA-2026-09-18 section 11, one to eight workers, arithmetic
// diagnostics): 0.43-0.82 of the whole-block order where a smaller size under
// a head replaced one larger block, 1.03-1.14 where it only added word steps
// to the same number of product pairs. Crediting it again needs the evidence
// named above and a check that the final recipe keeps the credited head (the
// cyclic ring may leave ring-dn < head limbs above the divisor, its cheaper
// pair a lower limit; complete blocks plus a padded one measured 1.22-1.28 of
// near-equal blocks at dn=387141..1310717 and 1.56 at 147454,
// results/divrem_stageA2_2026-09-18/wip2, W16, 8 executions), gated on a
// policy plan that reaches it.
// The 3/2 estimate exceeds the quotient limb by at most one.
inline constexpr unsigned head_correction_limit = 2;
} // namespace sbn::v3::divrem_tuning
