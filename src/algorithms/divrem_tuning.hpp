#pragma once
#include <stddef.h>
namespace sbn::v3::divrem_tuning {
// Initial shape thresholds for the exact division service. Zen5 AI Max 395,
// Clang 21.1.8. Calibrated by experiments/results/divrem_2026-09-18 (crossover
// sweep of the schoolbook entry against the block-Barrett service); the
// values only choose between exact algorithms and never change results.
// Forced-algorithm sweep (results/divrem_2026-09-18/crossover2): the block
// Barrett execute beats the 3/2 schoolbook from dn=64 on balanced shapes
// (1.8x at 64x128, 2.9x at 64x512) and from 12 quotient limbs at any dn
// (1.5-2.7x at 12, 1.4-5.6x at 16); at 8 quotient limbs the two are within
// noise. The thresholds below keep the schoolbook where it is at least as fast.
inline constexpr size_t schoolbook_max_divisor = 64;   // dn <= this: 3/2 schoolbook
inline constexpr size_t schoolbook_max_quotient = 8;   // quotient limbs <= this: schoolbook (O(dn) per limb)
// One use of a plan, complete (query, bind, divisor preparation, executions, unbind): the two thresholds above
// compare executions only, and a block-Barrett division first pays its planning (recipe searches of the query, the
// block inverse, divisor-side spectra and their tables): 90-130 us up to dn~1000, 190-270 us at 2^11..2^14, 360 us
// to 1 ms at 2^15..2^18 on one Zen 5 core, whatever the quotient length. The schoolbook pays none of it, so it
// serves a request whose schoolbook work beyond the execution-level threshold,
//   reuse_hint * (quotient limbs - schoolbook_max_quotient) * divisor limbs,
// is at most schoolbook_use_work limb steps. Measured equal-cost points of complete calls (experiments
// results/divrem_fresh_2026-09-19/crossover, forced algorithms in interleaved windows, one and sixteen workers
// alike): 335k (64d/d, dn~73), 405-412k (2d/d at dn~640, 8d/d at dn~243), 590-850k (quotients of 21-420 limbs on
// divisors of 2^11..2^17 limbs), 470k (2^18). The constant sits under the least of them: at the switch the
// schoolbook is level with or ahead of the blocks (0.83-0.98 of their complete call), never behind. A requested
// block size is a request for blocks and is not subject to this rule.
inline constexpr double schoolbook_use_work = 360000.0;
// Block inverse: up to this many limbs one schoolbook division U = floor((B^(2in) - 1) / Dtop) replaces the Newton
// ladder (its query, binding and execution). Equal complete cost near 880 limbs (the radix reciprocal's
// measurement, src/radix/programs.hpp); on complete divisions the step across the switch is +1.5-2 %
// (results/divrem_fresh_2026-09-19/crossover, list3). A binding that is re-prepared many times pays the
// schoolbook inverse each time (about twice the Newton execution at 340 limbs); the hint has no field for that.
inline constexpr size_t inverse_basecase_limbs = 864;
// Ordering price of that schoolbook inverse (a 2in/in division): 1.6 us at 64 limbs, 87 us at 512, 194 us at 768,
// 264 us at 896 (src/radix/programs.hpp, same routine) = 0.33-0.39 ns per limb squared.
inline constexpr double inverse_basecase_ns_per_limb2 = 0.33;
// Ordering price of the Newton route's planning (its query and binding), which a schoolbook inverse does not pay:
// the ladder's query alone measured 288 us at 1366 limbs, 418 us at 5462, 629 us at 21846 (experiments
// results/divrem_fresh_2026-09-19/profile-base). One constant for every Newton size, so the order among sizes that
// all take the ladder is what it was; it only weighs a ladder against a schoolbook inverse.
inline constexpr double inverse_newton_planning_ns = 300000.0;
// With those two prices the order also weighs one size the quotient-driven sizes (one to three blocks, or the
// fewest blocks and one more) do not contain: under a schoolbook inverse the modelled cost a*in^2 + executions *
// ceil(qn/in) * pair is least near in* = cbrt(executions * qn * pair / (2a)), pair being the best ordered size's
// product pair; the whole-block size at in* joins the order when it lies below the others and its estimate beats
// the best one's. Complete one-shot 2d/d calls by requested block count (experiments
// results/divrem_fresh_2026-09-19/crossover, list5/list6, one worker): dn=1024 189 us at three blocks, 145 at five
// to eight; 2048: 421 -> 292; 2896: 558 -> 387; 4096: 725 -> 472; the pairs of five to twelve blocks cost about
// what three cost, the inverse falls with in^2. Above dn~5000 the estimate declines it (the policy's plans there
// are the former ones, query differential plan-diff-*); a Newton-route size with more blocks, which measured up to
// 11 % better at dn 5793-9742, is not a candidate.
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
inline constexpr double cyclic_ring_fraction = 0.9;
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
