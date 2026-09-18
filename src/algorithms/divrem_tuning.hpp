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
// Candidate rule: complete dn-limb blocks (a full inverse for 2d/d) are
// considered when the head they leave is at most head_candidate_limbs, the
// divisor has at least head_candidate_min_divisor limbs, the team is wider
// than one worker and the prepared divisor serves at least
// head_candidate_min_applications block applications (reuse hint times
// complete blocks); the cost order still decides among candidates.
// Measured (experiments results/divrem_stageA_2026-09-18/final, arithmetic
// diagnostics): a full inverse never pays for a single application (one-shot
// 2d/d: 9-29% behind the best of two or three blocks at W16, 74% at 4096 W1)
// and lost 5-21% at two (one-shot 3d/d, dn=16384, W4, pre-amortization-2f87681);
// it pays back within 1-4 reuses, and every chosen point with seven or more
// applications at W4, W8 and W16 from dn=11585 up is faster than the previous
// default or level with it (0.49-1.01). Below dn=2896 the short-product cost
// order picks it wrongly (1.03-1.82, wip1) and at one worker it is mixed
// (0.90-1.12): neither is enabled.
// The candidate stands only with its word-division head. When the final
// cyclic recipe cannot serve the head (the ring leaves ring-dn < head limbs
// above the divisor, or its cheaper pair lowers the head limit) the policy
// returns to the other block sizes: complete blocks plus a padded one
// measured 1.22-1.28 of the previous default at dn=387141..1310717 and 1.56
// at 147454 (results/divrem_stageA2_2026-09-18/wip2, W16, 8 executions).
// Moving to the next ring with room instead was measured there and not
// adopted: 0.57-0.94 at six ring-growth representatives but 1.18 at
// dn=147454 (ring 1.28x); the cost order does not price that trade reliably.
inline constexpr size_t head_candidate_limbs = 8;
inline constexpr size_t head_candidate_min_divisor = 8192;
inline constexpr double head_candidate_min_applications = 4.0;
// The 3/2 estimate exceeds the quotient limb by at most one.
inline constexpr unsigned head_correction_limit = 2;
} // namespace sbn::v3::divrem_tuning
