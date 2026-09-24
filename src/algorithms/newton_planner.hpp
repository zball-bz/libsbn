#pragma once
#include "sbn3/newton.h"
#include "algorithms/newton_limits.hpp"
#include "product/window_group.hpp"
namespace sbn::v3::newton_detail {
inline constexpr uint64_t plan_magic = 0x53424e334e575431ULL;
inline constexpr uint64_t dyadic_plan_magic = 0x53424e3344595131ULL;
struct DyadicPlan {
    uint64_t marker=dyadic_plan_magic;
    size_t n=0,bytes=0;
    unsigned recipe=0,timing=0; // 0 word quotient, 1 u52 quotient, 2 local refinement
};
using Choice=product::WindowGroupChoice;
struct Plan {
    uint64_t marker = plan_magic, seal = 0;
    sbn3_newton_options options{};
    sbn3_newton_info info{};
    Choice choices[newton_limits::stages]{};
    // For local plans, choices' object bytes hold packed product records;
    // choice_count remains zero. These records are copied/decoded, never
    // interpreted as WindowGroupChoice objects.
    unsigned choice_count = 0,local_steps = 0;
    size_t shared_offset = 0, work1_offset = 0;
    size_t product_pool_offset = 0, product_pool_bytes = 0;
    bool compact = false;
    bool local = false;
};
static_assert(sizeof(Plan) <= sizeof(sbn3_newton_plan));
using Bundle=product::WindowGroupPlans;
enum class Cycle { Inverse, Rsqrt, Division };
bool choose_cycle(Plan &, Cycle, size_t m, size_t n, unsigned index, bool replay, Bundle &);
double cycle_cost(Cycle, const Bundle &, bool deep);
// Planning-speed device of choose_cycle: a lower bound of a lattice
// candidate's cold cost, known after its producer query alone. Candidates
// whose bound lies outside the selection window are not evaluated further;
// that is exact as long as the bound never exceeds the cost. The probe
// evaluates one candidate completely and returns both (test gate).
bool cycle_bound_probe(Cycle, size_t m, size_t n, const Choice &, bool compact, double &bound, double &cost);
} // namespace sbn::v3::newton_detail
