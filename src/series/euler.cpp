#include "sbn3/constants.h"
#include "sbn3/newton.h"
#include "series/arithmetic.hpp"
#include "series/scaled_ratio.hpp"
#include "common/identity.hpp"
#include "runtime/team.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <cstring>
#include <new>
#include <time.h>
namespace sbn::v3::series {
namespace {
constexpr uint64_t magic = 0x53424e3345554c31ULL;
struct Plan {
    uint64_t marker = magic, seal = 0;
    sbn3_e_options options{};
    sbn3_e_info info{};
    sbn3_series_info series{};
    sbn3_newton_plan division{};
    sbn3_newton_info finish{};
    size_t control_bytes = 0, values_offset = 0, T = 0, D = 0, A = 0, normalized_D = 0, Q = 0;
    size_t pool = 0, finish_pool = 0, finish_value_bytes = 0, metadata = 0, prepared = 0, scratch = 0;
};
static_assert(sizeof(Plan) <= sizeof(sbn3_e_plan));
struct Binding {
    Plan plan{};
    Formula formula{FormulaKind::Euler};
    FiniteBinding series{};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, values{}, metadata{}, prepared{}, scratch{};
    sbn3_series_values result{};
    bool used = false;
    sbn3_e_metrics metrics{};
};
size_t up(size_t n, size_t a = 128) {
    size_t r = 0;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "Euler layout alignment");
    return r;
}
uint64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
uint64_t hash(const Plan &p) {
    uint64_t h = identity::word(identity::fnv_seed, magic);
    for (uint64_t v : {p.info.fractional_limbs, p.info.working_limbs, p.info.terms, p.info.storage_bytes,
                       p.info.storage_alignment, p.series.schedule_id, p.finish.plan_id,
                       p.options.memory_budget, p.control_bytes, p.values_offset, p.T, p.D, p.A,
                       p.normalized_D, p.Q, p.pool, p.finish_pool, p.finish_value_bytes, p.metadata, p.prepared, p.scratch})
        h = identity::word(h, v);
    return h;
}
Plan load(const sbn3_e_plan &p) {
    Plan q{};
    std::memcpy(&q, p.opaque, sizeof q);
    require(q.marker == magic && q.seal == hash(q), SBN3_FATAL_ARGUMENT, "Euler plan identity");
    return q;
}
uint64_t terms(size_t precision) {
    const __uint128_t target = __uint128_t(precision) * 64 + 8;
    auto enough = [&](uint64_t n) { return factorial_log_bounds(0, n).lower >= target; };
    uint64_t hi = 2;
    while (!enough(hi))
        hi *= 2;
    uint64_t lo = 1;
    while (lo < hi) {
        const auto m = lo + (hi - lo) / 2;
        if (enough(m))
            hi = m;
        else
            lo = m + 1;
    }
    return lo;
}
FiniteFormula execution_formula(const Formula &formula) {
    auto f = finite_formula(formula);
    f.reuse_values = true;
    return f;
}
uint64_t *at(Binding &b, size_t relative) {
    return reinterpret_cast<uint64_t *>(b.arena->base + b.offset + relative);
}
Binding &get(sbn3_e_binding *p) {
    require(p, SBN3_FATAL_ARGUMENT, "Euler binding");
    auto &b = *reinterpret_cast<Binding *>(p);
    require(b.plan.marker == magic && pthread_equal(b.team->creator, pthread_self()) && !b.team->busy,
            SBN3_FATAL_LIFETIME, "Euler binding owner/state");
    return b;
}
void release_series(Binding &b) {
    for (auto *l : {&b.metadata, &b.prepared, &b.scratch})
        if (l->token)
            sbn3_arena_release(b.arena, l);
}
void finish_guard(const uint64_t *q, size_t drop) {
    bool low = true, high = true;
    for (size_t j = 1; j < drop; ++j) {
        low &= q[j] == 0;
        high &= q[j] == UINT64_MAX;
    }
    require(!(low && q[0] < 16) && !(high && q[0] > UINT64_MAX - 16), SBN3_FATAL_MATH,
            "Euler output guard separation");
}
} // namespace
} // namespace sbn::v3::series
using namespace sbn::v3;
using namespace sbn::v3::series;
extern "C" sbn3_query_result sbn3_e_query(size_t n, const sbn3_e_options *options, sbn3_e_plan *out,
                                          sbn3_e_info *info) {
    require(out && info, SBN3_FATAL_ARGUMENT, "Euler query output");
    *info = {};
    if (!n || n > (size_t(1) << 28))
        return SBN3_UNSUPPORTED;
    Plan p{};
    // Consider zero or one serial layer above the subtree forks. A second
    // layer duplicates more prepared tables without reducing the 1B scratch peak.
    p.options = options ? *options : sbn3_e_options{{16, 8, 1, 0, 0}, 0};
    p.info.fractional_limbs = n;
    p.info.working_limbs = std::max(size_t(4), n + 2);
    p.info.output_limbs = n + 1;
    p.info.terms = terms(p.info.working_limbs);
    p.info.workers = p.options.series.workers;
    Formula formula{FormulaKind::Euler};
    FinitePlan fp{};
    auto rc = finite_query(execution_formula(formula), {1, p.info.terms + 1}, 3, p.options.series, fp);
    if (rc != SBN3_SUPPORTED)
        return rc;
    p.series = fp.info;
    sbn3_newton_options no{p.info.workers, 0, 0, 0};
    rc = sbn3_newton_query(SBN3_NEWTON_DIVIDE, p.info.working_limbs, &no, &p.division, &p.finish);
    if (rc != SBN3_SUPPORTED)
        return rc;
    p.info.storage_alignment = std::max(size_t(128), p.finish.storage_alignment);
    p.control_bytes = up(sizeof(Binding));
    p.values_offset = p.control_bytes;
    size_t cursor = p.values_offset;
    auto words = [&](size_t count) {
        const size_t at = cursor;
        cursor = up(cursor + count * 8);
        return at;
    };
    p.T = words(std::max(p.series.output.limbs[0], p.info.working_limbs + 1));
    p.D = words(std::max(p.series.output.limbs[1], p.info.working_limbs));
    p.A = p.T;
    p.normalized_D = up(p.A + (p.info.working_limbs + 1) * 8);
    // Compaction is useful only if it releases a workspace-alignment page.
    // Otherwise keep D where it is and avoid an O(n) overlapping move.
    const size_t compact_pool=up(p.normalized_D+p.info.working_limbs*8,p.info.storage_alignment);
    if(up(p.D+p.info.working_limbs*8,p.info.storage_alignment)==compact_pool)p.normalized_D=p.D;
    p.Q = p.A; // division consumes the normalized numerator after its last read
    p.info.value_bytes = cursor - p.values_offset;
    p.pool = up(cursor, p.info.storage_alignment);
    p.finish_value_bytes = p.normalized_D + p.info.working_limbs * 8 - p.values_offset;
    p.finish_pool = up(p.values_offset + p.finish_value_bytes, p.info.storage_alignment);
    p.metadata = p.pool;
    p.prepared = up(p.metadata + p.series.plan_bytes, p.series.prepared_alignment);
    p.scratch = up(p.prepared + p.series.prepared_bytes, p.series.workspace_alignment);
    p.info.storage_bytes = std::max(p.scratch + p.series.workspace_bytes, p.finish_pool + p.finish.storage_bytes);
    p.info.series_prepared_bytes = p.series.prepared_bytes;
    p.info.series_workspace_bytes = p.series.workspace_bytes;
    p.info.finish_storage_bytes = p.finish.storage_bytes;
    p.info.serial_prefix = p.series.serial_prefix;
    p.seal = hash(p);
    p.info.plan_id = p.seal;
    *info = p.info;
    if (p.options.memory_budget && p.info.storage_bytes > p.options.memory_budget)
        return SBN3_QUERY_CAPACITY;
    std::memset(out, 0, sizeof *out);
    std::memcpy(out->opaque, &p, sizeof p);
    return SBN3_SUPPORTED;
}
extern "C" void sbn3_e_bind(const sbn3_e_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                            sbn3_e_binding **out) {
    require(opaque && arena && team && out, SBN3_FATAL_ARGUMENT, "Euler bind arguments");
    const auto p = load(*opaque);
    require(team->arena == arena && team->width == p.info.workers &&
                pthread_equal(team->creator, pthread_self()) && !team->busy,
            SBN3_FATAL_TEAM, "Euler bind team");
    require(offset <= arena->virtual_bytes && p.info.storage_bytes <= arena->virtual_bytes - offset &&
                !(uintptr_t(arena->base + offset) & (p.info.storage_alignment - 1)) &&
                arena->contains(offset, offset + p.info.storage_bytes) &&
                arena->unleased(offset, p.info.storage_bytes),
            SBN3_FATAL_WORKSPACE, "Euler prepared unleased range");
    auto control = arena->acquire(offset, p.control_bytes);
    auto *b = ::new (control.data) Binding{};
    b->plan = p;
    b->control = control;
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->values = arena->acquire(offset + p.values_offset, p.info.value_bytes);
    b->metadata = arena->acquire(offset + p.metadata, p.series.plan_bytes);
    b->prepared = arena->acquire(offset + p.prepared, p.series.prepared_bytes);
    b->scratch = arena->acquire(offset + p.scratch, p.series.workspace_bytes);
    FinitePlan fp{};
    fp.formula = execution_formula(b->formula);
    fp.options = p.options.series;
    fp.info = p.series;
    fp.spec = {fp.formula.recipe, {1, p.info.terms + 1}, 3, fp.formula.formula_id, fp.formula.parameter_id};
    finite_prepare(fp, *arena, *team, b->metadata, b->prepared, b->scratch, b->series);
    b->result.value[0].mantissa = {at(*b, p.T), p.series.output.limbs[0], 0, 0};
    b->result.value[1].mantissa = {at(*b, p.D), p.series.output.limbs[1], 0, 0};
    *out = reinterpret_cast<sbn3_e_binding *>(b);
}
static sbn3_const_limbs execute_e(Binding &b,sbn3_limbs out) {
    const auto &p=b.plan;
    const size_t n=p.info.working_limbs;
    require(!b.used,SBN3_FATAL_LIFETIME,"Euler single-use binding");
    b.used = true;
    uint64_t start = now();
    finite_execute(b.series, b.result);
    b.metrics.series_ns = now() - start;
    const auto &d = b.result.value[1].mantissa, &t = b.result.value[0].mantissa;
    require(d.size && !d.negative && !t.negative, SBN3_FATAL_MATH, "Euler finite signs");
    const int64_t shift = int64_t(64 * n) - int64_t(64 * d.size - __builtin_clzll(d.data[d.size - 1]));
    {
        ComputeLease compute(*b.arena);
        if (shift <= 0 && shift > -64) {
            parallel_limbs::right_shift(b.team, at(b, p.A), n + 1, t.size, unsigned(-shift));
            parallel_limbs::right_shift(b.team, at(b, p.D), n, d.size, unsigned(-shift));
        } else {
            scaled_slice(at(b, p.A), n + 1, {t.data, t.size, 0}, shift);
            scaled_slice(at(b, p.D), n, {d.data, d.size, 0}, shift);
        }
        if(p.normalized_D!=p.D)std::memmove(at(b, p.normalized_D), at(b, p.D), n * 8);
    }
    require(at(b, p.A)[n] <= 1 && (at(b, p.normalized_D)[n - 1] >> 63), SBN3_FATAL_MATH,
            "Euler normalized quotient");
    release_series(b);
    b.arena->release(b.values);
    b.values = b.arena->acquire(b.offset + p.values_offset, p.finish_value_bytes);
    start = now();
    sbn3_newton_binding *division = nullptr;
    sbn3_newton_bind(&p.division, b.arena, b.offset + p.finish_pool, b.team, &division);
    b.metrics.finish_prepare_ns = now() - start;
    start = now();
    const sbn3_newton_inputs in{{at(b, p.A), n + 1}, {at(b, p.normalized_D), n}, 0};
    sbn3_newton_execute(division, &in, {at(b, p.Q), n + 1});
    sbn3_newton_unbind(division);
    const size_t drop = n - p.info.fractional_limbs;
    auto *q = at(b, p.Q);
    finish_guard(q, drop);
    require(q[n] == 1, SBN3_FATAL_MATH, "Euler integer part");
    ++q[n];
    const sbn3_const_limbs result{q+drop,p.info.output_limbs};
    if(out.data)std::memmove(out.data,result.data,result.count*8);
    b.metrics.finish_ns = now() - start;
    return result;
}
extern "C" void sbn3_e_execute(sbn3_e_binding *ptr,sbn3_limbs out) {
    auto &b=get(ptr);const auto &p=b.plan;
    require(out.capacity>=p.info.output_limbs && out.data && !(uintptr_t(out.data)&63) &&
                (out.data==at(b,p.Q) || !overlaps(out.data,p.info.output_limbs*8,b.arena->base+b.offset,p.info.storage_bytes)),
            SBN3_FATAL_ARGUMENT,"Euler output/lifetime");
    (void)execute_e(b,out);
}
extern "C" sbn3_const_limbs sbn3_e_execute_inplace(sbn3_e_binding *ptr) {
    return execute_e(get(ptr),{});
}
extern "C" void sbn3_e_get_metrics(const sbn3_e_binding *p, sbn3_e_metrics *m) {
    require(m, SBN3_FATAL_ARGUMENT, "Euler metrics");
    *m = get(const_cast<sbn3_e_binding *>(p)).metrics;
}
extern "C" void sbn3_e_unbind(sbn3_e_binding *p) {
    auto &b = get(p);
    release_series(b);
    auto *arena = b.arena;
    auto values = b.values, control = b.control;
    b.plan.marker = 0;
    b.~Binding();
    arena->release(values);
    arena->release(control);
}
