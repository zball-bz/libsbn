#include "common/identity.hpp"
#include "sbn3/newton.h"
#include "algorithms/inverse_program.hpp"
#include "algorithms/newton_tuning.hpp"
#include "algorithms/newton_contract.hpp"
#include "algorithms/newton_limits.hpp"
#include "algorithms/newton_planner.hpp"
#include "algorithms/product_stage.hpp"
#include "algorithms/rsqrt.hpp"
#include "algorithms/divide_terminal.hpp"
#include "algorithms/sqrt2_rational.hpp"
#include "algorithms/sqrt2_finish.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cmath>
#include <new>
#include <string.h>
#include <time.h>
namespace sbn::v3 {
namespace {
constexpr uint64_t magic = 0x53424e334e575431ULL;
constexpr unsigned max_steps = newton_limits::stages, max_products = newton_limits::products,
                   max_spectra = newton_limits::spectra, max_leases = newton_limits::leases;
using newton_detail::Plan;
using newton_detail::Bundle;
struct Binding;
struct CompactState {
    Binding *owner = nullptr;
    Bundle *bundles = nullptr;
    ProductStage *stages = nullptr;
    unsigned index = 0, slot = UINT32_MAX, reservations = 0;
    bool active = false;
    sbn3_mul_binding *product = nullptr;
    sbn3_spectrum *spectrum = nullptr;
    sbn3_lease guard{}, tables{}, work{}, cache{};
};
struct Binding {
    uint64_t marker = magic;
    Plan plan{};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease leases[max_leases]{};
    unsigned lease_count = 0;
    sbn3_mul_binding *products[max_products]{};
    unsigned product_count = 0;
    sbn3_spectrum *spectra[max_spectra]{};
    unsigned spectrum_count = 0;
    InverseRung inv_steps[max_steps]{};
    RsqrtRung rs_steps[max_steps]{};
    RationalSqrt2Step rat_steps[max_steps]{};
    InverseProgram inv{};
    RsqrtProgram rs{};
    RationalSqrt2Program rat{};
    DivideTerminal division{};
    sbn3_mul_binding *checker = nullptr;
    uint64_t *result = nullptr, *A = nullptr, *D = nullptr, *U = nullptr, *Q = nullptr, *shared[2]{};
    uint64_t *division_input = nullptr; // dead inverse ping-pong slot after the final rung
    bool used = false;
    sbn3_newton_metrics metrics{};
    CompactState *compact = nullptr;
};
uint64_t feed(uint64_t h, uint64_t v) {
    return identity::word(h, v);
}
uint64_t seal(const Plan &p) {
    uint64_t h = feed(1469598103934665603ULL, p.marker);
    for (uint64_t x : {uint64_t(p.options.workers),
                       uint64_t(p.options.prime_count),
                       p.options.memory_budget,
                       uint64_t(p.options.timing),
                       uint64_t(p.info.kind),
                       p.info.precision_limbs,
                       p.info.storage_bytes,
                       p.info.plan_id,
                       uint64_t(p.info.workers),
                       uint64_t(p.info.products),
                       uint64_t(p.info.spectra),
                       uint64_t(p.info.stages),
                       uint64_t(p.info.lease_peak),
                       p.info.output_limbs,
                       p.info.storage_alignment,
                       p.info.control_bytes,
                       p.info.table_bytes,
                       p.info.product_workspace_bytes,
                       p.info.value_bytes,
                       p.info.spectrum_bytes,
                       p.info.shared_bytes,
                       p.info.setup_bytes,
                       p.shared_offset,
                       p.work1_offset,
                       p.product_pool_offset,
                       p.product_pool_bytes,
                       uint64_t(p.compact),
                       uint64_t(p.choice_count)})
        h = feed(h, x);
    for (unsigned k = 0; k < p.choice_count; ++k)
        for (uint64_t x :
             {uint64_t(p.choices[k].np), uint64_t(p.choices[k].T), uint64_t(p.choices[k].algorithm),
              uint64_t(p.choices[k].workers), p.choices[k].ring})
            h = feed(h, x);
    return h;
}
Plan load(const sbn3_newton_plan &p) {
    Plan q{};
    memcpy(&q, p.opaque, sizeof q);
    require(q.marker == magic && q.choice_count <= max_steps && q.seal == seal(q), SBN3_FATAL_ARGUMENT,
            "Newton plan identity");
    return q;
}
size_t aligned(size_t n, size_t a) {
    size_t r = 0;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "Newton layout alignment");
    return r;
}
size_t compact_state_offset() {
    return aligned(sizeof(Binding), 128);
}
size_t compact_bundles_offset() {
    return aligned(compact_state_offset() + sizeof(CompactState), 128);
}
size_t compact_stages_offset(unsigned count) {
    return aligned(compact_bundles_offset() + sizeof(Bundle) * count, 128);
}
size_t compact_control_bytes(unsigned count) {
    return aligned(compact_stages_offset(count) + sizeof(ProductStage) * count, 128);
}
struct PhaseLayout {
    size_t cache, table, work, bytes;
};
PhaseLayout phase_layout(const Bundle &v, unsigned slot) {
    const auto &i = v.infos[slot];
    const size_t cache = i.cached_mask ? v.future.storage_bytes : 0;
    const size_t table = aligned(cache, 128);
    const size_t work = aligned(table + i.mul.table_bytes, i.mul.workspace_alignment);
    return {cache, table, work, work + i.mul.workspace_bytes};
}
void compact_drop_product(CompactState &s) {
    if (!s.product)
        return;
    sbn3_mul_unbind(s.product);
    s.product = nullptr;
    s.owner->arena->release(s.work);
    s.owner->arena->release(s.tables);
    s.work = s.tables = {};
    s.slot = UINT32_MAX;
}
void compact_drop_cache(CompactState &s) {
    if (!s.spectrum)
        return;
    sbn3_spectrum_release(s.spectrum);
    s.spectrum = nullptr;
    s.owner->arena->release(s.cache);
    s.cache = {};
}
void compact_enter(void *ptr, unsigned index) {
    auto &s = *static_cast<CompactState *>(ptr);
    require(!s.active && s.guard.token && index < s.owner->plan.choice_count, SBN3_FATAL_LIFETIME,
            "Newton compact stage enter");
    s.owner->arena->release(s.guard);
    s.guard = {};
    s.index = index;
    s.active = true;
}
sbn3_mul_binding *compact_bind(void *ptr, unsigned slot) {
    auto &s = *static_cast<CompactState *>(ptr);
    const auto &v = s.bundles[s.index];
    require(s.active && slot < v.count, SBN3_FATAL_LIFETIME, "Newton compact product slot");
    if (s.product && s.slot == slot)
        return s.product;
    compact_drop_product(s);
    const auto &i = v.infos[slot];
    if (!i.cached_mask)
        compact_drop_cache(s);
    const auto layout = phase_layout(v, slot);
    auto &b = *s.owner;
    const size_t base = b.offset + b.plan.product_pool_offset;
    require(layout.bytes <= b.plan.product_pool_bytes, SBN3_FATAL_WORKSPACE, "Newton compact phase budget");
    if (i.cached_mask && !s.spectrum) {
        s.cache = b.arena->acquire(base, layout.cache);
        sbn3_spectrum_reserve_plan(&v.producer, SBN3_SPECTRUM_COLUMNS, v.future.generation, b.arena, &s.cache,
                                   &s.spectrum);
        ++s.reservations;
    }
    s.tables = b.arena->acquire(base + layout.table, i.mul.table_bytes);
    s.work = b.arena->acquire(base + layout.work, i.mul.workspace_bytes);
    sbn3_product_bind(&v.plans[slot], b.arena, &s.tables, &s.work, b.team,
                      i.cached_mask ? s.spectrum : nullptr, nullptr, &s.product);
    s.slot = slot;
    return s.product;
}
sbn3_spectrum *compact_cache(void *ptr) {
    return static_cast<CompactState *>(ptr)->spectrum;
}
void compact_leave(void *ptr) {
    auto &s = *static_cast<CompactState *>(ptr);
    require(s.active, SBN3_FATAL_LIFETIME, "Newton compact stage leave");
    compact_drop_product(s);
    compact_drop_cache(s);
    s.active = false;
    auto &b = *s.owner;
    s.guard = b.arena->acquire(b.offset + b.plan.product_pool_offset, b.plan.product_pool_bytes);
}
const ProductStageOps compact_ops{compact_enter, compact_bind, compact_cache, compact_leave};
sbn3_mul_options options(const Plan &p, size_t n) {
    sbn3_mul_options o{};
    o.workers = n < native_policy::newton_single_worker_below ? 1 : p.options.workers;
    o.prime_count = p.options.prime_count;
    o.borrow_output = 1;
    return o;
}
using newton_detail::Cycle;
using newton_detail::choose_cycle;
enum class Memory { Table, Workspace, Value, Spectrum };
struct Assembly {
    Plan &p;
    Binding *b;
    size_t cursor = aligned(sizeof(Binding), 128), max_ring = 0;
    size_t tables = 0, workspace = 0, values = 0, spectrum_memory = 0, max_align = 128;
    size_t pool_bytes = 0;
    size_t max_work0 = 0, max_work1 = 0;
    size_t inverse_work0 = 0, inverse_work1 = 0;
    unsigned products = 0, spectra = 0, leases = 0, choices = 0, stages = 0, workers = 1;
    uint64_t hash = 1469598103934665603ULL;
    sbn3_query_result status = SBN3_SUPPORTED;
    // Inverse/rsqrt have the same cold work in either placement. Only
    // division changes its spectrum-build count when the pool is compact.
    bool replay_choices;
    Assembly(Plan &plan, Binding *binding = nullptr, bool replay = true) : p(plan), b(binding), replay_choices(replay) {
        if (p.compact)
            cursor = compact_control_bytes(p.choice_count);
    }
    sbn3_lease memory(size_t bytes, size_t alignment, Memory kind) {
        if (!bytes || status != SBN3_SUPPORTED)
            return {};
        size_t at = 0, end = 0;
        if (!align_size(cursor, alignment, at) || !add_size(at, bytes, end)) {
            status = SBN3_QUERY_CAPACITY;
            return {};
        }
        cursor = end;
        max_align = std::max(max_align, alignment);
        ++leases;
        if (kind == Memory::Table)
            tables += bytes;
        else if (kind == Memory::Workspace)
            workspace += bytes;
        else if (kind == Memory::Value)
            values += bytes;
        else
            spectrum_memory += bytes;
        hash = feed(feed(feed(hash, bytes), alignment), uint64_t(kind));
        if (!b)
            return {};
        require(b->lease_count < max_leases, SBN3_FATAL_WORKSPACE, "Newton lease storage");
        auto l = b->arena->acquire(b->offset + at, bytes);
        b->leases[b->lease_count++] = l;
        return l;
    }
    uint64_t *words(size_t n) {
        if (n > SIZE_MAX / 8) {
            status = SBN3_QUERY_CAPACITY;
            return nullptr;
        }
        return static_cast<uint64_t *>(memory(n * 8, 128, Memory::Value).data);
    }
    sbn3_mul_binding *product(const sbn3_mul_plan &plan, const sbn3_product_info &info,
                              const sbn3_spectrum *cache = nullptr) {
        if (status != SBN3_SUPPORTED)
            return nullptr;
        if (products >= max_products) {
            status = SBN3_QUERY_CAPACITY;
            return nullptr;
        }
        hash = feed(feed(hash, info.mul.arithmetic_id), info.mul.execution_id);
        workers = std::max(workers, info.mul.workers);
        if (p.compact) {
            ++products;
            return nullptr;
        }
        auto t = memory(info.mul.table_bytes, 128, Memory::Table),
             w = memory(info.mul.workspace_bytes, info.mul.workspace_alignment, Memory::Workspace);
        ++products;
        if (!b)
            return nullptr;
        sbn3_mul_binding *result = nullptr;
        sbn3_product_bind(&plan, b->arena, &t, &w, b->team, info.cached_mask ? cache : nullptr, nullptr,
                          &result);
        b->products[b->product_count++] = result;
        return result;
    }
    sbn3_spectrum *cache(const sbn3_mul_plan &producer, const sbn3_product_info &info,
                         const sbn3_spectrum_desc &future) {
        if (status != SBN3_SUPPORTED)
            return nullptr;
        if (!future.basis_id)
            return nullptr;
        if (spectra >= max_spectra) {
            status = SBN3_QUERY_CAPACITY;
            return nullptr;
        }
        ++spectra;
        (void)info;
        hash = feed(feed(hash, future.basis_id), future.generation);
        if (p.compact)
            return nullptr;
        auto l = memory(future.storage_bytes, 128, Memory::Spectrum);
        if (!b)
            return nullptr;
        sbn3_spectrum *result = nullptr;
        sbn3_spectrum_reserve_plan(&producer, SBN3_SPECTRUM_COLUMNS, future.generation, b->arena, &l,
                                   &result);
        b->spectra[b->spectrum_count++] = result;
        return result;
    }
    bool cycle(Cycle kind, size_t m, size_t n, Bundle &v) {
        if (status != SBN3_SUPPORTED)
            return false;
        if (!choose_cycle(p, kind, m, n, choices, b != nullptr || (p.compact && (replay_choices || kind != Cycle::Division)), v)) {
            status = SBN3_QUERY_CAPACITY;
            return false;
        }
        if (p.compact) {
            for (unsigned k = 0; k < v.count; ++k) {
                const auto layout = phase_layout(v, k);
                pool_bytes = std::max(pool_bytes, layout.bytes);
                tables = std::max(tables, v.infos[k].mul.table_bytes);
                workspace = std::max(workspace, v.infos[k].mul.workspace_bytes);
                spectrum_memory = std::max(spectrum_memory, layout.cache);
                max_align = std::max(max_align, v.infos[k].mul.workspace_alignment);
                hash = feed(hash, layout.bytes);
            }
            if (b) {
                ::new (b->compact->bundles + choices) Bundle(v);
                ::new (b->compact->stages + choices) ProductStage{b->compact, choices, &compact_ops};
            }
        }
        ++choices;
        ++stages;
        max_ring = std::max(max_ring, v.producer_info.cyclic_limbs);
        const size_t work0 = kind == Cycle::Inverse ? v.producer_info.cyclic_limbs
                             : kind == Cycle::Rsqrt ? m + newton_contract::guard_words
                                                    : std::max(m + 1, newton_contract::residual_words(m, n));
        max_work0 = std::max(max_work0, work0);
        max_work1 = std::max(max_work1, v.producer_info.cyclic_limbs);
        for (unsigned k = 0; k < v.count; ++k)
            max_work1 = std::max(max_work1, sbn3_mul_output_capacity(&v.infos[k].mul));
        if (kind == Cycle::Inverse) {
            inverse_work0 = std::max(inverse_work0, work0);
            inverse_work1 = std::max(inverse_work1, v.producer_info.cyclic_limbs);
        }
        return true;
    }
    void inverse(size_t target, uint64_t *v0, uint64_t *v1) {
        size_t sizes[max_steps], m = target;
        unsigned count = 0;
        while (m > newton_limits::inverse_seed_words) {
            if (count == max_steps) {
                status = SBN3_QUERY_CAPACITY;
                return;
            }
            sizes[count++] = m;
            m = newton_contract::next_precision(m);
        }
        if (b)
            b->inv = {target, m, count, b->inv_steps, {v0, v1}};
        for (unsigned j = count; j-- > 0;) {
            const size_t n = sizes[j];
            Bundle v{};
            if (!cycle(Cycle::Inverse, m, n, v))
                return;
            auto *h = cache(v.producer, v.producer_info, v.future);
            auto *mul = product(v.plans[0], v.infos[0], h);
            if (b)
                b->inv_steps[count - 1 - j] = {m,       n,      v.producer_info.cyclic_limbs, mul, h, nullptr,
                                               nullptr, b->team};
            if (b && p.compact)
                b->inv_steps[count - 1 - j].stage = b->compact->stages + choices - 1;
            m = n;
        }
        if (b)
            b->U = count ? (count & 1 ? v1 : v0) : v0;
    }
    void rsqrt(size_t target, uint64_t *v0, uint64_t *v1) {
        size_t sizes[max_steps], m = target;
        unsigned count = 0;
        while (m > newton_limits::rsqrt_seed_words) {
            if (count == max_steps) {
                status = SBN3_QUERY_CAPACITY;
                return;
            }
            sizes[count++] = m;
            m = newton_contract::next_precision(m);
        }
        if (b)
            b->rs = {target, m, count, b->rs_steps, {v0, v1}};
        for (unsigned j = count; j-- > 0;) {
            const size_t n = sizes[j];
            Bundle v{};
            if (!cycle(Cycle::Rsqrt, m, n, v))
                return;
            auto *h = cache(v.producer, v.producer_info, v.future);
            auto *sq = product(v.plans[0], v.infos[0], h);
            auto *mul = product(v.plans[1], v.infos[1], h);
            if (b)
                b->rs_steps[count - 1 - j] = {
                    m, n, v.producer_info.cyclic_limbs, sq, mul, h, nullptr, nullptr, true, b->team};
            if (b && p.compact)
                b->rs_steps[count - 1 - j].stage = b->compact->stages + choices - 1;
            m = n;
        }
        if (b)
            b->result = count ? (count & 1 ? v1 : v0) : v0;
    }
    void division(size_t n) {
        const size_t m = newton_contract::next_precision(n);
        Bundle v{};
        if (!cycle(Cycle::Division, m, n, v))
            return;
        auto *h = cache(v.producer, v.producer_info, v.future);
        auto *mul = product(v.plans[0], v.infos[0], h);
        auto *res = product(v.plans[1], v.infos[1]);
        if (b)
            b->division = {m,
                           n,
                           v.producer_info.cyclic_limbs,
                           mul,
                           res,
                           h,
                           {nullptr, nullptr},
                           b->team,
                           std::max(m + 1, newton_contract::residual_words(m, n))};
        if (b && p.compact)
            b->division.stage = b->compact->stages + choices - 1;
        if (b) {
            b->division.output_capacity = v.producer_info.cyclic_limbs;
            for (unsigned k = 0; k < v.count; ++k)
                b->division.output_capacity =
                    std::max(b->division.output_capacity, sbn3_mul_output_capacity(&v.infos[k].mul));
        }
    }
    void rational(size_t digits, uint64_t *v0, uint64_t *v1, uint64_t *v2) {
        const unsigned last = rational_sqrt2_iterations(digits);
        if (b)
            b->rat = {last, b->rat_steps, {v0, v1, v2}};
        for (unsigned k = 1; k < last; ++k) {
            const size_t n = rational_sqrt2_capacity(k);
            auto o = options(p, n);
            sbn3_product_request r{};
            r.a_limbs = r.b_limbs = n;
            sbn3_mul_plan producer{}, mp{}, sp{};
            sbn3_product_info pi{}, mi{}, si{};
            sbn3_spectrum_desc future{};
            status = sbn3_product_query(&r, &o, &producer, &pi);
            if (status != SBN3_SUPPORTED)
                return;
            sbn3_spectrum *h = nullptr;
            if (sbn3_spectrum_query(&producer, SBN3_SPECTRUM_COLUMNS, k, &future) == SBN3_SUPPORTED) {
                h = cache(producer, pi, future);
                r.cached_a[0] = &future;
                status = sbn3_product_query(&r, &o, &mp, &mi);
                if (status != SBN3_SUPPORTED)
                    return;
            } else {
                mp = producer;
                mi = pi;
            }
            r.kind = SBN3_PRODUCT_SQR;
            r.b_limbs = 0;
            status = sbn3_product_query(&r, &o, &sp, &si);
            if (status != SBN3_SUPPORTED)
                return;
            auto *mul = product(mp, mi, h);
            auto *sq = product(sp, si, h);
            ++stages;
            if (b)
                b->rat_steps[k - 1] = {n, rational_sqrt2_capacity(k + 1), mul, sq, h};
        }
    }
    void run() {
        const auto kind = p.info.kind;
        const size_t n = p.info.precision_limbs;
        if (kind == SBN3_NEWTON_INVERSE) {
            // The retained path preserves the short-call layout. A compact
            // ladder sends the last output directly to caller storage.
            const size_t capacity =
                p.compact ? newton_contract::minimum_approximation_words(n, newton_limits::inverse_seed_words)
                          : n + 1;
            auto *a = words(capacity), *c = words(capacity);
            inverse(n, a, c);
        } else if (kind == SBN3_NEWTON_RSQRT || kind == SBN3_SQRT2_RSQRT) {
            const size_t precision = n + (kind == SBN3_SQRT2_RSQRT ? 2 : 0);
            // SQRT2 consumes its full intermediate internally; its final
            // value belongs to this binding rather than caller output.
            const size_t capacity =
                p.compact
                    ? newton_contract::minimum_approximation_words(precision, newton_limits::rsqrt_seed_words)
                    : precision + 1;
            auto *a = words(capacity), *c = words(capacity);
            rsqrt(precision, a, c);
        } else {
            const size_t precision = kind == SBN3_SQRT2_RATIONAL ? std::max(size_t(4), n + 2) : n,
                         m = newton_contract::next_precision(precision);
            size_t length =
                kind == SBN3_SQRT2_RATIONAL
                    ? std::max(rational_sqrt2_capacity(rational_sqrt2_iterations(n)) + 2, m + 1)
                    : m + 1;
            if (p.compact && kind == SBN3_NEWTON_DIVIDE)
                length = std::max(length, newton_contract::residual_words(m, precision));
            auto *a = words(length), *c = words(length);
            uint64_t *third = nullptr;
            if (kind == SBN3_SQRT2_RATIONAL) {
                third = words(length);
                rational(n, a, c, third);
            }
            inverse(m, a, c);
            division(precision);
            if (b && p.compact && kind == SBN3_NEWTON_DIVIDE)
                b->division_input = b->U == a ? c : a;
            if (kind == SBN3_SQRT2_RATIONAL) {
                auto *A = words(precision + 1), *D = words(precision + 1), *Q = words(precision + 1);
                if (b) {
                    b->A = A;
                    b->D = D;
                    b->Q = Q;
                }
            }
        }
        if (kind >= SBN3_SQRT2_RSQRT) {
            auto o = options(p, n);
            sbn3_product_request r{};
            r.kind = SBN3_PRODUCT_SQR;
            r.a_limbs = n + 1;
            sbn3_mul_plan q{};
            sbn3_product_info i{};
            status = sbn3_product_query(&r, &o, &q, &i);
            if (status != SBN3_SUPPORTED)
                return;
            auto *checker = product(q, i);
            if (b)
                b->checker = checker;
        }
    }
    bool finish() {
        if (status != SBN3_SUPPORTED)
            return false;
        const bool verify = p.info.kind >= SBN3_SQRT2_RSQRT;
        const bool division_union = p.compact && p.info.kind == SBN3_NEWTON_DIVIDE;
        const size_t work0_words = division_union ? inverse_work0 :
            std::max(p.compact ? max_work0 : max_ring, verify ? 2 * p.info.precision_limbs + 2 : 0);
        const size_t work0_bytes = bytes_for(work0_words, 8),
                     work1_bytes = bytes_for(division_union ? inverse_work1 : max_work1, 8);
        const size_t work1 = aligned(work0_bytes, 128);
        const size_t execute_bytes = division_union ? std::max(work1 + work1_bytes, bytes_for(max_work1, 8))
                                                     : work1 + work1_bytes;
        const size_t pool_at = aligned(cursor, max_align);
        const size_t shared = execute_bytes, shared_at = aligned(pool_at + pool_bytes, max_align);
        const unsigned peak = leases + 1 + (shared ? 1 : 0) + (p.compact ? 3 : 0);
        if (peak > max_leases) {
            status = SBN3_QUERY_CAPACITY;
            return false;
        }
        uint64_t key = hash;
        for (uint64_t v : {shared_at, shared, work1, uint64_t(peak), pool_at, pool_bytes})
            key = feed(key, v);
        if (!b) {
            auto &i = p.info;
            i.workers = workers;
            i.products = products;
            i.spectra = spectra;
            i.stages = stages;
            i.lease_peak = peak;
            i.control_bytes =
                p.compact ? compact_control_bytes(p.choice_count) : aligned(sizeof(Binding), 128);
            i.table_bytes = tables;
            i.product_workspace_bytes = workspace;
            i.value_bytes = values;
            i.spectrum_bytes = spectrum_memory;
            i.shared_bytes = shared;
            i.setup_bytes = 0;
            i.storage_alignment = max_align;
            i.storage_bytes = aligned(shared_at + shared, 128);
            i.plan_id = key;
            p.shared_offset = shared_at;
            p.work1_offset = work1;
            p.product_pool_offset = pool_at;
            p.product_pool_bytes = pool_bytes;
        } else {
            require(choices == p.choice_count && key == p.info.plan_id && shared_at == p.shared_offset &&
                        shared == p.info.shared_bytes && pool_at == p.product_pool_offset &&
                        pool_bytes == p.product_pool_bytes,
                    SBN3_FATAL_ARGUMENT, "Newton query/bind equivalence");
            if (p.compact)
                b->compact->guard = b->arena->acquire(b->offset + pool_at, pool_bytes);
            if (shared) {
                auto l = b->arena->acquire(b->offset + shared_at, shared);
                b->leases[b->lease_count++] = l;
                b->shared[0] = static_cast<uint64_t *>(l.data);
                b->shared[1] = reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(l.data) + work1);
            }
            for (size_t j = 0; j < b->inv.rung_count; ++j) {
                b->inv_steps[j].residual = b->shared[0];
                b->inv_steps[j].correction = b->shared[1];
            }
            for (size_t j = 0; j < b->rs.rung_count; ++j) {
                b->rs_steps[j].residual = b->shared[0];
                b->rs_steps[j].correction = b->shared[1];
            }
            b->division.work[0] = division_union ? b->division_input : b->shared[0];
            b->division.work[1] = division_union ? b->shared[0] : b->shared[1];
        }
        return true;
    }
};
Binding &binding(sbn3_newton_binding *p) {
    auto *b = reinterpret_cast<Binding *>(p);
    require(b && b->marker == magic, SBN3_FATAL_LIFETIME, "Newton binding");
    return *b;
}
const Binding &binding(const sbn3_newton_binding *p) {
    return binding(const_cast<sbn3_newton_binding *>(p));
}
void idle(const Binding &b) {
    require(pthread_equal(b.team->creator, pthread_self()) && !b.team->busy, SBN3_FATAL_TEAM,
            "Newton owner/idle");
}
uint64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
void value(const Binding &b, sbn3_const_limbs v, size_t count, sbn3_limbs out, bool consume = false) {
    require(v.count == count && (!count || v.data), SBN3_FATAL_ARGUMENT, "Newton input shape");
    const size_t bytes = bytes_for(count, 8);
    valid_span(v.data, bytes, "Newton input");
    require(!(reinterpret_cast<uintptr_t>(v.data) & 7) &&
                ((consume && v.data == out.data) || !overlaps(v.data, bytes, out.data, (b.plan.info.output_limbs) * 8)) &&
                !overlaps(v.data, bytes, b.arena->base + b.offset, b.plan.info.storage_bytes),
            SBN3_FATAL_ARGUMENT, "Newton value overlap");
    for (unsigned j = 0; j < b.team->width; ++j) {
        const auto &l = j ? b.team->stacks[j] : b.team->storage;
        require(!overlaps(v.data, bytes, l.data, l.bytes), SBN3_FATAL_ARGUMENT, "Newton input/team overlap");
    }
}
} // namespace
} // namespace sbn::v3
using namespace sbn::v3;
extern "C" sbn3_query_result sbn3_newton_query(sbn3_newton_kind kind, size_t n,
                                               const sbn3_newton_options *options, sbn3_newton_plan *out,
                                               sbn3_newton_info *info) {
    require(out && info, SBN3_FATAL_ARGUMENT, "Newton query output");
    *info = {};
    if (unsigned(kind) > SBN3_SQRT2_RATIONAL || !n || (kind == SBN3_NEWTON_DIVIDE && n < 4))
        return SBN3_UNSUPPORTED;
    if (n > newton_limits::precision_words)
        return SBN3_QUERY_CAPACITY;
    Plan p{};
    p.options = options ? *options : sbn3_newton_options{1, 0, 0, 0};
    if (!p.options.workers || p.options.workers > 32 || p.options.timing > 1 ||
        (p.options.prime_count && (p.options.prime_count < newton_limits::first_ntt_prime_count ||
                                   p.options.prime_count > newton_limits::last_ntt_prime_count)))
        return SBN3_UNSUPPORTED;
    p.info.kind = kind;
    p.info.precision_limbs = n;
    p.info.output_limbs = n + 1;
    Assembly a(p);
    a.run();
    if (!a.finish())
        return a.status;
    if (kind <= SBN3_NEWTON_DIVIDE && p.choice_count) {
        // Reuse placement-independent choices; compact division reselects
        // its final cycle with the extra spectrum-build cost. Include all
        // immutable recipe metadata when comparing the storage layouts.
        Plan compact = p;
        compact.compact = true;
        Assembly candidate(compact,nullptr,false);
        candidate.run();
        if (candidate.finish() && compact.info.storage_bytes < p.info.storage_bytes)
            p = compact;
    }
    *info = p.info;
    if (p.options.memory_budget && p.info.storage_bytes > p.options.memory_budget)
        return SBN3_QUERY_CAPACITY;
    p.seal = seal(p);
    memset(out, 0, sizeof *out);
    memcpy(out->opaque, &p, sizeof p);
    return SBN3_SUPPORTED;
}
extern "C" void sbn3_newton_bind(const sbn3_newton_plan *opaque, sbn3_arena *arena, size_t offset,
                                 sbn3_team *team, sbn3_newton_binding **out) {
    require(opaque && arena && team && out, SBN3_FATAL_ARGUMENT, "Newton bind arguments");
    const auto p = load(*opaque);
    require(team->arena == arena && team->width >= p.info.workers &&
                pthread_equal(team->creator, pthread_self()) && !team->busy,
            SBN3_FATAL_TEAM, "Newton bind team");
    size_t end = 0;
    require(add_size(offset, p.info.storage_bytes, end) && end <= arena->virtual_bytes &&
                arena->contains(offset, end) &&
                !(reinterpret_cast<uintptr_t>(arena->base + offset) & (p.info.storage_alignment - 1)) &&
                arena->unleased(offset, p.info.storage_bytes),
            SBN3_FATAL_WORKSPACE, "Newton prepared unleased range");
    sbn3_arena_stats stats{};
    sbn3_arena_get_stats(arena, &stats);
    require(stats.active_leases + p.info.lease_peak <= Arena::max_leases, SBN3_FATAL_WORKSPACE,
            "Newton lease capacity", p.info.lease_peak, Arena::max_leases - stats.active_leases);
    auto control = arena->acquire(offset, p.info.control_bytes);
    auto *b = ::new (control.data) Binding{};
    b->plan = p;
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->leases[b->lease_count++] = control;
    if (p.compact) {
        auto *base = static_cast<unsigned char *>(control.data);
        b->compact = ::new (base + compact_state_offset()) CompactState{};
        b->compact->owner = b;
        b->compact->bundles = reinterpret_cast<Bundle *>(base + compact_bundles_offset());
        b->compact->stages = reinterpret_cast<ProductStage *>(base + compact_stages_offset(p.choice_count));
    }
    Assembly assembly(b->plan, b);
    assembly.run();
    require(assembly.finish(), SBN3_FATAL_MATH, "Newton bind plan replay");
    *out = reinterpret_cast<sbn3_newton_binding *>(b);
}
extern "C" void sbn3_newton_execute(sbn3_newton_binding *opaque, const sbn3_newton_inputs *provided,
                                    sbn3_limbs output) {
    auto &b = binding(opaque);
    idle(b);
    require(!b.used, SBN3_FATAL_LIFETIME, "Newton binding already executed");
    const sbn3_newton_inputs empty{};
    const auto &in = provided ? *provided : empty;
    const auto &p = b.plan;
    const size_t n = p.info.precision_limbs, bytes = bytes_for(n + 1, 8);
    require(output.capacity >= n + 1 && !(reinterpret_cast<uintptr_t>(output.data) & 63), SBN3_FATAL_ARGUMENT,
            "Newton output span");
    valid_span(output.data, bytes, "Newton output");
    require(!overlaps(output.data, bytes, b.arena->base + b.offset, p.info.storage_bytes),
            SBN3_FATAL_ARGUMENT, "Newton output/storage overlap");
    for (unsigned j = 0; j < b.team->width; ++j) {
        const auto &l = j ? b.team->stacks[j] : b.team->storage;
        require(!overlaps(output.data, bytes, l.data, l.bytes), SBN3_FATAL_ARGUMENT,
                "Newton output/team overlap");
    }
    if (p.info.kind == SBN3_NEWTON_INVERSE || p.info.kind == SBN3_NEWTON_DIVIDE)
        value(b, in.denominator, n, output);
    if (p.info.kind == SBN3_NEWTON_DIVIDE)
        value(b, in.numerator, n + 1, output, true);
    b.used = true;
    const auto start = p.options.timing ? now() : 0;
    switch (p.info.kind) {
    case SBN3_NEWTON_INVERSE:
        inverse_program(b.inv, in.denominator.data, output.data);
        break;
    case SBN3_NEWTON_RSQRT:
        rsqrt_program(b.rs, in.radicand, output.data);
        break;
    case SBN3_NEWTON_DIVIDE:
        inverse_program(b.inv, in.denominator.data + n - b.inv.target, b.U);
        divide_terminal(b.division, in.numerator.data, in.denominator.data, b.U, output.data);
        break;
    case SBN3_SQRT2_RSQRT:
        rsqrt_program(b.rs, 2, b.result);
        sqrt2_from_rsqrt(output.data, n, b.result, b.rs.target);
        break;
    case SBN3_SQRT2_RATIONAL: {
        const auto v = rational_sqrt2(b.rat);
        normalize_rational_sqrt2(v, b.division.n, b.A, b.D);
        inverse_program(b.inv, b.D + b.division.n - b.inv.target, b.U);
        divide_terminal(b.division, b.A, b.D, b.U, b.Q);
        memcpy(output.data, b.Q + b.division.n - n, bytes);
        break;
    }
    default:
        fatal(SBN3_FATAL_ARGUMENT, "Newton operation");
    }
    const auto calculated = p.options.timing ? now() : 0;
    if (b.checker)
        b.metrics.unit_corrections = sqrt2_certify(b.checker, output.data, n, b.shared[0]);
    if (p.options.timing) {
        b.metrics.compute_ns = calculated - start;
        b.metrics.verify_ns = b.checker ? now() - calculated : 0;
    }
    b.metrics.products_executed = unsigned(2 * b.inv.rung_count + 2 * b.rs.rung_count);
    if (p.info.kind == SBN3_NEWTON_DIVIDE || p.info.kind == SBN3_SQRT2_RATIONAL)
        b.metrics.products_executed += 3;
    if (p.info.kind == SBN3_SQRT2_RATIONAL)
        b.metrics.products_executed += 2 * (b.rat.last_iteration - 1);
    if (b.checker)
        ++b.metrics.products_executed;
    b.metrics.spectra_computed = b.compact ? b.compact->reservations : b.spectrum_count;
}
extern "C" void sbn3_newton_get_metrics(const sbn3_newton_binding *opaque, sbn3_newton_metrics *out) {
    const auto &b = binding(opaque);
    idle(b);
    require(out, SBN3_FATAL_ARGUMENT, "Newton metrics");
    *out = b.metrics;
}
extern "C" void sbn3_newton_unbind(sbn3_newton_binding *opaque) {
    auto &b = binding(opaque);
    idle(b);
    for (unsigned j = 0; j < b.product_count; ++j)
        sbn3_mul_unbind(b.products[j]);
    for (unsigned j = 0; j < b.spectrum_count; ++j)
        sbn3_spectrum_release(b.spectra[j]);
    auto *arena = b.arena;
    const auto control = b.leases[0];
    if (b.compact) {
        require(!b.compact->active && !b.compact->product && !b.compact->spectrum, SBN3_FATAL_LIFETIME,
                "Newton compact final state");
        arena->release(b.compact->guard);
        b.compact->~CompactState();
    }
    for (unsigned j = b.lease_count; j-- > 1;)
        sbn3_arena_release(arena, &b.leases[j]);
    b.marker = 0;
    b.~Binding();
    arena->release(control);
}

#include "algorithms/newton_planner_impl.hpp"
