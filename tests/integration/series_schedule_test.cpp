#include "product_support.hpp"
#include "sbn3/series.h"
#include "series/formulas.hpp"
#include "series/schedule.hpp"
#include <algorithm>
#include <cmath>

struct Context {
    sbn3_series_recipe recipe = SBN3_SERIES_HYPERDESCENT;
    unsigned radix = 1;
    bool modular = false, large_bound = false;
    mutable size_t queries = 0;
    std::atomic<size_t> calls{0}, last_end{0};
    std::atomic<uint64_t> owner_min{UINT64_MAX};
    uintptr_t scratch_begin = 0, scratch_end = 0;
    const void *self = nullptr;
};
static sbn3_query_result bounds(const void *p, sbn3_series_range, uint64_t n, unsigned need,
                                sbn3_series_shape *s) {
    const auto &c = *static_cast<const Context *>(p);
    ++c.queries;
    if (n > SIZE_MAX / 16)
        return SBN3_QUERY_CAPACITY;
    *s = {};
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j))
            s->limbs[j] = c.large_bound ? size_t(n + 1) : 2;
    return SBN3_SUPPORTED;
}
static double work(const void *, sbn3_series_range r) {
    return double(r.end - r.begin) * (double(r.begin) + double(r.end) + 1) * 0.5;
}
static sbn3_query_result resources(const void *, const sbn3_series_stage *s, sbn3_series_resources *r) {
    size_t cap = 0;
    for (auto x : s->output.limbs)
        cap = std::max(cap, x);
    *r = {cap * 8 + 73, 256, double(s->max_terms) / s->workers + 1, 37, 128};
    return SBN3_SUPPORTED;
}
using I = __int128_t;
using U = __uint128_t;
static constexpr uint64_t prime = (uint64_t(1) << 61) - 1;
struct Number {
    I m = 0;
    int64_t e = 0;
};
static Number add(Number a, Number b, bool mod) {
    if (mod)
        return {I((U(a.m) + U(b.m)) % prime), 0};
    const int64_t e = std::min(a.e, b.e);
    assert(a.e - e < 100 && b.e - e < 100);
    return {a.m * I(U(1) << (a.e - e)) + b.m * I(U(1) << (b.e - e)), e};
}
static Number mul(Number a, Number b, bool mod) {
    return {mod ? I(U(a.m) * U(b.m) % prime) : a.m * b.m, a.e + b.e};
}
struct Triple {
    Number t, d, u;
};
static Triple term(Context &c, uint64_t k) {
    if (c.modular)
        return {{I(k % 97 + 1), 0}, {I(k % 101 + 1), 0}, {I(k % 7 + 1), 0}};
    const I sign = k & 1 ? -1 : 1;
    return {{sign * I(k + 3), c.recipe == SBN3_SERIES_BINARY_BBP ? -int64_t(k * c.radix) : 0},
            {I(k + 1), 0},
            {sign * 2, 0}};
}
static Triple combine(Context &c, Triple a, Triple b) {
    Number t = mul(a.t, b.d, c.modular);
    if (c.recipe == SBN3_SERIES_HYPERDESCENT)
        t = add(t, b.t, c.modular);
    else
        t = add(t, mul(c.recipe == SBN3_SERIES_COMMON_P2B3 ? a.u : a.d, b.t, c.modular), c.modular);
    return {t, mul(a.d, b.d, c.modular), mul(a.u, b.u, c.modular)};
}
static void put(sbn3_series_value &v, Number x) {
    U m = x.m < 0 ? U(-x.m) : U(x.m);
    assert(v.mantissa.capacity >= 2);
    v.mantissa.data[0] = uint64_t(m);
    v.mantissa.data[1] = uint64_t(m >> 64);
    v.mantissa.size = m >> 64 ? 2 : m ? 1 : 0;
    v.mantissa.negative = x.m < 0;
    v.exponent2 = x.e;
}
static Number read(const sbn3_series_value &v) {
    U m = 0;
    for (size_t j = v.mantissa.size; j-- > 0;)
        m = (m << 64) | v.mantissa.data[j];
    return {v.mantissa.negative ? -I(m) : I(m), v.exponent2};
}
static Triple read(const sbn3_series_values &v) {
    return {read(v.value[0]), read(v.value[1]), read(v.value[2])};
}
static void put(sbn3_series_values &v, Triple x, unsigned n) {
    if (n & 1)
        put(v.value[0], x.t);
    if (n & 2)
        put(v.value[1], x.d);
    if (n & 4)
        put(v.value[2], x.u);
}
static void touch(Context &c, const sbn3_series_stage *s, void *p, size_t n, sbn3_team_scope *scope) {
    assert(s->workers == sbn3_team_width(scope));
    assert(!(uintptr_t(p) & 255));
    assert(uintptr_t(p) >= c.scratch_begin && uintptr_t(p) + n <= c.scratch_end);
    memset(p, int(s->index & 255), n);
    c.calls.fetch_add(1, std::memory_order_relaxed);
    size_t end = uintptr_t(p) + n - c.scratch_begin, old = c.last_end.load();
    while (old < end && !c.last_end.compare_exchange_weak(old, end)) {
    }
}
static void leaf(void *p, const sbn3_series_stage *s, sbn3_series_range r, unsigned n, sbn3_series_values *v,
                 void *scratch, size_t bytes, sbn3_team_scope *scope) {
    auto &c = *static_cast<Context *>(p);
    assert(r.end - r.begin <= s->max_terms);
    if (sbn3_team_first_worker(scope) == 0) {
        auto old = c.owner_min.load();
        while (r.begin < old && !c.owner_min.compare_exchange_weak(old, r.begin)) {
        }
    }
    touch(c, s, scratch, bytes, scope);
    Triple result = term(c, r.begin);
    for (uint64_t k = r.begin + 1; k < r.end; ++k)
        result = combine(c, result, term(c, k));
    put(*v, result, n);
}
static void merge(void *p, const sbn3_series_stage *s, sbn3_series_range r, uint64_t split, unsigned n,
                  const sbn3_series_values *l, const sbn3_series_values *rr, sbn3_series_values *v,
                  void *scratch, size_t bytes, sbn3_team_scope *scope) {
    auto &c = *static_cast<Context *>(p);
    assert(r.begin < split && split < r.end);
    touch(c, s, scratch, bytes, scope);
    // Only requested dependencies are read: absent components are zero and
    // unused components of combine's return value are discarded.
    put(*v, combine(c, read(*l), read(*rr)), n);
}
static void one(Fixture &f, Context &c, uint64_t begin, uint64_t end, unsigned mask, unsigned batch,
                unsigned prefix, bool right_inline = false, bool separate_query_context = false) {
    c.queries = 0;
    c.calls = 0;
    c.last_end = 0;
    sbn3_series_spec spec{c.recipe, {begin, end}, mask, 1234, c.radix};
    sbn3_series_options options{sbn3_team_workers(f.team), batch, prefix, 0, 0};
    sbn3_series_oracle oracle{&c, bounds, work, resources};
    sbn3_series_info info{};
    struct QueryContext { uint64_t marker = 0x51554552594f4e4cULL; size_t calls = 0; } query_context;
    sbn::v3::series::ScheduleRefinement refinement{nullptr, nullptr, right_inline};
    if (separate_query_context) {
        c.self = &c;
        refinement.context = &c;
        refinement.prepared_context = &query_context;
        refinement.prepared_resources = [](const void *ptr, const sbn3_series_stage *s,
                                             sbn3_series_resources *r, sbn::v3::series::StagePreparation *) {
            auto &q = *const_cast<QueryContext *>(static_cast<const QueryContext *>(ptr));
            assert(q.marker == 0x51554552594f4e4cULL);
            ++q.calls;
            return resources(nullptr, s, r);
        };
        refinement.split_policy_id = 0x51554552594c4946ULL;
        refinement.split_point = [](const void *ptr, sbn3_series_range r, double fraction) {
            assert(static_cast<const Context *>(ptr)->self == ptr);
            const auto length = r.end - r.begin;
            return r.begin + std::clamp(uint64_t(double(length) * fraction), uint64_t(1), length - 1);
        };
    }
    allocation_watch_start();
    assert(sbn::v3::series::query_schedule(spec, options, oracle, info, refinement) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    sbn3_series_options reject = options;
    reject.workspace_budget = 1;
    sbn3_series_info need{};
    assert(sbn3_series_query(&spec, &reject, &oracle, &need) == SBN3_QUERY_CAPACITY &&
           need.workspace_bytes > 1);
    auto pl = f.allocate(info.plan_bytes, 64), scratch = f.allocate(up(info.workspace_bytes, 64) + 64, 64);
    sbn3_series_plan *plan = nullptr;
    allocation_watch_start();
    plan = sbn::v3::series::prepare_schedule(spec, options, oracle, info, pl.data, pl.bytes, refinement);
    assert(!allocation_watch_stop());
    const size_t preparation_calls = query_context.calls;
    if (separate_query_context) assert(preparation_calls > 0);
    query_context.marker = 0; // this context must never be consulted during execution
    auto prepared = f.allocate(info.prepared_bytes, info.prepared_alignment);
    memset(prepared.data, 0, prepared.bytes);
    struct Visit {
        unsigned char *data;
        size_t bytes, count;
    };
    Visit visit{static_cast<unsigned char *>(prepared.data), prepared.bytes, 0};
    allocation_watch_start();
    sbn3_series_visit(
        plan,
        [](void *arg, const sbn3_series_stage *stage, const sbn3_series_resources *r) {
            auto &v = *static_cast<Visit *>(arg);
            assert(stage->prepared_offset + r->prepared_bytes <= v.bytes);
            assert(!(stage->prepared_offset & (r->prepared_alignment - 1)));
            for (size_t j = 0; j < r->prepared_bytes; ++j) {
                auto &byte = v.data[stage->prepared_offset + j];
                assert(!byte);
                byte = 0xa5;
            }
            ++v.count;
        },
        &visit);
    assert(!allocation_watch_stop());
    assert(visit.count == info.stages);
    sbn3_series_values values{};
    for (unsigned j = 0; j < 3; ++j)
        if (mask & (1u << j)) {
            auto l = f.allocate(info.output.limbs[j] * 8, 64);
            values.value[j].mantissa = {static_cast<uint64_t *>(l.data), info.output.limbs[j], 0, 0};
        }
    c.scratch_begin = uintptr_t(scratch.data);
    c.scratch_end = c.scratch_begin + info.workspace_bytes;
    memset(static_cast<char *>(scratch.data) + info.workspace_bytes, 0x5a,
           scratch.bytes - info.workspace_bytes);
    sbn3_series_executor executor{&c, leaf, merge};
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        c.owner_min = UINT64_MAX;
        const size_t queries_before = c.queries;
        allocation_watch_start();
        sbn3_series_execute(plan, &executor, f.team, &values, scratch.data, info.workspace_bytes);
        assert(!allocation_watch_stop());
        assert(c.queries == queries_before);
        assert(query_context.calls == preparation_calls);
        if (right_inline && options.workers > 1 && !prefix)
            assert(c.owner_min >= begin + (end - begin) / 4);
        if (options.workers == 1)
            assert(c.owner_min == begin); // serial remains left before right
        Triple expected = term(c, begin);
        for (uint64_t k = begin + 1; k < end; ++k)
            expected = combine(c, expected, term(c, k));
        Number want[]{expected.t, expected.d, expected.u};
        for (unsigned j = 0; j < 3; ++j)
            if (mask & (1u << j)) {
                Number got = read(values.value[j]);
                assert(add(got, {-want[j].m, want[j].e}, false).m == 0);
            }
        for (size_t j = info.workspace_bytes; j < scratch.bytes; ++j)
            assert(static_cast<unsigned char *>(scratch.data)[j] == 0x5a);
    }
    assert(c.last_end <= info.workspace_bytes);
    printf("series recipe=%u [%llu,%llu) W%u need=%u batch=%u stages=%zu bytes=%zu highwater=%zu PASS\n",
           c.recipe, (unsigned long long)begin, (unsigned long long)end, options.workers, mask, batch,
           info.stages, info.workspace_bytes, c.last_end.load());
}
static void scaling() {
    Context c;
    c.large_bound = true;
    sbn3_series_spec s{SBN3_SERIES_COMMON_P2B3, {0, 352568359}, 3, 123, 0};
    sbn3_series_options o{16, 8, 2, 0, 0};
    sbn3_series_oracle oracle{&c, bounds, work, resources};
    sbn3_series_info i{};
    assert(sbn3_series_query(&s, &o, &oracle, &i) == SBN3_SUPPORTED);
    assert(i.stages < 4096 && c.queries < 10000 && i.plan_bytes < (1u << 20));
    const auto mathematical = i.mathematical_id;
    o.workers = 3;
    assert(sbn3_series_query(&s, &o, &oracle, &i) == SBN3_SUPPORTED && i.mathematical_id == mathematical);
    s.parameter_id = 1;
    assert(sbn3_series_query(&s, &o, &oracle, &i) == SBN3_SUPPORTED && i.mathematical_id != mathematical);
    printf("series 352M-term planning: stages=%zu metadata=%zu, no leaf enumeration PASS\n", i.stages,
           i.plan_bytes);
    o.workers = 0;
    assert(sbn3_series_query(&s, &o, &oracle, &i) == SBN3_UNSUPPORTED);
    o.workers = 16;
    auto broken = oracle;
    broken.bounds = [](const void *, sbn3_series_range, uint64_t, unsigned, sbn3_series_shape *) {
        return SBN3_QUERY_CAPACITY;
    };
    assert(sbn3_series_query(&s, &o, &broken, &i) == SBN3_QUERY_CAPACITY);
    broken = oracle;
    broken.work = [](const void *, sbn3_series_range) -> double { return NAN; };
    assert(sbn3_series_query(&s, &o, &broken, &i) == SBN3_UNSUPPORTED);
    broken = oracle;
    broken.resources = [](const void *, const sbn3_series_stage *, sbn3_series_resources *r) {
        *r = {0, 3, 1, 0, 0};
        return SBN3_SUPPORTED;
    };
    assert(sbn3_series_query(&s, &o, &broken, &i) == SBN3_UNSUPPORTED);

    // A constrained plan must actually change its concurrency schedule;
    // accepting a too-small budget or only testing rejection would miss this.
    o.max_serial_prefix = 0;
    o.workspace_budget = 0;
    sbn3_series_info unconstrained{};
    assert(sbn3_series_query(&s, &o, &oracle, &unconstrained) == SBN3_SUPPORTED);
    o.max_serial_prefix = 4;
    o.workspace_budget = unconstrained.workspace_bytes - 1;
    assert(sbn3_series_query(&s, &o, &oracle, &i) == SBN3_SUPPORTED && i.serial_prefix > 0 &&
           i.workspace_bytes <= o.workspace_budget);
    printf("series budget selects serial prefix: %zu -> %zu bytes, levels=%u PASS\n",
           unconstrained.workspace_bytes, i.workspace_bytes, i.serial_prefix);
}
static void formula_gates() {
    using namespace sbn::v3::series;
    for (uint64_t a = 1; a < 257; ++a)
        for (uint64_t b = a + 1; b < a + 37; ++b) {
            uint64_t exact = 0;
            for (uint64_t k = a; k < b; ++k)
                exact += 64 - __builtin_clzll(k);
            assert(integer_log_sum(a, b) == exact);
        }
    ref_int p, q, t, u, tmp, v, expected;
    ref_inits(p, q, t, u, tmp, v, expected, nullptr);
    for (auto kind : {FormulaKind::Euler, FormulaKind::Chudnovsky, FormulaKind::BinaryLog}) {
        Formula formula{kind, 9};
        const uint64_t first = kind == FormulaKind::Chudnovsky ? 0 : 1;
        for (uint64_t k : {first, uint64_t(1), uint64_t(17), uint64_t(65536), Formula::max_index}) {
            uint64_t storage[3][5]{};
            sbn3_series_values x{};
            for (unsigned j = 0; j < 3; ++j)
                x.value[j].mantissa = {storage[j], 5, 0, 0};
            unsigned mask = kind == FormulaKind::Chudnovsky ? 7 : 3;
            formula.leaf(k, mask, x);
            ref_set_ui(q, k);
            ref_set_ui(t, 1);
            ref_set_ui(u, 1);
            if (kind == FormulaKind::Chudnovsky) {
                if (!k) {
                    ref_set_ui(q, 1);
                    ref_set_ui(t, 13591409);
                } else {
                    ref_set_ui(u, 6 * k - 5);
                    ref_mul_ui(u, u, 2 * k - 1);
                    ref_mul_ui(u, u, 6 * k - 1);
                    ref_mul(q, q, q);
                    ref_mul_ui(q, q, k);
                    ref_mul_ui(q, q, 10939058860032000ULL);
                    ref_set_ui(t, k);
                    ref_mul_ui(t, t, 545140134);
                    ref_add_ui(t, t, 13591409);
                    ref_mul(t, t, u);
                    if (k & 1)
                        ref_neg(t, t);
                }
            }
            const ref_number *want[]{t, q, u};
            sbn3_series_shape shape{};
            assert(formula.bounds({k, k + 1}, 1, mask, shape) == SBN3_SUPPORTED);
            for (unsigned j = 0; j < 3; ++j)
                if (mask & (1u << j)) {
                    ref_import(v, x.value[j].mantissa.size, -1, 8, 0, 0, storage[j]);
                    if (x.value[j].mantissa.negative)
                        ref_neg(v, v);
                    assert(ref_cmp(v, want[j]) == 0 && x.value[j].mantissa.size <= shape.limbs[j]);
                }
            assert(x.value[0].exponent2 == (kind == FormulaKind::BinaryLog ? -int64_t(9 * k) : 0));
        }
        // Exact independent small BSR recurrence; compare every subinterval's
        // actual T,D,U bit lengths with the "any subrange" envelope contract.
        for (uint64_t a = first; a < 16; ++a) {
            ref_set_ui(t, 0);
            ref_set_ui(q, 1);
            ref_set_ui(u, 1);
            const uint64_t end = a + 17;
            for (uint64_t k = a; k < end; ++k) {
                uint64_t storage[3][5]{};
                sbn3_series_values x{};
                for (unsigned j = 0; j < 3; ++j)
                    x.value[j].mantissa = {storage[j], 5, 0, 0};
                formula.leaf(k, kind == FormulaKind::Chudnovsky ? 7 : 3, x);
                ref_import(p, x.value[0].mantissa.size, -1, 8, 0, 0, storage[0]);
                if (x.value[0].mantissa.negative)
                    ref_neg(p, p);
                ref_import(v, x.value[1].mantissa.size, -1, 8, 0, 0, storage[1]);
                ref_mul(t, t, v);
                if (kind == FormulaKind::BinaryLog) {
                    if (k > a)
                        ref_mul_2exp(t, t, formula.radix_bits);
                    ref_mul(p, p, q);
                } else if (kind == FormulaKind::Chudnovsky)
                    ref_mul(p, p, u);
                ref_add(t, t, p);
                ref_mul(q, q, v);
                if (kind == FormulaKind::Chudnovsky) {
                    ref_import(v, x.value[2].mantissa.size, -1, 8, 0, 0, storage[2]);
                    ref_mul(u, u, v);
                }
                sbn3_series_shape shape{};
                const unsigned need = kind == FormulaKind::Chudnovsky ? 7 : 3;
                assert(formula.bounds({first, end}, k - a + 1, need, shape) == SBN3_SUPPORTED);
                assert(ref_sizeinbase(t, 2) <= 64 * shape.limbs[0] &&
                       ref_sizeinbase(q, 2) <= 64 * shape.limbs[1]);
                if (need & 4)
                    assert(ref_sizeinbase(u, 2) <= 64 * shape.limbs[2]);
            }
        }
    }
    ref_clears(p, q, t, u, tmp, v, expected, nullptr);
    puts("Euler/Chudnovsky/BinaryLog exact wide leaves, integer bit bounds, exponent gates PASS");
}
int main() {
    formula_gates();
    scaling();
    {
        Fixture f(3, false);
        Context c;
        c.modular = true;
        one(f, c, 1, 4098, 3, 8, 0, false, true);
    }
    for (unsigned w : {1u, 3u, 16u}) {
        Fixture f(w, false);
        Context c;
        for (auto recipe : {SBN3_SERIES_HYPERDESCENT, SBN3_SERIES_COMMON_P2B3, SBN3_SERIES_BINARY_BBP}) {
            c.recipe = recipe;
            for (unsigned mask = 1; mask <= (recipe == SBN3_SERIES_COMMON_P2B3 ? 7u : 3u); ++mask)
                one(f, c, 2, 10, mask, 3, 2);
            for (unsigned r : {1u, 3u, 9u}) {
                c.radix = r;
                one(f, c, 1, 8, 3, 1, 0);
            }
            c.modular = true;
            one(f, c, 17, 4114, 3, 7, 0, true);
            one(f, c, 17, 100020, 3, 7, 2);
            c.modular = false;
        }
    }
    puts("generic series schedule/signed/exponent/needs/no-allocation gates PASS");
}
