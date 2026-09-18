#include "product/backend.hpp"
#include "product/cost_model.hpp"
#include "common/checked.hpp"
#include "common/small_checks.h"
using namespace sbn::v3;
const Backend *sbn::v3::backend_lookup(uint64_t id) noexcept {
    switch (id) {
    case 100:
        return &short_backend();
    case 101:
        return &fft_backend();
    case 4:
        return &np4_backend();
    case 5:
        return &np5_backend();
    case 6:
        return &np6_backend();
    case 7:
        return &np7_backend();
    case 8:
        return &np8_backend();
    case 9:
        return &np9_backend();
    case 10:
        return &np10_backend();
    default:
        return nullptr;
    }
}
static const Backend *lookup(uint64_t id) {return backend_lookup(id);}
using cost_model::deep_score;
static const sbn3_mul_info &mul_info(const sbn3_mul_info &i) {
    return i;
}
static const sbn3_mul_info &mul_info(const sbn3_product_info &i) {
    return i.mul;
}
static unsigned operation_kind(const sbn3_mul_info &) {
    return SBN3_PRODUCT_MUL;
}
static unsigned operation_kind(const sbn3_product_info &i) {
    return i.kind;
}
static bool cyclic_product(const sbn3_mul_info &) {
    return false;
}
static bool cyclic_product(const sbn3_product_info &i) {
    return i.cyclic_limbs != 0;
}
// Exact geometry and cached bases retain their constraints. Work counts use
// both operand lengths; unbalanced shapes must not fall into a padded deep tower.
using cost_model::small_domain;
template <class Info, class Query>
static sbn3_query_result select_plan(const sbn3_product_spec &s, const sbn3_mul_options &opt,
                                     sbn3_mul_plan &plan, Info &info, Query query, bool small) {
    bool found = false;
    double best = 0;
    auto rejection = SBN3_UNSUPPORTED;
    auto consider = [&](unsigned np, const sbn3_mul_options &options) {
        sbn3_mul_plan p{};
        Info ci{};
        const auto result = query(*lookup(np), options, p, ci);
        const auto &mi = mul_info(ci);
        if (result == SBN3_SUPPORTED) {
            const double cost = small ? cost_model::small_ntt(mi, operation_kind(ci) == SBN3_PRODUCT_MUL).nanoseconds
                                      : deep_score(mi, cyclic_product(ci));
            if (!found || cost < best) {
                found = true;
                best = cost;
                plan = p;
                info = ci;
            }
        } else if (!found && result == SBN3_QUERY_CAPACITY) {
            rejection = result;
            if (mi.workspace_bytes &&
                (!mul_info(info).workspace_bytes || mi.workspace_bytes < mul_info(info).workspace_bytes))
                info = ci;
        }
    };
    // NP9/10 admission is measured at W32 (actual 5B consumer shapes and
    // full-constant validation). Keep the other calibrated width domains.
    const unsigned first = opt.prime_count ? opt.prime_count : 4,
                   last = opt.prime_count ? opt.prime_count : (!small && opt.workers == 32 ? 10 : 8);
    if (!lookup(first) || !lookup(last))
        return SBN3_UNSUPPORTED;
    if (!small && opt.prime_count)
        return query(*lookup(first), opt, plan, info);
    for (unsigned np = first; np <= last; ++np) {
        consider(np, opt);
        if (!small)
            continue;
        // Obtain the widest certified digit independently of the workspace
        // filter. Every candidate is then checked against the original budget.
        auto seed = opt;
        seed.workspace_budget = 0;
        sbn3_mul_plan unused{};
        Info si{};
        if (query(*lookup(np), seed, unused, si) != SBN3_SUPPORTED)
            continue;
        const int widest = mul_info(si).trunk_bits;
        const int preferred = np == 4 ? 80 : 24 * int(np) - 16;
        const int digits[] = {widest, preferred, np == 4 ? 84 : widest};
        for (unsigned d = 0; d < 3; ++d) {
            const int T = digits[d];
            bool seen = false;
            for (unsigned j = 0; j < d; ++j)
                seen |= digits[j] == T;
            if (seen || T > widest || (opt.trunk_bits && T != opt.trunk_bits))
                continue;
            auto candidate = opt;
            candidate.trunk_bits = T;
            if (opt.algorithm != SBN3_MUL_BAILEY &&
                (std::max(s.a_limbs, s.b_limbs) <= (size_t(1) << 19) || opt.algorithm == SBN3_MUL_FLAT)) {
                candidate.algorithm = SBN3_MUL_FLAT;
                consider(np, candidate);
            }
            if (opt.algorithm == SBN3_MUL_FLAT)
                continue;
            candidate.algorithm = SBN3_MUL_BAILEY;
            const size_t need = (64 * s.a_limbs + T - 1) / T + (64 * s.b_limbs + T - 1) / T - 1;
            for (unsigned cl = 6; cl <= 11; ++cl) {
                const size_t rows = (need + (size_t(8) << cl) - 1) / (size_t(8) << cl);
                unsigned rl = 0;
                while ((size_t(1) << rl) < rows)
                    ++rl;
                if (rl < 4 || rl > 11)
                    continue;
                candidate.column_log2 = cl;
                candidate.row_log2 = rl;
                consider(np, candidate);
            }
        }
    }
    // Wide digits are useful where they keep the flat transform one level
    // smaller. This is an additional legal candidate, never a forced switch.
    if (small && !opt.prime_count && !opt.trunk_bits && opt.algorithm != SBN3_MUL_BAILEY &&
        std::max(s.a_limbs, s.b_limbs) <= (size_t(1) << 19)) {
        auto candidate = opt;
        candidate.algorithm = SBN3_MUL_FLAT;
        consider(10, candidate);
    }
    return found ? SBN3_SUPPORTED : rejection;
}
static bool short_options(const sbn3_mul_options &o) {
    return !o.prime_count && !o.prime_batch && !o.trunk_bits && !o.column_log2 && !o.row_log2 &&
           !o.crt_mode && !o.codec_mode && !o.fused_start_skew_us;
}
static bool native_available() {
    return __builtin_cpu_supports("avx512ifma") && __builtin_cpu_supports("avx512vbmi") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512cd") &&
           __builtin_cpu_supports("avx512vbmi2") && __builtin_cpu_supports("avx512vpopcntdq") &&
           __builtin_cpu_supports("pclmul") && __builtin_cpu_supports("vpclmulqdq");
}
template <class Info, class Query>
static sbn3_query_result select_family(const sbn3_product_spec &s, const sbn3_mul_options &opt,
                                       sbn3_mul_plan &plan, Info &info, Query query) {
    if (opt.algorithm >= SBN3_MUL_SCALAR)
        return query(short_backend(), opt, plan, info);
    const size_t lo = std::min(s.a_limbs, s.b_limbs), hi = std::max(s.a_limbs, s.b_limbs);
    if (opt.algorithm || !short_options(opt))
        return select_plan(s, opt, plan, info, query, small_domain(s, opt));
    if (lo <= 1 || hi <= 8) {
        auto o = opt;
        o.algorithm = SBN3_MUL_SCALAR;
        return query(short_backend(), o, plan, info);
    }
    bool found = false;
    double best = 0;
    auto rejection = SBN3_UNSUPPORTED;
    auto consider = [&](sbn3_query_result status, const sbn3_mul_plan &p, const Info &candidate,
                        double cost) {
        if (status == SBN3_SUPPORTED) {
            if (!found || cost < best) {
                found = true;
                best = cost;
                plan = p;
                info = candidate;
            }
        } else if (!found && status == SBN3_QUERY_CAPACITY) {
            rejection = status;
            const auto &i = mul_info(candidate);
            if (i.workspace_bytes &&
                (!mul_info(info).workspace_bytes || i.workspace_bytes < mul_info(info).workspace_bytes))
                info = candidate;
        }
    };
    for (unsigned alg : {unsigned(SBN3_MUL_SCALAR), unsigned(SBN3_MUL_U52), unsigned(SBN3_MUL_PQ16)}) {
        // Scalar remains a real low-memory candidate; do not enter a target
        // kernel or change algorithms at execution time on budget rejection.
        if (alg == SBN3_MUL_SCALAR && lo > 8 && hi > 256 && !opt.workspace_budget)
            continue;
        auto o = opt;
        o.algorithm = alg;
        sbn3_mul_plan p{};
        Info ci{};
        const auto status = query(short_backend(), o, p, ci);
        double cost = 0;
        if (status == SBN3_SUPPORTED) {
            const auto &i = mul_info(ci);
            cost = alg == SBN3_MUL_SCALAR ? scalar_product_cost(s.a_limbs, s.b_limbs, opt.workers)
                   : alg == SBN3_MUL_U52  ? u52_product_cost(s.a_limbs, s.b_limbs, opt.workers)
                                          : pq16_product_cost(i, s.a_limbs, s.b_limbs);
        }
        consider(status, p, ci, cost);
    }
    if (hi >= 1024) {
        auto o = opt;
        if (hi < (1u << 15))
            o.algorithm = SBN3_MUL_FLAT;
        sbn3_mul_plan p{};
        Info ci{};
        const bool small = small_domain(s, o);
        const auto status = select_plan(s, o, p, ci, query, small);
        double cost = 0;
        if (status == SBN3_SUPPORTED) {
            const auto &i = mul_info(ci);
            cost = hi < (1u << 15) ? short_flat_cost(i)
                   : small
                       ? cost_model::small_ntt(i, operation_kind(ci) == SBN3_PRODUCT_MUL).nanoseconds
                       : deep_score(i) * native_policy::deep_ns_per_work * 16 / std::min(opt.workers, 16u);
        }
        consider(status, p, ci, cost);
    }
    return found ? SBN3_SUPPORTED : rejection;
}
extern "C" sbn3_query_result sbn3_mul_query(const sbn3_product_spec *s, const sbn3_mul_options *o,
                                            sbn3_mul_plan *p, sbn3_mul_info *i) {
    require(s && p && i, SBN3_FATAL_ARGUMENT, "mul query arguments");
    *i = {};
    if (!native_available())
        return SBN3_UNSUPPORTED;
    const sbn3_mul_options defaults{1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    const auto &opt = o ? *o : defaults;
    return select_family(*s, opt, *p, *i,
                         [&](const Backend &b, const sbn3_mul_options &co, sbn3_mul_plan &cp,
                             sbn3_mul_info &ci) { return b.query(*s, co, cp, ci); });
}
extern "C" void sbn3_mul_bind(const sbn3_mul_plan *p, sbn3_arena *a, const sbn3_lease *t, const sbn3_lease *w,
                              sbn3_team *team, sbn3_mul_binding **out) {
    require(p && a && t && w && team && out, SBN3_FATAL_ARGUMENT, "mul bind arguments");
    const auto *b = lookup(p->opaque[1]);
    require(b, SBN3_FATAL_ARGUMENT, "mul plan backend");
    b->bind(*p, *a, *t, *w, *team, out);
}
static void binding_valid(const sbn3_mul_binding *b) {
    require(b && b->marker == 0x53424e3342494e44ULL, SBN3_FATAL_LIFETIME, "mul binding identity");
    // Short products must not pay seven unrelated backend calls per use.
    // Still compare registered addresses before dereferencing the vtable.
    if (b->backend == &short_backend() || b->backend == &fft_backend())
        return;
    for (unsigned np = 4; np <= 10; ++np)
        if (b->backend == lookup(np))
            return;
    fatal(SBN3_FATAL_LIFETIME, "mul binding backend");
}
extern "C" void sbn3_mul_execute(sbn3_mul_binding *b, sbn3_const_limbs a, sbn3_const_limbs y, sbn3_limbs r) {
    if constexpr (SBN3_CHECK_SMALL) {
        binding_valid(b);
        b->backend->execute(b, a, y, r);
    } else
        b->backend->execute(b, a, y, r);
}
extern "C" void sbn3_mul_execute_ptrs(sbn3_mul_binding *b, const uint64_t *a, const uint64_t *y,
                                      uint64_t *r) {
    if constexpr (SBN3_CHECK_SMALL)
        binding_valid(b);
    b->execute_fast(b, a, y, r);
}
extern "C" void sbn3_mul_execute_on_scope(sbn3_mul_binding *b, sbn3_team_scope *s, sbn3_const_limbs a,
                                          sbn3_const_limbs y, sbn3_limbs r) {
    if constexpr (SBN3_CHECK_SMALL) {
        binding_valid(b);
        require(s, SBN3_FATAL_ARGUMENT, "mul scope");
    }
    b->backend->execute_scope(b, s, a, y, r);
}
extern "C" void sbn3_mul_get_metrics(const sbn3_mul_binding *b, sbn3_mul_metrics *m) {
    binding_valid(b);
    require(m, SBN3_FATAL_ARGUMENT, "mul metrics");
    b->backend->metrics(b, *m);
}
extern "C" void sbn3_mul_unbind(sbn3_mul_binding *b) {
    binding_valid(b);
    b->backend->unbind(b);
}
extern "C" sbn3_query_result sbn3_product_query(const sbn3_product_request *s, const sbn3_mul_options *o,
                                                sbn3_mul_plan *p, sbn3_product_info *i) {
    require(s && p && i, SBN3_FATAL_ARGUMENT, "product query arguments");
    *i = {};
    const sbn3_mul_options defaults{1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    auto opt = o ? *o : defaults;
    if (!native_available())
        return SBN3_UNSUPPORTED;
    if (s->cyclic_limbs && opt.algorithm == SBN3_MUL_SCALAR)
        return short_backend().product_query(*s, opt, *p, *i);
    if (s->negacyclic)
        return fft_backend().product_query(*s, opt, *p, *i);
    if (s->window_limbs && !s->cyclic_limbs && s->kind != SBN3_PRODUCT_LOW && s->kind != SBN3_PRODUCT_HIGH)
        return SBN3_UNSUPPORTED;
    const bool small_window = s->kind == SBN3_PRODUCT_LOW || s->kind == SBN3_PRODUCT_HIGH;
    if (small_window || (s->kind == SBN3_PRODUCT_TMP && !s->cached_a[0] && !s->cached_a[1] && s->a_limbs &&
                         s->a_limbs <= 300 && s->b_limbs <= 8192 &&
                         (opt.algorithm == SBN3_MUL_U52 || (!opt.algorithm && short_options(opt))))) {
        if (!opt.algorithm)
            opt.algorithm = small_window ? SBN3_MUL_SCALAR : SBN3_MUL_U52;
        return short_backend().product_query(*s, opt, *p, *i);
    }
    if ((s->cached_a[0] && s->cached_a[0]->backend_id == 101) ||
        (s->cached_a[1] && s->cached_a[1]->backend_id == 101))
        return fft_backend().product_query(*s, opt, *p, *i);
    if (opt.algorithm == SBN3_MUL_PQ16 && (s->cyclic_limbs || s->kind == SBN3_PRODUCT_TMP))
        return fft_backend().product_query(*s, opt, *p, *i);
    for (auto *d : s->cached_a)
        if (d) {
            if (opt.prime_count && opt.prime_count != d->np)
                return SBN3_UNSUPPORTED;
            opt.prime_count = d->np;
        }
    if (!s->cyclic_limbs && !s->cached_a[0] && !s->cached_a[1] &&
        (s->kind == SBN3_PRODUCT_MUL || s->kind == SBN3_PRODUCT_SQR)) {
        const sbn3_product_spec spec{s->a_limbs, s->kind == SBN3_PRODUCT_SQR ? s->a_limbs : s->b_limbs};
        return select_family(spec, opt, *p, *i,
                             [&](const Backend &b, const sbn3_mul_options &co, sbn3_mul_plan &cp,
                                 sbn3_product_info &ci) { return b.product_query(*s, co, cp, ci); });
    }
    const sbn3_product_spec spec{s->a_limbs, s->kind == SBN3_PRODUCT_SQR ? s->a_limbs : s->b_limbs};
    const bool small = !s->cyclic_limbs && s->kind == SBN3_PRODUCT_MUL && !s->cached_a[0] &&
                       !s->cached_a[1] && small_domain(spec, opt);
    const auto status = select_plan(
        spec, opt, *p, *i,
        [&](const Backend &b, const sbn3_mul_options &co, sbn3_mul_plan &cp, sbn3_product_info &ci) {
            return b.product_query(*s, co, cp, ci);
        },
        small);
    if (!s->cached_a[0] && !s->cached_a[1] && !opt.algorithm && short_options(opt) &&
        (s->cyclic_limbs || s->kind == SBN3_PRODUCT_TMP)) {
        auto candidate = opt;
        candidate.algorithm = SBN3_MUL_PQ16;
        sbn3_mul_plan fp{};
        sbn3_product_info fi{};
        const auto fs = fft_backend().product_query(*s, candidate, fp, fi);
        if (fs == SBN3_SUPPORTED &&
            (status != SBN3_SUPPORTED ||
             pq16_product_cost(fi.mul, spec.a_limbs, spec.b_limbs) < cost_model::small_ntt(i->mul, false).nanoseconds)) {
            *p = fp;
            *i = fi;
            return SBN3_SUPPORTED;
        }
        if (status == SBN3_UNSUPPORTED && fs == SBN3_QUERY_CAPACITY) {
            *i = fi;
            return fs;
        }
    }
    return status;
}
static void spectrum_valid(const sbn3_spectrum *s) {
    bool registered = s && s->backend == &fft_backend();
    if (s)
        for (unsigned np = 4; np <= 10; ++np)
            registered |= s->backend == lookup(np);
    require(s && registered && s->marker == 0x53424e3353504543ULL, SBN3_FATAL_LIFETIME, "spectrum identity");
}
extern "C" void sbn3_product_bind(const sbn3_mul_plan *p, sbn3_arena *a, const sbn3_lease *t,
                                  const sbn3_lease *w, sbn3_team *team, const sbn3_spectrum *s0,
                                  const sbn3_spectrum *s1, sbn3_mul_binding **out) {
    require(p && a && t && w && team && out, SBN3_FATAL_ARGUMENT, "product bind arguments");
    if (s0)
        spectrum_valid(s0);
    if (s1)
        spectrum_valid(s1);
    const auto *b = lookup(p->opaque[1]);
    require(b, SBN3_FATAL_ARGUMENT, "product backend");
    b->product_bind(*p, *a, *t, *w, *team, s0, s1, out);
}
extern "C" void sbn3_product_execute(sbn3_mul_binding *b, const sbn3_product_inputs *in, sbn3_limbs out) {
    if constexpr (SBN3_CHECK_SMALL) {
        binding_valid(b);
        require(in, SBN3_FATAL_ARGUMENT, "product inputs");
    }
    b->backend->product_execute(b, nullptr, *in, out);
}
extern "C" void sbn3_product_execute_on_scope(sbn3_mul_binding *b, sbn3_team_scope *s,
                                              const sbn3_product_inputs *in, sbn3_limbs out) {
    if constexpr (SBN3_CHECK_SMALL) {
        binding_valid(b);
        require(s && in, SBN3_FATAL_ARGUMENT, "product scope arguments");
    }
    b->backend->product_execute(b, s, *in, out);
}
extern "C" void sbn3_product_get_metrics(const sbn3_mul_binding *b, sbn3_product_metrics *m) {
    binding_valid(b);
    require(m, SBN3_FATAL_ARGUMENT, "product metrics");
    b->backend->product_metrics(b, *m);
}
extern "C" void sbn3_spectrum_prepare(sbn3_mul_binding *b, sbn3_const_limbs a,
                                      sbn3_spectrum_frontier frontier, uint64_t generation, sbn3_arena *arena,
                                      const sbn3_lease *storage, sbn3_spectrum **out) {
    binding_valid(b);
    require(arena && storage && out && b->backend->prepare, SBN3_FATAL_ARGUMENT,
            "spectrum prepare unsupported/arguments");
    b->backend->prepare(b, a, frontier, generation, *arena, *storage, out);
}
extern "C" void sbn3_spectrum_describe(const sbn3_spectrum *s, sbn3_spectrum_desc *d) {
    spectrum_valid(s);
    require(d, SBN3_FATAL_ARGUMENT, "spectrum description");
    s->backend->describe(s, *d);
}
extern "C" sbn3_query_result sbn3_spectrum_query(const sbn3_mul_plan *p, sbn3_spectrum_frontier frontier,
                                                 uint64_t generation, sbn3_spectrum_desc *d) {
    require(p && d, SBN3_FATAL_ARGUMENT, "spectrum query arguments");
    const auto *backend = lookup(p->opaque[1]);
    require(backend, SBN3_FATAL_ARGUMENT, "spectrum query backend");
    if (!backend->spectrum_query)
        return SBN3_UNSUPPORTED;
    return backend->spectrum_query(*p, unsigned(frontier), generation, *d);
}
extern "C" void sbn3_spectrum_reserve(sbn3_mul_binding *b, sbn3_spectrum_frontier frontier,
                                      uint64_t generation, sbn3_arena *a, const sbn3_lease *storage,
                                      sbn3_spectrum **out) {
    binding_valid(b);
    require(a && storage && out && b->backend->spectrum_reserve, SBN3_FATAL_ARGUMENT,
            "spectrum reserve arguments/backend");
    b->backend->spectrum_reserve(b, unsigned(frontier), generation, *a, *storage, out);
}
extern "C" void sbn3_spectrum_compute(sbn3_mul_binding *b, sbn3_spectrum *s, sbn3_const_limbs a) {
    binding_valid(b);
    spectrum_valid(s);
    require((b->backend == s->backend || (b->backend == &short_backend() && s->backend == &fft_backend())) &&
                b->backend->spectrum_compute,
            SBN3_FATAL_ARGUMENT, "spectrum compute backend");
    b->backend->spectrum_compute(b, s, a);
}
extern "C" void sbn3_spectrum_compute_square(sbn3_mul_binding *b, sbn3_spectrum *s, sbn3_const_limbs a,
                                             sbn3_limbs out) {
    binding_valid(b);
    spectrum_valid(s);
    require(b->backend == s->backend && b->backend->spectrum_square, SBN3_FATAL_ARGUMENT,
            "fused spectrum square backend");
    b->backend->spectrum_square(b, s, a, out);
}
extern "C" void sbn3_spectrum_reserve_plan(const sbn3_mul_plan *p, sbn3_spectrum_frontier frontier,
                                           uint64_t generation, sbn3_arena *a, const sbn3_lease *storage,
                                           sbn3_spectrum **out) {
    require(p && a && storage && out, SBN3_FATAL_ARGUMENT, "spectrum reserve-plan arguments");
    const auto *backend = lookup(p->opaque[1]);
    require(backend && backend->spectrum_reserve_plan, SBN3_FATAL_ARGUMENT, "spectrum reserve-plan backend");
    backend->spectrum_reserve_plan(*p, unsigned(frontier), generation, *a, *storage, out);
}
extern "C" int sbn3_spectrum_can_apply(const sbn3_mul_plan *p, const sbn3_spectrum *s, unsigned term) {
    require(p, SBN3_FATAL_ARGUMENT, "spectrum plan");
    spectrum_valid(s);
    return p->opaque[1] == s->backend->id && s->backend->can_apply(*p, s, term);
}
extern "C" void sbn3_spectrum_retain(const sbn3_spectrum *s) {
    spectrum_valid(s);
    s->backend->spectrum_retain(s);
}
extern "C" void sbn3_spectrum_release(const sbn3_spectrum *s) {
    spectrum_valid(s);
    s->backend->spectrum_release(s);
}

extern "C" void sbn3_int_mul_execute(sbn3_mul_binding *b, sbn3_int *out, sbn3_int_view a, sbn3_int_view y) {
    require(out && a.negative <= 1 && y.negative <= 1, SBN3_FATAL_ARGUMENT, "integer product descriptors");
    size_t n = 0;
    require(add_size(a.size, y.size, n), SBN3_FATAL_SIZE, "integer product length");
    require(!overlaps(out, sizeof *out, out->data, bytes_for(out->capacity, 8)) &&
                !overlaps(out, sizeof *out, a.data, bytes_for(a.size, 8)) &&
                !overlaps(out, sizeof *out, y.data, bytes_for(y.size, 8)),
            SBN3_FATAL_ARGUMENT, "integer descriptor alias");
    sbn3_mul_execute(b, {a.data, a.size}, {y.data, y.size}, {out->data, out->capacity});
    while (n && !out->data[n - 1])
        --n;
    out->size = n;
    out->negative = n ? (a.negative ^ y.negative) : 0;
}
