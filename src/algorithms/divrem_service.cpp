// Exact integer division service: word/schoolbook for small shapes, block
// Barrett with a Newton block inverse and cached spectra otherwise.
// Derivations: experiments/docs/divrem-design-2026-09-18.md.
#include "sbn3/divrem.h"
#include "algorithms/divrem_tuning.hpp"
#include "algorithms/newton_tuning.hpp"
#include "algorithms/newton_limits.hpp"
#include "algorithms/local_inverse.hpp"
#include "common/checked.hpp"
#include "common/identity.hpp"
#include "product/cost_model.hpp"
#include "product/backend.hpp"
#include "product/root_prepare_cost.hpp"
#include "runtime/team.hpp"
#include "runtime/scratch.hpp"
#include "value/divrem_words.hpp"
#include "value/limbs.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <cmath>
#include <new>
#include <string.h>
#include <time.h>
namespace sbn::v3 {
namespace {
constexpr uint64_t magic = 0x53424e3344495652ULL; // "SBN3DIVR"
constexpr size_t page = 4096;
struct ProductChoice {
    unsigned np = 0, algorithm = 0, workers = 1;
    int T = 0;
    size_t ring = 0;
    unsigned cached = 0;
};
struct Plan {
    uint64_t marker = magic, seal = 0;
    sbn3_divrem_request request{};
    sbn3_divrem_options options{};
    sbn3_divrem_info info{};
    size_t in = 0, ring = 0, xlen = 0;
    size_t head = 0; // longest short head block served by word division
    ProductChoice u{}, t{};
    // Layout, byte offsets from the binding's range start.
    size_t persistent_offset = 0, d_offset = 0, dn_offset = 0, u_offset = 0, uspec_offset = 0, dspec_offset = 0;
    size_t utab_offset = 0, uwork_offset = 0, ttab_offset = 0, twork_offset = 0;
    size_t shared_offset = 0, newton_bytes = 0, newton_alignment = 0;
    size_t xbuf_offset = 0, tbuf_offset = 0, ubuf_offset = 0, pbuf_offset = 0, qbuf_offset = 0, rbuf_offset = 0;
    size_t tcap = 0, pcap = 0, scratch_words = 0;
    bool shared_products = false;
};
struct Queried {
    sbn3_mul_plan producer{}, consumer{};
    sbn3_product_info producer_info{}, consumer_info{};
    sbn3_spectrum_desc future{};
    bool cached = false;
};
// The opaque plan carries the resolved recipes: bind neither searches nor
// re-queries; it only checks the seal and rebuilds the layout from them.
struct Stored {
    Plan plan{};
    Queried uq{}, tq{};
    sbn3_newton_plan nplan{};
    sbn3_newton_info ninfo{};
};
static_assert(sizeof(Stored) <= sizeof(sbn3_divrem_plan));
struct Binding {
    uint64_t marker = magic;
    Plan plan{};
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    size_t offset = 0;
    sbn3_lease control{}, persistent{}, shared{}, utab{}, uwork{}, ttab{}, twork{}, uspec_lease{}, dspec_lease{};
    uint64_t *D = nullptr, *Dn = nullptr, *U = nullptr;
    unsigned shift = 0;
    uint64_t dinv = 0; // 3/2 reciprocal of the top two normalized divisor limbs
    bool prepared = false, shared_leased = false;
    Queried uq{}, tq{};
    sbn3_mul_binding *uprod = nullptr, *tprod = nullptr;
    sbn3_spectrum *uspec = nullptr, *dspec = nullptr;
    sbn3_newton_plan nplan{};
    sbn3_newton_info ninfo{};
    uint64_t *xbuf = nullptr, *tbuf = nullptr, *ubuf = nullptr, *pbuf = nullptr, *qbuf = nullptr, *rbuf = nullptr,
             *scratch = nullptr;
    sbn3_divrem_metrics metrics{};
};
uint64_t feed(uint64_t h, uint64_t v) {
    return identity::word(h, v);
}
uint64_t seal(const Plan &p) {
    uint64_t h = feed(1469598103934665603ULL, p.marker);
    const uint64_t words[] = {p.request.numerator_limbs, p.request.denominator_limbs, p.options.workers,
                              p.options.prime_count, p.options.memory_budget, p.options.reuse_hint,
                              p.options.block_limbs, p.options.residual, p.options.algorithm, p.options.timing, p.info.algorithm, p.info.storage_bytes,
                              p.info.plan_id, p.in, p.ring, p.xlen, p.head, p.u.np, p.u.algorithm, p.u.workers,
                              uint64_t(p.u.T), p.u.ring, p.u.cached, p.t.np, p.t.algorithm, p.t.workers,
                              uint64_t(p.t.T), p.t.ring, p.t.cached, p.persistent_offset, p.shared_offset,
                              p.newton_bytes, p.newton_alignment, p.tcap, p.pcap, p.scratch_words, uint64_t(p.shared_products)};
    for (uint64_t w : words)
        h = feed(h, w);
    return h;
}
void load(const sbn3_divrem_plan &p, Stored &s) {
    memcpy(&s, p.opaque, sizeof s);
    require(s.plan.marker == magic && s.plan.seal == seal(s.plan), SBN3_FATAL_ARGUMENT, "division plan identity");
}
size_t aligned(size_t n, size_t a) {
    size_t r = 0;
    require(align_size(n, a, r), SBN3_FATAL_SIZE, "division layout alignment");
    return r;
}
uint64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
sbn3_mul_options product_options(const ProductChoice &c) {
    sbn3_mul_options o{};
    o.workers = c.workers;
    o.prime_count = c.np;
    o.trunk_bits = c.T;
    o.algorithm = c.algorithm;
    o.borrow_output = 1;
    return o;
}
// Prepare fills the reserved spectrum through the cached consumer binding, and
// the spectrum build contract demands the builder's own representation
// (support, written slots, scale) to equal the reserved one. A consumer may
// legally read a wider or differently scaled spectrum, so query acceptance of
// the consumer does not imply that it can build: compare the spectrum the
// consumer plan itself would produce with the reserved description.
bool consumer_builds(const Queried &q, uint64_t generation) {
    sbn3_spectrum_desc own{};
    if (sbn3_spectrum_query(&q.consumer, SBN3_SPECTRUM_COLUMNS, generation, &own) != SBN3_SUPPORTED)
        return false;
    const auto &f = q.future;
    return own.seal == f.seal && own.basis_id == f.basis_id && own.backend_id == f.backend_id && own.np == f.np &&
           own.trunk_bits == f.trunk_bits && own.frontier == f.frontier && own.format_version == f.format_version &&
           own.codec_mode == f.codec_mode && own.C == f.C && own.M2 == f.M2 &&
           own.transform_trunks == f.transform_trunks && own.live_slots == f.live_slots &&
           own.written_slots == f.written_slots && own.source_limbs == f.source_limbs &&
           own.source_trunks == f.source_trunks && own.block_stride == f.block_stride &&
           own.storage_bytes == f.storage_bytes && own.table_bytes == f.table_bytes &&
           own.plane_bytes == f.plane_bytes && !memcmp(own.scale, f.scale, sizeof own.scale);
}
// The consumer of a queried producer's future spectrum.
bool query_consumer(sbn3_product_request r, const sbn3_mul_options &o, uint64_t generation, Queried &q) {
    if (sbn3_spectrum_query(&q.producer, SBN3_SPECTRUM_COLUMNS, generation, &q.future) != SBN3_SUPPORTED)
        return false;
    r.cached_a[0] = &q.future;
    return sbn3_product_query(&r, &o, &q.consumer, &q.consumer_info) == SBN3_SUPPORTED;
}
bool query_cached(sbn3_product_request r, const sbn3_mul_options &o, uint64_t generation, Queried &q) {
    return sbn3_product_query(&r, &o, &q.producer, &q.producer_info) == SBN3_SUPPORTED &&
           query_consumer(r, o, generation, q);
}
// The plain product of a recipe. The cached recipe's policy producer is that
// same query (the options do not depend on caching), so the two recipes of one
// product share its search.
struct Plain {
    bool queried = false, supported = false;
    sbn3_mul_plan plan{};
    sbn3_product_info info{};
};
// Term 0 (a) is the divisor-side operand that may be cached; term 1 (b) is fresh.
bool query_product(size_t a, size_t b, const ProductChoice &c, uint64_t generation, Queried &q, Plain *plain = nullptr) {
    q = {};
    auto o = product_options(c);
    sbn3_product_request r{};
    r.kind = SBN3_PRODUCT_MUL;
    r.a_limbs = a;
    r.b_limbs = b;
    r.cyclic_limbs = c.ring;
    bool supported = false;
    if (plain && plain->queried) {
        supported = plain->supported;
        q.producer = plain->plan;
        q.producer_info = plain->info;
    } else {
        supported = sbn3_product_query(&r, &o, &q.producer, &q.producer_info) == SBN3_SUPPORTED;
        if (plain) {
            plain->queried = true;
            plain->supported = supported;
            plain->plan = q.producer;
            plain->info = q.producer_info;
        }
    }
    if (!supported)
        return false;
    if (!c.cached) {
        q.consumer = q.producer;
        q.consumer_info = q.producer_info;
        return true;
    }
    if (!query_consumer(r, o, generation, q))
        return false;
    if (!consumer_builds(q, generation)) {
        // The policy producer of a linear product may take the full transform
        // where the geometry-pinned consumer truncates (different support and
        // scale). Select one construction representation at query time: the
        // producer is re-derived at the consumer's resolved family and
        // geometry, so reservation, build and application agree. A candidate
        // that still cannot be built by its consumer is not a plan.
        const auto &i = q.consumer_info.mul;
        if (i.algorithm != SBN3_MUL_BAILEY && i.algorithm != SBN3_MUL_FLAT)
            return false;
        o.prime_count = i.np;
        o.trunk_bits = int(i.trunk_bits);
        o.algorithm = i.algorithm;
        if (i.algorithm == SBN3_MUL_BAILEY) {
            o.column_log2 = unsigned(__builtin_ctzll(i.C));
            o.row_log2 = unsigned(__builtin_ctzll(i.M2));
        }
        q = {};
        if (!query_cached(r, o, generation, q) || !consumer_builds(q, generation))
            return false;
    }
    q.cached = true;
    return true;
}
struct Cost {
    double prepare = 0, apply = 0;
};
// The FFT family's tables (its own, or those of the spectrum it reads) are built by every binding and are not in
// root_prepare_cost, which prices the NTT families' roots only (divrem_tuning).
double table_setup(const Queried &q) {
    const auto &i = q.consumer_info.mul;
    if (i.algorithm != SBN3_MUL_PQ16)
        return 0;
    return divrem_tuning::fft_table_ns_per_byte * double(i.table_bytes + (q.cached ? q.future.table_bytes : 0));
}
Cost product_cost(const Queried &q, size_t a, size_t b) {
    const auto &i = q.consumer_info.mul;
    const bool deep = std::max(a, b) > native_policy::small_model_max_words;
    const double base = q.consumer_info.cyclic_limbs ? cost_model::cyclic_product(i, deep).nanoseconds
                                                     : cost_model::linear_product(i, a, b).nanoseconds;
    Cost c;
    c.apply = q.cached ? cost_model::cached_share(base) : base;
    c.prepare = root_prepare_cost(i) + table_setup(q) + (q.cached ? cost_model::prepare_share(base) : 0);
    return c;
}
// What one policy product search costs the query that makes it (divrem_tuning).
double search_price(size_t a, size_t b) {
    const size_t hi = std::max(a, b);
    return hi < 1024               ? divrem_tuning::product_search_short_ns
           : hi < (size_t(1) << 15) ? divrem_tuning::product_search_ns
                                    : divrem_tuning::product_search_small_domain_ns;
}
// The least a transform recipe over n limbs costs for these applications on the workers it would get, one spectrum
// kept (a third of the transforms once, two thirds per application): no recipe a search can return costs less.
double transform_floor(const Plan &p, size_t n, double applications) {
    return divrem_tuning::transform_floor_ns_per_limb * double(n) * (1 + 2 * applications) / 3 /
           double(newton_rung_workers(n, p.options.workers));
}
// A search can pay when the shorter operand is long enough for a transform at all and the recipe in hand costs more
// than that floor and the search together.
bool search_pays(const Plan &p, size_t a, size_t b, double applications, double in_hand) {
    return std::min(a, b) >= divrem_tuning::transform_min_limbs &&
           in_hand - transform_floor(p, a + b, applications) > search_price(a, b);
}
// The short product (no tables, no spectrum), a recipe of its own next to the policy's. It runs on one worker
// whatever the team (10120 x 461: 117 us on one worker, 114 on eight), and its one-worker model is the measured one.
ProductChoice short_product() {
    ProductChoice c{};
    c.algorithm = SBN3_MUL_U52;
    c.workers = 1;
    return c;
}
unsigned product_workers(const Plan &p, size_t n) {
    return newton_rung_workers(n, p.options.workers);
}
// Best recipe for the U product (in+1) x in: the short product while no search can pay, else the policy's linear
// product (cached when the family supports a reserved spectrum) unless the short product's preparation and
// applications together cost less.
bool choose_u(const Plan &p, size_t in, double applications, ProductChoice &out, Queried &q, Cost &cost) {
    bool found = false;
    auto total = [&](const Cost &k) { return k.prepare + applications * k.apply; };
    if (!p.options.prime_count) {
        const ProductChoice direct = short_product();
        Queried candidate{};
        if (query_product(in + 1, in, direct, 1, candidate)) {
            found = true;
            out = direct;
            q = candidate;
            cost = product_cost(candidate, in + 1, in);
            if (!search_pays(p, in + 1, in, applications, total(cost)))
                return true;
        }
    }
    ProductChoice c{};
    c.np = p.options.prime_count;
    c.workers = product_workers(p, in + 1);
    Plain plain{};
    bool policy = false;
    ProductChoice choice{};
    Queried queried{};
    Cost best{};
    for (unsigned cached : {1u, 0u}) {
        c.cached = cached;
        Queried candidate{};
        if (!query_product(in + 1, in, c, 1, candidate, &plain))
            continue;
        const Cost k = product_cost(candidate, in + 1, in);
        if (!policy || total(k) < total(best)) {
            policy = true;
            choice = c;
            queried = candidate;
            best = k;
        }
    }
    if (policy && (!found || total(best) < total(cost))) {
        found = true;
        out = choice;
        q = queried;
        cost = best;
    }
    return found;
}
// Residual recipes for T = D' x qhat, term 0 cached where possible, ordered by
// prepare + applications * apply within their family.
struct Residual {
    bool found = false;
    double total = INFINITY;
    ProductChoice choice{};
    Queried queried{};
    Cost cost{};
    size_t ring = 0;
};
void consider(Residual &b, size_t dn, size_t in, double applications, const ProductChoice &c, size_t r,
              Plain *plain = nullptr) {
    if (r && c.np && c.cached && b.found) {
        // Same exact cyclic geometry in producer and cached consumer. The
        // only downward cost change is Flat's one-buffer factor (>= .79).
        // Discard a loser from cheap geometry metadata; survivors still
        // construct and validate both ordinary plans and the cache contract.
        sbn3_product_request request{};
        request.a_limbs=dn;request.b_limbs=in;request.cyclic_limbs=r;
        sbn3_mul_info geometry{};
        const auto *backend=backend_lookup(c.np);
        if (backend && backend->geometry_query &&
            backend->geometry_query(request,product_options(c),geometry)==SBN3_SUPPORTED) {
            const double base=cost_model::cyclic_product(geometry,std::max(dn,in)>native_policy::small_model_max_words).nanoseconds;
            const double cached=base*(geometry.algorithm==SBN3_MUL_FLAT?.79:1.);
            const double bound=root_prepare_cost(geometry)+cost_model::prepare_share(base)+
                               applications*cost_model::cached_share(cached);
            if (bound>b.total) return;
        }
    }
    Queried candidate{};
    if (!query_product(dn, in, c, 2, candidate, plain))
        return;
    const Cost k = product_cost(candidate, dn, in);
    const double total = k.prepare + applications * k.apply;
    if (!std::isfinite(total) || total >= b.total)
        return;
    b.total = total;
    b.found = true;
    b.choice = c;
    b.queried = candidate;
    b.cost = k;
    b.ring = r;
}
// Linear family: dn+in output limbs. The short product first; the policy's product is searched when that search
// can pay, and not when the short product costs no more than the policy's recipe did at a larger block size of this
// request (transform: fewer, longer blocks transform no more limbs, so that recipe's cost does not fall with the
// block size).
void linear_residual(const Plan &p, size_t dn, size_t in, double applications, double &transform, Residual &lin) {
    if (!p.options.prime_count) {
        consider(lin, dn, in, applications, short_product(), 0);
        if (lin.found && ((std::isfinite(transform) && lin.total <= transform) || !search_pays(p, dn, in, applications, lin.total)))
            return;
    }
    ProductChoice linear{};
    linear.np = p.options.prime_count;
    linear.workers = product_workers(p, dn);
    Plain plain{};
    Residual policy{};
    for (unsigned cached : {1u, 0u}) {
        linear.cached = cached;
        consider(policy, dn, in, applications, linear, 0, &plain);
    }
    if (!policy.found)
        return;
    transform = std::min(transform, policy.total);
    // The search is made for a transform recipe. Another short product of the policy's replaces nothing: between
    // short products its model compares executions of balanced operands (a 300007 x 6 product: scalar by the
    // model, 622 us measured against 197 by u52).
    const unsigned algorithm = policy.queried.consumer_info.mul.algorithm;
    const bool transformed = algorithm != SBN3_MUL_SCALAR && algorithm != SBN3_MUL_U52;
    if (!lin.found || (transformed && policy.total < lin.total))
        lin = policy;
}
// Cyclic family: the ring lattice >= dn + guard, one visit per candidate recipe.
template <class Visit> void ring_lattice(const Plan &p, size_t dn, Visit visit) {
    const size_t minimum = dn + divrem_tuning::ring_guard_words;
    const unsigned width = product_workers(p, dn);
    const unsigned first = p.options.prime_count ? p.options.prime_count : newton_limits::first_ntt_prime_count,
                   last = p.options.prime_count ? p.options.prime_count : newton_limits::last_ntt_prime_count;
    if (!p.options.prime_count && minimum <= 512)
        visit(ProductChoice{0, SBN3_MUL_SCALAR, 1, 0, minimum, 0}, minimum);
    for (unsigned np = first; np <= last; ++np)
        for (int T = np == 4 ? 88 : 24 * int(np) - 8; T >= (np == 4 ? 80 : 24 * int(np) - 32); T -= np == 4 ? 4 : 8) {
            size_t r = 2 * size_t(T);
            while (r < minimum)
                r *= 2;
            for (unsigned algorithm : {unsigned(SBN3_MUL_FLAT), unsigned(SBN3_MUL_BAILEY)}) {
                if (algorithm == SBN3_MUL_FLAT && r > (size_t(1) << 19))
                    continue;
                if (algorithm == SBN3_MUL_BAILEY && r < 2048)
                    continue;
                visit(ProductChoice{np, algorithm, width, T, r, 1}, r);
            }
        }
    if (!p.options.prime_count && minimum <= 32768)
        for (unsigned radix : {1u, 3u, 5u, 7u}) {
            size_t branch = radix == 3 ? 256 : 128;
            while (radix * branch < 2 * minimum)
                branch *= 2;
            const size_t r = radix * branch / 2;
            if (r > 32768)
                continue;
            // At one worker the two widths are one candidate (its twin ties and is not taken).
            const unsigned widths[2] = {1u, width};
            for (unsigned k = 0; k < (width > 1 ? 2u : 1u); ++k)
                visit(ProductChoice{0, SBN3_MUL_PQ16, widths[k], 16, r, 1}, r);
        }
}
// The supported rings of the lattice (the expensive part of the query).
void cyclic_residual(const Plan &p, size_t dn, size_t in, double applications, Residual &cyc) {
    ring_lattice(p, dn, [&](const ProductChoice &c, size_t r) { consider(cyc, dn, in, applications, c, r); });
}
// Its least ring, supported or not: no query.
size_t least_ring(const Plan &p, size_t dn) {
    size_t least = SIZE_MAX;
    ring_lattice(p, dn, [&](const ProductChoice &, size_t r) { least = std::min(least, r); });
    return least;
}
bool short_recipe(const Residual &t) {
    const unsigned algorithm = t.queried.consumer_info.mul.algorithm;
    return algorithm == SBN3_MUL_SCALAR || algorithm == SBN3_MUL_U52;
}
// Linear and cyclic families are ranked separately by the cost model; the
// family is then chosen by the structural ring rule (divrem_tuning).
const Residual *residual_family(const Plan &p, size_t dn, size_t in, const Residual &lin, const Residual &cyc) {
    if (p.options.residual == 1 || !cyc.found)
        return lin.found ? &lin : nullptr;
    if (!lin.found || p.options.residual == 2)
        return &cyc;
    // Cached against cached: the ring rule; otherwise (a linear recipe that keeps no spectrum, a short product,
    // the scalar ring) the cost model decides, the FFT family's linear transform at its measured bias against a
    // ring (divrem_tuning: a lattice is searched only where rings pay).
    if (cyc.queried.cached && lin.queried.cached)
        return double(cyc.ring) <= divrem_tuning::cyclic_ring_fraction * double(dn + in) ? &cyc : &lin;
    const double bias = lin.queried.consumer_info.mul.algorithm == SBN3_MUL_PQ16 ? divrem_tuning::fft_linear_order_bias : 1.0;
    return cyc.total < bias * lin.total ? &cyc : &lin;
}
// The divisor length from which the ring family pays on the block product's team for these executions (divrem_tuning).
bool ring_pays(const Plan &p, size_t dn) {
    namespace t = divrem_tuning;
    const unsigned workers = product_workers(p, dn),
                   executions = std::clamp(p.options.reuse_hint, 1u, t::cyclic_ring_execution_limit);
    double least = t::cyclic_ring_min_limbs;
    if (workers > 1)
        least *= std::pow(double(workers), t::cyclic_ring_worker_scaling);
    if (executions > 1)
        least /= std::pow(double(executions), t::cyclic_ring_execution_scaling);
    return double(dn) >= least;
}
// Whether the ring lattice is searched for a block size (divrem_tuning): from the divisor length at which the ring
// family pays on the block product's team for these executions, and where a ring can be taken at all. Between two
// recipes that keep a spectrum the family rule decides by the ring alone, and no ring passes it when the lattice's
// least one does not. A linear transform without a spectrum is decided by the cost model: searched when the least
// ring passes that rule or the share of the linear output it drops is worth the lattice's price. Against a short
// product a ring is a transform recipe like the policy's: considered from the same operand length and when the
// recipe in hand costs more than the least such transform and the search; then the lattice's least ring is queried
// alone (a few queries against about a hundred), and the lattice is searched when that ring is modelled cheaper.
bool lattice_wanted(const Plan &p, size_t dn, size_t in, double applications, const Residual &lin) {
    namespace t = divrem_tuning;
    if (p.options.residual == 1)
        return false;
    if (p.options.residual == 2 || !lin.found)
        return true;
    if (!ring_pays(p, dn))
        return false;
    const size_t ring = least_ring(p, dn);
    const double fraction = double(ring) / double(dn + in);
    if (short_recipe(lin)) {
        if (in < t::transform_min_limbs || lin.total - transform_floor(p, ring, applications) <= t::lattice_search_ns)
            return false;
        Residual least{};
        ring_lattice(p, dn, [&](const ProductChoice &c, size_t r) {
            if (r == ring)
                consider(least, dn, in, applications, c, r);
        });
        return least.found && least.total < lin.total;
    }
    if (lin.queried.cached)
        return fraction <= t::cyclic_ring_fraction;
    return fraction <= t::cyclic_ring_fraction || (1 - fraction) * lin.total > t::lattice_search_ns;
}
// Block inverse: local u52 recurrence/exact correction without a product
// planner below the shared crossover, spectral Newton above it.
bool inverse_basecase(size_t in) {
    return in <= divrem_tuning::inverse_basecase_limbs;
}
double newton_estimate(const Plan &p, size_t in) {
    ProductChoice c{};
    c.np = p.options.prime_count;
    c.workers = product_workers(p, in);
    Queried q{};
    if (!query_product(in, in, c, 0, q))
        return INFINITY;
    return divrem_tuning::inverse_cost_ratio * cost_model::linear_product(q.consumer_info.mul, in, in).nanoseconds;
}
// Ordering price of the block inverse, by the route the plan would take: the
// local arithmetic, or the spectral Newton ladder with its planning.
double inverse_estimate(const Plan &p, size_t in) {
    if (inverse_basecase(in))
        return inverse_tuning::local_cost(in);
    return divrem_tuning::inverse_newton_planning_ns + newton_estimate(p, in);
}
// One head limb by word division: an O(dn) multiply-subtract pass.
double head_step_cost(const Plan &p, size_t dn) {
    const bool parallel = p.options.workers > 1 && dn >= parallel_limbs::minimum_parallel_words;
    const double limb = !parallel                                ? divrem_tuning::head_ns_per_limb
                        : dn <= divrem_tuning::head_cache_limbs ? divrem_tuning::head_parallel_ns_per_limb
                                                                : divrem_tuning::head_memory_ns_per_limb;
    return divrem_tuning::head_step_ns + double(dn) * limb;
}
// Longest short head block worth serving by word division instead of one
// padded block's product pair.
size_t head_limit(const Plan &p, size_t dn, size_t in, double pair_ns) {
    const double limit = std::floor(divrem_tuning::head_cost_margin * pair_ns / head_step_cost(p, dn));
    return limit >= double(in - 1) ? in - 1 : limit > 0 ? size_t(limit) : 0;
}
// One block size: the searches of the order (U recipe, linear residual family)
// and, once taken, the final recipes.
struct Candidate {
    size_t in = 0, ring = 0, head = 0;
    ProductChoice u{}, t{};
    Queried uq{}, tq{};
    double total = INFINITY, inverse = 0, pair = 0; // ordering prices: whole plan, block inverse, one block's product pair
    bool searched = false; // ucost and linear hold this size's searches
    Cost ucost{};
    Residual linear{};
};
double applications(const Plan &p, size_t in) {
    const size_t qn = p.info.quotient_limbs;
    return double(std::max(1u, p.options.reuse_hint)) * double((qn + in - 1) / in);
}
// The searches both the order and the final recipe need. A request for the
// cyclic residual only (experiments) has no linear family.
bool search(const Plan &p, size_t in, Candidate &c, double &transform) {
    c = {};
    c.in = in;
    if (!choose_u(p, in, applications(p, in), c.u, c.uq, c.ucost))
        return false;
    if (p.options.residual != 2)
        linear_residual(p, p.request.denominator_limbs, in, applications(p, in), transform, c.linear);
    c.searched = true;
    return true;
}
// Block-size ordering uses the linear residual recipe only; the cyclic
// lattice (the expensive part of the query) runs for the size that is taken
// and, where the first size is left without a ring, for the sizes next to it (rank).
// The total prices whole blocks, the quotient limbs a size leaves over as one
// more product pair: the word-division head serves what the taken size leaves
// over, it is no reason to take a size (divrem_tuning).
bool order(const Plan &p, size_t in, Candidate &c, double &transform) {
    if (!search(p, in, c, transform))
        return false;
    Residual lattice{};
    const Residual *t = &c.linear;
    if (p.options.residual == 2) { // no linear family to order by
        cyclic_residual(p, p.request.denominator_limbs, in, applications(p, in), lattice);
        t = &lattice;
    }
    if (!t->found)
        return false;
    c.inverse = inverse_estimate(p, in);
    c.pair = c.ucost.apply + t->cost.apply;
    c.total = c.inverse + c.ucost.prepare + t->cost.prepare + applications(p, in) * c.pair;
    return std::isfinite(c.total);
}
// The residual recipe of a block size and the head it leaves room for.
void adopt(const Plan &p, Candidate &c, const Residual &t) {
    const size_t dn = p.request.denominator_limbs, in = c.in;
    c.t = t.choice;
    c.tq = t.queried;
    c.ring = t.ring;
    // The residual buffer bounds the head: X = R*B^head + block needs dn+head limbs.
    const size_t xlen = c.ring ? c.ring : dn + in;
    c.head = std::min(head_limit(p, dn, in, c.ucost.apply + t.cost.apply), xlen - dn);
}
// Block-size candidates in cost order; take() hands them out one at a time so
// that the query can pass over a size whose storage exceeds the budget.
struct Ranking {
    Candidate sizes[4]{};
    unsigned ordered[4]{};
    unsigned count = 0, next = 0;
    double transform = INFINITY; // least complete cost of the policy's residual recipe at the sizes searched so far
    // The ring lattice of one block size at a time, constructed when it is searched (a recipe is 8 KB; the queries
    // of short divisions never search one).
    alignas(Residual) unsigned char lattice_bytes[sizeof(Residual)];
    Residual *lattice = nullptr;
    size_t lattice_in = 0;
    Ranking() {}
};
// The ring lattice of a block size where it is wanted (divrem_tuning): searched once while the size stays the last
// one asked for, so the order's search serves the final recipe.
const Residual &lattice(const Plan &p, Ranking &r, const Candidate &c) {
    const size_t dn = p.request.denominator_limbs;
    if (r.lattice && r.lattice_in == c.in)
        return *r.lattice;
    r.lattice = ::new (r.lattice_bytes) Residual{};
    r.lattice_in = c.in;
    if (lattice_wanted(p, dn, c.in, applications(p, c.in), c.linear))
        cyclic_residual(p, dn, c.in, applications(p, c.in), *r.lattice);
    return *r.lattice;
}
// Final recipes of a block size: the order's searches are not repeated, the
// cyclic lattice joins them and the family rule decides. The order priced
// whole blocks, so the final recipe owes it no head: the plan serves by word
// division what that recipe allows (nothing when the ring leaves no room above
// the divisor or the product pair is cheaper than the word steps) and pads a
// block otherwise.
bool finish(const Plan &p, Ranking &r, Candidate &c) {
    const size_t dn = p.request.denominator_limbs, in = c.in;
    double transform = INFINITY;
    if (!c.searched && !search(p, in, c, transform))
        return false;
    const Residual *t = residual_family(p, dn, in, c.linear, lattice(p, r, c));
    if (!t)
        return false;
    adopt(p, c, *t);
    return true;
}
// Under a memory budget the recipe taken for a block size may not fit where the other residual family of the same
// size does (a ring holds less than the linear product's workspace, and the lattice is not always searched).
bool other_family(const Plan &p, Candidate &c) {
    if (p.options.residual)
        return false;
    if (c.ring) {
        if (!c.linear.found)
            return false;
        adopt(p, c, c.linear);
        return true;
    }
    Residual cyclic{};
    cyclic_residual(p, p.request.denominator_limbs, c.in, applications(p, c.in), cyclic);
    if (!cyclic.found)
        return false;
    adopt(p, c, cyclic);
    return true;
}
void order_sizes(const Plan &p, Ranking &r) {
    const size_t dn = p.request.denominator_limbs, qn = p.info.quotient_limbs;
    size_t candidates[4]{};
    unsigned count = 0;
    auto add = [&](size_t in) {
        in = std::min(in, std::min(dn, qn));
        if (!in)
            return;
        for (unsigned j = 0; j < count; ++j)
            if (candidates[j] == in)
                return;
        candidates[count++] = in;
    };
    if (p.options.block_limbs)
        add(p.options.block_limbs);
    else if (qn <= dn) {
        for (size_t b = 1; b <= 3; ++b)
            add((qn + b - 1) / b);
    } else {
        const size_t bmin = (qn + dn - 1) / dn;
        add((qn + bmin - 1) / bmin);
        add((qn + bmin) / (bmin + 1));
        // Complete dn-limb blocks (a full inverse for 2d/d) are served on
        // request (block_limbs) and are not a policy candidate: see
        // divrem_tuning.
    }
    if (count == 1) { // nothing to order: searched once, by take()
        r.sizes[0].in = candidates[0];
        r.ordered[r.count++] = 0;
        return;
    }
    auto insert = [&](size_t in) {
        Candidate &c = r.sizes[r.count];
        if (!order(p, in, c, r.transform))
            return;
        unsigned at = r.count++;
        for (; at && c.total < r.sizes[r.ordered[at - 1]].total; --at)
            r.ordered[at] = r.ordered[at - 1];
        r.ordered[at] = r.count - 1;
    };
    for (unsigned j = 0; j < count; ++j)
        insert(candidates[j]);
    // The quotient-driven sizes can miss the local-inverse optimum. With
    // inverse price a*in^p and qn/in block pairs it is near
    // (executions*qn*pair/(p*a))^(1/(p+1)). Add that whole-block size, then
    // price its actual product recipes rather than pruning by larger ones.
    if (p.options.block_limbs || !r.count || r.count == 4)
        return;
    const Candidate &best = r.sizes[r.ordered[0]];
    size_t smallest = candidates[0];
    for (unsigned j = 1; j < count; ++j)
        smallest = std::min(smallest, candidates[j]);
    const double executions = std::max(1u, p.options.reuse_hint);
    const double star = std::pow(executions * double(qn) * best.pair / (1.5 * 1.7), 1./2.5);
    if (!(star >= 1) || star >= double(smallest))
        return;
    size_t in = size_t(std::min(star, double(divrem_tuning::inverse_basecase_limbs)));
    const size_t blocks = (qn + in - 1) / in;
    in = (qn + blocks - 1) / blocks;
    // Price this candidate's own shorter products. Multiplying the larger
    // candidate's pair price by the new block count is not a lower bound:
    // it excluded profitable schoolbook-inverse plans at 2k..5k limbs.
    if (in < smallest)
        insert(in);
}
// The order prices linear recipes. Where the ring family pays, the model has the FFT family's linear recipe at
// 0.9-1.0 of a ring recipe that measured 1.11-1.35 times faster (divrem_tuning), and an order of linear recipes
// passed over a size with a ring for one more block without one. So when the first size is left with that linear
// recipe, the next sizes whose linear totals lie within the measured bias are asked for their ring in turn, and the
// first that takes one goes first. A first size that takes a ring costs the one lattice the final recipe needs.
void rank(const Plan &p, Ranking &r) {
    order_sizes(p, r);
    const size_t dn = p.request.denominator_limbs;
    if (r.count < 2 || p.options.residual || !ring_pays(p, dn))
        return;
    const Candidate &first = r.sizes[r.ordered[0]];
    if (first.linear.queried.consumer_info.mul.algorithm != SBN3_MUL_PQ16 ||
        residual_family(p, dn, first.in, first.linear, lattice(p, r, first)) != &first.linear)
        return;
    for (unsigned j = 1; j < r.count; ++j) {
        const Candidate &c = r.sizes[r.ordered[j]];
        if (c.total > divrem_tuning::fft_linear_order_bias * first.total)
            return;
        const Residual &ring = lattice(p, r, c);
        if (residual_family(p, dn, c.in, c.linear, ring) == &ring) {
            const unsigned taken = r.ordered[j];
            for (unsigned k = j; k; --k)
                r.ordered[k] = r.ordered[k - 1];
            r.ordered[0] = taken;
            return;
        }
    }
}
// Next block size with its final recipes.
Candidate *take(const Plan &p, Ranking &r) {
    while (r.next < r.count) {
        Candidate &c = r.sizes[r.ordered[r.next++]];
        if (finish(p, r, c))
            return &c;
    }
    return nullptr;
}
// One-use cost (divrem_tuning): the schoolbook work of all expected executions beyond that level against the fixed
// cost of planning and preparing a block-Barrett division.
bool schoolbook_use(const Plan &p) {
    const double qn = double(p.info.quotient_limbs), executions = std::max(1u,p.options.reuse_hint);
    const size_t dn=p.request.denominator_limbs;
    double level = double(divrem_tuning::schoolbook_level_quotient);
    // Share the extra preparation pass across repeated uses where subsequent
    // value passes stay in one CCD's cache or actually use the parallel path.
    // In the serial memory-bound regime streaming the divisor still costs
    // that pass on every execute: do not extrapolate a cached crossover there.
    if(executions>1 && (dn<=divrem_tuning::short_cache_limbs ||
                       (p.options.workers>1 && dn>=parallel_limbs::minimum_parallel_words)))
        level-=1.-1./executions;
    const double work = executions * std::max(0.0, qn - level) * double(dn);
    return work <= divrem_tuning::schoolbook_use_work;
}
sbn3_query_result inverse_query(size_t in, const sbn3_newton_options &o, sbn3_newton_plan &plan, sbn3_newton_info &info) {
    if (!inverse_basecase(in))
        return sbn3_newton_query(SBN3_NEWTON_INVERSE, in, &o, &plan, &info);
    plan = {};
    info = {};
    info.kind = SBN3_NEWTON_INVERSE;
    info.precision_limbs = in;
    info.workers = 1;
    info.storage_bytes = local_inverse_bytes(in);
    info.storage_alignment = page;
    info.plan_id = feed(feed(1469598103934665603ULL, magic), in);
    return SBN3_SUPPORTED;
}
sbn3_newton_options newton_options(const Plan &p) {
    sbn3_newton_options o{};
    o.workers = p.options.workers;
    o.prime_count = p.options.prime_count;
    return o;
}
// Complete storage layout from the queried recipes. Returns false on size overflow.
bool layout(Plan &p, const Queried &uq, const Queried &tq, const sbn3_newton_info *ninfo) {
    auto &i = p.info;
    const size_t dn = p.request.denominator_limbs, nn = p.request.numerator_limbs, in = p.in;
    size_t cursor = aligned(sizeof(Binding), 128);
    i.control_bytes = cursor;
    i.table_bytes = i.product_workspace_bytes = i.spectrum_bytes = i.scratch_bytes = 0;
    size_t max_align = page;
    auto place = [&](size_t bytes, size_t alignment, size_t &offset) {
        size_t at = 0, end = 0;
        if (!align_size(cursor, alignment, at) || !add_size(at, bytes, end))
            return false;
        offset = at;
        cursor = end;
        max_align = std::max(max_align, alignment);
        return true;
    };
    size_t words = 0;
    if (i.algorithm == SBN3_DIVREM_BARRETT) {
        p.persistent_offset = aligned(cursor, page);
        cursor = p.persistent_offset;
        if (!place(bytes_for(dn, 8), 128, p.dn_offset) ||
            !place(bytes_for(in + 1, 8), 128, p.u_offset))
            return false;
        p.d_offset = p.dn_offset; // only the normalized divisor persists
        if (uq.cached && !place(uq.future.storage_bytes, page, p.uspec_offset))
            return false;
        if (tq.cached && !place(tq.future.storage_bytes, page, p.dspec_offset))
            return false;
        i.spectrum_bytes = (uq.cached ? uq.future.storage_bytes : 0) + (tq.cached ? tq.future.storage_bytes : 0);
        i.persistent_bytes = cursor - p.persistent_offset;
        // The inverse completes before either product is bound. Its entire
        // binding can share storage with the later product tables, mutable
        // workspaces and block scratch; only D/U/spectra persist outside it.
        p.newton_bytes = ninfo->storage_bytes;
        p.newton_alignment = std::max(ninfo->storage_alignment, page);
        p.shared_offset = aligned(cursor, p.newton_alignment);
        cursor = p.shared_offset;
        const auto &um = uq.consumer_info.mul, &tm = tq.consumer_info.mul;
        // Cached consumers borrow their immutable roots from the spectra.
        // Rebinding only their small control/codec state lets the two serial
        // products share one mutable workspace. Use it when it saves at least
        // one of the existing large aligned allocations, keeping tiny products
        // bound throughout to avoid adding a binder to their inner loop.
        const size_t work_alignment = std::max({um.workspace_alignment,tm.workspace_alignment,page});
        p.shared_products = uq.cached && tq.cached && work_alignment >= (size_t(1)<<20) &&
                            std::min(um.workspace_bytes,tm.workspace_bytes)>=work_alignment;
        if (p.shared_products) {
            i.table_bytes=std::max(um.table_bytes,tm.table_bytes);
            i.product_workspace_bytes=std::max(um.workspace_bytes,tm.workspace_bytes);
            if(!place(i.table_bytes,page,p.utab_offset) ||
               !place(i.product_workspace_bytes,work_alignment,p.uwork_offset))return false;
            p.ttab_offset=p.utab_offset;p.twork_offset=p.uwork_offset;
        } else {
            if (!place(um.table_bytes, page, p.utab_offset) ||
                !place(um.workspace_bytes, std::max(um.workspace_alignment, page), p.uwork_offset) ||
                !place(tm.table_bytes, page, p.ttab_offset) ||
                !place(tm.workspace_bytes, std::max(tm.workspace_alignment, page), p.twork_offset))
                return false;
            i.table_bytes = um.table_bytes + tm.table_bytes;
            i.product_workspace_bytes = um.workspace_bytes + tm.workspace_bytes;
        }
        p.xlen = p.ring ? p.ring : dn + in;
        if (p.head >= in || dn + p.head > p.xlen) // X = R*B^head + block lives in xbuf
            return false;
        p.tcap = std::max(sbn3_mul_output_capacity(&tm), tm.output_limbs);
        p.pcap = std::max(sbn3_mul_output_capacity(&um), um.output_limbs);
        if (!place(bytes_for(p.tcap, 8), page, p.tbuf_offset) ||
            !place(bytes_for(in, 8), page, p.ubuf_offset) || !place(bytes_for(p.pcap, 8), page, p.pbuf_offset) ||
            !place(bytes_for(in, 8), page, p.qbuf_offset) || !place(bytes_for(dn, 8), page, p.rbuf_offset))
            return false;
        p.xbuf_offset = p.tbuf_offset; // product T becomes the signed residual, after its last read
        i.scratch_bytes = cursor - p.tbuf_offset;
        const size_t newton_end = aligned(p.shared_offset + p.newton_bytes, page);
        cursor = std::max(cursor, newton_end);
        max_align = std::max(max_align, p.newton_alignment);
        i.shared_bytes = cursor - p.shared_offset;
        const size_t head = i.quotient_limbs % in;
        i.head_limbs = head <= p.head ? head : 0;
        i.blocks = unsigned(i.quotient_limbs / in + (head > p.head));
        i.products = 2;
        i.workers = std::max({1u, p.u.workers, p.t.workers, ninfo->workers});
        i.block_limbs = in;
        i.inverse_limbs = in;
        i.ring_limbs = p.ring;
    } else {
        p.persistent_offset = aligned(cursor, page);
        cursor = p.persistent_offset;
        if (!place(bytes_for(dn, 8), 128, p.d_offset))
            return false;
        i.persistent_bytes = cursor - p.persistent_offset;
        p.shared_offset = aligned(cursor, page);
        cursor = p.shared_offset;
        if (!add_size(nn, dn + 1, words) || !place(bytes_for(words, 8), page, p.xbuf_offset))
            return false;
        p.scratch_words = words;
        i.scratch_bytes = i.shared_bytes = cursor - p.shared_offset;
        i.workers = 1;
        i.blocks = 0;
        i.products = 0;
    }
    i.storage_alignment = max_align;
    i.storage_bytes = aligned(cursor, page);
    uint64_t key = feed(1469598103934665603ULL, i.algorithm);
    for (uint64_t v : {i.storage_bytes, i.storage_alignment, i.control_bytes, i.persistent_bytes, i.shared_bytes,
                       i.table_bytes, i.product_workspace_bytes, i.spectrum_bytes, i.scratch_bytes, p.xlen, p.head, p.tcap,
                       p.pcap, p.newton_bytes, uint64_t(p.shared_products), uq.consumer_info.mul.arithmetic_id, tq.consumer_info.mul.arithmetic_id,
                       uq.consumer_info.mul.execution_id, tq.consumer_info.mul.execution_id,
                       ninfo ? ninfo->plan_id : 0})
        key = feed(key, v);
    i.plan_id = key;
    return true;
}
Binding &binding(sbn3_divrem_binding *p) {
    auto *b = reinterpret_cast<Binding *>(p);
    require(b && b->marker == magic, SBN3_FATAL_LIFETIME, "division binding");
    return *b;
}
void idle(const Binding &b) {
    require(pthread_equal(b.team->creator, pthread_self()) && !b.team->busy, SBN3_FATAL_TEAM,
            "division owner/idle");
}
uint8_t *at(const Binding &b, size_t offset) {
    return b.arena->base + b.offset + offset;
}
void check_span(const Binding &b, const void *data, size_t bytes, const char *where) {
    valid_span(data, bytes, where);
    require(!(reinterpret_cast<uintptr_t>(data) & 7), SBN3_FATAL_ARGUMENT, where);
    require(!overlaps(data, bytes, b.arena->base + b.offset, b.plan.info.storage_bytes), SBN3_FATAL_ARGUMENT,
            "division value/storage overlap");
    for (unsigned j = 0; j < b.team->width; ++j) {
        const auto &l = j ? b.team->stacks[j] : b.team->storage;
        require(!overlaps(data, bytes, l.data, l.bytes), SBN3_FATAL_ARGUMENT, "division value/team overlap");
    }
}
void unbind_products(Binding &b) {
    if (b.uprod) {
        sbn3_mul_unbind(b.uprod);
        b.uprod = nullptr;
    }
    if (b.tprod) {
        sbn3_mul_unbind(b.tprod);
        b.tprod = nullptr;
    }
    for (auto *l : {&b.utab, &b.uwork, &b.ttab, &b.twork})
        if (l->token) {
            b.arena->release(*l);
            *l = {};
        }
}
sbn3_mul_binding *product_binding(Binding &b,bool inverse) {
    auto *&slot=inverse?b.uprod:b.tprod;
    if(slot)return slot;
    if(b.plan.shared_products)unbind_products(b);
    const auto &q=inverse?b.uq:b.tq;
    const auto &i=q.consumer_info.mul;
    auto &table=inverse?b.utab:b.ttab;
    auto &work=inverse?b.uwork:b.twork;
    table=b.arena->acquire(b.offset+(inverse?b.plan.utab_offset:b.plan.ttab_offset),i.table_bytes);
    work=b.arena->acquire(b.offset+(inverse?b.plan.uwork_offset:b.plan.twork_offset),i.workspace_bytes);
    sbn3_product_bind(&q.consumer,b.arena,&table,&work,b.team,inverse?b.uspec:b.dspec,nullptr,&slot);
    return slot;
}
void release_products(Binding &b) {
    unbind_products(b);
    if (b.uspec) {
        sbn3_spectrum_release(b.uspec);
        b.uspec = nullptr;
        b.arena->release(b.uspec_lease);
    }
    if (b.dspec) {
        sbn3_spectrum_release(b.dspec);
        b.dspec = nullptr;
        b.arena->release(b.dspec_lease);
    }
    if (b.shared_leased) {
        b.arena->release(b.shared);
        b.shared = {};
        b.shared_leased = false;
    }
}
// N' = N << shift over [lo, lo+count) of its nn+1 limbs, from the unshifted source.
void shifted_span(sbn3_team *team, uint64_t *dst, const uint64_t *N, size_t nn, size_t lo, size_t count,
                  unsigned shift) {
    parallel_limbs::each(team, count, parallel_limbs::parts(team, count), [&](size_t b, size_t e, unsigned) {
        for (size_t k = b; k < e; ++k) {
            const size_t j = lo + k;
            uint64_t v = j < nn ? N[j] << shift : 0;
            if (shift && j)
                v |= N[j - 1] >> (64 - shift);
            dst[k] = v;
        }
    });
}
void barrett_prepare(Binding &b, const uint64_t *D) {
    auto &p = b.plan;
    const size_t dn = p.request.denominator_limbs, in = p.in;
    release_products(b);
    b.prepared = false;
    b.shift = unsigned(__builtin_clzll(D[dn - 1]));
    if (b.shift)
        shifted_span(b.team, b.Dn, D, dn, 0, dn, b.shift);
    else
        parallel_limbs::copy(b.team, b.Dn, D, dn);
    // Block inverse of the top in limbs, in the shared union (unleased at this
    // point): exact U by the local recurrence, or the spectral Newton binding.
    // Both satisfy |U - B^(2in)/Dtop| < 3 with B^in <= U < 2B^in.
    if (inverse_basecase(in)) {
        auto scratch = b.arena->acquire(b.offset + p.shared_offset, p.newton_bytes);
        {
            Frame frame(*b.arena,scratch);
            local_inverse(b.U,b.Dn+dn-in,in,frame);
        }
        b.arena->release(scratch);
    } else {
        sbn3_newton_binding *inverse = nullptr;
        sbn3_newton_bind(&b.nplan, b.arena, b.offset + p.shared_offset, b.team, &inverse);
        sbn3_newton_inputs inputs{};
        inputs.denominator = {b.Dn + dn - in, in};
        sbn3_newton_execute(inverse, &inputs, {b.U, in + 1});
        sbn3_newton_unbind(inverse);
    }
    require(b.U[in] == 1, SBN3_FATAL_MATH, "division block inverse framing");
    b.dinv = divrem_words::invert_pi1(b.Dn[dn - 1], b.Dn[dn - 2]);
    // Persistent products and divisor-side spectra.
    b.shared = b.arena->acquire(b.offset + p.tbuf_offset, p.info.scratch_bytes);
    b.shared_leased = true;
    if (b.uq.cached) {
        b.uspec_lease = b.arena->acquire(b.offset + p.uspec_offset, b.uq.future.storage_bytes);
        sbn3_spectrum_reserve_plan(&b.uq.producer, SBN3_SPECTRUM_COLUMNS, b.uq.future.generation, b.arena,
                                   &b.uspec_lease, &b.uspec);
    }
    if (b.tq.cached) {
        b.dspec_lease = b.arena->acquire(b.offset + p.dspec_offset, b.tq.future.storage_bytes);
        sbn3_spectrum_reserve_plan(&b.tq.producer, SBN3_SPECTRUM_COLUMNS, b.tq.future.generation, b.arena,
                                   &b.dspec_lease, &b.dspec);
    }
    auto *u=product_binding(b,true);
    if (b.uspec)
        sbn3_spectrum_compute(u, b.uspec, {b.U, in + 1});
    auto *t=product_binding(b,false);
    if (b.dspec)
        sbn3_spectrum_compute(t, b.dspec, {b.Dn, dn});
    b.prepared = true;
}
// One exact quotient limb by word division. x holds X = R*B + n0 on dn+1
// limbs with R < D', so q = floor(X/D') < B. The step follows GMP's sbpi1_div_qr (3/2
// estimate q or q+1, multiply-subtract, add-back; notice in value/divrem_basecase.cpp); each
// team part subtracts its own slice of qhat*D' and the slice carries are stitched exactly.
uint64_t word_step(sbn3_team *team, uint64_t *x, const uint64_t *Dn, size_t dn, uint64_t dinv, uint64_t &addbacks) {
    const uint64_t d1 = Dn[dn - 1], d0 = Dn[dn - 2];
    uint64_t q = UINT64_MAX; // <x[dn],x[dn-1]> == <d1,d0>: the quotient limb is B-1
    if (x[dn] != d1 || x[dn - 1] != d0) {
        uint64_t r1, r0;
        udiv_qr_3by2(q, r1, r0, x[dn], x[dn - 1], x[dn - 2], d1, d0, dinv);
        (void)r1;
        (void)r0;
    }
    const unsigned parts = parallel_limbs::parts(team, dn);
    uint64_t high[32]{};
    parallel_limbs::each(team, dn, parts, [&](size_t b, size_t e, unsigned k) {
        high[k] = e > b ? sbn3i_submul_1(x + b, Dn + b, long(e - b), q) : 0;
    });
    uint64_t borrow = 0;
    for (unsigned k = 0; k < parts; ++k) {
        const size_t e = parallel_limbs::cut(dn, parts, k + 1);
        borrow += parallel_limbs::sub_word(x + e, dn + 1 - e, high[k]);
    }
    // X - qhat*D' > -B^(dn+1): at most one wrap, undone by the add-back.
    require(borrow <= 1, SBN3_FATAL_MATH, "division head borrow");
    for (unsigned k = 0; borrow; ++k) {
        require(k < divrem_tuning::head_correction_limit, SBN3_FATAL_MATH, "division head correction bound");
        --q;
        ++addbacks;
        borrow -= parallel_limbs::add_to(team, x, dn + 1, Dn, dn);
    }
    require(!x[dn], SBN3_FATAL_MATH, "division head remainder");
    return q;
}
// Replace T by [high:low] - T, zero extending the concatenation to n limbs.
// Both sources stay disjoint from T. Independent subtractions are joined by
// one borrow per partition; even an all-zero borrow chain visits O(n) words.
uint64_t residual_subtract(sbn3_team *team, uint64_t *t, size_t n, const uint64_t *low, size_t ln,
                          const uint64_t *high, size_t hn) {
    const unsigned parts = parallel_limbs::parts(team, n);
    uint64_t borrows[32]{};
    parallel_limbs::each(team, n, parts, [&](size_t begin, size_t end, unsigned k) {
        size_t at = begin;
        uint64_t borrow = 0;
        auto subtract = [&](const uint64_t *source, size_t count) {
            if (!count) return;
            const uint64_t next = sbn3i_sub_n(t + at, source, t + at, long(count));
            borrow = next + (borrow ? parallel_limbs::sub_word(t + at, count, 1) : 0);
            at += count;
        };
        if (at < ln) subtract(low + at, std::min(end, ln) - at);
        if (at < end && at < ln + hn) subtract(high + at - ln, std::min(end, ln + hn) - at);
        if (at < end) {
            uint64_t carry = 1 - borrow;
            for (; at < end; ++at) {
                const __uint128_t v = __uint128_t(~t[at]) + carry;
                t[at] = uint64_t(v);
                carry = uint64_t(v >> 64);
            }
            borrow = 1 - carry;
        }
        borrows[k] = borrow;
    });
    uint64_t borrow = 0;
    for (unsigned k = 0; k < parts; ++k) {
        const size_t begin = parallel_limbs::cut(n, parts, k), end = parallel_limbs::cut(n, parts, k + 1);
        borrow = borrows[k] + (borrow ? parallel_limbs::sub_word(t + begin, end - begin, 1) : 0);
    }
    return borrow;
}
bool residual_absolute(sbn3_team *team, uint64_t *r, size_t n) {
    if (!(r[n - 1] >> 63)) return false;
    const unsigned parts = parallel_limbs::parts(team, n);
    uint64_t any[32]{};
    parallel_limbs::each(team, n, parts, [&](size_t begin, size_t end, unsigned k) {
        uint64_t bits = 0;
        for (size_t j = begin; j < end; ++j) bits |= (r[j] = ~r[j]);
        any[k] = bits;
    });
    uint64_t bits = 0;
    for (unsigned k = 0; k < parts; ++k) bits |= any[k];
    // Folding after subtraction can represent zero as B^n-1. Its one's
    // complement is zero, with no negative sign and no quotient correction.
    return bits != 0;
}
void barrett_execute(Binding &b, const uint64_t *N, size_t nn, uint64_t *Q, uint64_t *R, sbn3_divrem_result &out) {
    auto &p = b.plan;
    auto *team = b.team;
    const size_t dn = p.request.denominator_limbs, in = p.in, L = p.xlen;
    const bool cyclic = p.ring != 0;
    const unsigned s = b.shift;
    const uint64_t *Dn = b.Dn;
    uint64_t *xbuf = b.xbuf, *tbuf = b.tbuf, *ubuf = b.ubuf, *pbuf = b.pbuf, *qbuf = b.qbuf, *rbuf = b.rbuf;
    // R = top dn limbs of N' (N'[nn] is the shifted-out top); Q < B^qn makes R < D'.
    const size_t qn = nn + 1 - dn;
    shifted_span(team, rbuf, N, nn, qn, dn, s);
    require(divrem_words::compare(rbuf, Dn, dn) < 0, SBN3_FATAL_MATH, "division normalized head");
    uint64_t corrections = 0;
    unsigned products = 0;
    size_t pos = qn;
    // A short first block (qn mod in limbs, e.g. the first quotient limb of
    // 2d/d under a full inverse) is exact word division over
    // X = R*B^head + N'[pos-head, pos); the blocks below are then complete.
    if (const size_t head = qn % in; head && head <= p.head) {
        pos -= head;
        shifted_span(team, xbuf, N, nn, pos, head, s);
        parallel_limbs::copy(team, xbuf + head, rbuf, dn);
        for (size_t j = head; j-- > 0;)
            Q[pos + j] = word_step(team, xbuf + j, Dn, dn, b.dinv, b.metrics.head_corrections);
        b.metrics.head_limbs += head;
        parallel_limbs::copy(team, rbuf, xbuf, dn);
    }
    while (pos) {
        const size_t ic = std::min(in, pos);
        pos -= ic;
        // qhat = floor(rtop * U / B^in), rtop = top ic limbs of R.
        parallel_limbs::copy(team, ubuf, rbuf + dn - ic, ic);
        parallel_limbs::fill(team, ubuf + ic, in - ic);
        sbn3_product_inputs pin{};
        if (!b.uspec)
            pin.a = {b.U, in + 1};
        pin.b = {ubuf, in};
        sbn3_product_execute(product_binding(b,true), &pin, {pbuf, p.pcap});
        ++products;
        parallel_limbs::copy(team, qbuf, pbuf + in, ic);
        if (!parallel_limbs::zero(team, pbuf + in + ic, in + 1 - ic))
            parallel_limbs::fill(team, qbuf, ic, UINT64_MAX);
        parallel_limbs::fill(team, qbuf + ic, in - ic);
        // T = qhat * D' (mod B^ring-1 on the cyclic recipe).
        sbn3_product_inputs tin{};
        if (!b.dspec)
            tin.a = {Dn, dn};
        tin.b = {qbuf, in};
        sbn3_product_execute(product_binding(b,false), &tin, {tbuf, p.tcap});
        ++products;
        // E = R*B^ic + N'block - T, |E| < 7 D' < B^(dn+1).
        // T's output storage becomes E. The U operand has already been read,
        // so reuse ubuf for the shifted low block instead of materializing X.
        shifted_span(team, ubuf, N, nn, pos, ic, s);
        const uint64_t borrow = residual_subtract(team, xbuf, L, ubuf, ic, rbuf, std::min(dn, L - ic));
        bool negative;
        if (cyclic) {
            if (borrow)
                limbs::cyclic_sub_power(xbuf, L, 0);
            if (ic + dn > L && parallel_limbs::add_to(team, xbuf, L, rbuf + L - ic, ic + dn - L))
                parallel_limbs::add_word(xbuf, L, 1);
            negative = residual_absolute(team, xbuf, L);
        } else {
            negative = borrow != 0;
            if (negative) {
                parallel_limbs::complement(team, xbuf, L);
                parallel_limbs::add_word(xbuf, L, 1);
            }
        }
        require(parallel_limbs::zero(team, xbuf + dn + 1, L - dn - 1), SBN3_FATAL_MATH, "division residual certificate");
        unsigned k = 0;
        auto above = [&](bool strict) {
            return xbuf[dn] != 0 || divrem_words::compare(xbuf, Dn, dn) > (strict ? 0 : -1);
        };
        if (negative) {
            // E = -m: add D' until nonnegative; the last step is R = D' - m.
            while (above(true)) {
                require(++k <= divrem_tuning::correction_limit, SBN3_FATAL_MATH, "division correction bound");
                parallel_limbs::sub_from(team, xbuf, dn + 1, Dn, dn);
            }
            require(!sbn3i_sub_n(xbuf, Dn, xbuf, long(dn)), SBN3_FATAL_MATH, "division negative residual");
            ++k;
            require(!parallel_limbs::sub_word(qbuf, ic, k), SBN3_FATAL_MATH, "division quotient underflow");
        } else {
            while (above(false)) {
                require(++k <= divrem_tuning::correction_limit, SBN3_FATAL_MATH, "division correction bound");
                parallel_limbs::sub_from(team, xbuf, dn + 1, Dn, dn);
            }
            if (k)
                require(!parallel_limbs::add_word(qbuf, ic, k), SBN3_FATAL_MATH, "division quotient overflow");
        }
        corrections += k;
        parallel_limbs::copy(team, Q + pos, qbuf, ic);
        parallel_limbs::copy(team, rbuf, xbuf, dn);
    }
    if (s)
        parallel_limbs::each(team, dn, parallel_limbs::parts(team, dn), [&](size_t begin, size_t end, unsigned) {
            for (size_t j = begin; j < end; ++j)
                R[j] = (rbuf[j] >> s) | (j + 1 < dn ? rbuf[j + 1] << (64 - s) : 0);
        });
    else
        parallel_limbs::copy(team, R, rbuf, dn);
    out.corrections = corrections;
    b.metrics.products_executed = products;
}
// The block plan of a request. Block sizes in cost order; under a memory budget
// the first whose storage fits, so a faster but larger size never displaces a
// plan the caller can hold. When none fits, info reports the least requirement.
// Kept out of line: the candidates' recipes are about 100 KB of frame, which
// the word and schoolbook queries (a microsecond) do not carry.
[[gnu::noinline]] sbn3_query_result plan_blocks(Plan &p, Stored &s, sbn3_divrem_info *info) {
    const size_t budget = p.options.memory_budget;
    Ranking ranking{};
    rank(p, ranking);
    const auto no = newton_options(p);
    sbn3_query_result failure = SBN3_UNSUPPORTED;
    bool failed = false;
    size_t inverse_in = 0;
    sbn3_query_result inverse = SBN3_UNSUPPORTED;
    auto fits = [&](const Candidate &c) {
        Plan sized = p;
        sized.in = c.in;
        sized.ring = c.ring;
        sized.head = c.head;
        sized.u = c.u;
        sized.t = c.t;
        if (inverse_in != c.in) {
            inverse = inverse_query(c.in, no, s.nplan, s.ninfo);
            inverse_in = c.in;
        }
        if (inverse != SBN3_SUPPORTED || !layout(sized, c.uq, c.tq, &s.ninfo)) {
            if (!failed)
                failure = inverse != SBN3_SUPPORTED ? inverse : SBN3_QUERY_CAPACITY;
            failed = true;
            return false;
        }
        if (budget && sized.info.storage_bytes > budget) {
            if (!info->storage_bytes || sized.info.storage_bytes < info->storage_bytes)
                *info = sized.info;
            return false;
        }
        p = sized;
        s.uq = c.uq;
        s.tq = c.tq;
        return true;
    };
    for (Candidate *taken; (taken = take(p, ranking));)
        if (fits(*taken) || (budget && other_family(p, *taken) && fits(*taken)))
            return SBN3_SUPPORTED;
    // A budget none of the ordered sizes meets: smaller blocks hold less. The least size is halved, a bounded number
    // of times (divrem_tuning); below that the requirement is reported.
    if (budget && !p.options.block_limbs && ranking.count) {
        size_t in = ranking.sizes[0].in;
        for (unsigned j = 1; j < ranking.count; ++j)
            in = std::min(in, ranking.sizes[j].in);
        Candidate &smaller = ranking.sizes[0]; // every ordered size has been tried
        for (unsigned k = 0; k < divrem_tuning::budget_halvings && in > 1; ++k) {
            const size_t qn = p.info.quotient_limbs, blocks = (qn + (in + 1) / 2 - 1) / ((in + 1) / 2);
            in = (qn + blocks - 1) / blocks;
            if (!search(p, in, smaller, ranking.transform) || !finish(p, ranking, smaller))
                break;
            if (fits(smaller) || (other_family(p, smaller) && fits(smaller)))
                return SBN3_SUPPORTED;
        }
    }
    return info->storage_bytes ? SBN3_QUERY_CAPACITY : failure;
}
// A memory budget that no block size meets: the schoolbook holds the least of all, and serves the request where
// its complete cost is bounded (divrem_tuning). Out of line, as plan_blocks: the plan value is reset here.
[[gnu::noinline]] bool schoolbook_under_budget(Plan &p, Stored &s, sbn3_divrem_info *info) {
    const size_t dn = p.request.denominator_limbs, qn = p.info.quotient_limbs, level = divrem_tuning::schoolbook_budget_quotient;
    const double work = double(std::max(1u, p.options.reuse_hint)) * double(qn > level ? qn - level : 0) * double(dn);
    if (p.options.algorithm || p.options.block_limbs ||
        (dn > divrem_tuning::schoolbook_budget_divisor && work > divrem_tuning::schoolbook_budget_work))
        return false;
    Plan q = p;
    q.info.algorithm = SBN3_DIVREM_SCHOOLBOOK;
    s = Stored{};
    if (!layout(q, s.uq, s.tq, &s.ninfo))
        return false;
    if (q.info.storage_bytes > p.options.memory_budget) { // the least requirement of the request is then the schoolbook's
        if (q.info.storage_bytes < info->storage_bytes)
            *info = q.info;
        return false;
    }
    p = q;
    return true;
}
} // namespace
} // namespace sbn::v3
using namespace sbn::v3;
extern "C" sbn3_query_result sbn3_divrem_query(const sbn3_divrem_request *request,
                                               const sbn3_divrem_options *options, sbn3_divrem_plan *out,
                                               sbn3_divrem_info *info) {
    require(request && out && info, SBN3_FATAL_ARGUMENT, "division query arguments");
    *info = {};
    Plan p{};
    p.request = *request;
    p.options = options ? *options : sbn3_divrem_options{1, 0, 0, 0, 0, 0, 0, 0};
    const size_t dn = request->denominator_limbs, nn = request->numerator_limbs;
    if (!dn || !p.options.workers || p.options.workers > 32 || p.options.timing > 1 || p.options.residual > 2 || p.options.algorithm > 2 ||
        (p.options.prime_count && (p.options.prime_count < newton_limits::first_ntt_prime_count ||
                                   p.options.prime_count > newton_limits::last_ntt_prime_count)))
        return SBN3_UNSUPPORTED;
    if (nn > newton_limits::precision_words || dn > newton_limits::precision_words)
        return SBN3_QUERY_CAPACITY;
    auto &i = p.info;
    i.numerator_limbs = nn;
    i.denominator_limbs = dn;
    i.quotient_limbs = nn >= dn ? nn - dn + 1 : 0;
    i.remainder_limbs = dn;
    if (dn <= 2)
        i.algorithm = SBN3_DIVREM_WORD;
    else if (p.options.algorithm)
        i.algorithm = p.options.algorithm == SBN3_DIVREM_BARRETT && i.quotient_limbs ? SBN3_DIVREM_BARRETT : SBN3_DIVREM_SCHOOLBOOK;
    else if (!i.quotient_limbs || dn <= divrem_tuning::schoolbook_max_divisor || (!p.options.block_limbs && schoolbook_use(p)))
        i.algorithm = SBN3_DIVREM_SCHOOLBOOK;
    else
        i.algorithm = SBN3_DIVREM_BARRETT;
    Stored s{};
    const size_t budget = p.options.memory_budget;
    if (i.algorithm == SBN3_DIVREM_BARRETT) {
        const auto planned = plan_blocks(p, s, info);
        if (planned != SBN3_SUPPORTED && !(planned == SBN3_QUERY_CAPACITY && budget && schoolbook_under_budget(p, s, info)))
            return planned;
        *info = p.info;
    } else {
        if (!layout(p, s.uq, s.tq, &s.ninfo))
            return SBN3_QUERY_CAPACITY;
        *info = p.info;
        if (budget && p.info.storage_bytes > budget)
            return SBN3_QUERY_CAPACITY;
    }
    p.seal = seal(p);
    s.plan = p;
    memset(out, 0, sizeof *out);
    memcpy(out->opaque, &s, sizeof s);
    return SBN3_SUPPORTED;
}
extern "C" void sbn3_divrem_bind(const sbn3_divrem_plan *opaque, sbn3_arena *arena, size_t offset, sbn3_team *team,
                                 sbn3_divrem_binding **out) {
    require(opaque && arena && team && out, SBN3_FATAL_ARGUMENT, "division bind arguments");
    Stored s{};
    load(*opaque, s);
    Plan p = s.plan;
    require(team->arena == arena && team->width >= p.info.workers && pthread_equal(team->creator, pthread_self()) &&
                !team->busy,
            SBN3_FATAL_TEAM, "division bind team");
    size_t end = 0;
    require(add_size(offset, p.info.storage_bytes, end) && end <= arena->virtual_bytes && arena->contains(offset, end) &&
                !(reinterpret_cast<uintptr_t>(arena->base + offset) & (p.info.storage_alignment - 1)) &&
                arena->unleased(offset, p.info.storage_bytes),
            SBN3_FATAL_WORKSPACE, "division prepared unleased range");
    // Layout replay from the stored recipes proves the plan's own consistency;
    // the product and Newton plans carry their own seals.
    const auto expected = p.info;
    require(layout(p, s.uq, s.tq, &s.ninfo) && p.info.plan_id == expected.plan_id &&
                p.info.storage_bytes == expected.storage_bytes,
            SBN3_FATAL_ARGUMENT, "division query/bind equivalence");
    const Queried &uq = s.uq, &tq = s.tq;
    const sbn3_newton_plan &nplan = s.nplan;
    const sbn3_newton_info &ninfo = s.ninfo;
    auto control = arena->acquire(offset, p.info.control_bytes);
    auto *b = ::new (control.data) Binding{};
    b->plan = p;
    b->arena = arena;
    b->team = team;
    b->offset = offset;
    b->control = control;
    b->uq = uq;
    b->tq = tq;
    b->nplan = nplan;
    b->ninfo = ninfo;
    // Divisor-side values keep their own lease; product tables, workspaces
    // and spectra are claimed exclusively by their bindings at prepare.
    b->D = reinterpret_cast<uint64_t *>(at(*b, p.d_offset));
    if (p.info.algorithm == SBN3_DIVREM_BARRETT) {
        b->persistent = arena->acquire(offset + p.d_offset, p.u_offset + bytes_for(p.in + 1, 8) - p.d_offset);
        b->Dn = reinterpret_cast<uint64_t *>(at(*b, p.dn_offset));
        b->U = reinterpret_cast<uint64_t *>(at(*b, p.u_offset));
        b->xbuf = reinterpret_cast<uint64_t *>(at(*b, p.xbuf_offset));
        b->tbuf = reinterpret_cast<uint64_t *>(at(*b, p.tbuf_offset));
        b->ubuf = reinterpret_cast<uint64_t *>(at(*b, p.ubuf_offset));
        b->pbuf = reinterpret_cast<uint64_t *>(at(*b, p.pbuf_offset));
        b->qbuf = reinterpret_cast<uint64_t *>(at(*b, p.qbuf_offset));
        b->rbuf = reinterpret_cast<uint64_t *>(at(*b, p.rbuf_offset));
    } else {
        b->persistent = arena->acquire(offset + p.d_offset, bytes_for(p.request.denominator_limbs, 8));
        b->scratch = reinterpret_cast<uint64_t *>(at(*b, p.xbuf_offset));
        b->shared = arena->acquire(offset + p.shared_offset, p.info.shared_bytes);
        b->shared_leased = true;
    }
    *out = reinterpret_cast<sbn3_divrem_binding *>(b);
}
extern "C" void sbn3_divrem_prepare(sbn3_divrem_binding *opaque, sbn3_const_limbs denominator) {
    auto &b = binding(opaque);
    idle(b);
    const auto &p = b.plan;
    const size_t dn = p.request.denominator_limbs;
    require(denominator.count == dn && denominator.data, SBN3_FATAL_ARGUMENT, "division divisor length");
    check_span(b, denominator.data, bytes_for(dn, 8), "division divisor");
    require(denominator.data[dn - 1] != 0, SBN3_FATAL_ARGUMENT, "division divisor top limb / zero divisor");
    const auto start = p.options.timing ? now() : 0;
    if (p.info.algorithm == SBN3_DIVREM_BARRETT)
        barrett_prepare(b, denominator.data);
    else {
        memcpy(b.D, denominator.data, dn * 8);
        b.prepared = true;
    }
    ++b.metrics.prepares;
    if (p.options.timing)
        b.metrics.prepare_ns = now() - start;
}
extern "C" void sbn3_divrem_execute(sbn3_divrem_binding *opaque, sbn3_const_limbs numerator, sbn3_limbs quotient,
                                    sbn3_limbs remainder, sbn3_divrem_result *result) {
    auto &b = binding(opaque);
    idle(b);
    require(b.prepared, SBN3_FATAL_LIFETIME, "division divisor not prepared");
    const auto &p = b.plan;
    const size_t dn = p.request.denominator_limbs, nn = numerator.count;
    require(nn <= p.request.numerator_limbs && (!nn || numerator.data) && remainder.data &&
                remainder.capacity >= dn && result,
            SBN3_FATAL_ARGUMENT, "division execute spans");
    const size_t qcount = nn >= dn ? nn - dn + 1 : 0;
    require(!qcount || (quotient.data && quotient.capacity >= qcount), SBN3_FATAL_ARGUMENT, "division quotient capacity");
    const size_t nb = bytes_for(nn, 8), qb = bytes_for(qcount, 8), rb = bytes_for(dn, 8);
    if (nn)
        check_span(b, numerator.data, nb, "division numerator");
    if (qcount)
        check_span(b, quotient.data, qb, "division quotient");
    check_span(b, remainder.data, rb, "division remainder");
    require(!overlaps(numerator.data, nb, quotient.data, qb) && !overlaps(numerator.data, nb, remainder.data, rb) &&
                !overlaps(quotient.data, qb, remainder.data, rb),
            SBN3_FATAL_ARGUMENT, "division value overlap");
    const auto start = p.options.timing ? now() : 0;
    *result = {};
    size_t nn_eff = nn;
    while (nn_eff && !numerator.data[nn_eff - 1])
        --nn_eff;
    size_t written = 0;
    if (nn_eff < dn) {
        memcpy(remainder.data, numerator.data, nn_eff * 8);
        memset(remainder.data + nn_eff, 0, (dn - nn_eff) * 8);
        b.metrics.products_executed = 0;
    } else if (p.info.algorithm == SBN3_DIVREM_BARRETT) {
        written = nn_eff - dn + 1;
        barrett_execute(b, numerator.data, nn_eff, quotient.data, remainder.data, *result);
    } else {
        written = nn_eff - dn + 1;
        divrem_words::schoolbook(quotient.data, remainder.data, numerator.data, nn_eff, b.D, dn, b.scratch);
        b.metrics.products_executed = 0;
    }
    if (qcount > written)
        memset(quotient.data + written, 0, (qcount - written) * 8);
    size_t qlen = written;
    while (qlen && !quotient.data[qlen - 1])
        --qlen;
    size_t rlen = dn;
    while (rlen && !remainder.data[rlen - 1])
        --rlen;
    result->quotient_limbs = qlen;
    result->remainder_limbs = rlen;
    b.metrics.corrections += result->corrections;
    ++b.metrics.executes;
    if (p.options.timing)
        b.metrics.execute_ns = now() - start;
}
extern "C" void sbn3_int_divrem_execute(sbn3_divrem_binding *opaque, sbn3_int *q, sbn3_int *r, sbn3_int_view n,
                                        sbn3_int_view d) {
    auto &b = binding(opaque);
    require(q && r && n.negative <= 1 && d.negative <= 1 && b.prepared, SBN3_FATAL_ARGUMENT, "integer division arguments");
    const size_t dn = b.plan.request.denominator_limbs;
    while (d.size && !d.data[d.size - 1])
        --d.size;
    while (n.size && !n.data[n.size - 1])
        --n.size;
    bool same = d.size == dn;
    if (same && b.plan.info.algorithm == SBN3_DIVREM_BARRETT && b.shift) {
        uint64_t difference = 0, carry = 0;
        for (size_t j = 0; j < dn; ++j) {
            const uint64_t word = d.data[j];
            difference |= ((word << b.shift) | carry) ^ b.Dn[j];
            carry = word >> (64 - b.shift);
        }
        same = !difference && !carry;
    } else if (same) same = !memcmp(d.data, b.D, dn * 8);
    require(same, SBN3_FATAL_ARGUMENT, "integer division prepared divisor");
    const size_t qn = n.size >= dn ? n.size - dn + 1 : 0;
    require(q->capacity >= qn && r->capacity >= dn, SBN3_FATAL_SIZE, "integer division capacity");
    sbn3_divrem_result out{};
    sbn3_divrem_execute(opaque, {n.data, n.size}, {q->data, q->capacity}, {r->data, r->capacity}, &out);
    q->size = out.quotient_limbs;
    q->negative = out.quotient_limbs ? (n.negative ^ d.negative) : 0;
    r->size = out.remainder_limbs;
    r->negative = out.remainder_limbs ? n.negative : 0;
}
extern "C" void sbn3_divrem_get_metrics(const sbn3_divrem_binding *opaque, sbn3_divrem_metrics *out) {
    const auto &b = binding(const_cast<sbn3_divrem_binding *>(opaque));
    require(out, SBN3_FATAL_ARGUMENT, "division metrics");
    *out = b.metrics;
}
extern "C" void sbn3_divrem_unbind(sbn3_divrem_binding *opaque) {
    auto &b = binding(opaque);
    idle(b);
    release_products(b);
    auto *arena = b.arena;
    const auto control = b.control;
    arena->release(b.persistent);
    b.marker = 0;
    b.~Binding();
    arena->release(control);
}
