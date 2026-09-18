// Exact integer division service: word/schoolbook for small shapes, block
// Barrett with a Newton block inverse and cached spectra otherwise.
// Derivations: experiments/docs/divrem-design-2026-09-18.md.
#include "sbn3/divrem.h"
#include "algorithms/divrem_tuning.hpp"
#include "algorithms/newton_tuning.hpp"
#include "algorithms/newton_limits.hpp"
#include "common/checked.hpp"
#include "common/identity.hpp"
#include "product/cost_model.hpp"
#include "product/root_prepare_cost.hpp"
#include "runtime/team.hpp"
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
                              p.newton_bytes, p.newton_alignment, p.tcap, p.pcap, p.scratch_words};
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
bool query_cached(sbn3_product_request r, const sbn3_mul_options &o, uint64_t generation, Queried &q) {
    if (sbn3_product_query(&r, &o, &q.producer, &q.producer_info) != SBN3_SUPPORTED ||
        sbn3_spectrum_query(&q.producer, SBN3_SPECTRUM_COLUMNS, generation, &q.future) != SBN3_SUPPORTED)
        return false;
    r.cached_a[0] = &q.future;
    return sbn3_product_query(&r, &o, &q.consumer, &q.consumer_info) == SBN3_SUPPORTED;
}
// Term 0 (a) is the divisor-side operand that may be cached; term 1 (b) is fresh.
bool query_product(size_t a, size_t b, const ProductChoice &c, uint64_t generation, Queried &q) {
    q = {};
    auto o = product_options(c);
    sbn3_product_request r{};
    r.kind = SBN3_PRODUCT_MUL;
    r.a_limbs = a;
    r.b_limbs = b;
    r.cyclic_limbs = c.ring;
    if (!c.cached) {
        if (sbn3_product_query(&r, &o, &q.producer, &q.producer_info) != SBN3_SUPPORTED)
            return false;
        q.consumer = q.producer;
        q.consumer_info = q.producer_info;
        return true;
    }
    if (!query_cached(r, o, generation, q))
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
Cost product_cost(const Queried &q, size_t a, size_t b) {
    const auto &i = q.consumer_info.mul;
    const bool deep = std::max(a, b) > native_policy::small_model_max_words;
    const double base = q.consumer_info.cyclic_limbs ? cost_model::cyclic_product(i, deep).nanoseconds
                                                     : cost_model::linear_product(i, a, b).nanoseconds;
    Cost c;
    c.apply = q.cached ? cost_model::cached_share(base) : base;
    c.prepare = root_prepare_cost(i) + (q.cached ? cost_model::prepare_share(base) : 0);
    return c;
}
unsigned product_workers(const Plan &p, size_t n) {
    return newton_rung_workers(n, p.options.workers);
}
// Best recipe for the U product (in+1) x in: linear, cached when the family
// supports a reserved spectrum.
bool choose_u(const Plan &p, size_t in, ProductChoice &out, Queried &q, Cost &cost) {
    ProductChoice c{};
    c.np = p.options.prime_count;
    c.workers = product_workers(p, in + 1);
    bool found = false;
    for (unsigned cached : {1u, 0u}) {
        c.cached = cached;
        Queried candidate{};
        if (!query_product(in + 1, in, c, 1, candidate))
            continue;
        const Cost k = product_cost(candidate, in + 1, in);
        if (!found || k.apply < cost.apply) {
            found = true;
            out = c;
            q = candidate;
            cost = k;
        }
    }
    return found;
}
// Best recipe for T = D' x qhat: linear (dn+in output) or cyclic at a
// supported ring >= dn + guard, term 0 cached where possible. Ordered by
// prepare + applications * apply.
bool choose_t(const Plan &p, size_t dn, size_t in, double applications, ProductChoice &out, Queried &q, Cost &cost,
              size_t &ring) {
    // Linear and cyclic families are ranked separately by the cost model; the
    // family is then chosen by the structural ring rule (divrem_tuning).
    struct Best {
        bool found = false;
        double total = INFINITY;
        ProductChoice choice{};
        Queried queried{};
        Cost cost{};
        size_t ring = 0;
    } lin, cyc;
    auto consider = [&](Best &b, const ProductChoice &c, size_t r) {
        Queried candidate{};
        if (!query_product(dn, in, c, 2, candidate))
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
    };
    auto take = [&](const Best &b) {
        out = b.choice;
        q = b.queried;
        cost = b.cost;
        ring = b.ring;
        return true;
    };
    ProductChoice linear{};
    linear.np = p.options.prime_count;
    linear.workers = product_workers(p, dn);
    if (p.options.residual != 2)
        for (unsigned cached : {1u, 0u}) {
            linear.cached = cached;
            consider(lin, linear, 0);
        }
    if (p.options.residual == 1)
        return lin.found && take(lin);
    const size_t minimum = dn + divrem_tuning::ring_guard_words;
    const unsigned width = product_workers(p, dn);
    const unsigned first = p.options.prime_count ? p.options.prime_count : newton_limits::first_ntt_prime_count,
                   last = p.options.prime_count ? p.options.prime_count : newton_limits::last_ntt_prime_count;
    if (!p.options.prime_count && minimum <= 512) {
        ProductChoice c{0, SBN3_MUL_SCALAR, 1, 0, minimum, 0};
        consider(cyc, c, minimum);
    }
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
                ProductChoice c{np, algorithm, width, T, r, 1};
                consider(cyc, c, r);
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
            for (unsigned w : {1u, width}) {
                ProductChoice c{0, SBN3_MUL_PQ16, w, 16, r, 1};
                consider(cyc, c, r);
            }
        }
    if (!cyc.found)
        return lin.found && take(lin);
    if (!lin.found || p.options.residual == 2)
        return take(cyc);
    // Cached against cached: the ring rule; otherwise (no cyclic spectrum, e.g.
    // the scalar ring) the cost model decides.
    if (cyc.queried.cached && lin.queried.cached)
        return take(double(cyc.ring) <= divrem_tuning::cyclic_ring_fraction * double(dn + in) ? cyc : lin);
    return take(cyc.total < lin.total ? cyc : lin);
}
double inverse_estimate(const Plan &p, size_t in) {
    ProductChoice c{};
    c.np = p.options.prime_count;
    c.workers = product_workers(p, in);
    Queried q{};
    if (!query_product(in, in, c, 0, q))
        return INFINITY;
    return divrem_tuning::inverse_cost_ratio * cost_model::linear_product(q.consumer_info.mul, in, in).nanoseconds;
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
    const double limit = std::floor(pair_ns / head_step_cost(p, dn));
    return limit >= double(in - 1) ? in - 1 : limit > 0 ? size_t(limit) : 0;
}
struct Candidate {
    size_t in = 0, ring = 0, head = 0;
    ProductChoice u{}, t{};
    Queried uq{}, tq{};
    double total = INFINITY;
};
// Block-size ordering uses the linear residual recipe only; the cyclic
// lattice (the expensive part of the query) runs once for the chosen size.
bool evaluate(Plan p, size_t in, Candidate &c, bool lattice) {
    const size_t dn = p.request.denominator_limbs, qn = p.info.quotient_limbs;
    const size_t blocks = (qn + in - 1) / in, head = qn % in;
    const double reuse = std::max(1u, p.options.reuse_hint);
    Cost uc{}, tc{};
    if (!choose_u(p, in, c.u, c.uq, uc))
        return false;
    if (!lattice && !p.options.residual)
        p.options.residual = 1;
    if (!choose_t(p, dn, in, reuse * double(blocks), c.t, c.tq, tc, c.ring))
        return false;
    c.in = in;
    // The residual buffer bounds the head: X = R*B^head + block needs dn+head limbs.
    const size_t xlen = c.ring ? c.ring : dn + in;
    c.head = std::min(head_limit(p, dn, in, uc.apply + tc.apply), xlen - dn);
    const bool word_head = head && head <= c.head;
    c.total = inverse_estimate(p, in) + uc.prepare + tc.prepare +
              reuse * (double(blocks - word_head) * (uc.apply + tc.apply) +
                       (word_head ? double(head) * head_step_cost(p, dn) : 0));
    return std::isfinite(c.total);
}
// Choose the block size and both products. With replay, the plan's stored
// choices are re-queried exactly.
bool assemble(Plan &p, bool replay, Queried &uq, Queried &tq) {
    const size_t dn = p.request.denominator_limbs, qn = p.info.quotient_limbs;
    if (replay) // exact re-query of stored choices (tests/diagnostics; bind uses the stored recipes)
        return query_product(p.in + 1, p.in, p.u, 1, uq) && query_product(dn, p.in, p.t, 2, tq);
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
        // Complete dn-limb blocks under a short word-division head (2d/d has
        // qn = dn+1: one full-inverse block plus the first quotient limb).
        // The cost order decides; a long head makes this a padded extra block.
        if (qn % dn <= divrem_tuning::head_candidate_limbs && dn >= divrem_tuning::head_candidate_min_divisor &&
            p.options.workers > 1)
            add(dn);
    }
    Candidate best{};
    for (unsigned j = 0; j < count; ++j) {
        Candidate c{};
        if (evaluate(p, candidates[j], c, count == 1) && c.total < best.total)
            best = c;
    }
    if (!std::isfinite(best.total))
        return false;
    if (count > 1) {
        Candidate full{};
        if (evaluate(p, best.in, full, true))
            best = full;
    }
    p.in = best.in;
    p.ring = best.ring;
    p.head = best.head;
    p.u = best.u;
    p.t = best.t;
    uq = best.uq;
    tq = best.tq;
    return true;
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
        if (!place(bytes_for(dn, 8), 128, p.d_offset) || !place(bytes_for(dn, 8), 128, p.dn_offset) ||
            !place(bytes_for(in + 1, 8), 128, p.u_offset))
            return false;
        if (uq.cached && !place(uq.future.storage_bytes, page, p.uspec_offset))
            return false;
        if (tq.cached && !place(tq.future.storage_bytes, page, p.dspec_offset))
            return false;
        i.spectrum_bytes = (uq.cached ? uq.future.storage_bytes : 0) + (tq.cached ? tq.future.storage_bytes : 0);
        const auto &um = uq.consumer_info.mul, &tm = tq.consumer_info.mul;
        if (!place(um.table_bytes, page, p.utab_offset) ||
            !place(um.workspace_bytes, std::max(um.workspace_alignment, page), p.uwork_offset) ||
            !place(tm.table_bytes, page, p.ttab_offset) ||
            !place(tm.workspace_bytes, std::max(tm.workspace_alignment, page), p.twork_offset))
            return false;
        i.table_bytes = um.table_bytes + tm.table_bytes;
        i.product_workspace_bytes = um.workspace_bytes + tm.workspace_bytes;
        i.persistent_bytes = cursor - p.persistent_offset;
        // Shared union: Newton inverse storage during prepare, block scratch during execute.
        p.newton_bytes = ninfo->storage_bytes;
        p.newton_alignment = std::max(ninfo->storage_alignment, page);
        p.shared_offset = aligned(cursor, p.newton_alignment);
        cursor = p.shared_offset;
        p.xlen = p.ring ? p.ring : dn + in;
        if (p.head >= in || dn + p.head > p.xlen) // X = R*B^head + block lives in xbuf
            return false;
        p.tcap = std::max(sbn3_mul_output_capacity(&tm), tm.output_limbs);
        p.pcap = std::max(sbn3_mul_output_capacity(&um), um.output_limbs);
        if (!place(bytes_for(p.xlen, 8), page, p.xbuf_offset) || !place(bytes_for(p.tcap, 8), page, p.tbuf_offset) ||
            !place(bytes_for(in, 8), page, p.ubuf_offset) || !place(bytes_for(p.pcap, 8), page, p.pbuf_offset) ||
            !place(bytes_for(in, 8), page, p.qbuf_offset) || !place(bytes_for(dn, 8), page, p.rbuf_offset))
            return false;
        i.scratch_bytes = cursor - p.shared_offset;
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
                       p.pcap, p.newton_bytes, uq.consumer_info.mul.arithmetic_id, tq.consumer_info.mul.arithmetic_id,
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
void release_products(Binding &b) {
    if (b.uprod) {
        sbn3_mul_unbind(b.uprod);
        b.uprod = nullptr;
    }
    if (b.tprod) {
        sbn3_mul_unbind(b.tprod);
        b.tprod = nullptr;
    }
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
    for (auto *l : {&b.utab, &b.uwork, &b.ttab, &b.twork})
        if (l->token) {
            b.arena->release(*l);
            *l = {};
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
    memcpy(b.D, D, dn * 8);
    b.shift = unsigned(__builtin_clzll(D[dn - 1]));
    if (b.shift)
        divrem_words::shift_left(b.Dn, D, dn, b.shift);
    else
        memcpy(b.Dn, D, dn * 8);
    // Block inverse of the top in limbs: the Newton binding lives in the
    // shared union, which is unleased at this point.
    sbn3_newton_binding *inverse = nullptr;
    sbn3_newton_bind(&b.nplan, b.arena, b.offset + p.shared_offset, b.team, &inverse);
    sbn3_newton_inputs inputs{};
    inputs.denominator = {b.Dn + dn - in, in};
    sbn3_newton_execute(inverse, &inputs, {b.U, in + 1});
    sbn3_newton_unbind(inverse);
    require(b.U[in] == 1, SBN3_FATAL_MATH, "division block inverse framing");
    b.dinv = divrem_words::invert_pi1(b.Dn[dn - 1], b.Dn[dn - 2]);
    // Persistent products and divisor-side spectra.
    b.shared = b.arena->acquire(b.offset + p.shared_offset, p.info.shared_bytes);
    b.shared_leased = true;
    const auto &um = b.uq.consumer_info.mul, &tm = b.tq.consumer_info.mul;
    b.utab = b.arena->acquire(b.offset + p.utab_offset, um.table_bytes);
    b.uwork = b.arena->acquire(b.offset + p.uwork_offset, um.workspace_bytes);
    b.ttab = b.arena->acquire(b.offset + p.ttab_offset, tm.table_bytes);
    b.twork = b.arena->acquire(b.offset + p.twork_offset, tm.workspace_bytes);
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
    sbn3_product_bind(&b.uq.consumer, b.arena, &b.utab, &b.uwork, b.team, b.uspec, nullptr, &b.uprod);
    sbn3_product_bind(&b.tq.consumer, b.arena, &b.ttab, &b.twork, b.team, b.dspec, nullptr, &b.tprod);
    if (b.uspec)
        sbn3_spectrum_compute(b.uprod, b.uspec, {b.U, in + 1});
    if (b.dspec)
        sbn3_spectrum_compute(b.tprod, b.dspec, {b.Dn, dn});
    b.prepared = true;
}
// One exact quotient limb by word division. x holds X = R*B + n0 on dn+1
// limbs with R < D', so q = floor(X/D') < B. The 3/2 estimate from the top
// limbs is q or q+1 (the sbpi1 schoolbook step); every team part subtracts
// its own slice of qhat*D' and the slice carries are stitched exactly.
uint64_t word_step(sbn3_team *team, uint64_t *x, const uint64_t *Dn, size_t dn, uint64_t dinv) {
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
        borrow -= parallel_limbs::add_to(team, x, dn + 1, Dn, dn);
    }
    require(!x[dn], SBN3_FATAL_MATH, "division head remainder");
    return q;
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
            Q[pos + j] = word_step(team, xbuf + j, Dn, dn, b.dinv);
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
        sbn3_product_execute(b.uprod, &pin, {pbuf, p.pcap});
        ++products;
        parallel_limbs::copy(team, qbuf, pbuf + in, ic);
        if (!parallel_limbs::zero(team, pbuf + in + ic, in + 1 - ic))
            parallel_limbs::fill(team, qbuf, ic, UINT64_MAX);
        parallel_limbs::fill(team, qbuf + ic, in - ic);
        // X = R*B^ic + N'block, folded once when it exceeds the ring.
        shifted_span(team, xbuf, N, nn, pos, ic, s);
        if (ic + dn <= L) {
            parallel_limbs::copy(team, xbuf + ic, rbuf, dn);
            parallel_limbs::fill(team, xbuf + ic + dn, L - ic - dn);
        } else {
            parallel_limbs::copy(team, xbuf + ic, rbuf, L - ic);
            if (parallel_limbs::add_to(team, xbuf, L, rbuf + (L - ic), ic + dn - L))
                parallel_limbs::add_word(xbuf, L, 1);
        }
        // T = qhat * D' (mod B^ring-1 on the cyclic recipe).
        sbn3_product_inputs tin{};
        if (!b.dspec)
            tin.a = {Dn, dn};
        tin.b = {qbuf, in};
        sbn3_product_execute(b.tprod, &tin, {tbuf, p.tcap});
        ++products;
        // E = X - T, |E| < 7 D' < B^(dn+1): exact signed lift.
        const uint64_t borrow = parallel_limbs::sub_from(team, xbuf, L, tbuf, L);
        bool negative;
        if (cyclic) {
            if (borrow)
                limbs::cyclic_sub_power(xbuf, L, 0);
            negative = parallel_limbs::cyclic_absolute(team, xbuf, L);
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
        divrem_words::shift_right(R, rbuf, dn, s);
    else
        parallel_limbs::copy(team, R, rbuf, dn);
    out.corrections = corrections;
    b.metrics.products_executed = products;
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
    else if (dn <= divrem_tuning::schoolbook_max_divisor || i.quotient_limbs <= divrem_tuning::schoolbook_max_quotient)
        i.algorithm = SBN3_DIVREM_SCHOOLBOOK;
    else
        i.algorithm = SBN3_DIVREM_BARRETT;
    Stored s{};
    if (i.algorithm == SBN3_DIVREM_BARRETT) {
        if (!assemble(p, false, s.uq, s.tq))
            return SBN3_UNSUPPORTED;
        const auto no = newton_options(p);
        const auto status = sbn3_newton_query(SBN3_NEWTON_INVERSE, p.in, &no, &s.nplan, &s.ninfo);
        if (status != SBN3_SUPPORTED)
            return status;
    }
    if (!layout(p, s.uq, s.tq, &s.ninfo))
        return SBN3_QUERY_CAPACITY;
    *info = p.info;
    if (p.options.memory_budget && p.info.storage_bytes > p.options.memory_budget)
        return SBN3_QUERY_CAPACITY;
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
    require(d.size == dn && !memcmp(d.data, b.D, dn * 8), SBN3_FATAL_ARGUMENT, "integer division prepared divisor");
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
