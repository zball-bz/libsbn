#include "product/backend.hpp"
#include "product/fft_backend.hpp"
#include "runtime/team.hpp"
#include "runtime/linux/sync.hpp"
#include "backend/u52/kernels.hpp"
#include "backend/pq16/kernels.hpp"
#include "core/x86_64/word.hpp"
#include <new>
#include <time.h>
#include <algorithm>
#ifndef SBN3_SHORT_TIMING
#define SBN3_SHORT_TIMING 0
#endif
#ifndef SBN3_SHORT_STATS
#define SBN3_SHORT_STATS (SBN3_CHECK_SMALL || SBN3_SHORT_TIMING)
#endif
namespace sbn::v3 {
namespace {
constexpr uint64_t magic = 0x53424e3353485254ULL, id = 100, binding_magic = 0x53424e3342494e44ULL;
struct Plan {
    uint64_t marker = magic, backend = id, seal = 0;
    size_t an = 0, bn = 0, ring = 0;
    unsigned kind = 0;
    sbn3_window_certificate window{};
    pq16::Shape shape{};
    sbn3_mul_info info{};
};
struct Binding {
    sbn3_mul_binding header{&short_backend(), binding_magic};
    Plan plan;
    Arena *arena;
    sbn3_team *team;
    sbn3_lease tables, work;
    Frame frame;
    const pq16::Tables *pq = nullptr;
    u52::Prepared u52_buffers{};
    void (*kernel)(sbn3_mul_binding *, const uint64_t *, const uint64_t *, uint64_t *) = nullptr;
    uint64_t executions = 0, last_ns = 0;
    uint32_t active = 0;
    size_t base_bytes = 0, steady_bytes = 0, table_used = 0;
    uintptr_t resources_begin = UINTPTR_MAX, resources_end = 0;
    Binding(const Plan &p, Arena &a, sbn3_team &t, const sbn3_lease &tl, const sbn3_lease &w)
        : plan(p), arena(&a), team(&t), tables(tl), work(w), frame(a, w) {}
};
static_assert(sizeof(Plan) <= sizeof(sbn3_mul_plan));
uint64_t hash(uint64_t h, uint64_t x) {
    for (unsigned j = 0; j < 8; ++j) {
        h = (h ^ (x & 255)) * 1099511628211ULL;
        x >>= 8;
    }
    return h;
}
uint64_t seal(const Plan &p) {
    uint64_t h = 1469598103934665603ULL;
    for (uint64_t x : {p.marker,
                       p.backend,
                       uint64_t(p.an),
                       uint64_t(p.bn),
                       uint64_t(p.ring),
                       uint64_t(p.kind),
                       p.window.offset_bits,
                       p.window.width_bits,
                       p.window.error_bits,
                       uint64_t(p.shape.nfull),
                       uint64_t(p.shape.branch),
                       uint64_t(p.shape.radix),
                       uint64_t(p.shape.centered),
                       uint64_t(p.shape.balanced),
                       uint64_t(p.shape.recipe),
                       uint64_t(p.shape.bits),
                       uint64_t(p.info.algorithm),
                       uint64_t(p.info.workers),
                       uint64_t(p.info.table_bytes),
                       uint64_t(p.info.workspace_bytes),
                       uint64_t(p.info.per_worker_bytes),
                       uint64_t(p.info.output_limbs),
                       p.info.basis_id,
                       p.info.arithmetic_id,
                       p.info.execution_id})
        h = hash(h, x);
    return h;
}
Plan load(const sbn3_mul_plan &p) {
    Plan q{};
    memcpy(&q, p.opaque, sizeof q);
    require(q.marker == magic && q.backend == id && q.seal == seal(q), SBN3_FATAL_ARGUMENT,
            "short plan identity");
    return q;
}
uint64_t now() {
    if constexpr (SBN3_SHORT_TIMING) {
        timespec t{};
        clock_gettime(CLOCK_MONOTONIC, &t);
        return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
    }
    return 0;
}
sbn3_query_result query_product(const sbn3_product_request &r, const sbn3_mul_options &o, sbn3_mul_plan &out,
                                sbn3_product_info &result) {
    result = {};
    const bool mid = r.kind == SBN3_PRODUCT_TMP,
               window = r.kind == SBN3_PRODUCT_LOW || r.kind == SBN3_PRODUCT_HIGH;
    if (r.kind != SBN3_PRODUCT_MUL && r.kind != SBN3_PRODUCT_SQR && !mid && !window)
        return SBN3_UNSUPPORTED;
    if (r.negacyclic || (!window && !r.cyclic_limbs && r.window_limbs))
        return SBN3_UNSUPPORTED;
    if (mid && (o.algorithm != SBN3_MUL_U52 || !r.a_limbs || r.a_limbs > 300 || r.b_limbs < r.a_limbs ||
                r.b_limbs > 8192))
        return SBN3_UNSUPPORTED;
    if (window && (o.algorithm != SBN3_MUL_SCALAR || r.a_limbs > 256 || r.b_limbs > 256 || !r.window_limbs ||
                   r.window_limbs > r.a_limbs + r.b_limbs))
        return SBN3_UNSUPPORTED;
    if (r.cyclic_limbs && (o.algorithm != SBN3_MUL_SCALAR || r.cyclic_limbs > 512 ||
                           r.a_limbs > r.cyclic_limbs || r.b_limbs > r.cyclic_limbs || mid || window))
        return SBN3_UNSUPPORTED;
    if (r.a1_limbs || r.b1_limbs || r.cached_a[0] || r.cached_a[1] ||
        (r.kind == SBN3_PRODUCT_SQR && r.b_limbs))
        return SBN3_UNSUPPORTED;
    if (!o.workers || o.workers > 32 || o.borrow_output > 2 || o.prime_count || o.prime_batch ||
        (o.trunk_bits && o.algorithm != SBN3_MUL_PQ16) || o.column_log2 || o.row_log2 || o.crt_mode ||
        o.codec_mode || o.fused_start_skew_us)
        return SBN3_UNSUPPORTED;
    if (o.algorithm < SBN3_MUL_SCALAR || o.algorithm > SBN3_MUL_PQ16)
        return SBN3_UNSUPPORTED;
    if (r.cyclic_limbs && r.window_limbs &&
        (r.window_limbs > r.cyclic_limbs ||
         r.a_limbs + (r.kind == SBN3_PRODUCT_SQR ? r.a_limbs : r.b_limbs) >= r.cyclic_limbs))
        return SBN3_UNSUPPORTED;
    Plan q{};
    q.an = r.a_limbs;
    q.bn = r.kind == SBN3_PRODUCT_SQR ? r.a_limbs : r.b_limbs;
    q.kind = r.kind;
    q.ring = r.cyclic_limbs;
    const size_t limit = o.algorithm == SBN3_MUL_SCALAR ||
                         (o.algorithm == SBN3_MUL_U52 && u52::streams(q.an,q.bn))
                             ? size_t(1) << 31 : size_t(1) << 20;
    if (q.an > limit || q.bn > limit)
        return SBN3_QUERY_CAPACITY;
    auto &i = q.info;
    i.algorithm = o.algorithm;
    i.workers = o.workers;
    i.output_limbs = q.ring ? (r.window_limbs ? r.window_limbs : q.ring) : q.an + q.bn;
    i.output_alignment = 8;
    i.workspace_alignment = 128;
    i.table_bytes = 128;
    i.trunk_bits = o.algorithm == SBN3_MUL_U52 ? 52 : o.algorithm == SBN3_MUL_PQ16 ? 16 : 64;
    i.digit_words = 1;
    i.nat = (q.an * 64 + i.trunk_bits - 1) / i.trunk_bits;
    i.nyt = (q.bn * 64 + i.trunk_bits - 1) / i.trunk_bits;
    if (o.algorithm == SBN3_MUL_U52)
        i.per_worker_bytes = mid ? u52::middle_scratch_bytes(q.an, q.bn) : u52::scratch_bytes(q.an, q.bn);
    if (mid) {
        const size_t rn = i.nyt - i.nat + 1;
        i.output_limbs = (52 * (rn + 2) + 63) / 64;
        q.window = {52 * (i.nat - 1), 52 * rn,
                    53 + uint64_t(i.nat > 1 ? 64 - __builtin_clzll(i.nat - 1) : 0)};
    }
    if (q.ring && r.window_limbs)
        q.window = {0, 64 * r.window_limbs, 0};
    if (window) {
        i.output_limbs = r.window_limbs;
        q.window = {r.kind == SBN3_PRODUCT_HIGH ? 64 * (q.an + q.bn - r.window_limbs) : 0,
                    64 * r.window_limbs, 0};
    }
    if (o.algorithm == SBN3_MUL_PQ16) {
        size_t control = 0;
        align_size(sizeof(Binding), 128, control);
        const bool square = q.kind == SBN3_PRODUCT_SQR;
        q.shape = pq16::select(q.an, q.bn, o.workers, unsigned(o.trunk_bits),
                               o.workspace_budget > control ? o.workspace_budget - control : 0, square);
        if (!q.shape.nfull && o.workspace_budget)
            q.shape = pq16::select(q.an, q.bn, o.workers, unsigned(o.trunk_bits), 0, square);
        if (!q.shape.nfull)
            return SBN3_UNSUPPORTED;
        i.trunk_bits = q.shape.bits;
        i.nat = (q.an * 64 + i.trunk_bits - 1) / i.trunk_bits;
        i.nyt = (q.bn * 64 + i.trunk_bits - 1) / i.trunk_bits;
        i.per_worker_bytes = pq16::scratch_bytes(q.shape, q.an, q.bn, o.workers, square);
        i.table_bytes = pq16::table_bytes(q.shape);
        i.transform_trunks = 2 * q.shape.nfull;
        i.C = q.shape.radix;
        i.M2 = q.shape.branch;
        i.table_entries = q.shape.branch;
        i.codec_mode = unsigned(q.shape.recipe) | (q.shape.balanced ? 8u : 0u);
    }
    size_t base = 0;
    align_size(sizeof(Binding), 128, base);
    i.workspace_bytes = base + i.per_worker_bytes;
    i.basis_id = hash(hash(hash(1469598103934665603ULL, i.algorithm), q.shape.nfull), q.shape.centered);
    if (q.shape.bits != 16)
        i.basis_id = hash(i.basis_id, q.shape.bits);
    i.arithmetic_id = hash(hash(hash(i.basis_id, q.an), q.bn), q.kind);
    if (window)
        i.arithmetic_id = hash(i.arithmetic_id, r.window_limbs);
    if (q.ring)
        i.arithmetic_id = hash(hash(i.arithmetic_id, q.ring), r.window_limbs);
    i.execution_id = hash(hash(hash(i.arithmetic_id, o.workers), i.workspace_bytes), i.table_bytes);
    if (q.shape.recipe != pq16::Recipe::PfaPQ)
        i.execution_id = hash(i.execution_id, uint64_t(q.shape.recipe));
    if (q.shape.balanced)
        i.basis_id = hash(i.basis_id, 8u), i.arithmetic_id = hash(i.arithmetic_id, 8u),
        i.execution_id = hash(i.execution_id, 8u);
    result.kind = r.kind;
    result.mul = i;
    result.window = q.window;
    result.cyclic_limbs = q.ring;
    if (o.algorithm == SBN3_MUL_PQ16) {
        sbn3_spectrum_desc future{};
        if (fft_spectrum_description(q.shape, q.an, SBN3_SPECTRUM_COLUMNS, 0, future) == SBN3_SUPPORTED) {
            result.spectrum_bytes = future.storage_bytes;
            result.spectrum_alignment = 128;
        }
    }
    if (o.workspace_budget && i.workspace_bytes > o.workspace_budget)
        return SBN3_QUERY_CAPACITY;
    q.seal = seal(q);
    memset(&out, 0, sizeof out);
    memcpy(out.opaque, &q, sizeof q);
    return SBN3_SUPPORTED;
}
sbn3_query_result query(const sbn3_product_spec &s, const sbn3_mul_options &o, sbn3_mul_plan &p,
                        sbn3_mul_info &i) {
    sbn3_product_request r{};
    r.a_limbs = s.a_limbs;
    r.b_limbs = s.b_limbs;
    sbn3_product_info result{};
    auto status = query_product(r, o, p, result);
    i = result.mul;
    return status;
}
Binding &get(sbn3_mul_binding *b) {
    return *reinterpret_cast<Binding *>(b);
}
const Binding &get(const sbn3_mul_binding *b) {
    return *reinterpret_cast<const Binding *>(b);
}
template <unsigned algorithm>
void fast_mul(sbn3_mul_binding *, const uint64_t *, const uint64_t *, uint64_t *);
template <unsigned n>
void fast_equal_size(sbn3_mul_binding *, const uint64_t *, const uint64_t *, uint64_t *);
void execute_ptrs_checked(sbn3_mul_binding *, const uint64_t *, const uint64_t *, uint64_t *);
void fast_window(sbn3_mul_binding *, const uint64_t *, const uint64_t *, uint64_t *);
void idle(const Binding &b) {
    require(pthread_equal(b.team->creator, pthread_self()) && !b.team->busy && !b.active, SBN3_FATAL_TEAM,
            "short binding owner/active");
}
void bind_product(const sbn3_mul_plan &p, sbn3_arena &a, const sbn3_lease &t, const sbn3_lease &w,
                  sbn3_team &team, const sbn3_spectrum *s0, const sbn3_spectrum *s1, sbn3_mul_binding **out) {
    auto q = load(p);
    require(!s0 && !s1, SBN3_FATAL_ARGUMENT, "short products have no spectrum");
    require(team.arena == &a && team.width >= q.info.workers && pthread_equal(team.creator, pthread_self()) &&
                !team.busy,
            SBN3_FATAL_TEAM, "short binding team");
    require(t.bytes >= q.info.table_bytes && w.bytes >= q.info.workspace_bytes &&
                !(uintptr_t(t.data) & 127) && !(uintptr_t(w.data) & 127) &&
                !overlaps(t.data, t.bytes, w.data, w.bytes),
            SBN3_FATAL_WORKSPACE, "short binding spans");
    for (auto l : {team.storage})
        require(!overlaps(l.data, l.bytes, t.data, t.bytes) && !overlaps(l.data, l.bytes, w.data, w.bytes),
                SBN3_FATAL_WORKSPACE, "short team control alias");
    for (unsigned k = 1; k < team.width; ++k)
        require(!overlaps(team.stacks[k].data, team.stacks[k].bytes, t.data, t.bytes) &&
                    !overlaps(team.stacks[k].data, team.stacks[k].bytes, w.data, w.bytes),
                SBN3_FATAL_WORKSPACE, "short stack alias");
    a.claim_unshared(t);
    a.claim_unshared(w);
    auto *b = new (w.data) Binding(q, a, team, t, w);
    auto enclose = [&](const sbn3_lease &l) {
        if (l.bytes) {
            const auto begin = uintptr_t(l.data);
            b->resources_begin = std::min(b->resources_begin, begin);
            b->resources_end = std::max(b->resources_end, begin + l.bytes);
        }
    };
    enclose(t);
    enclose(w);
    enclose(team.storage);
    for (unsigned k = 1; k < team.width; ++k)
        enclose(team.stacks[k]);
    align_size(sizeof(Binding), 128, b->base_bytes);
    b->frame.allocate(b->base_bytes, 128);
    if (q.info.algorithm == SBN3_MUL_U52 && q.kind != SBN3_PRODUCT_TMP)
        b->u52_buffers = u52::prepare_buffers(b->frame, q.an, q.bn);
    b->steady_bytes = b->frame.used();
    if (q.info.algorithm == SBN3_MUL_PQ16) {
        Frame tf(a, t);
        ComputeLease setup(a);
        b->pq = pq16::prepare(tf, q.shape);
        b->table_used = tf.used();
    }
    {
        if (q.ring || q.kind == SBN3_PRODUCT_TMP || q.kind == SBN3_PRODUCT_LOW || q.kind == SBN3_PRODUCT_HIGH)
            b->kernel = fast_window;
        else if (q.info.algorithm == SBN3_MUL_U52)
            b->kernel = fast_mul<SBN3_MUL_U52>;
        else if (q.info.algorithm == SBN3_MUL_PQ16 && q.info.workers == 1)
            b->kernel = fast_mul<SBN3_MUL_PQ16>;
        else if (q.info.algorithm == SBN3_MUL_SCALAR) {
            b->kernel = fast_mul<SBN3_MUL_SCALAR>;
            if (q.an == q.bn)
                switch (q.an) {
                case 1:
                    b->kernel = fast_equal_size<1>;
                    break;
                case 2:
                    b->kernel = fast_equal_size<2>;
                    break;
                case 3:
                    b->kernel = fast_equal_size<3>;
                    break;
                case 4:
                    b->kernel = fast_equal_size<4>;
                    break;
                case 5:
                    b->kernel = fast_equal_size<5>;
                    break;
                case 6:
                    b->kernel = fast_equal_size<6>;
                    break;
                }
        }
    }
    b->header.execute_fast = SBN3_CHECK_SMALL || !b->kernel ? execute_ptrs_checked : b->kernel;
    a.release(w);
    *out = &b->header;
}
void bind(const sbn3_mul_plan &p, sbn3_arena &a, const sbn3_lease &t, const sbn3_lease &w, sbn3_team &team,
          sbn3_mul_binding **out) {
    bind_product(p, a, t, w, team, nullptr, nullptr, out);
}
void validate(const Binding &b, const sbn3_product_inputs &in, sbn3_limbs out) {
    const auto &q = b.plan;
    require(in.a.count == q.an && !in.a1.count && !in.a1.data && !in.b1.count && !in.b1.data,
            SBN3_FATAL_ARGUMENT, "short input lengths");
    require(q.kind == SBN3_PRODUCT_SQR ? (!in.b.count && !in.b.data) : in.b.count == q.bn,
            SBN3_FATAL_ARGUMENT, "short second input");
    require(out.capacity >= q.info.output_limbs, SBN3_FATAL_SIZE, "short output capacity",
            q.info.output_limbs, out.capacity);
    const auto a = in.a, bv = q.kind == SBN3_PRODUCT_SQR ? in.a : in.b;
    const size_t ab = a.count * 8, bb = bv.count * 8, rb = q.info.output_limbs * 8;
    valid_span(a.data, ab, "short A");
    valid_span(bv.data, bb, "short B");
    valid_span(out.data, rb, "short result");
    require(!(uintptr_t(a.data) & 7) && !(uintptr_t(bv.data) & 7) && !(uintptr_t(out.data) & 7) &&
                !overlaps(out.data, rb, a.data, ab) && !overlaps(out.data, rb, bv.data, bb),
            SBN3_FATAL_ARGUMENT, "short alignment/alias");
    // This is only a fast negative test. Values in the envelope's gaps
    // remain legal and take the exact per-resource checks below.
    const auto *begin = reinterpret_cast<const void *>(b.resources_begin);
    const size_t bytes = b.resources_end - b.resources_begin;
    if (!overlaps(begin, bytes, a.data, ab) && !overlaps(begin, bytes, bv.data, bb) &&
        !overlaps(begin, bytes, out.data, rb))
        return;
    auto disjoint = [&](const sbn3_lease &l) {
        require(!overlaps(l.data, l.bytes, a.data, ab) && !overlaps(l.data, l.bytes, bv.data, bb) &&
                    !overlaps(l.data, l.bytes, out.data, rb),
                SBN3_FATAL_ARGUMENT, "short resource/value alias");
    };
    disjoint(b.tables);
    disjoint(b.work);
    disjoint(b.team->storage);
    for (unsigned k = 1; k < b.team->width; ++k)
        disjoint(b.team->stacks[k]);
}
struct Run {
    Binding *binding;
    sbn3_product_inputs input;
    sbn3_limbs out;
};
void compute(Run &r, sbn3_team_scope *scope) {
    auto &b = *r.binding;
    const auto &q = b.plan;
    uint32_t expected = 0;
    // A standalone W1 call is already excluded by creator + idle checks
    // and team.busy. Borrowed child scopes can race each other and retain
    // the independent binding CAS (including two W1 children).
    if (scope)
        require(
            __atomic_compare_exchange_n(&b.active, &expected, 1, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED),
            SBN3_FATAL_LIFETIME, "concurrent short binding");
    const auto a = r.input.a, y = q.kind == SBN3_PRODUCT_SQR ? a : r.input.b;
    if (q.info.algorithm != SBN3_MUL_PQ16 || q.info.workers == 1)
        b.kernel(&b.header, a.data, y.data, r.out.data);
    else {
        [[maybe_unused]] const auto start = now();
        pq16::multiply(r.out.data, a.data, a.count, y.data, y.count, *b.pq, b.frame, scope);
        if constexpr (SBN3_SHORT_STATS) {
            b.last_ns = now() - start;
            ++b.executions;
        }
    }
    require(b.frame.used() == b.steady_bytes, SBN3_FATAL_WORKSPACE, "short scratch rewind");
    if (scope)
        __atomic_store_n(&b.active, 0, __ATOMIC_RELEASE);
}
// Direct small-word cyclic convolution, modulo B^n-1. A row may cross
// the ring seam once; each arithmetic carry is itself folded at that seam.
// No full linear product or transform storage is materialized.
void cyclic_add(uint64_t *r, size_t n, size_t at, uint64_t carry) {
    while (carry) {
        const __uint128_t v = __uint128_t(r[at]) + carry;
        r[at] = uint64_t(v);
        carry = uint64_t(v >> 64);
        if (++at == n)
            at = 0;
    }
}
void cyclic_words(uint64_t *r, size_t n, const uint64_t *a, size_t an, const uint64_t *b, size_t bn) {
    memset(r, 0, n * 8);
    if (!an || !bn)
        return;
    for (size_t j = 0; j < bn; ++j) {
        if (!b[j])
            continue;
        const size_t first = std::min(an, n - j);
        uint64_t carry = sbn3i_addmul_1(r + j, a, long(first), b[j]);
        cyclic_add(r, n, j + first == n ? 0 : j + first, carry);
        if (first < an) {
            const size_t rest = an - first;
            carry = sbn3i_addmul_1(r, a + first, long(rest), b[j]);
            cyclic_add(r, n, rest, carry);
        }
    }
    uint64_t all = UINT64_MAX;
    for (size_t j = 0; j < n; ++j)
        all &= r[j];
    if (all == UINT64_MAX)
        memset(r, 0, n * 8);
}
// Exact short low triangle: excluded high rows never execute. Exact high
// basecase streams all lower diagonals' carries through three words and only
// stores the requested top window; this is not a claimed subquadratic high product.
void fast_window(sbn3_mul_binding *p, const uint64_t *a, const uint64_t *b, uint64_t *out) {
    auto &v = get(p);
    const auto &q = v.plan;
    [[maybe_unused]] const auto start = now();
    if (q.ring && !q.window.width_bits)
        cyclic_words(out, q.ring, a, q.an, b, q.bn);
    else if (q.kind == SBN3_PRODUCT_TMP)
        u52::middle(out, a, q.an, b, q.bn, v.frame);
    else if (q.kind == SBN3_PRODUCT_LOW || q.ring) {
        const size_t n = q.info.output_limbs;
        memset(out, 0, n * 8);
        for (size_t j = 0; j < q.bn && j < n; ++j) {
            const size_t len = std::min(q.an, n - j);
            if (!len)
                continue;
            const auto carry = sbn3i_addmul_1(out + j, a, long(len), b[j]);
            if (j + len < n)
                out[j + len] = carry;
        }
    } else {
        const size_t off = q.window.offset_bits / 64;
        uint64_t c0 = 0, c1 = 0, c2 = 0;
        for (size_t k = 0; k < q.an + q.bn; ++k) {
            const size_t lo = k >= q.bn ? k - q.bn + 1 : 0, hi = std::min(k + 1, q.an);
            for (size_t j = lo; j < hi; ++j) {
                const __uint128_t t = __uint128_t(a[j]) * b[k - j];
                const __uint128_t x = __uint128_t(c0) + uint64_t(t);
                c0 = uint64_t(x);
                const __uint128_t y = __uint128_t(c1) + uint64_t(t >> 64) + uint64_t(x >> 64);
                c1 = uint64_t(y);
                c2 += uint64_t(y >> 64);
            }
            if (k >= off)
                out[k - off] = c0;
            c0 = c1;
            c1 = c2;
            c2 = 0;
        }
    }
    if constexpr (SBN3_SHORT_STATS) {
        v.last_ns = now() - start;
        ++v.executions;
    }
}
template <unsigned algorithm>
void fast_mul(sbn3_mul_binding *p, const uint64_t *a, const uint64_t *y, uint64_t *out) {
    auto &b = get(p);
    const auto &q = b.plan;
    [[maybe_unused]] const auto start = now();
    if constexpr (algorithm == SBN3_MUL_SCALAR)
        mul_basecase_assumed(out, a, q.an, y, q.bn);
    else if constexpr (algorithm == SBN3_MUL_U52)
        u52::multiply_prepared(out, a, q.an, y, q.bn, b.u52_buffers, b.frame);
    else
        pq16::multiply(out, a, q.an, y, q.bn, *b.pq, b.frame, nullptr);
    if constexpr (SBN3_SHORT_STATS) {
        b.last_ns = now() - start;
        ++b.executions;
    }
}
template <unsigned n>
void fast_equal_size(sbn3_mul_binding *p, const uint64_t *a, const uint64_t *b, uint64_t *out) {
    (void)p;
    [[maybe_unused]] const auto start = now();
    if constexpr (n == 1) {
        const __uint128_t q = __uint128_t(a[0]) * b[0];
        out[0] = uint64_t(q);
        out[1] = uint64_t(q >> 64);
    } else if constexpr (n == 2) {
        const __uint128_t x = __uint128_t(a[0]) * b[0], y = __uint128_t(a[0]) * b[1],
                          z = __uint128_t(a[1]) * b[0], w = __uint128_t(a[1]) * b[1];
        const __uint128_t s = (x >> 64) + uint64_t(y) + uint64_t(z),
                          t = w + (y >> 64) + (z >> 64) + (s >> 64);
        out[0] = uint64_t(x);
        out[1] = uint64_t(s);
        out[2] = uint64_t(t);
        out[3] = uint64_t(t >> 64);
    } else
        sbn3i_mul_basecase_le6(out, a, n, b, n);
    if constexpr (SBN3_SHORT_STATS) {
        auto &v = get(p);
        v.last_ns = now() - start;
        ++v.executions;
    }
}
void action(void *arg, sbn3_team_scope *parent) {
    auto &r = *static_cast<Run *>(arg);
    auto &b = *r.binding;
    const auto &q = b.plan;
    require_scope_leader(parent);
    require(parent->team == b.team && parent->width >= q.info.workers, SBN3_FATAL_TEAM,
            "short execution scope");
    sbn3_team_scope scope{parent->team, parent->first, q.info.workers, false, parent->epoch};
    compute(r, &scope);
}
void product_execute(sbn3_mul_binding *p, sbn3_team_scope *scope, const sbn3_product_inputs &in,
                     sbn3_limbs out) {
    auto &b = get(p);
    if constexpr (!SBN3_CHECK_SMALL) {
        if (b.plan.info.algorithm != SBN3_MUL_PQ16 || b.plan.info.workers == 1) {
            b.kernel(p, in.a.data, b.plan.kind == SBN3_PRODUCT_SQR ? in.a.data : in.b.data, out.data);
            return;
        }
    }
    if (scope) {
        require_scope_leader(scope);
        require(scope->team == b.team && scope->width == b.plan.info.workers, SBN3_FATAL_TEAM,
                "short borrowed width");
    } else
        idle(b);
    validate(b, in, out);
    Run run{&b, in, out};
    if (scope)
        action(&run, scope);
    else if (b.plan.info.workers == 1) {
        // No callbacks or worker tasks occur in a serial short kernel.
        // Retain the arena episode and team/binding exclusion without
        // constructing a root scope or changing the thread's team TLS.
        b.team->busy = true;
        {
            ComputeLease episode(*b.arena);
            compute(run, nullptr);
            os::publish_stores();
        }
        b.team->busy = false;
    } else
        sbn3_team_run(b.team, action, &run);
}
void execute(sbn3_mul_binding *p, sbn3_const_limbs a, sbn3_const_limbs b, sbn3_limbs out) {
    auto &binding = get(p);
    if constexpr (!SBN3_CHECK_SMALL)
        if (binding.plan.info.algorithm != SBN3_MUL_PQ16 || binding.plan.info.workers == 1) {
            binding.kernel(p, a.data, b.data, out.data);
            return;
        }
    require(binding.plan.kind == SBN3_PRODUCT_MUL, SBN3_FATAL_ARGUMENT, "mul recipe");
    product_execute(p, nullptr, {a, b, {}, {}}, out);
}
void execute_ptrs_checked(sbn3_mul_binding *p, const uint64_t *a, const uint64_t *y, uint64_t *out) {
    const auto &q = get(p).plan;
    execute(p, {a, q.an}, {y, q.bn}, {out, q.info.output_limbs});
}
void execute_scope(sbn3_mul_binding *p, sbn3_team_scope *s, sbn3_const_limbs a, sbn3_const_limbs b,
                   sbn3_limbs out) {
    if constexpr (!SBN3_CHECK_SMALL)
        if (get(p).plan.info.algorithm != SBN3_MUL_PQ16 || get(p).plan.info.workers == 1) {
            get(p).kernel(p, a.data, b.data, out.data);
            return;
        }
    require(get(p).plan.kind == SBN3_PRODUCT_MUL, SBN3_FATAL_ARGUMENT, "mul recipe");
    product_execute(p, s, {a, b, {}, {}}, out);
}
void metrics(const sbn3_mul_binding *p, sbn3_mul_metrics &m) {
    const auto &b = get(p);
    idle(b);
    m = {b.frame.peak() - b.base_bytes, b.table_used, b.frame.peak(), b.executions, {b.last_ns, 0, 0, 0}};
}
void product_metrics(const sbn3_mul_binding *p, sbn3_product_metrics &m) {
    m = {};
    metrics(p, m.mul);
}
void unbind(sbn3_mul_binding *p) {
    auto &b = get(p);
    idle(b);
    auto *a = b.arena;
    const auto t = b.tables;
    b.header.marker = 0;
    b.~Binding();
    a->release(t);
}
struct Program {
    Plan plan{};
    const pq16::Tables *tables=nullptr;
};
size_t program_bytes(const sbn3_mul_plan &p) {
    const auto q=load(p);
    if(q.kind!=SBN3_PRODUCT_MUL || q.ring)return 0;
    size_t header=0,total=0;
    if(!align_size(sizeof(Program),128,header) || !add_size(header,q.info.table_bytes,total))return 0;
    return total;
}
const void *program_prepare(const sbn3_mul_plan &p,Frame &f) {
    const auto q=load(p);require(q.kind==SBN3_PRODUCT_MUL && !q.ring,SBN3_FATAL_ARGUMENT,"short program recipe");
    auto *v=::new(f.allocate(sizeof(Program),128)) Program{};v->plan=q;
    if(q.info.algorithm==SBN3_MUL_PQ16)v->tables=pq16::prepare(f,q.shape);
    return v;
}
SharedPreparation program_tables(const sbn3_mul_plan &p) {
    const auto q=load(p);const auto &s=q.shape;
    if(q.kind!=SBN3_PRODUCT_MUL || q.ring || q.info.algorithm!=SBN3_MUL_PQ16)return {};
    return {{0x5051313654423031ULL,s.nfull,s.branch,s.radix,s.centered,uint64_t(s.recipe),s.bits,s.balanced},
            q.info.table_bytes,128};
}
size_t program_local_bytes(const sbn3_mul_plan &) {return (sizeof(Program)+127)&~size_t(127);}
const void *program_tables_prepare(const sbn3_mul_plan &p,Frame &f) {return pq16::prepare(f,load(p).shape);}
const void *program_prepare_shared(const sbn3_mul_plan &p,Frame &f,const void *shared) {
    auto *v=::new(f.allocate(sizeof(Program),128)) Program{};v->plan=load(p);
    v->tables=static_cast<const pq16::Tables *>(shared);return v;
}
void program_execute(const void *p,Frame &f,sbn3_team_scope *scope,sbn3_const_limbs a,sbn3_const_limbs b,sbn3_limbs out) {
    const auto &v=*static_cast<const Program *>(p);
    if(v.plan.info.algorithm==SBN3_MUL_SCALAR)mul_basecase_assumed(out.data,a.data,a.count,b.data,b.count);
    else if(v.plan.info.algorithm==SBN3_MUL_U52)u52::multiply(out.data,a.data,a.count,b.data,b.count,f);
    else pq16::multiply(out.data,a.data,a.count,b.data,b.count,*v.tables,f,scope);
}
unsigned program_contract(const sbn3_mul_plan &p) {
    const auto q=load(p);
    // Centered PFA can consult raw input digits while fixing emission. Do not
    // advertise consume for it or direct u64 basecase. Short U52 converts both
    // inputs first. A long rectangular U52 product streams its long input;
    // bounded inputs can enter that path even if the planned shape does not.
    return program_bounded_inputs |
        ((q.info.algorithm==SBN3_MUL_U52 && std::max(q.an,q.bn)<=u52::strip_limbs) ||
         (q.info.algorithm==SBN3_MUL_PQ16 && !q.shape.centered)
             ? program_consume_inputs : 0u);
}
size_t program_pair_bytes(const sbn3_mul_plan &p) {
    const auto q=load(p);
    if(q.kind!=SBN3_PRODUCT_MUL || q.ring || q.info.algorithm!=SBN3_MUL_PQ16)return 0;
    const size_t cached=16*size_t(q.shape.nfull)+256;
    const size_t original=q.shape.centered?((8*std::max(q.an,q.bn)+127)&~size_t(127)):0;
    return cached+original+pq16::cached_scratch_bytes(q.shape,q.an,q.bn)+128;
}
void program_pair_execute(const void *ptr,Frame &f,sbn3_team_scope *scope,sbn3_const_limbs common,
                          sbn3_const_limbs x,sbn3_const_limbs y,sbn3_limbs out0,sbn3_limbs out1) {
    const auto &v=*static_cast<const Program *>(ptr);
    auto *cache=f.alloc<double>(2*size_t(v.plan.shape.nfull)+32);
    const uint64_t *original=common.data;
    if(v.plan.shape.centered){
        auto *copy=f.alloc<uint64_t>(common.count);
        memcpy(copy,common.data,common.count*8);original=copy;
    }
    pq16::forward_spectrum(cache,common.data,common.count,*v.tables,scope);
    {FrameMark mark(f);pq16::apply_spectrum(out0.data,cache,original,common.count,x.data,x.count,false,*v.tables,f,scope);}
    {FrameMark mark(f);pq16::apply_spectrum(out1.data,cache,original,common.count,y.data,y.count,false,*v.tables,f,scope);}
}
sbn3_query_result spectrum_query(const sbn3_mul_plan &p, unsigned f, uint64_t generation,
                                 sbn3_spectrum_desc &d) {
    const auto q = load(p);
    if (q.info.algorithm != SBN3_MUL_PQ16)
        return SBN3_UNSUPPORTED;
    return fft_spectrum_description(q.shape, q.an, f, generation, d);
}
void reserve_plan(const sbn3_mul_plan &p, unsigned f, uint64_t generation, sbn3_arena &a, const sbn3_lease &l,
                  sbn3_spectrum **out) {
    const auto q = load(p);
    require(q.info.algorithm == SBN3_MUL_PQ16, SBN3_FATAL_ARGUMENT, "short spectrum algorithm");
    fft_reserve_spectrum(q.shape, q.an, f, generation, a, l, out);
}
void reserve(sbn3_mul_binding *p, unsigned f, uint64_t generation, sbn3_arena &a, const sbn3_lease &l,
             sbn3_spectrum **out) {
    const auto &b = get(p);
    idle(b);
    require(b.plan.info.algorithm == SBN3_MUL_PQ16 && b.arena == &a, SBN3_FATAL_ARGUMENT,
            "short spectrum reserve");
    fft_reserve_spectrum(b.plan.shape, b.plan.an, f, generation, a, l, out);
}
void compute_spectrum(sbn3_mul_binding *p, sbn3_spectrum *s, sbn3_const_limbs input) {
    const auto &b = get(p);
    idle(b);
    require(b.plan.info.algorithm == SBN3_MUL_PQ16 && input.count == b.plan.an, SBN3_FATAL_ARGUMENT,
            "short spectrum input");
    for (const auto &l : {b.tables, b.work, b.team->storage})
        require(!overlaps(input.data, input.count * 8, l.data, l.bytes), SBN3_FATAL_ARGUMENT,
                "short spectrum resource alias");
    for (unsigned k = 1; k < b.team->width; ++k)
        require(!overlaps(input.data, input.count * 8, b.team->stacks[k].data, b.team->stacks[k].bytes),
                SBN3_FATAL_ARGUMENT, "short spectrum stack alias");
    fft_compute_spectrum(b.plan.shape, s, input, *b.team, b.plan.info.workers);
}
void prepare_spectrum(sbn3_mul_binding *p, sbn3_const_limbs input, unsigned f, uint64_t generation,
                      sbn3_arena &a, const sbn3_lease &l, sbn3_spectrum **out) {
    reserve(p, f, generation, a, l, out);
    compute_spectrum(p, *out, input);
}
} // namespace
const Backend &short_backend() noexcept {
    static const Backend b{
        .id = id,
        .name = "native scalar/u52/pq16",
        .query = query,
        .bind = bind,
        .execute = execute,
        .execute_scope = execute_scope,
        .metrics = metrics,
        .unbind = unbind,
        .product_query = query_product,
        .product_bind = bind_product,
        .product_execute = product_execute,
        .product_metrics = product_metrics,
        .prepare = prepare_spectrum,
        .describe = nullptr,
        .can_apply = nullptr,
        .spectrum_retain = nullptr,
        .spectrum_release = nullptr,
        .spectrum_query = spectrum_query,
        .spectrum_reserve = reserve,
        .spectrum_compute = compute_spectrum,
        .spectrum_reserve_plan = reserve_plan,
        .program_bytes = program_bytes,
        .program_prepare = program_prepare,
        .program_execute = program_execute,
        .program_contract = program_contract,
        .program_pair_bytes = program_pair_bytes,
        .program_pair_execute = program_pair_execute,
        .program_tables = program_tables,
        .program_local_bytes = program_local_bytes,
        .program_tables_prepare = program_tables_prepare,
        .program_prepare_shared = program_prepare_shared,
    };
    return b;
}
} // namespace sbn::v3
