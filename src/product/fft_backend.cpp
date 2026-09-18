#include "product/spectrum_contract.hpp"
#include "common/identity.hpp"
#include "product/fft_backend.hpp"
#include "runtime/team.hpp"
#include <new>
#include <initializer_list>
#include <algorithm>
#include <string.h>
namespace sbn::v3 {
namespace {
constexpr uint64_t id = 101, magic = 0x53424e3346465431ULL, spectrum_magic = 0x53424e3353504543ULL,
                   binding_magic = 0x53424e3342494e44ULL;
size_t up(size_t n) {
    size_t r = 0;
    require(align_size(n, 128, r), SBN3_FATAL_SIZE, "FFT cache layout");
    return r;
}
uint64_t hash(uint64_t h, uint64_t v) {
    return identity::word(h, v);
}
unsigned codec(pq16::Shape s) {
    return unsigned(s.recipe) | (s.balanced ? spectrum_contract::fft_codec::balanced : 0u) |
           (s.centered ? spectrum_contract::fft_codec::centered : 0u);
}
bool equal(pq16::Shape a, pq16::Shape b) {
    return a.nfull == b.nfull && a.branch == b.branch && a.radix == b.radix && a.bits == b.bits &&
           codec(a) == codec(b);
}
uint64_t basis(pq16::Shape s) {
    uint64_t h = 1469598103934665603ULL;
    for (uint64_t v :
         {id, uint64_t(s.nfull), uint64_t(s.branch), uint64_t(s.radix), uint64_t(s.bits), uint64_t(codec(s))})
        h = hash(h, v);
    return h;
}
uint64_t descriptor_seal(const sbn3_spectrum_desc &d) {
    uint64_t h = 1469598103934665603ULL;
    for (uint64_t v : {d.basis_id,
                       d.instance_id,
                       d.generation,
                       uint64_t(d.np),
                       uint64_t(d.trunk_bits),
                       uint64_t(d.frontier),
                       uint64_t(d.format_version),
                       d.C,
                       d.M2,
                       d.transform_trunks,
                       d.live_slots,
                       d.written_slots,
                       d.source_limbs,
                       d.source_trunks,
                       d.block_stride,
                       d.storage_bytes,
                       d.table_bytes,
                       d.plane_bytes,
                       d.backend_id,
                       uint64_t(d.codec_mode)})
        h = hash(h, v);
    for (auto v : d.scale)
        h = hash(h, v);
    return h;
}
struct Spectrum {
    sbn3_spectrum header{&fft_backend(), spectrum_magic};
    Arena *arena = nullptr;
    sbn3_lease storage{};
    uint64_t refs = 1;
    uint32_t state = spectrum_contract::reserved;
    pq16::Shape shape{};
    sbn3_spectrum_desc desc{};
    const pq16::Tables *tables = nullptr;
    double *data = nullptr;
    uint64_t *original = nullptr;
};
size_t storage_bytes(pq16::Shape s, size_t an) {
    return up(sizeof(Spectrum)) + up(pq16::table_bytes(s)) + up(16 * size_t(s.nfull)) +
           (s.centered ? up(8 * an) : 0) + 256;
}
pq16::Shape shape_of(const sbn3_spectrum_desc &d) {
    return {uint32_t(d.transform_trunks / 2),
            uint32_t(d.M2),
            uint32_t(d.C),
            bool(d.codec_mode & spectrum_contract::fft_codec::centered),
            pq16::Recipe(d.codec_mode & spectrum_contract::fft_codec::recipe_mask),
            d.trunk_bits,
            bool(d.codec_mode & spectrum_contract::fft_codec::balanced)};
}
bool valid(const sbn3_spectrum_desc &d) {
    if (d.backend_id != id || d.np || d.format_version != spectrum_contract::fft_aosoa_format ||
        d.frontier != 1 || d.codec_mode & ~spectrum_contract::fft_codec::allowed ||
        d.seal != descriptor_seal(d) || !d.C || d.C > 7 || d.M2 > (1u << 19) || !d.M2 ||
        (d.M2 & (d.M2 - 1)) || d.transform_trunks != 2 * d.C * d.M2 || !d.source_limbs ||
        d.source_limbs > (1u << 20))
        return false;
    const auto s = shape_of(d);
    if (s.bits < 16 || s.bits > 20 || (s.radix != 1 && s.radix != 3 && s.radix != 5 && s.radix != 7) ||
        unsigned(s.recipe) > 2 || s.nfull > (1u << 19) || d.basis_id != basis(s) || d.live_slots != s.nfull ||
        d.written_slots != s.nfull || d.block_stride != 128 ||
        d.source_trunks != (64 * d.source_limbs + s.bits - 1) / s.bits ||
        d.plane_bytes != 16 * size_t(s.nfull) || d.table_bytes != pq16::table_bytes(s) ||
        d.storage_bytes < storage_bytes(s, d.source_limbs) || d.scale[0] != 1)
        return false;
    for (unsigned k = 1; k < 10; ++k)
        if (d.scale[k])
            return false;
    return true;
}
Spectrum &spectrum(sbn3_spectrum *p) {
    auto *s = reinterpret_cast<Spectrum *>(p);
    require(s && s->header.backend == &fft_backend() && s->header.marker == spectrum_magic,
            SBN3_FATAL_LIFETIME, "FFT spectrum");
    return *s;
}
const Spectrum &spectrum(const sbn3_spectrum *p) {
    return spectrum(const_cast<sbn3_spectrum *>(p));
}
bool matches(const Spectrum &s, const sbn3_spectrum_desc &d) {
    if (!valid(s.desc) || !valid(d) || !equal(s.shape, shape_of(d)) ||
        s.desc.source_limbs != d.source_limbs || s.desc.generation != d.generation)
        return false;
    return d.instance_id ? s.desc.seal == d.seal : s.desc.storage_bytes >= d.storage_bytes;
}
void retain(const sbn3_spectrum *p) {
    auto &s = const_cast<Spectrum &>(spectrum(p));
    const auto n = __atomic_fetch_add(&s.refs, 1, __ATOMIC_RELAXED);
    require(n && n != UINT64_MAX, SBN3_FATAL_LIFETIME, "FFT spectrum retain");
}
void release(const sbn3_spectrum *p) {
    auto &s = const_cast<Spectrum &>(spectrum(p));
    const auto n = __atomic_fetch_sub(&s.refs, 1, __ATOMIC_ACQ_REL);
    require(n, SBN3_FATAL_LIFETIME, "FFT spectrum release");
    if (n == 1) {
        auto *a = s.arena;
        const auto l = s.storage;
        s.header.marker = 0;
        s.~Spectrum();
        a->release(l);
    }
}
struct Plan {
    uint64_t marker = magic, backend = id, seal = 0;
    size_t an = 0, bn = 0, ring = 0, prefix = 0;
    unsigned kind = 0;
    bool has_cache = false, plus = false;
    pq16::Shape shape{};
    sbn3_spectrum_desc cached{};
    sbn3_product_info info{};
};
static_assert(sizeof(Plan) <= sizeof(sbn3_mul_plan));
uint64_t seal(const Plan &p) {
    uint64_t h = hash(hash(magic, p.an), p.bn);
    for (uint64_t v :
         {uint64_t(p.kind), uint64_t(p.has_cache), uint64_t(p.plus), p.ring, p.prefix, basis(p.shape),
          p.cached.seal, p.info.mul.workspace_bytes, p.info.mul.table_bytes, p.info.mul.per_worker_bytes,
          uint64_t(p.info.mul.workers), p.info.mul.arithmetic_id, p.info.mul.execution_id}) {
        h = hash(h, v);
    }
    return h;
}
Plan load(const sbn3_mul_plan &p) {
    Plan q{};
    memcpy(&q, p.opaque, sizeof q);
    require(q.marker == magic && q.backend == id && q.seal == seal(q), SBN3_FATAL_ARGUMENT,
            "FFT cached plan");
    return q;
}
struct Binding {
    sbn3_mul_binding header{&fft_backend(), binding_magic};
    Plan plan{};
    Arena *arena;
    sbn3_team *team;
    sbn3_lease tables, work;
    Frame frame;
    const Spectrum *cache = nullptr;
    const pq16::Tables *math_tables = nullptr;
    bool last_build = false;
    uint32_t active = 0;
    uint64_t executions = 0;
    size_t base_bytes = 0;
    Binding(const Plan &p, Arena &a, sbn3_team &t, sbn3_lease tl, sbn3_lease w, const Spectrum *s)
        : plan(p), arena(&a), team(&t), tables(tl), work(w), frame(a, w), cache(s) {}
};
Binding &binding(sbn3_mul_binding *b) {
    return *reinterpret_cast<Binding *>(b);
}
const Binding &binding(const sbn3_mul_binding *b) {
    return *reinterpret_cast<const Binding *>(b);
}
void idle(const Binding &b) {
    require(pthread_equal(b.team->creator, pthread_self()) && !b.team->busy && !b.active, SBN3_FATAL_TEAM,
            "FFT binding owner");
}
sbn3_query_result query_product(const sbn3_product_request &r, const sbn3_mul_options &o, sbn3_mul_plan &out,
                                sbn3_product_info &info) {
    if (r.negacyclic > 1 || (r.negacyclic && (!r.cyclic_limbs || o.workers != 1)))
        return SBN3_UNSUPPORTED;
    info = {};
    const bool mid = r.kind == SBN3_PRODUCT_TMP;
    if ((r.kind != SBN3_PRODUCT_MUL && r.kind != SBN3_PRODUCT_SQR && !mid) || r.a1_limbs || r.b1_limbs ||
        r.cached_a[1] || (r.kind == SBN3_PRODUCT_SQR && r.b_limbs) || (mid && r.cyclic_limbs) ||
        (!r.cached_a[0] && !r.cyclic_limbs && !mid))
        return SBN3_UNSUPPORTED;
    if (!o.workers || o.workers > 32 || o.prime_count || o.prime_batch || o.column_log2 || o.row_log2 ||
        o.crt_mode || o.codec_mode || o.fused_start_skew_us || o.borrow_output > 2 ||
        (o.algorithm != SBN3_MUL_AUTO && o.algorithm != SBN3_MUL_PQ16))
        return SBN3_UNSUPPORTED;
    Plan p{};
    p.an = r.a_limbs;
    p.bn = r.kind == SBN3_PRODUCT_SQR ? r.a_limbs : r.b_limbs;
    p.kind = r.kind;
    p.ring = r.cyclic_limbs;
    p.plus = r.negacyclic;
    p.prefix = r.window_limbs;
    p.has_cache = r.cached_a[0] != nullptr;
    if (p.prefix && (!p.ring || p.plus || mid || p.prefix > p.ring || p.an + p.bn >= p.ring))
        return SBN3_UNSUPPORTED;
    if (!p.an || !p.bn || p.an > (1u << 20) || p.bn > (1u << 20) || (mid && p.bn < p.an))
        return SBN3_UNSUPPORTED;
    if (p.has_cache) {
        const auto &d = *r.cached_a[0];
        if (!valid(d) || d.source_limbs != p.an || (o.trunk_bits && unsigned(o.trunk_bits) != d.trunk_bits))
            return SBN3_UNSUPPORTED;
        p.shape = shape_of(d);
        p.cached = d;
        if (mid)
            p.ring = pq16::cyclic_period(p.shape);
    } else {
        if (o.trunk_bits && (o.trunk_bits < 16 || o.trunk_bits > 20))
            return SBN3_UNSUPPORTED;
        p.shape = p.plus ? pq16::plus_shape(p.ring)
                         : pq16::cyclic_shape(mid ? std::max(p.bn + 1, 2 * p.an) : p.ring,
                                              o.trunk_bits ? unsigned(o.trunk_bits) : 16);
        if (mid)
            p.ring = pq16::cyclic_period(p.shape);
    }
    if (p.ring) {
        if (p.ring != pq16::cyclic_period(p.shape) ||
            !(p.plus ? pq16::plus_supported(p.shape, p.an, p.bn)
                     : pq16::cyclic_supported(p.shape, p.an, p.bn)) ||
            (mid && p.ring < p.bn + 1))
            return SBN3_UNSUPPORTED;
    } else if (!pq16::supported(p.shape, p.an, p.bn, o.workers))
        return SBN3_UNSUPPORTED;
    if (p.shape.bits > 16 && o.workers != 1)
        return SBN3_UNSUPPORTED;
    auto &i = p.info.mul;
    i.algorithm = SBN3_MUL_PQ16;
    i.workers = o.workers;
    i.trunk_bits = p.shape.bits;
    i.digit_words = 1;
    i.C = p.shape.radix;
    i.M2 = p.shape.branch;
    i.transform_trunks = 2 * p.shape.nfull;
    i.nat = (64 * p.an + i.trunk_bits - 1) / i.trunk_bits;
    i.nyt = (64 * p.bn + i.trunk_bits - 1) / i.trunk_bits;
    i.output_limbs = p.prefix ? p.prefix
                     : mid    ? p.bn - p.an + 1
                     : p.ring ? p.ring + unsigned(p.plus)
                              : p.an + p.bn;
    i.output_alignment = 8;
    i.workspace_alignment = 128;
    i.table_bytes = p.has_cache ? 128 : pq16::table_bytes(p.shape);
    i.per_worker_bytes =
        p.ring ? (p.has_cache || p.kind == SBN3_PRODUCT_SQR ? 16 : 32) * size_t(p.shape.nfull) + 1024 +
                     (p.plus ? 16 * size_t(p.shape.nfull) : 0) + (mid ? p.ring * 8 + 128 : 0)
               : pq16::cached_scratch_bytes(p.shape, p.an, p.bn);
    i.workspace_bytes = up(sizeof(Binding)) + i.per_worker_bytes;
    i.table_entries = p.shape.branch;
    i.full = p.ring != 0;
    i.codec_mode = unsigned(p.shape.recipe) | (p.shape.balanced ? 8u : 0u);
    i.basis_id = basis(p.shape);
    i.arithmetic_id = hash(hash(hash(hash(i.basis_id, p.an), p.bn), p.kind), p.ring);
    if (p.prefix)
        i.arithmetic_id = hash(i.arithmetic_id, p.prefix);
    if (p.plus)
        i.arithmetic_id = hash(i.arithmetic_id, 1);
    i.execution_id = hash(hash(hash(i.arithmetic_id, o.workers), i.workspace_bytes), p.has_cache);
    p.info.kind = r.kind;
    p.info.cached_mask = p.has_cache;
    p.info.cyclic_limbs = r.cyclic_limbs;
    p.info.negacyclic = p.plus;
    p.info.spectrum_bytes = storage_bytes(p.shape, p.an);
    p.info.spectrum_alignment = 128;
    if (p.prefix)
        p.info.window = {0, 64 * p.prefix, 0};
    if (mid)
        p.info.window = {64 * (p.an - 1), 64 * (p.bn - p.an + 1), 1};
    info = p.info;
    if (o.workspace_budget && i.workspace_bytes > o.workspace_budget)
        return SBN3_QUERY_CAPACITY;
    p.seal = seal(p);
    memset(&out, 0, sizeof out);
    memcpy(out.opaque, &p, sizeof p);
    return SBN3_SUPPORTED;
}
void metrics(const sbn3_mul_binding *p, sbn3_mul_metrics &out) {
    const auto &b = binding(p);
    idle(b);
    out = {b.frame.peak() - b.base_bytes,
           b.plan.has_cache ? 0 : b.plan.info.mul.table_bytes,
           b.frame.peak(),
           b.executions,
           {}};
}
void product_metrics(const sbn3_mul_binding *p, sbn3_product_metrics &out) {
    const auto &b = binding(p);
    out = {};
    metrics(p, out.mul);
    if (b.executions) {
        out.row_forward = b.plan.has_cache ? (b.plan.kind == SBN3_PRODUCT_SQR ? 0 : 1)
                                           : (b.plan.kind == SBN3_PRODUCT_SQR ? 1 : 2);
        out.row_forward += b.last_build;
        out.row_inverse = 1;
    }
}
void forbidden_fast(sbn3_mul_binding *, const uint64_t *, const uint64_t *, uint64_t *) {
    fatal(SBN3_FATAL_ARGUMENT, "FFT cache requires product_execute");
}
void product_bind(const sbn3_mul_plan &p, sbn3_arena &a, const sbn3_lease &t, const sbn3_lease &w,
                  sbn3_team &team, const sbn3_spectrum *s0, const sbn3_spectrum *s1, sbn3_mul_binding **out) {
    const auto q = load(p);
    require(bool(s0) == q.has_cache && !s1, SBN3_FATAL_ARGUMENT, "FFT cached operand");
    const auto *s = s0 ? &spectrum(s0) : nullptr;
    if (s)
        require(matches(*s, q.cached), SBN3_FATAL_ARGUMENT, "FFT incompatible spectrum");
    require(team.arena == &a && team.width >= q.info.mul.workers &&
                pthread_equal(team.creator, pthread_self()) && !team.busy,
            SBN3_FATAL_TEAM, "FFT bind team");
    require(t.bytes >= q.info.mul.table_bytes && w.bytes >= q.info.mul.workspace_bytes &&
                !(uintptr_t(t.data) & 127) && !(uintptr_t(w.data) & 127) &&
                !overlaps(t.data, t.bytes, w.data, w.bytes) &&
                (!s || (!overlaps(s->storage.data, s->storage.bytes, t.data, t.bytes) &&
                        !overlaps(s->storage.data, s->storage.bytes, w.data, w.bytes))),
            SBN3_FATAL_WORKSPACE, "FFT binding storage");
    a.claim_unshared(t);
    a.claim_unshared(w);
    auto *b = new (w.data) Binding(q, a, team, t, w, s);
    b->header.execute_fast = forbidden_fast;
    b->base_bytes = up(sizeof(Binding));
    b->frame.allocate(b->base_bytes, 128);
    if (s) {
        retain(s0);
        b->math_tables = s->tables;
    } else {
        ComputeLease setup(a);
        Frame tf(a, t);
        b->math_tables = pq16::prepare(tf, q.shape);
    }
    a.release(w);
    *out = &b->header;
}
struct Call {
    Binding *b;
    sbn3_const_limbs a, fresh;
    sbn3_limbs out;
    Spectrum *build = nullptr;
};
void action(void *arg, sbn3_team_scope *scope) {
    auto &c = *static_cast<Call *>(arg);
    auto &b = *c.b;
    require_scope_leader(scope);
    require(scope->team == b.team && scope->width >= b.plan.info.mul.workers, SBN3_FATAL_TEAM,
            "FFT product scope");
    sbn3_team_scope sub{scope->team, scope->first, b.plan.info.mul.workers, false, scope->epoch};
    uint32_t expected = spectrum_contract::reserved;
    require(__atomic_compare_exchange_n(&b.active, &expected, 1, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED),
            SBN3_FATAL_LIFETIME, "concurrent FFT binding");
    b.last_build = c.build != nullptr;
    if (c.build)
        pq16::forward_spectrum(c.build->data, c.a.data, c.a.count, *b.math_tables, &sub);
    if (b.plan.ring) {
        FrameMark mark(b.frame);
        const bool mid = b.plan.kind == SBN3_PRODUCT_TMP;
        auto *sink = mid ? b.frame.alloc<uint64_t>(b.plan.ring) : c.out.data;
        if (b.plan.plus)
            pq16::plus_multiply(sink, c.a.data, b.plan.an, c.fresh.data, b.plan.bn,
                                b.plan.kind == SBN3_PRODUCT_SQR, b.cache ? b.cache->data : nullptr,
                                *b.math_tables, b.frame);
        else
            pq16::cyclic_multiply(sink, c.a.data, b.plan.an, c.fresh.data, b.plan.bn,
                                  b.plan.kind == SBN3_PRODUCT_SQR, b.cache ? b.cache->data : nullptr,
                                  *b.math_tables, b.frame, &sub, b.plan.prefix);
        if (mid)
            memcpy(c.out.data, sink + b.plan.an - 1, b.plan.info.mul.output_limbs * 8);
    } else
        pq16::apply_spectrum(c.out.data, b.cache->data, b.cache->original, b.plan.an, c.fresh.data, b.plan.bn,
                             b.plan.kind == SBN3_PRODUCT_SQR, *b.math_tables, b.frame, &sub);
    require(b.frame.used() == b.base_bytes, SBN3_FATAL_WORKSPACE, "FFT frame rewind");
    ++b.executions;
    __atomic_store_n(&b.active, 0, __ATOMIC_RELEASE);
}
void product_values(Binding &b, const sbn3_product_inputs &in, sbn3_limbs out, bool building = false) {
    const auto &q = b.plan;
    const size_t rb = bytes_for(q.info.mul.output_limbs, 8), bb = bytes_for(in.b.count, 8),
                 ab = bytes_for(in.a.count, 8);
    require((q.has_cache && !building ? (!in.a.data && !in.a.count) : in.a.count == q.an) && !in.a1.data &&
                !in.a1.count && !in.b1.data && !in.b1.count &&
                (q.kind == SBN3_PRODUCT_SQR ? !in.b.data && !in.b.count : in.b.count == q.bn),
            SBN3_FATAL_ARGUMENT, "FFT cached inputs");
    if (b.cache && !building)
        require(__atomic_load_n(&b.cache->state, __ATOMIC_ACQUIRE) == spectrum_contract::ready,
                SBN3_FATAL_LIFETIME, "FFT spectrum not completed");
    require(out.capacity >= q.info.mul.output_limbs && !(uintptr_t(out.data) & 7) &&
                !(uintptr_t(in.b.data) & 7),
            SBN3_FATAL_ARGUMENT, "FFT output shape");
    valid_span(out.data, rb, "FFT output");
    valid_span(in.b.data, bb, "FFT input");
    valid_span(in.a.data, ab, "FFT input A");
    require(!overlaps(out.data, rb, in.b.data, bb) && !overlaps(out.data, rb, in.a.data, ab) &&
                !(uintptr_t(in.a.data) & 7),
            SBN3_FATAL_ARGUMENT, "FFT value alias");
    auto disjoint = [&](const sbn3_lease &l) {
        require(!overlaps(l.data, l.bytes, out.data, rb) && !overlaps(l.data, l.bytes, in.b.data, bb) &&
                    !overlaps(l.data, l.bytes, in.a.data, ab),
                SBN3_FATAL_ARGUMENT, "FFT resource alias");
    };
    disjoint(b.tables);
    disjoint(b.work);
    if (b.cache)
        disjoint(b.cache->storage);
    disjoint(b.team->storage);
    for (unsigned k = 1; k < b.team->width; ++k)
        disjoint(b.team->stacks[k]);
}
void product_execute(sbn3_mul_binding *p, sbn3_team_scope *scope, const sbn3_product_inputs &in,
                     sbn3_limbs out) {
    auto &b = binding(p);
    if (!scope)
        idle(b);
    product_values(b, in, out);
    Call c{&b, in.a, in.b, out};
    if (scope)
        action(&c, scope);
    else
        sbn3_team_run(b.team, action, &c);
}

void unbind(sbn3_mul_binding *p) {
    auto &b = binding(p);
    idle(b);
    auto *a = b.arena;
    const auto t = b.tables;
    const auto *cache = b.cache ? &b.cache->header : nullptr;
    if (cache)
        release(cache);
    b.header.marker = 0;
    b.~Binding();
    a->release(t);
}
void describe(const sbn3_spectrum *p, sbn3_spectrum_desc &out) {
    out = spectrum(p).desc;
}
bool can_apply(const sbn3_mul_plan &p, const sbn3_spectrum *s, unsigned term) {
    if (term || !s)
        return false;
    const auto q = load(p);
    if (!q.has_cache)
        return false;
    const auto &sp = spectrum(s);
    return __atomic_load_n(&sp.state, __ATOMIC_ACQUIRE) == spectrum_contract::ready && matches(sp, q.cached);
}
sbn3_query_result query_spectrum(const sbn3_mul_plan &p, unsigned f, uint64_t g, sbn3_spectrum_desc &d) {
    const auto q = load(p);
    return fft_spectrum_description(q.shape, q.an, f, g, d);
}
void reserve_plan(const sbn3_mul_plan &p, unsigned f, uint64_t g, sbn3_arena &a, const sbn3_lease &l,
                  sbn3_spectrum **out) {
    const auto q = load(p);
    fft_reserve_spectrum(q.shape, q.an, f, g, a, l, out);
}
void reserve(sbn3_mul_binding *p, unsigned f, uint64_t g, sbn3_arena &a, const sbn3_lease &l,
             sbn3_spectrum **out) {
    const auto &b = binding(p);
    idle(b);
    fft_reserve_spectrum(b.plan.shape, b.plan.an, f, g, a, l, out);
}
void compute_spectrum(sbn3_mul_binding *p, sbn3_spectrum *s, sbn3_const_limbs a) {
    const auto &b = binding(p);
    idle(b);
    for (const auto &l : {b.tables, b.work})
        require(!overlaps(a.data, a.count * 8, l.data, l.bytes), SBN3_FATAL_ARGUMENT,
                "FFT producer resource alias");
    fft_compute_spectrum(b.plan.shape, s, a, *b.team, b.plan.info.mul.workers);
}
void prepare(sbn3_mul_binding *p, sbn3_const_limbs a, unsigned f, uint64_t g, sbn3_arena &arena,
             const sbn3_lease &l, sbn3_spectrum **out) {
    reserve(p, f, g, arena, l, out);
    compute_spectrum(p, *out, a);
}
sbn3_query_result query(const sbn3_product_spec &, const sbn3_mul_options &, sbn3_mul_plan &,
                        sbn3_mul_info &) {
    return SBN3_UNSUPPORTED;
}
void plain_bind(const sbn3_mul_plan &, sbn3_arena &, const sbn3_lease &, const sbn3_lease &, sbn3_team &,
                sbn3_mul_binding **) {
    fatal(SBN3_FATAL_ARGUMENT, "FFT cache requires product_bind");
}
void plain_execute(sbn3_mul_binding *, sbn3_const_limbs, sbn3_const_limbs, sbn3_limbs) {
    fatal(SBN3_FATAL_ARGUMENT, "FFT cache requires product_execute");
}
void plain_scope(sbn3_mul_binding *, sbn3_team_scope *, sbn3_const_limbs, sbn3_const_limbs, sbn3_limbs) {
    fatal(SBN3_FATAL_ARGUMENT, "FFT cache requires product_execute");
}
} // namespace
sbn3_query_result fft_spectrum_description(pq16::Shape s, size_t an, unsigned frontier, uint64_t generation,
                                           sbn3_spectrum_desc &out) {
    if (frontier != SBN3_SPECTRUM_COLUMNS || !an || an > (1u << 20) || !s.nfull)
        return SBN3_UNSUPPORTED;
    out = {};
    out.basis_id = basis(s);
    out.generation = generation;
    out.trunk_bits = s.bits;
    out.frontier = 1;
    out.format_version = spectrum_contract::fft_aosoa_format;
    out.C = s.radix;
    out.M2 = s.branch;
    out.transform_trunks = 2 * s.nfull;
    out.live_slots = out.written_slots = s.nfull;
    out.source_limbs = an;
    out.source_trunks = (64 * an + s.bits - 1) / s.bits;
    out.block_stride = 128;
    out.scale[0] = 1;
    out.storage_bytes = storage_bytes(s, an);
    out.table_bytes = pq16::table_bytes(s);
    out.plane_bytes = 16 * size_t(s.nfull);
    out.backend_id = id;
    out.codec_mode = codec(s);
    out.seal = descriptor_seal(out);
    return SBN3_SUPPORTED;
}
void fft_reserve_spectrum(pq16::Shape shape, size_t an, unsigned frontier, uint64_t generation, sbn3_arena &a,
                          const sbn3_lease &l, sbn3_spectrum **out) {
    sbn3_spectrum_desc d{};
    require(fft_spectrum_description(shape, an, frontier, generation, d) == SBN3_SUPPORTED &&
                l.bytes >= d.storage_bytes && !(uintptr_t(l.data) & 127),
            SBN3_FATAL_ARGUMENT, "FFT spectrum storage");
    a.claim_unshared(l);
    ComputeLease setup(a);
    Frame f(a, l);
    auto *s = new (f.allocate(sizeof(Spectrum), 128)) Spectrum{};
    s->arena = &a;
    s->storage = l;
    s->shape = shape;
    s->tables = pq16::prepare(f, shape);
    s->data = static_cast<double *>(f.allocate(d.plane_bytes, 128));
    if (shape.centered)
        s->original = static_cast<uint64_t *>(f.allocate(an * 8, 128));
    static uint64_t sequence = 0;
    d.instance_id = __atomic_add_fetch(&sequence, 1, __ATOMIC_RELAXED);
    require(d.instance_id, SBN3_FATAL_LIFETIME, "FFT spectrum generation");
    d.storage_bytes = l.bytes;
    d.seal = descriptor_seal(d);
    s->desc = d;
    *out = &s->header;
}
namespace {
void begin_fft_spectrum(pq16::Shape shape, Spectrum &s, sbn3_const_limbs a, sbn3_team &team,
                        unsigned workers) {
    require(equal(shape, s.shape) && a.count == s.desc.source_limbs && s.arena == team.arena &&
                team.width >= workers && pthread_equal(team.creator, pthread_self()) && !team.busy,
            SBN3_FATAL_ARGUMENT, "FFT spectrum producer");
    valid_span(a.data, bytes_for(a.count, 8), "FFT spectrum source");
    require(!(uintptr_t(a.data) & 7) && !overlaps(a.data, a.count * 8, s.storage.data, s.storage.bytes),
            SBN3_FATAL_ARGUMENT, "FFT spectrum input alias");
    require(!overlaps(a.data, a.count * 8, team.storage.data, team.storage.bytes), SBN3_FATAL_ARGUMENT,
            "FFT spectrum/team alias");
    for (unsigned k = 1; k < team.width; ++k)
        require(!overlaps(a.data, a.count * 8, team.stacks[k].data, team.stacks[k].bytes),
                SBN3_FATAL_ARGUMENT, "FFT spectrum/stack alias");
    uint32_t expected = spectrum_contract::reserved;
    require(__atomic_compare_exchange_n(&s.state, &expected, spectrum_contract::computing, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE),
            SBN3_FATAL_LIFETIME, "FFT spectrum already filled");
    if (s.original)
        memcpy(s.original, a.data, a.count * 8);
}
void compute_square(sbn3_mul_binding *p, sbn3_spectrum *handle, sbn3_const_limbs a, sbn3_limbs out) {
    auto &b = binding(p);
    idle(b);
    auto &s = spectrum(handle);
    require(b.plan.kind == SBN3_PRODUCT_SQR && b.cache == &s, SBN3_FATAL_ARGUMENT, "FFT fused square cache");
    product_values(b, {a, {}, {}, {}}, out, true);
    begin_fft_spectrum(b.plan.shape, s, a, *b.team, b.plan.info.mul.workers);
    Call c{&b, {s.original ? s.original : a.data, a.count}, {}, out, &s};
    sbn3_team_run(b.team, action, &c);
    __atomic_store_n(&s.state, spectrum_contract::ready, __ATOMIC_RELEASE);
}
} // namespace
void fft_compute_spectrum(pq16::Shape shape, sbn3_spectrum *p, sbn3_const_limbs a, sbn3_team &team,
                          unsigned workers) {
    auto &s = spectrum(p);
    begin_fft_spectrum(shape, s, a, team, workers);
    struct Work {
        Spectrum *s;
        const uint64_t *a;
        size_t n;
        unsigned workers;
    } work{&s, s.original ? s.original : a.data, a.count, workers};
    auto run = [](void *arg, sbn3_team_scope *scope) {
        const auto &w = *static_cast<Work *>(arg);
        sbn3_team_scope sub{scope->team, scope->first, w.workers, false, scope->epoch};
        pq16::forward_spectrum(w.s->data, w.a, w.n, *w.s->tables, &sub);
    };
    sbn3_team_run(&team, run, &work);
    __atomic_store_n(&s.state, spectrum_contract::ready, __ATOMIC_RELEASE);
}
const Backend &fft_backend() noexcept {
    static const Backend b{
        .id = id,
        .name = "native FFT spectrum",
        .query = query,
        .bind = plain_bind,
        .execute = plain_execute,
        .execute_scope = plain_scope,
        .metrics = metrics,
        .unbind = unbind,
        .product_query = query_product,
        .product_bind = product_bind,
        .product_execute = product_execute,
        .product_metrics = product_metrics,
        .prepare = prepare,
        .describe = describe,
        .can_apply = can_apply,
        .spectrum_retain = retain,
        .spectrum_release = release,
        .spectrum_query = query_spectrum,
        .spectrum_reserve = reserve,
        .spectrum_compute = compute_spectrum,
        .spectrum_reserve_plan = reserve_plan,
        .spectrum_square = compute_square,
    };
    return b;
}
} // namespace sbn::v3
