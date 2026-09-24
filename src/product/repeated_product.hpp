#pragma once
#include "product/backend.hpp"
#include "product/fixed_geometry.hpp"
#include "product/fft_choice.hpp"
#include "product/window_policy.hpp"
#include "common/identity.hpp"
namespace sbn::v3::product {
struct RepeatedProductChoice {
    unsigned np = 0, algorithm = 0, workers = 1;
    int T = 0;
    size_t ring = 0;
    unsigned cached = 0, pfa = 0; // explicit classic FFT recipe; no mixed-shape search on replay
};
struct RepeatedProductPlans {
    sbn3_mul_plan producer{}, consumer{};
    sbn3_product_info producer_info{}, consumer_info{};
    sbn3_spectrum_desc future{};
    bool cached = false;
};

struct RepeatedProductOptions {
    unsigned workers=1,prime_count=0,reuse_hint=0,residual=0;
    size_t uses=1;
    bool cancellation_bound=false;
};
inline bool repeated_product_available() noexcept {return native_available();}
inline uint64_t repeated_choice_identity(const RepeatedProductChoice &c) noexcept {
    uint64_t h=1469598103934665603ULL;
    for(uint64_t x:{uint64_t(c.np),uint64_t(c.algorithm),uint64_t(c.workers),uint64_t(c.T),uint64_t(c.ring),uint64_t(c.cached),uint64_t(c.pfa)})h=identity::word(h,x);
    return h;
}
inline sbn3_mul_options product_options(const RepeatedProductChoice &c) {
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
inline bool consumer_builds(const RepeatedProductPlans &q, uint64_t generation) {
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
inline bool query_consumer(sbn3_product_request r, const sbn3_mul_options &o, uint64_t generation, RepeatedProductPlans &q) {
    if (sbn3_spectrum_query(&q.producer, SBN3_SPECTRUM_COLUMNS, generation, &q.future) != SBN3_SUPPORTED)
        return false;
    r.cached_a[0] = &q.future;
    return sbn3_product_query(&r, &o, &q.consumer, &q.consumer_info) == SBN3_SUPPORTED;
}
inline bool query_cached(sbn3_product_request r, const sbn3_mul_options &o, uint64_t generation, RepeatedProductPlans &q) {
    const auto *backend=o.prime_count?backend_lookup(o.prime_count):nullptr;
    const auto status=backend?backend->product_query(r,o,q.producer,q.producer_info):
                              sbn3_product_query(&r,&o,&q.producer,&q.producer_info);
    return status == SBN3_SUPPORTED &&
           query_consumer(r, o, generation, q);
}
// Term 0 (a) is the divisor-side operand that may be cached; term 1 (b) is fresh.
inline bool query_product(size_t a, size_t b, const RepeatedProductChoice &c, uint64_t generation, RepeatedProductPlans &q) {
    q = {};
    auto o = product_options(c);
    sbn3_product_request r{};
    r.kind = SBN3_PRODUCT_MUL;
    r.a_limbs = a;
    r.b_limbs = b;
    r.cyclic_limbs = c.ring;
    bool supported = false;
    if(c.pfa){
        if(c.pfa!=1||c.np||c.ring)return false;
        const auto shape=pq16::execution_shape(pq16::query(a,b),c.workers);
        supported=shape.nfull&&short_fft_query(r,o,shape,q.producer,q.producer_info)==SBN3_SUPPORTED;
    }else if(c.np){
        const auto *backend=backend_lookup(c.np);
        supported=backend&&backend->product_query(r,o,q.producer,q.producer_info)==SBN3_SUPPORTED;
    }else supported=sbn3_product_query(&r,&o,&q.producer,&q.producer_info)==SBN3_SUPPORTED;
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
inline bool native_product(const RepeatedProductOptions &o,size_t a,size_t b,bool cyclic,RepeatedProductChoice &out) {
    FixedNttGeometry g;
    if(!fixed_ntt_geometry(a,b,cyclic?a:0,native_window_workers(std::max(a,b),o.workers),o.prime_count,g,size_t(1)<<19,0,o.uses>1))return false;
    out={g.np,g.algorithm,g.workers,g.T,g.ring,1,0};return true;
}
inline bool repeated_product_query(const RepeatedProductOptions &o,size_t a,size_t b,bool residual,RepeatedProductChoice &c,RepeatedProductPlans &q) {
    c={};const bool forced_ring=residual&&o.residual==2;
    const bool keep=o.workers>1||o.reuse_hint>1||o.uses>1;
    const auto materialize=[&]{if(c.algorithm!=SBN3_MUL_U52)c.cached=keep;return query_product(a,b,c,residual?2:1,q);};
    const unsigned width=native_window_workers(std::max(a,b),o.workers);
    // Increasing the rung team must not disable a ring already admitted
    // at eight workers. Beyond that point the work is bandwidth dominated.
    const size_t ring_threshold=32768*size_t(std::min(width,8u));
    const bool allow_ring=residual&&o.residual!=1&&
        (forced_ring||o.cancellation_bound||a>=ring_threshold||o.reuse_hint>=2);
    if(!o.prime_count&&!forced_ring&&
       (std::min(a,b)<128||(std::min(a,b)<=1024&&std::max(a,b)>=4*std::min(a,b)))){
        c.algorithm=SBN3_MUL_U52;
        return materialize();
    }
    if(!o.prime_count&&width==1){
        auto linear=pq16::query(a,b);linear.recipe=pq16::Recipe::PfaPQ;
        c.workers=native_window_workers(std::max(a,b),o.workers);c.algorithm=SBN3_MUL_PQ16;c.T=16;c.cached=1;
        if(allow_ring){
            const auto ring=pq16::cyclic_shape(a,16);
            if(pq16::cyclic_supported(ring,a,b)&&(forced_ring||ring.nfull<linear.nfull)){
                c.ring=pq16::cyclic_period(ring);
                return materialize();
            }
        }
        if(!allow_ring&&a+b<=262144&&linear.nfull&&pq16::supported(linear,a,b,c.workers)){
            c.pfa=1;return materialize();
        }
    }
    if(!native_product(o,a,b,allow_ring,c))return false;
    // A ring which cannot remove any product output has no reduction to do.
    if(c.ring>=a+b&&!forced_ring){
        if(!o.prime_count&&width==1&&a+b<=262144){
            auto s=pq16::query(a,b);s.recipe=pq16::Recipe::PfaPQ;
            if(s.nfull&&pq16::supported(s,a,b,1)){
                c={0,SBN3_MUL_PQ16,1,16,0,1,1};return materialize();
            }
        }
        if(!native_product(o,a,b,false,c))return false;
    }
    return materialize();
}
} // namespace sbn::v3::product
