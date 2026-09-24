#pragma once
#include "product/window.hpp"
#include "product/backend.hpp"
#include "product/fixed_geometry.hpp"
#include "product/window_fft.hpp"
#include "product/window_policy.hpp"
#include "backend/pq16/kernels.hpp"
#include "common/identity.hpp"
#include "product/low_lift.hpp"
#include "product/cost_model.hpp"
#include "product/root_prepare_cost.hpp"
#include "product/u52_execution_cost.hpp"

namespace sbn::v3::product {
// Product-owned physical selection. Arithmetic clients carry/replay it but
// do not select its codec, layout or ring. The public service ABI is unchanged.
struct WindowGroupChoice {
    unsigned np=0,algorithm=0,workers=1;
    int T=0;
    size_t ring=0;
};
struct WindowGroupPlans {
    sbn3_mul_plan producer{},plans[2]{};
    sbn3_product_info producer_info{},infos[2]{};
    sbn3_spectrum_desc future{};
    unsigned count=0;
    bool consume_residual=false;
    size_t repair_bytes=0;
};
struct WindowGroupOptions {unsigned workers=1,prime_count=0;};
inline uint64_t window_choice_identity(const WindowGroupChoice &c) noexcept {
    uint64_t h=1469598103934665603ULL;
    for(uint64_t x:{uint64_t(c.np),uint64_t(c.algorithm),uint64_t(c.workers),uint64_t(c.T),uint64_t(c.ring)})
        h=identity::word(h,x);
    return h;
}
inline bool materialize_window_group(const WindowGroupShape &s,const WindowGroupChoice &c,WindowGroupPlans &out) {
    const unsigned cancel=s.count-2;
    const auto &first=s.products[0],&last=s.products[s.count-1],&res=s.products[cancel];
    sbn3_mul_options o{};o.workers=c.workers;o.prime_count=c.np;o.trunk_bits=c.T;o.algorithm=c.algorithm;
    o.borrow_output=s.count==3?2:1;
    sbn3_product_request r{};r.kind=SBN3_PRODUCT_MUL;r.cyclic_limbs=c.ring;
    r.a_limbs=first.a.words;r.b_limbs=std::min(c.ring,std::max(first.b.words,last.b.words));
    out.repair_bytes=window_group_low_lift_bytes(s,c.ring);
    if(sbn3_product_query(&r,&o,&out.producer,&out.producer_info)!=SBN3_SUPPORTED)return false;
    if(c.algorithm==SBN3_MUL_SCALAR){out.future={};out.plans[0]=out.producer;out.infos[0]=out.producer_info;}
    else{
        if(sbn3_spectrum_query(&out.producer,SBN3_SPECTRUM_COLUMNS,res.bound.words,&out.future)!=SBN3_SUPPORTED)return false;
        r.cached_a[0]=&out.future;
        if(sbn3_product_query(&r,&o,&out.plans[0],&out.infos[0])!=SBN3_SUPPORTED)return false;
    }
    out.count=1;
    const auto *backend=backend_lookup(out.producer.opaque[1]);
    if(out.repair_bytes&&(!backend||!backend->product_scratch_bytes||!backend->with_product_scratch||
        backend->product_scratch_bytes(out.plans[0])<out.repair_bytes))return false;
    // Bounded live support lets the correction use the already-bound
    // cancellation program at every width without transforming zero tails.
    if(s.count==2&&c.algorithm!=SBN3_MUL_SCALAR&&
       (c.workers>1||(backend&&backend->product_execute_live))){
        out.consume_residual=backend&&backend->product_execute_live&&backend->product_live_contract&&
            (backend->product_live_contract(out.plans[0])&program_consume_inputs);
        return true;
    }
    if(s.count==3&&backend&&backend->product_fresh_supported&&backend->product_execute_fresh&&
       backend->product_fresh_supported(out.plans[0],res.a.words,std::min(res.b.words,c.ring)))return true;
    if(s.count==3){r.cached_a[0]=nullptr;r.a_limbs=res.a.words;r.b_limbs=std::min(res.b.words,c.ring);}
    else{r.a_limbs=last.a.words;r.b_limbs=last.b.words;}
    if(sbn3_product_query(&r,&o,&out.plans[1],&out.infos[1])!=SBN3_SUPPORTED)return false;
    out.count=2;return true;
}
// In addition to the enclosing choice, consider one smaller period per
// prime family, at that family's widest certified digit width. Arithmetic
// supplies a magnitude bound; only this provider may change physical period.
inline void smaller_ntt_window(const WindowGroupShape&s,WindowGroupChoice &choice){
    if(choice.workers!=1||choice.algorithm!=SBN3_MUL_FLAT)return;
    const auto &res=s.products[s.count-2];const size_t n=res.bound.words;
    const auto *baseline=backend_lookup(choice.np);sbn3_product_request r{};
    r.a_limbs=res.a.words;r.b_limbs=res.b.words;r.cyclic_limbs=choice.ring;
    sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_FLAT;o.prime_count=choice.np;o.trunk_bits=choice.T;
    sbn3_mul_info initial{};
    if(!baseline||baseline->geometry_query(r,o,initial)!=SBN3_SUPPORTED)return;
    auto volume=[](const sbn3_mul_info&i){return double(i.np)*i.transform_trunks*(i.np<=8?mul_np_cost[i.np]:1.);};
    const double unit=cost_model::cyclic_product(initial,false).nanoseconds/volume(initial);
    const double transforms=s.count==2?5./3.:8./3.;
    double best=unit*volume(initial)*transforms+root_prepare_cost(initial);
    for(unsigned np=4;np<=8;++np){
        auto *backend=backend_lookup(np);if(!backend||!backend->product_live_contract||!backend->with_product_scratch)continue;
        r.cyclic_limbs=0;r.b_limbs=res.b.words;o.prime_count=np;o.trunk_bits=0;
        sbn3_mul_info i{};if(backend->geometry_query(r,o,i)!=SBN3_SUPPORTED)continue;
        const int T=int(i.trunk_bits);size_t factor=1;while(2*factor*size_t(T)<n)factor*=2;
        const size_t ring=factor*size_t(T);
        size_t g=0;bool supported=ring<n;
        for(unsigned j=0;j<s.count;++j){const auto &p=s.products[j];
            supported&=p.a.words<ring&&p.b.words<=2*ring&&p.bound.words>=ring&&p.bound.high_exclusive<(uint64_t(1)<<63);
            g=std::max(g,p.bound.words>=ring?p.bound.words-ring+1:0);
            if(p.cancellation&&p.subtract.words>1)supported&=p.subtract.shift<ring&&p.subtract.words<=2*ring-p.subtract.shift;}
        if(!supported||g>ring/4||factor>native_policy::window_serial_flat_trunks/64)continue;
        r.cyclic_limbs=ring;r.b_limbs=std::min(res.b.words,ring);o.trunk_bits=T;
        if(backend->geometry_query(r,o,i)!=SBN3_SUPPORTED||volume(i)>=volume(initial)||window_group_low_lift_bytes(s,ring)>i.pool_bytes)continue;
        double work=unit*volume(i)*transforms+root_prepare_cost(i);
        for(unsigned j=0;j<s.count;++j){const size_t low=s.products[j].bound.words-ring+1;
            if(low<=8)continue;
            auto fft=local_window_detail::linear_shape(low,low);
            work+=fft.nfull?pq16::native_cost(fft,low,low):u52_execution_cost(low,low);}
        if(work<best){best=work;choice={np,SBN3_MUL_FLAT,1,T,ring};}
    }
}
inline bool compile_window_group(const WindowGroupShape &s,WindowGroupOptions o,WindowGroupChoice &choice,
                                  WindowGroupPlans &out,bool replay=false) {
    require((s.count==2||s.count==3)&&s.products[s.count-2].cancellation,
            SBN3_FATAL_ARGUMENT,"compiled product window group");
    if(replay)return materialize_window_group(s,choice,out);
    const auto &res=s.products[s.count-2];const size_t minimum=res.bound.words;
    const bool use_fft=!o.prime_count&&(o.workers==1||inline_window_preferred(minimum,0,o.workers));
    WindowGroupChoice c{};c.workers=native_window_workers(minimum,o.workers);
    if(!o.prime_count&&minimum<=256){c.algorithm=SBN3_MUL_SCALAR;c.workers=1;c.ring=minimum;}
    else if(use_fft){
        const auto shape=cyclic_window_fft(s);
        if(pq16::cyclic_supported(shape,res.a.words,res.b.words)){
            c.algorithm=SBN3_MUL_PQ16;c.workers=1;c.T=int(shape.bits);c.ring=pq16::cyclic_period(shape);
        }
    }
    if(!c.ring){
        FixedNttGeometry g;
        if(!fixed_ntt_geometry(res.a.words,res.b.words,minimum,c.workers,o.prime_count,g,
                               native_policy::window_parallel_flat_words,
                               c.workers==1?native_policy::window_serial_flat_trunks:0))return false;
        c={g.np,g.algorithm,g.workers,g.T,g.ring};
        if(!o.prime_count)smaller_ntt_window(s,c);
    }
    if(!materialize_window_group(s,c,out))return false;
    choice=c;return true;
}
} // namespace sbn::v3::product
