#pragma once
#include "product/window.hpp"
#include "product/cyclic_reconstruct.hpp"
#include "product/stage.hpp"
#include "product/build_apply.hpp"
#include "product/low_lift.hpp"
#include "product/difference.hpp"
#include "value/parallel_limbs.hpp"

namespace sbn::v3::product {
// Adapter for a precompiled group of cyclic products. All physical details
// stay here; its client uses the same windows as a full-product-only backend.
struct CyclicWindowSlot {
    sbn3_mul_binding *binding = nullptr;
    unsigned stage_slot = 0;
    size_t input_words = 0;
    uint64_t *input_padding = nullptr;
    bool cached = false, build_cache = false;
    sbn3_product_metrics *metrics = nullptr;
    bool fresh_in_cached = false;
};
struct CyclicWindowGroup {
    CyclicWindowSlot slots[3]{};
    sbn3_spectrum *cache = nullptr;
    const ProductStage *stage = nullptr;
    uint64_t *value = nullptr;
    size_t period = 0, capacity = 0, execute_capacity = 0;
    sbn3_team *team = nullptr;
    bool consume_value=false;
    size_t repair_bytes=0;
    uint64_t *witness=nullptr;
    size_t witness_words=0;
};
class CyclicWindowProducts {
    const CyclicWindowGroup &p_;
    ProductStageRun stage_;

    void run(unsigned slot, Span a, Span b) noexcept {
        require(slot < 3, SBN3_FATAL_ARGUMENT, "cyclic window slot");
        const auto &s = p_.slots[slot];
        auto *binding = stage_.bind(s.binding, s.stage_slot);
        if(b.words>p_.period){
            require(p_.repair_bytes&&b.words<=2*p_.period&&
                !overlaps(b.data,8*b.words,p_.value,8*p_.capacity),SBN3_FATAL_ARGUMENT,"folded cyclic input");
            memcpy(p_.value,b.data,8*p_.period);
            const auto carry=limbs::add_to(p_.value,p_.period,b.data+p_.period,b.words-p_.period);
            add_cyclic_word(p_.value,p_.period,carry);b={p_.value,p_.period};
        }
        const auto words = s.input_words ? s.input_words : b.words;
        require(words >= b.words && words <= p_.period,
                SBN3_FATAL_ARGUMENT, "cyclic window input support");
        const bool live=s.cached&&!s.build_cache&&binding->backend->product_execute_live;
        if (!live && (words != b.words || s.input_padding)) {
            require(s.input_padding, SBN3_FATAL_ARGUMENT, "cyclic window padding storage");
            parallel_limbs::copy(p_.team, s.input_padding, b.data, b.words);
            parallel_limbs::fill(p_.team, s.input_padding + b.words, words - b.words);
            b = {s.input_padding, words};
        }
        auto *cache = s.cached ? stage_.cache(p_.cache) : nullptr;
        const sbn3_limbs output{p_.value, p_.execute_capacity};
        if (cache && s.build_cache)
            spectrum_compute_multiply(binding, cache, {a.data,a.words}, {b.data,b.words}, output);
        else {
            const sbn3_product_inputs input{cache?sbn3_const_limbs{}:sbn3_const_limbs{a.data,a.words},
                                           {b.data,b.words},{},{}};
            if(s.fresh_in_cached){
                require(binding->backend->product_execute_fresh,SBN3_FATAL_ARGUMENT,"fresh product capability");
                binding->backend->product_execute_fresh(binding,nullptr,input,output);
            }else if(live)binding->backend->product_execute_live(binding,nullptr,input,output);
            else sbn3_product_execute(binding, &input, output);
        }
        if (s.metrics) sbn3_product_get_metrics(binding, s.metrics);
        if(!p_.repair_bytes)parallel_limbs::fill(p_.team, p_.value+p_.period, p_.capacity-p_.period);
    }
    void check_bound(MagnitudeBound b) const noexcept {
        if(!b.verify)return;
        require(b.high_exclusive && b.words < p_.capacity &&
                    p_.value[b.words] < b.high_exclusive &&
                    parallel_limbs::zero(p_.team,p_.value+b.words+1,p_.capacity-b.words-1),
                SBN3_FATAL_MATH,"cyclic product window bound");
    }
    WindowResult window(Window w,bool negative=false) const noexcept {
        require(w.origin <= p_.capacity && w.words <= p_.capacity-w.origin,
                SBN3_FATAL_ARGUMENT,"cyclic product window range");
        return {p_.value+w.origin,w.words,0,negative};
    }
    void subtract(ShiftedSpan sub) noexcept {
        if(!sub.value.words)return;
        if(sub.value.words==1&&sub.value.data[0]==1)
            limbs::cyclic_sub_power(p_.value,p_.period,sub.shift%p_.period);
        else {
            require(sub.shift<p_.period&&sub.value.words<=2*p_.period-sub.shift,
                    SBN3_FATAL_ARGUMENT,"cyclic cancellation addend");
            parallel_limbs::cyclic_sub_shifted(p_.team,p_.value,p_.period,sub.value.data,sub.value.words,sub.shift);
        }
    }
    WindowResult repaired(unsigned slot,Span a,Span b,ShiftedSpan sub,Window w,MagnitudeBound limit) noexcept {
        require(limit.words>=p_.period&&limit.high_exclusive<(uint64_t(1)<<63),SBN3_FATAL_ARGUMENT,"dynamic cyclic lift bound");
        const size_t g=limit.words-p_.period+1;
        require(g<=p_.period&&p_.capacity>=p_.period+g,SBN3_FATAL_ARGUMENT,"dynamic cyclic lift capacity");
        const auto &s=p_.slots[slot];auto *binding=stage_.bind(s.binding,s.stage_slot);
        require(binding->backend->with_product_scratch,SBN3_FATAL_ARGUMENT,"cyclic lift scratch loan");
        auto *witness=p_.witness?p_.witness:p_.value+p_.period;
        require((!p_.witness||(g<=p_.witness_words&&!overlaps(witness,8*g,p_.value,8*p_.capacity)))&&
                !overlaps(witness,8*g,a.data,8*a.words)&&!overlaps(witness,8*g,b.data,8*b.words)&&
                !overlaps(witness,8*g,sub.value.data,8*sub.value.words),SBN3_FATAL_ARGUMENT,"cyclic witness lifetime");
        struct Low {Span a,b;ShiftedSpan sub;size_t words;uint64_t *tail;} args{a,b,sub,g,witness};
        binding->backend->with_product_scratch(binding,&args,[](void *arg,Frame &scratch){
            auto &x=*static_cast<Low*>(arg);auto *low=low_lift_prepare(scratch,x.words,x.a,x.b,x.sub);
            memcpy(x.tail,low,8*x.words);
        });
        run(slot,a,b);subtract(sub);
        const bool negative=p_.witness?reconstruct_cyclic(p_.value,p_.period,witness,g):
                                      reconstruct_cyclic_tail(p_.value,p_.period,g);
        parallel_limbs::fill(p_.team,p_.value+p_.period+g,p_.capacity-p_.period-g);
        check_bound(limit);return window(w,negative);
    }
  public:
    explicit CyclicWindowProducts(const CyclicWindowGroup &p) noexcept : p_(p),stage_(p.stage) {
        require(p.period>=3 && p.capacity>=p.period && p.execute_capacity>=p.period,
                SBN3_FATAL_ARGUMENT,"cyclic window storage");
    }
    WindowResult multiply(unsigned slot,Span a,Span b,Window w,MagnitudeBound limit) noexcept {
        if(p_.repair_bytes){auto result=repaired(slot,a,b,{},w,limit);
            require(!result.negative,SBN3_FATAL_MATH,"positive dynamic cyclic lift");return result;}
        const bool lift=limit.words>=p_.period;
        uint64_t low[3];
        if(lift){
            require(a.words>=3&&b.words>=3&&limit.words-p_.period<=2&&
                        p_.capacity>=p_.period+3&&limit.high_exclusive<(uint64_t(1)<<63),
                    SBN3_FATAL_ARGUMENT,"cyclic product lift capacity");
            product_low_words(low,a.data,b.data);
        }
        run(slot,a,b);
        if(lift)require(!reconstruct_cyclic(p_.value,p_.period,low),SBN3_FATAL_MATH,"positive product lift");
        check_bound(limit);return window(w);
    }
    WindowResult cancel(unsigned slot,Span a,Span b,ShiftedSpan subtrahend,
                        Window w,MagnitudeBound limit) noexcept {
        if(p_.repair_bytes)return repaired(slot,a,b,subtrahend,w,limit);
        // This adapter admits one low-word lift for cancellation. Wider
        // signed lifts can be added independently of the arithmetic recipe.
        require(limit.words<=p_.period && limit.high_exclusive<(uint64_t(1)<<63),
                SBN3_FATAL_ARGUMENT,"cyclic cancellation bound");
        uint64_t low[1]={a.data[0]*b.data[0]};
        if(!subtrahend.shift&&subtrahend.value.words)low[0]-=subtrahend.value.data[0];
        run(slot,a,b);
        subtract(subtrahend);
        bool negative;
        if(limit.words==p_.period){
            require(p_.capacity>p_.period,SBN3_FATAL_ARGUMENT,"cyclic residual lift capacity");
            negative=reconstruct_cyclic(p_.value,p_.period,low);
        }else negative=parallel_limbs::cyclic_absolute(p_.team,p_.value,p_.period);
        check_bound(limit);return window(w,negative);
    }
    WindowResult retain(WindowResult r,uint64_t *destination) const noexcept {
        if(p_.consume_value)return r;
        require(destination,SBN3_FATAL_ARGUMENT,"cyclic retained window");
        parallel_limbs::copy(p_.team,destination,r.data,r.words);r.data=destination;return r;
    }
};
} // namespace sbn::v3::product
