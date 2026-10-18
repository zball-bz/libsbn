// SPDX-License-Identifier: LGPL-3.0-or-later
#include "planning/product.hpp"
#include "common/checked.hpp"
#include "backend/ntt_p48/root_bank.hpp"
#include "sbn3/product.h"
#include "product/backend.hpp"
#include "product/tuning_native.hpp"
#include "external/row_cache.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
namespace sbn::v3::planning {
namespace {
using external::SpilledProductPlan;using external::SpilledProductSpec;
constexpr size_t Q=external::chunk_bytes;
size_t floor_power(long double x,size_t maximum){
    if(!(x>=1))return 0;
    if(x>=maximum)return maximum;
    return std::bit_floor(size_t(x));
}
size_t minimum_stripe(unsigned T,size_t M){return std::min(M,std::max(size_t(1024),Q/(size_t(1)<<__builtin_ctz(T))));}
size_t minimum_period(const ExternalProductRequest&r){
    if(!r.middle)return std::max({r.cyclic.minimum_words,r.an,r.bn});
    size_t words;if(r.cyclic||!add_size(r.an,r.bn,words)||!r.middle.first||r.middle.first>=r.middle.end||r.middle.end>words)return 0;
    return std::max({r.an,r.bn,r.middle.end,words-r.middle.first+1});
}
bool cyclic_basis(const ExternalProductRequest&r,unsigned&np,unsigned&T,unsigned&queries){
    if(r.primes&&(r.primes<4||r.primes>10))return false;
    // One certified period per compiled prime basis, as in fixed_ntt_geometry.
    // External codecs require byte alignment; their complete ring cost is the
    // residue volume, without the unrelated native Bailey column-tower penalty.
    const unsigned first=r.primes?r.primes:4,last=r.primes?r.primes:10;
    const size_t need=minimum_period(r);if(!need)return false;
    bool found=false;double best=0;
    for(unsigned q=first;q<=last;++q){
        const auto*backend=backend_lookup(q);const auto*crt=product::p48_crt_backend(q);
        if(!backend||!backend->geometry_query||!crt)continue;
        unsigned maximum=r.trunk_bits;
        if(!maximum){
            sbn3_product_request request{};request.kind=r.square?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;
            request.a_limbs=r.an;request.b_limbs=r.square?0:r.bn;
            sbn3_mul_options options{};options.workers=r.workers;options.algorithm=SBN3_MUL_BAILEY;options.prime_count=q;
            sbn3_mul_info info{};++queries;
            if(backend->geometry_query(request,options,info)!=SBN3_SUPPORTED)continue;
            maximum=info.trunk_bits&~7u;
        }
        const unsigned minimum=q==4?80:24*q-32;
        if(maximum<minimum||maximum%8||!crt->prepared_bytes(maximum))continue;
        const size_t cells=need/(maximum/8)+(need%(maximum/8)!=0);
        if(!cells||cells>(uint64_t(1)<<32))continue;
        const size_t N=std::bit_ceil(cells);
        const unsigned width=r.trunk_bits?maximum:std::max(minimum,unsigned(8*(need/N+(need%N!=0))));
        size_t bits_a,bits_b;if(!mul_size(r.an,64,bits_a)||!mul_size(r.bn,64,bits_b))return false;
        const size_t na=bits_a/width+(bits_a%width!=0),nb=bits_b/width+(bits_b%width!=0);
        if(width>maximum||!crt->prepared_bytes(width)||!crt->capacity(width,std::min(na,nb)))continue;
        const double cost=double(q)*N*(q<=8?mul_np_cost[q]:1.);
        if(!found||cost<best){found=true;best=cost;np=q;T=width;}
    }
    return found;
}
bool disk_peak(const SpilledProductPlan&p,uint64_t other,uint64_t&out){
    external::RegionLayout a{},b{},r{};
    if(external::region_layout(p.forward[0].source_bytes,0,a)||external::region_layout(p.forward[1].source_bytes,0,b)||external::region_layout(p.stored_output_bytes,0,r))return false;
    uint64_t planes;if(__builtin_mul_overflow(p.plane_allocation_bytes,uint64_t(p.plane_count),&planes)||
       __builtin_add_overflow(other,a.allocation_bytes,&out)||(!p.spec.square&&__builtin_add_overflow(out,b.allocation_bytes,&out))||
       __builtin_add_overflow(out,r.allocation_bytes,&out)||__builtin_add_overflow(out,planes,&out))return false;
    if(p.spec.cyclic.stored_subtract){external::RegionLayout sub{};uint64_t bytes;
        if(__builtin_mul_overflow(uint64_t(p.spec.cyclic.subtract_source.storage_limbs),uint64_t(8),&bytes)||
           external::region_layout(bytes,0,sub)||__builtin_add_overflow(out,sub.allocation_bytes,&out))return false;
    }
    return true;
}
double price(const SpilledProductPlan&p,const ExternalProductProfile&f){
    const double cells=double(p.middle.spec.rows)*p.spec.inner_cells,primes=p.spec.primes;
    const unsigned log=unsigned(__builtin_ctzll(p.spec.outer_cells))+unsigned(__builtin_ctzll(p.spec.inner_cells));
    const double transforms=p.spec.square||p.cached_square?2:3;
    const unsigned outer_log=unsigned(__builtin_ctzll(p.spec.outer_cells));
    const double work=primes*cells*(transforms*log+16-(p.cached_a?(p.middle.spec.a_transformed?log:outer_log):0))*f.vector_level_ns/std::min(p.spec.workers,16u);
    const double raw_cells=(p.cached_square?0.:p.cached_a?double(p.spec.bn):double(p.spec.an+(p.spec.square?0:p.spec.bn)))*8/p.spec.trunk_bits;
    const double decode=raw_cells*((p.spec.primes+p.spec.prime_batch-1)/p.spec.prime_batch)*f.funnel_cell_ns/p.spec.workers;
    const unsigned depth=p.middle.row.depth,upper=(depth+2)/3;
    const double memory=(transforms-(p.cached_a&&p.middle.spec.a_transformed?1:0))*128*primes*cells*upper/f.memory_bytes_per_ns;
    const double bytes=double(p.normal_read_bytes)+p.normal_write_bytes;
    // Fractional request estimate ranks run lengths; byte admission is exact.
    const double run=std::min(double(f.target_transfer_bytes),double(p.forward[0].info.fragment_bytes));
    const double retention_copy=4.*p.spec.retained_bytes/f.memory_bytes_per_ns;
    // Resident forward reads the compact source directly. Charge its additional
    // residue scan and the byte-funnel sharing lost by one-prime execution.
    const double raw_copy=p.spec.raw_operand<2?double(p.raw_bytes)/f.memory_bytes_per_ns:0.;
    const double raw_decode=p.spec.raw_operand<2?
        double(p.spec.raw_operand?p.spec.bn:p.spec.an)*8/p.spec.trunk_bits*
        (p.spec.primes-(p.spec.primes+p.spec.prime_batch-1)/p.spec.prime_batch)*f.funnel_cell_ns/p.spec.workers:0.;
    return work+decode+memory+retention_copy+raw_copy+raw_decode+bytes/f.io_bytes_per_ns+bytes/run*f.request_ns;
}
struct Bounds {long double fixed_forward,fixed_inverse,A,B,worker_slope;};
Bounds bounds(const SpilledProductPlan&p){
    const long double C=p.spec.outer_cells,q=p.middle.spec.rows,M=p.spec.inner_cells,np=p.spec.primes,w=p.spec.workers,T=p.spec.trunk_bits;
    // Conservative alignment/header margins; no floating-point bound admits
    // memory. The generated request still needs its exact complete quote.
    const long double fw=std::max(p.forward[0].info.prepared_bytes,p.forward[1].info.prepared_bytes)+
        (1u<<20)+external::index_page_bytes+std::max(p.forward[0].input_index_bytes,p.forward[1].input_index_bytes)+64*C+64+32768;
    const long double inv=p.inverse.info.prepared_bytes+p.emit.crt->prepared_bytes(p.spec.trunk_bits)+Q*3+
        w*(sizeof(product::CrtCarry)+256+4096)+65536;
    return {fw,inv,64*np*q+w*T,q*M*(Q+sizeof(external::CarrySegment)),4*64*(C+1)*w};
}
// A cache-bounded row and exact lifetime replay. Initial stripe granularity
// uses a conservative one-chunk-per-segment allowance. A retained raw operand
// then solves the actual final-bank A*J+B/J bound inside its necessary prime
// phase. Prime batches balance a bounded number of IO lanes, without a grid.
bool streaming_recipe(const ExternalProductRequest&r,const ExternalProductBudget&budget,const ExternalProductProfile&profile,
                      unsigned np,unsigned T,size_t N,size_t live,size_t fixed,ExternalProductSelection&out){
    if(r.workers<4||!profile.active_row_bytes||!profile.io_workers)return false;
    const size_t cap=profile.active_row_bytes/(r.square?64:128);
    if(!cap)return false;
    const size_t M=std::min(N,std::max(std::max(size_t(1),N/8192),std::bit_floor(cap)));
    const size_t C=N/M,rows=live/M+(live%M!=0),minJ=minimum_stripe(T,M);
    if(rows<3||M<=minJ)return false;
    const long double A=(2.L*np*64+2.L*T)*rows,B=(long double)rows*M*(Q+sizeof(external::CarrySegment));
    size_t J=std::max(minJ,floor_power(std::sqrt(B/A),M/2));
    if(J<M/2&&A*(2*J)+B/(2*J)<A*J+B/J)J*=2;
    const unsigned lanes=std::min(profile.io_workers,r.workers/2),waves=(np+lanes-1)/lanes;
    SpilledProductSpec spec{r.an,r.bn,C,M,J,np,T,(np+waves-1)/waves,r.workers,r.square};
    spec.a_source=r.a_source;spec.b_source=r.b_source;spec.cyclic=r.cyclic;spec.middle=r.middle;
    spec.compact_middle=spec.duplex_middle=spec.stream_final=true;
    SpilledProductPlan base;
    ++out.summary_quotes;
    if(!external::query_spilled_product(spec,base)||!external::bound_spilled_output(base,r.stored_result_limbs))return false;
    const size_t room=budget.resident_bytes-fixed;
    bool found=false;SpilledProductPlan best{};uint64_t best_disk=0;double best_ns=0;
    auto admit=[&](SpilledProductPlan p){
        size_t total;if(!add_size(fixed,p.storage_bytes,total))return;
        out.smallest_quoted_resident=std::min(out.smallest_quoted_resident,total);
        if(p.storage_bytes>room)return;
        const size_t held=size_t(std::min(uint64_t((room-p.storage_bytes)/Q*Q),uint64_t(np)*p.plane_bytes));
        if(!external::retain_spilled_prefix(p,held)||p.normal_write_bytes>budget.write_limit)return;
        uint64_t disk;if(!disk_peak(p,budget.other_owned_disk_bytes,disk)||disk>budget.owned_disk_limit)return;
        p.middle.parallel_reads=true;for(auto&f:p.forward)f.parallel_writes=true;
        const double ns=price(p,profile);if(!std::isfinite(ns))return;
        // OOC's primary resource is transferred data. Equal-volume recipes use
        // the conservative arithmetic estimate as a tie-breaker only.
        const __uint128_t traffic=__uint128_t(p.normal_read_bytes)+p.normal_write_bytes;
        const __uint128_t previous=__uint128_t(best.normal_read_bytes)+best.normal_write_bytes;
        if(!found||traffic<previous||(traffic==previous&&ns<best_ns)){
            best=p;best_disk=disk;best_ns=ns;found=true;
        }
    };
    auto admit_raw=[&](SpilledProductPlan p){
        admit(p);
        if(p.storage_bytes>room&&external::fit_raw_prefix(p,room,budget.write_limit))admit(p);
    };
    admit(base);
    if(!r.square){
        const unsigned operand=r.an<=r.bn?0:1;auto raw=base;
        if(external::use_raw_operand(raw,operand)){
            (void)external::duplex_middle_within(raw,raw.storage_bytes);
            admit_raw(raw);
            const size_t phase=raw.raw_phase+std::max(external::forward_memory_workspace(raw.forward[operand]),raw.middle.storage_bytes);
            const auto&e=raw.emit;const auto&i=raw.inverse;
            const long double banks=raw.final_banks,tail_slope=64.L*rows/i.fused.tile_cells;
            const long double emit_slope=e.row_slots*(long double)T*(1+unsigned(e.cyclic.stored_subtract));
            const long double final_A=banks*(48.L*np*rows+tail_slope)+emit_slope;
            const long double final_B=(long double)rows*M*sizeof(external::CarrySegment);
            const long double constant=banks*((long double)i.storage_bytes-np*(long double)rows*J*48-tail_slope*J)+
                e.storage_bytes-e.carry.descriptor_bytes-emit_slope*J+16384;
            const long double allowance=(long double)phase-constant,disc=allowance*allowance-4*final_A*final_B;
            if(allowance>0&&disc>=0){
                long double upper=(allowance+std::sqrt(disc))/(2*final_A);
                const auto&f=raw.forward[1-operand];
                const size_t input_rows=size_t((f.kernel.input_cells+M-1)/M);
                const long double fw_constant=f.info.prepared_bytes+f.info.workspace_bytes+f.input_index_bytes+
                    (1u<<20)+external::index_page_bytes+64.L*C+32768;
                const long double fw_slope=(long double)input_rows*T+2.L*spec.prime_batch*rows*48;
                upper=std::min(upper,((long double)phase-fw_constant)/fw_slope);
                const size_t target=std::bit_ceil(std::min(M/2,std::max(size_t(1),profile.target_transfer_bytes/48)));
                const size_t wider=std::min(target,floor_power(upper,M/2));
                if(wider>J){auto larger=spec;larger.stripe_cells=wider;larger.compact_middle=false;
                    larger.raw_operand=operand;SpilledProductPlan p;++out.summary_quotes;
                    if(external::query_spilled_product(larger,p)&&external::bound_spilled_output(p,r.stored_result_limbs)&&
                       p.storage_bytes<=raw.storage_bytes)admit_raw(p);
                }
            }
        }
    }
    if(!found){
        // If the grant/write cap cannot afford overlapping banks, retain the
        // same arithmetic geometry and replay its serial storage lifetime.
        // Removing overlap frees bytes for the retained plane prefix.
        auto serial=spec;serial.compact_middle=serial.duplex_middle=serial.stream_final=false;
        SpilledProductPlan p;++out.summary_quotes;
        if(external::query_spilled_product(serial,p)&&external::bound_spilled_output(p,r.stored_result_limbs))admit(p);
    }
    if(!found)return false;
    // Forward staging uses fewer simultaneous primes than inverse+CRT. Give
    // that pass a larger transfer width from the already admitted phase pool.
    // Input and output planes keep the same row-major address mapping.
    size_t forward=std::bit_floor(std::min(M,std::max(size_t(1),profile.target_transfer_bytes/48)));
    for(unsigned k=0;k<(r.square?1u:2u);++k)if(best.spec.raw_operand!=k){
        const auto&f=best.forward[k];const size_t input_rows=size_t((f.kernel.input_cells+M-1)/M);
        const long double fixed_forward=f.info.prepared_bytes+f.info.workspace_bytes+f.transfer_bytes+
            external::index_page_bytes+f.input_index_bytes+64.L*C+32768;
        const long double slope=(long double)input_rows*T+2.L*best.spec.prime_batch*rows*48;
        forward=std::min(forward,floor_power(((long double)best.pool_bytes-fixed_forward)/slope,M));
    }
    if(forward>best.spec.forward_stripe_cells&&external::widen_forward_within(best,forward,best.storage_bytes))best_ns=price(best,profile);
    // A partially retained raw source is reread once per prime. Size its
    // staging by source bytes per transfer, independently of packed residues.
    // One direct lifetime replay may trade a little raw retention for fewer
    // requests. The existing byte/request cost decides; writes may not grow.
    if(best.raw_forward.storage_bytes){
        const size_t width=std::bit_floor(std::min(M,std::max(size_t(1),profile.target_transfer_bytes/T)));
        const size_t old=best.raw_forward.kernel.stripe_cells;
        if(width>old&&profile.io_bytes_per_ns>0&&profile.request_ns>0){
            auto next=best;next.spec.raw_stripe_cells=width;++out.summary_quotes;
            if(external::fit_raw_prefix(next,best.storage_bytes,budget.write_limit)&&next.normal_write_bytes<=best.normal_write_bytes&&
               next.raw_forward.kernel.stripe_cells>old){
                const double extra=(double(next.normal_read_bytes)-double(best.normal_read_bytes))/profile.io_bytes_per_ns;
                const double saved=(double(best.raw_read_bytes)/(old*double(T))-double(next.raw_read_bytes)/(next.raw_forward.kernel.stripe_cells*double(T)))*profile.request_ns;
                if(extra<saved){best=next;best_ns=price(best,profile);}
            }
        }
    }
    // Final IO need not inherit the raw-source staging width. Its live banks
    // cost K + A*J + B/J: packed coefficients/row outputs grow with J, carry
    // descriptors shrink with J. Solve that bound once inside the phase pool;
    // keep three banks and cache-sized arithmetic tiles. An exact quote admits
    // the one candidate without displacing retention or changing byte traffic.
    if(best.final_banks==3&&best.inverse.fused.workspace_bytes>=best.inverse.info.workspace_bytes&&best.emit.carry.stream_segments){
        const auto&i=best.inverse;const auto&e=best.emit;
        const long double old=i.spec.stripe_cells;
        const long double A=3.L*(48.L*np*rows+(long double)sizeof(product::CrtCarry)*rows/i.fused.tile_cells)+
            e.row_slots*(long double)T*(1+unsigned(e.cyclic.stored_subtract));
        const long double B=e.carry.descriptor_bytes*old;
        const long double K=best.emit_offset+(long double)e.storage_bytes-A*old-B/old+16384;
        const long double room=(long double)(best.storage_bytes-best.inverse_offset)-K,disc=room*room-4*A*B;
        if(room>0&&disc>=0){
            const size_t target=std::bit_floor(std::min(M/4,std::max(size_t(1),profile.target_transfer_bytes/48)));
            const size_t width=std::min(target,floor_power((room+std::sqrt(disc))/(2*A),M/4));
            if(width>i.spec.stripe_cells){++out.summary_quotes;
                if(external::widen_inverse_within(best,width,best.storage_bytes))best_ns=price(best,profile);}
        }
    }
    out.plan=best;out.total_resident_bytes=fixed+best.storage_bytes;out.owned_disk_bytes=best_disk;
    out.estimated_ns=best_ns;out.admitted=true;return true;
}
}
double estimate_external_product(const external::SpilledProductPlan&p,const ExternalProductProfile&f) noexcept {return price(p,f);}
double estimate_forward_rows(const external::ForwardPlanesPlan&p,const ExternalProductProfile&f) noexcept {
    const double cells=double(p.spec.live_rows)*p.spec.inner_cells,np=p.spec.prime_count;
    const double work=np*cells*(unsigned(__builtin_ctzll(p.spec.outer_cells))+4)*f.vector_level_ns/std::min(p.spec.workers,16u);
    const double decode=double(p.spec.input_limbs)*8/p.spec.trunk_bits*((p.spec.prime_count+p.spec.prime_batch-1)/p.spec.prime_batch)*f.funnel_cell_ns/p.spec.workers;
    const double bytes=double(p.source_read_bytes)+np*p.plane_bytes;
    const double run=std::min(double(f.target_transfer_bytes),double(p.info.fragment_bytes));
    return work+decode+112*np*cells/f.memory_bytes_per_ns+bytes/f.io_bytes_per_ns+bytes/run*f.request_ns;
}
double estimate_row_cache(const external::RowCachePlan&p,const ExternalProductProfile&f) noexcept {
    double cost=estimate_forward_rows(p.forward,f);
    if(p.kind==external::RowCacheKind::completed_columns){
        const auto&s=p.forward.spec;const double cells=double(s.live_rows)*s.inner_cells,np=s.prime_count;
        const unsigned log=unsigned(__builtin_ctzll(s.inner_cells));
        cost+=np*cells*log*f.vector_level_ns/p.column.workers;
        cost+=128*np*cells*((p.column.depth+2)/3+1)/f.memory_bytes_per_ns;
        cost+=double(s.input_limbs)*8/s.trunk_bits*(np-(s.prime_count+s.prime_batch-1)/s.prime_batch)*f.funnel_cell_ns/s.workers;
        cost+=(double(p.normal_read_bytes)-p.forward.source_read_bytes)/f.io_bytes_per_ns;
    }
    return cost;
}
bool complete_cache_within(external::RowCachePlan&cache,std::span<external::SpilledProductPlan *>consumers,size_t work,
                           const ExternalProductProfile&profile) noexcept {
    if(cache.kind!=external::RowCacheKind::outer_rows||consumers.size()<2)return false;
    external::RowCachePlan next;
    if(!external::query_row_cache(cache.forward,next,external::RowCacheKind::completed_columns)||next.storage_bytes!=cache.storage_bytes||
       next.build_bytes>work||next.normal_write_bytes>cache.normal_write_bytes||next.normal_read_bytes>cache.normal_read_bytes)return false;
    double before=estimate_row_cache(cache,profile),after=estimate_row_cache(next,profile);
    for(const auto*p:consumers){if(!p)return false;auto candidate=*p;
        if(!external::use_completed_cached_a(candidate))return false;
        before+=estimate_external_product(*p,profile);after+=estimate_external_product(candidate,profile);
    }
    if(!std::isfinite(after)||after>before)return false;
    cache=next;for(auto*p:consumers)p->middle.spec.a_transformed=true;
    return true;
}
bool prefer_raw_pair(external::RowCachePlan&cache,SpilledProductPlan&a,SpilledProductPlan&b,size_t limit,
                     const ExternalProductProfile&profile) noexcept {
    if(!cache.storage_bytes||!a.cached_a||!b.cached_a||a.cached_square||b.cached_square)return false;
    auto replay=[&](const SpilledProductPlan&old,SpilledProductPlan&next){
        auto spec=old.spec;spec.retained_bytes=0;spec.raw_operand=0;spec.raw_cache_bytes=SIZE_MAX;
        // A correction's current window can be narrower than its compiled
        // backing envelope (chunk-aligned residuals include a leading seam).
        // Rebuild against that envelope, then restore the current view. Merely
        // copying spec would shrink the pinned-index/source capacity.
        spec.a_source.storage_limbs=old.forward[0].source_capacity_bytes/8;
        spec.b_source.storage_limbs=old.forward[1].source_capacity_bytes/8;
        if(!external::query_spilled_product(spec,next)||!external::rebind_spilled_sources(next,old.spec.a_source,old.spec.b_source)||
           !external::bound_spilled_output(next,size_t(old.max_stored_output_bytes/8))||
           !external::project_spilled_output(next,size_t(old.output_first_byte/8),size_t((old.output_first_byte+old.stored_output_bytes)/8)))return false;
        next.emit.bound=old.emit.bound;next.emit.scalar=old.emit.scalar;
        if(next.storage_bytes>limit&&!external::fit_raw_prefix(next,limit))return false;
        if(next.storage_bytes>limit)return false;
        const size_t held=size_t(std::min(uint64_t((limit-next.storage_bytes)/Q*Q),uint64_t(next.spec.primes)*next.plane_bytes));
        if(!external::retain_spilled_prefix(next,held))return false;
        next.middle.parallel_reads=old.middle.parallel_reads;
        for(unsigned i=0;i<2;++i)next.forward[i].parallel_writes=old.forward[i].parallel_writes;
        return true;
    };
    SpilledProductPlan x,y;if(!replay(a,x)||!replay(b,y))return false;
    const __uint128_t before_w=__uint128_t(cache.normal_write_bytes)+a.normal_write_bytes+b.normal_write_bytes,
                      after_w=__uint128_t(x.normal_write_bytes)+y.normal_write_bytes;
    const __uint128_t before=before_w+cache.normal_read_bytes+a.normal_read_bytes+b.normal_read_bytes,
                      after=after_w+x.normal_read_bytes+y.normal_read_bytes;
    const double old_ns=estimate_row_cache(cache,profile)+price(a,profile)+price(b,profile),new_ns=price(x,profile)+price(y,profile);
    if(after_w>before_w||after>=before||!std::isfinite(new_ns)||new_ns>old_ns*profile.write_saving_slowdown)return false;
    cache={};a=x;b=y;return true;
}
ExternalPlanStatus choose_external_product(const ExternalProductRequest&r,const ExternalProductBudget&budget,
                                           const ExternalProductProfile&profile,ExternalProductSelection&out) noexcept {
    out={};
    if(!r.workers||r.workers>32||(r.square&&r.an!=r.bn)||!profile.leaf_cells||!std::has_single_bit(profile.leaf_cells)||
       !profile.target_transfer_bytes||!(profile.io_bytes_per_ns>0)||!(profile.memory_bytes_per_ns>0)||
       !(profile.request_ns>=0)||!(profile.vector_level_ns>=0)||!(profile.funnel_cell_ns>=0)||
       !(profile.write_saving_slowdown>=1)||!std::isfinite(profile.write_saving_slowdown)||
       !std::isfinite(profile.request_ns)||!std::isfinite(profile.io_bytes_per_ns)||!std::isfinite(profile.memory_bytes_per_ns)||
       !std::isfinite(profile.vector_level_ns)||!std::isfinite(profile.funnel_cell_ns))return ExternalPlanStatus::invalid;
    unsigned np=r.primes,T=r.trunk_bits;
    if(r.cyclic||r.middle){if(!cyclic_basis(r,np,T,out.basis_queries))return ExternalPlanStatus::unsupported;}
    else if(!np||!T){
        sbn3_mul_options o{};o.workers=r.workers;o.algorithm=SBN3_MUL_BAILEY;o.prime_count=np;o.trunk_bits=T?int(T):0;
        sbn3_product_request request{};request.kind=r.square?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;request.a_limbs=r.an;request.b_limbs=r.square?0:r.bn;
        sbn3_mul_plan plan{};sbn3_product_info info{};++out.basis_queries;
        if(sbn3_product_query(&request,&o,&plan,&info)!=SBN3_SUPPORTED||!info.mul.np)return ExternalPlanStatus::unsupported;
        np=info.mul.np;T=info.mul.trunk_bits&~7u; // byte-aligned external codec; strict capacity rechecked below
    }
    const auto *crt=product::p48_crt_backend(np);
    if(!crt||!crt->prepared_bytes(T))return ExternalPlanStatus::unsupported;
    size_t ab,bb;if(!mul_size(r.an,64,ab)||!mul_size(r.bn,64,bb))return ExternalPlanStatus::unsupported;
    const size_t na=ab/T+(ab%T!=0),nb=bb/T+(bb%T!=0);
    if(!crt->capacity(T,std::min(na,nb)))return ExternalPlanStatus::unsupported;
    size_t trunks=1;if(na&&nb){if(!add_size(na,nb,trunks))return ExternalPlanStatus::unsupported;--trunks;}
    if(r.cyclic||r.middle){size_t bits;
        if(!mul_size(minimum_period(r),64,bits))return ExternalPlanStatus::unsupported;
        trunks=bits/T+(bits%T!=0);}
    size_t live=trunks/8+(trunks%8!=0);
    if(!live||live>(uint64_t(1)<<32))return ExternalPlanStatus::unsupported;
    const size_t N=std::bit_ceil(live),shared=p48_root_bank::bytes();size_t fixed;
    if(r.cyclic||r.middle)live=N;
    if(!add_size(budget.other_live_bytes,shared,fixed)||fixed>=budget.resident_bytes)return ExternalPlanStatus::no_admitted_plan;
    if(streaming_recipe(r,budget,profile,np,T,N,live,fixed,out))return ExternalPlanStatus::ready;
    const size_t room=budget.resident_bytes-fixed;
    size_t cache_cells;if(!mul_size(profile.leaf_cells,r.workers,cache_cells))return ExternalPlanStatus::invalid;
    cache_cells=std::bit_ceil(std::min(N,cache_cells));
    const size_t minM=std::max(size_t(1),N/8192);
    const size_t M0=std::max(minM,cache_cells);
    const long double alpha=r.square?64:128;
    const long double balanced=std::cbrt(4.L*live*live*64*np*Q/(alpha*alpha));
    const size_t M1=std::max(M0,std::min(N,std::bit_ceil(std::max(size_t(1),size_t(std::min(balanced,(long double)N))))));
    struct Candidate {SpilledProductPlan plan{};size_t ram=0;uint64_t disk=0;double ns=0;};
    Candidate candidates[4]{};unsigned count=0;SpilledProductSpec seen[4]{};unsigned seen_count=0;
    auto quote=[&](SpilledProductSpec spec,SpilledProductPlan&base)->bool{
        spec.a_source=r.a_source;spec.b_source=r.b_source;spec.cyclic=r.cyclic;spec.middle=r.middle;
        for(unsigned i=0;i<seen_count;++i)if(seen[i].inner_cells==spec.inner_cells&&seen[i].stripe_cells==spec.stripe_cells&&seen[i].prime_batch==spec.prime_batch)return false;
        if(seen_count==4)return false;seen[seen_count++]=spec;++out.summary_quotes;
        if(!external::query_spilled_product(spec,base))return false;
        if(!external::bound_spilled_output(base,r.stored_result_limbs))return false;
        size_t need;if(!add_size(fixed,base.storage_bytes,need))return true;
        out.smallest_quoted_resident=std::min(out.smallest_quoted_resident,need);
        Candidate best{};bool found=false;
        auto placement=[&](SpilledProductPlan candidate){
            size_t ram;if(!add_size(fixed,candidate.storage_bytes,ram))return;
            out.smallest_quoted_resident=std::min(out.smallest_quoted_resident,ram);
            if(ram>budget.resident_bytes)return;
            const size_t held=size_t(std::min(uint64_t((budget.resident_bytes-ram)/Q*Q),uint64_t(np)*candidate.plane_bytes));
            if(!external::retain_spilled_prefix(candidate,held)||candidate.normal_write_bytes>budget.write_limit)return;
            uint64_t disk;if(!disk_peak(candidate,budget.other_owned_disk_bytes,disk)||disk>budget.owned_disk_limit)return;
            const double ns=price(candidate,profile);if(!std::isfinite(ns))return;
            if(!found||ns<best.ns||(ns<=best.ns*profile.write_saving_slowdown&&candidate.normal_write_bytes<best.plan.normal_write_bytes)){
                best={candidate,fixed+candidate.storage_bytes,disk,ns};found=true;
            }
        };
        placement(base);
        // One representation replay on the same arithmetic geometry. Compact
        // the smaller source; retained B bytes follow directly from the grant.
        auto raw=base;if(!r.square&&external::use_raw_operand(raw,r.an<=r.bn?0:1))placement(raw);
        if(found)candidates[count++]=best;
        return true;
    };
    const size_t geometries[]{M0,M1};
    for(unsigned k=0;k<2;++k){const size_t M=geometries[k];if(k&&M==M0)continue;
        const size_t C=N/M,rows=live/M+(live%M!=0),minJ=minimum_stripe(T,M);
        const long double A=64.L*np*rows+r.workers*T,B=(long double)rows*M*(Q+sizeof(external::CarrySegment));
        const long double optimum=std::sqrt(B/A);
        size_t J=std::max(minJ,floor_power(optimum,M));
        if(J<M){const size_t next=J*2;if(A*next+B/next<A*J+B/J)J=next;}
        SpilledProductSpec spec{r.an,r.bn,C,M,J,np,T,1,r.workers,r.square};SpilledProductPlan base{};
        if(!quote(spec,base)||!base.storage_bytes)continue;
        if(M==minJ)continue;
        const auto h=bounds(base);
        const long double available=(long double)room-base.pool;
        const long double allowance=available-h.fixed_inverse-h.worker_slope;
        const long double discriminant=allowance*allowance-4*h.A*h.B;
        if(allowance<=0||discriminant<0)continue;
        const long double upper=(allowance+std::sqrt(discriminant))/(2*h.A);
        const long double forward=(available-h.fixed_forward-h.worker_slope)/(C*(long double)T+48.L*rows);
        const size_t target=std::bit_ceil(std::min(M,std::max(size_t(1),profile.target_transfer_bytes/48+(profile.target_transfer_bytes%48!=0))));
        J=std::min(target,floor_power(std::min(upper,forward),M));if(J<minJ)continue;
        const long double maxF=(available-h.fixed_forward-C*(long double)T*J)/(h.worker_slope+48.L*rows*J);
        const long double maxI=(available-h.fixed_inverse-h.A*J-h.B/J)/h.worker_slope;
        const unsigned maxg=unsigned(std::max(1.L,std::min((long double)np,std::floor(std::min(maxF,maxI)))));
        const long double replay=double(r.an+(r.square?0:r.bn))*8/T*profile.funnel_cell_ns/r.workers;
        const long double mu=4.L/profile.io_bytes_per_ns,V=h.worker_slope+48.L*rows*J;
        const long double estimate=std::clamp(std::sqrt(std::max(0.L,replay*np/(mu*V))),1.L,(long double)maxg);
        const unsigned waves=std::max(1u,unsigned(std::llround(np/estimate)));
        const unsigned g=std::min(maxg,(np+waves-1)/waves);
        spec.stripe_cells=J;spec.prime_batch=g;SpilledProductPlan alternate{};quote(spec,alternate);
    }
    if(!count)return ExternalPlanStatus::no_admitted_plan;
    unsigned fastest=0;for(unsigned i=1;i<count;++i)if(candidates[i].ns<candidates[fastest].ns)fastest=i;
    unsigned chosen=fastest;
    for(unsigned i=0;i<count;++i)if(candidates[i].ns<=candidates[fastest].ns*profile.write_saving_slowdown &&
       (candidates[i].plan.normal_write_bytes<candidates[chosen].plan.normal_write_bytes ||
        (candidates[i].plan.normal_write_bytes==candidates[chosen].plan.normal_write_bytes&&candidates[i].ns<candidates[chosen].ns)))chosen=i;
    out.plan=candidates[chosen].plan;out.total_resident_bytes=candidates[chosen].ram;
    out.owned_disk_bytes=candidates[chosen].disk;out.estimated_ns=candidates[chosen].ns;
    // Reserve a streaming middle pass inside the already admitted grant before
    // assigning the remaining bytes to plane retention. One storage replay on
    // the chosen basis; no new geometry search and no relaxed write limit.
    if(!out.plan.middle.pipeline){auto streaming=out.plan;
        if(external::compact_middle_within(streaming,out.plan.storage_bytes)&&streaming.normal_write_bytes<=budget.write_limit){
            out.total_resident_bytes=out.total_resident_bytes-out.plan.storage_bytes+streaming.storage_bytes;
            out.plan=streaming;out.estimated_ns=price(streaming,profile);
        }
    }
    out.admitted=true;
    return ExternalPlanStatus::ready;
}
external::IoError run_selected_product(const ExternalProductSelection&s,external::Transport&t,const external::Region&a,
    const external::Region&b,external::Region&r,std::span<external::Region>temporary,std::span<uint8_t>workspace,sbn3_team&team,external::CyclicResidualResult*cyclic,const external::Region*subtract) noexcept {
    if(!s.admitted||!s.summary_quotes||t.counters.write_submitted>t.write_limit||s.plan.normal_write_bytes>t.write_limit-t.counters.write_submitted)
        return {external::IoCode::capacity,"selected product write admission",0,s.plan.normal_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    return external::run_spilled_product(s.plan,t,a,b,r,temporary,workspace,team,nullptr,cyclic,subtract);
}
} // namespace sbn::v3::planning
