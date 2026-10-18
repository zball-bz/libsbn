// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/inverse_planes.hpp"
#include "external/two_slot.hpp"
#include "common/checked.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cstring>
namespace sbn::v3::external {
bool query_inverse_planes(const product::OuterTransformSpec &s,InversePlanesPlan &out,bool fused) noexcept {
    InversePlanesPlan p{};p.spec=s;p.spec.input_cells=0;
    if(!product::p48_outer_query(p.spec,p.info))return false;
    if(fused&&!product::p48_outer_emit_query(p.spec,p.fused))return false;
    size_t raw,cells,cursor=0;
    if(!mul_size(s.inner_cells,48,raw)||!align_size(raw,chunk_bytes,p.row_stride)||
       __builtin_mul_overflow(p.row_stride,uint64_t(s.live_rows),&p.plane_bytes)||p.plane_bytes>INT64_MAX||
       !mul_size(s.prime_count,s.live_rows,cells)||!mul_size(cells,s.stripe_cells,cells)||!mul_size(cells,fused?48:64,p.coefficient_bytes)||
       (fused&&!add_size(p.coefficient_bytes,64,p.coefficient_bytes)))return false;
    auto put=[&](size_t bytes,size_t&offset,size_t alignment=64){return align_size(cursor,alignment,offset)&&add_size(offset,bytes,cursor);};
    if(!put(p.info.prepared_bytes,p.prepared)||!put(std::max(p.info.workspace_bytes,p.fused.workspace_bytes),p.work)||!put(p.coefficient_bytes,p.coefficients,4096)||
       !align_size(cursor,64,p.storage_bytes))return false;
    p.pipeline=!fused&&s.workers>1&&s.prime_batch<s.prime_count;
    out=p;return true;
}
void prepare_inverse_planes(const InversePlanesPlan&p,std::span<uint8_t>w,sbn3_team&t,InversePlanesBinding&b) noexcept {
    require(w.data()&&!(uintptr_t(w.data())&63)&&w.size()>=p.storage_bytes&&!t.busy&&t.width>=(p.pipeline?p.spec.workers:p.info.workers),
            SBN3_FATAL_WORKSPACE,"inverse stripe preparation");
    auto spec=p.spec;auto info=p.info;
    if(p.pipeline){--spec.workers;info.workers=std::min(info.workers,spec.workers);info.workspace_bytes=info.worker_bytes*info.workers;}
    product::p48_outer_prepare(spec,info,w.subspan(p.prepared,p.info.prepared_bytes));
    if(p.fused.workspace_bytes)std::memset(w.data()+p.coefficients+p.coefficient_bytes-64,0,64);
    b={&p,w,&t};
}
namespace {
struct Call {InversePlanesBinding*b;unsigned first,count;};
void transform(void*ptr,sbn3_team_scope*s){auto &c=*static_cast<Call *>(ptr);const auto&p=*c.b->plan;auto w=c.b->workspace;
    const size_t prime_bytes=p.spec.live_rows*p.spec.stripe_cells*64;
    product::p48_outer_inverse(w.subspan(p.prepared,p.info.prepared_bytes),c.first,c.count,
        w.subspan(p.coefficients+c.first*prime_bytes,c.count*prime_bytes),w.subspan(p.work,p.info.workspace_bytes),s);}
struct InverseRead {
    InversePlanesBinding*b;
    uint64_t column;
    std::span<ScratchPlane>planes;
    std::span<const PlaneView>views;
    TwoSlotExchange exchange{};
    IoError error{};
};
IoError read_batch(InverseRead&c,unsigned first,unsigned count){
    const auto&p=*c.b->plan;const size_t row_bytes=p.spec.stripe_cells*(p.fused.workspace_bytes?48:64),prime_bytes=p.spec.live_rows*row_bytes;
    if(c.planes[first].transport()->read_batch&&p.spec.stripe_cells*48%chunk_bytes==0){
        ReadRange ranges[io_batch_max];
        for(unsigned q=0;q<count;++q)for(size_t r=0;r<p.spec.live_rows;){size_t n=0;
            for(;n<io_batch_max&&r<p.spec.live_rows;++n,++r)ranges[n]={r*p.row_stride+c.column*48,
                c.b->workspace.subspan(p.coefficients+(first+q)*prime_bytes+r*row_bytes,p.spec.stripe_cells*48)};
            if(auto e=c.planes[first+q].read_many(c.views[first+q],{ranges,n}))return e;
        }
        return {};
    }
    for(unsigned q=0;q<count;++q)for(size_t r=0;r<p.spec.live_rows;++r){
        const uint64_t at=r*p.row_stride+c.column*48;
        auto target=c.b->workspace.subspan(p.coefficients+(first+q)*prime_bytes+r*row_bytes,p.spec.stripe_cells*48);
        if(auto e=c.planes[first+q].read(c.views[first+q],at,target))return e;
    }
    return {};
}
void inverse_io(void*ptr,sbn3_team_scope*){
    auto&c=*static_cast<InverseRead *>(ptr);const auto&p=*c.b->plan;
    unsigned job=0;
    for(unsigned first=0;first<p.spec.prime_count;first+=p.spec.prime_batch,++job){
        const unsigned slot=job%2,count=std::min(p.spec.prime_batch,p.spec.prime_count-first);
        if(job>=2&&!c.exchange.wait(slot,TwoSlotExchange::computed))return;
        if(auto e=read_batch(c,first,count)){c.error=e;c.exchange.stop();return;}
        c.exchange.publish(slot,TwoSlotExchange::ready);
    }
}
void inverse_compute(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<InverseRead *>(ptr);const auto&p=*c.b->plan;
    unsigned job=0;
    for(unsigned first=0;first<p.spec.prime_count;first+=p.spec.prime_batch,++job){
        const unsigned slot=job%2,count=std::min(p.spec.prime_batch,p.spec.prime_count-first);
        if(!c.exchange.wait(slot,TwoSlotExchange::ready))return;
        Call call{c.b,first,count};transform(&call,scope);
        if(c.exchange.stopped())return;
        c.exchange.publish(slot,TwoSlotExchange::computed);
    }
}
void inverse_pipeline(void*ptr,sbn3_team_scope*scope){
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,1,inverse_io,ptr,inverse_compute,ptr);
}
}
IoError read_inverse_stripe(InversePlanesBinding&b,uint64_t column,std::span<ScratchPlane>planes,CoefficientStripe&out) noexcept {
    out={};
    if(!b.plan||!b.team||b.team->busy)return {IoCode::invalid,"inverse stripe binding"};
    const auto&p=*b.plan;
    if(p.fused.workspace_bytes)return {IoCode::invalid,"fused inverse requires fused emission"};
    if(planes.size()!=p.spec.prime_count||column%p.spec.stripe_cells||column>p.spec.inner_cells-p.spec.stripe_cells)
        return {IoCode::invalid,"inverse stripe range",column};
    PlaneView views[10];
    for(size_t q=0;q<planes.size();++q){
        if(planes[q].logical_bytes()<p.plane_bytes)return {IoCode::capacity,"inverse stripe source",q,p.plane_bytes,planes[q].logical_bytes()};
        if(auto e=planes[q].view(views[q]))return e;
    }
    InverseRead read{&b,column,planes,{views,planes.size()}};
    if(p.pipeline){
        sbn3_team_run(b.team,inverse_pipeline,&read);if(read.error)return read.error;
    }else for(unsigned first=0;first<p.spec.prime_count;first+=p.spec.prime_batch){
        const unsigned count=std::min(p.spec.prime_batch,p.spec.prime_count-first);
        if(auto e=read_batch(read,first,count))return e;
        Call call{&b,first,count};sbn3_team_run(b.team,transform,&call);
    }
    out={{reinterpret_cast<const uint64_t *>(b.workspace.data()+p.coefficients),p.coefficient_bytes/8},
         p.spec.prime_count,p.spec.live_rows,p.spec.stripe_cells,column};return {};
}
IoError load_inverse_stripe(InversePlanesBinding&b,uint64_t column,std::span<ScratchPlane>planes,sbn3_team_scope*scope) noexcept {
    if(!b.plan||!b.team)return {IoCode::invalid,"inverse load binding"};const auto&p=*b.plan;
    if(planes.size()!=p.spec.prime_count||column%p.spec.stripe_cells||column>p.spec.inner_cells-p.spec.stripe_cells)
        return {IoCode::invalid,"inverse load range",column};
    PlaneView views[10];for(size_t q=0;q<planes.size();++q){
        if(planes[q].logical_bytes()<p.plane_bytes)return {IoCode::capacity,"inverse load source",q,p.plane_bytes,planes[q].logical_bytes()};
        if(auto e=planes[q].view(views[q]))return e;}
    InverseRead read{&b,column,planes,{views,planes.size()}};
    if(scope&&scope->width>1&&!planes[0].transport()->read_batch){
        require_scope_leader(scope);
        require(p.spec.stripe_cells*48%chunk_bytes==0&&planes[0].transport()->parallel_reads>=scope->width,
                SBN3_FATAL_ARGUMENT,"parallel inverse input chunks");
        struct Reads {InverseRead*call;IoError errors[10]{};} request{&read,{}};
        auto fn=[](void*ptr,uint64_t lo,uint64_t hi,unsigned){auto&c=*static_cast<Reads*>(ptr);
            for(auto q=lo;q<hi;++q)c.errors[q]=read_batch(*c.call,unsigned(q),1);};
        sbn3_team_for(scope,0,p.spec.prime_count,1,SBN3_STATIC,fn,&request);
        for(unsigned q=0;q<p.spec.prime_count;++q)if(request.errors[q])return request.errors[q];
        return {};
    }
    for(unsigned first=0;first<p.spec.prime_count;first+=p.spec.prime_batch)
        if(auto e=read_batch(read,first,std::min(p.spec.prime_batch,p.spec.prime_count-first)))return e;
    return {};
}
void transform_inverse_stripe(InversePlanesBinding&b,uint64_t column,sbn3_team_scope*scope,CoefficientStripe&out) noexcept {
    const auto&p=*b.plan;require_scope_leader(scope);
    require(scope->team==b.team&&scope->width>=p.info.workers&&!p.pipeline&&!p.fused.workspace_bytes&&column%p.spec.stripe_cells==0&&
        column<=p.spec.inner_cells-p.spec.stripe_cells,SBN3_FATAL_WORKSPACE,"inverse numeric stage");
    for(unsigned first=0;first<p.spec.prime_count;first+=p.spec.prime_batch){
        Call call{&b,first,std::min(p.spec.prime_batch,p.spec.prime_count-first)};transform(&call,scope);}
    out={{reinterpret_cast<const uint64_t*>(b.workspace.data()+p.coefficients),p.coefficient_bytes/8},
         p.spec.prime_count,p.spec.live_rows,p.spec.stripe_cells,column};
}
} // namespace sbn::v3::external
