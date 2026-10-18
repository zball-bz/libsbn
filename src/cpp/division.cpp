// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/division.hpp"
#include "sbn3/divrem.h"
#include "algorithms/divrem_prepared.hpp"
#include "runtime/resident.hpp"
#include "value/limbs.hpp"
#include "value/bitwise.hpp"
#include <new>

namespace sbn {
namespace {
using sbn::v3::overlaps;
using sbn::v3::ResidentSession;
using Round=division_rounding;
constexpr std::size_t max_words=(std::size_t(1)<<31)-16;
std::size_t aligned(std::size_t n,std::size_t a)noexcept {std::size_t r;detail::require(v3::align_size(n,a,r),SBN3_FATAL_SIZE,"division layout");return r;}
bool valid_mode(Round m)noexcept{return m>=Round::trunc&&m<=Round::euclidean;}
std::size_t qbound(std::size_t nn,std::size_t dn,Round mode)noexcept {
    if(!nn)return 0;
    if(dn==1)return nn;
    return (nn>=dn?nn-dn+1:0)+std::size_t(mode!=Round::trunc);
}
bool alias(std::optional<integer_ref> r,integer_view a)noexcept {
    return r&&overlaps(r->view().data(),r->capacity()*8,a.data(),a.size()*8);
}
struct Layout {
    division_requirements req{};bool tiny=false,shorter=false,local=false,compact=false,stageq=false,stager=false;
    std::size_t qat=0,rat=0,dat=0,workat=0,sessionat=0,workbytes=0,copyd=0,auxat=0,auxbytes=0;
};
// Keep the large service-plan stack out of direct word and local calls.
[[gnu::noinline]] sbn3_query_result query_service(std::size_t nn,std::size_t dn,unsigned workers,
                                                sbn3_divrem_plan*plan,sbn3_divrem_info&info,bool*compact=nullptr,std::size_t*auxiliary=nullptr)noexcept {
    const sbn3_divrem_request r{nn,dn};sbn3_divrem_options o{};o.workers=workers;
    if(plan){const auto st=sbn3_divrem_query(&r,&o,plan,&info);if(compact&&st==SBN3_SUPPORTED)*compact=plan->opaque[0]==v3::divrem_prepared::magic;if(auxiliary&&st==SBN3_SUPPORTED)*auxiliary=sbn3_remainder_plan_workspace_bytes(plan);return st;}
    sbn3_divrem_plan tmp;const auto st=sbn3_divrem_query(&r,&o,&tmp,&info);if(compact&&st==SBN3_SUPPORTED)*compact=tmp.opaque[0]==v3::divrem_prepared::magic;if(auxiliary&&st==SBN3_SUPPORTED)*auxiliary=sbn3_remainder_plan_workspace_bytes(&tmp);return st;
}
Layout layout(std::optional<integer_ref> q,std::optional<integer_ref> r,integer_view n,integer_view d,Round mode,
              unsigned workers,sbn3_divrem_plan*plan=nullptr)noexcept {
    Layout p;
    auto reject=[&](support s,const char*why){p.req.status=s;p.req.reason=why;return p;};
    if(!d.size())return reject(support::unsupported,"zero divisor");
    if(!valid_mode(mode)||!workers||workers>32||n.size()>max_words||d.size()>max_words)
        return reject(support::unsupported,"division shape/rounding/workers");
    p.req.quotient_limbs=qbound(n.size(),d.size(),mode);p.req.remainder_limbs=std::min(n.size(),d.size());
    // Rounded R can be D-|N| even for a short numerator.
    if(n.size()&&mode!=Round::trunc)p.req.remainder_limbs=d.size();
    p.req.output_limbs=p.req.quotient_limbs+p.req.remainder_limbs;
    if(q&&q->capacity()<p.req.quotient_limbs)return reject(support::capacity,"quotient capacity");
    if(r&&r->capacity()<p.req.remainder_limbs)return reject(support::capacity,"remainder capacity");
    if(q&&r&&(q->same_object(*r)||overlaps(q->view().data(),q->capacity()*8,r->view().data(),r->capacity()*8)))
        return reject(support::unsupported,"overlapping quotient/remainder");
    p.tiny=n.size()<=2&&d.size()<=2;
    p.shorter=n.size()<d.size();
    if(p.tiny)return p;
    // Short numerator needs no division kernel; only rounded subtraction and
    // deferred publication when an output aliases a still-live input.
    p.stageq=q&&(alias(q,n)||alias(q,d));
    p.stager=!r||alias(r,n)||alias(r,d);
    std::size_t at=0;
    if(p.stageq){p.qat=at;at+=8*std::max(std::size_t(1),p.req.quotient_limbs);}
    if(p.stager){p.rat=at;at+=8*std::max(std::size_t(1),p.req.remainder_limbs);}
    if(!p.shorter&&overlaps(n.data(),n.size()*8,d.data(),d.size()*8)){p.copyd=d.size();p.dat=at;at+=p.copyd*8;}
    if(p.shorter){p.req.scratch_bytes=at;return p;}
    // Direct one-shot local service is already optimized through 32768 limbs.
    // It does not construct an Arena/Team or a large opaque plan.
    p.local=workers==1&&d.size()<=32768;
    if(p.local){p.workbytes=q?sbn3_divrem_small_scratch_bytes(n.size(),d.size()):sbn3_remainder_small_scratch_bytes(n.size(),d.size());p.workat=aligned(at,128);}
    else {
        sbn3_divrem_info info{};const auto status=query_service(n.size(),d.size(),workers,plan,info,&p.compact,q?nullptr:&p.auxbytes);
        if(status!=SBN3_SUPPORTED)return reject(status==SBN3_QUERY_CAPACITY?support::capacity:support::unsupported,"division backend support");
        p.req.scratch_alignment=std::max(std::size_t(128),info.storage_alignment);p.workbytes=info.storage_bytes;
        if(p.auxbytes){p.auxat=aligned(at,64);at=p.auxat+p.auxbytes;}
        // General query can choose a compact scalar/DC plan for short quotients.
        if(!p.compact){p.sessionat=aligned(at,alignof(ResidentSession));at=p.sessionat+sizeof(ResidentSession);}
        p.workat=aligned(at,p.req.scratch_alignment);
    }
    p.req.scratch_bytes=p.workat+p.workbytes;return p;
}
template<class Fn>[[gnu::noinline]] void with_service_layout(std::optional<integer_ref>q,std::optional<integer_ref>r,
    integer_view n,integer_view d,Round mode,unsigned workers,Fn&&fn)noexcept{
    sbn3_divrem_plan plan;const auto p=layout(q,r,n,d,mode,workers,&plan);fn(p,&plan);
}
template<class Fn>void with_layout(std::optional<integer_ref>q,std::optional<integer_ref>r,
    integer_view n,integer_view d,Round mode,unsigned workers,Fn&&fn)noexcept{
    if(n.size()>=d.size()&&n.size()>2&&(workers!=1||d.size()>32768))
        with_service_layout(q,r,n,d,mode,workers,std::forward<Fn>(fn));
    else{const auto p=layout(q,r,n,d,mode,workers);fn(p,nullptr);}
}
void publish(std::optional<integer_ref> out,const std::uint64_t*data,std::size_t size,bool negative)noexcept {
    if(!out)return;
    auto*d=out->begin_write();if(size&&d->data!=data)std::memmove(d->data,data,size*8);
    d->size=size;d->negative=size&&negative;out->finish_write();
}
bool increment_quotient(Round mode,bool ns,bool ds,bool nonzero)noexcept {
    if(!nonzero)return false;
    return mode==Round::floor?(ns!=ds):mode==Round::ceil?(ns==ds):mode==Round::euclidean?ns:false;
}
void tiny(std::optional<integer_ref>q,std::optional<integer_ref>r,integer_view n,integer_view d,Round mode)noexcept {
    auto read=[](integer_view a){return (a.size()?__uint128_t(a.data()[0]):0)|(a.size()>1?__uint128_t(a.data()[1])<<64:0);};
    const auto N=read(n),D=read(d);auto Q=N/D,R=N%D;
    const bool bump=increment_quotient(mode,n.negative(),d.negative(),R!=0);
    bool rn=n.negative();if(bump){++Q;R=D-R;rn=!rn;}
    const std::uint64_t qw[2]{std::uint64_t(Q),std::uint64_t(Q>>64)},rw[2]{std::uint64_t(R),std::uint64_t(R>>64)};
    publish(q,qw,qw[1]?2:bool(qw[0]),n.negative()!=d.negative());
    publish(r,rw,rw[1]?2:bool(rw[0]),rn);
}
[[gnu::noinline]] void general_divide(const Layout&p,integer_view n,integer_view d,std::uint64_t*q,std::uint64_t*r,
                                     execution ex,sbn3_divrem_result&result,const sbn3_divrem_plan&plan)noexcept {
    auto*work=ex.scratch.data()+p.workat;sbn3_divrem_binding *b;
    ResidentSession*session=nullptr;
    if(plan.opaque[0]==v3::divrem_prepared::magic)b=v3::divrem_prepared::bind_raw(plan,work,p.workbytes);
    else {
        detail::require(!p.compact,SBN3_FATAL_MATH,"division control quote");
        session=::new(ex.scratch.data()+p.sessionat)ResidentSession({work,p.workbytes},ex.team);
        sbn3_divrem_bind(&plan,session->arena(),0,session->team(),&b);
    }
    sbn3_divrem_prepare(b,{d.data(),d.size()});
    if(q)sbn3_divrem_execute(b,{n.data(),n.size()},{q,n.size()-d.size()+1},{r,d.size()},&result);
    else sbn3_remainder_execute(b,{n.data(),n.size()},{r,d.size()},p.auxbytes?ex.scratch.data()+p.auxat:nullptr,p.auxbytes,&result);
    sbn3_divrem_unbind(b);if(session)session->~ResidentSession();
}
void run(const Layout&p,std::optional<integer_ref>qout,std::optional<integer_ref>rout,integer_view n,integer_view d,
         execution ex,Round mode,const sbn3_divrem_plan*plan)noexcept {
    detail::require(p.req.supported(),SBN3_FATAL_ARGUMENT,p.req.reason);
    if(p.tiny){tiny(qout,rout,n,d,mode);return;}
    detail::require(ex.scratch.size()>=p.req.scratch_bytes&&
        (!p.req.scratch_bytes||!(std::uintptr_t(ex.scratch.data())&(p.req.scratch_alignment-1))),
        SBN3_FATAL_WORKSPACE,"division scratch",p.req.scratch_bytes,ex.scratch.size());
    if(p.req.scratch_bytes){
        detail::require(!overlaps(ex.scratch.data(),p.req.scratch_bytes,n.data(),n.size()*8)&&
                        !overlaps(ex.scratch.data(),p.req.scratch_bytes,d.data(),d.size()*8),SBN3_FATAL_ARGUMENT,"division input/scratch alias");
        for(auto o:{qout,rout})if(o)detail::require(!overlaps(ex.scratch.data(),p.req.scratch_bytes,o->view().data(),o->capacity()*8),
                                                  SBN3_FATAL_ARGUMENT,"division output/scratch alias");
    }
    auto *q=!qout?nullptr:p.stageq?reinterpret_cast<std::uint64_t*>(ex.scratch.data()+p.qat):qout->begin_write()->data;
    auto *r=p.stager?reinterpret_cast<std::uint64_t*>(ex.scratch.data()+p.rat):rout->begin_write()->data;
    if(p.copyd){auto*D=reinterpret_cast<std::uint64_t*>(ex.scratch.data()+p.dat);std::memcpy(D,d.data(),p.copyd*8);d=integer_view{{D,p.copyd},d.negative()};}
    std::size_t qn=0,rn=0;
    if(p.shorter){if(n.size())std::memmove(r,n.data(),n.size()*8);rn=n.size();}
    else if(p.local){
        if(q)qn=sbn3_divrem_small(q,r,n.data(),n.size(),d.data(),d.size(),p.workbytes?ex.scratch.data()+p.workat:nullptr);
        else sbn3_remainder_small(r,n.data(),n.size(),d.data(),d.size(),p.workbytes?ex.scratch.data()+p.workat:nullptr);
        rn=v3::bits::normalized_size(r,d.size());
    }else{
        detail::require(!ex.scope,SBN3_FATAL_TEAM,"general division needs idle root Team");
        detail::require(plan,SBN3_FATAL_MATH,"missing division program");
        sbn3_divrem_result result{};general_divide(p,n,d,q,r,ex,result,*plan);qn=result.quotient_limbs;rn=result.remainder_limbs;
    }
    bool rnegative=n.negative();
    if(increment_quotient(mode,n.negative(),d.negative(),rn!=0)){
        if(q){std::size_t i=0;for(;i<qn&&q[i]==UINT64_MAX;++i)q[i]=0;
        if(i==qn)q[qn++]=1;else ++q[i];}
        // D - R, low-to-high, safe in place. Do not read unwritten R padding.
        std::uint64_t borrow=0;
        for(std::size_t j=0;j<d.size();++j){const auto a=d.data()[j],b=j<rn?r[j]:0;const auto t=a-b;const auto v=t-borrow;borrow=(a<b)|(t<borrow);r[j]=v;}
        detail::require(!borrow,SBN3_FATAL_MATH,"rounded division remainder");rn=v3::bits::normalized_size(r,d.size());rnegative=!rnegative;
    }
    publish(qout,q,qn,n.negative()!=d.negative());publish(rout,r,rn,rnegative);
}
integer selected(const integer&n,const integer&d,Round mode,bool quotient)noexcept {
    auto result=integer::with_resource_of(n);
    detail::require(!d.zero(),SBN3_FATAL_ARGUMENT,"zero divisor");
    result.reserve(quotient?qbound(n.size(),d.size(),mode):std::max(std::min(n.size(),d.size()),mode==Round::trunc?std::size_t(0):d.size()));
    std::optional<integer_ref> q=quotient?std::optional{result.ref()}:std::nullopt,r=quotient?std::nullopt:std::optional{result.ref()};
    with_layout(q,r,n.view(),d.view(),mode,1,[&](const Layout&p,const sbn3_divrem_plan*plan){
        detail::require(p.req.supported(),SBN3_FATAL_ARGUMENT,p.req.reason);workspace work(n.resource());work.reserve(p.req.scratch_bytes,p.req.scratch_alignment);
        run(p,q,r,n.view(),d.view(),work.on(),mode,plan);
    });return result;
}
}
division_requirements divrem_requirements(std::optional<integer_ref>q,std::optional<integer_ref>r,integer_view n,integer_view d,Round mode,unsigned workers)noexcept {
    return layout(q,r,n,d,mode,workers).req;
}
void detail::general_divrem_into(std::optional<integer_ref>q,std::optional<integer_ref>r,integer_view n,integer_view d,execution ex,Round mode)noexcept {
    with_layout(q,r,n,d,mode,ex.workers(),[&](const Layout&p,const sbn3_divrem_plan*plan){run(p,q,r,n,d,ex,mode,plan);});
}
division_result divrem(const integer&n,const integer&d,Round mode)noexcept {
    detail::require(!d.zero(),SBN3_FATAL_ARGUMENT,"zero divisor");
    division_result out{integer::with_resource_of(n),integer::with_resource_of(n)};
    out.quotient.reserve(qbound(n.size(),d.size(),mode));out.remainder.reserve(mode==Round::trunc?std::min(n.size(),d.size()):d.size());
    with_layout(out.quotient.ref(),out.remainder.ref(),n.view(),d.view(),mode,1,[&](const Layout&p,const sbn3_divrem_plan*plan){
        detail::require(p.req.supported(),SBN3_FATAL_ARGUMENT,p.req.reason);workspace work(n.resource());work.reserve(p.req.scratch_bytes,p.req.scratch_alignment);
        run(p,out.quotient.ref(),out.remainder.ref(),n.view(),d.view(),work.on(),mode,plan);
    });return out;
}
integer divide(const integer&n,const integer&d,Round mode)noexcept{return selected(n,d,mode,true);}
integer remainder(const integer&n,const integer&d,Round mode)noexcept{return selected(n,d,mode,false);}
integer operator/(const integer&n,const integer&d)noexcept{return divide(n,d);}
integer operator%(const integer&n,const integer&d)noexcept{return remainder(n,d);}
integer&integer::operator/=(const integer&d)noexcept{*this=divide(*this,d);return *this;}
integer&integer::operator%=(const integer&d)noexcept{*this=remainder(*this,d);return *this;}
integer exact_divide(const integer&n,const integer&d)noexcept{auto r=divrem(n,d);detail::require(r.remainder.zero(),SBN3_FATAL_MATH,"nonexact integer division");return std::move(r.quotient);}
std::optional<integer> try_exact_divide(const integer&n,const integer&d)noexcept{auto r=divrem(n,d);if(!r.remainder.zero())return std::nullopt;return std::move(r.quotient);}
bool is_divisible(const integer&n,const integer&d)noexcept{return d.zero()?n.zero():remainder(n,d).zero();}
bool is_congruent(const integer&a,const integer&b,const integer&m)noexcept{return m.zero()?a==b:is_divisible(a-b,m);}
}
