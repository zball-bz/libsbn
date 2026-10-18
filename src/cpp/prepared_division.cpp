// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/prepared_divisor.hpp"
#include "algorithms/divrem_prepared.hpp"
#include "algorithms/divisor_view.hpp"
#include "runtime/resident.hpp"
#include "value/bitwise.hpp"
#include <new>
namespace sbn {
namespace {
using Maybe=std::optional<integer_ref>;using Round=division_rounding;
using v3::overlaps;using v3::ResidentSession;
struct State {
    sbn3_divrem_binding*binding=nullptr;ResidentSession*session=nullptr;sbn3_team*team=nullptr;
    std::size_t nn=0,dn=0,bytes=0,remainder_bytes=0;bool negative=false;pthread_t owner{};
};
struct Storage {requirements need;std::size_t sessionat=0,workat=0,workbytes=0;bool compact=false;};
std::size_t align(std::size_t n,std::size_t a)noexcept{std::size_t x;detail::require(v3::align_size(n,a,x),SBN3_FATAL_SIZE,"divisor layout");return x;}
Storage plan_storage(std::size_t dn,std::size_t nn,unsigned workers,unsigned reuse,sbn3_divrem_plan&plan)noexcept{
    Storage s;sbn3_divrem_info info{};sbn3_divrem_request r{nn,dn};sbn3_divrem_options o{};o.workers=workers;o.reuse_hint=reuse;
    const auto st=sbn3_divrem_query(&r,&o,&plan,&info);s.need.status=st==SBN3_SUPPORTED?support::yes:st==SBN3_QUERY_CAPACITY?support::capacity:support::unsupported;
    if(!s.need.supported()){s.need.reason="prepared divisor shape/workers";return s;}
    s.compact=plan.opaque[0]==v3::divrem_prepared::magic;s.need.scratch_alignment=std::max(std::size_t(128),info.storage_alignment);
    s.sessionat=align(sizeof(State),alignof(ResidentSession));
    s.workat=align(s.compact?sizeof(State):s.sessionat+sizeof(ResidentSession),s.need.scratch_alignment);
    s.workbytes=info.storage_bytes;s.need.scratch_bytes=s.workat+s.workbytes;return s;
}
State*construct(const Storage&s,const sbn3_divrem_plan&plan,integer_view d,std::size_t nn,std::span<std::byte>data,sbn3_team*team)noexcept{
    detail::require(s.need.supported(),SBN3_FATAL_ARGUMENT,s.need.reason);
    detail::require(data.size()>=s.need.scratch_bytes&&!(uintptr_t(data.data())&(s.need.scratch_alignment-1)),
                    SBN3_FATAL_WORKSPACE,"prepared divisor storage",s.need.scratch_bytes,data.size());
    detail::require(!overlaps(data.data(),s.need.scratch_bytes,d.data(),d.size()*8),SBN3_FATAL_ARGUMENT,"prepare divisor/storage alias");
    auto*p=::new(data.data())State;p->nn=nn;p->dn=d.size();p->bytes=s.need.scratch_bytes;p->negative=d.negative();p->team=team;p->owner=pthread_self();
    auto*work=data.data()+s.workat;
    if(s.compact)p->binding=v3::divrem_prepared::bind_raw(plan,work,s.workbytes);
    else{p->session=::new(data.data()+s.sessionat)ResidentSession({work,s.workbytes},team);sbn3_divrem_bind(&plan,p->session->arena(),0,p->session->team(),&p->binding);}
    sbn3_divrem_prepare(p->binding,{d.data(),d.size()});p->remainder_bytes=sbn3_remainder_workspace_bytes(p->binding);return p;
}
std::size_t quotient_capacity(std::size_t nn,std::size_t dn,Round m)noexcept{
    if(!nn)return 0;if(dn==1)return nn;return (nn>=dn?nn-dn+1:0)+unsigned(m!=Round::trunc);
}
struct Outputs {division_requirements need;bool qstage=false,rstage=false;std::size_t qat=0,rat=0,auxat=0,auxbytes=0;};
Outputs quote(const State&s,Maybe q,Maybe r,integer_view n,Round mode)noexcept{
    Outputs p;auto reject=[&](support st,const char*why){p.need.status=st;p.need.reason=why;return p;};
    if(n.size()>s.nn||mode<Round::trunc||mode>Round::euclidean)return reject(support::unsupported,"prepared numerator bound/rounding");
    p.need.quotient_limbs=quotient_capacity(n.size(),s.dn,mode);p.need.remainder_limbs=mode==Round::trunc?std::min(n.size(),s.dn):(n.size()?s.dn:0);
    p.need.output_limbs=p.need.quotient_limbs+p.need.remainder_limbs;
    if(q&&q->capacity()<p.need.quotient_limbs)return reject(support::capacity,"prepared quotient capacity");
    if(r&&r->capacity()<p.need.remainder_limbs)return reject(support::capacity,"prepared remainder capacity");
    if(q&&r&&(q->same_object(*r)||overlaps(q->view().data(),q->capacity()*8,r->view().data(),r->capacity()*8)))return reject(support::unsupported,"prepared Q/R overlap");
    if(overlaps(n.data(),n.size()*8,&s,s.bytes))return reject(support::unsupported,"prepared numerator/storage overlap");
    for(auto out:{q,r})if(out&&overlaps(out->view().data(),out->capacity()*8,&s,s.bytes))return reject(support::unsupported,"prepared output/storage overlap");
    p.qstage=q&&overlaps(q->view().data(),q->capacity()*8,n.data(),n.size()*8);
    p.rstage=!r||overlaps(r->view().data(),r->capacity()*8,n.data(),n.size()*8);
    std::size_t at=0;
    if(p.qstage){p.qat=at;at+=8*p.need.quotient_limbs;}
    if(p.rstage){p.rat=at;at+=8*p.need.remainder_limbs;}
    p.need.scratch_alignment=8;
    if(!q&&n.size()>=s.dn&&s.remainder_bytes){p.auxbytes=s.remainder_bytes;p.auxat=align(at,64);at=p.auxat+p.auxbytes;p.need.scratch_alignment=64;}
    p.need.scratch_bytes=at;return p;
}
void publish(Maybe dst,const uint64_t*data,std::size_t n,bool negative)noexcept{
    if(!dst)return;auto*r=dst->begin_write();if(n&&r->data!=data)std::memmove(r->data,data,8*n);r->size=n;r->negative=n&&negative;dst->finish_write();
}
void execute(State&s,const Outputs&p,Maybe qo,Maybe ro,integer_view n,execution ex,Round mode)noexcept{
    detail::require(p.need.supported(),SBN3_FATAL_ARGUMENT,p.need.reason);
    detail::require(!ex.scope&&(!ex.team||ex.team==s.team)&&pthread_equal(s.owner,pthread_self()),SBN3_FATAL_TEAM,"prepared divisor owner/team");
    detail::require(ex.scratch.size()>=p.need.scratch_bytes&&(!p.need.scratch_bytes||!(uintptr_t(ex.scratch.data())&(p.need.scratch_alignment-1))),SBN3_FATAL_WORKSPACE,"prepared division output staging",p.need.scratch_bytes,ex.scratch.size());
    detail::require(!overlaps(ex.scratch.data(),p.need.scratch_bytes,&s,s.bytes)&&!overlaps(ex.scratch.data(),p.need.scratch_bytes,n.data(),n.size()*8),SBN3_FATAL_ARGUMENT,"prepared staging alias");
    for(auto out:{qo,ro})if(out)detail::require(!overlaps(ex.scratch.data(),p.need.scratch_bytes,out->view().data(),out->capacity()*8),SBN3_FATAL_ARGUMENT,"prepared staging/output alias");
    uint64_t dummyq=0,dummyr=0;
    auto*q=!qo?nullptr:p.qstage?(p.need.quotient_limbs?reinterpret_cast<uint64_t*>(ex.scratch.data()+p.qat):&dummyq):qo->begin_write()->data;
    auto*r=p.rstage?(p.need.remainder_limbs?reinterpret_cast<uint64_t*>(ex.scratch.data()+p.rat):&dummyr):ro->begin_write()->data;
    sbn3_divrem_result result{};
    if(n.size()<s.dn){if(n.size())std::memcpy(r,n.data(),8*n.size());result.remainder_limbs=n.size();}
    else if(q)sbn3_divrem_execute(s.binding,{n.data(),n.size()},{q,p.need.quotient_limbs},{r,p.need.remainder_limbs},&result);
    else sbn3_remainder_execute(s.binding,{n.data(),n.size()},{r,p.need.remainder_limbs},p.auxbytes?ex.scratch.data()+p.auxat:nullptr,p.auxbytes,&result);
    bool rn=n.negative();const bool bump=result.remainder_limbs&&(mode==Round::floor?n.negative()!=s.negative:mode==Round::ceil?n.negative()==s.negative:mode==Round::euclidean?n.negative():false);
    if(bump){
        if(q){std::size_t i=0;for(;i<result.quotient_limbs&&q[i]==UINT64_MAX;++i)q[i]=0;
        if(i==result.quotient_limbs)q[result.quotient_limbs++]=1;else ++q[i];}
        const auto d=v3::prepared_divisor_words(s.binding);uint64_t borrow=0;
        for(std::size_t j=0;j<s.dn;++j){const auto a=d.word(j),b=j<result.remainder_limbs?r[j]:0,t=a-b,v=t-borrow;borrow=(a<b)|(t<borrow);r[j]=v;}
        detail::require(!borrow,SBN3_FATAL_MATH,"prepared rounded remainder");result.remainder_limbs=v3::bits::normalized_size(r,s.dn);rn=!rn;
    }
    publish(qo,q,result.quotient_limbs,n.negative()!=s.negative);publish(ro,r,result.remainder_limbs,rn);
}
State&get(void*p)noexcept{detail::require(p,SBN3_FATAL_LIFETIME,"empty prepared divisor");return *static_cast<State*>(p);}
}
requirements prepared_divisor::storage_requirements(std::size_t dn,std::size_t nn,unsigned workers,unsigned reuse)noexcept{sbn3_divrem_plan plan;return plan_storage(dn,nn,workers,reuse,plan).need;}
prepared_divisor::prepared_divisor(integer_view d,std::size_t nn,memory_resource&r,sbn3_team*team,unsigned reuse)noexcept{
    sbn3_divrem_plan plan;const auto s=plan_storage(d.size(),nn,team?sbn3_team_workers(team):1,reuse,plan);detail::require(s.need.supported(),SBN3_FATAL_ARGUMENT,s.need.reason);
    allocation_=r.allocate(s.need.scratch_bytes,s.need.scratch_alignment);resource_=&r;
    state_=construct(s,plan,d,nn,{static_cast<std::byte*>(allocation_.data),allocation_.bytes},team);
}
prepared_divisor prepared_divisor::bind(integer_view d,std::size_t nn,std::span<std::byte>storage,sbn3_team*team,unsigned reuse)noexcept{
    sbn3_divrem_plan plan;const auto s=plan_storage(d.size(),nn,team?sbn3_team_workers(team):1,reuse,plan);return prepared_divisor(construct(s,plan,d,nn,storage,team));
}
void prepared_divisor::close()noexcept{
    if(!state_)return;auto&s=get(state_);detail::require(pthread_equal(s.owner,pthread_self()),SBN3_FATAL_TEAM,"prepared divisor destruction owner");
    sbn3_divrem_unbind(s.binding);if(s.session)s.session->~ResidentSession();s.~State();state_=nullptr;
    if(resource_)resource_->deallocate(allocation_);resource_=nullptr;allocation_={};
}
std::size_t prepared_divisor::storage_bytes()const noexcept{return get(state_).bytes;}
division_requirements prepared_divisor::query(Maybe q,Maybe r,integer_view n,Round mode)const noexcept{return quote(get(state_),q,r,n,mode).need;}
void prepared_divisor::execute(Maybe q,Maybe r,integer_view n,execution ex,Round mode)noexcept{
    auto&s=get(state_);const auto p=quote(s,q,r,n,mode);sbn::execute(s,p,q,r,n,ex,mode);
}
division_result prepared_divisor::divrem(const integer&n,Round mode)noexcept{
    auto&s=get(state_);division_result out{integer::with_resource_of(n),integer::with_resource_of(n)};
    out.quotient.reserve(quotient_capacity(n.size(),s.dn,mode));out.remainder.reserve(mode==Round::trunc?std::min(n.size(),s.dn):s.dn);
    execute(out.quotient.ref(),out.remainder.ref(),n.view(),{},mode);return out;
}
integer prepared_divisor::divide(const integer&n,Round mode)noexcept{
    auto out=integer::with_resource_of(n);out.reserve(quotient_capacity(n.size(),get(state_).dn,mode));
    const auto need=query(out.ref(),std::nullopt,n.view(),mode);detail::require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);
    workspace w(n.resource());w.reserve(need.scratch_bytes,need.scratch_alignment);execute(out.ref(),std::nullopt,n.view(),w.on(),mode);return out;
}
integer prepared_divisor::remainder(const integer&n,Round mode)noexcept{
    auto out=integer::with_resource_of(n);out.reserve(mode==Round::trunc?std::min(n.size(),get(state_).dn):get(state_).dn);
    const auto need=query(std::nullopt,out.ref(),n.view(),mode);detail::require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);
    workspace w(n.resource());w.reserve(need.scratch_bytes,need.scratch_alignment);execute(std::nullopt,out.ref(),n.view(),w.on(),mode);return out;
}
}
