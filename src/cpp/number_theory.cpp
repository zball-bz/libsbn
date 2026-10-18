// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/number_theory.hpp"
#include "sbn3/gcd.h"
#include "sbn3/jacobi.h"
#include "common/checked.hpp"
#include <array>
namespace sbn {
namespace {
using v3::overlaps;
using Maybe=std::optional<integer_ref>;
struct Layout {bezout_requirements need;std::array<std::size_t,3>offset{},capacity{};std::array<bool,3>stage{};std::size_t workat=0,workbytes=0;};
support status(sbn3_query_result s)noexcept{return s==SBN3_SUPPORTED?support::yes:s==SBN3_QUERY_CAPACITY?support::capacity:support::unsupported;}
bool input_alias(integer_ref out,integer_view a,integer_view b)noexcept{
    return overlaps(out.view().data(),out.capacity()*8,a.data(),a.size()*8)||overlaps(out.view().data(),out.capacity()*8,b.data(),b.size()*8);
}
Layout quote(std::array<Maybe,3>out,integer_view a,integer_view b,unsigned workers,bool extended)noexcept{
    Layout l;sbn3_gcd_info info{};sbn3_gcd_options opts{workers};
    const auto st=extended?sbn3_exgcd_query(a.size(),b.size(),unsigned(bool(out[1]))|unsigned(bool(out[2]))*2,&opts,&info):
                            sbn3_gcd_query(a.size(),b.size(),&opts,&info);
    l.need.status=status(st);l.need.reason="GCD size/workers";if(st!=SBN3_SUPPORTED)return l;
    l.need.reason="";l.capacity={info.gcd_capacity,info.x_capacity,info.y_capacity};l.need.gcd_limbs=l.capacity[0];l.need.x_limbs=l.capacity[1];l.need.y_limbs=l.capacity[2];
    l.need.output_limbs=l.capacity[0]+l.capacity[1]+l.capacity[2];l.need.scratch_alignment=info.scratch_alignment;
    std::size_t at=0;
    for(unsigned i=0;i<3;++i)if(out[i]){
        if(out[i]->capacity()<l.capacity[i]){l.need.status=support::capacity;l.need.reason="GCD destination capacity";return l;}
        for(unsigned j=0;j<i;++j)if(out[j]&&(out[i]->same_object(*out[j])||overlaps(out[i]->view().data(),out[i]->capacity()*8,out[j]->view().data(),out[j]->capacity()*8))){l.need.status=support::unsupported;l.need.reason="GCD output overlap";return l;}
        if(input_alias(*out[i],a,b)){l.stage[i]=true;l.offset[i]=at;at+=8*l.capacity[i];}
    }
    if(!v3::align_size(at,info.scratch_alignment,l.workat)){l.need.status=support::capacity;return l;}
    l.workbytes=info.scratch_bytes;l.need.scratch_bytes=l.workat+l.workbytes;return l;
}
void check_execution(const requirements&need,integer_view a,integer_view b,execution ex)noexcept{
    detail::require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);
    detail::require(!ex.scope,SBN3_FATAL_TEAM,"GCD/Jacobi needs idle root Team");
    detail::require(ex.scratch.size()>=need.scratch_bytes&&(!need.scratch_bytes||!(uintptr_t(ex.scratch.data())&(need.scratch_alignment-1))),SBN3_FATAL_WORKSPACE,"number theory scratch",need.scratch_bytes,ex.scratch.size());
    detail::require(!overlaps(ex.scratch.data(),need.scratch_bytes,a.data(),a.size()*8)&&!overlaps(ex.scratch.data(),need.scratch_bytes,b.data(),b.size()*8),SBN3_FATAL_ARGUMENT,"number theory scratch alias");
}
void run(const Layout&l,std::array<Maybe,3>out,integer_view a,integer_view b,execution ex,bool extended)noexcept{
    check_execution(l.need,a,b,ex);std::array<sbn3_int,3>values{};std::array<sbn3_int*,3>ptr{};
    for(unsigned j=0;j<3;++j)if(out[j]){
        detail::require(!overlaps(ex.scratch.data(),l.need.scratch_bytes,out[j]->view().data(),out[j]->capacity()*8),SBN3_FATAL_ARGUMENT,"GCD output/scratch alias");
        auto*dst=out[j]->begin_write();values[j]={l.stage[j]?reinterpret_cast<uint64_t*>(ex.scratch.data()+l.offset[j]):dst->data,l.capacity[j],0,0};ptr[j]=&values[j];
    }
    void*work=l.workbytes?ex.scratch.data()+l.workat:nullptr;
    if(extended)sbn3_int_exgcd(ptr[0],ptr[1],ptr[2],a.c_view(),b.c_view(),work,l.workbytes,ex.team);
    else {values[0].size=sbn3_gcd(values[0].data,a.data(),a.size(),b.data(),b.size(),work,l.workbytes,ex.team);values[0].negative=0;}
    for(unsigned j=0;j<3;++j)if(out[j]){auto*dst=out[j]->begin_write();if(l.stage[j]&&values[j].size)std::memmove(dst->data,values[j].data,8*values[j].size);dst->size=values[j].size;dst->negative=values[j].negative;out[j]->finish_write();}
}
}
requirements gcd_requirements(integer_ref r,integer_view a,integer_view b,unsigned workers)noexcept{return quote({r,std::nullopt,std::nullopt},a,b,workers,false).need;}
void gcd_into(integer_ref r,integer_view a,integer_view b,execution ex)noexcept{const auto l=quote({r,std::nullopt,std::nullopt},a,b,ex.workers(),false);run(l,{r,std::nullopt,std::nullopt},a,b,ex,false);}
integer gcd(const integer&a,const integer&b)noexcept{
    auto g=integer::with_resource_of(a);g.reserve(std::max(a.size(),b.size()));const auto l=quote({g.ref(),std::nullopt,std::nullopt},a.view(),b.view(),1,false);
    workspace w(a.resource());w.reserve(l.need.scratch_bytes,l.need.scratch_alignment);run(l,{g.ref(),std::nullopt,std::nullopt},a.view(),b.view(),w.on(),false);return g;
}
bezout_requirements extended_gcd_requirements(integer_ref g,Maybe x,Maybe y,integer_view a,integer_view b,unsigned workers)noexcept{return quote({g,x,y},a,b,workers,true).need;}
void extended_gcd_into(integer_ref g,Maybe x,Maybe y,integer_view a,integer_view b,execution ex)noexcept{const auto l=quote({g,x,y},a,b,ex.workers(),true);run(l,{g,x,y},a,b,ex,true);}
bezout_result extended_gcd(const integer&a,const integer&b)noexcept{
    sbn3_gcd_info info;detail::require(sbn3_exgcd_query(a.size(),b.size(),3,nullptr,&info)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"extended GCD size");
    bezout_result out{integer::with_resource_of(a),integer::with_resource_of(a),integer::with_resource_of(a)};
    out.gcd.reserve(info.gcd_capacity);out.x.reserve(info.x_capacity);out.y.reserve(info.y_capacity);
    workspace w(a.resource());w.reserve(info.scratch_bytes,info.scratch_alignment);extended_gcd_into(out.gcd.ref(),out.x.ref(),out.y.ref(),a.view(),b.view(),w.on());return out;
}
integer lcm(const integer&a,const integer&b)noexcept{if(a.zero()||b.zero())return integer::with_resource_of(a);auto g=gcd(a,b);return abs(exact_divide(a,g)*b);}
std::optional<integer> inverse_mod(const integer&a,const integer&m)noexcept{
    detail::require(!m.zero(),SBN3_FATAL_ARGUMENT,"zero inverse modulus");
    sbn3_gcd_info info;detail::require(sbn3_exgcd_query(a.size(),m.size(),1,nullptr,&info)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"modular inverse size");
    auto g=integer::with_resource_of(a),x=integer::with_resource_of(a);g.reserve(info.gcd_capacity);x.reserve(info.x_capacity);
    workspace w(a.resource());w.reserve(info.scratch_bytes,info.scratch_alignment);extended_gcd_into(g.ref(),x.ref(),std::nullopt,a.view(),m.view(),w.on());
    if(g!=1)return std::nullopt;return mod(x,m);
}
requirements jacobi_requirements(integer_view a,integer_view n,unsigned workers)noexcept{
    requirements r;if(n.zero()||n.negative()||!(n.data()[0]&1)){r.status=support::unsupported;r.reason="Jacobi denominator must be positive odd";return r;}
    sbn3_jacobi_info info{};sbn3_jacobi_options opts{workers};r.status=status(sbn3_jacobi_query(a.size(),n.size(),&opts,&info));
    r.scratch_bytes=info.scratch_bytes;r.scratch_alignment=info.scratch_alignment;r.reason=r.supported()?"":"Jacobi size/workers";return r;
}
int jacobi(integer_view a,integer_view n,execution ex)noexcept{
    const auto need=jacobi_requirements(a,n,ex.workers());check_execution(need,a,n,ex);return sbn3_int_jacobi(a.c_view(),n.c_view(),ex.scratch.data(),need.scratch_bytes,ex.team);
}
int jacobi(const integer&a,const integer&n)noexcept{const auto need=jacobi_requirements(a.view(),n.view());detail::require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);workspace w(a.resource());w.reserve(need.scratch_bytes,need.scratch_alignment);return jacobi(a.view(),n.view(),w.on());}
}
