// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/integer.hpp"
#include "sbn3/execution.hpp"
#include "product/program.hpp"
#include "backend/u52/kernels.hpp"
#include "core/x86_64/word.hpp"
#include "value/bitwise.hpp"
namespace sbn {
namespace {
using namespace sbn::v3;
struct Product {
    ProductProgramPlan p{};std::size_t bytes=0;bool square=false;
};
bool plan(Product&x,std::size_t an,std::size_t bn,unsigned workers,bool sq){
    sbn3_mul_options options{};options.workers=workers;options.borrow_output=1;
    if(sq){sbn3_product_request q{};q.kind=SBN3_PRODUCT_SQR;q.a_limbs=an;sbn3_product_info info{};
        if(sbn3_product_query(&q,&options,&x.p.plan,&info)!=SBN3_SUPPORTED)return false;
        x.p.info=info.mul;
        if(product_program_describe(an,0,x.p)!=SBN3_SUPPORTED)return false;
    }else if(product_program_query(an,bn,options,x.p)!=SBN3_SUPPORTED)return false;
    x.square=sq;
    x.bytes=x.p.prepared_bytes+x.p.info.workspace_bytes+x.p.info.workspace_alignment+384;
    return true;
}
bool alias(const sbn3_int&r,integer_view a,integer_view b,std::size_t written){
    return overlaps(r.data,written*8,a.data(),a.size()*8)||overlaps(r.data,written*8,b.data(),b.size()*8);
}
[[gnu::noinline]]requirements query_large(integer_ref dst,integer_view a,integer_view b,unsigned workers,bool sq){
    Product p;if(!plan(p,a.size(),b.size(),workers,sq))return {support::unsupported,0,128,0,"multiplication backend shape"};
    const auto view=dst.view();sbn3_int description{const_cast<std::uint64_t*>(view.data()),dst.capacity(),view.size(),unsigned(view.negative())};const auto*r=&description;const auto n=sbn3_mul_output_capacity(&p.p.info);
    const bool stage=r->capacity<n||(std::uintptr_t(r->data)&(p.p.info.output_alignment-1))||alias(*r,a,b,std::min(r->capacity,n));
    if(stage)p.bytes+=n*8+std::max(std::size_t(128),p.p.info.output_alignment);
    return {r->capacity<a.size()+b.size()?support::capacity:support::yes,p.bytes,std::max(std::size_t(128),p.p.info.workspace_alignment),a.size()+b.size(),""};
}
struct Run {const Product*p;integer_view a,b;sbn3_int*out;execution ex;bool stage;};
void run_product(void*opaque,sbn3_team_scope*scope){
    auto&r=*static_cast<Run*>(opaque);auto frame=Frame::external(r.ex.scratch.data(),r.ex.scratch.size());const auto&p=r.p->p;
    const auto n=sbn3_mul_output_capacity(&p.info);auto*dest=r.stage?static_cast<std::uint64_t*>(frame.allocate(n*8,std::max(std::size_t(64),p.info.output_alignment))):r.out->data;
    auto tables=frame.subframe(p.prepared_bytes+128,128);const auto program=product_program_prepare(p,tables);
    auto work=frame.subframe(p.info.workspace_bytes,p.info.workspace_alignment);
    product_program_execute(program,work,scope,{r.a.data(),r.a.size()},r.p->square?sbn3_const_limbs{}:sbn3_const_limbs{r.b.data(),r.b.size()},{dest,n});
    const std::size_t logical=r.a.size()+r.b.size();if(r.stage)std::memcpy(r.out->data,dest,logical*8);
    r.out->size=bits::normalized_size(r.out->data,logical);r.out->negative=r.out->size&&(r.a.negative()!=r.b.negative());
}
[[gnu::noinline]]void large_into(integer_ref dst,integer_view a,integer_view b,execution ex,bool sq){
    Product p;detail::require(plan(p,a.size(),b.size(),ex.workers(),sq),SBN3_FATAL_ARGUMENT,"C++ multiplication shape");
    auto*r=dst.begin_write();detail::require(r->capacity>=a.size()+b.size(),SBN3_FATAL_WORKSPACE,"C++ product output",a.size()+b.size(),r->capacity);
    const auto n=sbn3_mul_output_capacity(&p.p.info);bool stage=r->capacity<n||(std::uintptr_t(r->data)&(p.p.info.output_alignment-1))||alias(*r,a,b,std::min(r->capacity,n));
    const std::size_t need=p.bytes+(stage?n*8+std::max(std::size_t(128),p.p.info.output_alignment):0);
    detail::require(ex.scratch.size()>=need,SBN3_FATAL_WORKSPACE,"C++ product scratch",need,ex.scratch.size());
    Run run{&p,a,b,r,ex,stage};
    if(ex.scope)run_product(&run,ex.scope);else if(ex.team)sbn3_team_run(ex.team,run_product,&run);else serial_team_run(run_product,&run);dst.finish_write();
}
[[gnu::noinline]]integer large_owned(const integer&a,const integer&b,bool sq){
    Product p;detail::require(plan(p,a.size(),b.size(),1,sq),SBN3_FATAL_ARGUMENT,"C++ owning product shape");
    integer out=integer::with_resource_of(a);out.reserve(sbn3_mul_output_capacity(&p.p.info));workspace scratch(out.resource());scratch.reserve(p.bytes,std::max(std::size_t(128),p.p.info.workspace_alignment));
    auto dest=out.ref();const bool staged=bool(std::uintptr_t(out.data())&(p.p.info.output_alignment-1));
    if(staged)scratch.reserve(p.bytes+sbn3_mul_output_capacity(&p.p.info)*8+std::max(std::size_t(128),p.p.info.output_alignment),std::max(std::size_t(128),p.p.info.workspace_alignment));
    Run run{&p,a.view(),b.view(),dest.begin_write(),scratch.on(),staged};serial_team_run(run_product,&run);dest.finish_write();return out;
}
}
static requirements query_product(integer_ref dst,integer_view a,integer_view b,unsigned workers,bool sq)noexcept{
    if(!workers||workers>32||a.size()>SIZE_MAX/128||b.size()>SIZE_MAX/128)return {support::unsupported,0,128,0,"product domain"};
    if(a.zero()||b.zero())return {};
    const std::size_t n=a.size()+b.size();if((a.size()<=6&&b.size()<=6)||std::min(a.size(),b.size())<=1){
        const auto v=dst.view();sbn3_int r{const_cast<std::uint64_t*>(v.data()),dst.capacity(),v.size(),unsigned(v.negative())};return {r.capacity<n?support::capacity:support::yes,n>12&&alias(r,a,b,std::min(n,r.capacity))?n*8+63:0,64,n,""};}
    return query_large(dst,a,b,workers,sq);
}
requirements multiply_requirements(integer_ref dst,integer_view a,integer_view b,unsigned workers)noexcept{return query_product(dst,a,b,workers,false);}
requirements square_requirements(integer_ref dst,integer_view a,unsigned workers)noexcept{return query_product(dst,a,a,workers,true);}
void mul_into(integer_ref dst,integer_view a,integer_view b,execution ex)noexcept{
    auto*r=dst.begin_write();if(a.zero()||b.zero()){r->size=r->negative=0;dst.finish_write();return;}
    if((a.size()<=6&&b.size()<=6)||std::min(a.size(),b.size())<=1){
        const auto n=a.size()+b.size();detail::require(r->capacity>=n,SBN3_FATAL_WORKSPACE,"small product output",n,r->capacity);
        alignas(64) std::uint64_t small[12];std::uint64_t*out=r->data;
        if(alias(*r,a,b,n)){if(n<=12)out=small;else{auto f=sbn::v3::Frame::external(ex.scratch.data(),ex.scratch.size());out=f.alloc<std::uint64_t>(n);
                sbn3_mul_basecase(out,n,a.data(),a.size(),b.data(),b.size());std::memcpy(r->data,out,n*8);r->size=sbn::v3::bits::normalized_size(r->data,n);r->negative=r->size&&(a.negative()!=b.negative());dst.finish_write();return;}}
        sbn3_mul_basecase(out,n,a.data(),a.size(),b.data(),b.size());if(out!=r->data)std::memcpy(r->data,out,n*8);
        r->size=sbn::v3::bits::normalized_size(r->data,n);r->negative=r->size&&(a.negative()!=b.negative());dst.finish_write();return;
    }
    large_into(dst,a,b,ex,false);
}
void square_into(integer_ref r,integer_view a,execution ex)noexcept{
    if(a.size()<=6){mul_into(r,a,a,ex);return;}large_into(r,a,a,ex,true);
}
integer multiply(const integer&a,const integer&b)noexcept{
    if(a.zero()||b.zero())return integer::with_resource_of(a);
    if((a.size()<=6&&b.size()<=6)||std::min(a.size(),b.size())<=1){auto out=integer::with_resource_of(a);out.reserve(a.size()+b.size());mul_into(out.ref(),a.view(),b.view(),{});return out;}
    return large_owned(a,b,&a==&b);
}
integer square(const integer&a)noexcept{if(a.size()<=6)return multiply(a,a);return large_owned(a,a,true);}
integer&integer::operator*=(const integer&x)noexcept{auto out=multiply(*this,x);*this=std::move(out);return *this;}
}
