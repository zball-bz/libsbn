// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/power.hpp"
#include "product/program.hpp"
#include "value/bitwise.hpp"
#include <bit>
namespace sbn {
namespace {
using namespace sbn::v3;
struct Stage {ProductProgramPlan program{};std::size_t bytes=0,capacity=0,alignment=64;bool leaf=false;};
bool stage(Stage&s,std::size_t an,std::size_t bn,unsigned workers,bool square)noexcept{
    s.capacity=an+bn;
    if((an<=6&&bn<=6)||std::min(an,bn)<=1){s.leaf=true;return true;}
    sbn3_mul_options o{};o.workers=workers;o.borrow_output=1;
    if(square){sbn3_product_request r{};r.kind=SBN3_PRODUCT_SQR;r.a_limbs=an;sbn3_product_info info;
        if(sbn3_product_query(&r,&o,&s.program.plan,&info)!=SBN3_SUPPORTED)return false;
        s.program.info=info.mul;if(product_program_describe(an,0,s.program)!=SBN3_SUPPORTED)return false;
    }else if(product_program_query(an,bn,o,s.program)!=SBN3_SUPPORTED)return false;
    s.capacity=sbn3_mul_output_capacity(&s.program.info);s.alignment=std::max(s.program.info.workspace_alignment,s.program.info.output_alignment);
    s.bytes=s.program.prepared_bytes+s.program.info.workspace_bytes+s.program.info.workspace_alignment+384;return true;
}
std::size_t width(std::size_t bits,std::size_t exponent)noexcept{return (bits*exponent+63)/64;}
enum class Kind {constant,copy,bit,small,ladder};
struct Power {requirements need;Kind kind=Kind::constant;std::size_t bits=0,shift=0,buffer=0,work=0,alignment=64;};
Power quote(integer_view a,std::size_t e,unsigned workers)noexcept{
    Power p;p.need.scratch_alignment=64;
    if(!workers||workers>32||a.size()>((std::size_t(1)<<31)-16)){p.need.status=support::unsupported;p.need.reason="power shape/workers";return p;}
    if(!e||a.zero()||(a.size()==1&&a.data()[0]==1)){p.need.output_limbs=e&&a.zero()?0:1;return p;}
    p.bits=sbn3_int_bit_length(a.c_view());
    if(e==1){p.kind=Kind::copy;p.need.output_limbs=a.size();return p;}
    const __uint128_t result_bits=__uint128_t(p.bits)*e;
    if(result_bits>__uint128_t((std::size_t(1)<<31)-18)*64){p.need.status=support::unsupported;p.need.reason="power result domain";return p;}
    if(detail::word_population(a.data(),a.size())==1){p.kind=Kind::bit;p.shift=(p.bits-1)*e;p.need.output_limbs=p.shift/64+1;return p;}
    p.need.output_limbs=width(p.bits,e);
    if(result_bits<=128){p.kind=Kind::small;return p;}
    p.kind=Kind::ladder;p.buffer=a.size();std::size_t prefix=1;
    const unsigned top=unsigned(std::bit_width(e)-1);
    for(unsigned bit=top;bit-->0;){
        const std::size_t len=width(p.bits,prefix);Stage sq;
        if(!stage(sq,len,len,workers,true)){p.need.status=support::unsupported;p.need.reason="power square backend";return p;}
        p.buffer=std::max(p.buffer,sq.capacity);p.work=std::max(p.work,sq.bytes);p.alignment=std::max(p.alignment,sq.alignment);prefix*=2;
        if((e>>bit)&1){Stage mul;if(!stage(mul,width(p.bits,prefix),a.size(),workers,false)){p.need.status=support::unsupported;p.need.reason="power multiply backend";return p;}
            p.buffer=std::max(p.buffer,mul.capacity);p.work=std::max(p.work,mul.bytes);p.alignment=std::max(p.alignment,mul.alignment);++prefix;}
    }
    p.need.scratch_alignment=p.alignment;
    p.need.scratch_bytes=2*(p.buffer*8+p.alignment)+p.work+128;return p;
}
void product(const Stage&s,uint64_t*out,const uint64_t*a,std::size_t an,const uint64_t*b,std::size_t bn,
             Frame&space,sbn3_team_scope*scope,bool square)noexcept{
    if(s.leaf){sbn3_mul_basecase(out,an+bn,a,an,b,bn);return;}
    auto tables=space.subframe(s.program.prepared_bytes+128,128);const auto program=product_program_prepare(s.program,tables);
    auto work=space.subframe(s.program.info.workspace_bytes,s.program.info.workspace_alignment);
    product_program_execute(program,work,scope,{a,an},square?sbn3_const_limbs{}:sbn3_const_limbs{b,bn},{out,s.capacity});
}
struct Run {integer_ref out;integer_view a;std::size_t exponent;execution ex;Power p;};
void ladder(void*raw,sbn3_team_scope*scope){
    auto&r=*static_cast<Run*>(raw);auto frame=Frame::external(r.ex.scratch.data(),r.ex.scratch.size());
    auto*current=static_cast<uint64_t*>(frame.allocate(r.p.buffer*8,r.p.alignment));auto*next=static_cast<uint64_t*>(frame.allocate(r.p.buffer*8,r.p.alignment));
    std::memcpy(current,r.a.data(),r.a.size()*8);std::size_t prefix=1;
    const unsigned top=unsigned(std::bit_width(r.exponent)-1);
    for(unsigned bit=top;bit-->0;){
        const auto len=width(r.p.bits,prefix);Stage sq;detail::require(stage(sq,len,len,scope->width,true),SBN3_FATAL_MATH,"power square replay");
        {FrameMark mark(frame);auto work=frame.subframe(r.p.work,128);product(sq,next,current,len,current,len,work,scope,true);}
        prefix*=2;std::swap(current,next);
        if((r.exponent>>bit)&1){const auto len=width(r.p.bits,prefix);Stage mul;detail::require(stage(mul,len,r.a.size(),scope->width,false),SBN3_FATAL_MATH,"power multiply replay");
            {FrameMark mark(frame);auto work=frame.subframe(r.p.work,128);product(mul,next,current,len,r.a.data(),r.a.size(),work,scope,false);}++prefix;std::swap(current,next);}
    }
    const auto size=bits::normalized_size(current,r.p.need.output_limbs);auto*out=r.out.begin_write();if(size)std::memcpy(out->data,current,size*8);
    out->size=size;out->negative=size&&r.a.negative()&&(r.exponent&1);r.out.finish_write();
}
void execute(integer_ref out,integer_view a,std::size_t e,execution ex,const Power&p)noexcept{
    detail::require(p.need.supported(),SBN3_FATAL_ARGUMENT,p.need.reason);detail::require(out.capacity()>=p.need.output_limbs,SBN3_FATAL_WORKSPACE,"power output",p.need.output_limbs,out.capacity());
    if(p.kind==Kind::copy){assign_into(out,a);return;}
    if(p.kind==Kind::constant){auto*r=out.begin_write();r->size=p.need.output_limbs;if(r->size)r->data[0]=1;r->negative=r->size&&e&&(e&1)&&a.negative();out.finish_write();return;}
    if(p.kind==Kind::bit){auto*r=out.begin_write();std::memset(r->data,0,p.need.output_limbs*8);r->data[p.shift/64]=uint64_t(1)<<(p.shift%64);r->size=p.need.output_limbs;r->negative=a.negative()&&(e&1);out.finish_write();return;}
    if(p.kind==Kind::small){__uint128_t value=a.data()[0];if(a.size()>1)value|=__uint128_t(a.data()[1])<<64;__uint128_t acc=1;auto power=e;
        while(power){if(power&1)acc*=value;power>>=1;if(power)value*=value;}
        auto*r=out.begin_write();r->data[0]=uint64_t(acc);if(p.need.output_limbs>1)r->data[1]=uint64_t(acc>>64);r->size=acc>>64?2:1;r->negative=a.negative()&&(e&1);out.finish_write();return;}
    detail::require(ex.scratch.size()>=p.need.scratch_bytes&&!(uintptr_t(ex.scratch.data())&(p.need.scratch_alignment-1)),SBN3_FATAL_WORKSPACE,"power scratch",p.need.scratch_bytes,ex.scratch.size());
    detail::require(!overlaps(ex.scratch.data(),p.need.scratch_bytes,a.data(),a.size()*8)&&!overlaps(ex.scratch.data(),p.need.scratch_bytes,out.view().data(),out.capacity()*8),SBN3_FATAL_ARGUMENT,"power scratch alias");
    Run run{out,a,e,ex,p};if(ex.scope)ladder(&run,ex.scope);else if(ex.team)sbn3_team_run(ex.team,ladder,&run);else serial_team_run(ladder,&run);
}
}
requirements power_requirements(integer_ref out,integer_view a,std::size_t e,unsigned workers)noexcept{
    auto p=quote(a,e,workers);if(p.need.supported()&&out.capacity()<p.need.output_limbs){p.need.status=support::capacity;p.need.reason="power output capacity";}return p.need;
}
void power_into(integer_ref out,integer_view a,std::size_t e,execution ex)noexcept{const auto p=quote(a,e,ex.workers());execute(out,a,e,ex,p);}
integer pow(const integer&a,std::size_t e)noexcept{
    const auto p=quote(a.view(),e,1);detail::require(p.need.supported(),SBN3_FATAL_ARGUMENT,p.need.reason);
    auto out=integer::with_resource_of(a);out.reserve(p.need.output_limbs);workspace w(a.resource());w.reserve(p.need.scratch_bytes,p.need.scratch_alignment);execute(out.ref(),a.view(),e,w.on(),p);return out;
}
}
