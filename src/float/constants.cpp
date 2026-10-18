// Original libsbn code; LGPL-3.0-or-later.
#include "float/constants.hpp"
#include "float/agm.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <new>

namespace sbn::v3::floating {
bool real_append(std::size_t& n,std::size_t bytes,std::size_t alignment) noexcept {
    if(!bytes)return true;
    std::size_t at=0;return align_size(n,alignment,at)&&add_size(at,bytes,n);
}
std::int64_t real_top(float_view x) noexcept {
    require(x.finite()&&!x.zero(),SBN3_FATAL_MATH,"real nonzero finite value");
    const __int128 e=__int128(x.exponent2())+magnitude_bits(x.magnitude());
    require(e>=INT64_MIN&&e<=INT64_MAX,SBN3_FATAL_SIZE,"real top exponent");
    return std::int64_t(e);
}
float_view real_scaled(float_view x,std::int64_t shift) noexcept {
    const __int128 e=__int128(x.exponent2())+shift;
    require(e>=INT64_MIN&&e<=INT64_MAX,SBN3_FATAL_SIZE,"real scale exponent");
    return float_view::from_magnitude(x.magnitude().limbs(),std::int64_t(e),x.negative());
}
ErrorMagnitude real_upper(float_view x) noexcept {
    return x.zero()?ErrorMagnitude{}:ErrorMagnitude::power_of_two(real_top(x));
}
float_report real_finish(float_ref out,float_view x,ErrorMagnitude error,rounding mode) noexcept {
    auto report=round_into(out,x,mode);
    report.error=ErrorMagnitude::from(report.error).add(error).publish();return report;
}
RealArithmeticPlan real_arithmetic_query(bits precision,unsigned workers) noexcept {
    RealArithmeticPlan p;p.precision=precision.count;p.workers=workers;
    p.need.scratch_alignment=128;p.need.output_limbs=(precision.count+63)/64;
    if(!precision.count||precision.count>float_precision_limit||!workers||workers>32){
        p.need.status=support::unsupported;p.need.reason="real arithmetic shape";return p;
    }
    if(float_product_query(precision.count,precision.count+128,precision.count+128,false,workers,p.multiply)!=SBN3_SUPPORTED||
       float_product_query(precision.count,precision.count+128,0,true,workers,p.square)!=SBN3_SUPPORTED){
        p.need.status=support::unsupported;p.need.reason="real arithmetic product";return p;
    }
    p.root_need=root_program_requirements(precision,workers,RootKind::square_root);
    p.division_need=division_program_requirements(precision,workers,SBN3_NEWTON_DIVIDE);
    if(!p.root_need.supported()||!p.division_need.supported()){
        p.need.status=support::unsupported;p.need.reason="real arithmetic root/division";return p;
    }
    // Ask the actual public sum quote. Its capacity status is expected for
    // this storage-free descriptor; scratch depends only on precision and
    // the nonzero finite operand shapes, not on a fictitious value buffer.
    float_buffer descriptor({},precision);const std::uint64_t word=1;
    const auto one=float_view::from_magnitude({&word,1},0);
    p.sum_need=add_requirements(descriptor.ref(),one,one);
    require(p.sum_need.status==support::capacity,SBN3_FATAL_MATH,"sum descriptor quote");
    p.need.scratch_alignment=std::max({std::size_t(128),p.multiply.alignment,p.square.alignment,
        p.root_need.scratch_alignment,p.division_need.scratch_alignment,p.sum_need.scratch_alignment});
    auto& n=p.need.scratch_bytes;
    if(!real_append(n,p.multiply.scratch_bytes,p.multiply.alignment)||
       !real_append(n,p.square.scratch_bytes,p.square.alignment)||
       !real_append(n,p.root_need.scratch_bytes,p.root_need.scratch_alignment)||
       !real_append(n,p.division_need.scratch_bytes,p.division_need.scratch_alignment)||
       !real_append(n,p.sum_need.scratch_bytes,p.sum_need.scratch_alignment)){
        p.need.status=support::unsupported;p.need.reason="real arithmetic bytes";
    }
    return p;
}
void real_arithmetic_prepare(RealArithmetic& a,const RealArithmeticPlan& p,Frame& frame) noexcept {
    a.plan=p;
    auto take=[&](std::size_t n,std::size_t alignment){return std::span<std::byte>{
        n?static_cast<std::byte*>(frame.allocate(n,alignment)):nullptr,n};};
    a.multiply=float_product_prepare(a.plan.multiply,take(p.multiply.scratch_bytes,p.multiply.alignment));
    a.square=float_product_prepare(a.plan.square,take(p.square.scratch_bytes,p.square.alignment));
    a.root=root_program_prepare(bits{p.precision},p.workers,RootKind::square_root,
        take(p.root_need.scratch_bytes,p.root_need.scratch_alignment));
    a.division=division_program_prepare(bits{p.precision},p.workers,SBN3_NEWTON_DIVIDE,
        take(p.division_need.scratch_bytes,p.division_need.scratch_alignment));
    a.sums=take(p.sum_need.scratch_bytes,p.sum_need.scratch_alignment);
}
float_report real_add(RealArithmetic& a,float_ref out,float_view x,float_view y) noexcept {
    return sum_finite(out,x,y,execution{a.sums,nullptr,nullptr},rounding::nearest_even);
}
float_report real_mul(RealArithmetic& a,float_ref out,float_view x,float_view y,execution ex) noexcept {
    return float_product_execute(a.multiply,out,x,y,ex,rounding::nearest_even);
}
float_report real_square(RealArithmetic& a,float_ref out,float_view x,execution ex) noexcept {
    return float_product_execute(a.square,out,x,{},ex,rounding::nearest_even);
}
float_report real_root(RealArithmetic& a,float_ref out,float_view x,execution ex) noexcept {
    return root_program_execute(a.root,out,x,ex,rounding::nearest_even);
}
float_report real_divide(RealArithmetic& a,float_ref out,float_view x,float_view y,execution ex) noexcept {
    return division_program_execute(a.division,out,x,y,ex,rounding::nearest_even);
}
std::size_t cold_constant_working_precision(bits p) noexcept {
    if(!p.count||p.count>float_precision_limit)return 0;
    std::size_t target=0,working=0;if(!add_size(p.count,32,target))return 0;
    std::size_t iterations=1;while((std::size_t(4)<<iterations)<target+11)++iterations;
    if(!add_size(target,iterations+32,working)||working>float_precision_limit)return 0;return working;
}
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e33434f4e31);
struct ConstantState {
    std::uint64_t marker=0;std::uint32_t active=0;
    std::size_t precision=0,working=0,iterations=0,bytes=0;
    unsigned workers=1;ConstantKind kind=ConstantKind::pi;
    RealArithmetic arithmetic{};ColdConstantPrepared pi{};FloatAgmPrepared agm{};
    ConstantBankBinding supplied_pi{};
    float_buffer* v[8]{};
};
struct Quote {
    requirements need{},pi{},agm{};RealArithmeticPlan arithmetic{};
    std::size_t working=0,iterations=0,limbs=0;
};
Quote quote(bits p,unsigned workers,ConstantKind kind,ConstantBankBinding supplied_pi={}) noexcept {
    Quote q;q.need.scratch_alignment=128;q.need.output_limbs=(p.count+63)/64;
    if(!p.count||p.count>float_precision_limit||!workers||workers>32||
       (kind!=ConstantKind::pi&&kind!=ConstantKind::log2)){
        q.need.status=support::unsupported;q.need.reason="cold constant shape";return q;
    }
    std::size_t target=0;
    if(!add_size(p.count,32,target))return {{support::unsupported,0,128,0,"constant precision"}};
    q.iterations=1;
    while((std::size_t(4)<<q.iterations)<target+11)++q.iterations;
    if(!add_size(target,q.iterations+32,q.working)||q.working>float_precision_limit){
        q.need.status=support::unsupported;q.need.reason="cold constant working precision";return q;
    }
    q.limbs=(q.working+63)/64;q.arithmetic=real_arithmetic_query(bits{q.working},workers);
    if(!q.arithmetic.need.supported()){q.need=q.arithmetic.need;return q;}
    q.need.scratch_alignment=q.arithmetic.need.scratch_alignment;
    if(kind==ConstantKind::log2){
        if(supplied_pi.supplied()){
            if(!constant_bank_supports(supplied_pi,{q.working,0})){
                q.need.status=support::unsupported;q.need.reason="log2 supplied pi precision/radius";return q;}
        }else q.pi=cold_constant_requirements(bits{q.working},workers,ConstantKind::pi);
        q.agm=agm_program_requirements(bits{q.working},workers);
        if(!q.pi.supported()||!q.agm.supported()){
            q.need.status=support::unsupported;q.need.reason="log2 dependencies";return q;
        }
        q.need.scratch_alignment=std::max({q.need.scratch_alignment,q.pi.scratch_alignment,q.agm.scratch_alignment});
    }
    auto& n=q.need.scratch_bytes;bool ok=real_append(n,sizeof(ConstantState),128);
    for(unsigned i=0;i<8;++i)ok=ok&&real_append(n,sizeof(float_buffer),64)&&real_append(n,q.limbs*8,64);
    ok=ok&&real_append(n,q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment)&&
        real_append(n,q.pi.scratch_bytes,q.pi.scratch_alignment)&&real_append(n,q.agm.scratch_bytes,q.agm.scratch_alignment);
    if(!ok){q.need.status=support::unsupported;q.need.reason="cold constant bytes";}
    return q;
}
struct Call {ConstantState* s;float_ref out;rounding mode;float_report report{};};
void calculate(void* opaque,sbn3_team_scope* scope) {
    auto& call=*static_cast<Call*>(opaque);auto& s=*call.s;auto& a=s.arithmetic;
    const execution ex{{},nullptr,scope};const std::uint64_t one_word=1;
    const auto one=float_view::from_magnitude({&one_word,1},0);
    auto ref=[&](unsigned i){return s.v[i]->ref();};auto val=[&](unsigned i){return s.v[i]->view();};
    if(s.kind==ConstantKind::pi){
        round_into(ref(0),one);real_root(a,ref(1),real_scaled(one,-1),ex);
        round_into(ref(2),real_scaled(one,-2));
        for(std::size_t j=0;j<s.iterations;++j){
            real_add(a,ref(3),real_scaled(val(0),-1),real_scaled(val(1),-1));
            real_add(a,ref(4),real_scaled(val(0),-1),real_scaled(val(1),-1).negated());
            real_square(a,ref(5),val(4),ex);
            real_add(a,ref(2),val(2),real_scaled(val(5),std::int64_t(j)).negated());
            real_mul(a,ref(6),val(0),val(1),ex);real_root(a,ref(1),val(6),ex);
            round_into(ref(0),val(3));
        }
        real_square(a,ref(3),val(0),ex);real_divide(a,ref(4),val(3),val(2),ex);
        // Exact a_N^2/t_N lies in the proved simple GL bracket. The fixed
        // guard schedule exceeds both the bracket and forward-roundoff
        // budgets in agm-cold-constants-log-design by over sixteen bits.
        call.report=real_finish(call.out,val(4),ErrorMagnitude::power_of_two(-std::int64_t(s.precision)-16),call.mode);
    }else{
        if(s.supplied_pi.supplied())constant_bank_prefix(s.supplied_pi,ConstantKind::pi,ref(0));
        else cold_constant_execute(s.pi,ref(0),ex);
        const std::uint64_t m=(s.precision+33)/2+3;
        agm_program_execute(s.agm,ref(1),one,real_scaled(one,2-std::int64_t(m)),ex);
        const std::uint64_t twice_m=2*m;
        const auto multiplier=float_view::from_magnitude({&twice_m,1},0);
        real_mul(a,ref(2),val(1),multiplier,ex);real_divide(a,ref(3),val(0),val(2),ex);
        // 0 <= pi/(2m AGM(1,2^(2-m))) - log2 <= 2^(5-2m).
        call.report=real_finish(call.out,val(3),ErrorMagnitude::power_of_two(-std::int64_t(s.precision)-16),call.mode);
    }
}
}
requirements cold_constant_requirements(bits p,unsigned workers,ConstantKind kind,ConstantBankBinding supplied_pi) noexcept{return quote(p,workers,kind,supplied_pi).need;}
ColdConstantPrepared cold_constant_prepare(bits precision,unsigned workers,ConstantKind kind,
                                           std::span<std::byte> storage,ConstantBankBinding supplied_pi) noexcept {
    require(!supplied_pi.query_only(),SBN3_FATAL_ARGUMENT,"constant producer needs bound pi values");
    const auto q=quote(precision,workers,kind,supplied_pi);
    require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
        SBN3_FATAL_WORKSPACE,"cold constant storage",q.need.scratch_bytes,storage.size());
    require(!overlaps(storage.data(),q.need.scratch_bytes,supplied_pi.values().storage.data(),supplied_pi.values().storage.size()),SBN3_FATAL_ARGUMENT,"cold constant borrowed pi/work alias");
    auto frame=Frame::external(storage.data(),q.need.scratch_bytes);
    auto* s=::new(frame.allocate(sizeof(ConstantState),128))ConstantState{};
    s->precision=precision.count;s->working=q.working;s->iterations=q.iterations;
    s->workers=workers;s->kind=kind;s->bytes=q.need.scratch_bytes;s->supplied_pi=supplied_pi;
    for(auto& v:s->v){auto* raw=frame.allocate(sizeof(float_buffer),64);auto* data=frame.alloc<std::uint64_t>(q.limbs);
        v=::new(raw)float_buffer({data,q.limbs},bits{q.working});}
    {auto work=frame.subframe(q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment);
        real_arithmetic_prepare(s->arithmetic,q.arithmetic,work);}
    if(kind==ConstantKind::log2){
        if(!supplied_pi.supplied()){
            auto* pi=static_cast<std::byte*>(frame.allocate(q.pi.scratch_bytes,q.pi.scratch_alignment));
            s->pi=cold_constant_prepare(bits{q.working},workers,ConstantKind::pi,{pi,q.pi.scratch_bytes});}
        auto* agm=static_cast<std::byte*>(frame.allocate(q.agm.scratch_bytes,q.agm.scratch_alignment));
        s->agm=agm_program_prepare(bits{q.working},workers,{agm,q.agm.scratch_bytes});
    }
    s->marker=magic;return {s,storage.first(q.need.scratch_bytes)};
}
float_report cold_constant_execute(const ColdConstantPrepared& prepared,float_ref out,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(ConstantState),
        SBN3_FATAL_LIFETIME,"cold constant identity");
    auto& s=*static_cast<ConstantState*>(prepared.state);
    require(s.marker==magic&&s.bytes==prepared.storage.size()&&out.precision().count==s.precision&&ex.workers()>=s.workers,
        SBN3_FATAL_ARGUMENT,"cold constant program");
    check_output(out,{});const auto* dest=detail::float_access::begin(out);
    if(s.supplied_pi.supplied()&&!s.supplied_pi.pending())constant_bank_check_output(s.supplied_pi,out);
    require(!overlaps(prepared.storage.data(),prepared.storage.size(),dest->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"constant output alias");
    std::uint32_t idle=0;
    require(__atomic_compare_exchange_n(&s.active,&idle,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"constant concurrent use");
    Call call{&s,out,mode,{}};
    if(ex.scope)calculate(&call,ex.scope);else if(ex.team)sbn3_team_run(ex.team,calculate,&call);else serial_team_run(calculate,&call);
    __atomic_store_n(&s.active,0,__ATOMIC_RELEASE);return call.report;
}
}
