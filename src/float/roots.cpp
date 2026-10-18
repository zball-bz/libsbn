// Original libsbn code; LGPL-3.0-or-later.
#include "float/roots.hpp"
#include "float/diagnostics.hpp"
#include "algorithms/rsqrt.hpp"
#include "algorithms/newton_limits.hpp"
#include "product/program.hpp"
#include "core/x86_64/word.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <new>

namespace sbn::v3::floating {
namespace {
// All lengths are fractional limbs unless explicitly named words/bytes.
// See docs/bigfloat-roots-design.md for the <2 B^-n kernel bound.
struct Stage {
    ProductProgramPlan product{};
    ProductProgram prepared{};
    std::size_t q=0,capacity=0;
    bool leaf=false,fused_sqrt=false;
};
struct Quote {
    requirements need{};
    std::size_t n=0,target=0,count=0,q=0,capacity=10,phase=0,tables=0,alignment=128;
    bool trivial=false;
};
struct Run {
    Stage* stages=nullptr;
    Quote quote{};
    std::uint64_t *a=nullptr,*y=nullptr,*z=nullptr,*rho=nullptr,*product=nullptr;
    void* phase=nullptr;
    std::size_t phase_bytes=0;
};
struct RootState {
    std::uint64_t marker=0;
    std::size_t precision=0;
    unsigned workers=0;
    std::uint32_t active=0;
    RootKind kind=RootKind::square_root;
    Run run{};
};
constexpr std::uint64_t root_magic=UINT64_C(0x53424e33524f4f54);
std::size_t predecessor(std::size_t n) noexcept { return n/2+1; }

bool power_four(float_view x,std::int64_t& exponent) noexcept {
    if(!x.finite()||x.negative()||x.magnitude().size()!=1)return false;
    const auto a=x.magnitude().data()[0];
    if(!std::has_single_bit(a))return false;
    const auto e=x.exponent2()+std::int64_t(std::countr_zero(a));
    if(e%2)return false;
    exponent=e/2;
    return true;
}
Quote program_shape(bits precision,unsigned workers,RootKind kind) noexcept {
    Quote p;
    const auto bits=precision.count;
    if(!bits||bits>float_precision_limit||!workers||workers>32||
       (kind!=RootKind::square_root&&kind!=RootKind::reciprocal_square_root)){
        p.need={support::unsupported,0,128,0,"float root precision/workers"};return p;
    }
    p.need.output_limbs=(bits+63)/64;
    // Two complete guard limbs keep the approximation error well below the
    // final mantissa-rounding error, including non-limb-aligned precisions.
    p.n=p.need.output_limbs+2;
    if(p.n>newton_limits::precision_words-4){p.need.status=support::unsupported;p.need.reason="float root product domain";return p;}
    p.q=p.n+2;
    p.target=kind==RootKind::square_root?predecessor(p.n):p.n;
    for(auto n=p.target;n>2;n=predecessor(n))++p.count;
    if(kind==RootKind::square_root)++p.count;
    return p;
}
Quote shape(float_ref out,float_view x,unsigned workers,RootKind kind) noexcept {
    auto p=program_shape(out.precision(),workers,kind);
    // A trivial root needs no product domain at all. Other invalid program
    // shapes have n==0 and retain their ordinary unsupported result.
    if(!p.need.supported()&&!p.n)return p;
    if(x.magnitude().size()>float_precision_limit/64){
        p.need={support::unsupported,0,128,0,"float root input length"};return p;
    }
    if(x.finite()&&!x.zero()){
        const __int128 top=__int128(x.exponent2())+magnitude_bits(x.magnitude());
        if(top<-float_exponent_limit||top>float_exponent_limit){
            p.need={support::unsupported,0,128,0,"float root input exponent"};return p;
        }
    }
    if(out.capacity()<p.need.output_limbs){p.need.status=support::capacity;p.need.reason="float root output capacity";return p;}
    std::int64_t e=0;
    p.trivial=!x.finite()||x.zero()||x.negative()||power_four(x,e);
    if(p.trivial){p.need.status=support::yes;p.need.reason="";}
    return p;
}

// Query keeps only this one temporary descriptor on the stack. Execution
// writes every descriptor into its caller-owned metadata before the seed.
bool compile_stage(Quote& quote,Stage* destination,std::size_t q,unsigned workers,bool fused) noexcept {
    const auto words=q+1;
    if(destination){destination->q=q;destination->fused_sqrt=fused;destination->leaf=words<=6;}
    std::size_t capacity=2*words;
    if(words>6){
        ProductProgramPlan temporary;
        auto& p=destination?destination->product:temporary;
        sbn3_mul_options options{};options.workers=workers;options.borrow_output=1;
        if(product_program_query(words,words,options,p)!=SBN3_SUPPORTED)return false;
        capacity=sbn3_mul_output_capacity(&p.info);
        const auto alignment=std::max<std::size_t>(128,std::max(p.info.workspace_alignment,p.info.output_alignment));
        std::size_t tables=0;
        if(!add_size(p.prepared_bytes,256,tables)||!add_size(quote.tables,tables,quote.tables))return false;
        quote.phase=std::max(quote.phase,p.info.workspace_bytes);
        quote.alignment=std::max(quote.alignment,alignment);
    }
    quote.capacity=std::max(quote.capacity,capacity);
    if(destination)destination->capacity=capacity;
    return true;
}
bool compile(Quote& p,Stage* stages,unsigned workers,RootKind kind) noexcept {
    std::size_t index=p.count-(kind==RootKind::square_root);
    for(auto n=p.target;n>2;n=predecessor(n)){
        --index;
        if(!compile_stage(p,stages?stages+index:nullptr,n+2,workers,false))return false;
    }
    if(kind==RootKind::square_root&&!compile_stage(p,stages?stages+p.count-1:nullptr,p.n+2,workers,true))return false;
    std::size_t bytes=0,part=0;
    if(!mul_size(p.count,sizeof(Stage),bytes)||!add_size(bytes,sizeof(RootState)+256,bytes)||
       !add_size(bytes,p.tables,bytes)||
       !mul_size(4*(p.q+1),sizeof(std::uint64_t),part)||!add_size(bytes,part,bytes)||
       !mul_size(p.capacity,sizeof(std::uint64_t),part)||!add_size(bytes,part,bytes)||
       !add_size(bytes,7*p.alignment,bytes)||!add_size(bytes,p.phase,bytes))return false;
    p.need.scratch_bytes=bytes;p.need.scratch_alignment=p.alignment;
    return true;
}

std::uint64_t word_at(integer_view x,std::int64_t bit) noexcept {
    if(bit<0)return bit<=-64?0:x.data()[0]<<unsigned(-bit);
    const auto i=std::size_t(bit/64);const auto shift=unsigned(bit%64);
    if(i>=x.size())return 0;
    return (x.data()[i]>>shift)|(shift&&i+1<x.size()?x.data()[i+1]<<(64-shift):0);
}
void scaled_copy(std::uint64_t* out,std::size_t words,integer_view x,std::int64_t shift) noexcept {
    for(std::size_t i=0;i<words;++i)out[i]=word_at(x,std::int64_t(64*i)-shift);
}
void extend(std::uint64_t* y,std::size_t old,std::size_t q) noexcept {
    require(q>=old,SBN3_FATAL_MATH,"root precision extension");
    const auto shift=q-old;
    std::memmove(y+shift,y,(old+1)*8);
    std::fill_n(y,shift,0);
}
void truncate(std::uint64_t* y,std::size_t q,std::size_t n) noexcept {
    require(q>=n,SBN3_FATAL_MATH,"root precision truncation");
    std::memmove(y,y+q-n,(n+1)*8);
}
void slice(std::uint64_t* out,const std::uint64_t* product,std::size_t q,bool half=false) noexcept {
    for(std::size_t i=0;i<=q;++i){
        out[i]=product[q+i];
        if(half)out[i]=(out[i]>>1)|(product[q+i+1]<<63);
    }
}
// t is nonnegative. Replace it by |1-t|, returning the sign of 1-t.
bool residual(std::uint64_t* t,std::size_t q) noexcept {
    bool negative=t[q]>1;
    if(t[q]==1)for(std::size_t i=0;i<q;++i)negative|=t[i]!=0;
    unsigned borrow=0;
    for(std::size_t i=0;i<=q;++i){
        const std::uint64_t one=i==q?1:0;
        const auto a=negative?t[i]:one,b=negative?one:t[i];
        const __uint128_t sub=__uint128_t(b)+borrow;
        t[i]=a-std::uint64_t(sub);borrow=__uint128_t(a)<sub;
    }
    require(!borrow,SBN3_FATAL_MATH,"root residual sign");
    return negative;
}
void combine(std::uint64_t* out,const std::uint64_t* a,const std::uint64_t* correction,
             std::size_t words,bool subtract) noexcept {
    unsigned carry=0;
    for(std::size_t i=0;i<words;++i){
        if(subtract){const __uint128_t v=__uint128_t(correction[i])+carry;
            const auto old=a[i];out[i]=old-std::uint64_t(v);carry=__uint128_t(old)<v;
        }else{const __uint128_t v=__uint128_t(a[i])+correction[i]+carry;out[i]=std::uint64_t(v);carry=unsigned(v>>64);}
    }
    require(!carry,SBN3_FATAL_MATH,"root correction range");
}
struct Products {
    const ProductProgram* program;
    Frame* work;
    sbn3_team_scope* scope;
    std::uint64_t* output;
    std::size_t words,capacity;
    void operator()(const std::uint64_t* a,const std::uint64_t* b) const noexcept {
        if(program)product_program_execute(*program,*work,scope,{a,words},{b,words},{output,capacity});
        else mul_basecase_assumed(output,a,words,b,words);
    }
};
void refine(const std::uint64_t* a,std::uint64_t* y,std::uint64_t* z,std::uint64_t* rho,
            std::size_t q,const Products& products,bool fused) noexcept {
    if(fused){
        products(a,y);slice(z,products.output,q);       // h = a*y
        products(z,y);slice(rho,products.output,q);     // h*y
        const bool negative=residual(rho,q);
        products(z,rho);slice(y,products.output,q,true);// y is dead: correction
        combine(y,z,y,q+1,negative);
    }else{
        products(y,y);slice(z,products.output,q);
        products(a,z);slice(rho,products.output,q);
        const bool negative=residual(rho,q);
        products(y,rho);slice(z,products.output,q,true);
        combine(y,y,z,q+1,negative);
    }
}
void seed(const std::uint64_t* a,std::size_t aq,std::uint64_t* y,std::uint64_t* z,
          std::uint64_t* rho,std::uint64_t* product) noexcept {
    const auto leading=(a[aq]<<62)|(a[aq-1]>>2);
    require(a[aq]>=1&&a[aq]<=3,SBN3_FATAL_MATH,"root normalized radicand");
    alignas(64)std::uint64_t r[3];rsqrt_seed(r,leading,2);
    scaled_copy(y,5,integer_view{{r,2}},159); // 2^31 R/B^2 on the B^-4 grid
    const Products products{nullptr,nullptr,nullptr,product,5,10};
    refine(a+aq-4,y,z,rho,4,products,false);
    refine(a+aq-4,y,z,rho,4,products,false);
    truncate(y,4,2);
}
void calculate(void* opaque,sbn3_team_scope* scope) {
    auto& r=*static_cast<Run*>(opaque);
    seed(r.a,r.quote.q,r.y,r.z,r.rho,r.product);
    std::size_t current=2;
    for(std::size_t i=0;i<r.quote.count;++i){
        const auto& s=r.stages[i];const auto q=s.q;
        extend(r.y,current,q);
        if(s.leaf){
            const Products products{nullptr,nullptr,scope,r.product,q+1,s.capacity};
            refine(r.a+r.quote.q-q,r.y,r.z,r.rho,q,products,s.fused_sqrt);
        }else{
            auto work=Frame::external(r.phase,r.phase_bytes);
            const Products products{&s.prepared,&work,scope,r.product,q+1,s.capacity};
            refine(r.a+r.quote.q-q,r.y,r.z,r.rho,q,products,s.fused_sqrt);
        }
        current=q-2;truncate(r.y,q,current);
        require(r.y[current]<=2,SBN3_FATAL_MATH,"root iterate normalization");
    }
    require(current==r.quote.n,SBN3_FATAL_MATH,"root completed ladder");
}
float_report add_root_error(float_report report,std::int64_t exponent) noexcept {
    if(report.error.kind==error_kind::unbounded)return report;
    if(report.error.kind==error_kind::finite)exponent=std::max(exponent,report.error.exponent2)+1;
    report.error=error_bound::power_of_two(exponent);
    return report;
}
bool trivial(float_ref out,float_view x,rounding mode,RootKind kind,float_report& report) noexcept {
    if(x.is_nan()){report=special(out,float_class::quiet_nan,false);return true;}
    if(x.zero()){
        report=kind==RootKind::square_root?special(out,float_class::finite,x.negative()):
            special(out,float_class::infinity,x.negative(),float_flag::divide_by_zero);
        return true;
    }
    if(x.negative()){report=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
    if(!x.finite()){
        report=kind==RootKind::square_root?special(out,float_class::infinity,false):special(out,float_class::finite,false);
        return true;
    }
    std::int64_t exponent=0;
    if(power_four(x,exponent)){
        const std::uint64_t one=1;
        report=round_magnitude(out,integer_view{{&one,1}},kind==RootKind::square_root?exponent:-exponent,false,mode);
        return true;
    }
    return false;
}
class RootUse {
    RootState& state_;
public:
    explicit RootUse(RootState& state) noexcept:state_(state){
        std::uint32_t expected=0;
        require(__atomic_compare_exchange_n(&state_.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),
                SBN3_FATAL_LIFETIME,"concurrent prepared root");
    }
    ~RootUse(){__atomic_store_n(&state_.active,0,__ATOMIC_RELEASE);}
    RootUse(const RootUse&)=delete;
    RootUse&operator=(const RootUse&)=delete;
};
}

FloatDiagnostic root_diagnostic(float_ref out,float_view x,bool reciprocal,unsigned workers) noexcept {
    const auto kind=reciprocal?RootKind::reciprocal_square_root:RootKind::square_root;
    auto q=shape(out,x,workers,kind);FloatDiagnostic d;d.recipe=q.trivial?"analytic":"geometric_newton";
    if(!q.trivial&&q.need.supported()&&!compile(q,nullptr,workers,kind))q.need.status=support::unsupported;
    d.need=q.need;d.working_bits=q.n*64;d.metadata_bytes=sizeof(RootState)+q.count*sizeof(Stage);
    d.table_bytes=q.tables;d.constant_value_bytes=0;d.constant_work_bytes=0;return d;
}
requirements root_program_requirements(bits precision,unsigned workers,RootKind kind) noexcept {
    auto p=program_shape(precision,workers,kind);
    if(!p.need.supported())return p.need;
    if(!compile(p,nullptr,workers,kind))return {support::unsupported,0,128,p.need.output_limbs,"float root product shape"};
    return p.need;
}
FloatRootPrepared root_program_prepare(bits precision,unsigned workers,RootKind kind,std::span<std::byte> storage) noexcept {
    auto p=program_shape(precision,workers,kind);
    require(p.need.supported(),SBN3_FATAL_ARGUMENT,p.need.reason);
    require(storage.size()>=sizeof(RootState)+p.count*sizeof(Stage)+256&&
            !(std::uintptr_t(storage.data())&127),SBN3_FATAL_WORKSPACE,"float root plan storage");
    auto frame=Frame::external(storage.data(),storage.size());
    auto* state=::new(frame.allocate(sizeof(RootState),128))RootState{};
    auto* stages=frame.alloc<Stage>(p.count);
    for(std::size_t i=0;i<p.count;++i)::new(stages+i)Stage{};
    require(compile(p,stages,workers,kind),SBN3_FATAL_ARGUMENT,"float root compile");
    require(storage.size()>=p.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(p.alignment-1)),
            SBN3_FATAL_WORKSPACE,"float root scratch",p.need.scratch_bytes,storage.size());
    for(std::size_t i=0;i<p.count;++i)if(!stages[i].leaf){
        auto tables=frame.subframe(stages[i].product.prepared_bytes+128,128);
        stages[i].prepared=product_program_prepare(stages[i].product,tables);
    }
    auto value=[&](std::size_t words){return static_cast<std::uint64_t*>(frame.allocate(words*8,p.alignment));};
    auto* a=value(p.q+1);auto* y=value(p.q+1);auto* z=value(p.q+1);auto* rho=value(p.q+1);
    auto* product=value(p.capacity);auto* phase=frame.allocate(p.phase,p.alignment);
    state->precision=precision.count;state->workers=workers;state->kind=kind;
    state->run={stages,p,a,y,z,rho,product,phase,p.phase};
    state->marker=root_magic;
    return {state,storage.first(p.need.scratch_bytes)};
}
float_report root_program_execute(const FloatRootPrepared& prepared,float_ref out,float_view x,
                                  execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(RootState)&&
            !(std::uintptr_t(prepared.state)&127),SBN3_FATAL_LIFETIME,"prepared root storage");
    auto& state=*static_cast<RootState*>(prepared.state);
    require(state.marker==root_magic&&state.run.quote.need.scratch_bytes==prepared.storage.size(),
            SBN3_FATAL_LIFETIME,"prepared root identity");
    require(out.precision().count==state.precision&&ex.workers()>=state.workers,
            SBN3_FATAL_ARGUMENT,"prepared root precision/workers");
    validate(x);check_output(out,x);
    const auto* output=detail::float_access::begin(out);
    require(!overlaps(prepared.storage.data(),prepared.storage.size(),x.magnitude().data(),x.magnitude().size()*8)&&
            !overlaps(prepared.storage.data(),prepared.storage.size(),output->data,out.capacity()*8),
            SBN3_FATAL_ARGUMENT,"prepared root value/storage alias");
    RootUse exclusive(state);
    float_report report;
    if(trivial(out,x,mode,state.kind,report))return report;
    auto& run=state.run;const auto& p=run.quote;
    // Exact-base output/input aliases are safe: stage every demanded source
    // bit before the output emitter writes any caller-owned value.
    const auto length=std::int64_t(magnitude_bits(x.magnitude()));
    const auto top=x.exponent2()+length-1;
    const auto exponent=top/2-std::int64_t(top%2<0);
    const auto shift=std::int64_t(64*p.q)+x.exponent2()-2*exponent;
    scaled_copy(run.a,p.q+1,x.magnitude(),shift);
    if(ex.scope)calculate(&run,ex.scope);
    else if(ex.team)sbn3_team_run(ex.team,calculate,&run);
    else serial_team_run(calculate,&run);
    const auto scale=(state.kind==RootKind::square_root?exponent:-exponent)-std::int64_t(64*p.n);
    report=round_magnitude(out,integer_view{{run.y,p.n+1}},scale,false,mode);
    return add_root_error(report,scale+1);
}
requirements root_requirements(float_ref out,float_view x,unsigned workers,RootKind kind) noexcept {
    auto p=shape(out,x,workers,kind);
    if(!p.need.supported()||p.trivial)return p.need;
    if(!compile(p,nullptr,workers,kind))return {support::unsupported,0,128,p.need.output_limbs,"float root product shape"};
    return p.need;
}
float_report root_into(float_ref out,float_view x,execution ex,rounding mode,RootKind kind) noexcept {
    validate(x);check_output(out,x);
    float_report report;
    if(trivial(out,x,mode,kind,report))return report;
    const auto* output=detail::float_access::begin(out);
    require(!overlaps(ex.scratch.data(),ex.scratch.size(),x.magnitude().data(),x.magnitude().size()*8)&&
            !overlaps(ex.scratch.data(),ex.scratch.size(),output->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"float root scratch alias");
    const auto prepared=root_program_prepare(out.precision(),ex.workers(),kind,ex.scratch);
    return root_program_execute(prepared,out,x,ex,mode);
}
}

namespace sbn {
requirements sqrt_requirements(float_ref out,float_view x,unsigned workers) noexcept {
    return v3::floating::root_requirements(out,x,workers,v3::floating::RootKind::square_root);
}
requirements rsqrt_requirements(float_ref out,float_view x,unsigned workers) noexcept {
    return v3::floating::root_requirements(out,x,workers,v3::floating::RootKind::reciprocal_square_root);
}
float_report sqrt_into(float_ref out,float_view x,execution ex,rounding mode) noexcept {
    return v3::floating::root_into(out,x,ex,mode,v3::floating::RootKind::square_root);
}
float_report rsqrt_into(float_ref out,float_view x,execution ex,rounding mode) noexcept {
    return v3::floating::root_into(out,x,ex,mode,v3::floating::RootKind::reciprocal_square_root);
}
}
