// Original libsbn code; LGPL-3.0-or-later.
#include "float/power.hpp"
#include "float/division.hpp"
#include "float/error.hpp"
#include "product/program.hpp"
#include "core/x86_64/word.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <new>

namespace sbn::v3::floating {
namespace {
using Wide=__int128_t;
constexpr std::uint64_t magic=UINT64_C(0x53424e33504f5749);
struct PowerState {
    std::uint64_t marker=0,magnitude=0;
    std::int64_t exponent=0;
    std::size_t precision=0,working=0,words=0,bytes=0,capacity=0,work_bytes=0;
    unsigned workers=1,exponent_bits=0;
    bool square=false,multiply=false,leaf=false,wide=false,reciprocate=false;
    std::uint32_t active=0;
    ProductProgramPlan square_plan{},multiply_plan{};
    ProductProgram square_program{},multiply_program{};
    FloatDivisionPrepared reciprocal{};
    float_buffer* reciprocal_result=nullptr;
    std::uint64_t *base=nullptr,*current=nullptr,*product=nullptr;
    void* work=nullptr;
    Wide base_top=0,current_top=0;
};
struct Quote {
    requirements need{};
    requirements reciprocal{};
    std::uint64_t magnitude=0;
    std::size_t working=0,words=0,capacity=0,work_bytes=0,tables_bytes=0;
    unsigned exponent_bits=0;
    bool square=false,multiply=false,leaf=false;
};
std::uint64_t absolute_exponent(std::int64_t exponent) noexcept {
    return exponent<0?std::uint64_t(0)-std::uint64_t(exponent):std::uint64_t(exponent);
}
Wide add_wide(Wide a,Wide b) noexcept {
    Wide out;require(!__builtin_add_overflow(a,b,&out),SBN3_FATAL_SIZE,"float power exponent sum");return out;
}
Wide multiply_wide(Wide a,Wide b) noexcept {
    Wide out;require(!__builtin_mul_overflow(a,b,&out),SBN3_FATAL_SIZE,"float power exponent product");return out;
}
std::int64_t narrow(Wide a) noexcept {
    require(a>=std::numeric_limits<std::int64_t>::min()&&a<=std::numeric_limits<std::int64_t>::max(),
            SBN3_FATAL_SIZE,"float power final exponent");return std::int64_t(a);
}
error_bound power_bound(Wide exponent) noexcept {
    if(exponent<std::numeric_limits<std::int64_t>::min())
        return error_bound::power_of_two(std::numeric_limits<std::int64_t>::min());
    if(exponent>std::numeric_limits<std::int64_t>::max())return error_bound::unbounded();
    return error_bound::power_of_two(std::int64_t(exponent));
}
bool power_two(float_view x,std::int64_t& exponent) noexcept {
    if(!x.finite()||x.magnitude().size()!=1||!std::has_single_bit(x.magnitude().data()[0]))return false;
    exponent=x.exponent2()+std::countr_zero(x.magnitude().data()[0]);return true;
}
float_report exact_power_two(float_ref out,Wide exponent,bool negative,rounding mode) noexcept {
    const auto top=add_wide(exponent,1);
    if(top>float_exponent_limit)return special(out,float_class::infinity,negative,float_flag::overflow|float_flag::rounded);
    if(top<-Wide(float_exponent_limit)){
        auto report=special(out,float_class::finite,negative,float_flag::underflow|float_flag::rounded);
        report.error=power_bound(exponent);return report;
    }
    const std::uint64_t one=1;
    return round_magnitude(out,integer_view{{&one,1}},narrow(exponent),negative,mode);
}
bool trivial(float_ref out,float_view x,std::int64_t exponent,rounding mode,float_report& report) noexcept {
    const auto magnitude=absolute_exponent(exponent);
    const bool negative=x.negative()&&(magnitude&1);
    if(!exponent){report=exact_power_two(out,0,false,mode);return true;}
    if(exponent==1){report=round_into(out,x,mode);return true;}
    if(x.is_nan()){report=special(out,float_class::quiet_nan,false);return true;}
    if(x.zero()){
        report=exponent<0?special(out,float_class::infinity,negative,float_flag::divide_by_zero):
            special(out,float_class::finite,negative);return true;
    }
    if(!x.finite()){
        report=exponent<0?special(out,float_class::finite,negative):special(out,float_class::infinity,negative);
        return true;
    }
    std::int64_t e=0;
    if(power_two(x,e)){
        report=exact_power_two(out,multiply_wide(Wide(e),Wide(exponent)),negative,mode);return true;
    }
    return false;
}
bool trivial_shape(float_view x,std::int64_t exponent) noexcept {
    std::int64_t ignored=0;
    return !exponent||exponent==1||!x.finite()||x.zero()||power_two(x,ignored);
}

bool product_plan(Quote& q,unsigned workers,bool square,ProductProgramPlan* destination) noexcept {
    if(q.leaf)return true;
    ProductProgramPlan temporary;auto& p=destination?*destination:temporary;
    sbn3_mul_options options{};options.workers=workers;options.borrow_output=1;
    if(square){
        sbn3_product_request request{};request.kind=SBN3_PRODUCT_SQR;request.a_limbs=q.words;
        sbn3_product_info info{};
        if(sbn3_product_query(&request,&options,&p.plan,&info)!=SBN3_SUPPORTED)return false;
        p.info=info.mul;
        if(product_program_describe(q.words,0,p)!=SBN3_SUPPORTED)return false;
    }else if(product_program_query(q.words,q.words,options,p)!=SBN3_SUPPORTED)return false;
    q.capacity=std::max(q.capacity,sbn3_mul_output_capacity(&p.info));
    q.work_bytes=std::max(q.work_bytes,p.info.workspace_bytes);
    q.need.scratch_alignment=std::max({q.need.scratch_alignment,p.info.workspace_alignment,p.info.output_alignment});
    std::size_t bytes=0;
    return add_size(p.prepared_bytes,256,bytes)&&add_size(q.tables_bytes,bytes,q.tables_bytes);
}
Quote quote(bits precision,std::int64_t exponent,unsigned workers,PowerState* state=nullptr,
            std::uint64_t wide_degree=0) noexcept {
    Quote q;q.need.scratch_alignment=128;
    if(!precision.count||precision.count>float_precision_limit||!workers||workers>32){
        q.need.status=support::unsupported;q.need.reason="float power precision/workers";return q;
    }
    q.need.output_limbs=(precision.count+63)/64;
    const bool reciprocate=!wide_degree&&exponent<0;
    q.magnitude=wide_degree?wide_degree:absolute_exponent(exponent);q.exponent_bits=unsigned(std::bit_width(q.magnitude));
    if(!wide_degree&&(!exponent||exponent==1)){q.need.scratch_bytes=sizeof(PowerState)+128;return q;}
    if(!add_size(precision.count,std::size_t(q.exponent_bits)+16,q.working)||q.working>float_precision_limit){
        q.need.status=support::unsupported;q.need.reason="float power working precision";return q;
    }
    q.words=(q.working+63)/64;q.square=q.magnitude>1;
    q.multiply=!std::has_single_bit(q.magnitude);q.leaf=q.words<=6;
    q.capacity=q.square?2*q.words:0;
    if((q.square&&!product_plan(q,workers,true,state?&state->square_plan:nullptr))||
       (q.multiply&&!product_plan(q,workers,false,state?&state->multiply_plan:nullptr))){
        q.need.status=support::unsupported;q.need.reason="float power product shape";return q;
    }
    if(reciprocate){
        q.reciprocal=division_program_requirements(bits{q.working},workers,SBN3_NEWTON_INVERSE);
        if(!q.reciprocal.supported()){
            q.need.status=q.reciprocal.status;q.need.reason="float power reciprocal shape";return q;
        }
        q.need.scratch_alignment=std::max(q.need.scratch_alignment,q.reciprocal.scratch_alignment);
    }
    std::size_t bytes=sizeof(PowerState),values=0;
    if(!mul_size(q.words*(2+unsigned(reciprocate))+q.capacity,8,values)||
       !add_size(bytes,values,bytes)||!add_size(bytes,q.tables_bytes,bytes)||
       !add_size(bytes,q.work_bytes,bytes)||!add_size(bytes,q.reciprocal.scratch_bytes,bytes)||
       !add_size(bytes,reciprocate?sizeof(float_buffer):0,bytes)||
       !add_size(bytes,12*q.need.scratch_alignment,bytes)){
        q.need.status=support::unsupported;q.need.reason="float power storage extent";return q;
    }
    q.need.scratch_bytes=bytes;return q;
}

std::uint64_t word_at(integer_view input,std::int64_t bit) noexcept {
    if(bit<0)return bit<=-64?0:input.data()[0]<<unsigned(-bit);
    const auto i=std::size_t(bit/64);const unsigned shift=unsigned(bit%64);
    return i<input.size()?(input.data()[i]>>shift)|
        (shift&&i+1<input.size()?input.data()[i+1]<<(64-shift):0):0;
}
// Retain exactly q leading significant bits (with virtual low zero padding).
// No omitted low range is scanned; its error is bounded analytically.
void normalized_prefix(std::uint64_t* out,std::size_t words,std::size_t q,integer_view input) noexcept {
    const auto cut=std::int64_t(magnitude_bits(input))-std::int64_t(q);
    for(std::size_t j=0;j<words;++j)out[j]=word_at(input,std::int64_t(64*j)+cut);
}
void product(PowerState& s,bool square,sbn3_team_scope* scope) noexcept {
    if(s.leaf)mul_basecase_assumed(s.product,s.current,s.words,square?s.current:s.base,s.words);
    else{
        auto work=Frame::external(s.work,s.work_bytes);
        const auto& program=square?s.square_program:s.multiply_program;
        product_program_execute(program,work,scope,{s.current,s.words},square?sbn3_const_limbs{}:
            sbn3_const_limbs{s.base,s.words},{s.product,s.capacity});
    }
    const integer_view value{{s.product,2*s.words}};
    const auto length=magnitude_bits(value);
    require(length==2*s.working||length+1==2*s.working,SBN3_FATAL_MATH,"float power product normalization");
    const auto top=add_wide(s.current_top,square?s.current_top:s.base_top);
    s.current_top=add_wide(top,Wide(length)-2*Wide(s.working));
    normalized_prefix(s.current,s.words,s.working,value);
}
void action(void* opaque,sbn3_team_scope* scope){
    auto& s=*static_cast<PowerState*>(opaque);
    if(s.reciprocate){
        const auto input=float_view::from_magnitude({s.base,s.words},-std::int64_t(s.working));
        const auto report=division_program_execute(s.reciprocal,s.reciprocal_result->ref(),{},input,
                                                   execution{{},nullptr,scope},rounding::nearest_even);
        const auto value=s.reciprocal_result->view();
        require(value.finite()&&!value.zero()&&!value.negative()&&report.error.kind!=error_kind::unbounded,
                SBN3_FATAL_MATH,"float power normalized reciprocal");
        const auto top=value.exponent2()+std::int64_t(magnitude_bits(value.magnitude()));
        // The reciprocal service's <1-ulp bound, plus the original prefix,
        // is covered by the 16*2^-q base-relative allowance in the proof.
        require(report.error.exact()||report.error.exponent2<=top-std::int64_t(s.working),
                SBN3_FATAL_MATH,"float power reciprocal accuracy");
        normalized_prefix(s.base,s.words,s.working,value.magnitude());
        s.base_top=add_wide(Wide(top),-s.base_top);
    }
    std::memcpy(s.current,s.base,s.words*8);s.current_top=s.base_top;
    for(unsigned bit=s.exponent_bits-1;bit-->0;){
        product(s,true,scope);
        if((s.magnitude>>bit)&1)product(s,false,scope);
    }
}
float_report finish(const PowerState& s,float_ref out,bool negative,rounding mode) noexcept {
    const auto top=s.current_top;
    if(top>Wide(float_exponent_limit)+2)
        return special(out,float_class::infinity,negative,float_flag::overflow|float_flag::rounded);
    if(top<-Wide(float_exponent_limit)-2){
        auto report=special(out,float_class::finite,negative,float_flag::underflow|float_flag::rounded);
        report.error=power_bound(add_wide(top,1));return report;
    }
    auto report=round_magnitude(out,integer_view{{s.current,s.words}},
        narrow(add_wide(top,-Wide(s.working))),negative,mode);
    if(has_flag(report.flags,float_flag::underflow))report.error=power_bound(add_wide(top,1));
    else if(report.error.kind!=error_kind::unbounded){
        const auto error_exponent=add_wide(top,Wide(7+s.exponent_bits)-Wide(s.working));
        report.error=ErrorMagnitude::from(report.error).add(ErrorMagnitude::from(power_bound(error_exponent))).publish();
    }
    return report;
}
void check_storage(std::span<std::byte> storage,float_ref out,float_view input) noexcept {
    const auto* value=detail::float_access::begin(out);
    require(!overlaps(storage.data(),storage.size(),input.magnitude().data(),input.magnitude().size()*8)&&
            !overlaps(storage.data(),storage.size(),value->data,((out.precision().count+63)/64)*8),
            SBN3_FATAL_ARGUMENT,"float power value/storage alias");
}
struct Use {
    PowerState& state;
    explicit Use(PowerState& s):state(s){std::uint32_t expected=0;
        require(__atomic_compare_exchange_n(&state.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),
                SBN3_FATAL_LIFETIME,"concurrent prepared float power");}
    ~Use(){__atomic_store_n(&state.active,0,__ATOMIC_RELEASE);}
};
}

requirements power_program_requirements(bits precision,std::int64_t exponent,unsigned workers) noexcept {
    return quote(precision,exponent,workers).need;
}
static FloatPowerPrepared prepare_power(bits precision,std::int64_t exponent,unsigned workers,
                                        std::span<std::byte> storage,std::uint64_t wide_degree) noexcept {
    require(storage.size()>=sizeof(PowerState)+128&&!(std::uintptr_t(storage.data())&127),
            SBN3_FATAL_WORKSPACE,"float power program metadata");
    auto frame=Frame::external(storage.data(),storage.size());
    auto* state=::new(frame.allocate(sizeof(PowerState),128))PowerState{};
    const auto q=quote(precision,exponent,workers,state,wide_degree);
    require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
            SBN3_FATAL_WORKSPACE,"float power program storage",q.need.scratch_bytes,storage.size());
    state->precision=precision.count;state->exponent=exponent;state->workers=workers;
    state->wide=wide_degree!=0;state->reciprocate=!wide_degree&&exponent<0;
    state->magnitude=q.magnitude;state->exponent_bits=q.exponent_bits;state->working=q.working;
    state->words=q.words;state->bytes=q.need.scratch_bytes;state->capacity=q.capacity;state->work_bytes=q.work_bytes;
    state->square=q.square;state->multiply=q.multiply;state->leaf=q.leaf;
    if(wide_degree||(exponent&&exponent!=1)){
        if(q.square&&!q.leaf){auto tables=frame.subframe(state->square_plan.prepared_bytes+128,128);
            state->square_program=product_program_prepare(state->square_plan,tables);}
        if(q.multiply&&!q.leaf){auto tables=frame.subframe(state->multiply_plan.prepared_bytes+128,128);
            state->multiply_program=product_program_prepare(state->multiply_plan,tables);}
        state->base=frame.alloc<std::uint64_t>(q.words);state->current=frame.alloc<std::uint64_t>(q.words);
        state->product=static_cast<std::uint64_t*>(frame.allocate(q.capacity*8,q.need.scratch_alignment));
        state->work=frame.allocate(q.work_bytes,q.need.scratch_alignment);
        if(state->reciprocate){
            auto* result=frame.alloc<std::uint64_t>(q.words);
            state->reciprocal_result=::new(frame.allocate(sizeof(float_buffer),alignof(float_buffer)))
                float_buffer({result,q.words},bits{q.working});
            auto* region=static_cast<std::byte*>(frame.allocate(q.reciprocal.scratch_bytes,q.reciprocal.scratch_alignment));
            state->reciprocal=division_program_prepare(bits{q.working},workers,SBN3_NEWTON_INVERSE,
                {region,q.reciprocal.scratch_bytes});
        }
    }
    state->marker=magic;return {state,storage.first(q.need.scratch_bytes)};
}
FloatPowerPrepared power_program_prepare(bits precision,std::int64_t exponent,unsigned workers,
                                        std::span<std::byte> storage) noexcept {
    return prepare_power(precision,exponent,workers,storage,0);
}
float_report power_program_execute(const FloatPowerPrepared& prepared,float_ref out,float_view input,
                                  execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(PowerState),
            SBN3_FATAL_LIFETIME,"prepared float power state");
    auto& state=*static_cast<PowerState*>(prepared.state);
    require(state.marker==magic&&!state.wide&&state.bytes==prepared.storage.size()&&out.precision().count==state.precision&&
            ex.workers()>=state.workers,SBN3_FATAL_ARGUMENT,"prepared float power identity/shape");
    validate(input);check_output(out,input);check_storage(prepared.storage,out,input);Use exclusive(state);
    float_report report;if(trivial(out,input,state.exponent,mode,report))return report;
    const bool negative=input.negative()&&(state.magnitude&1);
    const auto input_top=input.exponent2()+std::int64_t(magnitude_bits(input.magnitude()));
    normalized_prefix(state.base,state.words,state.working,input.magnitude());
    state.base_top=input_top;
    if(ex.scope)action(&state,ex.scope);else if(ex.team)sbn3_team_run(ex.team,action,&state);else serial_team_run(action,&state);
    return finish(state,out,negative,mode);
}
requirements wide_power_program_requirements(bits precision,std::uint64_t degree,unsigned workers) noexcept {
    if(!degree)return {support::unsupported,0,128,0,"wide power zero degree"};
    return quote(precision,0,workers,nullptr,degree).need;
}
WidePositivePowerPrepared wide_power_program_prepare(bits precision,std::uint64_t degree,unsigned workers,
                                                      std::span<std::byte> storage) noexcept {
    require(degree,SBN3_FATAL_ARGUMENT,"wide power zero degree");
    const auto p=prepare_power(precision,0,workers,storage,degree);return {p.state,p.storage};
}
WidePositivePowerValue wide_power_program_execute(const WidePositivePowerPrepared& prepared,float_view input,
                                                  execution ex) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(PowerState),
            SBN3_FATAL_LIFETIME,"prepared wide power state");
    auto& state=*static_cast<PowerState*>(prepared.state);
    require(state.marker==magic&&state.wide&&state.bytes==prepared.storage.size()&&ex.workers()>=state.workers,
            SBN3_FATAL_ARGUMENT,"prepared wide power identity/shape");
    validate(input);
    require(input.finite()&&!input.zero()&&!input.negative(),SBN3_FATAL_ARGUMENT,"wide power positive input");
    require(!overlaps(prepared.storage.data(),prepared.storage.size(),input.magnitude().data(),input.magnitude().size()*8),
            SBN3_FATAL_ARGUMENT,"wide power input/storage alias");
    Use exclusive(state);
    normalized_prefix(state.base,state.words,state.working,input.magnitude());
    state.base_top=input.exponent2()+std::int64_t(magnitude_bits(input.magnitude()));
    if(ex.scope)action(&state,ex.scope);else if(ex.team)sbn3_team_run(ex.team,action,&state);else serial_team_run(action,&state);
    // Every prefix/product is downward. Weighted logarithmic deficit is
    // <8*N*2^-q, hence (P-center)/center <16*N*2^-q.
    return {float_view::from_magnitude({state.current,state.words},-std::int64_t(state.working)),
            state.current_top,ErrorMagnitude::power_of_two(std::int64_t(state.exponent_bits)+4-std::int64_t(state.working))};
}
}

namespace sbn {
requirements powi_requirements(float_ref out,float_view input,std::int64_t exponent,unsigned workers) noexcept {
    const auto p=out.precision().count;
    if(!p||p>float_precision_limit||!workers||workers>32)return {support::unsupported,0,128,0,"float power precision/workers"};
    v3::floating::validate(input);
    const auto words=(p+63)/64;
    if(out.capacity()<words)return {support::capacity,0,128,words,"float power output capacity"};
    if(v3::floating::trivial_shape(input,exponent))return {support::yes,0,128,words,""};
    return v3::floating::power_program_requirements(out.precision(),exponent,workers);
}
float_report powi_into(float_ref out,float_view input,std::int64_t exponent,execution ex,rounding mode) noexcept {
    using namespace v3::floating;
    validate(input);check_output(out,input);
    float_report report;if(trivial(out,input,exponent,mode,report))return report;
    check_storage(ex.scratch,out,input);
    const auto prepared=power_program_prepare(out.precision(),exponent,ex.workers(),ex.scratch);
    return power_program_execute(prepared,out,input,ex,mode);
}
}
