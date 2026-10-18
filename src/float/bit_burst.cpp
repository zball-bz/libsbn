// Original libsbn code; LGPL-3.0-or-later.
#include "float/bit_burst.hpp"
#include "float/error.hpp"
#include "algorithms/normalized_newton.hpp"
#include "series/arithmetic.hpp"
#include "series/scaled_ratio.hpp"
#include "product/program.hpp"
#include "core/x86_64/word.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <new>

namespace sbn::v3::floating {
namespace {
using Wide=__int128_t;
constexpr std::uint64_t magic=UINT64_C(0x53424e3342425552);
constexpr std::size_t maximum_slices=64,table_capacity=512,fixed_banks=12;
enum Bank : unsigned {accumulator,cosine,increment,slice_cosine,temp0,temp1,temp2,temp3,one,two,argument,sinc};
struct ProductShape {
    ProductProgramPlan plan{};
    std::size_t words=0,capacity=0,phase_bytes=0,alignment=128,local_bytes=0;
    bool leaf=true,square=false;
};
struct Facts {
    std::size_t a=0,b=0,terms=0,factor_bits=0,leaf_limbs=0;
    bool trigonometric=false;
};
struct SliceShape {Facts facts{};series::FiniteFrameInfo tree{};std::size_t argument_words=0;};
struct Slice {Facts facts{};series::FiniteFramePrepared tree{};ProductShape square{};};
struct Quote {
    BitBurstLayout layout{};NormalizedNewtonInfo ratio{};
    ProductShape product{},square{};
    std::array<SliceShape,maximum_slices> slices{};
    std::size_t directory=0,argument_words=0,squared_words=0,raw_words=0,root_words[3]{};
};
struct State {
    std::uint64_t marker=0;std::uint32_t active=0;
    BitBurstLayout layout{};BitBurstStats stats{};BitBurstKind kind=BitBurstKind::exp;
    unsigned workers=1;std::size_t fractional=0,words=0,bytes=0;
    Slice* slices=nullptr;ProductShape product_shape{},square_shape{};ProductProgram product{},square{};
    NormalizedNewtonPrepared ratio{};
    ProductTableRef* directory=nullptr;std::size_t directory_count=0,directory_capacity=0;
    sbn3_int values[fixed_banks]{};sbn3_series_values root{};
    std::uint64_t *argument=nullptr,*squared=nullptr,*raw=nullptr,*ratio_a=nullptr,*ratio_d=nullptr,*ratio_q=nullptr;
    std::size_t argument_capacity=0,squared_capacity=0,raw_capacity=0;
    std::span<std::byte> phase{};
};
std::int64_t narrow(Wide x) noexcept {require(x>=INT64_MIN&&x<=INT64_MAX,SBN3_FATAL_SIZE,"bit-burst exponent");return std::int64_t(x);}
std::size_t lg(std::size_t x) noexcept{return x<=1?0:std::bit_width(x-1);}
bool append(std::size_t& cursor,std::size_t bytes,std::size_t alignment) noexcept {
    if(!bytes)return true;std::size_t at=0;return align_size(cursor,alignment,at)&&add_size(at,bytes,cursor);
}
bool append_words(std::size_t& cursor,std::size_t n,std::size_t alignment=64) noexcept {
    std::size_t bytes=0;return mul_size(n,8,bytes)&&append(cursor,bytes,alignment);
}
bool product_shape(std::size_t words,bool square,unsigned workers,ProductShape& p) noexcept {
    p={};p.words=words;p.square=square;p.leaf=words<=6;p.capacity=2*words;
    if(p.leaf)return true;
    sbn3_mul_options options{};options.workers=workers;options.borrow_output=1;
    if(square){sbn3_product_request request{};request.kind=SBN3_PRODUCT_SQR;request.a_limbs=words;sbn3_product_info info{};
        if(sbn3_product_query(&request,&options,&p.plan.plan,&info)!=SBN3_SUPPORTED)return false;
        p.plan.info=info.mul;if(product_program_describe(words,0,p.plan)!=SBN3_SUPPORTED)return false;
    }else if(product_program_query(words,words,options,p.plan)!=SBN3_SUPPORTED)return false;
    p.capacity=sbn3_mul_output_capacity(&p.plan.info);
    p.alignment=std::max({std::size_t(128),p.plan.info.workspace_alignment,p.plan.info.output_alignment,p.plan.tables.alignment});
    if(!add_size(p.plan.local_bytes,128,p.local_bytes))return false;
    std::size_t tables=0;
    if(!add_size(p.plan.prepared_bytes,128,tables)||!append(p.phase_bytes,tables,128)||
       !append(p.phase_bytes,p.plan.info.workspace_bytes,p.plan.info.workspace_alignment))return false;
    return true;
}
sbn3_query_result slice_bounds(const void* raw,sbn3_series_range range,std::uint64_t terms,unsigned need,sbn3_series_shape* out) {
    const auto& f=*static_cast<const Facts*>(raw);*out={};
    const std::size_t begin=f.trigonometric?0:1,end=f.terms+begin;
    if(range.begin<begin||range.end>end||range.begin>=range.end)return SBN3_UNSUPPORTED;
    terms=std::min<std::uint64_t>(terms,range.end-range.begin);
    // Every exp subrange has |T/D| <= |z|/(1-|z|)<1; every even
    // subrange has |T/D|<=1/(1-|z|^2)<2. After alignment to -n*b
    // (or -2*n*b), these bounds include all sums without a log(n) loss.
    const __uint128_t F=__uint128_t(terms)*f.factor_bits,degree=f.trigonometric?2:1;
    const __uint128_t bits[]{__uint128_t(terms)*degree*f.b+F+2,F+1,__uint128_t(terms)*degree*(f.b-f.a)+1};
    for(unsigned j=0;j<3;++j)if(need&(1u<<j)){
        if(bits[j]>SIZE_MAX-63)return SBN3_QUERY_CAPACITY;out->limbs[j]=(std::size_t(bits[j])+63)/64;
    }
    return SBN3_SUPPORTED;
}
struct SliceRun {const Facts* facts;integer_view value;bool negative=false,sinc=false;};
void put(sbn3_series_value& out,integer_view value,std::int64_t exponent,bool negative) {
    require(value.size()<=out.mantissa.capacity,SBN3_FATAL_WORKSPACE,"bit-burst leaf capacity");
    if(value.size())std::memcpy(out.mantissa.data,value.data(),value.size()*8);
    out.mantissa.size=value.size();out.mantissa.negative=value.size()&&negative;out.exponent2=exponent;
}
void slice_leaf(const void* raw,std::uint64_t k,unsigned need,sbn3_series_values* out) {
    const auto& run=*static_cast<const SliceRun*>(raw);const auto& f=*run.facts;
    __uint128_t denominator=k;
    if(f.trigonometric){const __uint128_t j=2*__uint128_t(k)+(run.sinc?2:1);denominator=j*(j+1);}
    require(denominator,SBN3_FATAL_MATH,"bit-burst positive denominator");
    const std::uint64_t scalar[]{std::uint64_t(denominator),std::uint64_t(denominator>>64)};
    const integer_view d{{scalar,2}};
    if(need&SBN3_SERIES_T)put(out->value[0],f.trigonometric?d:run.value,
        f.trigonometric?0:-narrow(f.b),!f.trigonometric&&run.negative);
    if(need&SBN3_SERIES_D)put(out->value[1],d,0,false);
    if(need&SBN3_SERIES_U)put(out->value[2],run.value,-narrow((f.trigonometric?Wide(2):Wide(1))*f.b),
        f.trigonometric||run.negative);
}
series::FiniteFormula formula(const Facts& facts) noexcept {
    series::FiniteFormula f;f.context=&facts;f.recipe=SBN3_SERIES_COMMON_P2B3;
    f.formula_id=facts.trigonometric?UINT64_C(0x4242555253545343):UINT64_C(0x4242555253544558);
    f.parameter_id=(UINT64_C(0x9e3779b97f4a7c15)*facts.a)^facts.b^(std::uint64_t(facts.terms)<<1);
    f.max_leaf_limbs=facts.leaf_limbs;f.bounds=slice_bounds;f.leaf=slice_leaf;f.parameterized_leaves=true;
    f.work=[](const void* raw,sbn3_series_range range){const auto& x=*static_cast<const Facts*>(raw);
        return double(range.end-range.begin)*double(x.b+x.factor_bits+1);};
    return f;
}
sbn3_series_options options(unsigned workers) noexcept {
    // Single leaves avoid a scalar long-by-long batch when A grows to O(W)
    // bits. The existing scheduler retains compact serial-depth plans.
    sbn3_series_options o{};o.workers=workers;o.leaf_terms=1;return o;
}
Quote quote(bits absolute,unsigned workers,BitBurstKind kind) noexcept {
    Quote q;auto& info=q.layout;auto& need=info.need;need.scratch_alignment=128;
    if(!absolute.count||absolute.count>float_precision_limit||!workers||workers>32||
       (kind!=BitBurstKind::exp&&kind!=BitBurstKind::expm1&&kind!=BitBurstKind::sincos)){
        need.status=support::unsupported;need.reason="bit-burst request";return q;
    }
    std::size_t working=0;
    if(!add_size(absolute.count,64+4*lg(absolute.count+2),working)||!add_size(working,63,working)){
        need.status=support::unsupported;need.reason="bit-burst precision extent";return q;
    }
    working&=~std::size_t(63);const auto L=working/64;
    // This is the existing scaled-ratio service's checked limb domain, not
    // a cap on dyadic chunk width or a fallback to many fixed-size slices.
    if(L<2||L>(std::size_t(1)<<29)){
        need.status=support::unsupported;need.reason="bit-burst ratio precision domain";return q;
    }
    info.absolute_bits=absolute.count;info.working_bits=working;info.halvings=lg(working)+3;
    const bool trig=kind==BitBurstKind::sincos;std::size_t a=info.halvings-1,records=0;
    std::size_t metadata=sizeof(State),phase=0,table_budget=0,slice_table_budget=0;
    if(!product_shape(L+1,false,workers,q.product)||(trig&&!product_shape(L+1,true,workers,q.square))||
       normalized_newton_query(SBN3_NEWTON_DIVIDE,L+2,workers,q.ratio)!=SBN3_SUPPORTED){
        need.status=support::unsupported;need.reason="bit-burst arithmetic shape";return q;
    }
    need.scratch_alignment=std::max({need.scratch_alignment,q.product.alignment,q.square.alignment,q.ratio.alignment});
    phase=std::max({q.ratio.workspace_bytes,q.product.plan.info.workspace_bytes,q.square.plan.info.workspace_bytes});
    q.raw_words=std::max(q.product.capacity,q.square.capacity);
    for(const auto* p:{&q.product,&q.square})if(!p->leaf&&p->plan.tables.bytes){
        std::size_t extent=0;
        if(!add_size(p->plan.tables.bytes,p->plan.tables.alignment-1,extent)||!add_size(table_budget,extent,table_budget)){
            need.status=support::unsupported;need.reason="bit-burst table extent";return q;
        }
    }
    while(a<working){
        if(info.slices==maximum_slices){need.status=support::unsupported;need.reason="bit-burst slice count";return q;}
        auto& slice=q.slices[info.slices++];auto& f=slice.facts;
        f.a=a;f.b=std::min(2*a,working);f.trigonometric=trig;
        const auto divisor=trig?2*a:a;f.terms=(working+2+divisor-1)/divisor;
        f.factor_bits=trig?2*std::bit_width(2*f.terms+3):std::bit_width(f.terms);
        f.leaf_limbs=std::max<std::size_t>(2,((trig?2:1)*(f.b-f.a)+63)/64);
        slice.argument_words=(f.b-f.a+63)/64;
        const auto begin=trig?0u:1u;
        if(series::finite_frame_query(formula(f),{begin,f.terms+begin},SBN3_SERIES_T|SBN3_SERIES_D,options(workers),slice.tree)!=SBN3_SUPPORTED){
            need.status=support::unsupported;need.reason="bit-burst finite series shape";return q;
        }
        need.scratch_alignment=std::max(need.scratch_alignment,slice.tree.alignment);
        phase=std::max(phase,slice.tree.phase_bytes);slice_table_budget=std::max(slice_table_budget,slice.tree.series.prepared_bytes);
        if(!add_size(records,slice.tree.series.stages,records)){need.status=support::unsupported;need.reason="bit-burst record count";return q;}
        q.argument_words=std::max(q.argument_words,slice.argument_words);
        info.max_leaf_limbs=std::max(info.max_leaf_limbs,f.leaf_limbs);
        for(unsigned j=0;j<3;++j){q.root_words[j]=std::max(q.root_words[j],slice.tree.series.output.limbs[j]);
            info.max_exact_limbs=std::max(info.max_exact_limbs,q.root_words[j]);}
        if(trig){ProductShape sq;if(!product_shape(slice.argument_words,true,workers,sq)){
                need.status=support::unsupported;need.reason="bit-burst slice square shape";return q;}
            need.scratch_alignment=std::max(need.scratch_alignment,sq.alignment);phase=std::max(phase,sq.phase_bytes);
            q.squared_words=std::max(q.squared_words,sq.capacity);
            std::size_t extent=0;if(!add_size(sq.plan.tables.bytes,sq.plan.tables.alignment,extent)){
                need.status=support::unsupported;need.reason="bit-burst slice table extent";return q;}
            slice_table_budget=std::max(slice_table_budget,extent);
        }
        a=f.b;
    }
    if(working<absolute.count+info.halvings+lg(info.slices+1)+16){need.status=support::unsupported;need.reason="bit-burst error budget";return q;}
    q.directory=records>=table_capacity/2?table_capacity:std::min(table_capacity,2*records+info.slices+4);
    if(!append(metadata,info.slices*sizeof(Slice),128)||!append(metadata,q.directory*sizeof(ProductTableRef),128)){
        need.status=support::unsupported;need.reason="bit-burst metadata extent";return q;
    }
    for(std::size_t i=0;i<info.slices;++i)if(!append(metadata,q.slices[i].tree.metadata_bytes,q.slices[i].tree.alignment)){
        need.status=support::unsupported;need.reason="bit-burst series metadata extent";return q;
    }
    if(!append(metadata,q.ratio.metadata_bytes,q.ratio.alignment)||!append(metadata,q.product.local_bytes,128)||
       !append(metadata,q.square.local_bytes,128)||!add_size(table_budget,slice_table_budget,table_budget)){
        need.status=support::unsupported;need.reason="bit-burst retained metadata extent";return q;
    }
    info.metadata_bytes=metadata;info.shared_table_budget=table_budget;info.phase_bytes=phase;
    std::size_t total=metadata;
    if(!append(total,table_budget,need.scratch_alignment)){need.status=support::unsupported;need.reason="bit-burst shared tables";return q;}
    const auto values_begin=total;
    bool ok=true;
    for(unsigned i=0;i<fixed_banks&&ok;++i)ok=append_words(total,L+2,64);
    for(unsigned i=0;i<3&&ok;++i)ok=append_words(total,q.root_words[i],64);
    ok=ok&&append_words(total,q.argument_words,64)&&append_words(total,q.squared_words,need.scratch_alignment)&&
        append_words(total,q.raw_words,need.scratch_alignment)&&append_words(total,L+3,64)&&
        append_words(total,L+2,64)&&append_words(total,L+3,64);
    info.live_value_bytes=total-values_begin;
    if(!ok||!append(total,phase,need.scratch_alignment)){need.status=support::unsupported;need.reason="bit-burst live storage";return q;}
    need.scratch_bytes=total;return q;
}
ProductTablePoolView pool(const State& s) noexcept{return {s.directory,s.directory_count};}
struct Cache {State& state;Frame& tables;};
void cache_product(void* raw,const ProductProgramPlan& p) {
    auto& c=*static_cast<Cache*>(raw);auto& s=c.state;const auto request=product_program_tables(p);
    if(!request.bytes||pool(s).find(request)||s.directory_count==s.directory_capacity)return;
    std::size_t at=0,end=0;
    if(!align_size(c.tables.used(),request.alignment,at)||!add_size(at,request.bytes,end)||end>c.tables.capacity())return;
    auto storage=c.tables.subframe(request.bytes,request.alignment);const auto* backend=backend_lookup(p.plan.opaque[1]);
    const void* data=backend->program_tables_prepare(p.plan,storage);
    require(data==storage.data(),SBN3_FATAL_MATH,"bit-burst shared table base");
    s.directory[s.directory_count++]={request,data};
}
ProductProgram prepare_fixed(const ProductShape& shape,void* region,ProductTablePoolView tables) {
    if(shape.leaf)return {};
    auto frame=Frame::external(region,shape.local_bytes);
    const auto* shared=tables.find(shape.plan.tables);
    require(!shape.plan.tables.bytes||shared,SBN3_FATAL_WORKSPACE,"bit-burst fixed shared table");
    return product_program_prepare(shape.plan,frame,shared);
}
std::size_t normalized_size(const std::uint64_t* p,std::size_t n) noexcept {while(n&&!p[n-1])--n;return n;}
sbn3_int_view view(const sbn3_int& x) noexcept{return {x.data,x.size,x.negative};}
void constant(State& s,unsigned bank,std::uint64_t value) noexcept {
    auto& x=s.values[bank];std::fill_n(x.data,x.capacity,0);x.data[s.fractional]=value;x.size=value?s.fractional+1:0;x.negative=0;
}
void copy(State& s,unsigned out,unsigned in) noexcept {
    const auto value=s.values[in];auto& dst=s.values[out];
    if(value.size)std::memmove(dst.data,value.data,value.size*8);dst.size=value.size;dst.negative=value.negative;
}
void add(State& s,unsigned out,unsigned a,unsigned b,bool subtract=false) noexcept {
    auto x=view(s.values[a]),y=view(s.values[b]);if(subtract&&y.size)y.negative^=1;
    sbn3_int_add(&s.values[out],x,y);require(s.values[out].size<=s.words,SBN3_FATAL_MATH,"bit-burst fixed sum range");
}
void multiply(State& s,unsigned out,unsigned a,unsigned b,sbn3_team_scope* scope,bool square=false) noexcept {
    auto& x=s.values[a];auto& y=s.values[square?a:b];const auto negative=x.negative^y.negative;
    std::fill(x.data+x.size,x.data+s.words,0);if(x.data!=y.data)std::fill(y.data+y.size,y.data+s.words,0);
    const auto& shape=square?s.square_shape:s.product_shape;
    if(shape.leaf)mul_basecase_assumed(s.raw,x.data,s.words,y.data,s.words);
    else{auto work=Frame::external(s.phase.data(),s.phase.size());
        product_program_execute(square?s.square:s.product,work,scope,{x.data,s.words},square?sbn3_const_limbs{}:sbn3_const_limbs{y.data,s.words},
                                {s.raw,s.raw_capacity});}
    auto& result=s.values[out];std::memcpy(result.data,s.raw+s.fractional,s.words*8);
    result.size=normalized_size(result.data,s.words);result.negative=result.size&&negative;
}
void square_slice(State& s,const Slice& slice,sbn3_team_scope* scope) noexcept {
    const auto& shape=slice.square;
    if(shape.leaf){mul_basecase_assumed(s.squared,s.argument,shape.words,s.argument,shape.words);return;}
    auto phase=Frame::external(s.phase.data(),s.phase.size());ProductProgram program;
    {auto tables=phase.subframe(shape.plan.prepared_bytes+128,128);
        program=product_program_prepare(shape.plan,tables,pool(s).find(shape.plan.tables));}
    auto work=phase.subframe(shape.plan.info.workspace_bytes,shape.plan.info.workspace_alignment);
    product_program_execute(program,work,scope,{s.argument,shape.words},{},{s.squared,s.squared_capacity});
}
void evaluate_slice(State& s,const Slice& slice,const SliceRun& run,unsigned output,sbn3_team_scope* scope) noexcept {
    {auto work=Frame::external(s.phase.data(),s.phase.size());
        series::finite_frame_execute(slice.tree,&run,s.root,work,scope,pool(s));}
    const auto& t=s.root.value[0];const auto& d=s.root.value[1];auto& out=s.values[output];
    if(!t.mantissa.size){out.size=out.negative=0;return;}
    const series::ScaledMagnitude numerator{{t.mantissa.data,t.mantissa.size,0},t.exponent2};
    const series::ScaledMagnitude denominator{{d.mantissa.data,d.mantissa.size,d.mantissa.negative},d.exponent2};
    series::ratio_inputs(numerator,denominator,s.fractional,s.ratio_a,s.ratio_d);
    {auto work=Frame::external(s.phase.data(),s.phase.size());
        normalized_newton_execute(s.ratio,{s.ratio_a,s.fractional+3},{s.ratio_d,s.fractional+2},
                                  {s.ratio_q,s.fractional+3},work,scope);}
    const auto value=series::ratio_result(s.ratio_q,s.fractional);
    std::memcpy(out.data,value.data,value.count*8);out.size=normalized_size(out.data,value.count);out.negative=out.size&&t.mantissa.negative;
}
void compute(State& s,float_view input,sbn3_team_scope* scope) noexcept {
    s.stats.nonzero_slices=s.stats.max_runtime_leaf_limbs=0;
    constant(s,one,1);constant(s,two,2);constant(s,accumulator,0);constant(s,cosine,1);
    const bool trig=s.kind==BitBurstKind::sincos;
    for(std::size_t i=0;i<s.layout.slices;++i){const auto& slice=s.slices[i];const auto& f=slice.facts;
        const auto count=(f.b-f.a+63)/64;
        series::scaled_slice(s.argument,count,input.magnitude().c_view(),narrow(Wide(input.exponent2())+f.b-s.layout.halvings));
        if((f.b-f.a)%64)s.argument[count-1]&=(UINT64_C(1)<<((f.b-f.a)%64))-1;
        const auto used=normalized_size(s.argument,count);if(!used)continue;++s.stats.nonzero_slices;
        if(trig){
            square_slice(s,slice,scope);const auto squared=normalized_size(s.squared,2*count);
            s.stats.max_runtime_leaf_limbs=std::max(s.stats.max_runtime_leaf_limbs,squared);
            SliceRun run{&f,integer_view{{s.squared,squared}},false,false};
            evaluate_slice(s,slice,run,slice_cosine,scope);run.sinc=true;evaluate_slice(s,slice,run,sinc,scope);
            auto& arg=s.values[argument];series::scaled_slice(arg.data,s.words,{s.argument,used,0},narrow(Wide(s.layout.working_bits)-f.b));
            arg.size=normalized_size(arg.data,s.words);arg.negative=0;
            multiply(s,increment,argument,sinc,scope);
            multiply(s,temp0,accumulator,slice_cosine,scope);multiply(s,temp1,cosine,increment,scope);
            multiply(s,temp2,cosine,slice_cosine,scope);multiply(s,temp3,accumulator,increment,scope);
            add(s,accumulator,temp0,temp1);add(s,cosine,temp2,temp3,true);
        }else{
            s.stats.max_runtime_leaf_limbs=std::max(s.stats.max_runtime_leaf_limbs,used);
            SliceRun run{&f,integer_view{{s.argument,used}},input.negative(),false};
            evaluate_slice(s,slice,run,increment,scope);
            multiply(s,temp0,accumulator,increment,scope);add(s,temp1,accumulator,increment);add(s,accumulator,temp1,temp0);
        }
    }
    for(std::size_t i=0;i<s.layout.halvings;++i){
        if(trig){
            multiply(s,temp0,accumulator,cosine,scope);sbn3_int_lshift(&s.values[temp0],view(s.values[temp0]),1);
            multiply(s,temp1,cosine,cosine,scope,true);multiply(s,temp2,accumulator,accumulator,scope,true);
            add(s,cosine,temp1,temp2,true);copy(s,accumulator,temp0);
        }else{add(s,temp0,two,accumulator);multiply(s,temp1,accumulator,temp0,scope);copy(s,accumulator,temp1);}
    }
    if(s.kind==BitBurstKind::exp)add(s,accumulator,one,accumulator);
}
std::int64_t error_exponent(const State& s) noexcept {
    // U=2^-W. Exp slices have error <3U; one increment product adds <U.
    // Thus v=1+u after K compositions differs from exp(r/2^s) by
    // <=16(K+1)U, including the omitted input window. In reconstruction
    // v_next=v*v+e, |e|<U. For |h|=|log(v)-r_j|<=1/16, v>1/2 and
    // |h_next|<=2|h|+8U; the resulting absolute error is bounded by
    // 512*2^s*(K+1)U. For trig, complex z=C+iS has slice norm error
    // <=8U and composition/doubling rounding norm <=3U; the analogous
    // envelope is <128*2^s*(K+1)U. The checked W condition makes both
    // small-error inductions self-consistent. Publish 2048*2^s*(K+1)U
    // rounded upward in the K factor, then add final output quantization.
    return narrow(Wide(s.layout.halvings)+lg(s.layout.slices+1)+11-s.layout.working_bits);
}
float_report emit(State& s,float_ref out,unsigned bank,bool negative_zero,rounding mode) noexcept {
    const auto& value=s.values[bank];auto report=round_magnitude(out,integer_view{{value.data,value.size}},
        -narrow(s.layout.working_bits),value.size?bool(value.negative):negative_zero,mode);
    report.error=ErrorMagnitude::from(report.error).add(ErrorMagnitude::power_of_two(error_exponent(s))).publish();return report;
}
bool domain(float_view x) noexcept {
    if(!x.finite())return false;validate(x);if(x.zero())return true;
    const auto top=x.exponent2()+std::int64_t(magnitude_bits(x.magnitude()));if(top<=0)return true;if(top>1)return false;
    const std::uint64_t one=1;return compare(float_view::from_magnitude(x.magnitude().limbs(),x.exponent2()),
        float_view::from_magnitude({&one,1},0))!=std::partial_ordering::greater;
}
void aliases(std::span<std::byte> storage,float_ref out,float_view input) noexcept {
    check_output(out,input);const auto* destination=detail::float_access::begin(out);
    require(!overlaps(storage.data(),storage.size(),input.magnitude().data(),input.magnitude().size()*8)&&
            !overlaps(storage.data(),storage.size(),destination->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"bit-burst value/storage alias");
}
struct Use {State& state;explicit Use(State& s):state(s){std::uint32_t expected=0;
    require(__atomic_compare_exchange_n(&s.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"concurrent bit-burst program");}
    ~Use(){__atomic_store_n(&state.active,0,__ATOMIC_RELEASE);}};
State& state_of(const FloatBitBurstPrepared& p) noexcept {
    require(p.state&&p.state==p.storage.data()&&p.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"bit-burst prepared state");
    auto& s=*static_cast<State*>(p.state);require(s.marker==magic&&s.bytes==p.storage.size(),SBN3_FATAL_LIFETIME,"bit-burst prepared identity");return s;
}
}

BitBurstLayout bit_burst_program_layout(bits absolute,unsigned workers,BitBurstKind kind) noexcept{return quote(absolute,workers,kind).layout;}
requirements bit_burst_program_requirements(bits absolute,unsigned workers,BitBurstKind kind) noexcept{return quote(absolute,workers,kind).layout.need;}
FloatBitBurstPrepared bit_burst_program_prepare(bits absolute,unsigned workers,BitBurstKind kind,std::span<std::byte> storage) noexcept {
    const auto q=quote(absolute,workers,kind);const auto& need=q.layout.need;require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);
    require(storage.size()>=need.scratch_bytes&&!(std::uintptr_t(storage.data())&(need.scratch_alignment-1)),
            SBN3_FATAL_WORKSPACE,"bit-burst prepared storage",need.scratch_bytes,storage.size());
    auto frame=Frame::external(storage.data(),storage.size());auto* state=::new(frame.allocate(sizeof(State),128))State{};
    auto& s=*state;s.layout=q.layout;s.kind=kind;s.workers=workers;s.fractional=q.layout.working_bits/64;s.words=s.fractional+1;s.bytes=need.scratch_bytes;
    s.product_shape=q.product;s.square_shape=q.square;
    s.slices=static_cast<Slice*>(frame.allocate(s.layout.slices*sizeof(Slice),128));
    for(std::size_t i=0;i<s.layout.slices;++i)::new(s.slices+i)Slice{};
    s.directory_capacity=q.directory;s.directory=static_cast<ProductTableRef*>(frame.allocate(q.directory*sizeof(ProductTableRef),128));
    for(std::size_t i=0;i<q.directory;++i)::new(s.directory+i)ProductTableRef{};
    for(std::size_t i=0;i<s.layout.slices;++i){auto& slice=s.slices[i];slice.facts=q.slices[i].facts;const auto& f=slice.facts;
        auto metadata=frame.subframe(q.slices[i].tree.metadata_bytes,q.slices[i].tree.alignment);
        const auto begin=f.trigonometric?0u:1u;
        slice.tree=series::finite_frame_prepare(formula(f),{begin,f.terms+begin},SBN3_SERIES_T|SBN3_SERIES_D,options(workers),metadata);
        if(f.trigonometric)require(product_shape(q.slices[i].argument_words,true,workers,slice.square),SBN3_FATAL_ARGUMENT,"bit-burst square preparation");
    }
    {auto metadata=frame.subframe(q.ratio.metadata_bytes,q.ratio.alignment);
        s.ratio=normalized_newton_prepare(SBN3_NEWTON_DIVIDE,s.fractional+2,workers,metadata);}
    auto region=[&](std::size_t bytes,std::size_t alignment){return bytes?frame.allocate(bytes,alignment):nullptr;};
    void* fixed_product=region(q.product.local_bytes,128);void* fixed_square=region(q.square.local_bytes,128);
    void* table_storage=region(s.layout.shared_table_budget,need.scratch_alignment);
    {auto tables=Frame::external(table_storage,s.layout.shared_table_budget);Cache cache{s,tables};
        if(!q.product.leaf)cache_product(&cache,q.product.plan);if(!q.square.leaf)cache_product(&cache,q.square.plan);
        for(std::size_t i=0;i<s.layout.slices;++i){series::finite_frame_visit_products(s.slices[i].tree,cache_product,&cache);
            if(kind==BitBurstKind::sincos&&!s.slices[i].square.leaf)cache_product(&cache,s.slices[i].square.plan);}
        s.stats.shared_table_bytes=tables.used();}
    s.product=prepare_fixed(q.product,fixed_product,pool(s));s.square=prepare_fixed(q.square,fixed_square,pool(s));
    auto words=[&](std::size_t count,std::size_t alignment=64){return static_cast<std::uint64_t*>(region(count*8,alignment));};
    for(auto& value:s.values)value={words(s.fractional+2),s.fractional+2,0,0};
    for(unsigned i=0;i<3;++i)s.root.value[i]={{words(q.root_words[i]),q.root_words[i],0,0},0};
    s.argument=words(q.argument_words);s.argument_capacity=q.argument_words;
    s.squared=words(q.squared_words,need.scratch_alignment);s.squared_capacity=q.squared_words;
    s.raw=words(q.raw_words,need.scratch_alignment);s.raw_capacity=q.raw_words;
    s.ratio_a=words(s.fractional+3);s.ratio_d=words(s.fractional+2);s.ratio_q=words(s.fractional+3);
    s.phase={static_cast<std::byte*>(region(s.layout.phase_bytes,need.scratch_alignment)),s.layout.phase_bytes};
    s.marker=magic;return {state,storage.first(need.scratch_bytes)};
}
BitBurstStats bit_burst_statistics(const FloatBitBurstPrepared& prepared) noexcept{return state_of(prepared).stats;}
float_report exp_bit_burst_execute(const FloatBitBurstPrepared& prepared,float_ref out,float_view input,execution ex,rounding mode) noexcept {
    auto& s=state_of(prepared);require(s.kind!=BitBurstKind::sincos&&domain(input)&&ex.workers()>=s.workers,SBN3_FATAL_ARGUMENT,"bit-burst exp input/shape");
    aliases(prepared.storage,out,input);Use exclusive(s);
    if(input.zero()){const std::uint64_t one=1;return s.kind==BitBurstKind::expm1?special(out,float_class::finite,input.negative()):
        round_magnitude(out,integer_view{{&one,1}},0,false,mode);}
    struct Run {State* state;float_view input;};Run run{&s,input};
    auto action=[](void* p,sbn3_team_scope* scope){auto& r=*static_cast<Run*>(p);compute(*r.state,r.input,scope);};
    if(ex.scope)action(&run,ex.scope);else if(ex.team)sbn3_team_run(ex.team,action,&run);else serial_team_run(action,&run);
    return emit(s,out,accumulator,s.kind==BitBurstKind::expm1&&input.negative(),mode);
}
TrigSincosReport sincos_bit_burst_execute(const FloatBitBurstPrepared& prepared,float_ref sine,float_ref cosine_out,float_view input,
                                         execution ex,rounding mode) noexcept {
    auto& s=state_of(prepared);require(s.kind==BitBurstKind::sincos&&domain(input)&&ex.workers()>=s.workers,SBN3_FATAL_ARGUMENT,"bit-burst sincos input/shape");
    aliases(prepared.storage,sine,input);aliases(prepared.storage,cosine_out,input);
    require(!overlaps(detail::float_access::begin(sine)->data,sine.capacity()*8,detail::float_access::begin(cosine_out)->data,cosine_out.capacity()*8),
            SBN3_FATAL_ARGUMENT,"bit-burst paired output alias");Use exclusive(s);
    if(input.zero()){const std::uint64_t one=1;return {special(sine,float_class::finite,input.negative()),
        round_magnitude(cosine_out,integer_view{{&one,1}},0,false,mode)};}
    struct Run {State* state;float_view input;};Run run{&s,input};
    auto action=[](void* p,sbn3_team_scope* scope){auto& r=*static_cast<Run*>(p);compute(*r.state,r.input,scope);};
    if(ex.scope)action(&run,ex.scope);else if(ex.team)sbn3_team_run(ex.team,action,&run);else serial_team_run(action,&run);
    if(s.values[accumulator].size&&input.negative())s.values[accumulator].negative^=1;
    const auto sr=emit(s,sine,accumulator,input.negative(),mode);const auto cr=emit(s,cosine_out,cosine,false,mode);return {sr,cr};
}

namespace {
bool tiny_point(float_view x,std::size_t p,BitBurstKind kind) noexcept {
    if(x.zero())return true;const auto T=x.exponent2()+std::int64_t(magnitude_bits(x.magnitude()));
    return kind==BitBurstKind::sincos?Wide(2)*T<=-Wide(p)-10:T<=-Wide(p)-8;
}
std::size_t tiny_bytes(float_view x,std::size_t p,BitBurstKind kind) noexcept {
    if(kind==BitBurstKind::exp||x.zero()||magnitude_bits(x.magnitude())<=p)return 0;
    const auto words=(std::min(float_precision_limit,p+32)+63)/64;return words<=8?0:words*8+63;
}
float_report tiny_exp(float_ref out,float_view x,rounding mode) noexcept {
    const auto exponent=x.zero()?0:narrow(Wide(x.exponent2())+magnitude_bits(x.magnitude())+1);
    const bool zero=x.zero();
    const std::uint64_t one=1;auto report=round_magnitude(out,integer_view{{&one,1}},0,false,mode);
    if(!zero)report.error=error_bound::power_of_two(exponent);return report;
}
float_report tiny_linear(float_ref out,float_view x,unsigned power,execution ex,rounding mode) noexcept {
    if(x.zero())return special(out,float_class::finite,x.negative());
    const auto T=narrow(Wide(x.exponent2())+magnitude_bits(x.magnitude()));
    auto error=ErrorMagnitude::power_of_two(narrow(Wide(power)*T));float_report report;
    if(magnitude_bits(x.magnitude())<=out.precision().count)report=round_into(out,x,mode);
    else{
        const auto W=std::min(float_precision_limit,out.precision().count+32),count=(W+63)/64;
        alignas(64)std::uint64_t local[8];std::uint64_t* data=local;
        if(count>8){require(ex.scratch.size()>=count*8+63,SBN3_FATAL_WORKSPACE,"bit-burst tiny prefix storage");
            const auto address=(std::uintptr_t(ex.scratch.data())+63)&~std::uintptr_t(63);data=reinterpret_cast<std::uint64_t*>(address);}
        const auto input=x.magnitude();const auto width=magnitude_bits(input),cut=width>W?width-W:0,kept=width-cut;
        series::scaled_slice(data,(kept+63)/64,input.c_view(),-narrow(cut));
        report=round_magnitude(out,integer_view{{data,(kept+63)/64}},narrow(Wide(x.exponent2())+cut),x.negative(),mode);
        if(cut)error=error.add(ErrorMagnitude::power_of_two(T-narrow(W)));
    }
    report.error=ErrorMagnitude::from(report.error).add(error).publish();return report;
}
bits absolute_for(std::size_t p,float_view input,BitBurstKind kind) noexcept {
    const auto T=input.zero()?0:narrow(Wide(input.exponent2())+magnitude_bits(input.magnitude()));
    const auto extra=kind==BitBurstKind::exp?Wide(0):std::max<Wide>(0,-Wide(T));
    const auto target=Wide(p)+extra+8;return {target>SIZE_MAX?SIZE_MAX:std::size_t(target)};
}
}
requirements bit_burst_requirements(float_ref out,float_view input,BitBurstKind kind,unsigned workers) noexcept {
    const auto p=out.precision().count,words=(p+63)/64;
    if(!p||p>float_precision_limit||!workers||workers>32||
       (kind!=BitBurstKind::exp&&kind!=BitBurstKind::expm1)||!domain(input))
        return {support::unsupported,0,128,words,"bit-burst point input/shape"};
    if(out.capacity()<words)return {support::capacity,0,128,words,"bit-burst output capacity"};
    if(tiny_point(input,p,kind))return {support::yes,tiny_bytes(input,p,kind),64,words,""};
    auto need=bit_burst_program_requirements(absolute_for(p,input,kind),workers,kind);need.output_limbs=words;return need;
}
float_report bit_burst_into(float_ref out,float_view input,BitBurstKind kind,execution ex,rounding mode) noexcept {
    const auto p=out.precision().count;require((kind==BitBurstKind::exp||kind==BitBurstKind::expm1)&&domain(input),SBN3_FATAL_ARGUMENT,"bit-burst point input");
    aliases(ex.scratch,out,input);
    if(tiny_point(input,p,kind))return kind==BitBurstKind::exp?tiny_exp(out,input,mode):tiny_linear(out,input,2,ex,mode);
    const auto program=bit_burst_program_prepare(absolute_for(p,input,kind),ex.workers(),kind,ex.scratch);
    return exp_bit_burst_execute(program,out,input,ex,mode);
}
requirements bit_burst_sincos_requirements(float_ref sine,float_ref cosine_out,float_view input,unsigned workers) noexcept {
    const auto p=std::max(sine.precision().count,cosine_out.precision().count),words=(p+63)/64;
    if(!sine.precision().count||!cosine_out.precision().count||p>float_precision_limit||!workers||workers>32||!domain(input))
        return {support::unsupported,0,128,words,"bit-burst paired input/shape"};
    if(sine.capacity()<(sine.precision().count+63)/64||cosine_out.capacity()<(cosine_out.precision().count+63)/64)
        return {support::capacity,0,128,words,"bit-burst paired output capacity"};
    if(tiny_point(input,p,BitBurstKind::sincos))return {support::yes,tiny_bytes(input,sine.precision().count,BitBurstKind::sincos),64,words,""};
    auto need=bit_burst_program_requirements(absolute_for(p,input,BitBurstKind::sincos),workers,BitBurstKind::sincos);need.output_limbs=words;return need;
}
TrigSincosReport bit_burst_sincos_into(float_ref sine,float_ref cosine_out,float_view input,execution ex,rounding mode) noexcept {
    const auto p=std::max(sine.precision().count,cosine_out.precision().count);require(domain(input),SBN3_FATAL_ARGUMENT,"bit-burst paired input");
    aliases(ex.scratch,sine,input);aliases(ex.scratch,cosine_out,input);
    require(!overlaps(detail::float_access::begin(sine)->data,sine.capacity()*8,detail::float_access::begin(cosine_out)->data,cosine_out.capacity()*8),
            SBN3_FATAL_ARGUMENT,"bit-burst paired output alias");
    if(tiny_point(input,p,BitBurstKind::sincos)){
        const auto T=input.zero()?0:narrow(Wide(input.exponent2())+magnitude_bits(input.magnitude()));
        const bool zero=input.zero();const auto sr=tiny_linear(sine,input,3,ex,mode);const std::uint64_t one=1;
        auto cr=round_magnitude(cosine_out,integer_view{{&one,1}},0,false,mode);if(!zero)cr.error=error_bound::power_of_two(narrow(Wide(2)*T-1));
        return {sr,cr};
    }
    const auto program=bit_burst_program_prepare(absolute_for(p,input,BitBurstKind::sincos),ex.workers(),BitBurstKind::sincos,ex.scratch);
    return sincos_bit_burst_execute(program,sine,cosine_out,input,ex,mode);
}
}
