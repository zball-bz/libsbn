// Original libsbn code; LGPL-3.0-or-later.
#include "float/product.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

namespace sbn::v3::floating {
namespace {
constexpr std::size_t prefix_guard_bits=128, small_prefix_limbs=6;

std::size_t prefix_bits(std::size_t p, float_view x) noexcept {
    return std::min(magnitude_bits(x.magnitude()),p+prefix_guard_bits);
}

bool append(std::size_t& cursor,std::size_t bytes,std::size_t alignment,
            std::size_t& offset) noexcept {
    return align_size(cursor,alignment,offset)&&add_size(offset,bytes,cursor);
}

std::int64_t exponent(__int128 e) noexcept {
    require(e>=std::numeric_limits<std::int64_t>::min()&&
            e<=std::numeric_limits<std::int64_t>::max(),
            SBN3_FATAL_SIZE,"float product exponent");
    return std::int64_t(e);
}

struct Prefix {
    std::int64_t exponent2=0;
    bool truncated=false;
};

// The low omitted range is deliberately never scanned. Its full possible
// magnitude is bounded analytically, even when those bits happen to be zero.
Prefix stage_prefix(std::uint64_t* out,std::size_t words,std::size_t bound,
                    std::size_t precision,float_view x) noexcept {
    const auto input=x.magnitude();
    const auto bits=magnitude_bits(input);
    const auto kept=std::min(bits,precision+prefix_guard_bits);
    require(kept<=bound&&words>=(kept+63)/64,SBN3_FATAL_ARGUMENT,
            "float product prefix shape",kept,bound);
    const auto cut=bits-kept, first=cut/64;
    const unsigned shift=unsigned(cut%64);
    const auto count=(kept+63)/64;
    for(std::size_t i=0;i<count;++i){
        const auto at=first+i;
        out[i]=(input.data()[at]>>shift)|
            (shift&&at+1<input.size()?input.data()[at+1]<<(64-shift):0);
    }
    std::fill(out+count,out+words,0);
    return {exponent(__int128(x.exponent2())+cut),cut!=0};
}

// Only well-separated range cases bypass the product. Boundary cases pass
// through the common rounder; the error of a zero output uses the full input
// bound, never a potentially smaller top bit of the truncated product.
bool special_product(float_ref out,float_view a,float_view b,float_report& report) noexcept {
    const bool negative=a.negative()!=b.negative();
    if(a.is_nan()||b.is_nan()){
        report=special(out,float_class::quiet_nan,false);return true;
    }
    if(!a.finite()||!b.finite()){
        report=(a.zero()||b.zero())
            ?special(out,float_class::quiet_nan,false,float_flag::invalid)
            :special(out,float_class::infinity,negative);
        return true;
    }
    if(a.zero()||b.zero()){
        report=special(out,float_class::finite,negative);return true;
    }
    const auto top=__int128(a.exponent2())+magnitude_bits(a.magnitude())+
                   __int128(b.exponent2())+magnitude_bits(b.magnitude());
    if(top-1>float_exponent_limit){
        report=special(out,float_class::infinity,negative,float_flag::overflow|float_flag::rounded);
        return true;
    }
    if(top<-__int128(float_exponent_limit)-1){
        report=special(out,float_class::finite,negative,float_flag::underflow|float_flag::rounded);
        report.error=error_bound::power_of_two(exponent(top));return true;
    }
    return false;
}

float_report finish(float_ref out,const std::uint64_t* product,std::size_t words,
                    Prefix a,Prefix b,float_view original_a,float_view original_b,
                    rounding mode) noexcept {
    // Observe original input metadata before an exact-base alias overwrites it.
    const auto full_top=exponent(__int128(original_a.exponent2())+
        magnitude_bits(original_a.magnitude())+__int128(original_b.exponent2())+
        magnitude_bits(original_b.magnitude()));
    const bool negative=original_a.negative()!=original_b.negative();
    auto report=round_magnitude(out,integer_view{{product,words}},
        exponent(__int128(a.exponent2)+b.exponent2),negative,mode);
    if(has_flag(report.flags,float_flag::underflow)){
        report.error=error_bound::power_of_two(full_top);
    }else if((a.truncated||b.truncated)&&out.view().finite()&&!out.view().zero()){
        // Rounding the approximate mantissa is not directed rounding of the
        // full mathematical product. Nearest contributes <=1/2 ulp plus a
        // <2^-126-ulp prefix error. Other modes can need just over one ulp.
        const auto value=out.view();
        const auto ulp=exponent(__int128(value.exponent2())+
            magnitude_bits(value.magnitude())-out.precision().count);
        report.error=error_bound::power_of_two(ulp+(mode==rounding::nearest_even?0:1));
    }
    return report;
}

float_report small_product(float_ref out,float_view a,float_view b,bool square,
                           rounding mode) noexcept {
    if(square)b=a;
    const auto p=out.precision().count,ab=prefix_bits(p,a),bb=prefix_bits(p,b);
    const auto an=(ab+63)/64,bn=(bb+63)/64;
    require(an&&bn&&an<=small_prefix_limbs&&bn<=small_prefix_limbs,
            SBN3_FATAL_ARGUMENT,"small float product shape");
    alignas(64)std::uint64_t aw[small_prefix_limbs],bw[small_prefix_limbs];
    alignas(64)std::uint64_t product[2*small_prefix_limbs];
    const auto ap=stage_prefix(aw,an,ab,p,a);
    const auto bp=square?ap:stage_prefix(bw,bn,bb,p,b);
    sbn3_mul_basecase(product,an+bn,aw,an,square?aw:bw,bn);
    return finish(out,product,an+bn,ap,bp,a,b,mode);
}

void check_scratch(const FloatProductPrepared& prepared,float_ref out,
                   float_view a,float_view b) noexcept {
    const auto scratch=prepared.storage;
    for(auto v:{a,b})require(!overlaps(scratch.data(),scratch.size(),
        v.magnitude().data(),v.magnitude().size()*8),SBN3_FATAL_ARGUMENT,
        "float product input scratch alias");
    const auto* dst=detail::float_access::begin(out);
    require(!overlaps(scratch.data(),scratch.size(),dst->data,
        ((out.precision().count+63)/64)*8),SBN3_FATAL_ARGUMENT,
        "float product output scratch alias");
    require(!overlaps(prepared.plan,sizeof(*prepared.plan),scratch.data(),scratch.size()),
        SBN3_FATAL_ARGUMENT,"float product plan scratch alias");
}

struct ProductCall {const FloatProductPrepared* prepared;};
void product_action(void* opaque,sbn3_team_scope* scope){
    const auto& b=*static_cast<ProductCall*>(opaque)->prepared;
    const auto& p=*b.plan;
    auto work=Frame::external(b.work,p.product.info.workspace_bytes);
    product_program_execute(b.product,work,scope,{b.a,p.a_limbs},
        p.square?sbn3_const_limbs{}:sbn3_const_limbs{b.b,p.b_limbs},
        {b.result,p.product_capacity});
}

[[gnu::noinline]]requirements query_large(float_ref out,float_view a,float_view b,
                                        bool square,unsigned workers) noexcept {
    FloatProductPlan plan{};
    if(float_product_query(out.precision().count,magnitude_bits(a.magnitude()),
        magnitude_bits(b.magnitude()),square,workers,plan)!=SBN3_SUPPORTED)
        return {support::unsupported,0,64,(out.precision().count+63)/64,
                "float product backend shape"};
    const auto n=(out.precision().count+63)/64;
    return {out.capacity()<n?support::capacity:support::yes,plan.scratch_bytes,
            plan.alignment,n,""};
}

[[gnu::noinline]]float_report large_product(float_ref out,float_view a,float_view b,
                                          bool square,execution ex,rounding mode) noexcept {
    FloatProductPlan plan{};
    require(float_product_query(out.precision().count,magnitude_bits(a.magnitude()),
        magnitude_bits(b.magnitude()),square,ex.workers(),plan)==SBN3_SUPPORTED,
        SBN3_FATAL_ARGUMENT,"float product backend shape");
    // Check before table preparation touches the supplied scratch.
    FloatProductPrepared check{};check.plan=&plan;check.storage=ex.scratch;
    check_scratch(check,out,a,b);
    const auto prepared=float_product_prepare(plan,ex.scratch);
    return float_product_execute(prepared,out,a,b,ex,mode);
}

requirements query_product(float_ref out,float_view a,float_view b,
                           bool square,unsigned workers) noexcept {
    const auto p=out.precision().count,n=(p+63)/64;
    if(!p||p>float_precision_limit||!workers||workers>32)
        return {support::unsupported,0,64,0,"float product precision/workers"};
    if(square)b=a;
    if(!a.finite()||!b.finite()||a.zero()||b.zero()||
       (prefix_bits(p,a)<=64*small_prefix_limbs&&prefix_bits(p,b)<=64*small_prefix_limbs))
        return {out.capacity()<n?support::capacity:support::yes,0,64,n,""};
    return query_large(out,a,b,square,workers);
}
}

sbn3_query_result float_product_query(std::size_t precision,std::size_t a_bits,
    std::size_t b_bits,bool square,unsigned workers,FloatProductPlan& out) noexcept {
    if(!precision||precision>float_precision_limit||!a_bits||a_bits>float_precision_limit||
       (!square&&(!b_bits||b_bits>float_precision_limit))||!workers||workers>32)
        return SBN3_UNSUPPORTED;
    FloatProductPlan p{};
    p.precision_bits=precision;p.workers=workers;p.square=square;
    p.a_prefix_bits=std::min(a_bits,precision+prefix_guard_bits);
    p.b_prefix_bits=square?p.a_prefix_bits:std::min(b_bits,precision+prefix_guard_bits);
    p.a_limbs=(p.a_prefix_bits+63)/64;p.b_limbs=(p.b_prefix_bits+63)/64;
    p.product_limbs=p.a_limbs+p.b_limbs;
    p.small=p.a_limbs<=small_prefix_limbs&&p.b_limbs<=small_prefix_limbs;
    if(p.small){p.product_capacity=p.product_limbs;out=p;return SBN3_SUPPORTED;}
    sbn3_mul_options options{};options.workers=workers;options.borrow_output=1;
    sbn3_query_result rc;
    if(square){
        sbn3_product_request request{};request.kind=SBN3_PRODUCT_SQR;request.a_limbs=p.a_limbs;
        sbn3_product_info info{};
        rc=sbn3_product_query(&request,&options,&p.product.plan,&info);
        if(rc!=SBN3_SUPPORTED)return rc;
        p.product.info=info.mul;
        rc=product_program_describe(p.a_limbs,0,p.product);
    }else rc=product_program_query(p.a_limbs,p.b_limbs,options,p.product);
    if(rc!=SBN3_SUPPORTED)return rc;
    p.product_capacity=sbn3_mul_output_capacity(&p.product.info);
    p.alignment=std::max({std::size_t(128),p.product.info.workspace_alignment,
                          p.product.info.output_alignment});
    std::size_t cursor=0,bytes=0;
    if(!mul_size(p.a_limbs,8,bytes)||!append(cursor,bytes,64,p.a_offset))return SBN3_UNSUPPORTED;
    if(square)p.b_offset=p.a_offset;
    else if(!mul_size(p.b_limbs,8,bytes)||!append(cursor,bytes,64,p.b_offset))return SBN3_UNSUPPORTED;
    if(!mul_size(p.product_capacity,8,bytes)||
       !append(cursor,bytes,std::max(std::size_t(64),p.product.info.output_alignment),p.product_offset)||
       !add_size(p.product.prepared_bytes,128,p.tables_bytes)||
       !append(cursor,p.tables_bytes,128,p.tables_offset)||
       !append(cursor,p.product.info.workspace_bytes,p.product.info.workspace_alignment,p.work_offset)||
       !add_size(cursor,p.alignment-1,p.scratch_bytes))return SBN3_UNSUPPORTED;
    p.payload_bytes=cursor;out=p;return SBN3_SUPPORTED;
}

FloatProductPrepared float_product_prepare(const FloatProductPlan& p,
    std::span<std::byte> scratch) noexcept {
    require(p.precision_bits&&p.a_limbs&&p.b_limbs,SBN3_FATAL_ARGUMENT,"float product plan");
    FloatProductPrepared b{};b.plan=&p;
    if(p.small)return b;
    require(scratch.size()>=p.scratch_bytes,SBN3_FATAL_WORKSPACE,"float product scratch",
            p.scratch_bytes,scratch.size());
    require(!overlaps(&p,sizeof(p),scratch.data(),scratch.size()),SBN3_FATAL_ARGUMENT,
            "float product plan scratch alias");
    auto frame=Frame::external(scratch.data(),scratch.size());
    auto* base=static_cast<std::byte*>(frame.allocate(p.payload_bytes,p.alignment));
    b.storage=scratch;
    b.a=reinterpret_cast<std::uint64_t*>(base+p.a_offset);
    b.b=reinterpret_cast<std::uint64_t*>(base+p.b_offset);
    b.result=reinterpret_cast<std::uint64_t*>(base+p.product_offset);
    b.work=base+p.work_offset;
    auto tables=Frame::external(base+p.tables_offset,p.tables_bytes);
    b.product=product_program_prepare(p.product,tables);
    return b;
}

float_report float_product_execute(const FloatProductPrepared& b,float_ref out,
    float_view a,float_view c,execution ex,rounding mode) noexcept {
    require(b.plan&&out.precision().count==b.plan->precision_bits,SBN3_FATAL_ARGUMENT,
            "float product output precision");
    const auto& p=*b.plan;if(p.square)c=a;
    validate(a);validate(c);check_output(out,a,c);
    if(!p.small)check_scratch(b,out,a,c);
    float_report report;
    if(special_product(out,a,c,report))return report;
    require(prefix_bits(p.precision_bits,a)<=p.a_prefix_bits&&
            prefix_bits(p.precision_bits,c)<=p.b_prefix_bits,
            SBN3_FATAL_ARGUMENT,"float product input prefix bound");
    if(p.small)return small_product(out,a,c,p.square,mode);
    require(ex.workers()>=p.workers&&b.product.prepared&&b.a&&b.b&&b.result&&
            b.storage.size()>=p.scratch_bytes,SBN3_FATAL_WORKSPACE,"prepared float product");
    const auto ap=stage_prefix(b.a,p.a_limbs,p.a_prefix_bits,p.precision_bits,a);
    const auto cp=p.square?ap:stage_prefix(b.b,p.b_limbs,p.b_prefix_bits,p.precision_bits,c);
    ProductCall call{&b};
    if(ex.scope)product_action(&call,ex.scope);
    else if(ex.team)sbn3_team_run(ex.team,product_action,&call);
    else serial_team_run(product_action,&call);
    return finish(out,b.result,p.product_limbs,ap,cp,a,c,mode);
}
}

namespace sbn {
requirements multiply_requirements(float_ref out,float_view a,float_view b,unsigned workers) noexcept {
    return v3::floating::query_product(out,a,b,false,workers);
}
requirements square_requirements(float_ref out,float_view a,unsigned workers) noexcept {
    return v3::floating::query_product(out,a,a,true,workers);
}
float_report mul_into(float_ref out,float_view a,float_view b,execution ex,rounding mode) noexcept {
    using namespace v3::floating;
    validate(a);validate(b);check_output(out,a,b);
    float_report report;if(special_product(out,a,b,report))return report;
    const auto p=out.precision().count;
    if(prefix_bits(p,a)<=64*small_prefix_limbs&&prefix_bits(p,b)<=64*small_prefix_limbs)
        return small_product(out,a,b,false,mode);
    return large_product(out,a,b,false,ex,mode);
}
float_report square_into(float_ref out,float_view a,execution ex,rounding mode) noexcept {
    using namespace v3::floating;
    validate(a);check_output(out,a);
    float_report report;if(special_product(out,a,a,report))return report;
    if(prefix_bits(out.precision().count,a)<=64*small_prefix_limbs)
        return small_product(out,a,a,true,mode);
    return large_product(out,a,a,true,ex,mode);
}
}
