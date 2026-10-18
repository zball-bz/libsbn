// Original libsbn code; LGPL-3.0-or-later.
#include "float/fma.hpp"
#include <algorithm>
#include <limits>

namespace sbn::v3::floating {
namespace {
constexpr std::size_t small_operand_limbs=6;

std::size_t sum_bytes(std::size_t precision,bool addend_nonzero) noexcept {
    return precision<=256||!addend_nonzero?0:((precision+63)/64+2)*8+63;
}
bool append(std::size_t& cursor,std::size_t bytes,std::size_t alignment,
            std::size_t& offset) noexcept {
    return align_size(cursor,alignment,offset)&&add_size(offset,bytes,cursor);
}
std::int64_t product_exponent(float_view a,float_view b) noexcept {
    const auto e=__int128(a.exponent2())+b.exponent2();
    require(e>=std::numeric_limits<std::int64_t>::min()&&
            e<=std::numeric_limits<std::int64_t>::max(),
            SBN3_FATAL_SIZE,"float FMA product exponent");
    return std::int64_t(e);
}
void validate_arguments(float_ref out,float_view a,float_view b,float_view c) noexcept {
    validate(a);validate(b);validate(c);check_output(out,a,b);check_output(out,c);
}
void check_scratch(float_ref out,float_view a,float_view b,float_view c,
                   std::span<std::byte> scratch) noexcept {
    for(auto v:{a,b,c})require(!overlaps(scratch.data(),scratch.size(),
        v.magnitude().data(),v.magnitude().size()*8),SBN3_FATAL_ARGUMENT,
        "float FMA input scratch alias");
    const auto* destination=detail::float_access::begin(out);
    require(!overlaps(scratch.data(),scratch.size(),destination->data,
        ((out.precision().count+63)/64)*8),SBN3_FATAL_ARGUMENT,
        "float FMA output scratch alias");
}

bool special_fma(float_ref out,float_view a,float_view b,float_view c,
                 execution ex,rounding mode,float_report& result) noexcept {
    if(a.is_nan()||b.is_nan()||c.is_nan()){
        result=special(out,float_class::quiet_nan,false);return true;
    }
    const bool negative=a.negative()!=b.negative();
    if(!a.finite()||!b.finite()){
        if(a.zero()||b.zero()||(!c.finite()&&c.negative()!=negative))
            result=special(out,float_class::quiet_nan,false,float_flag::invalid);
        else result=special(out,float_class::infinity,negative);
        return true;
    }
    if(!c.finite()){
        result=special(out,float_class::infinity,c.negative());return true;
    }
    if(a.zero()||b.zero()){
        result=sum_finite(out,float_view::from_magnitude({},0,negative),c,ex,mode);
        return true;
    }
    return false;
}

float_report small_fma(float_ref out,float_view a,float_view b,float_view c,
                       execution ex,rounding mode) noexcept {
    alignas(64)std::uint64_t product[2*small_operand_limbs];
    const auto an=a.magnitude().size(),bn=b.magnitude().size();
    require(an&&bn&&an<=small_operand_limbs&&bn<=small_operand_limbs,
            SBN3_FATAL_ARGUMENT,"small float FMA shape");
    const auto exponent=product_exponent(a,b);
    const bool negative=a.negative()!=b.negative();
    // The product is separate from all public values. No destination write
    // happens until sum_finite has consumed both this product and c.
    sbn3_mul_basecase(product,an+bn,a.magnitude().data(),an,b.magnitude().data(),bn);
    return sum_finite(out,float_view::from_magnitude({product,an+bn},exponent,negative),c,ex,mode);
}

struct ProductCall {
    const ProductProgram* program;
    void* work;
    std::size_t work_bytes;
    sbn3_const_limbs a,b;
    sbn3_limbs output;
};
void product_action(void* opaque,sbn3_team_scope* scope){
    auto& c=*static_cast<ProductCall*>(opaque);auto work=Frame::external(c.work,c.work_bytes);
    product_program_execute(*c.program,work,scope,c.a,c.b,c.output);
}

[[gnu::noinline]]requirements query_large(float_ref out,float_view a,float_view b,
                                        float_view c,unsigned workers) noexcept {
    FloatFmaPlan plan{};
    if(float_fma_query(out.precision().count,a.magnitude().size(),b.magnitude().size(),
        !c.zero(),workers,plan)!=SBN3_SUPPORTED)
        return {support::unsupported,0,64,(out.precision().count+63)/64,"float FMA backend shape"};
    const auto words=(out.precision().count+63)/64;
    return {out.capacity()<words?support::capacity:support::yes,
            plan.scratch_bytes,plan.alignment,words,""};
}
[[gnu::noinline]]float_report large_fma(float_ref out,float_view a,float_view b,float_view c,
                                      execution ex,rounding mode) noexcept {
    FloatFmaPlan plan{};
    require(float_fma_query(out.precision().count,a.magnitude().size(),b.magnitude().size(),
        !c.zero(),ex.workers(),plan)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"float FMA backend shape");
    return float_fma_execute(plan,out,a,b,c,ex,mode);
}
}

sbn3_query_result float_fma_query(std::size_t precision,std::size_t an,std::size_t bn,
    bool addend_nonzero,unsigned workers,FloatFmaPlan& out) noexcept {
    if(!precision||precision>float_precision_limit||!an||!bn||
       an>float_precision_limit/64||bn>float_precision_limit/64||!workers||workers>32)
        return SBN3_UNSUPPORTED;
    FloatFmaPlan p{};p.precision_bits=precision;p.a_limbs=an;p.b_limbs=bn;
    p.product_limbs=an+bn;p.workers=workers;
    p.addition_bytes=sum_bytes(precision,addend_nonzero);
    p.small=an<=small_operand_limbs&&bn<=small_operand_limbs;
    if(p.small){
        p.product_capacity=p.product_limbs;p.scratch_bytes=p.addition_bytes;
        p.pool_bytes=p.addition_bytes;out=p;return SBN3_SUPPORTED;
    }
    sbn3_mul_options options{};options.workers=workers;options.borrow_output=1;
    const auto rc=product_program_query(an,bn,options,p.product);
    if(rc!=SBN3_SUPPORTED)return rc;
    p.product_capacity=sbn3_mul_output_capacity(&p.product.info);
    const auto pool_alignment=std::max(std::size_t(128),p.product.info.workspace_alignment);
    p.alignment=std::max(pool_alignment,p.product.info.output_alignment);
    std::size_t cursor=0,product_bytes=0;
    if(!add_size(p.product.prepared_bytes,128,p.tables_bytes)||
       !align_size(p.tables_bytes,p.product.info.workspace_alignment,p.work_offset)||
       !add_size(p.work_offset,p.product.info.workspace_bytes,p.multiply_bytes))
        return SBN3_UNSUPPORTED;
    p.pool_bytes=std::max(p.multiply_bytes,p.addition_bytes);
    if(!mul_size(p.product_capacity,8,product_bytes)||
       !append(cursor,product_bytes,std::max(std::size_t(64),p.product.info.output_alignment),p.product_offset)||
       !append(cursor,p.pool_bytes,pool_alignment,p.pool_offset)||
       !add_size(cursor,p.alignment-1,p.scratch_bytes))return SBN3_UNSUPPORTED;
    p.payload_bytes=cursor;out=p;return SBN3_SUPPORTED;
}

float_report float_fma_execute(const FloatFmaPlan& p,float_ref out,float_view a,
    float_view b,float_view c,execution ex,rounding mode) noexcept {
    validate_arguments(out,a,b,c);
    require(out.precision().count==p.precision_bits,SBN3_FATAL_ARGUMENT,"float FMA output precision");
    float_report report;if(special_fma(out,a,b,c,ex,mode,report))return report;
    require(a.magnitude().size()==p.a_limbs&&b.magnitude().size()==p.b_limbs&&
            p.addition_bytes>=sum_bytes(p.precision_bits,!c.zero()),
            SBN3_FATAL_ARGUMENT,"float FMA compiled input shape");
    require(ex.scratch.size()>=p.scratch_bytes,SBN3_FATAL_WORKSPACE,"float FMA scratch",
            p.scratch_bytes,ex.scratch.size());
    check_scratch(out,a,b,c,ex.scratch);
    require(!overlaps(&p,sizeof(p),ex.scratch.data(),ex.scratch.size()),
            SBN3_FATAL_ARGUMENT,"float FMA plan scratch alias");
    if(p.small)return small_fma(out,a,b,c,ex,mode);
    require(ex.workers()>=p.workers,SBN3_FATAL_TEAM,"float FMA team width");
    const auto exponent=product_exponent(a,b);
    const bool negative=a.negative()!=b.negative();
    auto frame=Frame::external(ex.scratch.data(),ex.scratch.size());
    auto* base=static_cast<std::byte*>(frame.allocate(p.payload_bytes,p.alignment));
    auto* product=reinterpret_cast<std::uint64_t*>(base+p.product_offset);
    auto* pool=base+p.pool_offset;
    {
        // Product tables die before the same pool becomes the streaming
        // addition window. The exact product slot remains live throughout.
        auto tables=Frame::external(pool,p.tables_bytes);
        const auto program=product_program_prepare(p.product,tables);
        ProductCall call{&program,pool+p.work_offset,p.product.info.workspace_bytes,
            {a.magnitude().data(),p.a_limbs},{b.magnitude().data(),p.b_limbs},
            {product,p.product_capacity}};
        if(ex.scope)product_action(&call,ex.scope);
        else if(ex.team)sbn3_team_run(ex.team,product_action,&call);
        else serial_team_run(product_action,&call);
    }
    const auto value=float_view::from_magnitude({product,p.product_limbs},exponent,negative);
    return sum_finite(out,value,c,{{pool,p.addition_bytes},ex.team,ex.scope},mode);
}
}

namespace sbn {
requirements fma_requirements(float_ref out,float_view a,float_view b,float_view c,unsigned workers) noexcept {
    const auto p=out.precision().count;
    if(!p||p>float_precision_limit||!workers||workers>32)
        return {support::unsupported,0,64,0,"float FMA precision/workers"};
    const auto words=(p+63)/64;
    if(!a.finite()||!b.finite()||!c.finite()||a.zero()||b.zero())
        return {out.capacity()<words?support::capacity:support::yes,0,64,words,""};
    if(a.magnitude().size()<=6&&b.magnitude().size()<=6)
        return {out.capacity()<words?support::capacity:support::yes,
                v3::floating::sum_bytes(p,!c.zero()),64,words,""};
    return v3::floating::query_large(out,a,b,c,workers);
}
float_report fma_into(float_ref out,float_view a,float_view b,float_view c,
                      execution ex,rounding mode) noexcept {
    using namespace v3::floating;validate_arguments(out,a,b,c);
    float_report report;if(special_fma(out,a,b,c,ex,mode,report))return report;
    if(a.magnitude().size()<=6&&b.magnitude().size()<=6){
        v3::require(ex.scratch.size()>=sum_bytes(out.precision().count,!c.zero()),
                SBN3_FATAL_WORKSPACE,"small float FMA scratch",
                sum_bytes(out.precision().count,!c.zero()),ex.scratch.size());
        check_scratch(out,a,b,c,ex.scratch);return small_fma(out,a,b,c,ex,mode);
    }
    return large_fma(out,a,b,c,ex,mode);
}
}
