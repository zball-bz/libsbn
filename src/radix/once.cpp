#include "radix/once.hpp"
#include "sbn3/decimal.h"
#include "radix/parse_tree.hpp"
#include "runtime/arena.hpp"
#include "runtime/team.hpp"
#include "common/small_checks.h"
#include <cstring>

namespace sbn::v3::radix {
namespace {
bool valid_options(const sbn3_radix_options *o) noexcept {
    return !o||(o->workers&&o->workers<=32&&o->use_alphabet<=1);
}
template<class Info>void check_workspace(const sbn3_radix_workspace *w,const Info &info) noexcept {
    require(w&&w->arena&&w->team&&w->bytes>=info.storage_bytes&&
        !(uintptr_t(w->arena->base+w->offset)&(info.storage_alignment-1)),
        SBN3_FATAL_WORKSPACE,"radix once workspace",info.storage_bytes,w?w->bytes:0);
}
[[gnu::noinline]] void format_planned(const sbn3_format_spec *s,const sbn3_radix_options *o,const uint64_t *in,
                                    uint8_t *out,const sbn3_radix_workspace *w,sbn3_format_once_result *result) noexcept {
    if(format_once_local(*s,o,in,out,w,result))return;
    sbn3_format_plan plan;sbn3_format_info info;
    require(sbn3_format_query(s,o,&plan,&info)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"format once request");
    check_workspace(w,info);sbn3_format_binding *b;
    sbn3_format_bind(&plan,w->arena,w->offset,w->team,&b);
    sbn3_format_result r;sbn3_format_execute(b,{in,s->limbs},out,&r);sbn3_format_unbind(b);
    *result={r.integer_first,info.integer_digits,r.fraction_digits,r.exact_fallback};
}
[[gnu::noinline]] void parse_planned(const sbn3_parse_spec *s,const sbn3_radix_options *o,const uint8_t *in,
                                   uint64_t *out,const sbn3_radix_workspace *w,sbn3_parse_once_result *result) noexcept {
    sbn3_parse_plan plan;sbn3_parse_info info;
    require(sbn3_parse_query(s,o,&plan,&info)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"parse once request");
    check_workspace(w,info);sbn3_parse_binding *b;
    sbn3_parse_bind(&plan,w->arena,w->offset,w->team,&b);
    sbn3_parse_result r;sbn3_parse_execute(b,in,{out,info.limbs},&r);sbn3_parse_unbind(b);
    *result={r.valid,r.invalid_index,r.valid?info.limbs:0,r.exact_fallback};
}
[[maybe_unused,gnu::noinline]] void format_check(const sbn3_format_spec *s,const sbn3_radix_options *o,const uint64_t *in,
                                  const uint8_t *out,const sbn3_radix_workspace *w,const sbn3_format_once_result *result) noexcept {
    sbn3_format_info info;
    require(s&&in&&out&&result&&sbn3_format_once_query(s,o,&info)==SBN3_SUPPORTED&&!(uintptr_t(in)&7)&&
        !overlaps(in,s->limbs*8,out,info.digit_bytes),SBN3_FATAL_ARGUMENT,"format once buffers");
    if(info.storage_bytes){check_workspace(w,info);const auto *work=w->arena->base+w->offset;
        require(!overlaps(work,info.storage_bytes,in,s->limbs*8)&&!overlaps(work,info.storage_bytes,out,info.digit_bytes),
            SBN3_FATAL_ARGUMENT,"format once workspace overlap");}
}
[[maybe_unused,gnu::noinline]] void parse_check(const sbn3_parse_spec *s,const sbn3_radix_options *o,const uint8_t *in,
                                 const uint64_t *out,const sbn3_radix_workspace *w,const sbn3_parse_once_result *result) noexcept {
    sbn3_parse_info info;
    require(s&&in&&out&&result&&sbn3_parse_once_query(s,o,&info)==SBN3_SUPPORTED&&!(uintptr_t(out)&63)&&
        !overlaps(in,s->integer_digits+s->fraction_digits,out,info.limbs*8),SBN3_FATAL_ARGUMENT,"parse once buffers");
    if(info.storage_bytes){check_workspace(w,info);const auto *work=w->arena->base+w->offset;
        require(!overlaps(work,info.storage_bytes,in,s->integer_digits+s->fraction_digits)&&!overlaps(work,info.storage_bytes,out,info.limbs*8),
            SBN3_FATAL_ARGUMENT,"parse once workspace overlap");}
}
[[gnu::noinline]] bool parse_integer_local(const sbn3_radix_workspace *w,uint64_t *out,size_t words,
                                          const uint8_t *in,size_t digits) noexcept {
    const size_t bytes=parse_local_bytes(words);OnceSpace work(w,bytes,64);
    auto frame=Frame::external(work.data,bytes);
    return parse_local(parse_leaf_plan(10),decimal_ascii_plan(),in,digits,out,frame);
}
}
// Larger groups own a temporary result array. Keep that frame out of both
// the u64/u128 dispatcher and the single-block path.
[[gnu::noinline]] bool once_integer_parse_kernel(uint64_t *out,size_t words,const uint8_t *in,size_t digits) noexcept {
    if(digits<=128)return parse_integer_blocks(out,words,in,digits,128,decimal_ascii_plan());
    return parse_leaf_exact(parse_leaf_plan(10),decimal_ascii_plan(),in,digits,out,words);
}
}
using namespace sbn::v3;
using namespace sbn::v3::radix;
extern "C" sbn3_query_result sbn3_format_once_query(const sbn3_format_spec *s,const sbn3_radix_options *o,sbn3_format_info *info){
    require(s&&info,SBN3_FATAL_ARGUMENT,"format once query");*info={};
    if(!valid_options(o)||unsigned(s->mode)>1||!s->limbs||s->fraction_digits>(uint64_t(1)<<40))return SBN3_UNSUPPORTED;
    const auto kind=format_once_kind(*s,o);
    if(kind==OnceKind::planned){
        sbn3_query_result rc;if(format_once_local_query(*s,o,*info,rc))return rc;
        sbn3_format_plan plan;return sbn3_format_query(s,o,&plan,info);}
    info->workers=1;info->storage_alignment=64;
    if(kind==OnceKind::integer){info->integer_digits=once_integer_width(s->limbs);info->fraction_offset=info->digit_bytes=info->integer_digits;}
    else{info->fraction_digits=s->fraction_digits;info->digit_bytes=(s->fraction_digits+63)&~uint64_t(63);}
    return SBN3_SUPPORTED;
}
extern "C" sbn3_query_result sbn3_parse_once_query(const sbn3_parse_spec *s,const sbn3_radix_options *o,sbn3_parse_info *info){
    require(s&&info,SBN3_FATAL_ARGUMENT,"parse once query");*info={};
    if(!valid_options(o))return SBN3_UNSUPPORTED;
    const auto kind=parse_once_kind(*s,o);
    if(kind==OnceKind::planned){sbn3_parse_plan plan;return sbn3_parse_query(s,o,&plan,info);}
    info->workers=1;info->storage_alignment=64;
    info->limbs=kind==OnceKind::integer?once_integer_limbs(s->integer_digits):s->fraction_bits/64;
    if(kind==OnceKind::integer&&s->integer_digits>512){info->storage_bytes=info->workspace_bytes=parse_local_bytes(info->limbs);info->lease_peak=SBN3_CHECK_SMALL?1:0;}
    info->exponent2=-int64_t(s->fraction_bits);
    if(o&&o->memory_budget&&info->storage_bytes>o->memory_budget)return SBN3_QUERY_CAPACITY;
    return SBN3_SUPPORTED;
}
extern "C" void sbn3_format_once(const sbn3_format_spec *s,const sbn3_radix_options *o,const uint64_t *in,uint8_t *out,
                                 const sbn3_radix_workspace *w,sbn3_format_once_result *result){
    if constexpr(SBN3_CHECK_SMALL)format_check(s,o,in,out,w,result);
    const auto kind=format_once_kind(*s,o);
    if(kind==OnceKind::integer){
        const size_t width=once_integer_width(s->limbs);
        const size_t first=s->limbs==1?format_limb(out,in[0],decimal_ascii_plan()):decimal_write_padded(out,width,in,s->limbs);
        *result={first,width,0,0};return;
    }
    if(kind==OnceKind::fraction){decimal_fraction_write(out,s->fraction_digits,in,s->limbs);*result={0,0,s->fraction_digits,0};return;}
    format_planned(s,o,in,out,w,result);
}
extern "C" void sbn3_parse_once(const sbn3_parse_spec *s,const sbn3_radix_options *o,const uint8_t *in,uint64_t *out,
                                const sbn3_radix_workspace *w,sbn3_parse_once_result *result){
    if constexpr(SBN3_CHECK_SMALL)parse_check(s,o,in,out,w,result);
    const auto kind=parse_once_kind(*s,o);bool valid;size_t words;
    if(kind==OnceKind::integer){
        const auto digits=s->integer_digits;
        words=digits<=38?1+unsigned(digits>19):once_integer_limbs(digits);
        if(digits<=38)valid=sbn3_decimal_read_small(out,words,reinterpret_cast<const char*>(in),digits)==SBN3_DECIMAL_OK;
        else if(digits<=512)valid=once_integer_parse_kernel(out,words,in,digits);
        else valid=parse_integer_local(w,out,words,in,digits);
    }else if(kind==OnceKind::fraction){words=s->fraction_bits/64;valid=decimal_fraction_read(out,words,in,s->fraction_digits);}
    else{parse_planned(s,o,in,out,w,result);return;}
    *result={unsigned(valid),valid?0:validate_digits(in,s->integer_digits+s->fraction_digits,10,decimal_ascii_plan().decode),valid?words:0,0};
}
