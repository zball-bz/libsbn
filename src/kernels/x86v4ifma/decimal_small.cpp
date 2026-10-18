#include "decimal_small.hpp"
#include "radix/decimal_small.hpp"
#include "common/small_checks.h"
#include <cassert>

namespace d=sbn::v3::tiny_decimal;
static inline void domain(size_t n) noexcept {
    if constexpr(SBN3_CHECK_SMALL)assert(n&&n<=SBN3_DECIMAL_SMALL_LIMBS);
}
// Keep the bounded multiword frames out of ordinary u64/u128 C calls too;
// ThinLTO is an optional further improvement, not required for this property.
[[gnu::noinline]] static char *write_multiple(char *out,const uint64_t *in,size_t n){
    uint64_t work[SBN3_DECIMAL_SMALL_LIMBS],chunks[2*SBN3_DECIMAL_SMALL_LIMBS];
    return d::write_words<19>(out,in,n,work,chunks);
}
[[gnu::noinline]] static sbn3_decimal_status read_multiple(uint64_t *out,size_t n,const char *in,size_t count){
    uint64_t work[SBN3_DECIMAL_SMALL_LIMBS];return d::read_words(out,n,in,count,work);
}
[[gnu::noinline]] static char *write_fraction_multiple(char *out,const uint64_t *in,size_t n,size_t count){
    uint64_t work[SBN3_DECIMAL_SMALL_LIMBS];return d::write_fraction_words<19>(out,in,n,count,work);
}
[[gnu::noinline]] static sbn3_decimal_status read_fraction_multiple(uint64_t *out,size_t n,const char *in,size_t count){
    alignas(64) uint64_t work[sbn::v3::radix::decimal_fraction_read_limbs];
    return d::read_fraction_words<19>(out,n,in,count,work);
}
extern "C" [[gnu::always_inline]] char *sbn3_decimal_write_small(char *out,const uint64_t *in,size_t n){
    domain(n);
    if(n==1)return d::write64(out,in[0]);
    if(n==2)return d::write128(out,{in[0],in[1]});
    return write_multiple(out,in,n);
}
extern "C" [[gnu::always_inline]] sbn3_decimal_status sbn3_decimal_read_small(uint64_t *out,size_t n,const char *in,size_t count){
    domain(n);
    if(n==1)return d::read64(out,in,count);
    if(n==2){sbn3_uint128 v;const auto status=d::read128(&v,in,count);if(status==SBN3_DECIMAL_OK)memcpy(out,&v,16);return status;}
    return read_multiple(out,n,in,count);
}
extern "C" [[gnu::always_inline]] char *sbn3_decimal_write_fraction(char *out,const uint64_t *in,size_t n,size_t count){
    domain(n);
    if(n==1)return d::write_fraction_fixed<1>(out,in,count);
    if(n==2)return d::write_fraction_fixed<2>(out,in,count);
    return write_fraction_multiple(out,in,n,count);
}
extern "C" [[gnu::always_inline]] sbn3_decimal_status sbn3_decimal_read_fraction(uint64_t *out,size_t n,const char *in,size_t count){
    domain(n);
    if(n==1)return d::read_fraction_fixed<1>(out,in,count);
    if(n==2)return d::read_fraction_fixed<2>(out,in,count);
    return read_fraction_multiple(out,n,in,count);
}
namespace sbn::v3::radix {
size_t decimal_write_padded(uint8_t *output,size_t width,const uint64_t *in,size_t n) noexcept {
    uint64_t work[decimal_write_limbs],chunks[2*decimal_write_limbs];auto *out=reinterpret_cast<char*>(output);
    const size_t used=size_t(d::write_words<19>(out,in,n,work,chunks)-out),first=width-used;
    memmove(out+first,out,used);memset(out,'0',first);return used==1&&out[first]=='0'?width:first;
}
void decimal_fraction_write(uint8_t *out,size_t count,const uint64_t *in,size_t n) noexcept {
    auto *text=reinterpret_cast<char*>(out);
    if(n==1){d::write_fraction_fixed<1>(text,in,count);return;}
    if(n==2){d::write_fraction_fixed<2>(text,in,count);return;}
    uint64_t work[decimal_fraction_write_limbs];d::write_fraction_words<19>(text,in,n,count,work);
}
bool decimal_fraction_read(uint64_t *out,size_t n,const uint8_t *in,size_t count) noexcept {
    auto *text=reinterpret_cast<const char*>(in);
    if(n==1)return d::read_fraction_fixed<1>(out,text,count)==SBN3_DECIMAL_OK;
    if(n==2)return d::read_fraction_fixed<2>(out,text,count)==SBN3_DECIMAL_OK;
    return read_fraction_multiple(out,n,text,count)==SBN3_DECIMAL_OK;
}
}
