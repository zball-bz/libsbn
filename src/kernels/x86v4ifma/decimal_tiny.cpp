#include "decimal_tiny.hpp"
extern "C" [[gnu::always_inline]] char *sbn3_decimal_write_u64(char *out,uint64_t value) {
    return sbn::v3::tiny_decimal::write64(out,value);
}
extern "C" [[gnu::always_inline]] char *sbn3_decimal_write_u128(char *out,sbn3_uint128 value) {
    return sbn::v3::tiny_decimal::write128(out,value);
}
extern "C" [[gnu::always_inline]] sbn3_decimal_status sbn3_decimal_read_u64(uint64_t *out,const char *p,size_t n) {
    return sbn::v3::tiny_decimal::read64(out,p,n);
}
extern "C" [[gnu::always_inline]] sbn3_decimal_status sbn3_decimal_read_u128(sbn3_uint128 *out,const char *p,size_t n) {
    return sbn::v3::tiny_decimal::read128(out,p,n);
}
