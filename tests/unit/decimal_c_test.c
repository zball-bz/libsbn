#include "sbn3/decimal.h"
#include <assert.h>
#include <string.h>
int main(void) {
    char buf[39]; uint64_t word=0; sbn3_uint128 wide={0,0};
    const uint64_t maximum=UINT64_MAX;
    char *end=sbn3_decimal_write_u64(buf,maximum);
    assert(end-buf==20&&!memcmp(buf,"18446744073709551615",20));
    assert(sbn3_decimal_read_u64(&word,buf,20)==SBN3_DECIMAL_OK&&word==maximum);
    end=sbn3_decimal_write_u128(buf,(sbn3_uint128){maximum,maximum});
    assert(end-buf==39&&!memcmp(buf,"340282366920938463463374607431768211455",39));
    assert(sbn3_decimal_read_u128(&wide,buf,39)==SBN3_DECIMAL_OK&&wide.lo==maximum&&wide.hi==maximum);
    uint64_t in[3]={0,0,1},back[3];char more[60];
    end=sbn3_decimal_write_small(more,in,3);
    assert(sbn3_decimal_read_small(back,3,more,(size_t)(end-more))==SBN3_DECIMAL_OK&&!memcmp(in,back,sizeof in));
    word=UINT64_C(1)<<63;
    assert(sbn3_decimal_write_fraction(buf,&word,1,2)==buf+2&&!memcmp(buf,"50",2));
    assert(sbn3_decimal_read_fraction(&word,1,"125",3)==SBN3_DECIMAL_OK&&word==(UINT64_C(1)<<61));
    return 0;
}
