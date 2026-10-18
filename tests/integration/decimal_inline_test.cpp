#include "sbn3/decimal.h"
#include <cassert>
#include <cstdio>

extern "C" [[gnu::noinline]] bool decimal_inline_roundtrip64(uint64_t x){
    char text[20];uint64_t out;
    const auto end=sbn3_decimal_write_u64(text,x);
    return sbn3_decimal_read_u64(&out,text,size_t(end-text))==SBN3_DECIMAL_OK&&out==x;
}
extern "C" [[gnu::noinline]] bool decimal_inline_roundtrip128(sbn3_uint128 x){
    char text[39];sbn3_uint128 out;
    const auto end=sbn3_decimal_write_u128(text,x);
    return sbn3_decimal_read_u128(&out,text,size_t(end-text))==SBN3_DECIMAL_OK&&out.lo==x.lo&&out.hi==x.hi;
}
extern "C" [[gnu::noinline]] char *decimal_inline_u8(char *out,uint8_t x){
    return sbn3_decimal_write_u64(out,x);
}
extern "C" [[gnu::noinline]] bool decimal_inline_fraction64(uint64_t x){
    char text[19];uint64_t back;
    sbn3_decimal_write_fraction(text,&x,1,19);
    const auto status=sbn3_decimal_read_fraction(&back,1,text,19);
    return status==SBN3_DECIMAL_OK&&back<=x&&x-back<=2;
}
extern "C" [[gnu::noinline]] bool decimal_inline_fraction128(sbn3_uint128 x){
    char text[128];uint64_t input[2]{x.lo,x.hi},back[2];
    sbn3_decimal_write_fraction(text,input,2,128);
    return sbn3_decimal_read_fraction(back,2,text,128)==SBN3_DECIMAL_OK&&back[0]==x.lo&&back[1]==x.hi;
}
int main(int argc,char **){
    uint64_t x=uint64_t(argc)+0x748131a562b98203ULL;
    for(unsigned j=0;j<10000;++j){x^=x<<13;x^=x>>7;x^=x<<17;
        assert(decimal_inline_roundtrip64(x));assert(decimal_inline_roundtrip128({x,x^(x>>20)}));
        assert(decimal_inline_fraction64(x));assert(decimal_inline_fraction128({x,x^(x>>20)}));}
    for(unsigned j=0;j<256;++j){char a[20],b[20];
        auto *ae=decimal_inline_u8(a,uint8_t(j)),*be=sbn3_decimal_write_u64(b,j);
        assert(ae-a==be-b);for(size_t k=0;k<size_t(ae-a);++k)assert(a[k]==b[k]);}
    puts("public-header decimal LTO roundtrip PASS");
}
