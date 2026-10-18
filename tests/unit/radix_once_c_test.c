#include "sbn3/radix.h"
#include <assert.h>
#include <string.h>

int main(void){
    sbn3_radix_options options={0};options.workers=1;options.use_alphabet=1;
    memcpy(options.alphabet,"0123456789",10);
    unsigned char text[128];uint64_t word=42;
    sbn3_format_spec f={10,1,0,0,SBN3_RADIX_EXACT};sbn3_format_info fi;sbn3_format_once_result fr;
    assert(sbn3_format_once_query(&f,&options,&fi)==SBN3_SUPPORTED&&fi.storage_bytes==0&&fi.digit_bytes==64);
    sbn3_format_once(&f,&options,&word,text,0,&fr);
    assert(fr.integer_first==62&&fr.integer_digits==64&&text[62]=='4'&&text[63]=='2');
    for(unsigned j=0;j<62;++j)assert(text[j]=='0');
    word=UINT64_C(1)<<63;f.exponent2=-64;f.fraction_digits=3;
    sbn3_format_once(&f,&options,&word,text,0,&fr);
    assert(fr.integer_first==0&&fr.fraction_digits==3&&!memcmp(text,"500",3));
    _Alignas(64) uint64_t out[8];memset(out,0xa5,sizeof out);
    sbn3_parse_spec p={10,20,0,0};sbn3_parse_info pi;sbn3_parse_once_result pr;
    assert(sbn3_parse_once_query(&p,&options,&pi)==SBN3_SUPPORTED&&pi.storage_bytes==0&&pi.limbs==2);
    sbn3_parse_once(&p,&options,(const unsigned char*)"18446744073709551615",out,0,&pr);
    assert(pr.valid&&pr.limbs==2&&out[0]==UINT64_MAX&&out[1]==0&&out[2]==UINT64_C(0xa5a5a5a5a5a5a5a5));
    p.integer_digits=0;p.fraction_digits=3;p.fraction_bits=64;
    sbn3_parse_once(&p,&options,(const unsigned char*)"125",out,0,&pr);
    assert(pr.valid&&out[0]==(UINT64_C(1)<<61));
    sbn3_parse_once(&p,&options,(const unsigned char*)"1x5",out,0,&pr);
    assert(!pr.valid&&pr.limbs==0&&pr.invalid_index==1);
    options.workers=0;assert(sbn3_parse_once_query(&p,&options,&pi)==SBN3_UNSUPPORTED);
    options.workers=1;options.memory_budget=1;
    assert(sbn3_parse_once_query(&p,&options,&pi)==SBN3_SUPPORTED);
    p.integer_digits=1000;p.fraction_digits=p.fraction_bits=0;
    assert(sbn3_parse_once_query(&p,&options,&pi)==SBN3_QUERY_CAPACITY&&pi.storage_bytes>1);
    return 0;
}
