#include "sbn3/decimal.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using W=__uint128_t;
using Value=std::array<uint64_t,8>;
static uint64_t seed=0x43210450abba9876ULL;
static uint64_t random_word(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static size_t integer_reference(char *out,Value a,size_t n){
    char reverse[160];size_t count=0;
    do{uint64_t carry=0;for(size_t j=n;j-->0;){const W x=(W(carry)<<64)|a[j];a[j]=uint64_t(x/10);carry=uint64_t(x%10);}
        reverse[count++]=char('0'+carry);while(n&&a[n-1]==0)--n;}while(n);
    for(size_t j=0;j<count;++j)out[j]=reverse[count-1-j];return count;
}
static void format_fraction_reference(char *out,Value a,size_t n,size_t count){
    for(size_t k=0;k<count;++k){uint64_t carry=0;
        for(size_t j=0;j<n;++j){const W x=W(a[j])*10+carry;a[j]=uint64_t(x);carry=uint64_t(x>>64);}
        out[k]=char('0'+carry);}
}
static Value parse_fraction_reference(const char *in,size_t count,size_t n){
    Value a{};
    for(size_t k=count;k-->0;){uint64_t carry=uint64_t(in[k]-'0');
        for(size_t j=n;j-->0;){const W x=(W(carry)<<64)|a[j];a[j]=uint64_t(x/10);carry=uint64_t(x%10);}}
    return a;
}
static void check(Value a,size_t n,size_t digits){
    char out[1026],gold[1024];Value back;back.fill(0xdeadbeef);
    const size_t length=integer_reference(gold,a,n);memset(out,0x7f,sizeof out);
    allocation_watch_start();
    auto *end=sbn3_decimal_write_small(out+1,a.data(),n);
    auto status=sbn3_decimal_read_small(back.data(),n,out+1,length);
    assert(!allocation_watch_stop());
    assert(end==out+1+length&&!memcmp(out+1,gold,length)&&out[0]==0x7f&&out[length+1]==0x7f);
    if(status!=SBN3_DECIMAL_OK||memcmp(back.data(),a.data(),n*8)){
        fprintf(stderr,"integer parse n=%zu length=%zu status=%u text=%.*s\n",n,length,unsigned(status),int(length),out+1);
        for(size_t j=0;j<n;++j)fprintf(stderr,"word %zu: %016lx != %016lx\n",j,back[j],a[j]);
        assert(false);}
    format_fraction_reference(gold,a,n,digits);memset(out,0x7f,sizeof out);
    allocation_watch_start();end=sbn3_decimal_write_fraction(out+1,a.data(),n,digits);
    assert(!allocation_watch_stop());
    assert(end==out+1+digits&&!memcmp(out+1,gold,digits)&&out[0]==0x7f&&out[digits+1]==0x7f);
    const Value expected=parse_fraction_reference(gold,digits,n);
    allocation_watch_start();status=sbn3_decimal_read_fraction(back.data(),n,out+1,digits);
    assert(!allocation_watch_stop());
    assert(status==SBN3_DECIMAL_OK&&!memcmp(back.data(),expected.data(),n*8));
    if(digits>=64*n)assert(!memcmp(back.data(),a.data(),n*8));
    for(size_t j=0;j<digits;++j)out[j+1]=char('0'+random_word()%10);
    const auto random_expected=parse_fraction_reference(out+1,digits,n);
    assert(sbn3_decimal_read_fraction(back.data(),n,out+1,digits)==SBN3_DECIMAL_OK&&!memcmp(back.data(),random_expected.data(),n*8));
}
int main(){
    for(size_t n=1;n<=8;++n){
        for(size_t digits:{0u,1u,7u,8u,15u,16u,17u,19u,20u,31u,32u,33u,38u,39u,40u,63u,64u,65u,127u,128u,129u,511u,512u,1024u}){
            for(unsigned c=0;c<12;++c){Value a{};
                for(size_t j=0;j<n;++j)a[j]=c==0?0:c==1?~uint64_t(0):c==2?uint64_t(1)<<63:random_word();
                check(a,n,digits);}}
        Value a{};for(unsigned bit=0;bit<64*n;++bit){a={};a[bit/64]=uint64_t(1)<<(bit%64);check(a,n,64*n);}
        // The first integer beyond capacity, represented independently.
        char overflow[161];Value maximum{};for(size_t j=0;j<n;++j)maximum[j]=UINT64_MAX;
        size_t count=integer_reference(overflow,maximum,n);size_t k=count;
        while(k&&overflow[k-1]=='9')overflow[--k]='0';assert(k);++overflow[k-1];
        Value out;out.fill(71);const auto before=out;
        assert(sbn3_decimal_read_small(out.data(),n,overflow,count)==SBN3_DECIMAL_OVERFLOW&&out==before);
        overflow[count-1]='x';assert(sbn3_decimal_read_small(out.data(),n,overflow,count)==SBN3_DECIMAL_INVALID&&out==before);
        for(unsigned c=0;c<256;++c)if(c<'0'||c>'9'){
            char text[65];memset(text,'9',sizeof text);
            for(unsigned pos:{0u,15u,16u,31u,32u,64u}){text[pos]=char(c);
                assert(sbn3_decimal_read_fraction(out.data(),n,text,sizeof text)==SBN3_DECIMAL_INVALID&&out==before);
                text[pos]='9';}}
    }
    const size_t page=size_t(sysconf(_SC_PAGESIZE));
    char *p=static_cast<char*>(mmap(nullptr,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    assert(p!=MAP_FAILED&&mprotect(p+page,page,PROT_NONE)==0);
    for(size_t n=1;n<=8;++n)for(size_t count=1;count<=160;++count){
        char *text=p+page-count;memset(text,'1',count);Value a{};
        assert(sbn3_decimal_read_fraction(a.data(),n,text,count)==SBN3_DECIMAL_OK);
        auto expected=parse_fraction_reference(text,count,n);assert(a==expected);
        assert(sbn3_decimal_write_fraction(text,a.data(),n,count)==p+page);
    }
    assert(munmap(p,2*page)==0);
    puts("decimal small: 1..8 limbs, exact fractions, overflow, invalid bytes, guard pages, allocation freedom PASS");
}
