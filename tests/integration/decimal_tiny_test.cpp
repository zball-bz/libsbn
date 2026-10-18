#include "sbn3/decimal.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using W=__uint128_t;
static uint64_t state=0x1370ace95ba72149ULL;
static uint64_t random_word(){state^=state<<13;state^=state>>7;state^=state<<17;return state;}
static unsigned reference(char *out,W x){
    char r[40];unsigned n=0;do{r[n++]=char('0'+x%10);x/=10;}while(x);
    for(unsigned j=0;j<n;++j)out[j]=r[n-1-j];return n;
}
static void check(W x){
    char out[64],expected[40];const unsigned n=reference(expected,x);memset(out,0x7f,sizeof out);
    const sbn3_uint128 v{uint64_t(x),uint64_t(x>>64)};
    sbn3_uint128 back{91,92};
    allocation_watch_start();
    auto *end=sbn3_decimal_write_u128(out+1,v);
    const auto ok=sbn3_decimal_read_u128(&back,out+1,n);
    assert(!allocation_watch_stop());
    assert(end==out+n+1&&!memcmp(out+1,expected,n));
    assert(out[0]==0x7f&&out[n+1]==0x7f&&ok==SBN3_DECIMAL_OK);
    assert(back.lo==v.lo&&back.hi==v.hi);
    if(!v.hi){
        memset(out,0x7f,sizeof out);uint64_t u=91;
        allocation_watch_start();end=sbn3_decimal_write_u64(out+1,v.lo);
        const auto status=sbn3_decimal_read_u64(&u,out+1,n);
        assert(!allocation_watch_stop());assert(end==out+n+1&&!memcmp(out+1,expected,n));
        assert(out[0]==0x7f&&out[n+1]==0x7f&&status==SBN3_DECIMAL_OK&&u==v.lo);
    }
}
int main(){
    for(unsigned j=0;j<100000;++j)check(j);
    for(unsigned k=0;k<128;++k){const W x=W(1)<<k;check(x);check(x-1);check(x+1);}
    W p=1;for(unsigned k=0;k<=38;++k){check(p);check(p-1);check(p+1);p*=10;}
    check(~W(0));check(~W(0)-1);
    for(unsigned j=0;j<50000;++j){check(random_word());check((W(random_word())<<64)|random_word());}
    uint64_t lo=17;sbn3_uint128 wide{18,19};
    assert(sbn3_decimal_read_u64(&lo,"18446744073709551616",20)==SBN3_DECIMAL_OVERFLOW&&lo==17);
    assert(sbn3_decimal_read_u128(&wide,"340282366920938463463374607431768211456",39)==SBN3_DECIMAL_OVERFLOW&&wide.lo==18&&wide.hi==19);
    assert(sbn3_decimal_read_u64(&lo,nullptr,0)==SBN3_DECIMAL_INVALID);
    char zeros[80];memset(zeros,'0',sizeof zeros);
    assert(sbn3_decimal_read_u64(&lo,zeros,sizeof zeros)==SBN3_DECIMAL_OK&&lo==0);
    assert(sbn3_decimal_read_u128(&wide,zeros,sizeof zeros)==SBN3_DECIMAL_OK&&!wide.lo&&!wide.hi);
    zeros[0]='1';assert(sbn3_decimal_read_u128(&wide,zeros,sizeof zeros)==SBN3_DECIMAL_OVERFLOW);
    zeros[79]='x';assert(sbn3_decimal_read_u128(&wide,zeros,sizeof zeros)==SBN3_DECIMAL_INVALID);
    const size_t page=size_t(sysconf(_SC_PAGESIZE));
    auto *p0=static_cast<char*>(mmap(nullptr,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    assert(p0!=MAP_FAILED&&mprotect(p0+page,page,PROT_NONE)==0);
    for(unsigned n=1;n<=39;++n){
        char *in=p0+page-n;memset(in,'1',n);
        assert(sbn3_decimal_read_u128(&wide,in,n)==SBN3_DECIMAL_OK);
        for(unsigned at=0;at<n;++at)for(unsigned c=0;c<256;++c){
            in[at]=char(c);const auto status=sbn3_decimal_read_u128(&wide,in,n);
            if(c<'0'||c>'9')assert(status==SBN3_DECIMAL_INVALID);
            else assert(status!=SBN3_DECIMAL_INVALID);
            if(n<=20){const auto s=sbn3_decimal_read_u64(&lo,in,n);
                if(c<'0'||c>'9')assert(s==SBN3_DECIMAL_INVALID);else assert(s!=SBN3_DECIMAL_INVALID);}
            in[at]='1';
        }
        W x=1;for(unsigned j=1;j<n;++j)x*=10;
        auto *end=sbn3_decimal_write_u128(in,{uint64_t(x),uint64_t(x>>64)});
        assert(end==p0+page);
    }
    assert(munmap(p0,2*page)==0);
    puts("decimal tiny: boundaries, 200k values, overflow/invalid priority, guard pages and no allocation PASS");
}
