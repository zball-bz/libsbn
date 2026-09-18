#include "common/small_checks.h"
#include "sbn3/base.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>

void ref_mul(uint64_t *, const uint64_t *, size_t, const uint64_t *, size_t);
static uint64_t seed = 0x36ec39ab61958bc1ULL;
static uint64_t rnd(void) { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; }

int main(void) {
    uint64_t a[129], b[129], r[260], want[260];
    unsigned cases = 0;
    assert(sbn3_add_n(NULL,NULL,NULL,0)==0 && sbn3_sub_n(NULL,NULL,NULL,0)==0);
    for (size_t an=0;an<=128;++an) for(size_t bn=0;bn<=128;++bn) {
        for(int pattern=0;pattern<4;++pattern) {
            for(size_t i=0;i<an;++i) a[i]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(i&1?0:UINT64_MAX):rnd();
            for(size_t i=0;i<bn;++i) b[i]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(i&1?UINT64_MAX:0):rnd();
            memset(r,0xa5,sizeof r); memset(want,0,sizeof want);
            ref_mul(want,a,an,b,bn);
            sbn3_mul_basecase(r,an+bn,a,an,b,bn);
            assert(memcmp(r,want,(an+bn)*8)==0);
            assert(r[an+bn]==0xa5a5a5a5a5a5a5a5ULL);
            ++cases;
        }
    }
    for (size_t n=1;n<=128;++n) for(int k=0;k<64;++k) {
        uint64_t cy=0,br=0;
        for(size_t i=0;i<n;++i) {
            a[i]=rnd();b[i]=rnd();
            unsigned __int128 v=(unsigned __int128)a[i]+b[i]+cy;
            want[i]=(uint64_t)v;cy=(uint64_t)(v>>64);
        }
        assert(sbn3_add_n(r,a,b,n)==cy && !memcmp(r,want,n*8));
        memcpy(r,a,n*8);assert(sbn3_add_n(r,r,b,n)==cy && !memcmp(r,want,n*8));
        for(size_t i=0;i<n;++i) {
            unsigned __int128 sub=(unsigned __int128)b[i]+br;
            want[i]=a[i]-(uint64_t)sub;br=(unsigned __int128)a[i]<sub;
        }
        assert(sbn3_sub_n(r,a,b,n)==br && !memcmp(r,want,n*8));
        memcpy(r,b,n*8);assert(sbn3_sub_n(r,a,r,n)==br && !memcmp(r,want,n*8));
    }
    for(unsigned sa=0;sa<2;++sa)for(unsigned sb=0;sb<2;++sb) {
        a[0]=UINT64_MAX;a[1]=0;b[0]=17;
        sbn3_int out={r,260,0,0};
        sbn3_int_mul_basecase(&out,(sbn3_int_view){a,2,sa},(sbn3_int_view){b,1,sb});
        assert(out.size==2 && out.negative==(sa^sb) && r[0]==UINT64_MAX-16 && r[1]==16);
        a[0]=0;sbn3_int_mul_basecase(&out,(sbn3_int_view){a,2,sa},(sbn3_int_view){b,1,sb});
        assert(out.size==0 && out.negative==0);
        sbn3_int_mul_basecase(&out,(sbn3_int_view){NULL,0,sa},(sbn3_int_view){b,1,sb});
        assert(out.size==0 && out.negative==0);
    }
    for(size_t n=8191;n<=8193;++n){
        uint64_t *x=malloc(n*8),*got=malloc((n+3)*8),*ref=malloc((n+2)*8);assert(x&&got&&ref);
        for(unsigned pattern=0;pattern<4;++pattern){uint64_t y[2]={pattern?rnd():UINT64_MAX,pattern?rnd():UINT64_MAX};
            for(size_t j=0;j<n;++j)x[j]=pattern==0?UINT64_MAX:pattern==1?0:rnd();memset(ref,0,(n+2)*8);ref_mul(ref,x,n,y,2);
            got[n+2]=0x239ae51ca57ffdddULL;sbn3_mul_basecase(got,n+2,x,n,y,2);assert(!memcmp(got,ref,(n+2)*8)&&got[n+2]==0x239ae51ca57ffdddULL);
            sbn3_mul_basecase(got,n+2,y,2,x,n);assert(!memcmp(got,ref,(n+2)*8)&&got[n+2]==0x239ae51ca57ffdddULL);}
        free(x);free(got);free(ref);
    }
#if SBN3_CHECK_SMALL
    pid_t child=fork();assert(child>=0);
    if(!child) {
        struct rlimit lim={0,0};setrlimit(RLIMIT_CORE,&lim);
        sbn3_mul_basecase(a,4,a,2,b,2);_exit(99);
    }
    int status;assert(waitpid(child,&status,0)==child);
    assert(WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);
#endif
    printf("word: %u products, small checks=%d, add/sub alias gates OK; %s\n",cases,SBN3_CHECK_SMALL,sbn3_build_version());
    return 0;
}
