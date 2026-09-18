#include "product_support.hpp"
static void one(unsigned np,size_t n,unsigned workers){
    Fixture f(workers);sbn3_mul_options o{};o.workers=workers;o.prime_count=np;o.algorithm=SBN3_MUL_FLAT;
    sbn3_mul_info info{};auto *binding=f.bind(n,n,o,info);
    const bool packed=(np==8&&info.M2>=32768)||(np==10&&info.M2>=16384);
    if(packed)assert(info.per_worker_bytes<96*info.M2);
    auto *a=f.guarded(n),*b=f.guarded(n);const size_t rn=up(2*n,8);auto *out=f.guarded(rn);
    for(unsigned pattern=0;pattern<4;++pattern){
        for(size_t j=0;j<n;++j){a[j]=pattern==0?UINT64_MAX:random_word();b[j]=pattern==0?UINT64_MAX:random_word();}
        if(pattern==1)memset(b,0,n*8);if(pattern==2){a[n-1]=0;b[n-1]=0;}memset(out,0x57,rn*8);
        allocation_watch_start();sbn3_mul_execute(binding,{a,n},{b,n},{out,2*n});assert(!allocation_watch_stop());verify_product(a,n,b,n,out);
        for(size_t j=2*n;j<rn;++j)assert(out[j]==0x5757575757575757ULL);
        sbn3_mul_metrics m{};sbn3_mul_get_metrics(binding,&m);assert(m.worker_peak_bytes<=info.per_worker_bytes && m.workspace_used_bytes<=info.workspace_bytes);
    }
    printf("NP%u n%zu W%u M%zu packed%d: GMP/guard/one-array/worker placement PASS\n",np,n,workers,info.M2,packed);
}
int main(){for(unsigned np:{8u,10u})for(size_t n:{65536u,185364u,202140u,220436u})one(np,n,16);one(8,202140,3);one(10,202140,5);one(10,202140,4);one(10,202140,8);}
