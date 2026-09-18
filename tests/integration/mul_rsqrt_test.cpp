#include "product_support.hpp"
#include "algorithms/mul_rsqrt.hpp"
#include <algorithm>
using namespace sbn::v3;
static void certificate(const uint64_t *q,const uint64_t *x,size_t n,uint64_t a) {
    ref_int Q,X,S,L,H;ref_inits(Q,X,S,L,H,nullptr);
    ref_import(Q,n+1,-1,8,0,0,q);ref_import(X,n+1,-1,8,0,0,x);ref_mul(S,Q,Q);
    if(ref_cmp_ui(X,mul_rsqrt_error)>=0){
        ref_sub_ui(L,X,mul_rsqrt_error);ref_mul(L,L,L);ref_mul_ui(L,L,a);assert(ref_cmp(L,S)<0);
    }
    ref_add_ui(H,X,mul_rsqrt_error);ref_mul(H,H,H);ref_mul_ui(H,H,a);assert(ref_cmp(H,S)>0);
    ref_clears(Q,X,S,L,H,nullptr);
}
static void one(size_t n,unsigned workers,uint64_t a,unsigned pattern) {
    Fixture f(workers,false);MulRsqrtPlan plan{};
    allocation_watch_start();assert(mul_rsqrt_query(n,workers,plan)==SBN3_SUPPORTED);assert(!allocation_watch_stop());
    const size_t offset=up(f.base+f.cursor,plan.alignment)-f.base;
    f.cursor=up(offset+plan.storage_bytes,4096)+4096;
    sbn3_error error{};assert(sbn3_arena_prepare(f.arena,offset,plan.storage_bytes,&error)==SBN3_OK);
    auto *values=static_cast<uint64_t *>(f.allocate((plan.value_words+16)*8,128).data);
    std::vector<uint64_t> q(n+1),previous(n+1);
    for(size_t j=0;j<=n;++j)q[j]=pattern==0?0:pattern==1?UINT64_MAX:random_word();
    for(unsigned repeat=0;repeat<2;++repeat){
        memset(values,0xa5,plan.value_words*8);
        memcpy(values+plan.input_at,q.data(),(n+1)*8);
        for(unsigned j=0;j<16;++j)values[plan.value_words+j]=0x38cf726318ab4475ULL;
        allocation_watch_start();auto out=mul_rsqrt_execute(plan,a,values,plan.value_words,*f.arena,offset,*f.team);
        assert(!allocation_watch_stop());assert(out.data==values+plan.result_at && out.count==n+1);
        assert(f.arena->unleased(offset,plan.storage_bytes));
        certificate(q.data(),out.data,n,a);
        if(repeat)assert(!memcmp(previous.data(),out.data,(n+1)*8));else memcpy(previous.data(),out.data,(n+1)*8);
        for(unsigned j=0;j<16;++j)assert(values[plan.value_words+j]==0x38cf726318ab4475ULL);
    }
    printf("mul rsqrt n=%zu W%u a=%llu pattern=%u: <2^38 ulp / alias / no allocation / rebind PASS; scratch=%zu values=%zu\n",n,workers,(unsigned long long)a,pattern,plan.storage_bytes,plan.value_words*8);fflush(stdout);
}
int main(){
    for(size_t n:{4u,5u,6u,7u,16u,63u,128u,512u})
        for(uint64_t a:{uint64_t(1),uint64_t(2),uint64_t(4),uint64_t(17),uint64_t(10005),uint64_t(1)<<32,UINT64_MAX})one(n,1,a,unsigned(a%3));
    for(size_t n:{4u,5u,128u,1024u})for(unsigned pattern:{1u,2u})one(n,1,UINT64_MAX,pattern);
    for(unsigned w:{3u,16u})for(size_t n:{8192u,65536u,65537u})one(n,w,10005,2);
}
