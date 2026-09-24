#include "product_support.hpp"
#include "algorithms/fused_divrem.hpp"
#include "sbn3/divrem.h"
using namespace sbn::v3;
int main(){
    unsigned checked=0;
    for(unsigned workers:{1u,16u})for(size_t dn:{129ul,4097ul,32768ul,65537ul,131073ul,230195ul,387141ul,524288ul})for(size_t divisor:{5ul,2ul}){
        const size_t nn=dn+(dn+divisor-1)/divisor,qn=nn-dn+1;
        sbn3_divrem_request request{nn,dn};sbn3_divrem_options options{};options.workers=workers;options.timing=1;
        sbn3_divrem_plan plan{};sbn3_divrem_info info{};
        allocation_watch_start();const auto rc=fused_divrem_query(&request,&options,&plan,&info);assert(!allocation_watch_stop()&&rc==SBN3_SUPPORTED);
        if(dn>=8192){sbn3_divrem_plan automatic;sbn3_divrem_info selected{};
            assert(sbn3_divrem_query(&request,&options,&automatic,&selected)==SBN3_SUPPORTED&&selected.products==4);}
        const auto saved=plan;options.memory_budget=info.storage_bytes-1;sbn3_divrem_info need{};
        assert(fused_divrem_query(&request,&options,&plan,&need)==SBN3_QUERY_CAPACITY&&need.storage_bytes==info.storage_bytes&&!memcmp(&plan,&saved,sizeof plan));options.memory_budget=0;
        Fixture f(workers,false);const size_t at=up(f.base+f.cursor,info.storage_alignment)-f.base;f.cursor=at+info.storage_bytes;
        sbn3_error error{};assert(sbn3_arena_prepare(f.arena,at,info.storage_bytes,&error)==SBN3_OK);
        auto *D=f.guarded(dn),*N=f.guarded(nn),*Q=f.guarded(qn),*R=f.guarded(dn);
        sbn3_divrem_binding *b=nullptr;allocation_watch_start();sbn3_divrem_bind(&plan,f.arena,at,f.team,&b);assert(!allocation_watch_stop());
        ref_int d,n,q,r,x;ref_inits(d,n,q,r,x,nullptr);
        for(unsigned pattern=0;pattern<4;++pattern){
            for(size_t j=0;j<dn;++j)D[j]=pattern==2?UINT64_MAX:random_word();
            if(pattern==1)D[dn-1]=1;else D[dn-1]|=uint64_t(1)<<63;
            ref_import(d,dn,-1,8,0,0,D);
            allocation_watch_start();sbn3_divrem_prepare(b,{D,dn});assert(!allocation_watch_stop());memset(D,0,dn*8);
            sbn3_arena_stats before{};sbn3_arena_get_stats(f.arena,&before);
            for(size_t actual:{nn,nn-1,dn,dn-1}){
                for(size_t j=0;j<actual;++j)N[j]=pattern==2?UINT64_MAX:random_word();
                if(pattern==3){memset(N,0,nn*8);if(actual>=dn){ref_set_ui(q,1);ref_mul_2exp(q,q,64*(actual-dn));ref_sub_ui(q,q,1);ref_mul(n,q,d);ref_export(N,nullptr,-1,8,0,0,n);}}
                ref_import(n,actual,-1,8,0,0,N);sbn3_divrem_result result{};memset(Q,0xa5,qn*8);
                allocation_watch_start();sbn3_divrem_execute(b,{N,actual},{Q,qn},{R,dn},&result);assert(!allocation_watch_stop()&&result.corrections<=8);
                ref_import(q,result.quotient_limbs,-1,8,0,0,Q);ref_import(r,dn,-1,8,0,0,R);ref_mul(x,q,d);ref_add(x,x,r);assert(ref_cmp(n,x)==0&&ref_cmp(r,d)<0);
                sbn3_arena_stats after{};sbn3_arena_get_stats(f.arena,&after);assert(before.active_leases==after.active_leases);++checked;
            }
        }
        ref_clears(d,n,q,r,x,nullptr);sbn3_divrem_metrics metrics{};sbn3_divrem_get_metrics(b,&metrics);
        assert(metrics.prepares==4&&metrics.executes==16);allocation_watch_start();sbn3_divrem_unbind(b);assert(!allocation_watch_stop());
    }
    printf("general fused exact service: %u exact/reprepare/shorter-numerator/lease/budget cases PASS\n",checked);
}
