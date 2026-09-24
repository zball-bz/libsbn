#include "product_support.hpp"
#include "algorithms/local_divrem.hpp"
#include "runtime/scratch.hpp"
using namespace sbn::v3;
int main(){
    unsigned checked=0;
    for(size_t dn:{65ul,257ul,1025ul,2049ul,4097ul,8193ul,16384ul,32768ul})for(size_t divisor:{5ul,2ul}){
        const size_t nn=dn+(dn+divisor-1)/divisor;const auto p=local_fused_division_plan(nn,dn,2);
        Fixture f(1,false);auto lease=f.allocate(p.storage_bytes,128);
        auto *D=f.guarded(dn),*N=f.guarded(nn),*Q=f.guarded(nn-dn+1),*R=f.guarded(dn);
        ref_int d,n,q,r,x;ref_inits(d,n,q,r,x,nullptr);
        for(unsigned pattern=0;pattern<5;++pattern){
            for(size_t j=0;j<dn;++j)D[j]=pattern==2?UINT64_MAX:random_word();
            if(pattern==1)D[dn-1]=1;else D[dn-1]|=uint64_t(1)<<63;
            if(pattern==4){memset(D,0,dn*8);D[dn-1]=uint64_t(1)<<63;D[0]=1;}
            ref_import(d,dn,-1,8,0,0,D);
            auto frame=Frame::external(lease.data,p.storage_bytes);
            allocation_watch_start();const auto prepared=local_division_prepare(p,D,frame);assert(!allocation_watch_stop());
            const auto retained=frame.used();assert(retained<=p.persistent_bytes);memset(D,0,dn*8);
            for(size_t actual:{nn,nn-1,dn,dn-1}){
                for(size_t j=0;j<actual;++j)N[j]=pattern==2?UINT64_MAX:random_word();
                if(pattern==3){memset(N,0,nn*8);if(actual>=dn){
                    ref_set_ui(q,1);ref_mul_2exp(q,q,64*(actual-dn));ref_sub_ui(q,q,1);ref_mul(n,q,d);ref_export(N,nullptr,-1,8,0,0,n);}}
                ref_import(n,actual,-1,8,0,0,N);const size_t qn=actual>=dn?actual-dn+1:0;
                memset(Q,0xa5,(nn-dn+1)*8);memset(R,0xa5,dn*8);
                allocation_watch_start();local_division_apply(p,prepared,Q,R,N,actual,frame);assert(!allocation_watch_stop());
                assert(frame.used()==retained&&frame.peak()<=p.storage_bytes);
                ref_import(q,qn,-1,8,0,0,Q);ref_import(r,dn,-1,8,0,0,R);ref_mul(x,q,d);ref_add(x,x,r);
                assert(ref_cmp(x,n)==0&&ref_cmp(r,d)<0);++checked;
            }
        }
        ref_clears(d,n,q,r,x,nullptr);
    }
    printf("fused exact quotient/remainder: %u normalized/unnormalized/reprepared/shorter/exact-multiple cases PASS\n",checked);
}
