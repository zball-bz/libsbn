#include "product_support.hpp"
#include "algorithms/local_inverse.hpp"
#include "runtime/scratch.hpp"
using namespace sbn::v3;
int main(){
    unsigned checked=0;
    for(size_t n:{4ul,5ul,7ul,16ul,17ul,31ul,65ul,129ul,257ul,511ul,513ul,1023ul,1025ul,2049ul,4097ul,8193ul,16384ul,16387ul,32766ul}){
        Fixture f(1,false);const size_t bytes=local_divide_bytes(n);auto work=f.allocate(bytes,128);
        auto *d=f.guarded(n),*a=f.guarded(n+1),*out=f.guarded(n+1),*copy=f.guarded(n+1);
        for(unsigned pattern=0;pattern<5;++pattern){
            for(size_t j=0;j<n;++j){d[j]=pattern==1?0:pattern==2?UINT64_MAX:random_word();a[j]=random_word();}
            d[n-1]|=uint64_t(1)<<63;a[n]=pattern==3?0:1;
            if(pattern==1&&n>1)d[0]=1;
            if(pattern==4)memset(a,0,(n+1)*8);
            memcpy(copy,a,(n+1)*8);memset(out,0xa5,(n+1)*8);
            auto frame=Frame::external(work.data,bytes);
            allocation_watch_start();local_divide(out,a,d,n,frame);
            assert(!allocation_watch_stop()&&!frame.used()&&frame.peak()<=bytes);
            ref_int D,A,Q,E,L;ref_inits(D,A,Q,E,L,nullptr);
            ref_import(D,n,-1,8,0,0,d);ref_import(A,n+1,-1,8,0,0,a);ref_import(Q,n+1,-1,8,0,0,out);
            ref_mul_2exp(A,A,64*n);ref_mul(E,Q,D);ref_sub(E,E,A);ref_abs(E,E);ref_mul_ui(L,D,3);
            assert(ref_cmp(E,L)<0);ref_clears(D,A,Q,E,L,nullptr);
            allocation_watch_start();local_divide(copy,copy,d,n,frame);assert(!allocation_watch_stop());
            assert(!memcmp(copy,out,(n+1)*8));++checked;
        }
    }
    printf("local fused quotient: %u strict <3-ulp / consumed numerator / scratch bounds cases PASS\n",checked);
}
