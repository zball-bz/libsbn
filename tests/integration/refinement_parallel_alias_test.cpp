#include "product_support.hpp"
#include "algorithms/refinement.hpp"
#include "runtime/team.hpp"
using namespace sbn::v3;
// D=B^n-B^(n-m), U=B^m+1 gives E=-B^(n-m). Its correction window is zero.
// These exact windows isolate output assembly while satisfying the same
// normalized reciprocal preconditions as an ordinary product provider.
struct ExactWindows {
    const uint64_t *rho,*zero;size_t rn,count;
    product::WindowResult cancel(unsigned,product::Span,product::Span,product::ShiftedSpan,
                                 product::Window w,product::MagnitudeBound){assert(w.words==rn);return {rho,rn,0,true};}
    product::WindowResult multiply(unsigned,product::Span,product::Span,product::Window w,
                                   product::MagnitudeBound){assert(w.words==count);return {zero,count,0,false};}
    product::WindowResult retain(product::WindowResult r,uint64_t *){return r;}
};
int main(){
    unsigned checked=0;
    for(unsigned workers:{1u,16u})for(unsigned overlap:{2u,3u,17u}){
        const size_t shift=(size_t(1)<<20)+3,m=shift+overlap-1,n=m+shift;
        Fixture f(workers,false);auto *d=f.guarded(n),*out=f.guarded(n+1),*rho=f.guarded(n-m+3),*zero=f.guarded(n-m+1);
        memset(d,0,n*8);std::fill(d+shift,d+n,UINT64_MAX);memset(out,0,(n+1)*8);out[0]=out[m]=1;
        memset(rho,0,(n-m+3)*8);memset(zero,0,(n-m+1)*8);
        if(shift>=m-2)rho[shift-(m-2)]=1;
        ExactWindows products{rho,zero,n-m+3,n-m+1};const auto epoch=f.team->epoch;
        allocation_watch_start();bounded_refinement<RefinementKind::Inverse>({m,n,nullptr,nullptr,f.team},products,nullptr,d,out,out);
        assert(!allocation_watch_stop());
        for(size_t j=0;j<=n;++j)assert(out[j]==uint64_t(j==shift||j==n));
        if(workers>1&&overlap<=3)assert(f.team->epoch>=epoch+3);
        ++checked;
    }
    printf("parallel in-place reciprocal assembly: %u exact-window/overlap/fallback cases PASS\n",checked);
}
