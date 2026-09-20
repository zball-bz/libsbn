// Exact guarded middle windows; independent full-product reference.
#include "product_support.hpp"
#include "backend/u52/kernels.hpp"
#include "runtime/scratch.hpp"
#include "radix/geometry.hpp"
using namespace sbn::v3;
int main(){
    unsigned checked=0,repairs=0,actual_carries=0;
    ref_int A,B,P,Q,E,difference,modulus;ref_inits(A,B,P,Q,E,difference,modulus,nullptr);
    for(size_t an:{4ul,9ul,17ul,33ul,65ul,97ul,128ul,193ul,257ul,298ul,300ul,
                   333ul,416ul,511ul,595ul,700ul,831ul,832ul})
      for(size_t bn:{an,an+7,an+an/2,an+an/3+1,an+127,2*an+1,4*an+3,8192ul}){
        const size_t na=(64*an+51)/52,nb=(64*bn+51)/52,words=(52*(nb-na+7)+63)/64;
        const uint64_t start=52*(na-5),window=64*(an-1),shift=window-start;
        const size_t width=bn-an+1;
        assert(shift>=128 && 2*na<(uint64_t(1)<<12));
        Fixture f(1);auto lease=f.allocate(u52::middle_guard_scratch_bytes(an,bn),64);Frame work(*f.arena,lease);
        auto *a=f.guarded(an),*b=f.guarded(bn),*m=f.guarded(words),*got=f.guarded(width);
        for(unsigned pattern=0;pattern<12;++pattern){
            for(size_t j=0;j<an;++j)a[j]=pattern==0?UINT64_MAX:pattern<3?0:random_word();
            for(size_t j=0;j<bn;++j)b[j]=pattern<2?UINT64_MAX:pattern==2?0:random_word();
            if(pattern==1)a[0]=1;
            allocation_watch_start();
            {ComputeLease computing(*f.arena);u52::middle_guard(m,a,an,b,bn,work);}
            assert(!allocation_watch_stop()&&!work.used());
            radix::take_window(got,width,m,words,shift);
            ref_import(A,an,-1,8,0,0,a);ref_import(B,bn,-1,8,0,0,b);ref_mul(P,A,B);
            ref_fdiv_q_2exp(Q,P,window);ref_fdiv_r_2exp(Q,Q,64*width);
            ref_import(E,width,-1,8,0,0,got);
            ref_set_ui(modulus,1);ref_mul_2exp(modulus,modulus,64*width);
            ref_sub(difference,Q,E);ref_mod(difference,difference,modulus);
            assert(ref_cmp_ui(difference,1)<=0);
            const bool repair=m[1]==UINT64_MAX;
            if(!repair)assert(!ref_sgn(difference));
            repairs+=repair;actual_carries+=ref_sgn(difference)!=0;++checked;
        }
    }
    ref_clears(A,B,P,Q,E,difference,modulus,nullptr);
    assert(repairs&&actual_carries);
    printf("middle windows: %u exact-or-one-low certificates, %u guarded repairs, %u actual carries PASS\n",checked,repairs,actual_carries);
}
