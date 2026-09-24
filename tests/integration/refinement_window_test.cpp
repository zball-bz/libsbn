#include "product_support.hpp"
#include "algorithms/reciprocal.hpp"
#include "algorithms/inverse_seed.hpp"
#include "product/program_windows.hpp"
#include <algorithm>
using namespace sbn::v3;

// A full-product-only backend with no FFT/NTT/IFMA knowledge. The shared
// arithmetic recipe and generic window adapter must work unchanged on it.
struct PortableMultiply {
    unsigned calls = 0;
    static size_t workspace_bytes(size_t,size_t) noexcept {return 0;}
    void operator()(uint64_t *out, const uint64_t *a, size_t an,
                    const uint64_t *b, size_t bn, Frame &) noexcept {
        ++calls;memset(out, 0, (an+bn)*8);
        for(size_t i=0;i<an;++i){
            uint64_t carry=0;
            for(size_t j=0;j<bn;++j){
                const unsigned __int128 v=(unsigned __int128)a[i]*b[j]+out[i+j]+carry;
                out[i+j]=uint64_t(v);carry=uint64_t(v>>64);
            }
            out[i+bn]=carry;
        }
    }
};
void inverse(uint64_t *out,const uint64_t *d,size_t n,Frame &scratch,PortableMultiply &mul){
    product::FullWindowFactory<PortableMultiply> factory{mul};
    FrameMark mark(scratch);auto work=scratch.subframe(reciprocal_bytes(n,factory),64);
    reciprocal(out,d,n,work,factory);
}

void check(const uint64_t *q,const uint64_t *d,const uint64_t *a,size_t n){
    ref_int D,A,Q,E,L;ref_inits(D,A,Q,E,L,nullptr);
    ref_import(D,n,-1,8,0,0,d);ref_import(Q,n+1,-1,8,0,0,q);
    if(a){ref_import(A,n+1,-1,8,0,0,a);ref_mul_2exp(A,A,64*n);}
    else{ref_set_ui(A,1);ref_mul_2exp(A,A,128*n);}
    ref_mul(E,Q,D);ref_sub(E,E,A);ref_abs(E,E);ref_mul_ui(L,D,3);
    assert(ref_cmp(E,L)<0);ref_clears(D,A,Q,E,L,nullptr);
}
void window_gate(){
    Fixture f(1,false);auto lease=f.allocate(4096,64);auto scratch=Frame::external(lease.data,lease.bytes);
    auto *full=scratch.alloc<uint64_t>(24);PortableMultiply mul;
    product::FullProductWindows<PortableMultiply> windows(mul,full,24,scratch);
    uint64_t a[5],b[7],c[4],expected[24];ref_int A,B,C,E;ref_inits(A,B,C,E,nullptr);unsigned count=0;
    for(unsigned pattern=0;pattern<4;++pattern){
        for(auto &x:a)x=pattern==0?UINT64_MAX:random_word();
        for(auto &x:b)x=pattern==1?UINT64_MAX:random_word();
        for(auto &x:c)x=pattern==2?0:random_word();
        ref_import(A,5,-1,8,0,0,a);ref_import(B,7,-1,8,0,0,b);ref_mul(E,A,B);
        memset(expected,0,sizeof expected);ref_export(expected,nullptr,-1,8,0,0,E);
        for(size_t offset:{0ul,3ul,8ul}){
            allocation_watch_start();const auto w=windows.multiply(0,{a,5},{b,7},{offset,3,0});
            assert(!allocation_watch_stop()&&!w.negative&&!w.error_bits&&!memcmp(w.data,expected+offset,24));++count;
        }
        for(size_t shift:{0ul,8ul,16ul}){
            ref_import(C,4,-1,8,0,0,c);ref_mul_2exp(C,C,64*shift);ref_mul(E,A,B);ref_sub(E,E,C);
            const bool negative=ref_sgn(E)<0;ref_abs(E,E);
            memset(expected,0,sizeof expected);ref_export(expected,nullptr,-1,8,0,0,E);
            allocation_watch_start();const auto w=windows.cancel(0,{a,5},{b,7},{{c,4},shift},{0,24,0},{});
            assert(!allocation_watch_stop()&&w.negative==negative&&!w.error_bits&&!memcmp(w.data,expected,sizeof expected));++count;
        }
    }
    ref_clears(A,B,C,E,nullptr);printf("generic product windows/cancellation: %u independent integer cases PASS\n",count);
}
void program_gate(){
    unsigned checked=0;
    for(unsigned algorithm:{unsigned(SBN3_MUL_SCALAR),unsigned(SBN3_MUL_U52),unsigned(SBN3_MUL_PQ16),unsigned(SBN3_MUL_FLAT)})
    for(size_t n:{17ul,65ul,129ul,257ul}){
        Fixture f(1,false);auto *d=f.guarded(n),*a=f.guarded(n+1),*out=f.guarded(n+1);
        for(size_t j=0;j<n;++j){d[j]=random_word();a[j]=random_word();}d[n-1]|=uint64_t(1)<<63;a[n]=1;
        product::ProgramMultiply mul;mul.options.workers=1;mul.options.algorithm=algorithm;
        if(algorithm==SBN3_MUL_FLAT)mul.options.prime_count=6;
        product::FullWindowFactory<product::ProgramMultiply> factory{mul};
        const size_t bytes=std::max(reciprocal_bytes(n,factory),quotient_bytes(n,factory));
        auto lease=f.allocate(bytes,128);auto frame=Frame::external(lease.data,bytes);
        struct Job {size_t n;const uint64_t *a,*d;uint64_t *out;Frame *frame;product::ProgramMultiply *mul;bool divide;};
        for(bool divide:{false,true}){
            Job job{n,a,d,out,&frame,&mul,divide};allocation_watch_start();
            sbn3_team_run(f.team,[](void *ptr,sbn3_team_scope *scope){auto &j=*static_cast<Job*>(ptr);j.mul->scope=scope;
                product::FullWindowFactory<product::ProgramMultiply> provider{*j.mul};
                if(j.divide)quotient(j.out,j.a,j.d,j.n,*j.frame,provider);
                else reciprocal(j.out,j.d,j.n,*j.frame,provider);
            },&job);
            assert(!allocation_watch_stop()&&!frame.used()&&frame.peak()<=bytes);check(out,d,divide?a:nullptr,n);++checked;
        }
    }
    printf("ordinary product-program backend: %u unchanged reciprocal/quotient driver cases PASS\n",checked);
}
int main(){
    window_gate();
    program_gate();
    unsigned checked=0;
    for(size_t n:{4ul,5ul,7ul,15ul,16ul,17ul,31ul,32ul,33ul,65ul,129ul,257ul,511ul,513ul}){
        Fixture f(1,false);auto lease=f.allocate(256*n+4096,64);
        auto *d=f.guarded(n),*a=f.guarded(n+1),*q=f.guarded(n+1),*inplace=f.guarded(n+1);
        PortableMultiply mul;
        for(unsigned pattern=0;pattern<6;++pattern){
            for(size_t j=0;j<n;++j){d[j]=pattern<2?0:pattern==2?UINT64_MAX:random_word();a[j]=random_word();}
            d[n-1]|=uint64_t(1)<<63;if(pattern==1)d[0]=1;
            a[n]=pattern==3?0:1;if(pattern==4)memset(a,0,(n+1)*8);
            auto frame=Frame::external(lease.data,lease.bytes);
            allocation_watch_start();inverse(q,d,n,frame,mul);
            assert(!allocation_watch_stop()&&!frame.used());check(q,d,nullptr,n);++checked;
            {product::FullWindowFactory<PortableMultiply> factory{mul};FrameMark mark(frame);
             auto work=frame.subframe(quotient_bytes(n,factory),64);
             allocation_watch_start();quotient(q,a,d,n,work,factory);assert(!allocation_watch_stop());
             check(q,d,a,n);++checked;}

            const size_t m=newton_contract::next_precision(n),rn=newton_contract::residual_words(m,n);
            auto *u=frame.alloc<uint64_t>(n+1),*saved=frame.alloc<uint64_t>(m+1);
            auto *rho=frame.alloc<uint64_t>(rn),*coarse=frame.alloc<uint64_t>(m+1),*full=frame.alloc<uint64_t>(n+m+3);
            inverse(u,d+n-m,m,frame,mul);memcpy(saved,u,(m+1)*8);
            product::FullProductWindows<PortableMultiply> products(mul,full,n+m+3,frame);
            const unsigned before=mul.calls;
            allocation_watch_start();bounded_refinement<RefinementKind::Inverse>({m,n,nullptr,rho},products,nullptr,d,u,u);
            assert(!allocation_watch_stop()&&mul.calls==before+2);check(u,d,nullptr,n);++checked;
            memcpy(u,saved,(m+1)*8);memcpy(inplace,a,(n+1)*8);
            allocation_watch_start();bounded_refinement<RefinementKind::Quotient>({m,n,coarse,rho},products,a,d,u,q);
            assert(!allocation_watch_stop()&&mul.calls==before+5);check(q,d,a,n);++checked;
            allocation_watch_start();bounded_refinement<RefinementKind::Quotient>({m,n,coarse,inplace},products,inplace,d,u,inplace);
            assert(!allocation_watch_stop()&&mul.calls==before+8&&!memcmp(q,inplace,(n+1)*8));++checked;
            assert(frame.peak()<=lease.bytes);
        }
    }
    printf("generic refinement/full-product backend: %u strict error/alias/no-allocation cases PASS\n",checked);
}
