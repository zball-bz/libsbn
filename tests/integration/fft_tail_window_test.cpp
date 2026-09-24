#include "product_support.hpp"
#include "product/local_windows.hpp"
#include "algorithms/refinement.hpp"
using namespace sbn::v3;
using namespace sbn::v3::product;

static void kernel_gate(){
    unsigned cases=0;
    for(unsigned bits:{17u,18u,19u,20u})for(unsigned radix:{1u,3u,5u,7u})for(bool balanced:{false,true}){
        pq16::Shape shape{radix*128,128,radix,false,pq16::Recipe::CooleyTukeyPQ,bits,balanced};
        if(!pq16::cyclic_supported(shape,1,1))continue;
        for(;;){auto larger=shape;larger.branch*=2;larger.nfull*=2;
            if(!pq16::cyclic_supported(larger,1,1))break;shape=larger;}
        const size_t ring=pq16::cyclic_period(shape),an=ring/2-1,bn=ring;
        Fixture f(1);auto tables=f.allocate(pq16::table_bytes(shape)+256,128);
        auto work=f.allocate(pq16::scratch_bytes(shape,an,bn)+512,128),cache=f.allocate(16*shape.nfull+128,128);
        auto tf=Frame::external(tables.data,tables.bytes),wf=Frame::external(work.data,work.bytes);
        const auto *t=pq16::prepare(tf,shape);auto *a=f.guarded(an),*b=f.guarded(bn),*out=f.guarded(ring);
        for(unsigned pattern=0;pattern<4;++pattern){
            for(size_t j=0;j<an;++j)a[j]=pattern==0?0:pattern==1?UINT64_MAX:random_word();
            for(size_t j=0;j<bn;++j)b[j]=pattern==1?UINT64_MAX:pattern==2?(j&1?0:UINT64_MAX):random_word();
            ref_int A,B,P,R,E,H,Limit;ref_inits(A,B,P,R,E,H,Limit,nullptr);
            ref_import(A,an,-1,8,0,0,a);ref_import(B,bn,-1,8,0,0,b);ref_mul(P,A,B);
            ref_set_ui(R,1);ref_mul_2exp(R,R,64*ring);ref_sub_ui(R,R,1);ref_mod(P,P,R);
            ref_set_ui(Limit,1);ref_mul_2exp(Limit,Limit,40);
            auto *saved=static_cast<double*>(cache.data);pq16::forward_spectrum(saved,a,an,*t,nullptr);
            const std::vector<double> cache_copy(saved,saved+2*shape.nfull);
            for(size_t origin:{size_t(1),size_t(bits),size_t(bits+1),ring/2,ring/2+1})for(bool cached:{false,true}){
                const size_t untouched=bits*(origin/bits);
                for(size_t j=0;j<ring;++j)out[j]=0x8cb83a559633eea1ULL;
                SBN3_FRAME_POISON(out,8*untouched);
                allocation_watch_start();pq16::cyclic_tail_multiply(out,a,an,b,bn,cached?saved:nullptr,*t,wf,origin);
                assert(!allocation_watch_stop()&&!wf.used());SBN3_FRAME_UNPOISON(out,8*untouched);
                for(size_t j=0;j<untouched;++j)assert(out[j]==0x8cb83a559633eea1ULL);
                assert(!memcmp(saved,cache_copy.data(),16*shape.nfull));
                ref_fdiv_q_2exp(E,P,64*origin);ref_import(A,ring-origin,-1,8,0,0,out+origin);
                ref_sub(A,A,E);ref_set_ui(H,1);ref_mul_2exp(H,H,64*(ring-origin));ref_mod(A,A,H);
                ref_sub(B,H,A);if(ref_cmp(A,B)>0)ref_set(A,B);
                assert(ref_cmp(A,Limit)<0);++cases;
            }
            ref_clears(A,B,P,R,E,H,Limit,nullptr);
        }
    }
    printf("FFT tail raw modular error/cache/padding: %u cases PASS\n",cases);
}

static void cancellation_gate(){
    unsigned cases=0;
    for(size_t n:{2048ul,4096ul,6597ul,6889ul,10624ul}){
        const size_t m=newton_contract::next_precision(n),an=m+1,cn=an+n;
        auto shape=refinement_products<RefinementKind::Inverse>(m,n);
        shape.products[0].subtract={cn,0,false};
        const auto p=local_windows_query(shape);
        assert(p.cancel_shape.recipe==pq16::Recipe::CooleyTukeyPQ&&p.cancel_shape.bits>16);
        const auto w=shape.products[0].window;const auto bound=shape.products[0].bound;
        Fixture f(1);auto space=f.allocate(p.storage_bytes,128);auto *a=f.guarded(an),*b=f.guarded(n),*c=f.guarded(cn);
        for(unsigned pattern=0;pattern<3;++pattern){
            for(size_t j=0;j<an;++j)a[j]=pattern==0?UINT64_MAX:random_word();a[an-1]=1;
            for(size_t j=0;j<n;++j)b[j]=pattern==1?UINT64_MAX:random_word();b[n-1]|=uint64_t(1)<<63;
            ref_int A,B,P,C,E,W,G,D,Limit;ref_inits(A,B,P,C,E,W,G,D,Limit,nullptr);
            ref_import(A,an,-1,8,0,0,a);ref_import(B,n,-1,8,0,0,b);ref_mul(P,A,B);
            ref_set_ui(Limit,1);ref_mul_2exp(Limit,Limit,64);
            for(int residual=-3;residual<=3;++residual){
                ref_set_ui(E,unsigned(residual<0?-residual:residual));
                if(residual<=-2||residual>=2)ref_mul_2exp(E,E,64*w.origin);
                if(residual<0)ref_neg(E,E);ref_sub(C,P,E);
                memset(c,0,cn*8);ref_export(c,nullptr,-1,8,0,0,C);
                ref_abs(W,E);ref_fdiv_q_2exp(W,W,64*w.origin);if(residual<0)ref_neg(W,W);
                auto frame=Frame::external(space.data,space.bytes);
                {allocation_watch_start();LocalWindowProducts products(p,frame);
                 const auto result=products.cancel(0,{a,an},{b,n},{{c,cn},0},w,bound);
                 assert(!allocation_watch_stop()&&result.error_bits==64&&result.words==w.words);
                 ref_import(G,result.words,-1,8,0,0,result.data);if(result.negative)ref_neg(G,G);
                 ref_sub(D,G,W);ref_abs(D,D);assert(ref_cmp(D,Limit)<0);}
                assert(!frame.used()&&frame.peak()<=space.bytes);++cases;
            }
            ref_clears(A,B,P,C,E,W,G,D,Limit,nullptr);
        }
    }
    printf("FFT tail signed cancellation/near zero: %u independent cases PASS\n",cases);
}
int main(){kernel_gate();cancellation_gate();}
