// Short high-half contract against the oracle: n words r with
// 0 <= floor(A*B/2^(64n)) - r <= 1. The L0 header kernels (every variant,
// n = 1..8), u52::mulhigh and u52::sqrhigh (every n in 1..mulhigh_limit, which
// includes all truncation-column classes K = t and K = t-1), for distinct
// inputs, a == b and the explicit square. The u52 entries run with inputs and
// output ending at a guard page and at interior unaligned offsets, on scratch
// filled with garbage, without allocation, within the queried scratch.
// Words are compared only between kernels with the same kept terms: the scalar
// kernels truncate at a 2^64 column, the u52 kernels at a 2^52 digit column, so
// the two families may differ by one (both within the contract, each gated by
// the oracle alone). Same terms: the L0 variants (default/asm/u128) and squares
// with mulhigh(a, a) within the scalar family; mulhigh(a, a) with mulhigh(a, copy
// of a) at every n (one family per n, a == b takes that family's square
// kernel); sqrhigh with mulhigh(a, a) where both run the same family (n <= 14
// scalar, n >= 21 u52; at 15..20 sqrhigh is scalar and mulhigh u52).
#include "../integration/product_support.hpp"
#include "runtime/scratch.hpp"
#include "backend/u52/kernels.hpp"
#include "core/x86_64/gen/mulhigh.hpp"
using namespace sbn::v3;

// floor(A*B/2^(64n)) by the oracle
static std::vector<uint64_t> exact_high(const uint64_t *a,const uint64_t *b,size_t n){
    ref_int A,B,P;ref_inits(A,B,P,nullptr);
    ref_import(A,n,-1,8,0,0,a);ref_import(B,n,-1,8,0,0,b);ref_mul(P,A,B);ref_fdiv_q_2exp(P,P,64*n);
    std::vector<uint64_t> h(n,0);size_t count=0;ref_export(h.data(),&count,-1,8,0,0,P);
    ref_clears(A,B,P,nullptr);assert(count<=n);
    return h;
}
static bool contract(const uint64_t *a,const uint64_t *b,const uint64_t *r,size_t n){
    const auto h=exact_high(a,b,n);std::vector<uint64_t> d(n);
    if(ref_sub_n(d.data(),h.data(),r,n) || d[0]>1)return false;
    for(size_t i=1;i<n;++i)if(d[i])return false;
    return true;
}
// 0 random, 1 all ones (largest omission), 2 low half ones, 3 near ones,
// 4 leading zero limbs, 5 sparse
static void fill(uint64_t *x,size_t n,unsigned mode){
    for(size_t i=0;i<n;++i){
        uint64_t w=random_word();
        if(mode==1)w=UINT64_MAX;
        else if(mode==2 && i<n/2)w=UINT64_MAX;
        else if(mode==3)w=UINT64_MAX-(w&0xff);
        else if(mode==4 && i>=n-(n+3)/4)w=0;
        else if(mode==5)w=(w&7)?0:random_word();
        x[i]=w;
    }
}
// variant 0 default, 1 asm (n >= 3), 2 u128 (n <= 4)
template<int n> static bool l0_run(unsigned v,bool sq,uint64_t *r,const uint64_t *a,const uint64_t *b){
    if(v==0){if(sq)l0::sqrhigh<n>(r,a);else l0::mulhigh<n>(r,a,b);return true;}
    if constexpr(n>=3)if(v==1){if(sq)l0::sqrhigh_asm<n>(r,a);else l0::mulhigh_asm<n>(r,a,b);return true;}
    if constexpr(n<=4)if(v==2){if(sq)l0::sqrhigh_u128<n>(r,a);else l0::mulhigh_u128<n>(r,a,b);return true;}
    return false;
}
template<int n> static void l0_case(size_t &checks){
    constexpr uint64_t canary=0x5a5a5a5a5a5a5a5aull;
    for(unsigned mode=0;mode<6;++mode)for(int rep=0;rep<500;++rep){
        uint64_t a[n],b[n],want[n];fill(a,n,mode);fill(b,n,mode);
        for(unsigned same=0;same<2;++same){
            const uint64_t *y=same?a:b;
            l0::mulhigh<n>(want,a,y);assert(contract(a,y,want,n));
            for(unsigned sq=0;sq<=same;++sq)for(unsigned v=0;v<3;++v){
                uint64_t r[n+4];for(auto &w:r)w=canary;
                if(!l0_run<n>(v,sq,r,a,y))continue;
                assert(!memcmp(r,want,8*n));
                for(size_t i=n;i<n+4;++i)assert(r[i]==canary);
                ++checks;
            }
        }
    }
}
int main(){
    size_t l0_checks=0,entry_checks=0;
    l0_case<1>(l0_checks);l0_case<2>(l0_checks);l0_case<3>(l0_checks);l0_case<4>(l0_checks);
    l0_case<5>(l0_checks);l0_case<6>(l0_checks);l0_case<7>(l0_checks);l0_case<8>(l0_checks);

    constexpr size_t L=u52::mulhigh_limit;constexpr uint64_t canary=0xc3c3c3c3c3c3c3c3ull;
    Fixture f(1);const size_t scratch=u52::mulhigh_scratch_bytes(L);
    auto memory=f.allocate(scratch,64);
    uint64_t *ga=f.guarded(L),*gb=f.guarded(L),*gr=f.guarded(L),*gc=f.guarded(L);
    auto *pool=static_cast<uint64_t *>(f.allocate(4*8*(L+16),64).data);
    size_t classes[2]={0,0};
    for(size_t n=1;n<=L;++n){
        const size_t t=64*n/52;classes[(2*t-1<=(1ull<<(64*n%52)))?0:1]++;
        const size_t need=u52::mulhigh_scratch_bytes(n);assert(need<=scratch);
        for(unsigned layout=0;layout<2;++layout){
            // 0: every span ends at a guard page; 1: interior, unaligned, with canaries
            const size_t s=1+n%7;
            uint64_t *a=layout?pool+s:ga+L-n,*b=layout?pool+(L+16)+s+1:gb+L-n,
                     *r=layout?pool+2*(L+16)+s+2:gr+L-n,*c=layout?pool+3*(L+16)+s+3:gc+L-n;
            for(unsigned mode=0;mode<6;++mode)for(int rep=0;rep<4;++rep){
                fill(a,n,mode);fill(b,n,mode);memcpy(c,a,8*n);
                std::vector<uint64_t> first(n);
                for(unsigned kind=0;kind<4;++kind){
                    // 0 distinct inputs; 1 a == b (the family's square kernel); 2 a copy of
                    // a, distinct pointers (its product kernel); 3 sqrhigh (scalar to 20)
                    const uint64_t *y=kind==0?b:kind==2?c:a;
                    memset(memory.data,0xa5,scratch);
                    if(layout){r[-1]=canary;r[n]=canary;}
                    Frame frame(*f.arena,memory);                // peak() of this call alone
                    allocation_watch_start();
                    {ComputeLease computing(*f.arena);
                     if(kind==3)u52::sqrhigh(r,a,n,frame);else u52::mulhigh(r,a,y,n,frame);}
                    assert(!allocation_watch_stop());assert(!frame.used());
                    assert(frame.peak()<=(kind==3?u52::sqrhigh_scratch_bytes(n):need));
                    if(layout)assert(r[-1]==canary && r[n]==canary);
                    if(kind==1)memcpy(first.data(),r,8*n);
                    const bool same_terms=kind==2 || n<=14 || n>=21;
                    if(kind>=2 && same_terms && memcmp(first.data(),r,8*n)){
                        fprintf(stderr,"words differ n=%zu layout=%u mode=%u kind=%u\n",n,layout,mode,kind);assert(false);}
                    if(!contract(a,y,r,n)){fprintf(stderr,"contract n=%zu layout=%u mode=%u kind=%u\n",n,layout,mode,kind);assert(false);}
                    ++entry_checks;
                }
            }
        }
    }
    // A worked cross-family case (n = 18, a = b): A = 2^1152 - 2^1092 - 1, A^2 mod 2^1152 =
    // 2^1093 + 1. The scalar polynomial omits 16*2^1088 + 1 (not above that remainder), the
    // u52 one (K = 22) omits 21*2^1144 + 2^1093 + 1 (above it): sqrhigh, scalar at 18, is the
    // exact high half, mulhigh with a copy, u52 at 18, is one less. Each kernel computes
    // exactly its own truncation polynomial; both meet the contract.
    {
        constexpr size_t n=18;
        uint64_t *a=pool+1,*c=pool+(L+16)+2,*r=pool+2*(L+16)+3,*s=pool+3*(L+16)+3;
        for(size_t i=0;i<n;++i)a[i]=UINT64_MAX;
        a[n-1]=UINT64_MAX-16;memcpy(c,a,8*n);
        {Frame frame(*f.arena,memory);ComputeLease computing(*f.arena);
         u52::sqrhigh(s,a,n,frame);u52::mulhigh(r,a,c,n,frame);}
        auto h=exact_high(a,a,n);
        assert(!memcmp(s,h.data(),8*n));
        for(size_t i=0;i<n && !h[i]--;++i){}                  // h - 1
        assert(!memcmp(r,h.data(),8*n));
        entry_checks+=2;
    }
    printf("{\"status\":\"passed\",\"l0_checks\":%zu,\"entry_checks\":%zu,\"n_max\":%zu,\"column_t\":%zu,\"column_t_minus_1\":%zu}\n",
           l0_checks,entry_checks,L,classes[0],classes[1]);
}
