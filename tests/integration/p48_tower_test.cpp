// Compressed p48 root tower: every accessor against an independent scalar
// tower built with the historical recurrence (128-bit mulmod and division).
#define CR_NP 10
#define SBN3_P48_NS tower_gate
#include "product_support.hpp"
#include "backend/ntt_p48/engine.hpp"
#include <vector>
using namespace sbn::v3;
using namespace sbn::v3::tower_gate;
namespace {
uint64_t rec_of(uint64_t c,uint64_t p){return uint64_t((u128(c)<<52)/p);}
// Independent generator search and tower (no constexpr table, no vector code).
uint64_t generator(uint64_t p){
    std::vector<uint64_t> f;uint64_t n=p-1;
    for(uint64_t d=2;u128(d)*d<=n;++d)if(n%d==0){f.push_back(d);while(n%d==0)n/=d;}
    if(n>1)f.push_back(n);
    for(uint64_t g=2;;++g){bool ok=true;for(auto q:f)if(powm(g,(p-1)/q,p)==1)ok=false;if(ok)return g;}
}
struct Reference {std::vector<uint64_t> w,wi;};
Reference reference(unsigned lg,uint64_t p){
    Reference r;r.w.assign(size_t(1)<<lg,0);r.wi.assign(size_t(1)<<lg,0);r.w[0]=r.wi[0]=1;const uint64_t g=generator(p);
    for(unsigned t=1;t<=lg;++t){const uint64_t z=powm(g,(p-1)>>t,p),zi=invm(z,p),blk=uint64_t(1)<<(t-1);
        for(uint64_t s=0;s<blk;++s){r.w[blk+s]=mulm(z,r.w[s],p);r.wi[blk+s]=mulm(zi,r.wi[s],p);}}
    return r;
}
uint64_t lane(V v,unsigned k){alignas(64) uint64_t x[8];_mm512_store_si512(x,v);return x[k];}
}
int main(){
    Fixture fixture(1,false);size_t checked=0;const auto tables=fixture.allocate(size_t(64)<<20,128);
    for(unsigned lg:{1u,2u,3u,4u,5u,6u,9u,13u,17u}){
        Frame frame(*fixture.arena,tables);Primes primes{};
        const size_t before=frame.used();primes.init(frame,lg);
        size_t expected=0;
        for(unsigned q=0;q<NP;++q){expected=(expected+63)&~size_t(63);expected+=tower_bytes(lg);expected=(expected+63)&~size_t(63);expected+=tower_zero_slots*8;}
        assert(frame.used()-before==expected);
        for(unsigned q=0;q<NP;++q){
            const auto &P=primes.P[q];const uint64_t p=P.p;const auto ref=reference(lg,p);const size_t n=size_t(1)<<lg;
            assert(P.lg==lg && P.e && P.ez);
            for(size_t k=0;k<n;++k){
                assert(mulm(ref.w[k],ref.wi[k],p)==1);
                assert(wval(P,k)==ref.w[k] && wival(P,k)==ref.wi[k]);
                if(!(k&1)){assert(twr<false>(P,k/2)==rec_of(ref.w[k],p) && twr<true>(P,k/2)==rec_of(ref.wi[k],p));checked+=2;}
            }
            // Block cursors: every level-k child of every node, both directions.
            for(size_t j=0;j<n/2;++j)for(size_t k=1;k<=8;k<<=1)for(size_t c=0;c<k;++c){
                const size_t node=k*j+c;if(node>=n/2)continue;
                const uint64_t forward=twk<false,false>(twcur<false,false>(P,j),k,c),inverse=twk<true,false>(twcur<true,false>(P,j),k,c);
                assert(forward==rec_of(ref.w[2*node],p) && inverse==rec_of(ref.wi[2*node],p));checked+=2;
            }
            // Eight consecutive leaf roots and eight even nodes.
            for(size_t k0=0;k0+8<=n;k0+=8)for(unsigned i=0;i<8;++i){
                assert(lane(tower8<false>(P,k0),i)==rec_of(ref.w[k0+i],p));
                assert(lane(tower8<true>(P,k0),i)==rec_of(ref.wi[k0+i],p));checked+=2;
            }
            for(size_t j8=0;8*j8+8<=n/2;++j8)for(unsigned i=0;i<8;++i){
                assert(lane(tower_even8<false>(P,j8),i)==rec_of(ref.w[2*(8*j8+i)],p));
                assert(lane(tower_even8<true>(P,j8),i)==rec_of(ref.wi[2*(8*j8+i)],p));checked+=2;
            }
            // Halved inverse twiddle pair, exact in both members.
            const uint64_t inv2=(p+1)/2;
            for(size_t j=0;j<n/2;++j){
                const vcc got=wi2_of(P,j,primes.V[q].p);const uint64_t want=mulm(ref.wi[2*j],inv2,p);
                for(unsigned i=0;i<8;i+=7)assert(lane(got.c,i)==want && lane(got.rec,i)==rec_of(want,p));
                ++checked;
            }
        }
    }
    // Bailey factor tables: F[d][k*M2+b] = rec(w_d[2^k b]), including a row
    // length that is not a multiple of the vector width.
    for(size_t M2:{1u,2u,4u,8u,16u,32u,512u})for(unsigned lgC:{0u,3u,7u}){
        Frame frame(*fixture.arena,tables);Primes primes{};
        unsigned lgM=0;while((size_t(1)<<lgM)<M2)++lgM;
        const unsigned lg=lgC+lgM+1;primes.init(frame,lg);primes.factors_for(frame,M2,lgC);
        for(unsigned q=0;q<NP;++q){const auto &P=primes.P[q];const auto ref=reference(lg,P.p);
            assert(P.M2f==M2 && P.lgF==lgC);
            for(unsigned k=0;k<=lgC;++k)for(size_t b=0;b<M2;++b){
                assert(P.F[0][k*M2+b]==rec_of(ref.w[(size_t(1)<<k)*b],P.p));
                assert(P.F[1][k*M2+b]==rec_of(ref.wi[(size_t(1)<<k)*b],P.p));checked+=2;
            }
        }
    }
    printf("{\"status\":\"passed\",\"checked\":%zu,\"primes\":%u}\n",checked,unsigned(NP));
}
