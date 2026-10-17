// L0 inline full products and squares (core/x86_64/gen/mulfull.hpp) against the oracle: r[0..na+nb) =
// A*B exactly for every 1 <= na, nb <= 8 (both orders), r[0..2n) = A^2 for n = 1..8. Every variant
// (default, asm where generated, u128), distinct inputs and a == b, structured inputs (all ones, carry
// chains, leading zero limbs, sparse); canary words on both sides of r catch stray stores, and the
// inputs must be left as the caller wrote them.
#include "../integration/product_support.hpp"
#include "core/x86_64/gen/mulfull.hpp"
using namespace sbn::v3;

static std::vector<uint64_t> exact(const uint64_t *a,size_t na,const uint64_t *b,size_t nb){
    ref_int A,B,P;ref_inits(A,B,P,nullptr);
    ref_import(A,na,-1,8,0,0,a);ref_import(B,nb,-1,8,0,0,b);ref_mul(P,A,B);
    std::vector<uint64_t> p(na+nb,0);size_t count=0;ref_export(p.data(),&count,-1,8,0,0,P);
    ref_clears(A,B,P,nullptr);assert(count<=na+nb);
    return p;
}
// 0 random, 1 all ones, 2 low half ones, 3 near ones, 4 leading zero limbs, 5 sparse
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
constexpr uint64_t canary=0x5a5a5a5a5a5a5a5aull;
// variant 0 default, 1 asm (memory rows, one row), 2 u128, 3 reg (register window, nb >= 2)
template<int na,int nb> static bool mul_run(unsigned v,uint64_t *r,const uint64_t *a,const uint64_t *b){
    if(v==0){l0::mul<na,nb>(r,a,b);return true;}
    if constexpr(na>=nb && na>1 && nb==1)if(v==1){l0::mul_asm<na,nb>(r,a,b);return true;}
    if(v==2){l0::mul_u128<na,nb>(r,a,b);return true;}
    if constexpr(na>=nb && nb>=2)if(v==3){l0::mul_reg<na,nb>(r,a,b);return true;}
    return false;
}
template<int n> static bool sqr_run(unsigned v,uint64_t *r,const uint64_t *a){
    if(v==0){l0::sqr<n>(r,a);return true;}
    if constexpr(n>=2)if(v==1){l0::sqr_asm<n>(r,a);return true;}
    if(v==2){l0::sqr_u128<n>(r,a);return true;}
    return false;
}
template<int na,int nb> static void mul_case(size_t &checks){
    for(unsigned mode=0;mode<6;++mode)for(int rep=0;rep<200;++rep){
        uint64_t a[na],b[nb];fill(a,na,mode);fill(b,nb,mode);
        uint64_t a0[na],b0[nb];memcpy(a0,a,sizeof a);memcpy(b0,b,sizeof b);
        for(unsigned same=0;same<(na==nb?2u:1u);++same){
            const uint64_t *y=same?a:b;
            const auto want=exact(a,na,y,nb);
            for(unsigned v=0;v<4;++v){
                uint64_t buf[na+nb+2];for(auto &w:buf)w=canary;
                if(!mul_run<na,nb>(v,buf+1,a,y))continue;
                assert(!memcmp(buf+1,want.data(),8*(na+nb)) && buf[0]==canary && buf[na+nb+1]==canary);
                assert(!memcmp(a,a0,sizeof a) && !memcmp(b,b0,sizeof b));
                ++checks;
            }
        }
    }
}
template<int n> static void sqr_case(size_t &checks){
    for(unsigned mode=0;mode<6;++mode)for(int rep=0;rep<200;++rep){
        uint64_t a[n];fill(a,n,mode);
        uint64_t a0[n];memcpy(a0,a,sizeof a);
        const auto want=exact(a,n,a,n);
        for(unsigned v=0;v<3;++v){
            uint64_t buf[2*n+2];for(auto &w:buf)w=canary;
            if(!sqr_run<n>(v,buf+1,a))continue;
            assert(!memcmp(buf+1,want.data(),16*n) && buf[0]==canary && buf[2*n+1]==canary && !memcmp(a,a0,sizeof a));
            ++checks;
        }
    }
}
template<int na,int nb> static void row(size_t &checks){
    mul_case<na,nb>(checks);
    if constexpr(nb<8)row<na,nb+1>(checks);
}
template<int na> static void all(size_t &mul_checks,size_t &sqr_checks){
    row<na,1>(mul_checks);sqr_case<na>(sqr_checks);
    if constexpr(na<8)all<na+1>(mul_checks,sqr_checks);
}
int main(){
    size_t mul_checks=0,sqr_checks=0;
    all<1>(mul_checks,sqr_checks);
    printf("{\"status\":\"passed\",\"mul_checks\":%zu,\"sqr_checks\":%zu,\"n_max\":8}\n",mul_checks,sqr_checks);
}
