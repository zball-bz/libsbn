#include "product_support.hpp"
#include "runtime/scratch.hpp"
#include "backend/u52/kernels.hpp"
using namespace sbn::v3;
static void one(size_t an,size_t bn,u52::Algorithm root){
    Fixture f(1);auto memory=f.allocate(u52::scratch_bytes(an,bn,root),64);Frame frame(*f.arena,memory);
    auto *a=f.guarded(an),*b=f.guarded(bn),*p=f.guarded(an+bn);
    for(unsigned kind=0;kind<4;++kind){
        for(size_t j=0;j<an;++j)a[j]=kind==0?UINT64_MAX:random_word();
        for(size_t j=0;j<bn;++j)b[j]=kind==0?UINT64_MAX:random_word();
        if(kind==2 && root==u52::Algorithm::automatic){for(size_t j=an/2;j<an;++j)a[j]=0;}
        else{a[an-1]|=1ull<<63;b[bn-1]|=1ull<<63;}
        if(kind==3 && root==u52::Algorithm::automatic)memset(b,0,bn*8);
        memset(p,0xaa,(an+bn)*8);allocation_watch_start();
        {ComputeLease computing(*f.arena);auto used=u52::multiply(p,a,an,b,bn,frame,root);if(root!=u52::Algorithm::automatic)assert(used==root);}
        assert(!allocation_watch_stop());assert(!frame.used());verify_product(a,an,b,bn,p);
        assert(frame.peak()<=u52::scratch_bytes(an,bn,root));frame.reset();
    }
}
int main(){
    for(size_t n=1;n<=100;++n){one(n,n,u52::Algorithm::automatic);one(n,1,u52::Algorithm::automatic);}
    for(size_t n:{127u,128u,129u,255u,256u,257u,426u,532u,619u,1024u,4096u,8192u})
        for(size_t m:{n,n/2,n/3,1ul})one(n,m,u52::Algorithm::automatic);
    for(size_t n:{32u,64u})one(n,n,u52::Algorithm::basecase);
    for(size_t n:{64u,128u,257u,512u})one(n,n,u52::Algorithm::karatsuba);
    for(size_t n:{128u,256u,512u}){
        one(n,n,u52::Algorithm::toom33);
        one(3*n,2*n,u52::Algorithm::toom32);
        one(2*n,n,u52::Algorithm::toom42);
        one(8*n,n,u52::Algorithm::stripmine);
    }
    puts("u52 port: GMP / algorithms / ratios / dirty outputs / leading zeros / guards / allocation / Frame bound PASS");
}
