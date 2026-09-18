#include "product_support.hpp"
#include "algorithms/sqrt2_finish.hpp"
#include "value/limbs.hpp"
using namespace sbn::v3;
static void scenario(size_t n,unsigned workers){
    Fixture f(workers);sbn3_product_request req{};req.kind=SBN3_PRODUCT_SQR;req.a_limbs=n+1;
    sbn3_mul_options options{};options.workers=workers;sbn3_mul_plan plan{};sbn3_product_info info{};auto *b=f.product(req,options,info,plan);
    auto *z=f.guarded(n+1),*square=f.guarded(up(2*n+2,8));
    ref_int exact,got,sq,expected;ref_inits(exact,got,sq,expected,nullptr);ref_set_ui(exact,1);ref_mul_2exp(exact,exact,128*n+1);ref_sqrt(exact,exact);
    for(int delta=-2;delta<=2;++delta){
        ref_set(got,exact);if(delta<0)ref_sub_ui(got,got,unsigned(-delta));else ref_add_ui(got,got,unsigned(delta));
        memset(z,0,(n+1)*8);size_t count;ref_export(z,&count,-1,8,0,0,got);
        allocation_watch_start();const unsigned corrections=sqrt2_certify(b,z,n,square);assert(!allocation_watch_stop());
        assert(corrections==unsigned(delta<0?-delta:delta));ref_import(got,n+1,-1,8,0,0,z);assert(ref_cmp(got,exact)==0);
        ref_mul(expected,got,got);ref_import(sq,2*n+2,-1,8,0,0,square);assert(ref_cmp(sq,expected)==0);
    }
    ref_clears(exact,got,sq,expected,nullptr);
    printf("sqrt2 integer-square certificate n=%zu W%u: offsets -2..+2, one square, exact floor PASS\n",n,workers);
}
int main(){for(size_t n:{1u,7u,64u,1024u,4096u})scenario(n,1);scenario(32768,16);puts("sqrt2 certificate gates PASS");}
