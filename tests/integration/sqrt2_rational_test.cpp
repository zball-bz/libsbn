#include "newton_setup.hpp"
#include "algorithms/sqrt2_rational.hpp"
#include "algorithms/sqrt2_finish.hpp"
using namespace sbn::v3;
struct PreparedRational {
    std::vector<RationalSqrt2Step> steps;std::vector<sbn3_spectrum *> handles;RationalSqrt2Program program{};
    PreparedRational(Fixture &f,size_t digits){
        program.last_iteration=rational_sqrt2_iterations(digits);steps.reserve(program.last_iteration-1);
        for(unsigned k=1;k<program.last_iteration;++k){const size_t n=rational_sqrt2_capacity(k);
            sbn3_product_request req{};req.a_limbs=req.b_limbs=n;sbn3_mul_options o{};o.workers=sbn3_team_workers(f.team);
            sbn3_product_info info{};sbn3_mul_plan plan{};const auto first=f.leases.size();auto *multiply=f.product(req,o,info,plan);
            sbn3_spectrum_desc future{};sbn3_spectrum *cache=nullptr;
            if(sbn3_spectrum_query(&plan,SBN3_SPECTRUM_COLUMNS,k,&future)==SBN3_SUPPORTED){
                auto storage=f.allocate(future.storage_bytes,64);sbn3_spectrum_reserve(multiply,SBN3_SPECTRUM_COLUMNS,k,f.arena,&storage,&cache);
                req.cached_a[0]=&future;auto *cached=f.product(req,o,info,plan,cache);f.unbind(multiply);f.retire(first+1);f.retire(first);multiply=cached;handles.push_back(cache);
            }
            req.kind=SBN3_PRODUCT_SQR;req.b_limbs=0;auto *square=f.product(req,o,info,plan,cache);
            steps.push_back({n,rational_sqrt2_capacity(k+1),multiply,square,cache});
        }
        program.steps=steps.data();for(auto &p:program.values)p=f.guarded(up(rational_sqrt2_capacity(program.last_iteration)+2,8));
    }
    ~PreparedRational(){for(auto *h:handles)sbn3_spectrum_release(h);}
};
static void test(size_t digits,unsigned workers){
    // The primitive suites use per-buffer guard pages. This whole-program
    // fixture packs adjacent leases, as a preallocated client would, avoiding
    // artificial fragmentation of the arena's resident-region descriptors.
    Fixture f(workers,false);PreparedRational rational(f,digits);
    const size_t n=digits+2<4?4:digits+2,m=n/2+1;PreparedInverse inverse(f,m);PreparedDivision division(f,m,n);
    sbn3_product_request req{};req.kind=SBN3_PRODUCT_SQR;req.a_limbs=digits+1;sbn3_mul_options options{};options.workers=workers;
    sbn3_mul_plan plan{};sbn3_product_info info{};auto *checker=f.product(req,options,info,plan);
    auto *A=f.guarded(n+1),*D=f.guarded(n+1),*U=f.guarded(m+1),*Q=f.guarded(n+1),*z=f.guarded(digits+1),*square=f.guarded(up(2*digits+2,8));
    allocation_watch_start();const auto value=rational_sqrt2(rational.program);normalize_rational_sqrt2(value,n,A,D);
    inverse.execute(D+n-m,U);division.execute(A,D,U,Q);memcpy(z,Q+n-digits,(digits+1)*8);const unsigned corrections=sqrt2_certify(checker,z,digits,square);
    assert(!allocation_watch_stop());
    ref_int p,q,delta,t,want,got;ref_inits(p,q,delta,t,want,got,nullptr);
    ref_import(p,value.limbs,-1,8,0,0,value.p);ref_import(q,value.limbs,-1,8,0,0,value.q);ref_mul(delta,p,p);ref_mul(t,q,q);ref_mul_2exp(t,t,1);ref_sub(delta,delta,t);assert(ref_cmp_ui(delta,1)==0);
    ref_set_ui(want,1);ref_mul_2exp(want,want,64*digits);assert(ref_cmp(t,want)>=0);
    ref_set_ui(want,1);ref_mul_2exp(want,want,128*digits+1);ref_sqrt(want,want);ref_import(got,digits+1,-1,8,0,0,z);assert(ref_cmp(got,want)==0);
    ref_mul(t,got,got);ref_import(want,2*digits+2,-1,8,0,0,square);assert(ref_cmp(t,want)==0);ref_clears(p,q,delta,t,want,got,nullptr);
    printf("sqrt2 rational n=%zu W%u iteration=%u P/Q capacity=%zu corrections=%u prefix=%llx.%016llx: Pell/exact floor/one terminal divide/no allocation PASS\n",digits,workers,rational.program.last_iteration,value.limbs,corrections,(unsigned long long)z[digits],(unsigned long long)z[digits-1]);fflush(stdout);
}
int main(){
    for(size_t n:{1u,2u,3u,7u,16u,64u,129u,1024u,8192u,65536u})test(n,1);
    test(1024,16);test(65536,16);puts("rational Newton sqrt2 route gates PASS");
}
