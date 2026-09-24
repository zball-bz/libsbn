#include "product_support.hpp"
#include "product/local_program.hpp"
#include "algorithms/local_inverse.hpp"
#include "algorithms/reciprocal.hpp"
#include "algorithms/newton_planner.hpp"
#include <tuple>
using namespace sbn::v3;
using namespace sbn::v3::product;
static bool same(pq16::Shape a,pq16::Shape b){
    return std::tie(a.nfull,a.branch,a.radix,a.centered,a.recipe,a.bits,a.balanced)==
           std::tie(b.nfull,b.branch,b.radix,b.centered,b.recipe,b.bits,b.balanced);
}
static void equivalence(){
    unsigned cases=0;
    for(size_t n=508;n<=16384;n+=1+n/31)for(bool division:{false,true}){
        LocalWindowProgram program{};allocation_watch_start();
        const auto bytes=local_refinement_compile(n,division,program);assert(!allocation_watch_stop());
        const auto old=division?quotient_bytes(n,LocalWindowFactory{}):reciprocal_bytes(n,LocalWindowFactory{});
        assert(bytes==old);
        for(unsigned j=0;j<program.count;++j){const auto &step=program.steps[j];
            const size_t length=step.words[0],m=newton_contract::next_precision(length);
            const auto shape=step.words[1]>>8==3?refinement_products<RefinementKind::Quotient>(m,length):
                                                refinement_products<RefinementKind::Inverse>(m,length);
            const auto base=local_windows_query(shape);
            const auto peel=peel_window_possible(shape)?peeled_windows_query(shape,base):PeeledWindowPlan{};
            assert(step.matches(shape)&&bool(step.words[1]&1)==bool(peel.ring));
            if(peel.ring){const auto q=step.peeled_plan();
                assert(same(q.fft,peel.fft)&&std::tie(q.ring,q.capacity,q.bytes,q.residual_words,q.shared_mask,q.count)==
                    std::tie(peel.ring,peel.capacity,peel.bytes,peel.residual_words,peel.shared_mask,peel.count));
            }else{const auto q=step.local_plan();
                assert(same(q.shared,base.shared)&&same(q.cancel_shape,base.cancel_shape)&&
                    std::tie(q.value_words,q.correction_words,q.work_bytes,q.storage_bytes,q.count,q.cancellation,q.shared_mask,q.middle)==
                    std::tie(base.value_words,base.correction_words,base.work_bytes,base.storage_bytes,base.count,base.cancellation,base.shared_mask,base.middle));
            }
            ++cases;
        }
    }
    printf("local program G0: %u descriptor/size equivalence cases PASS\n",cases);
}
static void seal(){
    Fixture f(1);sbn3_newton_plan plan{};sbn3_newton_info info{};const sbn3_newton_options options{1,0,0,0};
    assert(sbn3_newton_query(SBN3_NEWTON_INVERSE,1024,&options,&plan,&info)==SBN3_SUPPORTED);
    newton_detail::Plan detail{};memcpy(&detail,plan.opaque,sizeof detail);assert(detail.local_steps);
    const size_t at=up(f.cursor,info.storage_alignment);sbn3_error error{};
    assert(sbn3_arena_prepare(f.arena,at,info.storage_bytes,&error)==SBN3_OK);
    sbn3_newton_binding *binding=nullptr;sbn3_newton_bind(&plan,f.arena,at,f.team,&binding);sbn3_newton_unbind(binding);
    const pid_t child=fork();assert(child>=0);
    if(!child){
        rlimit no_core{0,0};setrlimit(RLIMIT_CORE,&no_core);
        auto bad=plan;bad.opaque[offsetof(newton_detail::Plan,choices)/8+2]^=1;
        sbn3_newton_bind(&bad,f.arena,at,f.team,&binding);_exit(0);
    }
    int status=0;assert(waitpid(child,&status,0)==child&&WIFSIGNALED(status)&&WTERMSIG(status)==SIGABRT);
    puts("local program descriptor corruption rejected before bind PASS");
}
int main(){equivalence();seal();}
