#include "algorithms/inverse_program.hpp"
#include "algorithms/inverse_seed.hpp"
#include "algorithms/local_inverse.hpp"
#include "runtime/scratch.hpp"
#include "common/checked.hpp"
namespace sbn::v3 {
void inverse_program(const InverseProgram &p, const uint64_t *D, uint64_t *U) noexcept {
    require(p.target && p.seed_limbs <= p.target && p.seed_limbs >= 1 &&
                (p.local_seed_bytes?(p.seed_limbs<=local_inverse_max_limbs||local_refinement_supported(p.seed_limbs,false)):
                                    p.seed_limbs<=15) &&
                p.rung_count <= 32,
            SBN3_FATAL_ARGUMENT, "inverse program shape");
    auto seed=[&](uint64_t *out,const uint64_t *d){
        if(p.local_seed_bytes){
            require(p.local_seed_workspace,SBN3_FATAL_WORKSPACE,"local inverse prefix workspace");
            auto f=Frame::external(p.local_seed_workspace,p.local_seed_bytes);
            local_inverse_approximate(out,d,p.seed_limbs,f);
        }else inverse_seed(out,d,p.seed_limbs);
    };
    if (!p.rung_count) {
        require(p.target == p.seed_limbs, SBN3_FATAL_ARGUMENT, "inverse seed target");
        seed(U, D);
        return;
    }
    seed(U, D + p.target - p.seed_limbs);
    size_t m = p.seed_limbs;
    for (size_t j = 0; j < p.rung_count; ++j) {
        const auto &r = p.rungs[j];
        require(r.m == m && r.n <= p.target, SBN3_FATAL_ARGUMENT, "inverse precision ladder");
        inverse_rung(r, D + p.target - r.n, U, U);
        m = r.n;
    }
    require(m == p.target, SBN3_FATAL_ARGUMENT, "inverse incomplete ladder");
}
} // namespace sbn::v3
