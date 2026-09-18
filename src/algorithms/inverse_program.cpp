#include "algorithms/inverse_program.hpp"
#include "algorithms/inverse_seed.hpp"
#include "common/checked.hpp"
namespace sbn::v3 {
void inverse_program(const InverseProgram &p, const uint64_t *D, uint64_t *U) noexcept {
    require(p.target && p.seed_limbs <= p.target && p.seed_limbs >= 1 && p.seed_limbs <= 15 &&
                p.rung_count <= 32,
            SBN3_FATAL_ARGUMENT, "inverse program shape");
    if (!p.rung_count) {
        require(p.target == p.seed_limbs, SBN3_FATAL_ARGUMENT, "inverse seed target");
        inverse_seed(U, D, p.target);
        return;
    }
    inverse_seed(p.values[0], D + p.target - p.seed_limbs, p.seed_limbs);
    size_t m = p.seed_limbs;
    unsigned live = 0;
    for (size_t j = 0; j < p.rung_count; ++j) {
        const auto &r = p.rungs[j];
        require(r.m == m && r.n <= p.target, SBN3_FATAL_ARGUMENT, "inverse precision ladder");
        uint64_t *out = j + 1 == p.rung_count ? U : p.values[live ^ 1];
        inverse_rung(r, D + p.target - r.n, p.values[live], out);
        m = r.n;
        live ^= 1;
    }
    require(m == p.target, SBN3_FATAL_ARGUMENT, "inverse incomplete ladder");
}
} // namespace sbn::v3
