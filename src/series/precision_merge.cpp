// SPDX-License-Identifier: LGPL-3.0-or-later
#include "series/precision_merge.hpp"
#include "series/limited.hpp"
#include <cstring>
namespace sbn::v3::series {
void precision_merge_action(void *p, sbn3_team_scope *scope) {
    auto &c = *static_cast<PrecisionMergeCall *>(p);
    auto multiply = [&](sbn3_series_value &out, const sbn3_series_value &x, const sbn3_series_value &y, const ProductProgram *selected=nullptr) {
        const auto *program=selected?selected:c.program;
        const auto &xm = x.mantissa, &ym = y.mantissa;
        const size_t used = xm.size + ym.size;
        const unsigned negative = xm.negative ^ ym.negative;
        require(xm.size <= program->an && ym.size <= program->bn, SBN3_FATAL_WORKSPACE,
                "PSR product bounds");
        require(!__builtin_add_overflow(x.exponent2, y.exponent2, &out.exponent2), SBN3_FATAL_SIZE,
                "PSR product exponent");
        if (!xm.size || !ym.size) {
            out.mantissa.size = out.mantissa.negative = 0;
            out.exponent2 = 0;
            return;
        }
        const bool direct = !c.fallback;
        auto *destination = direct ? out.mantissa.data : c.fallback;
        if (program->contract & program_bounded_inputs) {
            product_program_bounded(
                *program, *c.work, scope, {xm.data, xm.size}, {ym.data, ym.size},
                {destination, direct ? out.mantissa.capacity : program->an + program->bn}, direct);
        } else {
            std::memcpy(c.a, xm.data, xm.size * 8);
            std::memset(c.a + xm.size, 0, (program->an - xm.size) * 8);
            std::memcpy(c.b, ym.data, ym.size * 8);
            std::memset(c.b + ym.size, 0, (program->bn - ym.size) * 8);
            product_program_execute(*program, *c.work, scope, {c.a, program->an}, {c.b, program->bn},
                                    {destination, program->an + program->bn});
        }
        if (!direct)
            std::memmove(out.mantissa.data, destination, used * 8);
        out.mantissa.size = used;
        out.mantissa.negative = negative;
        // PrecisionMergeCall temporaries retain the full backing slot until the merge
        // finishes. Dropped low words need no physical compaction here.
        limit_window(out, c.precision + 1);
    };
    const auto &v = *c.block;
    auto &r = *c.result;
    // V1: consume block U to form U*N_tail. N_tail then dies, so its slot
    // saves T before V0 consumes Q and overwrites the old T/Q pair.
    // Hyperdescent has U == 1; BinaryBBP is additive (T = T_L D_R + D_L T_R),
    // so the block's own denominator plays the U role and attenuation is
    // certified by the terms' absolute binary exponents, not by |U/D|.
    if (c.recipe == SBN3_SERIES_HYPERDESCENT)
        copy_limited(c.y, r.value[0], c.precision + 1);
    else if (c.recipe == SBN3_SERIES_BINARY_BBP)
        multiply(c.y, v.value[1], r.value[0]);
    else
        multiply(c.y, v.value[2], r.value[0], c.contraction);
    copy_limited(r.value[0], v.value[0], c.precision);
    // Both saved T and old denominator are now consumed into their combined
    // contiguous output region. Restore the new denominator only after N.
    sbn3_series_value sum{{r.value[0].mantissa.data, 2 * c.pair_words, 0, 0}, 0};
    if (c.program->pair_workspace_bytes && v.value[1].mantissa.size &&
        r.value[0].mantissa.size && r.value[1].mantissa.size) {
        const auto &common = r.value[1];
        const auto &x = v.value[1];
        const auto &y = r.value[0];
        const unsigned sign0 = common.mantissa.negative ^ x.mantissa.negative;
        const unsigned sign1 = common.mantissa.negative ^ y.mantissa.negative;
        int64_t e0 = 0, e1 = 0;
        require(!__builtin_add_overflow(common.exponent2, x.exponent2, &e0) &&
                    !__builtin_add_overflow(common.exponent2, y.exponent2, &e1),
                SBN3_FATAL_SIZE, "PSR paired product exponent");
        const size_t n0 = common.mantissa.size + x.mantissa.size;
        const size_t n1 = common.mantissa.size + y.mantissa.size;
        product_program_pair(*c.program, *c.work, scope, {common.mantissa.data, common.mantissa.size},
                             {x.mantissa.data, x.mantissa.size}, {y.mantissa.data, y.mantissa.size},
                             {c.x.mantissa.data, c.x.mantissa.capacity},
                             {sum.mantissa.data, sum.mantissa.capacity}, true);
        c.x.mantissa.size = n0; c.x.mantissa.negative = sign0; c.x.exponent2 = e0;
        sum.mantissa.size = n1; sum.mantissa.negative = sign1; sum.exponent2 = e1;
        limit_window(c.x, c.precision + 1);
        limit_window(sum, c.precision + 1);
    } else {
        multiply(c.x, v.value[1], r.value[1]);
        multiply(sum, r.value[0], r.value[1]);
    }
    add_limited<true>(r.value[0], sum, c.y, c.precision);
    copy_limited(r.value[1], c.x, c.precision);
}
} // namespace sbn::v3::series
