// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "product/program.hpp"

namespace sbn::v3::floating {
// Exact full-product recipe. Unlike FloatProductPlan, this never truncates
// either multiplicand: cancellation against the addend can expose any bit.
struct FloatFmaPlan {
    ProductProgramPlan product{};
    std::size_t precision_bits=0,a_limbs=0,b_limbs=0,product_limbs=0,product_capacity=0;
    std::size_t product_offset=0,pool_offset=0,tables_bytes=0,work_offset=0;
    std::size_t multiply_bytes=0,addition_bytes=0,pool_bytes=0,payload_bytes=0;
    std::size_t scratch_bytes=0,alignment=64;
    unsigned workers=1;
    bool small=false;
};
// Nonzero finite multiplicands of these exact limb lengths. A plan with
// addend_nonzero=true also accepts zero addends. No inputs are retained.
sbn3_query_result float_fma_query(std::size_t precision_bits,std::size_t a_limbs,
    std::size_t b_limbs,bool addend_nonzero,unsigned workers,FloatFmaPlan&) noexcept;
// Compiled execution, including allocation-free table preparation. No query
// occurs after entry. The full product survives pool reuse by the final sum.
float_report float_fma_execute(const FloatFmaPlan&,float_ref,float_view a,float_view b,
    float_view c,execution={},rounding=rounding::nearest_even) noexcept;
}
