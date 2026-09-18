#pragma once
#include "product/backend.hpp"
#include "common/checked.hpp"
namespace sbn::v3 {
// Internal cached-MUL episode. No new plan, lease or buffer is needed.
// Native providers include cache preparation in the returned product metrics;
// older providers retain their separate prepare/apply accounting.
inline void spectrum_compute_multiply(sbn3_mul_binding *product,sbn3_spectrum *cache,
                                       sbn3_const_limbs a,sbn3_const_limbs b,sbn3_limbs out) {
    require(product && cache && product->backend,SBN3_FATAL_ARGUMENT,"build/apply handles");
    if(product->backend->spectrum_multiply) {
        require(product->backend==cache->backend,SBN3_FATAL_ARGUMENT,"build/apply backend");
        product->backend->spectrum_multiply(product,cache,a,b,out);
    } else {
        sbn3_spectrum_compute(product,cache,a);
        const sbn3_product_inputs in{{},b,{},{}};
        sbn3_product_execute(product,&in,out);
    }
}
} // namespace sbn::v3
