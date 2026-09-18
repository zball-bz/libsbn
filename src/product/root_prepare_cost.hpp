#pragma once
#include "sbn3/product.h"
namespace sbn::v3 {
// Generated from config/tuning/native-root-setup.json.
// Data SHA-256: 5ec333eded1765c5936ae2668b04807fb55fa7eb32a888322108e301ff10d0ba
// Cold constructor calibration, Zen5/Clang 21.1.8. Counts
// actual root/factor entries; it does not change transform admissibility.
inline double root_prepare_cost(const sbn3_mul_info &i) {
    if(!i.np)return 0;
    constexpr double c[]={0.10630585212477935,0.45053547850785952,472.67815040524312,79.591243785789629};
    return c[0]*i.np*double(i.table_entries)+c[1]*i.np*double(i.factor_levels)*i.M2+
           c[2]*i.np+c[3]*i.np*(i.factor_levels?i.root_order_log2:0);
}
}
