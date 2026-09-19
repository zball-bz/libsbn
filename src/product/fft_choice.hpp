#pragma once
#include "product/backend.hpp"
#include "backend/pq16/kernels.hpp"
namespace sbn::v3 {
// An exact FFT winner, not an instruction to run its shape search again.
// Entry 127 is a private transcript tag; the executable backend remains 100.
inline bool is_fft_shape_choice(ProductChoice c) noexcept {return ((c>>1)&127)==127;}
inline ProductChoice fft_shape_choice(pq16::Shape s) noexcept {
    if(!s.branch || (s.branch&(s.branch-1)))return 0;
    return 255 | uint64_t(__builtin_ctz(s.branch))<<8 | uint64_t(s.radix)<<14 |
           uint64_t(s.bits)<<17 | uint64_t(s.recipe)<<22 | uint64_t(s.balanced)<<24 | uint64_t(s.centered)<<25;
}
inline bool fft_shape_of_choice(ProductChoice c,pq16::Shape &s) noexcept {
    if(!is_fft_shape_choice(c) || !(c&1) || c>>26)return false;
    const unsigned lg=unsigned(c>>8&63),m=unsigned(c>>14&7),b=unsigned(c>>17&31),r=unsigned(c>>22&3);
    if(lg<6||lg>19||(m!=1&&m!=3&&m!=5&&m!=7)||b<16||b>20||r>2)return false;
    s={m*(1u<<lg),1u<<lg,m,bool(c>>25&1),pq16::Recipe(r),b,bool(c>>24&1)};return true;
}
sbn3_query_result short_fft_query(const sbn3_product_request &,const sbn3_mul_options &,pq16::Shape,
                                   sbn3_mul_plan &,sbn3_product_info &) noexcept;
ProductChoice short_fft_choice(const sbn3_mul_plan &) noexcept;
}
