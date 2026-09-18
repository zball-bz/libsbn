#pragma once
#include "product/backend.hpp"
#include "backend/pq16/kernels.hpp"
namespace sbn::v3 {
sbn3_query_result fft_spectrum_description(pq16::Shape,size_t,unsigned,uint64_t,sbn3_spectrum_desc &);
void fft_reserve_spectrum(pq16::Shape,size_t,unsigned,uint64_t,sbn3_arena &,const sbn3_lease &,sbn3_spectrum **);
void fft_compute_spectrum(pq16::Shape,sbn3_spectrum *,sbn3_const_limbs,sbn3_team &,unsigned);
}
