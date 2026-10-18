// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "product/backend.hpp"
#include "backend/pq16/kernels.hpp"
namespace sbn::v3 {
// Private canonical matrix episodes. No public MUL plan or operand cache is
// fabricated. A ring request fixes its exact period after shape selection.
pq16::Shape fft_matrix_shape(size_t minimum,size_t maximum) noexcept;
bool fft_matrix_query(size_t a,size_t b,size_t ring,unsigned workers,pq16::Shape,
                      sbn3_mul_plan &,sbn3_mul_info &) noexcept;
}
