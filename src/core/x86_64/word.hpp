#pragma once
#include <stdint.h>
#include <stddef.h>
namespace sbn::v3 {
void mul_basecase_assumed(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t) noexcept;
// The explicit square of n limbs into 2n (r disjoint from a): straight-line kernels for
// n = 2..14, the product kernels otherwise. Plain products test the pointer only for 7..14.
void sqr_basecase_assumed(uint64_t *,const uint64_t *,size_t) noexcept;
}
extern "C" {
uint64_t sbn3i_add_n(uint64_t *, const uint64_t *, const uint64_t *, long);
uint64_t sbn3i_sub_n(uint64_t *, const uint64_t *, const uint64_t *, long);
uint64_t sbn3i_mul_1(uint64_t *, const uint64_t *, long, uint64_t);
uint64_t sbn3i_addmul_1(uint64_t *, const uint64_t *, long, uint64_t);
uint64_t sbn3i_addmul_2(uint64_t *, const uint64_t *, long, const uint64_t *);
uint64_t sbn3i_addmul_1_adx(uint64_t *, const uint64_t *, long, uint64_t);
uint64_t sbn3i_addmul_2_adx(uint64_t *, const uint64_t *, long, const uint64_t *);
void sbn3i_mul_basecase_le6(uint64_t *, const uint64_t *, uint64_t, const uint64_t *, uint64_t);
void sbn3i_mul_basecase_le16(uint64_t *, const uint64_t *, uint64_t, const uint64_t *, uint64_t);
void sbn3i_sqr_basecase_2(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_3(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_4(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_5(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_6(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_7(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_8(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_9(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_10(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_11(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_12(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_13(uint64_t *, const uint64_t *);
void sbn3i_sqr_basecase_14(uint64_t *, const uint64_t *);
// high_fixed.S: short high halves, 0 <= floor(A*B/2^(64n)) - r <= 1 (register columns to 10)
void sbn3i_mulhigh_5(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_mulhigh_6(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_mulhigh_7(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_mulhigh_8(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_mulhigh_9(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_mulhigh_10(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_mulhigh_11(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_mulhigh_12(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_mulhigh_13(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_mulhigh_14(uint64_t *, const uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_5(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_6(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_7(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_8(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_9(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_10(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_11(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_12(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_13(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_14(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_15(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_16(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_17(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_18(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_19(uint64_t *, const uint64_t *);
void sbn3i_sqrhigh_20(uint64_t *, const uint64_t *);
}
