#pragma once
#include <stdint.h>
#include <stddef.h>
namespace sbn::v3 {
void mul_basecase_assumed(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t) noexcept;
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
}
