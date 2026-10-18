#pragma once
#include <stddef.h>
#include <stdint.h>
// Published, prefix-stable integer roots. Independent of NP, codec, source
// length and scale; no first-use initialization or operation-owned cache.
namespace sbn::v3::p48_root_bank {
inline constexpr unsigned prime_count=10,tower_log=14,factor_log=13;
inline constexpr size_t even_count=size_t(1)<<(tower_log-1),factor_stride=256;
inline constexpr uint64_t primes[prime_count]={281346127691777ULL,281333242789889ULL,281255933378561ULL,
    281212983705601ULL,281204393771009ULL,281161444098049ULL,281135674294273ULL,281088429654017ULL,
    281019710177281ULL,280710472531969ULL};
const uint64_t *tower(unsigned prime,unsigned log) noexcept;
const uint64_t *zero_image(unsigned prime) noexcept;
const uint64_t *factors(unsigned prime,unsigned direction,size_t columns,unsigned log_rows) noexcept;
size_t bytes() noexcept;
}
