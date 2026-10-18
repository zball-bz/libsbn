// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "backend/ntt_p48/root_bank.hpp"
#include "backend/pq16/root_bank.hpp"
namespace sbn::v3::external {
inline size_t engine_shared_bytes() noexcept {
    const size_t bytes=p48_root_bank::bytes()+pq16::root_bank::bytes+sizeof(pq16::PQ16_ROOT_COARSE)+sizeof(pq16::PQ16_ROOT_FINE)+256;
    return (bytes+4095)&~size_t(4095);
}
// Sequential phases share one aligned arena interval. Round its available
// size down BEFORE filling it with retained planes, so final alignment cannot
// invalidate a phase that consumed the last integrity chunk of the RAM grant.
inline size_t aligned_phase_budget(size_t total,size_t fixed,size_t alignment) noexcept {
    return fixed<=total?fixed+((total-fixed)&~(alignment-1)):total;
}
}
