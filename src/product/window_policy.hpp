#pragma once
#include "product/native_capabilities.hpp"
#include "tuning/native_policy.hpp"
#include <algorithm>
namespace sbn::v3::product {
// Maximum worker requests do not disable a useful serial product family.
// The crossover belongs to the backend/machine profile, not the division
// recurrence. Explicit prime selection remains an arithmetic constraint.
inline bool inline_window_preferred(size_t span,unsigned prime_count=0,unsigned maximum_workers=1) noexcept {
    const size_t limit=maximum_workers==1?2*native_policy::window_inline_words-2:
        std::min(native_policy::window_inline_words,native_policy::window_parallel_fft_below-1);
    return !prime_count&&native_available()&&span<=limit;
}
inline size_t window_seed_limit(unsigned maximum) noexcept {
    return maximum==1?native_policy::window_inline_words:
        std::min(native_policy::window_inline_words,native_policy::window_parallel_fft_below-1);
}
inline unsigned native_window_workers(size_t span,unsigned maximum) noexcept {
    const unsigned cap=span<native_policy::newton_single_worker_below?1:
                       span<=native_policy::newton_middle_band_max?native_policy::newton_middle_workers:maximum;
    return std::min(maximum,cap);
}
} // namespace sbn::v3::product
