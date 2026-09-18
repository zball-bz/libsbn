#pragma once
#include <stddef.h>
namespace sbn::v3::cohort_policy {
// Generated from native-cohort.json. This is a locality preference, not a
// correctness/admission limit. Uncalibrated cohorts keep their prior policy.
inline constexpr size_t llc_bytes = 67108864;
inline constexpr unsigned measured_cohorts[] = {32};
inline size_t serial_cache_share(unsigned cohort_workers) noexcept {
    for (unsigned w : measured_cohorts)
        if (w == cohort_workers) return llc_bytes / w;
    return 0;
}
}
