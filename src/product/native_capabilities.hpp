#pragma once
namespace sbn::v3 {
// One gate for the current native-only bundle, including private metadata
// queries compiled in its ISA islands. Keep baseline callers out of those
// TUs when the required feature closure is absent.
inline bool native_available() noexcept {
    return __builtin_cpu_supports("avx512ifma") && __builtin_cpu_supports("avx512vbmi") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512cd") &&
           __builtin_cpu_supports("avx512vbmi2") && __builtin_cpu_supports("avx512vpopcntdq") &&
           __builtin_cpu_supports("pclmul") && __builtin_cpu_supports("vpclmulqdq");
}
}
