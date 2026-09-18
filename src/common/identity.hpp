#pragma once
#include <stdint.h>

namespace sbn::v3::identity {
// Identity serialization is explicitly little-endian, independent of struct
// padding. Each format still owns its field order and version.
inline constexpr uint64_t fnv_seed = 1469598103934665603ULL;
inline constexpr uint64_t fnv_prime = 1099511628211ULL;
// One multiply and one fold per 64-bit field. Every step is a bijection of the
// running state, so no field is lost. Identities guard against accidental
// corruption and name plans within one process; they are not persisted, not a
// content hash, and not comparable across library versions. (The earlier
// byte-wise FNV-1a cost eight dependent multiplies per field and dominated
// planning time: a product plan seal covers well over a hundred fields.)
inline uint64_t word(uint64_t hash, uint64_t value) noexcept {
    hash = (hash ^ value) * 0x9e3779b97f4a7c15ULL;
    return hash ^ (hash >> 32);
}
} // namespace sbn::v3::identity
