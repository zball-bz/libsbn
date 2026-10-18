#pragma once
#include <stdint.h>
#include <stddef.h>

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
// Four independent chains hide multiply latency in long opaque-plan seals.
// Ordering and length remain significant. Each word has a bijective path to
// the result, so every single-word mutation is detected, like word() itself.
// This is still an in-process accidental-corruption identity, not a MAC.
inline uint64_t words(uint64_t hash,const uint64_t *data,size_t count) noexcept {
    if(count<16){for(size_t j=0;j<count;++j)hash=word(hash,data[j]);return hash;}
    uint64_t a=hash,b=hash^1,c=hash^2,d=hash^3;
    size_t j=0;
    for(;j+4<=count;j+=4){a=word(a,data[j]);b=word(b,data[j+1]);c=word(c,data[j+2]);d=word(d,data[j+3]);}
    hash=word(word(word(word(word(hash,count),a),b),c),d);
    for(;j<count;++j)hash=word(hash,data[j]);
    return hash;
}
} // namespace sbn::v3::identity
