// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
namespace sbn::v3::external {
struct RegionId {
    uint64_t uuid_low = 0, uuid_high = 0, incarnation = 0;
    bool operator==(const RegionId &) const = default;
};
uint64_t chunk_checksum(RegionId, uint64_t chunk, std::span<const uint8_t>) noexcept;
uint64_t metadata_checksum(std::span<const uint8_t>) noexcept;
inline void put64(uint8_t *p, uint64_t value) noexcept {
    for (unsigned i = 0; i < 8; ++i) p[i] = uint8_t(value >> (8*i));
}
inline uint64_t get64(const uint8_t *p) noexcept {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= uint64_t(p[i]) << (8*i);
    return value;
}
} // namespace sbn::v3::external
