// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
namespace sbn::v3::product {
struct CrtCarry {uint64_t limb[8]{};};
struct CrtBackend {
    unsigned primes;
    size_t (*prepared_bytes)(unsigned trunk_bits) noexcept;
    bool (*capacity)(unsigned trunk_bits,uint64_t terms) noexcept;
    void (*prepare)(unsigned trunk_bits,std::span<uint8_t>) noexcept;
    // Normalized coefficient residues, lazy [0,4p), with the scale selected
    // by preparation (ordinary prepare selects unscaled). Eight adjacent
    // coefficients per cell; prime_stride is in uint64 words. Emits cells*T
    // bytes with zero incoming carry and returns its exact carry-out. Output
    // has 256 extra writable bytes for the existing packer's footprint.
    void (*emit)(std::span<const uint8_t>,const uint64_t *,size_t prime_stride,size_t cells,
                 std::span<uint8_t>,CrtCarry &) noexcept;
    // Explicit input representation: direct CRT may consume coefficients
    // multiplied by this per-prime factor (one for the Garner backend).
    uint64_t (*input_scale)(unsigned prime) noexcept;
    void (*prepare_scaled)(unsigned trunk_bits,bool input_preconditioned,std::span<uint8_t>) noexcept;
};
const CrtBackend *p48_crt_backend(unsigned primes) noexcept;
} // namespace sbn::v3::product
