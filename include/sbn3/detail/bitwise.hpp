// Original libsbn code; LGPL-3.0-or-later. Scalar ABI to the private ISA island.
#pragma once
#include <cstddef>
#include <cstdint>
namespace sbn::detail {
// Canonical lengths: bit63 carries the sign, low bits the normalized limb
// count. This is an internal call convention, never a stored value format.
std::size_t bit_and(std::uint64_t*,const std::uint64_t*,std::size_t,const std::uint64_t*,std::size_t)noexcept;
std::size_t bit_or(std::uint64_t*,const std::uint64_t*,std::size_t,const std::uint64_t*,std::size_t)noexcept;
std::size_t bit_xor(std::uint64_t*,const std::uint64_t*,std::size_t,const std::uint64_t*,std::size_t)noexcept;
std::size_t word_population(const std::uint64_t*,std::size_t)noexcept;
}
