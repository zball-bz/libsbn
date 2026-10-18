// Original libsbn code; LGPL-3.0-or-later. Scalar ABI to shared local division.
#pragma once
#include <cstddef>
#include <cstdint>
namespace sbn::detail {
struct division_sizes {std::size_t quotient,remainder;};
division_sizes local_divrem_checked(std::uint64_t*q,std::uint64_t*r,const std::uint64_t*n,std::size_t nn,
                                    const std::uint64_t*d,std::size_t dn,void*scratch,std::size_t bytes)noexcept;
}
