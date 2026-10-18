// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/constant_bank.hpp"
namespace sbn::v3::floating::constant_bank_detail {
inline constexpr std::uint64_t value_magic=UINT64_C(0x53424e3343425631);
struct Values {
    std::uint64_t marker=0;std::size_t bytes=0,pi_bits=0,log2_bits=0;
    std::uint32_t ready=0,pi_ready=0,log2_ready=0;
    float_buffer* pi=nullptr;float_buffer* log2=nullptr;float_report pi_report{},log2_report{};
};
const Values* checked(ConstantBankView) noexcept;
bool radius_good(float_view,error_bound,std::size_t,ConstantKind) noexcept;
}
