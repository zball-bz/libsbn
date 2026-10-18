#pragma once
#include <array>
#include <stdint.h>
namespace sbn::v3::radix::leaf_detail {
struct alignas(64) LimbRadix {uint64_t divisor[8]{},reciprocal[8]{};unsigned active=0;};
consteval auto make_limb_radices(){
    std::array<LimbRadix,65> out{};
    for(unsigned b=3;b<=63;++b){
        uint64_t step=1;for(unsigned j=0;j<8;++j)step*=b;
        __uint128_t power=step;
        for(unsigned j=1;j<8&&power<=UINT64_MAX;++j){
            out[b].divisor[j]=uint64_t(power);
            out[b].reciprocal[j]=uint64_t((__uint128_t(1)<<64)/power);
            out[b].active|=1u<<j;power*=step;
        }
    }
    return out;
}
inline constexpr auto limb_radices=make_limb_radices();
}
