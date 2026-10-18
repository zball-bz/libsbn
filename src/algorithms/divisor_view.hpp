// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/divrem.h"
namespace sbn::v3 {
// Read-only normalized magnitude already owned by a prepared divider. Used
// for signed rounding without copying D or scanning it at every execution.
struct DivisorWords {
    const uint64_t*data;size_t limbs;unsigned shift,radix_bits;
    uint64_t word(size_t j)const noexcept {
        if(radix_bits==64)return (data[j]>>shift)|(shift&&j+1<limbs?data[j+1]<<(64-shift):0);
        const size_t bit=shift+64*j,k=bit/52;const unsigned s=bit%52;
        uint64_t w=(data[k]>>s)|(data[k+1]<<(52-s));if(s>40)w|=data[k+2]<<(104-s);return w;
    }
};
DivisorWords prepared_divisor_words(sbn3_divrem_binding*)noexcept;
}
