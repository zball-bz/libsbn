// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include "external/store.hpp"
namespace sbn::v3::external {
// Immutable original Region/checksum domain. A view changes neither the file
// identity nor physical offsets in its integrity index. The arithmetic input
// can zero-extend this window to its planned limb capacity.
struct SourceWindow {
    size_t storage_limbs=SIZE_MAX,first_limb=0,limbs=SIZE_MAX;
    // Shift the bounded original magnitude first, then select virtual limbs.
    // Bounds prevent a prefix view from importing discarded neighbouring words.
    size_t lower_limb=0,upper_limb=SIZE_MAX;
    int bit_shift=0; // -63..63; positive is left
    size_t low_zero_words=0; // virtual word i reads physical i-low_zero_words
    bool ones_complement=false; // complement bounded stored words BEFORE shift/virtual guards
    bool operator==(const SourceWindow &) const = default;
};
inline bool resolve_source_window(size_t capacity,SourceWindow &s) noexcept {
    if(s.storage_limbs==SIZE_MAX)s.storage_limbs=capacity;
    if(s.limbs==SIZE_MAX)s.limbs=capacity;
    if(s.upper_limb==SIZE_MAX)s.upper_limb=s.storage_limbs;
    if(s.bit_shift < -63 || s.bit_shift > 63 || s.lower_limb>s.upper_limb||s.upper_limb>s.storage_limbs)return false;
    const size_t extra=s.bit_shift>0;
    if(s.upper_limb>SIZE_MAX-extra||s.low_zero_words>SIZE_MAX-s.upper_limb-extra)return false;
    const size_t end=s.upper_limb+s.low_zero_words+extra;
    if(s.limbs>capacity||s.first_limb<s.lower_limb||s.first_limb>end||s.limbs>end-s.first_limb)return false;
    // Discard equal leading virtual coordinates/bias; the extracted value and
    // original physical checksum domain are unchanged. Prefix views then have
    // one canonical identity whether formed directly or from a wider view.
    const size_t skip=std::min(s.low_zero_words,s.first_limb-s.lower_limb);
    s.first_limb-=skip;s.low_zero_words-=skip;return true;
}
// Exact physical word interval read for a nonempty virtual subwindow.
void source_word_range(const SourceWindow &,size_t offset,size_t count,size_t &begin,size_t &end) noexcept;
// capacity >= (count+2)*8 for shifted input; shift zero can read directly.
// Result occupies the first count words. No temporary file or allocation.
IoError read_source_words(ValueReader &,const SourceWindow &,size_t offset,size_t count,std::span<uint8_t> capacity) noexcept;
} // namespace sbn::v3::external
