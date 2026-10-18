#pragma once
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3::radix {
inline constexpr unsigned fixed_power_levels=9;
struct FixedPower { const uint64_t *data=nullptr; size_t capacity=0; };
// Published mathematical constants odd^(64*2^level), odd=3,5,...,63.
// Entries are 64-byte aligned and include at least eight zero padding words.
FixedPower fixed_power(unsigned odd,unsigned level) noexcept;
size_t fixed_power_bytes() noexcept;
struct ParseLeafPlan {
    unsigned twos=0;
    size_t limbs[9]{},power_limbs[3]{};
    const uint64_t *powers[3]{};
};
const ParseLeafPlan &parse_leaf_plan(unsigned base) noexcept;
// Exact normalized reciprocals for small integer-to-fraction scaling.
// Prefixes are shared by bases with the same odd part. Larger shapes use
// the ordinary runtime inverse; the published bank has a fixed size bound.
inline constexpr unsigned fixed_integer_inverse_words=264,fixed_integer_inverse_fragments=176;
struct FixedIntegerInverse {const uint64_t *data=nullptr;unsigned shift=0;};
FixedIntegerInverse fixed_integer_inverse(unsigned base,unsigned fragments,size_t words) noexcept;
}
