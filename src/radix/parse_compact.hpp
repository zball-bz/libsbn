#pragma once
// Small parse trees use the same leaves, powers and multiplication kernels as
// the staged tree. Their complete geometry fits in the unused child-plan area
// of the public plan: no stage directory or product-service binding is needed.
#include "radix/parse_tree.hpp"
#include "radix/power_bank.hpp"
#include "product/local_product.hpp"
#include "value/divrem_words.hpp"
namespace sbn::v3::radix {
inline constexpr size_t compact_parse_fragments=size_t(1)<<fixed_power_levels;
inline constexpr size_t compact_parse_precision=group_fragments*parse_leaf_limbs;
struct CompactParsePlan {
    struct Node {unsigned digits=0,level=0;int low=0,high=0;unsigned product=0;
        size_t words=0,shift=0,work=0;};
    struct Power {unsigned level=0,product=0;size_t words=0;};
    ParseLeafPlan leaf{};
    Node nodes[12]{};
    product::LocalProductPlan products[24]{};
    Power powers[9]{};
    unsigned base=0,node_count=0,product_count=0,power_count=0;
    int root=0,seed_level=-1;
    uint64_t seed[8]{};
    size_t seed_words=0,power_words=0,number_words=0,numerator_words=0,denominator_words=0;
    size_t persistent_bytes=0,work_bytes=0;
    int64_t shift=0;
    bool native_division=false;
};
struct CompactParseState {
    divrem_words::PreparedDivisor word{};
    u52::DivisionDivisor native{};
};
bool compact_parse_query(unsigned base,size_t digits,size_t fraction_digits,uint64_t bits,CompactParsePlan &) noexcept;
uint64_t compact_parse_identity(const CompactParsePlan &) noexcept;
void compact_parse_prepare(const CompactParsePlan &,CompactParseState &,void *storage) noexcept;
bool compact_parse_execute(const CompactParsePlan &,const CompactParseState &,const DigitPlan &,
                           const uint8_t *,size_t digits,uint64_t bits,uint64_t *,size_t out_words,void *storage) noexcept;
}
