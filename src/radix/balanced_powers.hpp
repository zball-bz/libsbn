#pragma once
#include "radix/programs.hpp"
#include "radix/plan_slots.hpp"
namespace sbn::v3::radix {
struct RailProductPlan;
struct RailProduct;
// At most two adjacent exponents per depth, for each of the two format roots,
// plus their short leaf powers. Only requested powers survive preparation;
// construction-only half powers live in the setup workspace.
inline constexpr unsigned max_balanced_powers=4*max_rail+8;
struct BalancedPowers {
    struct Entry {
        uint64_t exponent=0;
        size_t limbs=0,offset=0;
        unsigned parent=0,fixed_level=UINT32_MAX;
        bool retained=false,service=false,cached_square=false;
        ProductChoice square_choice=0;
    };
    PlanView<Entry,max_balanced_powers> entries;
    BalancedPowers() noexcept {}
    unsigned count=0;
    size_t total_limbs=0,auxiliary_bytes=0,setup_bytes=0;
    int request(const BaseInfo&,uint64_t exponent,bool retained=true) noexcept;
    int find(uint64_t exponent)const noexcept;
    sbn3_query_result finish(const BaseInfo&,unsigned workers,PlanTranscript*,const RailProductPlan *const *caches=nullptr) noexcept;
};
void balanced_powers_build(const BaseInfo&,const BalancedPowers&,unsigned workers,Arena&,sbn3_team&,
                           sbn3_lease&,uint64_t*,const uint64_t**,RailProduct *const *caches=nullptr) noexcept;
}
