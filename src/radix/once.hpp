#pragma once
#include "sbn3/radix.h"
#include "radix/decimal_small.hpp"
#include "radix/digit_bounds.hpp"
#include "radix/leaf.hpp"
#include "runtime/arena.hpp"
#include "runtime/team.hpp"
#include "common/small_checks.h"
namespace sbn::v3::radix {
enum class OnceKind : unsigned { planned,integer,fraction };
inline bool once_ascii(const sbn3_radix_options *o) noexcept {
    return o&&o->use_alphabet==1&&decimal_alphabet(o->alphabet);
}
inline OnceKind format_once_kind(const sbn3_format_spec &s,const sbn3_radix_options *o) noexcept {
    if(s.base!=10||!once_ascii(o))return OnceKind::planned;
    if(s.exponent2==0&&s.fraction_digits==0&&s.limbs&&s.limbs<=decimal_write_limbs)
        return OnceKind::integer;
    if(s.mode==SBN3_RADIX_EXACT&&s.limbs&&s.limbs<=decimal_fraction_write_limbs&&s.exponent2==-int64_t(64*s.limbs))
        return OnceKind::fraction;
    return OnceKind::planned;
}
inline OnceKind parse_once_kind(const sbn3_parse_spec &s,const sbn3_radix_options *o) noexcept {
    if(s.base!=10||!once_ascii(o))return OnceKind::planned;
    constexpr size_t max_digits=size_t((__uint128_t(64*256)*inverse_logs[10].down)>>64);
    if(s.fraction_digits==0&&s.fraction_bits==0&&s.integer_digits&&s.integer_digits<=max_digits)
        return OnceKind::integer;
    if(s.integer_digits==0&&s.fraction_digits&&s.fraction_digits<=1920&&s.fraction_bits&&
       s.fraction_bits<=64*decimal_fraction_read_limbs&&!(s.fraction_bits%64)&&
       (s.fraction_bits/64)*((s.fraction_digits+18)/19)<=120)return OnceKind::fraction;
    return OnceKind::planned;
}
inline size_t once_integer_width(size_t limbs) noexcept {
    return size_t((integer_digit_capacity(64*limbs,10)+63)&~uint64_t(63));
}
inline size_t once_integer_limbs(size_t digits) noexcept {
    return size_t((__uint128_t(digits)*(log2_floor_q56[10]+1)+((uint64_t(1)<<62)-1))>>62);
}
// Also used by the matched-kernel diagnostic; no selection or binding here.
bool once_integer_parse_kernel(uint64_t *,size_t,const uint8_t *,size_t) noexcept;
struct OnceSpace {
    Arena *arena=nullptr;sbn3_lease lease{};void *data=nullptr;
    OnceSpace(const sbn3_radix_workspace *w,size_t bytes,size_t alignment) noexcept {
        if(!bytes)return;
        require(w&&w->arena&&w->team&&bytes<=w->bytes,SBN3_FATAL_WORKSPACE,"radix once work",bytes,w?w->bytes:0);
        data=w->arena->base+w->offset;
        if constexpr(SBN3_CHECK_SMALL){
            size_t end;
            require(!(uintptr_t(data)&(alignment-1))&&add_size(w->offset,bytes,end)&&end<=w->arena->virtual_bytes&&
                w->arena->contains(w->offset,end)&&w->arena->unleased(w->offset,bytes),SBN3_FATAL_WORKSPACE,"radix once range");
            require(w->arena->accepts_team(w->team->arena)&&pthread_equal(w->team->creator,pthread_self())&&!w->team->busy,
                SBN3_FATAL_TEAM,"radix once owner");
            arena=w->arena;lease=arena->acquire(w->offset,bytes);
        }
    }
    ~OnceSpace(){if constexpr(SBN3_CHECK_SMALL)if(arena)arena->release(lease);}
};
bool format_once_local_query(const sbn3_format_spec &,const sbn3_radix_options *,sbn3_format_info &,sbn3_query_result &) noexcept;
bool format_once_local(const sbn3_format_spec &,const sbn3_radix_options *,const uint64_t *,uint8_t *,
                       const sbn3_radix_workspace *,sbn3_format_once_result *) noexcept;
// Caller-owned raw lowering of the same local recipes. No arena or Team
// descriptor; complete scratch is returned by the matching local query.
bool format_once_local_raw(const sbn3_format_spec &,const sbn3_radix_options *,const uint64_t *,uint8_t *,
                           void *,size_t,sbn3_format_once_result *) noexcept;
bool parse_once_local_query(const sbn3_parse_spec &,const sbn3_radix_options *,sbn3_parse_info &,sbn3_query_result &) noexcept;
bool parse_once_local_raw(const sbn3_parse_spec &,const sbn3_radix_options *,const uint8_t *,uint64_t *,
                          void *,size_t,sbn3_parse_once_result *) noexcept;
}
