#pragma once
#include <stddef.h>
#include <stdint.h>
#include "sbn3/team.h"
namespace sbn::v3 {
class Frame;
namespace pq16 {
enum class Recipe : uint8_t {PfaPQ=0,CooleyTukeyPQ=1,RightAngle=2};
struct Shape {uint32_t nfull,branch,radix;bool centered;Recipe recipe=Recipe::PfaPQ;unsigned bits=16;bool balanced=false;}; // balanced: locally signed digits (wide codec only)
// Geometry remains separate from the supported execution recipe. 16-bit odd
// radix at W1 up to N=20480 runs the right-angle recipe (2026-09-08 public
// scan: -4.1% at those 21 lengths; M3 24576 / M7 28672 stayed 1-3% faster
// on PFA and keep it). pow2 keeps the fused PQ pipeline, wider teams and the
// centered band keep PFA. CT/PQ remains the wide-codec recipe.
constexpr Shape execution_shape(Shape s,unsigned workers) noexcept {
    if(workers==1&&!s.centered&&s.bits==16&&s.radix!=1&&s.nfull<=20480)s.recipe=Recipe::RightAngle;
    else s.recipe=workers==1&&!s.centered&&s.nfull>=2048&&s.nfull<=32768&&
        (s.radix==5||(s.radix==7&&s.branch<=2048))?
        Recipe::CooleyTukeyPQ:Recipe::PfaPQ;
    return s;
}
struct Tables;
Shape query(size_t an,size_t bn,unsigned minimum_pow2=128) noexcept;
Shape select(size_t an,size_t bn,unsigned workers=1,unsigned bits=0,size_t scratch_budget=0,bool square=false,
             double preparation_ns_per_byte_per_use=0) noexcept;
size_t table_bytes(Shape) noexcept;
size_t scratch_bytes(Shape,size_t an,size_t bn,unsigned workers=1,bool square=false) noexcept;
Tables *prepare(Frame &,Shape) noexcept;
void multiply(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t,
              const Tables &,Frame &,sbn3_team_scope *scope=nullptr) noexcept;
bool supported(Shape,size_t an,size_t bn,unsigned workers) noexcept;
size_t cached_scratch_bytes(Shape,size_t an,size_t bn) noexcept;
void forward_spectrum(double *,const uint64_t *,size_t,const Tables &,sbn3_team_scope *) noexcept;
void apply_spectrum(uint64_t *,const double *,const uint64_t *original,size_t an,
                     const uint64_t *fresh,size_t bn,bool square,const Tables &,Frame &,sbn3_team_scope *) noexcept;
Shape plus_shape(size_t minimum_limbs) noexcept;
bool plus_supported(Shape,size_t an,size_t bn) noexcept;
void plus_multiply(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t,bool,
                   const double *,const Tables &,Frame &) noexcept;
constexpr size_t cyclic_period(Shape s) noexcept {return size_t(s.nfull)*s.bits/32;}
Shape cyclic_shape(size_t minimum_limbs,unsigned bits=16) noexcept;
bool cyclic_supported(Shape,size_t an,size_t bn) noexcept;
void cyclic_multiply(uint64_t *,const uint64_t *,size_t,const uint64_t *,size_t,bool square,
                       const double *cached,const Tables &,Frame &,sbn3_team_scope *,size_t prefix=0) noexcept;
}
}
