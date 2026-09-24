#pragma once
#include "value/limbs.hpp"
#include <algorithm>
namespace sbn::v3::limbs {
// Reduce L+delta modulo B^n+1, 0<=L<B^n and |delta|<2^62.
// The n+1-word canonical form includes the endpoint B^n.
inline void plus_adjust(uint64_t *r,size_t n,int64_t delta) noexcept {
    r[n]=0;
    if(delta>=0){const uint64_t d=uint64_t(delta);
        if(add_to(r,n,&d,1)){
            if(zero(r,n))r[n]=1;
            else{const uint64_t one=1;sub_from(r,n,&one,1);}
        }
    }else{const uint64_t d=uint64_t(-delta);
        if(sub_from(r,n,&d,1)){const uint64_t one=1;r[n]=add_to(r,n,&one,1);}
    }
}
inline void plus_sub_part(uint64_t *r,size_t n,size_t at,const uint64_t *a,size_t an) noexcept {
    const uint64_t high=r[n],borrow=sub_from(r+at,n-at,a,an);
    plus_adjust(r,n,int64_t(borrow)-int64_t(high));
}
inline void plus_add_part(uint64_t *r,size_t n,size_t at,const uint64_t *a,size_t an) noexcept {
    const uint64_t high=r[n],carry=add_to(r+at,n-at,a,an);
    plus_adjust(r,n,-int64_t(high)-int64_t(carry));
}
inline void plus_sub_power(uint64_t *r,size_t n,size_t exponent) noexcept {
    const uint64_t one=1;
    if(exponent<n)plus_sub_part(r,n,exponent,&one,1);
    else plus_add_part(r,n,exponent-n,&one,1);
}
inline void plus_sub_shifted(uint64_t *r,size_t n,const uint64_t *a,size_t an,size_t shift) noexcept {
    const size_t first=std::min(an,n-shift);
    plus_sub_part(r,n,shift,a,first);
    if(an>first)plus_add_part(r,n,0,a+first,an-first);
}
inline bool plus_absolute(uint64_t *r,size_t n) noexcept {
    const bool negative=r[n]||(r[n-1]>>63);
    if(!r[n]&&r[n-1]==(uint64_t(1)<<63)&&zero(r,n-1))return false;
    if(negative){for(size_t j=0;j<n;++j)r[j]=~r[j];const uint64_t two=2;add_to(r,n,&two,1);r[n]=0;}
    return negative;
}
}
