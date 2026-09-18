/* The generic basecase loop is adapted from libsbn/include/sbn/scalar.h,
 * itself a GMP generic mul_basecase port. See scalar-donor-notice.txt. */
#include "common/checked.hpp"
#include "common/small_checks.h"
#include "core/x86_64/word.hpp"
#include <string.h>
using namespace sbn::v3;
#ifndef SBN3_MUL2_STREAM
#define SBN3_MUL2_STREAM 2
#endif
/* Two ordinary schoolbook carry chains, interleaved so A is read once and
 * no partial result is read back. Each 128-bit sum is at most 2^128-1. */
[[maybe_unused]] static void mul2_stream(uint64_t *r,const uint64_t *a,size_t n,const uint64_t *b){
    const uint64_t b0=b[0],b1=b[1];uint64_t previous=a[0];
    __uint128_t first=__uint128_t(previous)*b0;r[0]=uint64_t(first);
    __uint128_t previous_high=__uint128_t(previous)*b1;
    uint64_t ca=uint64_t(first>>64),cb=0;
#pragma clang loop unroll_count(4)
    for(size_t i=1;i<n;++i){
        const uint64_t current=a[i];const __uint128_t low=__uint128_t(current)*b0+ca;
        const __uint128_t high=previous_high+uint64_t(low)+cb;
        previous_high=__uint128_t(current)*b1;
        ca=uint64_t(low>>64);cb=uint64_t(high>>64);r[i]=uint64_t(high);
    }
    const __uint128_t last=previous_high+ca+cb;
    r[n]=uint64_t(last);r[n+1]=uint64_t(last>>64);
}

static void addsub_args(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n) {
    require(n <= static_cast<size_t>(LONG_MAX), SBN3_FATAL_SIZE, "addsub length");
    const size_t nb = bytes_for(n, 8);
    valid_span(r, nb, "addsub output"); valid_span(a, nb, "addsub a"); valid_span(b, nb, "addsub b");
    require((r == a || !overlaps(r, nb, a, nb)) &&
            (r == b || !overlaps(r, nb, b, nb)), SBN3_FATAL_ARGUMENT, "addsub overlap");
}
extern "C" uint64_t sbn3_add_n(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n) {
    addsub_args(r, a, b, n);
    return n ? sbn3i_add_n(r, a, b, static_cast<long>(n)) : 0;
}
extern "C" uint64_t sbn3_sub_n(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n) {
    addsub_args(r, a, b, n);
    return n ? sbn3i_sub_n(r, a, b, static_cast<long>(n)) : 0;
}
extern "C" void sbn3_mul_basecase(uint64_t *r, size_t capacity,
                                   const uint64_t *a, size_t an,
                                   const uint64_t *b, size_t bn) {
    (void)capacity;
#if SBN3_CHECK_SMALL
    size_t rn;
    require(add_size(an, bn, rn) && rn <= capacity && rn <= static_cast<size_t>(LONG_MAX),
            SBN3_FATAL_SIZE, "basecase capacity", an, capacity);
    const size_t rb = bytes_for(rn, 8), ab = bytes_for(an, 8), bb = bytes_for(bn, 8);
    valid_span(r, rb, "basecase output"); valid_span(a, ab, "basecase a"); valid_span(b, bb, "basecase b");
    require(!overlaps(r, rb, a, ab) && !overlaps(r, rb, b, bb), SBN3_FATAL_ARGUMENT, "basecase overlap");
#endif
    mul_basecase_assumed(r,a,an,b,bn);
}
void sbn::v3::mul_basecase_assumed(uint64_t *r,const uint64_t *a,size_t an,const uint64_t *b,size_t bn) noexcept {
    if (!an || !bn) { if (an+bn) memset(r, 0, (an+bn)*8); return; }
    if(an==1 && bn==1){const __uint128_t p=__uint128_t(a[0])*b[0];r[0]=uint64_t(p);r[1]=uint64_t(p>>64);return;}
    if (an <= 6 && bn <= 6) { sbn3i_mul_basecase_le6(r, a, an, b, bn); return; }
    if (an < bn) { const uint64_t *p = a; a = b; b = p; size_t n = an; an = bn; bn = n; }
    if constexpr(SBN3_MUL2_STREAM)if(bn==2 && (SBN3_MUL2_STREAM==1 || an>=8192)){mul2_stream(r,a,an,b);return;}
    const long n = static_cast<long>(an);
    r[an] = sbn3i_mul_1(r, a, n, b[0]);
    ++r; ++b; --bn;
    while (bn >= 2) {
        r[an + 1] = sbn3i_addmul_2(r, a, n, b);
        r += 2; b += 2; bn -= 2;
    }
    if (bn) r[an] = sbn3i_addmul_1(r, a, n, b[0]);
}
extern "C" void sbn3_int_mul_basecase(sbn3_int *out,sbn3_int_view a,sbn3_int_view b) {
    require(out && a.negative<=1 && b.negative<=1,SBN3_FATAL_ARGUMENT,"integer magnitude/sign");
    size_t n=0;require(add_size(a.size,b.size,n),SBN3_FATAL_SIZE,"integer result size");
    require(!overlaps(out,sizeof *out,out->data,bytes_for(n,8)),SBN3_FATAL_ARGUMENT,"integer descriptor overlap");
    sbn3_mul_basecase(out->data,out->capacity,a.data,a.size,b.data,b.size);
    while(n && out->data[n-1]==0)--n;
    out->size=n;out->negative=n?(a.negative^b.negative):0;
}
