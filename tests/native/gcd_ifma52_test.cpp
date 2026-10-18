// Original libsbn tests; LGPL-3.0-or-later.
#include "kernels/x86v4ifma/gcd_ifma52.hpp"
#include <cassert>
#include <cstring>
#include <cstdio>
#include <initializer_list>

extern "C" void gcd_ifma52_test() {
    namespace k=sbn::v3::gcd_small::ifma52;
    using U=uint64_t;using I=__int128;
    constexpr U mask=k::digit_mask;
    U seed=0x284feca09bdf1637ULL;
    auto random=[&](){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;};
    // Scalar signed 128-bit products are the oracle. Three consecutive
    // blocks also test the only carry state that crosses a vector boundary.
    const U edges[]{0,1,2,mask/2,mask-2,mask-1,mask};
    for(unsigned test=0;test<100000;++test) {
        const U c=test<49?edges[test/7]:random()&mask;
        const U d=test<49?edges[test%7]:random()&mask;
        I scalar=int64_t(random()&mask)-int64_t(random()&mask);
        auto state=_mm512_set1_epi64(U(scalar));
        for(unsigned block=0;block<3;++block) {
            alignas(64) U a[8],b[8],want[8],got[8];
            for(unsigned i=0;i<8;++i) {
                a[i]=test%4?random()&mask:edges[(i+block)%7];
                b[i]=test%4?random()&mask:edges[(i+2*block)%7];
                const I value=I(a[i])*c-I(b[i])*d+scalar;
                want[i]=U(value)&mask;scalar=value>>52;
            }
            const auto row=k::difference(_mm512_load_si512(a),_mm512_load_si512(b),
                _mm512_set1_epi64(c),_mm512_set1_epi64((-d)&mask),d?255:0);
            _mm512_store_si512(got,k::carry(row.lo,row.hi,state));
            const I last=int64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(state,3),1));
            assert(!memcmp(got,want,sizeof(got))&&last==scalar);
        }
    }
    // Directed carry/borrow chains, including a chain through all three
    // blocks, and termination at every digit position. No random endpoint
    // hit is needed to cover the exact repair path.
    for(int sign:{-1,1})for(unsigned stop=0;stop<=24;++stop) {
        I scalar=sign;auto state=_mm512_set1_epi64(sign);
        for(unsigned block=0;block<3;++block) {
            alignas(64) int64_t lo[8];alignas(64) U want[8],got[8];
            for(unsigned i=0;i<8;++i) {
                lo[i]=sign>0?int64_t(mask):0;
                if(8*block+i==stop)lo[i]=int64_t(mask/2);
                const I value=I(lo[i])+scalar;want[i]=U(value)&mask;scalar=value>>52;
            }
            _mm512_store_si512(got,k::carry(_mm512_load_si512(lo),_mm512_setzero_si512(),state));
            const I last=int64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(state,3),1));
            assert(!memcmp(got,want,sizeof(got))&&last==scalar);
        }
    }
    puts("GCD IFMA: 300000 signed row blocks, 50 directed carry/borrow chains PASS");
}
