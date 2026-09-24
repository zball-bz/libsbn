#include "backend/u52/island.hpp"
#include <algorithm>
namespace sbn::v3::u52 {
namespace {
static size_t digits(size_t n){return (64*n+51)/52;}
static size_t prepared_bytes(size_t n) noexcept {return 64*((digits(n)+23)/8);}
static size_t work_bytes(size_t n) noexcept {return prepared_bytes(n)+64;}
static const uint64_t *prepare_digits(const uint64_t*a,size_t n,Frame&space) noexcept {
    auto *storage=static_cast<uint64_t*>(space.allocate(prepared_bytes(n),64));
    memset(storage,0,prepared_bytes(n));auto *p=storage+8;
    u52_from_u64(reinterpret_cast<sb_pvec>(p),a,n);return p;
}
static void apply_high(uint64_t*out,const uint64_t*a,size_t an,const uint64_t*b,size_t bn,Frame&space) noexcept {
    require(an>=3&&an<=512&&bn&&bn<=an,SBN3_FATAL_ARGUMENT,"high guard shape");
    FrameMark mark(space);const auto *v=prepare_digits(b,bn,space);
    const size_t na=digits(an),nb=digits(bn),first=(64*an-128)/52,end=digits(an+bn);
    // Omitted lower diagonals contribute a carry <min(na,nb)*2^52.
    // The 128-bit guard makes its quotient contribution <2^-65, hence the
    // integer high product is an underestimate by at most one.
    memset(out,0,bn*8);__uint128_t carry=0;
    constexpr uint64_t mask=(uint64_t(1)<<52)-1;
    for(size_t k=first;k<end;k+=8){
        __m512i lo0=_mm512_setzero_si512(),hi0=lo0,lo1=lo0,hi1=lo0;
        size_t t=k>=na?k-na+1:0;const size_t stop=std::min(nb,k+8);
        for(;t+1<stop;t+=2){
            const auto x0=_mm512_loadu_si512(a+(static_cast<ptrdiff_t>(k)-static_cast<ptrdiff_t>(t)));
            const auto x1=_mm512_loadu_si512(a+(static_cast<ptrdiff_t>(k)-static_cast<ptrdiff_t>(t+1)));
            const auto y0=_mm512_set1_epi64(v[t]),y1=_mm512_set1_epi64(v[t+1]);
            lo0=_mm512_madd52lo_epu64(lo0,x0,y0);hi0=_mm512_madd52hi_epu64(hi0,x0,y0);
            lo1=_mm512_madd52lo_epu64(lo1,x1,y1);hi1=_mm512_madd52hi_epu64(hi1,x1,y1);
        }
        if(t<stop){const auto x=_mm512_loadu_si512(a+(static_cast<ptrdiff_t>(k)-static_cast<ptrdiff_t>(t))),y=_mm512_set1_epi64(v[t]);
            lo0=_mm512_madd52lo_epu64(lo0,x,y);hi0=_mm512_madd52hi_epu64(hi0,x,y);}
        alignas(64) uint64_t low[8],high[8];
        _mm512_store_si512(low,_mm512_add_epi64(lo0,lo1));_mm512_store_si512(high,_mm512_add_epi64(hi0,hi1));
        for(size_t j=0;j<8&&k+j<end;++j){
            const __uint128_t sum=carry+low[j]+(__uint128_t(high[j])<<52);
            const uint64_t digit=uint64_t(sum)&mask;carry=sum>>52;
            const ptrdiff_t bit=ptrdiff_t(52*(k+j))-ptrdiff_t(64*an);
            if(bit<0){if(bit> -52)out[0]|=digit>>unsigned(-bit);continue;}
            const size_t word=size_t(bit)/64;const unsigned shift=unsigned(bit)%64;
            if(word<bn)out[word]|=digit<<shift;
            if(shift>12&&word+1<bn)out[word+1]|=digit>>(64-shift);
        }
    }
}

} // namespace
size_t high_prefix_scratch_bytes(size_t bn) noexcept {
    require(bn&&bn<=510,SBN3_FATAL_SIZE,"high prefix size");
    return prepared_bytes(bn+2)+work_bytes(bn)+128;
}
void high_prefix(uint64_t*out,const uint64_t*a,size_t an,const uint64_t*b,size_t bn,Frame&space) noexcept {
    require(bn&&bn<=510&&an>=bn+2,SBN3_FATAL_ARGUMENT,"high prefix geometry");
    FrameMark mark(space);const size_t keep=bn+2;
    // Removing the lower A words adds <B^-2 to the omitted fraction.
    // Together with the guarded band this remains strictly below one ulp.
    const auto *top=prepare_digits(a+an-keep,keep,space);
    apply_high(out,top,keep,b,bn,space);
}
} // namespace sbn::v3::u52
