#pragma once
#include <utility>
namespace sbn::v3::u52 {
// Quotient estimates need only the upper band of an 18x18 product.
// Keep terms i+j>=Start. With B=2^52 the omitted positive integer is
// <18*B^(Start+1); the quotient funnel starts at 52*(Start+3)+51 bits,
// so the additional under-estimate is <2^-150 quotient ulps. The exact
// remainder/correction loop remains authoritative at a rounding boundary.
template<unsigned Start,unsigned J,unsigned K=0>
[[gnu::always_inline]] inline void dc_estimate_segment(const sb_vec (&a)[3],sb_vec b,sb_vec (&lo)[(44-Start)/8],
                               sb_vec (&hi)[(44-Start)/8]) {
    constexpr unsigned first=Start-J,count=18-first,offset=8*K;
    if constexpr(offset<count){
        constexpr unsigned index=(first+offset)/8,shift=(first+offset)%8;
        // Keep the common reciprocal in registers. Re-loading every shifted
        // window costs more load/unaligned work than the removed products.
        const sb_vec x=shift?sb_alignr64(index+1<3?a[index+1]:sb_zero(),a[index],shift):a[index];
        lo[K]=sb_madd52lo(lo[K],x,b);hi[K]=sb_madd52hi(hi[K],x,b);
    }
    if constexpr(K+1<(44-Start)/8)dc_estimate_segment<Start,J,K+1>(a,b,lo,hi);
}
template<unsigned Start,unsigned J>
[[gnu::always_inline]] inline void dc_estimate_row(const sb_vec (&a)[3],const sb_limb *b,sb_vec (&lo)[(44-Start)/8],
                           sb_vec (&hi)[(44-Start)/8]) {
    if constexpr(J<=Start && Start-J<18){
        dc_estimate_segment<Start,J>(a,sb_set1_64(b[J]),lo,hi);
    }else if constexpr(J>Start){
        static_assert(Start==16 && (J==17 || J==18));
        constexpr unsigned shift=8-(J-Start);
        const sb_vec y=sb_set1_64(b[J]);sb_vec l[3],h[3];
        for(unsigned k=0;k<3;++k){
            l[k]=sb_madd52lo(sb_zero(),a[k],y);h[k]=sb_madd52hi(sb_zero(),a[k],y);
        }
        lo[0]=sb_add(lo[0],sb_alignr64(l[0],sb_zero(),shift));
        hi[0]=sb_add(hi[0],sb_alignr64(h[0],sb_zero(),shift));
        for(unsigned k=1;k<3;++k){
            lo[k]=sb_add(lo[k],sb_alignr64(l[k],l[k-1],shift));
            hi[k]=sb_add(hi[k],sb_alignr64(h[k],h[k-1],shift));
        }
    }
}
template<unsigned Start,size_t... J>
inline void dc_estimate_upper_impl(sb_limb *out,const sb_limb *a,const sb_limb *b,std::index_sequence<J...>) {
    static_assert(Start==16 || Start==24 || Start==28);
    const sb_vec common[3]={sb_load((sb_cpvec)a),sb_load((sb_cpvec)(a+8)),sb_load((sb_cpvec)(a+16),3)};
    sb_vec lo[(44-Start)/8]{},hi[(44-Start)/8]{};
    (dc_estimate_row<Start,J>(common,b,lo,hi),...);
    sb_vec previous=sb_zero();
    for(unsigned k=0;k<(44-Start)/8;++k){
        sb_store((sb_pvec)(out+Start+8*k),sb_add(lo[k],sb_alignr64(hi[k],previous,7)));
        previous=hi[k];
    }
}
template<unsigned Start,unsigned InputDigits=18>
inline void dc_estimate_upper(sb_limb *out,const sb_limb *a,const sb_limb *b) {
    static_assert(InputDigits==18 || InputDigits==19);
    dc_estimate_upper_impl<Start>(out,a,b,std::make_index_sequence<InputDigits>{});
}
} // namespace sbn::v3::u52
