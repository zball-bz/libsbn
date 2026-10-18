#pragma once
#include "runtime/scratch.hpp"
#include "radix/geometry.hpp"
#include <array>
#include <utility>
namespace sbn::v3::radix::leaf_detail {
// A right split omits all columns below its two guard digits. Their carry
// into those digits is < 64*2^52, hence the requested integer window is at
// most one below its exact value. Each retained node has >=16 guard bits
// beyond its last overlap digit. The sum of all local errors is <1 there.
inline void block_product(__m512i *out,unsigned right,const __m512i *x,unsigned n,
                          const FractionBlockPlan &p,unsigned level){
    const auto zero=_mm512_setzero_si512(),mask=_mm512_set1_epi64((uint64_t(1)<<52)-1);
    const unsigned lo=n-right,first=lo>1?lo-2:0;
    auto carry=zero;
    for(unsigned col=first;col<n;++col){
        __m512i low4[4]{carry,zero,zero,zero},high4[4]{zero,zero,zero,zero};
        unsigned j=p.begin[level];const unsigned end=std::min(col+1,p.end[level]);
        // Four independent accumulators hide IFMA latency on both issue
        // pipes. Each column's total remains below 64*2^52.
        for(;j+4<=end;j+=4){
            #pragma clang loop unroll(full)
            for(unsigned k=0;k<4;++k){
                const auto c=_mm512_set1_epi64(p.power[level][j+k]),v=_mm512_load_si512(x+col-j-k);
                low4[k]=_mm512_madd52lo_epu64(low4[k],v,c);high4[k]=_mm512_madd52hi_epu64(high4[k],v,c);
            }
        }
        auto low=_mm512_add_epi64(_mm512_add_epi64(low4[0],low4[1]),_mm512_add_epi64(low4[2],low4[3]));
        auto high=_mm512_add_epi64(_mm512_add_epi64(high4[0],high4[1]),_mm512_add_epi64(high4[2],high4[3]));
        for(;j<end;++j){const auto c=_mm512_set1_epi64(p.power[level][j]),v=_mm512_load_si512(x+col-j);
            low=_mm512_madd52lo_epu64(low,v,c);high=_mm512_madd52hi_epu64(high,v,c);}
        carry=_mm512_add_epi64(high,_mm512_srli_epi64(low,52));
        if(col>=lo)_mm512_store_si512(out+col-lo,_mm512_and_si512(low,mask));
    }
}
template<unsigned N>[[gnu::noinline]]
void block_terminal(uint64_t *out,const __m512i *input,uint64_t radix){
    __m512i x[N];const auto zero=_mm512_setzero_si512(),b=_mm512_set1_epi64(radix),mask=_mm512_set1_epi64((uint64_t(1)<<52)-1);
    #pragma clang loop unroll(full)
    for(unsigned j=0;j<N;++j)x[j]=_mm512_load_si512(input+j);
    #pragma clang loop unroll(full)
    for(unsigned r=0;r<9;++r){auto high=zero,carry=zero;
        #pragma clang loop unroll(full)
        for(unsigned j=0;j<N;++j){auto low=_mm512_madd52lo_epu64(high,x[j],b);
            high=_mm512_madd52hi_epu64(zero,x[j],b);low=_mm512_add_epi64(low,carry);
            carry=_mm512_srli_epi64(low,52);x[j]=_mm512_and_si512(low,mask);}
        _mm512_store_si512(out+8*r,_mm512_add_epi64(high,carry));
    }
}
inline void block_subtree(uint64_t *out,const __m512i *x,unsigned n,unsigned fragments,
                          const DigitPlan &dp,const FractionBlockPlan &p,Frame &space){
    if(fragments==1){const unsigned keep=std::min(n,p.precision[1]);x+=n-keep;
        #define LEAF_CASE(N) case N:block_terminal<N>(out,x,dp.b8);break;
        switch(keep){LEAF_CASE(1) LEAF_CASE(2) LEAF_CASE(3) LEAF_CASE(4) LEAF_CASE(5)
            LEAF_CASE(6) LEAF_CASE(7) LEAF_CASE(8) LEAF_CASE(9) LEAF_CASE(10)
            default:fatal(SBN3_FATAL_ARGUMENT,"block tree terminal precision");}
        #undef LEAF_CASE
        return;
    }
    const unsigned left=fragments/2,right=fragments-left,level=left-1;
    {
        FrameMark mark(space);const unsigned rn=std::min(n,p.precision[right]);
        auto *value=space.alloc<__m512i>(rn);
        block_product(value,rn,x,n,p,level);
        block_subtree(out+72*left,value,rn,right,dp,p,space);
    }
    const unsigned ln=std::min(n,p.precision[left]);
    block_subtree(out,x+n-ln,ln,left,dp,p,space);
}
}
namespace sbn::v3::radix {
namespace leaf_detail {
template<unsigned Base>constexpr FractionBlockPlan make_fraction_block_plan(){
    FractionBlockPlan p{};
    if constexpr(Base>=3&&Base<64&&(Base&(Base-1))){
        for(unsigned f=1;f<=16;++f)p.precision[f]=unsigned((
            uint64_t((__uint128_t(log2_floor_q56[Base]+1)*(64*f+8))>>56)+2+guard_bits+51)/52);
        uint64_t value[64]{1};unsigned exponent=0;
        for(unsigned k=0;k<8;++k){
            for(;exponent<64u*(k+1);++exponent){uint64_t carry=0;
                for(unsigned j=0;j<64;++j){const uint64_t v=value[j]*Base+carry;
                    value[j]=v&((uint64_t(1)<<52)-1);carry=v>>52;}}
            for(unsigned j=0;j<64;++j)p.power[k][j]=value[j];
            while(p.begin[k]<64&&!p.power[k][p.begin[k]])++p.begin[k];
            p.end[k]=64;while(p.end[k]>p.begin[k]&&!p.power[k][p.end[k]-1])--p.end[k];
        }
    }
    return p;
}
template<unsigned Base>inline constexpr auto fraction_block_constant=make_fraction_block_plan<Base>();
template<size_t... Bases>constexpr auto block_constant_directory(std::index_sequence<Bases...>){
    return std::array<const FractionBlockPlan *,sizeof...(Bases)>{&fraction_block_constant<Bases>...};
}
inline constexpr auto block_constants=block_constant_directory(std::make_index_sequence<65>{});
inline constexpr size_t block_constant_bytes=sizeof(block_constants)+65*sizeof(FractionBlockPlan);
}
const FractionBlockPlan &fraction_block_plan(unsigned base) noexcept {
    require(base>=3&&base<64&&(base&(base-1)),SBN3_FATAL_ARGUMENT,"block tree base");
    return *leaf_detail::block_constants[base];
}
size_t fraction_block_tree_scratch_bytes(unsigned limbs,unsigned fragments) noexcept {
    // Root fraction plus at most four live right-child fractions. The
    // output-word matrix survives the recursion.
    const size_t n=(64*limbs+51)/52;
    return 64*(9*fragments+5*n+8);
}
void emit_fraction_block_tree(const FractionBlock *blocks,unsigned count,const DigitPlan &dp,
                              const FractionBlockPlan &p,Frame &space) noexcept {
    require(count&&count<=8,SBN3_FATAL_ARGUMENT,"block tree lanes");
    const unsigned limbs=blocks[0].limbs,fragments=blocks[0].fragments,n=(64*limbs+51)/52;
    require(limbs&&limbs<=fraction_block_tree_limbs&&fragments&&fragments<=16,SBN3_FATAL_ARGUMENT,"block tree shape");
    for(unsigned u=1;u<count;++u)require(blocks[u].limbs==limbs&&blocks[u].fragments==fragments,SBN3_FATAL_ARGUMENT,"block tree homogeneous batch");
    FrameMark mark(space);auto *words=space.alloc<uint64_t>(72*fragments);auto *x=space.alloc<__m512i>(n);
    for(unsigned j=0;j<n;++j){alignas(64)uint64_t lane[8]{};const int bit=int(64*limbs)-int(52*(n-j));
        const unsigned lo=bit<0?0:unsigned(bit),skip=lo-bit,word=lo/64,r=lo%64;
        for(unsigned u=0;u<count;++u){uint64_t v=blocks[u].value[word]>>r;
            if(r&&word+1<limbs)v|=blocks[u].value[word+1]<<(64-r);
            lane[u]=(v<<skip)&((uint64_t(1)<<52)-1);}
        _mm512_store_si512(x+j,_mm512_load_si512(lane));
    }
    leaf_detail::block_subtree(words,x,n,fragments,dp,p,space);
    const auto radix=_mm512_set1_epi64(dp.b8),one=_mm512_set1_epi64(1),zero=_mm512_setzero_si512();
    const __mmask8 active=__mmask8((1u<<count)-1);
    for(unsigned f=fragments-1;f-->0;){const auto next=_mm512_load_si512(words+72*(f+1)),overlap=_mm512_load_si512(words+72*f+64);
        const auto plus=_mm512_add_epi64(overlap,one);
        const auto wrapped=_mm512_mask_sub_epi64(plus,_mm512_cmpeq_epi64_mask(plus,radix),plus,radix);
        require(!(active&~(_mm512_cmpeq_epi64_mask(overlap,next)|_mm512_cmpeq_epi64_mask(wrapped,next))),SBN3_FATAL_MATH,"block tree seam");
        auto carry=__mmask8(active&_mm512_cmpeq_epi64_mask(plus,radix)&_mm512_cmpeq_epi64_mask(next,zero));
        for(unsigned r=8;carry&&r-->0;){auto value=_mm512_load_si512(words+72*f+8*r);
            value=_mm512_mask_add_epi64(value,carry,value,one);
            carry=__mmask8(carry&_mm512_cmpeq_epi64_mask(value,radix));
            value=_mm512_mask_sub_epi64(value,carry,value,radix);_mm512_store_si512(words+72*f+8*r,value);}
    }
    const leaf_detail::Emitter emit(dp);
    if(count<4){
        // For short batches put the eight words of one fragment in SIMD
        // lanes, so byte emission has no inactive lanes or transpose.
        const auto index=_mm512_setr_epi64(0,8,16,24,32,40,48,56);
        for(unsigned f=0;f<fragments;++f)for(unsigned u=0;u<count;++u)
            _mm512_storeu_si512(blocks[u].output+64*f,emit(_mm512_i64gather_epi64(index,words+72*f+u,8)));
    }else for(unsigned f=0;f<fragments;++f){__m512i digits[8];
        for(unsigned r=0;r<8;++r)digits[r]=emit(_mm512_load_si512(words+72*f+8*r));
        leaf_detail::transpose(digits);
        for(unsigned u=0;u<count;++u)_mm512_storeu_si512(blocks[u].output+64*f,digits[u]);
    }
    for(unsigned u=0;u<count;++u){*blocks[u].first=words[u];*blocks[u].overlap=words[72*(fragments-1)+64+u];}
}
}
