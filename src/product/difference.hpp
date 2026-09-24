#pragma once
#include "product/window.hpp"
#include "product/cyclic_reconstruct.hpp"
#include "value/parallel_limbs.hpp"
namespace sbn::v3::product {
// A normalized low window followed by a disjoint high span. This describes
// both a full shifted value and a streamed concatenation without copying it.
struct DifferenceValue {
    Span low{};
    size_t origin=0,low_words=0;
    unsigned bit_shift=0;
    Span high{};
    size_t words() const noexcept {return low_words+high.words;}
    uint64_t word(size_t j) const noexcept {
        if(j>=low_words){j-=low_words;return j<high.words?high.data[j]:0;}
        const size_t at=origin+j;
        uint64_t x=at<low.words?low.data[at]<<bit_shift:0;
        if(bit_shift&&at&&at-1<low.words)x|=low.data[at-1]>>(64-bit_shift);
        return x;
    }
};
inline uint64_t subtract_value(sbn3_team *team,uint64_t *out,size_t count,const DifferenceValue &v) noexcept {
    require(v.bit_shift<64,SBN3_FATAL_ARGUMENT,"difference normalization");
    const unsigned parts=parallel_limbs::parts(team,count);uint64_t borrows[32]{};
    parallel_limbs::each(team,count,parts,[&](size_t begin,size_t end,unsigned rank){
        size_t at=begin;uint64_t borrow=0;
        auto subtract=[&](const uint64_t *p,size_t n){if(!n)return;
            const auto next=sbn3i_sub_n(out+at,p,out+at,long(n));
            borrow=next+(borrow?parallel_limbs::sub_word(out+at,n,1):0);at+=n;};
        if(at<v.low_words){
            const size_t limit=std::min(end,v.low_words);
            if(!v.bit_shift&&v.origin+at<v.low.words)
                subtract(v.low.data+v.origin+at,std::min(limit-at,v.low.words-v.origin-at));
            for(;at<limit;++at){const unsigned __int128 amount=(unsigned __int128)out[at]+borrow;
                const uint64_t x=v.word(at);out[at]=x-uint64_t(amount);borrow=(unsigned __int128)x<amount;}
        }
        if(at<end&&at<v.words())subtract(v.high.data+at-v.low_words,std::min(end,v.words())-at);
        if(at<end){uint64_t carry=1-borrow;
            for(;at<end;++at){const unsigned __int128 x=(unsigned __int128)(~out[at])+carry;
                out[at]=uint64_t(x);carry=uint64_t(x>>64);}borrow=1-carry;}
        borrows[rank]=borrow;
    });
    uint64_t borrow=0;
    for(unsigned rank=0;rank<parts;++rank){const size_t a=parallel_limbs::cut(count,parts,rank),b=parallel_limbs::cut(count,parts,rank+1);
        borrow=borrows[rank]+(borrow?parallel_limbs::sub_word(out+a,b-a,1):0);}
    return borrow;
}
inline void add_cyclic_word(uint64_t *out,size_t period,uint64_t word) noexcept {
    while(word)word=parallel_limbs::add_word(out,period,word);
}
// Add an exact rank-one high contribution to a linear or cyclic product.
inline void add_shifted_word_product(uint64_t *out,size_t capacity,size_t period,Span a,size_t shift,
                                      uint64_t high,sbn3_team *team=nullptr) noexcept {
    if(!high)return;
    if(!period){
        require(shift<=capacity&&a.words<capacity-shift,SBN3_FATAL_ARGUMENT,"shifted product capacity");
        if(high==1)require(!parallel_limbs::add_to(team,out+shift,capacity-shift,a.data,a.words),SBN3_FATAL_MATH,"shifted product carry");
        else{const auto carry=sbn3i_addmul_1(out+shift,a.data,long(a.words),high);
            require(!parallel_limbs::add_word(out+shift+a.words,capacity-shift-a.words,carry),SBN3_FATAL_MATH,"shifted product carry");}
        return;
    }
    require(shift<period,SBN3_FATAL_ARGUMENT,"cyclic shifted product support");
    if(a.words>period){
        size_t at=0,pos=shift;
        while(at<a.words){const size_t count=std::min(a.words-at,period-pos);
            const auto carry=sbn3i_addmul_1(out+pos,a.data+at,long(count),high);
            if(pos+count==period)add_cyclic_word(out,period,carry);
            else if(parallel_limbs::add_word(out+pos+count,period-pos-count,carry))add_cyclic_word(out,period,1);
            at+=count;pos=(pos+count)%period;
        }
        return;
    }
    const size_t first=std::min(a.words,period-shift);
    if(high==1){
        if(parallel_limbs::add_to(team,out+shift,period-shift,a.data,first))add_cyclic_word(out,period,1);
        if(a.words>first&&parallel_limbs::add_to(team,out,period,a.data+first,a.words-first))add_cyclic_word(out,period,1);
    }else{
        const auto carry=sbn3i_addmul_1(out+shift,a.data,long(first),high);
        if(shift+first==period)add_cyclic_word(out,period,carry);
        else if(parallel_limbs::add_word(out+shift+first,period-shift-first,carry))add_cyclic_word(out,period,1);
        if(first<a.words){const size_t rest=a.words-first;const auto next=sbn3i_addmul_1(out,a.data+first,long(rest),high);
            if(parallel_limbs::add_word(out+rest,period-rest,next))add_cyclic_word(out,period,1);}
    }
}
// Input contains a full product or its residue modulo B^period-1. The exact
// low product word disambiguates cancellation outside the centered ring.
// The caller proves |difference|<B^(period+1)/2. fold is needed only when a
// normalized (rather than already contiguous high) source crosses the ring.
inline WindowResult product_difference(uint64_t *value,size_t capacity,size_t period,const DifferenceValue &x,
                                        uint64_t exact_product_low,uint64_t *fold=nullptr,sbn3_team *team=nullptr) noexcept {
    require(!period||capacity>period,SBN3_FATAL_ARGUMENT,"difference lift capacity");
    const size_t span=period?period:capacity;
    const uint64_t low=x.word(0)-exact_product_low;
    const uint64_t borrow=subtract_value(team,value,span,x);
    bool negative=borrow!=0;
    if(period){
        if(borrow)limbs::cyclic_sub_power(value,period,0);
        if(x.words()>period){
            const size_t extra=x.words()-period;require(extra<=period,SBN3_FATAL_ARGUMENT,"difference fold support");
            const uint64_t *source;
            if(period>=x.low_words)source=x.high.data+period-x.low_words;
            else{
                require(fold,SBN3_FATAL_ARGUMENT,"difference fold storage");
                parallel_limbs::each(team,extra,parallel_limbs::parts(team,extra),[&](size_t a,size_t b,unsigned){
                    for(size_t j=a;j<b;++j)fold[j]=x.word(period+j);});source=fold;
            }
            if(parallel_limbs::add_to(team,value,period,source,extra))add_cyclic_word(value,period,1);
        }
        if(period<parallel_limbs::minimum_parallel_words){const uint64_t witness[1]={low};negative=reconstruct_cyclic(value,period,witness);}
        else{
            const int64_t wraps=static_cast<int64_t>(value[0]-low);value[period]=uint64_t(wraps);
            if(wraps>=0)parallel_limbs::sub_word(value,period+1,uint64_t(wraps));
            else parallel_limbs::add_word(value,period+1,uint64_t(0)-uint64_t(wraps));
            negative=value[period]>>63;
            if(negative){parallel_limbs::complement(team,value,period+1);parallel_limbs::add_word(value,period+1,1);}
        }
    }else if(negative){parallel_limbs::complement(team,value,capacity);parallel_limbs::add_word(value,capacity,1);}
    return {value,period?period+1:capacity,0,negative};
}
} // namespace sbn::v3::product
