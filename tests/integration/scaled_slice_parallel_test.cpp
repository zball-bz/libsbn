#include "product_support.hpp"
#include "series/scaled_ratio.hpp"
#include "value/parallel_limbs.hpp"
#include <algorithm>
#include <cstdint>
using namespace sbn::v3;
static uint64_t word(const std::vector<uint64_t> &in,size_t used,int64_t index){
    return index<0 || uint64_t(index)>=used?0:in[size_t(index)];
}
static std::vector<uint64_t> reference(const std::vector<uint64_t> &in,size_t used,size_t count,int64_t shift){
    std::vector<uint64_t> out(count);
    const int64_t bits=-shift;int64_t q=bits/64;if(bits%64<0)--q;
    const unsigned r=unsigned(bits-q*64);
    for(size_t j=0;j<count;++j){const int64_t at=q+int64_t(j);
        out[j]=word(in,used,at)>>r;if(r)out[j]|=word(in,used,at+1)<<(64-r);
    }
    return out;
}
static void run(size_t base,unsigned workers,bool large){
    Fixture f(workers,true);const size_t capacity=base+64;
    auto *input=f.guarded(capacity),*output=f.guarded(capacity),*readonly=f.guarded(base+11);
    std::vector<uint64_t> seed(capacity);
    for(auto &x:seed)x=random_word();
    size_t cases=0;
    const int64_t shifts[]{-100000,-577,-512,-511,-385,-64,-1,0,1,63,64,449,512,513,100000};
    for(int64_t shift:shifts)for(bool alias:{false,true}){
        const size_t used=base+11,count=large?(shift>=0?base+19:base-7):base;
        std::copy(seed.begin(),seed.end(),input);std::copy_n(seed.data(),used,readonly);std::fill(output,output+capacity,UINT64_C(0x6db235891ce74af0));
        auto wanted=reference(seed,used,count,shift);
        auto *dst=alias?input:output;
        allocation_watch_start();series::scaled_slice(dst,count,{alias?input:readonly,used,0},shift,f.team);assert(!allocation_watch_stop());
        if(count)assert(!memcmp(dst,wanted.data(),count*8));
        for(size_t j=count;j<capacity;++j)assert(dst[j]==(alias?seed[j]:UINT64_C(0x6db235891ce74af0)));
        if(!alias)assert(!memcmp(readonly,seed.data(),used*8));++cases;
    }
    if(!large){
        std::fill(output,output+capacity,UINT64_MAX);
        series::scaled_slice(output,base,{nullptr,0,0},0,f.team);
        for(size_t j=0;j<base;++j)assert(!output[j]);
    }
    printf("scaled slice W%u base%zu: %zu shifts/aliases, independent bit oracle, untouched tails and no allocation PASS\n",workers,base,cases);
}
int main(){
    for(size_t n:{0u,1u,7u,17u,63u})run(n,3,false);
    for(unsigned w:{3u,16u,32u})run(parallel_limbs::minimum_parallel_words+65,w,true);
}
