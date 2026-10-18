// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/checksum.hpp"
#include "common/m64.hpp"
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <vector>
// A separate scalar one-shot reference hashes an explicitly concatenated
// string, independently of the production segmented AVX-512 traversal.
#define XXH_INLINE_ALL
#define XXH_NO_STDLIB
#define XXH_VECTOR XXH_SCALAR
#include "../../third_party/xxhash/xxhash.h"
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using namespace sbn::v3::external;
static uint64_t reference(RegionId id,uint64_t k,const uint8_t*p,size_t n,std::vector<uint8_t>&joined){
    joined.resize(48+n);auto*h=joined.data();put64(h,0x314b48434e4253ULL);put64(h+8,id.uuid_low);put64(h+16,id.uuid_high);
    put64(h+24,id.incarnation);put64(h+32,k);put64(h+40,n);if(n)std::memcpy(h+48,p,n);
    return XXH3_64bits(h,48+n);
}
int main(){
    for(size_t n=0;n<1025;++n)for(unsigned pattern=0;pattern<3;++pattern){
        std::vector<uint8_t> data(n*8+1);uint64_t sum=0,rng=29;
        for(size_t j=0;j<n;++j){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
            const uint64_t x=pattern==0?rng:pattern==1?UINT64_MAX:UINT64_MAX-j;
            std::memcpy(data.data()+1+8*j,&x,8);sum=sbn::v3::m64::reduce(__uint128_t(sum)+x);}
        assert(sbn::v3::m64::words(data.data()+1,n)==sum&&sbn::v3::m64::words_scalar(data.data()+1,n)==sum);
    }
    constexpr size_t maximum=1u<<20;std::vector<uint8_t>input(maximum+64),joined;uint64_t rng=173;
    for(auto&x:input){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;x=uint8_t(rng);}
    size_t cases=0;
    for(size_t n=0;n<=17003;++n)for(unsigned off:{0u,1u,15u,31u,63u}){
        const RegionId id{UINT64_MAX-n,47+off,1+n%13};const uint64_t k=n*97+off;
        const auto expected=reference(id,k,input.data()+off,n,joined);
        allocation_watch_start();const auto got=chunk_checksum(id,k,{input.data()+off,n});assert(!allocation_watch_stop());
        assert(got==expected);assert(metadata_checksum({input.data()+off,n})==XXH3_64bits(input.data()+off,n));++cases;
    }
    for(size_t n:{size_t(32767),size_t(32768),size_t(65537),size_t(100000),maximum}){
        assert(chunk_checksum({1,2,3},UINT64_MAX,{input.data(),n})==reference({1,2,3},UINT64_MAX,input.data(),n,joined));++cases;
    }
    auto*map=static_cast<uint8_t*>(mmap(nullptr,maximum+8192,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));assert(map!=MAP_FAILED);
    assert(!mprotect(map,4096,PROT_NONE)&&!mprotect(map+4096+maximum,4096,PROT_NONE));
    // Both beginning and end guards; all stripe and short/long transition cuts.
    for(size_t n:{size_t(0),size_t(1),size_t(191),size_t(192),size_t(193),size_t(975),size_t(976),size_t(977),size_t(16368),size_t(16383),size_t(16384),size_t(16385),maximum})for(bool end:{false,true}){
        uint8_t*p=end?map+4096+maximum-n:map+4096;if(n)std::memcpy(p,input.data(),n);
        assert(chunk_checksum({31,47,3},17,{p,n})==reference({31,47,3},17,p,n,joined));++cases;
    }
    assert(!munmap(map,maximum+8192));
    assert(chunk_checksum({1,2,3},4,{})==reference({1,2,3},4,nullptr,0,joined));
    printf("Checksum: %zu scalar concat comparisons; all lengths 0..17003, unaligned/tail/guard/large cases, allocation-free PASS\n",cases);
}
