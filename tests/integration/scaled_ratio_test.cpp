#include "series/scaled_ratio.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <vector>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::v3::series::scaled_slice;
static uint64_t random_state=0x8421b9ce6132ULL;
static uint64_t random_word(){random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return random_state;}
// Independent bit-position lookup, retained from the original implementation.
static uint64_t reference(const uint64_t *a,size_t n,size_t j,int64_t shift){
    const __int128 bit=__int128(j)*64-shift;
    if(bit<0)return bit>-64&&n?a[0]<<unsigned(-bit):0;
    if(bit/64>=n)return 0;
    const size_t word=size_t(bit/64);const unsigned rem=unsigned(bit%64);
    return rem?(a[word]>>rem)|((word+1<n?a[word+1]:0)<<(64-rem)):a[word];
}
static uint64_t cases=0,words=0;
static void one(size_t n,size_t outn,int64_t shift,unsigned pattern){
    std::vector<uint64_t> source(n),dest(outn+2,0x9c4a77ed2bULL),alias(std::max(n,outn)+2,0x9c4a77ed2bULL);
    for(size_t j=0;j<n;++j)source[j]=pattern==0?random_word():pattern==1?UINT64_MAX:pattern==2?uint64_t(1)<<(j%64):0;
    std::copy(source.begin(),source.end(),alias.begin()+1);
    allocation_watch_start();
    scaled_slice(dest.data()+1,outn,{source.data(),n,0},shift);
    scaled_slice(alias.data()+1,outn,{alias.data()+1,n,0},shift);
    assert(!allocation_watch_stop());
    for(size_t j=0;j<outn;++j){const auto want=reference(source.data(),n,j,shift);assert(dest[j+1]==want&&alias[j+1]==want);}
    assert(dest.front()==0x9c4a77ed2bULL&&dest.back()==0x9c4a77ed2bULL);
    assert(alias.front()==0x9c4a77ed2bULL&&alias.back()==0x9c4a77ed2bULL);
    for(size_t j=outn;j<n;++j)assert(alias[j+1]==source[j]);
    ++cases;words+=outn;
}
int main(){
    for(size_t n=0;n<=20;++n)for(size_t out=0;out<=20;++out){
        for(int64_t shift=-130;shift<=130;++shift)one(n,out,shift,unsigned((shift+130)%4));
        for(int64_t shift:{INT64_MIN,INT64_MIN+1,INT64_MAX,INT64_MAX-1,int64_t(-4096),int64_t(4096)})one(n,out,shift,0);
    }
    for(size_t n:{size_t(257),size_t(4095),size_t((1<<20)+3)})
        for(int64_t shift:{int64_t(-2049),int64_t(-65),int64_t(-64),int64_t(-1),int64_t(0),int64_t(1),int64_t(64),int64_t(65),int64_t(2049)})
            for(size_t out:{n-3,n,n+7})one(n,out,shift,0);
    printf("scaled slice: %llu shapes, %llu output words, disjoint/exact alias, extreme shifts, zero/truncation/padding and allocation gates PASS\n",(unsigned long long)cases,(unsigned long long)words);
}
