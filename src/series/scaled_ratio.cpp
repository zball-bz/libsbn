#include "series/scaled_ratio.hpp"
#include "common/checked.hpp"
#include "value/parallel_limbs.hpp"
#include <cstring>
namespace sbn::v3::series {
namespace {
// Each partition snapshots at most one cache line of cross-partition input
// plus a shift carry word. All snapshots precede all stores. Wider alias
// dependencies retain the existing serial routine; no temporary allocation.
bool parallel_slice(sbn3_team *team,uint64_t *dst,const uint64_t *src,size_t n,
                    bool alias,__int128 q,unsigned bits) noexcept {
    if(!n)return true;
    if(!bits && dst==src)return true;
    const unsigned parts=parallel_limbs::parts(team,n);
    if(parts==1)return false;
    if(!alias){
        parallel_limbs::each(team,n,parts,[&](size_t b,size_t e,unsigned){
            if(!bits){if(e>b)std::memcpy(dst+b,src+b,(e-b)*8);}
            else for(size_t j=b;j<e;++j)dst[j]=(src[j]<<bits)|(*(src+j-1)>>(64-bits));
        });
        return true;
    }
    if(q<-8 || q>8)return false;
    const int delta=-int(q);
    const bool forward=delta>0;
    const unsigned halo=unsigned(forward?delta:-delta+int(bits!=0));
    if(!halo || halo>8 || n/parts<halo)return false;
    alignas(64) uint64_t saved[32][9]{};
    for(unsigned k=0;k<parts;++k){
        const size_t b=parallel_limbs::cut(n,parts,k),e=parallel_limbs::cut(n,parts,k+1);
        const size_t at=forward?e-halo:b;
        const auto *p=src+at;
        if(bits)--p;
        for(unsigned j=0;j<halo+unsigned(bits!=0);++j)saved[k][j]=p[j];
    }
    parallel_limbs::each(team,n,parts,[&](size_t b,size_t e,unsigned k){
        if(forward){
            const size_t end=e-halo;
            if(!bits){
                if(end>b)std::memmove(dst+b,src+b,(end-b)*8);
                std::memcpy(dst+end,saved[k],halo*8);
            }else{
                for(size_t j=b;j<end;++j)dst[j]=(src[j]<<bits)|(*(src+j-1)>>(64-bits));
                for(unsigned j=0;j<halo;++j)dst[end+j]=(saved[k][j+1]<<bits)|(saved[k][j]>>(64-bits));
            }
        }else{
            const size_t begin=b+halo;
            if(!bits){
                if(e>begin)std::memmove(dst+begin,src+begin,(e-begin)*8);
                std::memcpy(dst+b,saved[k],halo*8);
            }else{
                for(size_t j=e;j-->begin;)dst[j]=(src[j]<<bits)|(*(src+j-1)>>(64-bits));
                for(unsigned j=0;j<halo;++j)dst[b+j]=(saved[k][j+1]<<bits)|(saved[k][j]>>(64-bits));
            }
        }
    });
    return true;
}
}
void scaled_slice(uint64_t *out, size_t count, sbn3_int_view in, int64_t shift, sbn3_team *team) noexcept {
    require(count <= size_t(INT64_MAX / 64), SBN3_FATAL_SIZE, "scaled slice length");
    if (!count) return;
    if (!in.size) { std::memset(out, 0, count * 8); return; }
    // shift = 64*q + r, 0 <= r < 64. Only these scalar bounds need
    // signed wide arithmetic; the payload loops have contiguous word indices.
    const unsigned r = unsigned(uint64_t(shift) & 63);
    const __int128 q = (__int128(shift) - r) / 64;
    auto clip = [&](const __int128 j) -> size_t {
        return j < 0 ? 0 : j > count ? count : size_t(j);
    };
    const size_t first = clip(q), last = clip(q + in.size + unsigned(r != 0));
    if (!r) {
        if (last > first && !parallel_slice(team,out+first,in.data+size_t(__int128(first)-q),
                                           last-first,out==in.data,q,0))
            std::memmove(out+first,in.data+size_t(__int128(first)-q),(last-first)*8);
    } else {
        // Read both partial words before writing an aliased interior. Prefix
        // and suffix zeroing must likewise wait until all source reads finish.
        const bool low_live = q >= 0 && q < count;
        const __int128 top = q + in.size;
        const bool high_live = top >= 0 && top < count;
        const uint64_t low = low_live ? in.data[0] << r : 0;
        const uint64_t high = high_live ? in.data[in.size - 1] >> (64 - r) : 0;
        const size_t begin = clip(q + 1), end = clip(q + in.size);
        if (end > begin) {
            auto *dst = out + begin;
            const auto *src = in.data + size_t(__int128(begin) - q);
            const size_t n = end - begin;
            if(!parallel_slice(team,dst,src,n,out==in.data,q,r)){
                if(out==in.data && shift>0)
                    for(size_t j=n;j--;)dst[j]=(src[j]<<r)|(*(src+j-1)>>(64-r));
                else for(size_t j=0;j<n;++j)dst[j]=(src[j]<<r)|(*(src+j-1)>>(64-r));
            }
        }
        if (low_live) out[size_t(q)] = low;
        if (high_live) out[size_t(top)] = high;
    }
    parallel_limbs::fill(team,out,first);
    parallel_limbs::fill(team,out+last,count-last);
}
void ratio_inputs(ScaledMagnitude numerator, ScaledMagnitude denominator, size_t fractional, uint64_t *A,
                  uint64_t *D, sbn3_team *team) noexcept {
    const auto d = denominator.value;
    require(fractional >= 2 && fractional <= (size_t(1) << 29) && d.size && d.data[d.size - 1] && !d.negative,
            SBN3_FATAL_ARGUMENT, "scaled ratio domain");
    const size_t n = fractional + 2;
    const __int128 dbits = __int128(d.size) * 64 - __builtin_clzll(d.data[d.size - 1]);
    const __int128 common = __int128(n) * 64 - dbits - denominator.exponent2;
    const __int128 ds = common + denominator.exponent2, as = common + numerator.exponent2 - 64;
    require(ds >= INT64_MIN && ds <= INT64_MAX && as >= INT64_MIN && as <= INT64_MAX, SBN3_FATAL_SIZE,
            "scaled ratio shift");
    if (numerator.value.size) {
        const auto a = numerator.value;
        const __int128 abits =
            __int128(a.size) * 64 - __builtin_clzll(a.data[a.size - 1]) + numerator.exponent2;
        require(abits - (dbits + denominator.exponent2) <= 63, SBN3_FATAL_MATH,
                "scaled ratio magnitude certificate");
    }
    scaled_slice(D, n, d, int64_t(ds), team);
    scaled_slice(A, n + 1, numerator.value, int64_t(as), team);
    require((D[n - 1] >> 63) && A[n] == 0, SBN3_FATAL_MATH, "scaled ratio normalization");
}
sbn3_const_limbs ratio_result(const uint64_t *quotient, size_t fractional) noexcept {
    require(quotient[fractional + 2] == 0, SBN3_FATAL_MATH, "scaled ratio quotient width");
    return {quotient + 1, fractional + 1};
}
void guard_separated(const uint64_t *value, size_t dropped, uint64_t error, bool allow_lower_boundary) noexcept {
    require(dropped >= 1 && error < UINT64_MAX / 2, SBN3_FATAL_ARGUMENT, "constant guard parameters");
    bool low = true, high = true;
    for (size_t j = 1; j < dropped; ++j) {
        low &= value[j] == 0;
        high &= value[j] == UINT64_MAX;
    }
    require((allow_lower_boundary || !(low && value[0] < error)) && !(high && value[0] > UINT64_MAX - error), SBN3_FATAL_MATH,
            "constant output guard separation");
}
// Retain the established serial entry and its call-site ABI.
void scaled_slice(uint64_t *out,size_t count,sbn3_int_view in,int64_t shift) noexcept {
    scaled_slice(out,count,in,shift,nullptr);
}
} // namespace sbn::v3::series
