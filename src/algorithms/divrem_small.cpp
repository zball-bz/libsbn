#include "sbn3/divrem.h"
#include "sbn3/detail/division.hpp"
#include "algorithms/small_division.hpp"
#include "algorithms/local_divrem.hpp"
#include "backend/u52/kernels.hpp"
#include "common/checked.hpp"
#include "common/small_checks.h"
#include "product/native_capabilities.hpp"
#include "runtime/scratch.hpp"
#include "value/divrem_words.hpp"
#include <cstring>
using namespace sbn::v3;
namespace {
size_t storage_bytes(size_t nn,size_t dn,bool native) noexcept {
    if(native)if(const auto block=local_division_block(nn,dn))return local_division_auto_plan(nn,dn,block).storage_bytes;
    if(native)return u52::divide_scratch_bytes(nn,dn)+64;
    return nn<dn || dn==1?0:8*(nn+1+(dn>2?dn:0));
}
size_t remainder_bytes(size_t nn,size_t dn,bool native)noexcept{
    if(nn<dn||dn<=2)return 0;
    if(native)if(const auto block=local_division_block(nn,dn))return local_division_remainder_bytes(local_division_auto_plan(nn,dn,block));
    if(native)return u52::remainder_scratch_bytes(nn,dn)+64;
    return 8*(nn+dn+1);
}
}
extern "C" size_t sbn3_remainder_small_scratch_bytes(size_t nn,size_t dn){
    require(dn&&dn<=local_division_max_limbs&&nn<=(size_t(1)<<40),SBN3_FATAL_ARGUMENT,"local remainder size");
    return remainder_bytes(nn,dn,small_division_u52(nn,dn)&&native_available());
}
extern "C" void sbn3_remainder_small(uint64_t*r,const uint64_t*n,size_t nn,const uint64_t*d,size_t dn,void*scratch){
    require(dn&&d&&d[dn-1],SBN3_FATAL_ARGUMENT,"local remainder divisor");
    const bool native=small_division_u52(nn,dn)&&native_available();const auto bytes=remainder_bytes(nn,dn,native);
    if constexpr(SBN3_CHECK_SMALL){
        require(dn<=local_division_max_limbs&&nn<=(size_t(1)<<40),SBN3_FATAL_ARGUMENT,"local remainder size");
        const void*spans[]{r,n,d,scratch};const size_t sizes[]{8*dn,8*nn,8*dn,bytes};
        for(unsigned i=0;i<4;++i){valid_span(spans[i],sizes[i],"local remainder span");for(unsigned j=0;j<i;++j)require(!overlaps(spans[i],sizes[i],spans[j],sizes[j]),SBN3_FATAL_ARGUMENT,"local remainder overlap");}
    }
    if(nn<dn){if(nn)std::memcpy(r,n,nn*8);std::memset(r+nn,0,(dn-nn)*8);return;}
    if(!native){auto*words=static_cast<uint64_t*>(scratch);const auto p=divrem_words::prepare(d,dn,dn<=2?nullptr:words+nn+1);divrem_words::remainder_prepared(r,n,nn,p,words);return;}
    if(const auto block=local_division_block(nn,dn)){const auto plan=local_division_auto_plan(nn,dn,block);auto frame=Frame::external(scratch,bytes);local_division_execute(plan,nullptr,r,n,nn,d,frame);return;}
    auto frame=Frame::external(scratch,bytes);u52::remainder(r,n,nn,d,dn,frame);
}
extern "C" size_t sbn3_divrem_small_scratch_bytes(size_t nn,size_t dn) {
    require(dn && dn<=local_division_max_limbs && nn<=(size_t(1)<<40),SBN3_FATAL_ARGUMENT,"local division size");
    return storage_bytes(nn,dn,small_division_u52(nn,dn)&&native_available());
}
template<bool Checked>static size_t run_small(uint64_t *q,uint64_t *r,const uint64_t *n,size_t nn,
                                    const uint64_t *d,size_t dn,void *scratch,size_t available) {
    require(dn && d && d[dn-1],SBN3_FATAL_ARGUMENT,"small division divisor");
    const bool native=small_division_u52(nn,dn)&&native_available();
    if constexpr(SBN3_CHECK_SMALL){
        require(dn<=local_division_max_limbs && nn<=(size_t(1)<<40),SBN3_FATAL_ARGUMENT,"local division size");
        const size_t qn=nn>=dn?nn-dn+1:0,bytes=storage_bytes(nn,dn,native);
        const void *spans[]={q,r,n,d,scratch};
        const size_t sizes[]={8*qn,8*dn,8*nn,8*dn,bytes};
        for(unsigned i=0;i<5;++i){
            require(!sizes[i] || spans[i],SBN3_FATAL_ARGUMENT,"small division span");
            for(unsigned j=0;j<i;++j)require(!overlaps(spans[i],sizes[i],spans[j],sizes[j]),SBN3_FATAL_ARGUMENT,"small division overlap");
        }
    }
    if(!native){if constexpr(Checked){const size_t need=nn<dn||dn==1?0:8*(nn+1+(dn>2?dn:0));require(available>=need,SBN3_FATAL_WORKSPACE,"checked scalar division scratch",need,available);}
        return divrem_words::schoolbook(q,r,n,nn,d,dn,static_cast<uint64_t *>(scratch));}
    if(const auto block=local_division_block(nn,dn)){
        const auto plan=local_division_auto_plan(nn,dn,block);
        if constexpr(Checked)require(available>=plan.storage_bytes,SBN3_FATAL_WORKSPACE,"checked FFT division scratch",plan.storage_bytes,available);
        auto frame=Frame::external(scratch,plan.storage_bytes);
        local_division_execute(plan,q,r,n,nn,d,frame);
        size_t count=nn-dn+1;while(count&&!q[count-1])--count;return count;
    }
    // The recipe is already known to be DC; do not run the policy a third
    // time merely to recover this fixed kernel's scratch bound.
    const auto bytes=u52::divide_scratch_bytes(nn,dn)+64;
    if constexpr(Checked)require(available>=bytes,SBN3_FATAL_WORKSPACE,"checked native division scratch",bytes,available);
    auto frame=Frame::external(scratch,bytes);
    u52::divide(q,r,n,nn,d,dn,frame);
    size_t count=nn-dn+1;while(count&&!q[count-1])--count;return count;
}
extern "C" size_t sbn3_divrem_small(uint64_t*q,uint64_t*r,const uint64_t*n,size_t nn,const uint64_t*d,size_t dn,void*scratch){
    return run_small<false>(q,r,n,nn,d,dn,scratch,0);
}
sbn::detail::division_sizes sbn::detail::local_divrem_checked(uint64_t*q,uint64_t*r,const uint64_t*n,size_t nn,const uint64_t*d,size_t dn,void*scratch,size_t bytes)noexcept{
    // The public C++ two-word quote needs no caller scratch. Retain that
    // contract while sharing the word reciprocal kernel with larger calls.
    uint64_t small[3];if(nn<=2&&dn==2){scratch=small;bytes=sizeof small;}
    const auto qs=run_small<true>(q,r,n,nn,d,dn,scratch,bytes);size_t rs=dn;while(rs&&!r[rs-1])--rs;return {qs,rs};
}
