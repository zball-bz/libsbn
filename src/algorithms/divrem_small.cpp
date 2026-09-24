#include "sbn3/divrem.h"
#include "algorithms/small_division.hpp"
#include "algorithms/local_divrem.hpp"
#include "backend/u52/kernels.hpp"
#include "common/checked.hpp"
#include "common/small_checks.h"
#include "product/native_capabilities.hpp"
#include "runtime/scratch.hpp"
#include "value/divrem_words.hpp"
using namespace sbn::v3;
namespace {
size_t storage_bytes(size_t nn,size_t dn,bool native) noexcept {
    if(native)if(const auto block=local_division_block(nn,dn))return local_division_auto_plan(nn,dn,block).storage_bytes;
    if(native)return u52::divide_scratch_bytes(nn,dn)+64;
    return nn<dn || dn==1?0:8*(nn+1+(dn>2?dn:0));
}
}
extern "C" size_t sbn3_divrem_small_scratch_bytes(size_t nn,size_t dn) {
    require(dn && dn<=local_division_max_limbs && nn<=(size_t(1)<<40),SBN3_FATAL_ARGUMENT,"local division size");
    return storage_bytes(nn,dn,small_division_u52(nn,dn)&&native_available());
}
extern "C" size_t sbn3_divrem_small(uint64_t *q,uint64_t *r,const uint64_t *n,size_t nn,
                                    const uint64_t *d,size_t dn,void *scratch) {
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
    if(!native)return divrem_words::schoolbook(q,r,n,nn,d,dn,static_cast<uint64_t *>(scratch));
    if(const auto block=local_division_block(nn,dn)){
        const auto plan=local_division_auto_plan(nn,dn,block);
        auto frame=Frame::external(scratch,plan.storage_bytes);
        local_division_execute(plan,q,r,n,nn,d,frame);
        size_t count=nn-dn+1;while(count&&!q[count-1])--count;return count;
    }
    // The recipe is already known to be DC; do not run the policy a third
    // time merely to recover this fixed kernel's scratch bound.
    auto frame=Frame::external(scratch,u52::divide_scratch_bytes(nn,dn)+64);
    u52::divide(q,r,n,nn,d,dn,frame);
    size_t count=nn-dn+1;while(count&&!q[count-1])--count;return count;
}
