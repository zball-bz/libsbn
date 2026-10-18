// Original libsbn code; LGPL-3.0-or-later.
#include "float/short_division.hpp"
#include "algorithms/local_divrem.hpp"
#include "algorithms/local_inverse.hpp"
#include "algorithms/normalized_local.hpp"
#include "algorithms/normalized_newton.hpp"
#include "product/program.hpp"
#include "product/difference.hpp"
#include "value/divrem_words.hpp" // Retains the word-kernel donor notice.
#include "value/limbs.hpp"
#include "runtime/scratch.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e3353444956);
struct State {
    std::uint64_t marker=0;
    ShortDivisionInfo info{};
    std::size_t extent=0,nn=0,phase_bytes=0,product_capacity=0;
    LocalDivisionPlan local_plan{};
    NormalizedLocalPrepared local_inverse{};
    NormalizedNewtonPrepared inverse{};
    ProductProgram product{};
};
struct Recipe {
    ShortDivisionInfo info{};
    std::size_t nn=0,phase_bytes=0,product_capacity=0,table_bytes=0;
    LocalDivisionPlan local_plan{};
    NormalizedLocalInfo local_inverse{};
    NormalizedNewtonInfo inverse{};
    ProductProgramPlan product{};
};
bool append(std::size_t& cursor,std::size_t bytes,std::size_t alignment) noexcept {
    std::size_t at=0;return align_size(cursor,alignment,at)&&add_size(at,bytes,cursor);
}
bool words(std::size_t& cursor,std::size_t count,std::size_t alignment) noexcept {
    std::size_t bytes=0;return mul_size(count,8,bytes)&&append(cursor,bytes,alignment);
}
sbn3_query_result quote(sbn3_newton_kind kind,std::size_t n,std::size_t k,unsigned workers,Recipe& r) noexcept {
    if((kind!=SBN3_NEWTON_DIVIDE&&kind!=SBN3_NEWTON_INVERSE)||k<3||k>=n/2||
       !workers||workers>32||n>(std::size_t(1)<<34)+1||
       !add_size(n,k,r.nn)||!add_size(r.nn,1,r.nn))return SBN3_UNSUPPORTED;
    auto& info=r.info;info.kind=kind;info.n=n;info.k=k;info.workers=workers;info.local=k<=local_division_max_limbs;
    std::size_t metadata=sizeof(State),work=0;
    if(info.local){
        const auto m=std::min(k,local_inverse_max_limbs);
        r.local_plan=local_division_plan_for_block(r.nn,k,m);
        const auto rc=normalized_local_query(NormalizedLocalKind::inverse,m,r.local_inverse);
        if(rc!=SBN3_SUPPORTED)return rc;
        info.alignment=std::max(info.alignment,r.local_inverse.alignment);
        if(!append(metadata,r.local_inverse.metadata_bytes,r.local_inverse.alignment))return SBN3_QUERY_CAPACITY;
        // The compiled inverse can need more phase scratch than the older
        // dynamic seed. D/U persist while that phase runs; it dies before
        // divisor spectra and the quotient-block work are materialized.
        std::size_t seed=0;
        if(!words(seed,k,64)||!words(seed,m+1,64)||
           !append(seed,r.local_inverse.workspace_bytes,r.local_inverse.alignment)||
           !add_size(seed,256,seed))return SBN3_QUERY_CAPACITY;
        r.phase_bytes=std::max(r.local_plan.storage_bytes,seed);
    }else{
        auto rc=normalized_newton_query(SBN3_NEWTON_INVERSE,k,workers,r.inverse);
        if(rc!=SBN3_SUPPORTED)return rc;
        sbn3_mul_options options{};options.workers=workers;options.borrow_output=1;
        rc=product_program_query(k,k,options,r.product);if(rc!=SBN3_SUPPORTED)return rc;
        info.alignment=std::max({info.alignment,r.inverse.alignment,r.product.info.workspace_alignment,r.product.info.output_alignment});
        if(!add_size(r.product.prepared_bytes,128,r.table_bytes)||
           !append(metadata,r.inverse.metadata_bytes,r.inverse.alignment)||
           !append(metadata,r.table_bytes,128))return SBN3_QUERY_CAPACITY;
        r.product_capacity=std::max(2*k+2,sbn3_mul_output_capacity(&r.product.info));
        r.phase_bytes=std::max(r.inverse.workspace_bytes,r.product.info.workspace_bytes);
    }
    // N and R are common. Local replay first uses R to normalize D, then
    // reuses it as the remainder after prepare has copied the divisor.
    if(!words(work,r.nn,info.alignment)||!words(work,k,info.alignment))return SBN3_QUERY_CAPACITY;
    if(!info.local&&(!words(work,k,info.alignment)||!words(work,k+1,info.alignment)||
                    !words(work,r.product_capacity,info.alignment)||!words(work,k,info.alignment)))return SBN3_QUERY_CAPACITY;
    if(!append(work,r.phase_bytes,info.alignment)||!add_size(metadata,info.alignment-1,info.metadata_bytes))return SBN3_QUERY_CAPACITY;
    info.workspace_bytes=work;
    std::size_t total=0;if(!add_size(info.metadata_bytes,work,total)||!add_size(total,info.alignment-1,info.storage_bytes))return SBN3_QUERY_CAPACITY;
    return SBN3_SUPPORTED;
}
void normalize(std::uint64_t* out,sbn3_const_limbs raw) noexcept {
    const unsigned shift=unsigned(__builtin_clzll(raw.data[raw.count-1]));
    if(shift)require(!divrem_words::shift_left(out,raw.data,raw.count,shift),SBN3_FATAL_MATH,"short divisor normalization");
    else std::memcpy(out,raw.data,raw.count*8);
}
std::uint64_t shifted_word(sbn3_const_limbs input,std::size_t i,std::int64_t shift) noexcept {
    const __int128 bit=__int128(64)*i-shift;
    if(bit<0)return bit<=-64?0:input.data[0]<<unsigned(-bit);
    const __int128 index=bit/64;const unsigned part=unsigned(bit%64);
    if(index>=input.count)return 0;
    const auto j=std::size_t(index);
    return (input.data[j]>>part)|(part&&j+1<input.count?input.data[j+1]<<(64-part):0);
}
void native_blocks(const State& state,std::uint64_t* Q,std::uint64_t* R,const std::uint64_t* N,
                    const std::uint64_t* D,std::uint64_t* U,std::uint64_t* V,std::uint64_t* small,
                    Frame& phase,sbn3_team_scope* scope) noexcept {
    const auto k=state.info.k,nn=state.nn,qn=state.info.n+2;
    normalized_newton_execute(state.inverse,{}, {D,k},{U,k+1},phase,scope);
    require(U[k]==1,SBN3_FATAL_MATH,"short normalized inverse leading word");
    for(std::size_t j=0;j<k;++j)R[j]=qn+j<nn?N[qn+j]:0;
    std::size_t pos=qn;
    while(pos){
        const auto count=std::min(k,pos);pos-=count;
        std::copy_n(R+k-count,count,small);std::fill(small+count,small+k,0);
        product_program_execute(state.product,phase,scope,{U,k},{small,k},{V,state.product_capacity});
        auto* block=Q+pos;
        if(sbn3i_add_n(block,V+k,R+k-count,long(count)))std::fill_n(block,count,UINT64_MAX);
        std::copy_n(block,count,small);std::fill(small+count,small+k,0);
        product_program_execute(state.product,phase,scope,{D,k},{small,k},{V,state.product_capacity});
        const auto length=k+count+1;V[k+count]=0;
        const product::DifferenceValue input{{N,nn},pos,count,0,{R,k}};
        const bool negative=product::product_difference(V,length,0,input,block[0]*D[0]).negative;
        unsigned corrections=0;
        if(negative){
            while(V[k]||divrem_words::compare(V,D,k)>0){
                require(++corrections<=8,SBN3_FATAL_MATH,"short negative correction");limbs::sub_from(V,length,D,k);
            }
            sbn3i_sub_n(V,D,V,long(k));++corrections;
            const std::uint64_t c=corrections;
            require(!limbs::sub_from(block,count,&c,1),SBN3_FATAL_MATH,"short quotient underflow");
        }else{
            while(V[k]||divrem_words::compare(V,D,k)>=0){
                require(++corrections<=8,SBN3_FATAL_MATH,"short positive correction");limbs::sub_from(V,length,D,k);
            }
            const std::uint64_t c=corrections;
            require(!limbs::add_to(block,count,&c,1),SBN3_FATAL_MATH,"short quotient overflow");
        }
        std::memcpy(R,V,k*8);
    }
}
}
sbn3_query_result short_division_query(sbn3_newton_kind kind,std::size_t n,std::size_t k,
                                      unsigned workers,ShortDivisionInfo& info) noexcept {
    Recipe r;const auto rc=quote(kind,n,k,workers,r);info=rc==SBN3_SUPPORTED?r.info:ShortDivisionInfo{};return rc;
}
ShortDivisionPrepared short_division_prepare(sbn3_newton_kind kind,std::size_t n,std::size_t k,
                                             unsigned workers,Frame& metadata) noexcept {
    Recipe r;require(quote(kind,n,k,workers,r)==SBN3_SUPPORTED,SBN3_FATAL_ARGUMENT,"short division shape");
    auto* state=::new(metadata.allocate(sizeof(State),128))State{};
    state->info=r.info;state->nn=r.nn;state->phase_bytes=r.phase_bytes;state->product_capacity=r.product_capacity;
    if(r.info.local){
        state->local_plan=r.local_plan;
        state->local_inverse=normalized_local_prepare(NormalizedLocalKind::inverse,r.local_plan.block_limbs,metadata);
    }else{
        state->inverse=normalized_newton_prepare(SBN3_NEWTON_INVERSE,k,workers,metadata);
        auto tables=metadata.subframe(r.table_bytes,128);
        state->product=product_program_prepare(r.product,tables);
    }
    state->extent=std::size_t(metadata.data()+metadata.used()-reinterpret_cast<std::uint8_t*>(state));
    require(state->extent<=r.info.metadata_bytes,SBN3_FATAL_WORKSPACE,"short division metadata quote");
    state->marker=magic;return {state,state->info};
}
void short_division_execute(const ShortDivisionPrepared& prepared,sbn3_const_limbs A,sbn3_const_limbs D,
                             sbn3_limbs Q,Frame& work,sbn3_team_scope* scope,std::int64_t a_shift) noexcept {
    require_scope_leader(scope);require(prepared.state,SBN3_FATAL_LIFETIME,"short division prepared state");
    const auto& state=*static_cast<const State*>(prepared.state);const auto& info=state.info;
    require(state.marker==magic&&prepared.info.n==info.n&&prepared.info.k==info.k&&prepared.info.kind==info.kind&&
            prepared.info.workspace_bytes==info.workspace_bytes,SBN3_FATAL_LIFETIME,"short division prepared identity");
    require(scope->width>=info.workers,SBN3_FATAL_TEAM,"short division scope width");
    const bool divide=info.kind==SBN3_NEWTON_DIVIDE;
    require(D.data&&D.count==info.k&&D.data[info.k-1]&&Q.data&&Q.capacity>=info.n+2,
            SBN3_FATAL_ARGUMENT,"short division values");
    require(divide?(A.data&&A.count&&A.count<=(std::size_t(1)<<34)&&A.data[A.count-1]&&
                    __int128(64)*(A.count-1)+std::bit_width(A.data[A.count-1])+a_shift==__int128(64)*info.n):(!A.data&&!A.count),
            SBN3_FATAL_ARGUMENT,"short division numerator");
    const auto available=work.capacity()-work.used();
    auto* begin=work.data()+work.used();
    require(available>=info.workspace_bytes&&!(std::uintptr_t(begin)&(info.alignment-1)),
            SBN3_FATAL_WORKSPACE,"short division work",info.workspace_bytes,available);
    require(!overlaps(&state,state.extent,begin,available),SBN3_FATAL_ARGUMENT,"short division metadata/work alias");
    const sbn3_const_limbs values[]{A,D,{Q.data,info.n+2}};
    for(unsigned i=0;i<3;++i)if(values[i].count){
        const auto v=values[i];valid_span(v.data,v.count*8,"short division span");
        require(!(std::uintptr_t(v.data)&7)&&!overlaps(v.data,v.count*8,&state,state.extent)&&
                !overlaps(v.data,v.count*8,begin,available),SBN3_FATAL_ARGUMENT,"short division resource alias");
        if(i==2)for(unsigned j=0;j<2;++j)require(!overlaps(v.data,v.count*8,values[j].data,values[j].count*8),
                                                SBN3_FATAL_ARGUMENT,"short division output alias");
    }
    FrameMark lifetime(work);
    auto bank=[&](std::size_t count){return static_cast<std::uint64_t*>(work.allocate(count*8,info.alignment));};
    auto* N=bank(state.nn);auto* R=bank(info.k);
    std::fill_n(N,state.nn,0);
    if(divide){
        for(std::size_t i=0;i<info.n;++i)N[info.k+i]=shifted_word(A,i,a_shift);
        require(N[info.k+info.n-1]>>63,SBN3_FATAL_MATH,"short staged numerator normalization");
    }else N[info.n+info.k]=1;
    if(info.local){
        normalize(R,D);
        auto phase=work.subframe(state.phase_bytes,info.alignment);
        const auto divisor=local_division_prepare_compiled(state.local_plan,R,phase,state.local_inverse);
        local_division_apply(state.local_plan,divisor,Q.data,R,N,state.nn,phase);
    }else{
        auto* divisor=bank(info.k);auto* U=bank(info.k+1);auto* V=bank(state.product_capacity);auto* small=bank(info.k);
        normalize(divisor,D);auto phase=work.subframe(state.phase_bytes,info.alignment);
        native_blocks(state,Q.data,R,N,divisor,U,V,small,phase,scope);
    }
    require(!Q.data[info.n+1]&&Q.data[info.n]<=2,SBN3_FATAL_MATH,"short quotient normalization");
}
}
