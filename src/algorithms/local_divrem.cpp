#include "algorithms/local_divrem.hpp"
#include "algorithms/local_inverse.hpp"
#include "algorithms/newton_contract.hpp"
#include "product/difference.hpp"
#include "product/local_product.hpp"
#include "runtime/scratch.hpp"
#include "value/divrem_words.hpp"
#include "value/limbs.hpp"
#include <algorithm>
#include <string.h>
namespace sbn::v3 {
namespace {
constexpr size_t align(size_t n) noexcept {return (n+63)&~size_t(63);}
uint64_t shifted(const uint64_t*n,size_t nn,size_t at,unsigned shift) noexcept {
    uint64_t out=at<nn?n[at]<<shift:0;
    if(shift&&at&&at-1<nn)out|=n[at-1]>>(64-shift);
    return out;
}
uint64_t head_word(uint64_t*x,const uint64_t*d,size_t dn,uint64_t inverse,uint64_t&corrections) noexcept {
    const uint64_t d1=d[dn-1],d0=d[dn-2];uint64_t q=UINT64_MAX;
    if(x[dn]!=d1||x[dn-1]!=d0){
        uint64_t r1,r0;
        udiv_qr_3by2(q,r1,r0,x[dn],x[dn-1],x[dn-2],d1,d0,inverse);
        (void)r1;(void)r0;
    }
    if(!q)return 0; // A zero leading quotient already leaves X<D.
    const auto carry=sbn3i_submul_1(x,d,long(dn),q);
    unsigned borrow=x[dn]<carry;x[dn]-=carry;
    while(borrow){
        require(++corrections<=2,SBN3_FATAL_MATH,"local head correction");--q;
        borrow-=unsigned(limbs::add_to(x,dn+1,d,dn));
    }
    require(!x[dn],SBN3_FATAL_MATH,"local head remainder");return q;
}
}
LocalDivisionPlan local_division_plan(size_t nn,size_t dn,unsigned pieces) noexcept {
    require(dn>=3&&dn<=local_division_max_limbs&&nn>=dn&&nn<=(size_t(1)<<40)&&pieces>=1&&pieces<=3,
            SBN3_FATAL_ARGUMENT,"local division geometry");
    const size_t words=nn>dn?std::min(nn-dn,dn):1;
    return local_division_plan_for_block(nn,dn,std::min<size_t>(local_inverse_max_limbs,(words+pieces-1)/pieces));
}
LocalDivisionPlan local_division_plan_for_block(size_t nn,size_t dn,size_t block_limbs,unsigned reuse_hint) noexcept {
    require(dn>=3&&dn<=local_division_max_limbs&&nn>=dn&&nn<=(size_t(1)<<40)&&
            block_limbs&&block_limbs<=dn&&block_limbs<=local_inverse_max_limbs,
            SBN3_FATAL_ARGUMENT,"local division block geometry");
    LocalDivisionPlan p;p.numerator_limbs=nn;p.denominator_limbs=dn;
    p.block_limbs=block_limbs;
    const size_t m=p.block_limbs;
    const size_t qn=nn-dn+1,head=local_division_head(qn,m);
    const bool keep=reuse_hint>1||qn-head>m;
    p.quotient=product::local_product_query(m,m,false,keep);p.residual=product::local_product_query(dn,m,true,keep);
    // The two spectra are different values, but equal transform geometries
    // use one immutable table allocation for the lifetime of this call.
    if(local_division_shares_tables(p))p.residual.persistent_bytes-=product::local_product_table_bytes(p.residual);
    const size_t divisor=align(dn*8)+align((m+1)*8);
    p.persistent_bytes=align(divisor+p.quotient.persistent_bytes+p.residual.persistent_bytes+512);
    p.work_bytes=align((dn+m+2)*8)+
                 std::max(p.quotient.work_bytes,p.residual.work_bytes)+512;
    // Inverse scratch dies before the spectra are built; reserve a phase
    // maximum rather than adding it to retained spectra and execution work.
    p.storage_bytes=std::max(p.persistent_bytes+p.work_bytes,
                            divisor+local_inverse_approximate_bytes(m)+256)+128;
    return p;
}
LocalDivisionPlan local_fused_division_plan(size_t nn,size_t dn,unsigned reuse_hint) noexcept {
    require(nn>dn&&dn<=local_division_max_limbs&&nn-dn+3<=dn&&nn-dn+3<=local_divide_max_limbs,
            SBN3_FATAL_ARGUMENT,"local fused exact geometry");
    LocalDivisionPlan p;p.numerator_limbs=nn;p.denominator_limbs=dn;p.fused_precision=nn-dn+3;
    const size_t qn=nn-dn+1,m=newton_contract::next_precision(p.fused_precision);p.block_limbs=m;
    // U is prepared as an integer; terminal scratch owns any temporary
    // representation. Only the complete divisor product can be retained.
    p.quotient.an=p.quotient.bn=m;p.residual=product::local_product_query(dn,qn-1,true,reuse_hint>1);
    const size_t divisor=align(dn*8)+align((m+1)*8);
    p.persistent_bytes=align(divisor+p.residual.persistent_bytes+512);
    p.work_bytes=align((p.fused_precision+1)*8)+512+
        std::max(local_divide_terminal_bytes(p.fused_precision),align((dn+qn+1)*8)+p.residual.work_bytes);
    p.storage_bytes=std::max(p.persistent_bytes+p.work_bytes,divisor+local_inverse_approximate_bytes(m)+256)+128;
    return p;
}
LocalDivisionDivisor local_division_prepare(const LocalDivisionPlan&p,const uint64_t*d,Frame&space) noexcept {
    const size_t dn=p.denominator_limbs,m=p.block_limbs;
    require(d&&d[dn-1],SBN3_FATAL_ARGUMENT,"local division divisor");
    const size_t begin=space.used();
    auto *D=space.alloc<uint64_t>(dn),*U=space.alloc<uint64_t>(m+1);
    LocalDivisionDivisor v;v.divisor=D;v.shift=unsigned(__builtin_clzll(d[dn-1]));
    if(v.shift)divrem_words::shift_left(D,d,dn,v.shift);else memcpy(D,d,dn*8);
    local_inverse_approximate(U,D+dn-m,m,space);
    v.quotient=product::local_product_prepare(p.quotient,U,space);
    v.residual=product::local_product_prepare(p.residual,D,space,local_division_shares_tables(p)?v.quotient.tables:nullptr);
    v.head_inverse=divrem_words::invert_pi1(D[dn-1],D[dn-2]);
    require(space.used()-begin<=p.persistent_bytes,SBN3_FATAL_WORKSPACE,"local retained divisor bound");
    return v;
}
namespace {
uint64_t fused_apply(const LocalDivisionPlan&p,const LocalDivisionDivisor &prepared,
                    uint64_t*q,uint64_t*r,const uint64_t*n,size_t nn,Frame&space,
                    LocalDivisionMetrics *metrics) noexcept {
    FrameMark mark(space);const size_t dn=p.denominator_limbs,t=p.fused_precision,qn=nn-dn+1;
    const size_t planned_qn=p.numerator_limbs-dn+1;const unsigned shift=prepared.shift;
    const auto *D=prepared.divisor;auto *estimate=space.alloc<uint64_t>(t+1);
    for(size_t j=0;j<=t;++j)estimate[j]=shifted(n,nn,dn+j,shift);
    local_divide_terminal(estimate,estimate,D+dn-t,prepared.quotient.original,t,space);
    bool overflow=false;for(size_t j=qn;j<=t;++j)overflow|=estimate[j]!=0;
    if(overflow){memset(estimate,0xff,qn*8);memset(estimate+qn,0,(t+1-qn)*8);}
    const size_t ring=p.residual.ring,L=ring?ring+1:dn+planned_qn+1;
    auto *value=space.alloc<uint64_t>(dn+planned_qn+1);
    product::local_product_apply(p.residual,prepared.residual,value,estimate,planned_qn-1,space);
    if(!ring)value[L-2]=value[L-1]=0;
    product::add_shifted_word_product(value,L,ring,{D,dn},planned_qn-1,estimate[planned_qn-1]);
    const product::DifferenceValue input{{n,nn},0,nn+1,shift,{}};
    const auto difference=product::product_difference(value,L,ring,input,estimate[0]*D[0],r);
    const bool negative=difference.negative;
    unsigned corrections=0;
    if(negative){
        while(value[dn]||divrem_words::compare(value,D,dn)>0){require(++corrections<=8,SBN3_FATAL_MATH,"fused exact negative correction");limbs::sub_from(value,L,D,dn);}
        sbn3i_sub_n(value,D,value,long(dn));++corrections;const uint64_t c=corrections;
        require(!limbs::sub_from(estimate,qn,&c,1),SBN3_FATAL_MATH,"fused exact quotient underflow");
    }else{
        while(value[dn]||divrem_words::compare(value,D,dn)>=0){require(++corrections<=8,SBN3_FATAL_MATH,"fused exact positive correction");limbs::sub_from(value,L,D,dn);}
        const uint64_t c=corrections;require(!limbs::add_to(estimate,qn,&c,1),SBN3_FATAL_MATH,"fused exact quotient overflow");
    }
    memcpy(q,estimate,qn*8);memcpy(r,value,dn*8);if(shift)divrem_words::shift_right(r,r,dn,shift);
    if(metrics)*metrics={0,0,4};return corrections;
}
}
uint64_t local_division_apply(const LocalDivisionPlan&p,const LocalDivisionDivisor &prepared,
                             uint64_t*q,uint64_t*r,const uint64_t*n,size_t nn,Frame&space,
                             LocalDivisionMetrics *metrics) noexcept {
    const size_t dn=p.denominator_limbs,m=p.block_limbs;
    require(nn<=p.numerator_limbs,SBN3_FATAL_ARGUMENT,"local division numerator");
    if(metrics)*metrics={};
    if(nn<dn){if(nn)memcpy(r,n,nn*8);memset(r+nn,0,(dn-nn)*8);return 0;}
    if(p.fused_precision)return fused_apply(p,prepared,q,r,n,nn,space,metrics);
    const size_t qn=nn-dn+1;FrameMark mark(space);
    auto *value=space.alloc<uint64_t>(dn+m+2),*x=value;
    // The quotient estimate is retired into caller-owned Q before the
    // residual product starts. Both outputs fit this one transient buffer.
    auto *estimate=value;
    const unsigned shift=prepared.shift;
    const auto *D=prepared.divisor;
    // U=B^m+u: qhat=rtop+high(rtop*u). The leading one need not enlarge
    // a transform. Its contribution is an exact add, followed by saturation.
    for(size_t j=0;j<dn;++j)r[j]=shifted(n,nn,qn+j,shift);
    size_t pos=qn;uint64_t total_corrections=0,head_corrections=0;unsigned products=0;
    const size_t head=local_division_head(qn,m);
    if(head){
        pos-=head;for(size_t j=0;j<head;++j)x[j]=shifted(n,nn,pos+j,shift);
        memcpy(x+head,r,dn*8);const auto inverse=prepared.head_inverse;
        for(size_t j=head;j-->0;){uint64_t corrections=0;q[pos+j]=head_word(x+j,D,dn,inverse,corrections);head_corrections+=corrections;}
        memcpy(r,x,dn*8);
    }
    while(pos){
        const size_t ic=std::min(m,pos);pos-=ic;
        product::local_product_apply(p.quotient,prepared.quotient,estimate,r+dn-ic,ic,space,{m,ic,1});
        auto *block=q+pos;
        if(sbn3i_add_n(block,estimate+m,r+dn-ic,long(ic)))memset(block,0xff,ic*8);
        product::local_product_apply(p.residual,prepared.residual,value,block,ic,space);
        products+=2;
        const size_t period=p.residual.ring,length=period?period+1:dn+ic+1;
        if(!period)x[dn+ic]=0;
        const product::DifferenceValue input{{n,nn},pos,ic,shift,{r,dn}};
        const bool negative=product::product_difference(x,length,period,input,block[0]*D[0]).negative;
        unsigned corrections=0;
        if(negative){
            while(x[dn]||divrem_words::compare(x,D,dn)>0){require(++corrections<=8,SBN3_FATAL_MATH,"local negative correction");limbs::sub_from(x,length,D,dn);}
            sbn3i_sub_n(x,D,x,long(dn));++corrections;
            const uint64_t c=corrections;require(!limbs::sub_from(block,ic,&c,1),SBN3_FATAL_MATH,"local quotient underflow");
        }else{
            while(x[dn]||divrem_words::compare(x,D,dn)>=0){require(++corrections<=8,SBN3_FATAL_MATH,"local positive correction");limbs::sub_from(x,length,D,dn);}
            const uint64_t c=corrections;require(!limbs::add_to(block,ic,&c,1),SBN3_FATAL_MATH,"local quotient overflow");
        }
        total_corrections+=corrections;memcpy(r,x,dn*8);
    }
    if(shift)divrem_words::shift_right(r,r,dn,shift);
    if(metrics)*metrics={head,head_corrections,products};
    return total_corrections;
}
uint64_t local_division_execute(const LocalDivisionPlan&p,uint64_t*q,uint64_t*r,const uint64_t*n,size_t nn,
                               const uint64_t*d,Frame&space,LocalDivisionMetrics *metrics) noexcept {
    require(nn<=p.numerator_limbs&&d&&d[p.denominator_limbs-1],SBN3_FATAL_ARGUMENT,"local division input");
    if(nn<p.denominator_limbs){
        if(nn)memcpy(r,n,nn*8);
        memset(r+nn,0,(p.denominator_limbs-nn)*8);
        if(metrics)*metrics={};
        return 0;
    }
    FrameMark mark(space);
    const auto prepared=local_division_prepare(p,d,space);
    return local_division_apply(p,prepared,q,r,n,nn,space,metrics);
}
}
