// SPDX-License-Identifier: LGPL-3.0-or-later
#define CR_NP 10
#define SBN3_P48_NS p48_external_rows
#include "backend/ntt_p48/col.hpp"
#include "product/modular_row.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <new>

namespace sbn::v3::product {
namespace {
namespace p = ::sbn::v3::p48_external_rows;
constexpr uint64_t magic = 0x53424e33524f5731ULL;
constexpr size_t leaf_limit = 4096;
struct alignas(64) Prepared {
    uint64_t marker = magic;
    ModularRowSpec spec{};
    ModularRowInfo info{};
    p::Prime prime{};
    p::PrimeV pv{};
    uint64_t roots[2][p::tower_root_slots]{};
    uint64_t tau[2][33]{};
    p::cc normalize{}, correction{}, global_wrap{};
    p::Pk52 codec{};
};
uint64_t root_at(const Prepared &c, unsigned inverse, size_t index) {
    uint64_t r = 1;
    while(index){const unsigned bit=unsigned(__builtin_ctzll(index));
        r=p::mulm(r,c.roots[inverse][bit+1],c.prime.p);index&=index-1;}
    return r;
}
p::cc twiddle(const Prepared &c, unsigned inverse, unsigned depth, size_t node) {
    const unsigned log = unsigned(__builtin_ctzll(c.spec.cells));
    const uint64_t r=root_at(c,inverse,2*node), t=c.tau[inverse][log-depth-1];
    return p::cc_of(t==1?r:p::mulm(r,t,c.prime.p),c.prime.p);
}
void leaf_table(const Prepared &c, bool inverse, size_t task, uint64_t *heap) {
    heap[0]=0;
    const unsigned log=unsigned(__builtin_ctzll(c.info.leaf_cells));
    for(size_t nb=1,level=0;nb<c.info.leaf_cells;nb*=2,++level){
        const uint64_t r=root_at(c,inverse,2*nb*task);
        const auto factor=p::cc_of(p::mulm(c.tau[inverse][log-unsigned(level)-1],r,c.prime.p),c.prime.p);
        if(nb<8){for(size_t t=0;t<nb;++t){
            const uint64_t rec=inverse?p::twr<true>(c.prime,t):p::twr<false>(c.prime,t);
            heap[nb+t]=p::cc_of(p::mulm(p::cc_ld(rec,c.prime.p).c,factor.c,c.prime.p),c.prime.p).rec;
        }}else for(size_t t=0;t<nb;t+=8){
            const p::V rec=inverse?p::tower_even8<true>(c.prime,t/8):p::tower_even8<false>(c.prime,t/8);
            p::V value=_mm512_madd52hi_epu64(p::vset(1),rec,c.pv.p);
            value=p::shp(p::b52(value,p::vset(factor.c),p::vset(factor.rec),c.pv.pn),c.pv.p);
            _mm512_storeu_si512(heap+nb+t,p::vrec_of(value,c.pv));
        }
    }
}
struct Run {const Prepared *c;uint64_t *a,*b;uint8_t *scratch;
    const uint8_t*packed_a=nullptr;uint8_t*packed_b=nullptr;};
template<bool Inverse,unsigned Factor>
__attribute__((always_inline)) inline void butterfly(p::V *x,const p::vcc *t,const p::PrimeV &pv) {
    static_assert(Factor==2||Factor==4||Factor==8);
    if constexpr(Factor==8){p::m8_core<Inverse,true>(x,t[0],t[1],t[2],t[3],t[4],t[5],t[6],pv);}
    else if constexpr(Factor==2){p::bf<Inverse>(x[0],x[1],t[0].c,t[0].rec,pv);}
    else if constexpr(!Inverse){
        p::bf<false>(x[0],x[2],t[0].c,t[0].rec,pv);p::bf<false>(x[1],x[3],t[0].c,t[0].rec,pv);
        p::bf<false>(x[0],x[1],t[1].c,t[1].rec,pv);p::bf<false>(x[2],x[3],t[2].c,t[2].rec,pv);
    }else{
        p::bf<true>(x[0],x[1],t[1].c,t[1].rec,pv);p::bf<true>(x[2],x[3],t[2].c,t[2].rec,pv);
        p::bf<true>(x[0],x[2],t[0].c,t[0].rec,pv);p::bf<true>(x[1],x[3],t[0].c,t[0].rec,pv);
    }
}
__attribute__((always_inline)) inline p::V load_packed(const uint8_t*src,size_t at,const p::Pk52&codec){
    // Exactly six words at every slot, including the final one: no per-cell
    // tail branch and no requirement for readable bytes after the row.
    return _mm512_maskz_permutexvar_epi8(codec.m48,codec.g48,_mm512_maskz_loadu_epi64(0x3f,src+48*at));
}
template<bool Inverse,unsigned Factor,bool PackedIn,bool PackedOut>
__attribute__((always_inline)) inline void upper_butterfly(uint64_t*data,size_t pos,size_t stride,const p::vcc*t,const Prepared&c,
                                                          const uint8_t*input,uint8_t*output){
    static_assert(!PackedIn||!Inverse);static_assert(!PackedOut||Inverse);
    p::V x[Factor];for(unsigned k=0;k<Factor;++k){const size_t at=pos+k*stride;
        if constexpr(PackedIn)x[k]=load_packed(input,at,c.codec);
        else x[k]=_mm512_load_si512(data+8*at);
    }
    const auto&pv=c.pv;
    butterfly<Inverse,Factor>(x,t,pv);
    for(unsigned k=0;k<Factor;++k){const size_t at=pos+k*stride;
        if constexpr(PackedOut)p::stslot<48>(output+48*at,x[k],c.codec,pv);
        else _mm512_store_si512(data+8*at,x[k]);
    }
}
template<bool Inverse,unsigned Factor,bool PackedIn,bool PackedOut,unsigned Inputs>
__attribute__((always_inline)) inline void upper_range(Run&run,size_t pos,size_t count,size_t stride,const p::vcc*t){
    const auto&c=*run.c;auto*a=run.a;auto*b=run.b;const auto*pa=run.packed_a;auto*pb=run.packed_b;
    for(size_t j=0;j<count;++j,++pos){
        if constexpr(Inputs&1)upper_butterfly<Inverse,Factor,PackedIn,PackedOut>(a,pos,stride,t,c,pa,nullptr);
        if constexpr(Inputs&2)upper_butterfly<Inverse,Factor,PackedIn,PackedOut>(b,pos,stride,t,c,pb,pb);
    }
}
template<bool Inverse,unsigned Levels,bool PackedIn,bool PackedOut>
void upper_fixed(Run &run,unsigned depth,unsigned rank,unsigned width) {
    constexpr unsigned levels=Levels,factor=1u<<Levels;
    const auto &c=*run.c; const size_t length=c.spec.cells>>depth,e=length>>levels;
    const size_t jobs=c.spec.cells/factor,lo=jobs*rank/width,hi=jobs*(rank+1)/width;
    const unsigned inputs=Inverse?2:unsigned(c.spec.operation==RowOperation::product&&run.a!=run.b&&!c.spec.a_transformed)|
        (unsigned(!c.spec.b_transformed)<<1);
    for(size_t j=lo;j<hi;){
        const size_t node=j/e,end=std::min(hi,(node+1)*e);
        p::vcc t[7];unsigned at=0;
        for(unsigned l=0;l<levels;++l)for(unsigned k=0;k<(1u<<l);++k)
            t[at++]=p::vcc_of(twiddle(c,Inverse,depth+l,(node<<l)+k));
        size_t pos=node*length+(j-node*e);
        if constexpr(Inverse)upper_range<Inverse,factor,PackedIn,PackedOut,2>(run,pos,end-j,e,t);
        else switch(inputs){
            case 1:upper_range<Inverse,factor,PackedIn,PackedOut,1>(run,pos,end-j,e,t);break;
            case 2:upper_range<Inverse,factor,PackedIn,PackedOut,2>(run,pos,end-j,e,t);break;
            case 3:upper_range<Inverse,factor,PackedIn,PackedOut,3>(run,pos,end-j,e,t);break;
        }
        j=end;
    }
}
template<bool Inverse,bool PackedIn=false,bool PackedOut=false>
void upper(Run&run,unsigned depth,unsigned levels,unsigned rank,unsigned width){
    switch(levels){
    case 1:upper_fixed<Inverse,1,PackedIn,PackedOut>(run,depth,rank,width);break;
    case 2:upper_fixed<Inverse,2,PackedIn,PackedOut>(run,depth,rank,width);break;
    case 3:upper_fixed<Inverse,3,PackedIn,PackedOut>(run,depth,rank,width);break;
    default:fatal(SBN3_FATAL_MATH,"modular row radix",levels,3);
    }
}
__attribute__((always_inline)) inline void product_leaf(const Prepared&c,p::V*a,p::V*b,p::cc prefix,size_t count){
        if(count>=8)for(size_t j=0;j<count;j+=8){
            const p::V rec=p::tower8<false>(c.prime,j);
            p::V roots=_mm512_madd52hi_epu64(p::vset(1),rec,c.pv.p);
            roots=p::shp(p::b52(roots,p::vset(prefix.c),p::vset(prefix.rec),c.pv.pn),c.pv.p);
            p::conv8x8<true>(b+j,a+j,p::vrec_of(roots,c.pv),c.pv,p::vset(c.correction.c),p::vset(c.correction.rec));
        }else for(size_t j=0;j<count;++j){
            const auto wrap=p::cc_of(p::mulm(prefix.c,root_at(c,0,j),c.prime.p),c.prime.p);
            p::conv8(b+j,a+j,wrap,c.pv);
            b[j]=p::b52(b[j],p::vset(c.correction.c),p::vset(c.correction.rec),c.pv.pn);
        }
}
void leaf(Run &run, size_t task, uint64_t *heap) {
    const auto &c=*run.c;const size_t n=c.info.leaf_cells,at=task*n;
    p::Prime prime=c.prime;prime.heap=heap;
    auto *a=reinterpret_cast<p::V *>(run.a?run.a+8*at:nullptr),*b=reinterpret_cast<p::V *>(run.b+8*at);
    if(c.spec.operation!=RowOperation::inverse&&(!c.spec.b_transformed||
       (c.spec.operation==RowOperation::product&&a!=b&&!c.spec.a_transformed))){
        leaf_table(c,false,task,heap);
        if(c.spec.operation==RowOperation::product && a!=b&&!c.spec.a_transformed)p::full<false,false,1,true>(reinterpret_cast<uint64_t *>(a),n,1,prime,c.pv);
        if(!c.spec.b_transformed)p::full<false,false,1,true>(reinterpret_cast<uint64_t *>(b),n,1,prime,c.pv);
    }
    if(c.spec.operation==RowOperation::product){
        const auto prefix=p::cc_of(p::mulm(c.global_wrap.c,root_at(c,0,at),c.prime.p),c.prime.p);
        product_leaf(c,a,b,prefix,n);
        leaf_table(c,true,task,heap);
        p::full<true,false,1,true>(reinterpret_cast<uint64_t*>(b),n,1,prime,c.pv);
        return;
    }
    if(c.spec.operation!=RowOperation::forward){
        leaf_table(c,true,task,heap);
        p::full<true,false,1,true>(reinterpret_cast<uint64_t *>(b),n,1,prime,c.pv,nullptr,
                                 c.spec.operation==RowOperation::inverse?&c.normalize:nullptr);
    }
}
void worker(void *ptr,sbn3_team_region *region) {
    auto &r=*static_cast<Run *>(ptr);const auto &c=*r.c;
    const unsigned rank=region?sbn3_region_rank(region):0,width=region?sbn3_region_width(region):1;
    auto join=[&]{if(region)sbn3_region_barrier(region);};
    if(c.spec.operation!=RowOperation::inverse&&(!c.spec.b_transformed||
       (c.spec.operation==RowOperation::product&&r.a!=r.b&&!c.spec.a_transformed)))
        for(unsigned d=0;d<c.info.depth;){const unsigned k=std::min(3u,c.info.depth-d);
            if(!d&&r.packed_b)upper<false,true>(r,d,k,rank,width);
            else upper<false>(r,d,k,rank,width);
            join();d+=k;}
    auto *heap=reinterpret_cast<uint64_t *>(r.scratch+rank*c.info.worker_bytes);
    // Upper stages own contiguous index ranges. Keep adjacent leaves on the
    // same worker instead of scattering each worker across the entire row.
    // Counts still differ by at most one and every independent leaf is covered.
    const size_t first=c.info.leaf_count*rank/width,last=c.info.leaf_count*(rank+1)/width;
    for(size_t task=first;task<last;++task)leaf(r,task,heap);
    join();
    if(c.spec.operation!=RowOperation::forward)
        for(unsigned end=c.info.depth;end;){const unsigned d=(end-1)/3*3;
            if(!d&&r.packed_b)upper<true,false,true>(r,d,end-d,rank,width);
            else upper<true>(r,d,end-d,rank,width);
            join();end=d;}
}
const Prepared &get(std::span<const uint8_t> storage) {
    require(storage.data() && storage.size()>=sizeof(Prepared) && !(uintptr_t(storage.data())&63),SBN3_FATAL_ARGUMENT,"modular row prepared span");
    const auto &c=*reinterpret_cast<const Prepared *>(storage.data());
    require(c.marker==magic,SBN3_FATAL_ARGUMENT,"modular row state");return c;
}
struct PackedRun {
    const PackedRowBatchSpec *spec;
    const ModularRowInfo *info;
    std::span<const uint8_t>packed_a;
    std::span<uint8_t>packed_b,prepared,scratch;
    std::span<uint64_t>a,b;
};
void packed_worker(void*ptr,sbn3_team_region*region){
    auto&r=*static_cast<PackedRun *>(ptr);const auto&s=*r.spec;
    const unsigned rank=region?sbn3_region_rank(region):0,width=region?sbn3_region_width(region):1;
    auto join=[&]{if(region)sbn3_region_barrier(region);};
    const size_t lo=s.cells*rank/width,hi=s.cells*(rank+1)/width;
    const unsigned outer_log=s.global_log-unsigned(__builtin_ctzll(s.cells));
    for(uint64_t row=0;row<s.rows;++row){
        if(!rank){const uint64_t index=s.first_row+row;
            const uint64_t frequency=outer_log?__builtin_bitreverse64(index)>>(64-outer_log):0;
            ModularRowSpec q{s.operation,s.prime,s.global_log,s.workers,s.cells,frequency,
                p48_row_coset(s.prime,s.global_log,frequency),s.product_scale,s.a_transformed,s.b_transformed};
            p48_row_prepare(q,*r.info,r.prepared);
        }
        join();const auto&c=get(r.prepared);
        const bool second=s.operation==RowOperation::product&&!s.square;
        const bool fused_edges=r.info->depth!=0;
        const bool fused_output=fused_edges&&s.operation!=RowOperation::forward;
        const auto*ap=second?r.packed_a.data()+row*s.row_stride:r.packed_b.data()+row*s.row_stride;
        auto*bp=r.packed_b.data()+row*s.row_stride;
        const bool unpack_a=second&&(!fused_edges||s.a_transformed);
        const bool unpack_b=!fused_edges||s.operation==RowOperation::inverse||s.b_transformed;
        if(unpack_a||unpack_b){
            for(size_t j=lo;j<hi;++j){
                if(unpack_a)_mm512_store_si512(r.a.data()+8*j,load_packed(ap,j,c.codec));
                if(unpack_b)_mm512_store_si512(r.b.data()+8*j,load_packed(bp,j,c.codec));
            }
            join();
        }
        Run arithmetic{&c,second?r.a.data():r.b.data(),r.b.data(),r.scratch.data(),fused_edges?ap:nullptr,fused_edges?bp:nullptr};worker(&arithmetic,region);
        if(!fused_output){
            for(size_t j=lo;j<hi;++j)p::stslot<48>(bp+48*j,_mm512_load_si512(r.b.data()+8*j),c.codec,c.pv);
            join();
        }
    }
}
void blocked_worker(void*ptr,sbn3_team_region*region){
    auto&r=*static_cast<Run*>(ptr);const auto&c=*r.c;
    const unsigned rank=region?sbn3_region_rank(region):0,width=region?sbn3_region_width(region):1;
    auto join=[&]{if(region)sbn3_region_barrier(region);};
    const size_t step=std::min(c.spec.cells,p48_row_io_cells),blocks=c.spec.cells/step;
    for(size_t block=rank;block<blocks;block+=width){const size_t first=block*step;
        auto unpack=[&](uint64_t*v){const auto*src=reinterpret_cast<const uint8_t*>(v+first*8);
            for(size_t j=step;j-- >0;)_mm512_store_si512(v+8*(first+j),p::ldslot<48>(src+48*j,c.codec));};
        if(r.a!=r.b)unpack(r.a);unpack(r.b);
    }
    join();worker(&r,region);
    for(size_t block=rank;block<blocks;block+=width){const size_t first=block*step;
        auto*dst=reinterpret_cast<uint8_t*>(r.b+first*8);
        for(size_t j=0;j<step;++j)p::stslot<48>(dst+48*j,_mm512_load_si512(r.b+8*(first+j)),c.codec,c.pv);
    }
    join();
}
}
uint64_t p48_row_prime(unsigned q) noexcept {return q<p::NPMAX?p::PR[q]:0;}
uint64_t p48_row_coset(unsigned q,unsigned log,uint64_t prefix) noexcept {
    require(q<p::NPMAX && log<=32 && prefix<(uint64_t(1)<<log),SBN3_FATAL_ARGUMENT,"modular row coset");
    const uint64_t root=log?p::powm(p::fixed_setup.prime[q].root32,uint64_t(1)<<(32-log),p::PR[q]):1;
    return p::powm(root,prefix,p::PR[q]);
}
bool p48_row_query(const ModularRowSpec &s, ModularRowInfo &out) noexcept {
    if(unsigned(s.operation)>unsigned(RowOperation::product) || s.prime>=p::NPMAX || !s.cells || (s.cells&(s.cells-1)) ||
       s.global_log>32 || !s.workers || s.workers>32 || !s.local_tau || s.local_tau>=p::PR[s.prime] || s.product_scale>=p::PR[s.prime]||
       (s.operation!=RowOperation::product&&(s.a_transformed||s.b_transformed)))return false;
    const unsigned log=unsigned(__builtin_ctzll(s.cells));
    if(log>s.global_log || s.global_prefix>=(uint64_t(1)<<(s.global_log-log)))return false;
    ModularRowInfo i{};i.leaf_cells=std::min(s.cells,leaf_limit);i.leaf_count=s.cells/i.leaf_cells;
    i.depth=unsigned(__builtin_ctzll(i.leaf_count));i.workers=unsigned(std::min(size_t(s.workers),i.leaf_count));
    if(!align_size(sizeof(Prepared),64,i.prepared_bytes) || !mul_size(s.cells,64,i.array_bytes) ||
       !mul_size(i.leaf_cells+8,8,i.worker_bytes) || !align_size(i.worker_bytes,64,i.worker_bytes) ||
       !mul_size(i.worker_bytes,i.workers,i.scratch_bytes))return false;
    i.shared_bytes=p48_root_bank::bytes();out=i;return true;
}
void p48_row_prepare(const ModularRowSpec &s,const ModularRowInfo &info,std::span<uint8_t> storage) noexcept {
    ModularRowInfo check{};
    require(p48_row_query(s,check) && check.prepared_bytes==info.prepared_bytes && check.scratch_bytes==info.scratch_bytes &&
            storage.data() && storage.size()>=check.prepared_bytes && !(uintptr_t(storage.data())&63),SBN3_FATAL_ARGUMENT,"modular row preparation");
    auto &c=*::new(storage.data())Prepared{};c.spec=s;c.info=check;
    c.prime.p=p::PR[s.prime];c.prime.p2=2*c.prime.p;
    uint64_t inverse=1;for(unsigned i=0;i<6;++i)inverse*=2-c.prime.p*inverse;
    c.prime.J=(0-inverse)&p::M52;c.prime.r1=uint64_t((p::u128(1)<<52)/c.prime.p);
    c.prime.inv2=p::cc_of((c.prime.p+1)/2,c.prime.p);
    c.prime.e=p48_root_bank::tower(s.prime,p48_root_bank::tower_log);
    c.prime.ez=p48_root_bank::zero_image(s.prime);c.prime.lg=p48_root_bank::tower_log;c.pv.init(c.prime);
    for(unsigned d=0;d<2;++d)p::tower_roots(c.roots[d],s.prime,d!=0);
    c.tau[0][0]=s.local_tau;c.tau[1][0]=p::invm(s.local_tau,c.prime.p);
    for(unsigned d=0;d<2;++d)for(unsigned k=1;k<=32;++k)c.tau[d][k]=p::mulm(c.tau[d][k-1],c.tau[d][k-1],c.prime.p);
    const uint64_t invn=p::inverse_power2_or_general(s.cells,c.prime.p);
    c.normalize=p::cc_of(invn,c.prime.p);
    c.correction=p::cc_of(p::mulm(p::mulm(uint64_t((p::u128(1)<<52)%c.prime.p),s.product_scale,c.prime.p),invn,c.prime.p),c.prime.p);
    c.global_wrap=p::cc_of(p::powm(s.global_log?c.roots[0][s.global_log]:1,s.global_prefix,c.prime.p),c.prime.p);
    c.codec=p::pk52_mk();
}
void p48_row_execute(std::span<const uint8_t> prepared,std::span<uint8_t> scratch,std::span<uint64_t>a,
                     std::span<uint64_t>b,sbn3_team_scope *scope) noexcept {
    const auto &c=get(prepared);const size_t bytes=c.info.array_bytes;
    require(b.data() && scratch.data() && b.size()>=bytes/8 && !(uintptr_t(b.data())&63) && scratch.size()>=c.info.scratch_bytes &&
        !(uintptr_t(scratch.data())&63) && scope && sbn3_team_width(scope)>=c.info.workers,
        SBN3_FATAL_WORKSPACE,"modular row arrays/scratch");
    if(c.spec.operation==RowOperation::product)require(a.data() && a.size()>=bytes/8 && !(uintptr_t(a.data())&63) &&
        (a.data()==b.data()?c.spec.a_transformed==c.spec.b_transformed:!overlaps(a.data(),bytes,b.data(),bytes)),SBN3_FATAL_ARGUMENT,"modular row product alias");
    for(const void *array:{static_cast<const void *>(b.data()),c.spec.operation==RowOperation::product?static_cast<const void *>(a.data()):nullptr})
        if(array)require(!overlaps(array,bytes,prepared.data(),prepared.size()) && !overlaps(array,bytes,scratch.data(),scratch.size()),SBN3_FATAL_ARGUMENT,"modular row resource alias");
    require(!overlaps(prepared.data(),prepared.size(),scratch.data(),scratch.size()),SBN3_FATAL_ARGUMENT,"modular row preparation alias");
    require_scope_leader(scope);Run r{&c,a.data(),b.data(),scratch.data()};
    if(c.info.workers==1)worker(&r,nullptr);
    else{ sbn3_team_scope part{scope->team,scope->first,c.info.workers,false,scope->epoch};sbn3_team_region_run(&part,worker,&r); }
}
void p48_row_blocked_product(std::span<const uint8_t> prepared,std::span<uint8_t> scratch,std::span<uint64_t>a,
                            std::span<uint64_t>b,sbn3_team_scope*scope) noexcept {
    const auto&c=get(prepared);const size_t bytes=c.info.array_bytes;
    require(c.spec.operation==RowOperation::product&&a.data()&&b.data()&&scratch.data()&&
        a.size()>=bytes/8&&b.size()>=bytes/8&&!(uintptr_t(a.data())&63)&&!(uintptr_t(b.data())&63)&&
        scratch.size()>=c.info.scratch_bytes&&!(uintptr_t(scratch.data())&63)&&scope&&sbn3_team_width(scope)>=c.info.workers,
        SBN3_FATAL_WORKSPACE,"blocked row arrays/scratch");
    require(a.data()==b.data()?c.spec.a_transformed==c.spec.b_transformed:!overlaps(a.data(),bytes,b.data(),bytes),
        SBN3_FATAL_ARGUMENT,"blocked row operand alias");
    for(const void*array:{a.data(),b.data()})require(!overlaps(array,bytes,prepared.data(),prepared.size())&&
        !overlaps(array,bytes,scratch.data(),scratch.size()),SBN3_FATAL_ARGUMENT,"blocked row resource alias");
    require(!overlaps(prepared.data(),prepared.size(),scratch.data(),scratch.size()),SBN3_FATAL_ARGUMENT,"blocked row preparation alias");
    require_scope_leader(scope);Run run{&c,a.data(),b.data(),scratch.data()};
    if(c.info.workers==1)blocked_worker(&run,nullptr);
    else{sbn3_team_scope part{scope->team,scope->first,c.info.workers,false,scope->epoch};sbn3_team_region_run(&part,blocked_worker,&run);}
}
void p48_row_unpack(std::span<const uint8_t> prepared,std::span<const uint8_t> packed,std::span<uint64_t>words) noexcept {
    const auto &c=get(prepared);const size_t n=c.spec.cells;
    require(packed.data() && words.data() && packed.size()>=n*48+16 && words.size()>=n*8 && !(uintptr_t(words.data())&63),SBN3_FATAL_WORKSPACE,"row unpack footprint");
    const bool same=packed.data()==reinterpret_cast<const uint8_t *>(words.data());
    require(same || !overlaps(packed.data(),n*48+16,words.data(),n*64),SBN3_FATAL_ARGUMENT,"row unpack overlap");
    if(same)for(size_t i=n;i-- >0;)_mm512_store_si512(words.data()+8*i,p::ldslot<48>(packed.data()+48*i,c.codec));
    else for(size_t i=0;i<n;++i)_mm512_store_si512(words.data()+8*i,p::ldslot<48>(packed.data()+48*i,c.codec));
}
void p48_row_pack(std::span<const uint8_t> prepared,std::span<const uint64_t>words,std::span<uint8_t>packed) noexcept {
    const auto &c=get(prepared);const size_t n=c.spec.cells;
    require(packed.data() && words.data() && packed.size()>=n*48 && words.size()>=n*8 && !(uintptr_t(words.data())&63),SBN3_FATAL_WORKSPACE,"row pack footprint");
    require(packed.data()==reinterpret_cast<const uint8_t *>(words.data()) || !overlaps(packed.data(),n*48,words.data(),n*64),SBN3_FATAL_ARGUMENT,"row pack overlap");
    for(size_t i=0;i<n;++i)p::stslot<48>(packed.data()+48*i,_mm512_load_si512(words.data()+8*i),c.codec,c.pv);
}
void p48_rows_packed(const PackedRowBatchSpec&s,const ModularRowInfo&info,std::span<const uint8_t>pa,std::span<uint8_t>pb,
    std::span<uint8_t>prepared,std::span<uint8_t>scratch,std::span<uint64_t>a,std::span<uint64_t>b,sbn3_team_scope*scope) noexcept {
    ModularRowInfo check{};ModularRowSpec q{s.operation,s.prime,s.global_log,s.workers,s.cells,0,1,s.product_scale,s.a_transformed,s.b_transformed};
    const bool second=s.operation==RowOperation::product&&!s.square;
    uint64_t bytes=0;
    require(p48_row_query(q,check)&&s.rows&&s.row_stride>=s.cells*48&&
        !__builtin_mul_overflow(s.rows-1,s.row_stride,&bytes)&&!__builtin_add_overflow(bytes,uint64_t(s.cells*48),&bytes)&&
        s.first_row<(uint64_t(1)<<(s.global_log-unsigned(__builtin_ctzll(s.cells))))&&
        s.rows<=(uint64_t(1)<<(s.global_log-unsigned(__builtin_ctzll(s.cells))))-s.first_row&&
        check.prepared_bytes==info.prepared_bytes&&check.scratch_bytes==info.scratch_bytes&&check.workers==info.workers&&
        prepared.data()&&!(uintptr_t(prepared.data())&63)&&prepared.size()>=info.prepared_bytes&&
        scratch.data()&&!(uintptr_t(scratch.data())&63)&&scratch.size()>=info.scratch_bytes&&
        pb.data()&&pb.size()>=bytes&&b.data()&&!(uintptr_t(b.data())&63)&&b.size()>=s.cells*8&&
        (!second||(pa.data()&&pa.size()>=bytes&&a.data()&&!(uintptr_t(a.data())&63)&&a.size()>=s.cells*8))&&
        (!s.square||s.a_transformed==s.b_transformed)&&
        scope&&sbn3_team_width(scope)>=info.workers,SBN3_FATAL_WORKSPACE,"packed modular row grants");
    const std::span<const uint8_t>grants[]{prepared,scratch,
        {reinterpret_cast<uint8_t *>(b.data()),check.array_bytes},
        {second?reinterpret_cast<uint8_t *>(a.data()):nullptr,second?check.array_bytes:0}};
    for(unsigned i=0;i<4;++i){const auto span=grants[i];
        require(!overlaps(span.data(),span.size(),pb.data(),size_t(bytes))&&
            (!second||!overlaps(span.data(),span.size(),pa.data(),size_t(bytes))),SBN3_FATAL_ARGUMENT,"packed row alias");
        for(unsigned j=0;j<i;++j)require(!overlaps(span.data(),span.size(),grants[j].data(),grants[j].size()),
                                       SBN3_FATAL_ARGUMENT,"packed row work overlap");
    }
    require(!second||!overlaps(pa.data(),size_t(bytes),pb.data(),size_t(bytes)),SBN3_FATAL_ARGUMENT,"packed operands overlap");
    require_scope_leader(scope);PackedRun run{&s,&check,pa,pb,prepared,scratch,a,b};
    if(info.workers==1)packed_worker(&run,nullptr);
    else{ sbn3_team_scope part{scope->team,scope->first,info.workers,false,scope->epoch};sbn3_team_region_run(&part,packed_worker,&run); }
}
} // namespace sbn::v3::product
