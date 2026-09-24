#pragma once
// Compact denominator services. No product/Newton binding exists in these
// recipes; their retained state is exactly the state used by the thin kernels.
#include "sbn3/divrem.h"
#include "algorithms/local_divrem.hpp"
#include "algorithms/small_division.hpp"
#include "backend/u52/kernels.hpp"
#include "common/identity.hpp"
#include "common/small_checks.h"
#include "runtime/team.hpp"
#include "runtime/scratch.hpp"
#include "value/divrem_words.hpp"
#include <algorithm>
#include <cstring>
#include <new>
#include <time.h>

namespace sbn::v3::divrem_prepared {
constexpr uint64_t magic=0x53424e3344505231ULL;
constexpr size_t up(size_t n) noexcept {return (n+127)&~size_t(127);}
union Geometry {
    LocalDivisionPlan local;
    Geometry() noexcept {}
    ~Geometry() noexcept = default;
};
struct Plan {
    uint64_t marker;
    size_t nn,dn,bytes,work_at,work_bytes;
    unsigned algorithm,timing;
    Geometry geometry;
};
// Only common lifetime facts are in the header. A word divisor does not
// carry native-vector or FFT geometry, and only requested timing has a tail.
constexpr unsigned ready=1, timed=2;
using ExecuteFn=void (*)(sbn3_divrem_binding *,sbn3_const_limbs,sbn3_limbs,sbn3_limbs,sbn3_divrem_result *);
template<size_t Dn> void execute_word(sbn3_divrem_binding *,sbn3_const_limbs,sbn3_limbs,sbn3_limbs,sbn3_divrem_result *);
void execute_compact(sbn3_divrem_binding *,sbn3_const_limbs,sbn3_limbs,sbn3_limbs,sbn3_divrem_result *);
inline ExecuteFn executor(unsigned algorithm,size_t dn,unsigned flags) noexcept {
    if constexpr(!SBN3_CHECK_SMALL){
        if(algorithm<=SBN3_DIVREM_SCHOOLBOOK&&!(flags&timed))
            return dn==1?execute_word<1>:dn==2?execute_word<2>:execute_word<0>;
    }
    return execute_compact;
}
struct Binding {
    ExecuteFn run;
    size_t nn,dn,bytes; // nn: diagnostics; bytes: diagnostics or non-scalar prepare
    uint64_t prepares,executes;
    unsigned algorithm,flags;
    sbn3_team *team; // diagnostics only; compact arithmetic is single-worker
    uint64_t diagnostic_lease;
};
[[gnu::always_inline]] inline void initialize(Binding &b,size_t nn,size_t dn,size_t bytes,
                                             unsigned algorithm,unsigned flags,sbn3_team *team,uint64_t token) noexcept {
    b.run=executor(algorithm,dn,flags);b.dn=dn;
    b.prepares=bool(flags&ready);b.executes=0;b.algorithm=algorithm;b.flags=flags;
    if constexpr(SBN3_CHECK_SMALL){b.nn=nn;b.bytes=bytes;b.team=team;b.diagnostic_lease=token;}
    else if(algorithm>=SBN3_DIVREM_BARRETT)b.bytes=bytes;
}
struct WordBinding {
    Binding head;
    union { divrem_words::PreparedDivisor state; };
    uint64_t *scratch;
    WordBinding() noexcept {}
    ~WordBinding() noexcept = default;
};
struct NativeBinding {
    Binding head;
    union { u52::DivisionDivisor state; };
    unsigned char *work;size_t work_bytes;
    NativeBinding() noexcept {}
    ~NativeBinding() noexcept = default;
};
struct LocalBinding {
    Binding head;
    union { LocalDivisionPlan plan; };
    union { LocalDivisionDivisor state; };
    unsigned char *work;size_t work_bytes;
    uint64_t corrections,head_limbs,head_corrections;
    unsigned products;
    LocalBinding() noexcept {}
    ~LocalBinding() noexcept = default;
};
struct Timing {uint64_t prepare_ns,execute_ns;};
constexpr size_t record_bytes(unsigned algorithm) noexcept {
    return algorithm==SBN3_DIVREM_DC?sizeof(NativeBinding):
           algorithm==SBN3_DIVREM_BARRETT?sizeof(LocalBinding):sizeof(WordBinding);
}
constexpr size_t control_bytes(unsigned algorithm,unsigned timing) noexcept {
    return up(record_bytes(algorithm)+(timing?sizeof(Timing):0));
}
struct WordLayout {size_t work_at,work_bytes,bytes;};
inline WordLayout word_layout(size_t nn,size_t dn,unsigned timing) noexcept {
    const size_t at=control_bytes(SBN3_DIVREM_WORD,timing)+(dn<=2?0:up(dn*8));
    const size_t work=dn==1?0:(nn+1)*8;
    return {at,work,up(at+work)};
}
static_assert(sizeof(Plan)<=1024&&control_bytes(SBN3_DIVREM_WORD,0)==128&&
              control_bytes(SBN3_DIVREM_DC,1)<=1024&&control_bytes(SBN3_DIVREM_BARRETT,1)<=1024);
inline WordBinding &word(Binding &b) noexcept {return *reinterpret_cast<WordBinding *>(&b);}
inline NativeBinding &native(Binding &b) noexcept {return *reinterpret_cast<NativeBinding *>(&b);}
inline LocalBinding &local(Binding &b) noexcept {return *reinterpret_cast<LocalBinding *>(&b);}
inline Timing &timing(Binding &b) noexcept {
    return *reinterpret_cast<Timing *>(reinterpret_cast<unsigned char *>(&b)+record_bytes(b.algorithm));
}
inline bool is_prepared(const Binding &b) noexcept {return b.flags&ready;}
inline uint64_t now() noexcept {
    timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);
    return uint64_t(t.tv_sec)*1000000000+uint64_t(t.tv_nsec);
}
[[gnu::always_inline]] inline bool quote(size_t nn,size_t dn,unsigned algorithm,size_t block,unsigned timing,
                   Plan &p,sbn3_divrem_info &i,bool describe=true,unsigned reuse_hint=0,bool allow_fused=false) noexcept {
    p.marker=magic;p.nn=nn;p.dn=dn;p.algorithm=algorithm;p.timing=timing;
    const size_t control=control_bytes(algorithm,timing);
    size_t persistent=0,work=0,total=0;
    if(block){
        ::new(&p.geometry.local) LocalDivisionPlan(allow_fused?local_division_auto_plan(nn,dn,block,reuse_hint):
                                                 local_division_plan_for_block(nn,dn,block,reuse_hint));
        persistent=p.geometry.local.persistent_bytes;work=p.geometry.local.work_bytes;total=p.geometry.local.storage_bytes;
    }else if(algorithm==SBN3_DIVREM_DC){
        persistent=u52::divide_divisor_bytes(dn);work=u52::divide_work_bytes(nn,dn);
        total=up(persistent)+work;
    }else{
        const auto scalar=word_layout(nn,dn,timing);
        persistent=scalar.work_at-control;work=scalar.work_bytes;total=scalar.bytes-control;
    }
    p.work_at=control+up(persistent);p.work_bytes=work;
    p.bytes=up(control+total);
    if(!describe)return true;
    i={};i.numerator_limbs=nn;i.denominator_limbs=i.remainder_limbs=dn;
    i.quotient_limbs=nn>=dn?nn-dn+1:0;i.algorithm=algorithm;i.workers=1;
    i.control_bytes=control;i.persistent_bytes=up(persistent);
    i.storage_bytes=p.bytes;i.storage_alignment=128;
    i.shared_bytes=p.bytes-p.work_at;i.scratch_bytes=work;
    if(block){
        const auto &l=p.geometry.local;
        i.block_limbs=i.inverse_limbs=l.block_limbs;i.ring_limbs=l.residual.ring;
        i.head_limbs=local_division_head(i.quotient_limbs,l.block_limbs);
        i.blocks=unsigned((i.quotient_limbs-i.head_limbs+l.block_limbs-1)/l.block_limbs);i.products=2;
        if(l.fused_precision){i.block_limbs=i.quotient_limbs;i.head_limbs=0;i.blocks=1;i.products=4;}
        i.table_bytes=(l.quotient.shape.nfull?pq16::table_bytes(l.quotient.shape):0)+
          (l.residual.shape.nfull&&!local_division_shares_tables(l)?pq16::table_bytes(l.residual.shape):0);
        i.spectrum_bytes=16*((l.quotient.cached?size_t(l.quotient.shape.nfull):0)+(l.residual.cached?size_t(l.residual.shape.nfull):0));
        i.product_workspace_bytes=std::max(l.quotient.work_bytes,l.residual.work_bytes);
    }
    // Scalar requests have two <2^31 lengths; their layout is a pure function
    // of those lengths and timing. Encode that identity exactly, without a
    // hash. The high bit separates it from the transform/native family.
    if(algorithm<=SBN3_DIVREM_SCHOOLBOOK)
        i.plan_id=(uint64_t(1)<<63)|(uint64_t(dn)<<32)|(uint64_t(nn)<<1)|timing;
    else{
        uint64_t id=identity::word(magic,(uint64_t(nn)*0xd6e8feb86659fd93ULL)^
            (uint64_t(dn)*0xa0761d6478bd642fULL)^uint64_t(p.bytes)^(uint64_t(algorithm)<<56));
        if(block){
            id=identity::word(id,block);
            id=identity::word(id,p.geometry.local.fused_precision);
            for(const auto *s:{&p.geometry.local.quotient.shape,&p.geometry.local.residual.shape}){
                id=identity::word(id,(uint64_t(s->nfull)<<32)|s->branch);
                id=identity::word(id,uint64_t(s->radix)|(uint64_t(s->recipe)<<8)|
                    (uint64_t(s->bits)<<16)|(uint64_t(s->centered)<<24)|(uint64_t(s->balanced)<<25));
            }
            id=identity::word(id,p.geometry.local.residual.ring);
            id=identity::word(id,uint64_t(p.geometry.local.quotient.cached)|(uint64_t(p.geometry.local.residual.cached)<<1));
        }
        i.plan_id=id&~(uint64_t(1)<<63);
    }
    return true;
}
inline void store(const Plan &p,sbn3_divrem_plan *out) noexcept {
    // Unused ABI envelope bytes are deliberately untouched. They are not a
    // serialized representation or part of the plan identity.
    const size_t bytes=p.algorithm==SBN3_DIVREM_BARRETT?sizeof p:offsetof(Plan,geometry);
    std::memcpy(out->opaque,&p,bytes);
}
inline Binding &get(sbn3_divrem_binding *p) noexcept {
    auto *b=reinterpret_cast<Binding *>(p);
    require(b&&b->run,SBN3_FATAL_LIFETIME,"prepared division binding");
    if constexpr(SBN3_CHECK_SMALL)
        require(b->run==executor(b->algorithm,b->dn,b->flags),SBN3_FATAL_LIFETIME,"prepared division executor");
    return *b;
}
inline void idle(const Binding &b) noexcept {
    if constexpr(SBN3_CHECK_SMALL)
        require(!b.team->busy&&pthread_equal(b.team->creator,pthread_self()),SBN3_FATAL_TEAM,"prepared division owner");
}
inline void span(const Binding &b,const void *data,size_t bytes) noexcept {
    valid_span(data,bytes,"prepared division span");
    require(!(uintptr_t(data)&7)&&!overlaps(data,bytes,&b,b.bytes),SBN3_FATAL_ARGUMENT,"prepared division alias");
    for(unsigned j=0;j<b.team->width;++j){
        const auto &l=j?b.team->stacks[j]:b.team->storage;
        require(!overlaps(data,bytes,l.data,l.bytes),SBN3_FATAL_ARGUMENT,"prepared division team alias");
    }
}
[[gnu::always_inline]] inline void bind_plan(const Plan &p,sbn3_arena *arena,size_t offset,sbn3_team *team,
                                             sbn3_divrem_binding **out) noexcept {
    require(arena&&team&&out&&p.marker==magic&&team->arena==arena,SBN3_FATAL_TEAM,"prepared division bind");
    auto *data=arena->base+offset;
    require(!(uintptr_t(data)&127),SBN3_FATAL_WORKSPACE,"prepared division range");
    uint64_t token=0;
    if constexpr(SBN3_CHECK_SMALL){
        require(team->width&&!team->busy&&pthread_equal(team->creator,pthread_self()),SBN3_FATAL_TEAM,"prepared division owner");
        require(arena->unleased(offset,p.bytes),SBN3_FATAL_WORKSPACE,"prepared division exclusive range");
        token=arena->acquire(offset,p.bytes).token;
    }
    Binding *b;
    if(p.algorithm==SBN3_DIVREM_BARRETT){
        auto *v=::new(data)LocalBinding;
        ::new(&v->plan)LocalDivisionPlan(p.geometry.local);
        v->work=data+p.work_at;v->work_bytes=p.work_bytes;
        v->corrections=v->head_limbs=v->head_corrections=0;v->products=0;b=&v->head;
    }else if(p.algorithm==SBN3_DIVREM_DC){
        auto *v=::new(data)NativeBinding;v->work=data+p.work_at;v->work_bytes=p.work_bytes;b=&v->head;
    }else{
        auto *v=::new(data)WordBinding;v->scratch=reinterpret_cast<uint64_t *>(data+p.work_at);b=&v->head;
    }
    initialize(*b,p.nn,p.dn,p.bytes,p.algorithm,p.timing?timed:0,team,token);
    if(p.timing)::new(data+record_bytes(p.algorithm))Timing{};
    *out=reinterpret_cast<sbn3_divrem_binding *>(b);
}
[[gnu::always_inline]] inline void bind(const sbn3_divrem_plan *opaque,sbn3_arena *arena,size_t offset,sbn3_team *team,
                                       sbn3_divrem_binding **out) noexcept {
    if constexpr(!SBN3_CHECK_SMALL){
        unsigned algorithm,timing;
        const auto *bytes=reinterpret_cast<const unsigned char *>(opaque->opaque);
        std::memcpy(&algorithm,bytes+offsetof(Plan,algorithm),sizeof algorithm);
        std::memcpy(&timing,bytes+offsetof(Plan,timing),sizeof timing);
        if(algorithm<=SBN3_DIVREM_SCHOOLBOOK&&!timing){
            size_t dn,work_at;
            std::memcpy(&dn,bytes+offsetof(Plan,dn),sizeof dn);
            std::memcpy(&work_at,bytes+offsetof(Plan,work_at),sizeof work_at);
            require(arena&&team&&out&&team->arena==arena,SBN3_FATAL_TEAM,"prepared division bind");
            auto *data=arena->base+offset;
            require(!(uintptr_t(data)&127),SBN3_FATAL_WORKSPACE,"prepared division range");
            auto *b=::new(data)WordBinding;b->scratch=reinterpret_cast<uint64_t *>(data+work_at);
            initialize(b->head,0,dn,0,algorithm,0,team,0);
            *out=reinterpret_cast<sbn3_divrem_binding *>(b);return;
        }
    }
    Plan p;std::memcpy(&p,opaque->opaque,offsetof(Plan,geometry));
    if(p.algorithm==SBN3_DIVREM_BARRETT)std::memcpy(&p.geometry.local,
        reinterpret_cast<const char *>(opaque->opaque)+offsetof(Plan,geometry),sizeof p.geometry.local);
    bind_plan(p,arena,offset,team,out);
}
[[gnu::always_inline]] inline void prepare(Binding &b,sbn3_const_limbs d) noexcept {
    idle(b);
    require(d.count==b.dn&&d.data&&d.data[b.dn-1],SBN3_FATAL_ARGUMENT,"prepared division divisor");
    if constexpr(SBN3_CHECK_SMALL){
        span(b,d.data,b.dn*8);
    }
    const bool clocks=b.flags&timed;
    const auto start=clocks?now():0;
    auto *data=reinterpret_cast<unsigned char *>(&b);
    const size_t control=control_bytes(b.algorithm,clocks);
    if(b.algorithm==SBN3_DIVREM_BARRETT||b.algorithm==SBN3_DIVREM_DC){
        auto memory=Frame::external(data+control,b.bytes-control);
        if(b.algorithm==SBN3_DIVREM_BARRETT){auto &v=local(b);
            ::new(&v.state)LocalDivisionDivisor(local_division_prepare(v.plan,d.data,memory));
        }else ::new(&native(b).state)u52::DivisionDivisor(u52::divide_prepare(d.data,b.dn,memory));
    }else ::new(&word(b).state)divrem_words::PreparedDivisor(
        divrem_words::prepare(d.data,b.dn,reinterpret_cast<uint64_t *>(data+control),true));
    b.flags|=ready;++b.prepares;
    if(clocks)timing(b).prepare_ns=now()-start;
}
[[gnu::always_inline]] inline void execute(Binding &b,sbn3_const_limbs n,sbn3_limbs q,sbn3_limbs r,
                                          sbn3_divrem_result *out) noexcept {
    idle(b);const size_t dn=b.dn,qn=n.count>=dn?n.count-dn+1:0;
    if constexpr(SBN3_CHECK_SMALL){
        require(is_prepared(b)&&n.count<=b.nn&&(!n.count||n.data)&&r.data&&r.capacity>=dn&&out&&
                (!qn||(q.data&&q.capacity>=qn)),SBN3_FATAL_ARGUMENT,"prepared division execution");
        span(b,n.data,n.count*8);span(b,q.data,qn*8);span(b,r.data,dn*8);
        require(!overlaps(n.data,n.count*8,q.data,qn*8)&&!overlaps(n.data,n.count*8,r.data,dn*8)&&
                !overlaps(q.data,qn*8,r.data,dn*8),SBN3_FATAL_ARGUMENT,"prepared division values");
    }
    const bool clocks=b.flags&timed;const auto start=clocks?now():0;
    uint64_t corrections=0;size_t qs=0;
    if(b.algorithm<=SBN3_DIVREM_SCHOOLBOOK){
        // The common scalar kernel already trims N, clears the whole Q span
        // and returns its length. Do not normalize and emit that work twice.
        auto &v=word(b);qs=divrem_words::schoolbook_prepared(q.data,r.data,n.data,n.count,v.state,v.scratch);
    }else{
        size_t used=n.count;
        while(used&&!n.data[used-1])--used;
        if(b.algorithm==SBN3_DIVREM_BARRETT){
            auto &v=local(b);auto memory=Frame::external(v.work,v.work_bytes);LocalDivisionMetrics m;
            corrections=local_division_apply(v.plan,v.state,q.data,r.data,n.data,used,memory,&m);
            v.products=m.products;v.head_limbs+=m.head_limbs;v.head_corrections+=m.head_corrections;v.corrections+=corrections;
        }else if(b.algorithm==SBN3_DIVREM_DC){
            auto &v=native(b);auto memory=Frame::external(v.work,v.work_bytes);
            u52::divide_prepared(q.data,r.data,n.data,used,v.state,memory);
        }
        const size_t written=used>=dn?used-dn+1:0;
        if(qn>written)std::memset(q.data+written,0,(qn-written)*8);
        qs=written;while(qs&&!q.data[qs-1])--qs;
    }
    size_t rs=dn;while(rs&&!r.data[rs-1])--rs;
    *out={qs,rs,corrections};++b.executes;
    if(clocks)timing(b).execute_ns=now()-start;
}
inline bool same_divisor(Binding &b,sbn3_int_view d) noexcept {
    if(d.size!=b.dn)return false;
    if(b.algorithm==SBN3_DIVREM_DC){
        const auto &p=native(b).state;uint64_t difference=0;
        for(size_t j=0;j<d.size;++j){
            const size_t bit=p.shift+64*j,k=bit/52;const unsigned s=bit%52;
            uint64_t w=(p.digits[k]>>s)|(p.digits[k+1]<<(52-s));
            if(s>40)w|=p.digits[k+2]<<(104-s);
            difference|=w^d.data[j];
        }
        return !difference;
    }
    const bool barrett=b.algorithm==SBN3_DIVREM_BARRETT;
    const unsigned s=barrett?local(b).state.shift:word(b).state.shift;
    if(d.size==1)return (word(b).state.short_words[0]>>s)==d.data[0];
    const auto *D=barrett?local(b).state.divisor:
                  d.size==2?word(b).state.short_words:word(b).state.divisor;
    uint64_t difference=0,carry=0;
    for(size_t j=0;j<d.size;++j){difference|=((d.data[j]<<s)|carry)^D[j];carry=s?d.data[j]>>(64-s):0;}
    return !difference&&!carry;
}
inline sbn3_divrem_metrics metrics(Binding &b) noexcept {
    sbn3_divrem_metrics m{};m.prepares=b.prepares;m.executes=b.executes;
    if(b.flags&timed){m.prepare_ns=timing(b).prepare_ns;m.execute_ns=timing(b).execute_ns;}
    if(b.algorithm==SBN3_DIVREM_BARRETT){const auto &v=local(b);
        m.corrections=v.corrections;m.head_limbs=v.head_limbs;m.head_corrections=v.head_corrections;m.products_executed=v.products;
    }
    return m;
}
inline void unbind(Binding &b) noexcept {
    idle(b);const unsigned algorithm=b.algorithm;
    sbn3_arena *arena=nullptr;sbn3_lease lease{};
    if constexpr(SBN3_CHECK_SMALL){arena=static_cast<sbn3_arena *>(b.team->arena);lease={&b,b.bytes,b.diagnostic_lease};}
    b.run=nullptr;
    if(algorithm==SBN3_DIVREM_BARRETT)local(b).~LocalBinding();
    else if(algorithm==SBN3_DIVREM_DC)native(b).~NativeBinding();
    else word(b).~WordBinding();
    if constexpr(SBN3_CHECK_SMALL)arena->release(lease);
    else {(void)arena;(void)lease;}
}
} // namespace sbn::v3::divrem_prepared
