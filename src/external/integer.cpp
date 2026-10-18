// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/integer.hpp"
#include "common/checked.hpp"
#include "common/m61.hpp"
#include "external/trace.hpp"
#include "core/x86_64/word.hpp"
#include "value/bitwise.hpp"
#include "external/two_slot.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cstring>
#include <utility>
namespace sbn::v3::external {
namespace {
constexpr size_t maximum_batch=1u<<20;
// An arbitrary bit shift is a view, never a stored zero/shift materialization.
IoError window(ValueReader &reader,const LinearInput&input,int64_t low,size_t first,
               size_t n,std::span<uint64_t> out,std::span<uint8_t> raw) {
    const size_t count=input.limbs();const int64_t exponent=input.exponent2;
    if(!count){std::fill_n(out.data(),n,uint64_t(0));return {};}
    const __int128 shift=__int128(exponent)-low;
    const unsigned bits=unsigned(shift%64);
    const __int128 start=__int128(first)-shift/64;
    const __int128 from=std::max(__int128(0),start-(bits!=0));
    const __int128 end=std::min(__int128(count),start+n);
    if(from>=end){std::fill_n(out.data(),n,uint64_t(0));return {};}
    const size_t loaded=size_t(end-from);
    const __int128 lo=std::max(start,from),hi=std::min(start+n,end+unsigned(bits!=0));
    const size_t begin=size_t(lo-start),stop=size_t(hi-start),source=size_t(lo-from);
    if(!bits){
        if(input.value||input.resident.data()){
            // The caller reserves two readable/writable view guards. Plain
            // views read directly into their arithmetic operand array.
            auto target=std::span(reinterpret_cast<uint8_t*>(out.data()+begin),(out.size()-begin)*8);
            if(auto e=read_source_words(reader,input.source,size_t(from),loaded,target))return e;
        }else{require(from==0&&loaded==1,SBN3_FATAL_MATH,"linear scalar window");out[begin]=input.scalar;}
        std::fill_n(out.data(),begin,uint64_t(0));std::fill(out.data()+stop,out.data()+n,uint64_t(0));return {};
    }
    // Two explicit zero neighbours turn a shifted bounded view into one
    // contiguous loop. Resolve wide coordinates once, never per limb.
    auto*data=reinterpret_cast<uint64_t*>(raw.data());data[0]=0;
    if(input.value||input.resident.data()){
        if(auto e=read_source_words(reader,input.source,size_t(from),loaded,raw.subspan(8)))return e;
    }else{require(from==0&&loaded==1,SBN3_FATAL_MATH,"linear scalar window");data[1]=input.scalar;}
    data[loaded+1]=0;
    std::fill_n(out.data(),begin,uint64_t(0));std::fill(out.data()+stop,out.data()+n,uint64_t(0));
    for(size_t j=0;j<stop-begin;++j)out[begin+j]=(data[source+j+1]<<bits)|(data[source+j]>>(64-bits));
    return {};
}
uint64_t stitch(uint64_t*data,size_t n,bool subtract,uint64_t incoming,uint64_t high){
    if(incoming){const size_t pivot=subtract?bits::first_nonzero(data,n):bits::first_not_ones(data,n);
        if(pivot)std::memset(data,subtract?255:0,pivot*8);
        if(pivot<n)data[pivot]+=subtract?UINT64_MAX:1;else high=1;}
    return high;
}
void check_bound(const StoredSumPlan&p,std::span<const uint64_t>data,size_t first){
    const size_t n=data.size();
    if(p.bound.verify&&p.bound.high_exclusive&&first+n>p.bound.words){size_t begin=0;
        if(first<=p.bound.words){begin=p.bound.words-first;
            require(data[begin]<p.bound.high_exclusive,SBN3_FATAL_MATH,"stored linear magnitude bound");++begin;}
        require(bits::first_nonzero(data.data()+begin,n-begin)==n-begin,SBN3_FATAL_MATH,"stored linear high support");}
}
void add_residue(uint64_t&sum,uint64_t term){sum+=term;if(sum>=m61::modulus)sum-=m61::modulus;}
uint64_t at_word(uint64_t residue,size_t word){return m61::multiply(residue,uint64_t(1)<<((3*(word%61))%61));}
struct LinearTotals {uint64_t carry=0,rx=0,ry=0,rr=0;bool nonzero=false;WordExtent extent{};};
struct LinearPipeline {
    const StoredSumPlan&p;Transport&t;ValueReader&a,&b;const LinearInput&x,&y;ValueWriter&writer;
    std::span<uint8_t>w;bool subtract,swap;LinearTotals&total;
    TwoSlotExchange exchange{};IoError error{};
    size_t words()const{return p.transfer_bytes/8;}
    size_t jobs()const{return (p.result_limbs+words()-1)/words();}
    std::span<uint64_t>av(unsigned slot){return {reinterpret_cast<uint64_t*>(w.data()+(slot?p.a2:p.a)),words()+2};}
    std::span<uint64_t>bv(unsigned slot){return {reinterpret_cast<uint64_t*>(w.data()+(slot?p.b2:p.b)),words()+2};}
    void fail(IoError e){error=e;exchange.stop();}
    bool write(size_t job){const unsigned slot=unsigned(job%2);if(!exchange.wait(slot,TwoSlotExchange::computed))return false;
        const size_t first=job*words(),n=std::min(words(),p.result_limbs-first);
        const size_t lo=std::max(first,p.first_output_limb),hi=std::min(first+n,p.end_output_limb);
        if(lo<hi)if(auto e=writer.append({reinterpret_cast<const uint8_t*>(av(slot).data()+lo-first),(hi-lo)*8})){fail(e);return false;}
        return true;
    }
};
void linear_io(void*ptr,sbn3_team_scope*){
    auto&c=*static_cast<LinearPipeline*>(ptr);
    for(size_t job=0;job<c.jobs();++job){const unsigned slot=unsigned(job%2);
        if(job>=2&&!c.write(job-2))return;
        if(c.t.cancel.test()){c.fail({IoCode::cancelled,"linear pipeline boundary"});return;}
        const size_t first=job*c.words(),n=std::min(c.words(),c.p.result_limbs-first);
        auto raw=c.w.subspan(c.p.raw,c.p.transfer_bytes+32);
        if(auto e=window(c.a,c.x,c.p.exponent,first,n,c.av(slot),raw)){c.fail(e);return;}
        if(auto e=window(c.b,c.y,c.p.exponent,first,n,c.bv(slot),raw)){c.fail(e);return;}
        c.exchange.publish(slot,TwoSlotExchange::ready);
    }
    for(size_t job=c.jobs()-std::min(size_t(2),c.jobs());job<c.jobs();++job)if(!c.write(job))return;
}
void compute_chunk(const StoredSumPlan&p,std::span<uint64_t>a,std::span<const uint64_t>b,size_t first,
                   bool subtract,bool swap,LinearTotals&total){
    const size_t n=a.size();m61::add_at(total.rx,a.data(),n,first);m61::add_at(total.ry,b.data(),n,first);
    const uint64_t high=subtract?sbn3i_sub_n(a.data(),swap?b.data():a.data(),swap?a.data():b.data(),long(n)):
                                  sbn3i_add_n(a.data(),a.data(),b.data(),long(n));
    total.carry=stitch(a.data(),n,subtract,total.carry,high);
    total.nonzero|=bits::first_nonzero(a.data(),n)!=n;check_bound(p,a,first);
    const uint64_t checksum=m61::words(a.data(),n);add_residue(total.rr,at_word(checksum,first));
    const size_t lo=std::max(first,p.first_output_limb),hi=std::min(first+n,p.end_output_limb);
    if(lo<hi){const auto selected=a.subspan(lo-first,hi-lo);
        total.extent.observe(lo-p.first_output_limb,{reinterpret_cast<const uint8_t*>(selected.data()),selected.size()*8},false);
        add_residue(total.extent.residue_m61,at_word(lo==first&&hi==first+n?checksum:m61::words(selected.data(),selected.size()),lo-p.first_output_limb));}
}
void linear_compute(void*ptr,sbn3_team_scope*){
    auto&c=*static_cast<LinearPipeline*>(ptr);
    for(size_t j=0;j<c.jobs();++j){const unsigned slot=unsigned(j%2);if(!c.exchange.wait(slot,TwoSlotExchange::ready))return;
        const size_t first=j*c.words(),n=std::min(c.words(),c.p.result_limbs-first);
        compute_chunk(c.p,c.av(slot).first(n),c.bv(slot).first(n),first,c.subtract,c.swap,c.total);
        c.exchange.publish(slot,TwoSlotExchange::computed);
    }
}
void linear_pipeline(void*ptr,sbn3_team_scope*scope){auto&c=*static_cast<LinearPipeline*>(ptr);
    sbn3_team_scope group{scope->team,scope->first,c.p.workers,false,scope->epoch};
    sbn3_team_invoke2(&group,SBN3_PARALLEL_CHILDREN,1,linear_io,ptr,linear_compute,ptr);
}
}
bool limit_stored_integer(StoredInteger &v,size_t words) noexcept {
    if(!v.value||!v.extent_known||v.value.region().layout.logical_bytes%8)return false;
    auto source=v.source();if(!resolve_source_window(source.limbs,source))return false;
    const size_t end=v.first_limb+v.limbs();
    if(v.extent.end>source.storage_limbs||v.extent.end>end)return false;
    if(v.extent.end<=v.first_limb){v.word_count=0;v.first_limb=0;v.exponent2=0;v.negative=false;return true;}
    if(v.extent.first>=v.extent.end)return false;
    size_t first=std::max(v.first_limb,size_t(v.extent.first));
    if(words&&v.extent.end-first>words)first=size_t(v.extent.end)-words;
    const __int128 exponent=__int128(v.exponent2)+__int128(first-v.first_limb)*64;
    if(exponent>INT64_MAX)return false;
    v.exponent2=int64_t(exponent);v.first_limb=first;v.word_count=size_t(v.extent.end)-first;return true;
}
bool limit_resident_input(LinearInput&v,const WordExtent&extent,size_t words) noexcept {
    auto s=v.source;
    if(v.value||!v.resident.data()||v.resident.size()>SIZE_MAX/8||s.storage_limbs!=v.resident.size()||
       !resolve_source_window(v.limbs(),s)||s.bit_shift||s.low_zero_words||s.ones_complement||
       extent.end>s.storage_limbs||extent.end>s.first_limb+s.limbs)return false;
    if(extent.end<=s.first_limb){s.first_limb=s.limbs=s.lower_limb=s.upper_limb=0;v.source=s;v.exponent2=0;v.negative=false;return true;}
    if(extent.first>=extent.end)return false;
    size_t first=std::max(s.first_limb,size_t(extent.first));if(words&&extent.end-first>words)first=size_t(extent.end)-words;
    const __int128 exponent=__int128(v.exponent2)+__int128(first-s.first_limb)*64;if(exponent>INT64_MAX)return false;
    s.first_limb=s.lower_limb=first;s.upper_limb=size_t(extent.end);s.limbs=s.upper_limb-first;
    v.source=s;v.exponent2=int64_t(exponent);return true;
}
bool normalized_source(const StoredInteger&v,size_t words,SourceWindow&out) noexcept {
    if(!v.value||v.negative||!v.extent_known||!v.extent.top_word||!words||v.extent.end<=v.first_limb||
       v.extent.end>v.first_limb+v.limbs())return false;
    const size_t have=size_t(v.extent.end)-v.first_limb,zeros=words>have?words-have:0;
    out={size_t(v.value.region().layout.logical_bytes/8),zeros?v.first_limb:size_t(v.extent.end)-words,words,v.first_limb,size_t(v.extent.end),int(__builtin_clzll(v.extent.top_word)),zeros};
    return resolve_source_window(words,out);
}
bool query_stored_sum(size_t an,int64_t ae,size_t bn,int64_t be,StoredSumPlan &out,size_t precision,unsigned workers) noexcept {
    if(!workers||workers>32)return false;
    StoredSumPlan p{};p.an=an;p.bn=bn;p.ae=ae;p.be=be;
    p.exponent=an?(bn?std::min(ae,be):ae):(bn?be:0);
    p.precision=precision;
    if(precision && (an||bn)){
        if(an&&bn&&(__int128(ae)-be)%64)return false;
        __int128 high=an?__int128(ae)+__int128(an)*64:__int128(be)+__int128(bn)*64;
        if(bn)high=std::max(high,__int128(be)+__int128(bn)*64);
        const __int128 low=std::max(__int128(p.exponent),high-(__int128(precision)+1)*64);
        if(low<INT64_MIN||low>INT64_MAX)return false;p.exponent=int64_t(low);
    }
    auto span=[&](size_t limbs,int64_t exponent){
        const __int128 delta=__int128(exponent)-p.exponent;
        return limbs?std::max(__int128(0),__int128(limbs)+(delta>=0?(delta+63)/64:delta/64)):0;
    };
    const __int128 n=std::max(span(an,ae),span(bn,be))+1;
    if(n>SIZE_MAX || !mul_size(size_t(n),8,p.output_bytes))return false;p.result_limbs=size_t(n);
    if(!align_size(std::min(size_t(p.output_bytes),maximum_batch),chunk_bytes,p.transfer_bytes))return false;
    const size_t batch=p.transfer_bytes;
    p.workers=workers>=2&&p.result_limbs>batch/8?2:1;
    p.end_output_limb=p.result_limbs;
    RegionLayout result{};if(region_layout(p.output_bytes,0,result))return false;
    p.output_allocation=result.allocation_bytes;
    if(__builtin_add_overflow(result.data_bytes,result.index_bytes,&p.write_bytes))return false;
    size_t at=0;auto put=[&](size_t size,size_t alignment,size_t &offset){return align_size(at,alignment,offset)&&add_size(offset,size,at);};
    for(unsigned i=0;i<3;++i)if(!put(batch,4096,p.io[i])||!put(index_page_bytes,4096,p.index[i]))return false;
    if(!put(batch+16,4096,p.a)||!put(batch+16,4096,p.b)||!put(batch+32,64,p.raw)||
       (p.workers>1&&(!put(batch+16,4096,p.a2)||!put(batch+16,4096,p.b2)))||!align_size(at,4096,p.storage_bytes))return false;
    out=p;return true;
}
bool project_stored_sum(StoredSumPlan&p,product::Window window,product::MagnitudeBound bound) noexcept {
    if(p.precision||window.error_bits||window.origin>p.result_limbs||window.words>p.result_limbs-window.origin)return false;
    size_t bytes;if(!mul_size(window.words,8,bytes))return false;RegionLayout layout{};
    if(region_layout(bytes,0,layout))return false;
    p.first_output_limb=window.origin;p.end_output_limb=window.origin+window.words;p.output_bytes=bytes;
    p.output_allocation=layout.allocation_bytes;p.write_bytes=layout.data_bytes+layout.index_bytes;p.bound=bound;return true;
}
IoError run_stored_linear(const StoredSumPlan&p,Transport&t,ValueStore&store,
                      const LinearInput&x,const LinearInput&y,StoredInteger&out,
                      std::span<uint8_t>w,sbn3_team*team,sbn3_team_scope*scope) noexcept {
    OperationSpan trace(t,"stored_linear",p.an,p.bn);
    auto valid=[&](const LinearInput&v){auto source=v.source;
        if(v.resident.data())return !v.value&&v.resident.size()<=SIZE_MAX/8&&source.storage_limbs==v.resident.size()&&
            resolve_source_window(v.limbs(),source)&&!overlaps(v.resident.data(),v.resident.size()*8,w.data(),p.storage_bytes);
        return !v.value||(*v.value&&v.value->store()==&store&&!(v.value->region().layout.logical_bytes%8)&&
            source.storage_limbs==v.value->region().layout.logical_bytes/8&&resolve_source_window(v.limbs(),source));};
    if(!valid(x)||!valid(y)||out.value||x.limbs()!=p.an||y.limbs()!=p.bn||
       x.exponent2!=p.ae||y.exponent2!=p.be||
       !w.data()||(uintptr_t(w.data())&4095)||w.size()<p.storage_bytes||(team&&(scope||team->busy||team->width<p.workers))||
       (scope&&(!scope->team||scope->active||scope->width<p.workers)))
        return {IoCode::invalid,"stored sum binding",0,p.storage_bytes,w.size()};
    if(scope)require_scope_leader(scope);
    if(t.counters.write_submitted>t.write_limit||p.write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"stored sum write limit",0,p.write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    RegionLease result;if(auto e=store.reserve(p.output_bytes,RegionKind::value,result))return e;
    const size_t batch=p.transfer_bytes,words=batch/8;
    StoreBuffers io[3];for(unsigned j=0;j<3;++j)io[j]={w.subspan(p.io[j],batch),w.subspan(p.index[j],index_page_bytes)};
    ValueReader a,b;
    if(x.value)if(auto e=a.open(t,x.value->region(),io[0]))return e;
    if(y.value)if(auto e=b.open(t,y.value->region(),io[1]))return e;
    if(x.resident.data())if(auto e=a.open_memory({reinterpret_cast<const uint8_t *>(x.resident.data()),x.resident.size()*8}))return e;
    if(y.resident.data())if(auto e=b.open_memory({reinterpret_cast<const uint8_t *>(y.resident.data()),y.resident.size()*8}))return e;
    auto av=std::span(reinterpret_cast<uint64_t *>(w.data()+p.a),words+2);
    auto bv=std::span(reinterpret_cast<uint64_t *>(w.data()+p.b),words+2);
    const auto raw=w.subspan(p.raw,batch+32);
    auto read=[&](size_t first,size_t n)->IoError{
        if(t.cancel.test())return {IoCode::cancelled,"stored sum boundary"};
        if(auto e=window(a,x,p.exponent,first,n,av,raw))return e;
        return window(b,y,p.exponent,first,n,bv,raw);
    };
    const bool subtract=x.negative!=y.negative;bool swap=false,negative=x.negative;
    if(subtract){
        int comparison=0;
        for(size_t end=p.result_limbs;end && !comparison;){
            const size_t n=std::min(words,end),first=end-n;
            if(auto e=read(first,n))return e;
            for(size_t j=n;j-- >0;)if(av[j]!=bv[j]){comparison=av[j]<bv[j]?-1:1;break;}
            end=first;
        }
        swap=comparison<0;negative=comparison==0?false:(swap?y.negative:x.negative);
    }
    LinearTotals totals;auto&extent=totals.extent;
    const bool parallel=(team||scope)&&p.workers>1;ValueWriter writer;if(auto e=writer.open(t,result.region(),io[2],nullptr))return e;
    auto&carry=totals.carry;auto&rx=totals.rx;auto&ry=totals.ry;auto&rr=totals.rr;auto&nonzero=totals.nonzero;
    if(parallel){
        LinearPipeline call{p,t,a,b,x,y,writer,w,subtract,swap,totals};TeamWaitScope waits(scope?*scope->team:*team,10000);
        if(scope)linear_pipeline(&call,scope);else sbn3_team_run(team,linear_pipeline,&call);if(call.error)return call.error;
    }else for(size_t first=0;first<p.result_limbs;){
        const size_t n=std::min(words,p.result_limbs-first);
        if(auto e=read(first,n))return e;
        compute_chunk(p,av.first(n),bv.first(n),first,subtract,swap,totals);
        const size_t begin=std::max(first,p.first_output_limb),end=std::min(first+n,p.end_output_limb);
        if(begin<end)if(auto e=writer.append({reinterpret_cast<const uint8_t *>(av.data()+begin-first),(end-begin)*8}))return e;
        first+=n;
    }
    if(swap)std::swap(rx,ry);
    const uint64_t expected=subtract?(rx>=ry?rx-ry:rx+m61::modulus-ry):m61::reduce(__uint128_t(rx)+ry);
    require(!carry && rr==expected,SBN3_FATAL_MATH,"stored sum carry/m61");
    if(auto e=writer.seal())return e;
    if(auto e=result.publish(out.value))return e;
    const __int128 exponent=__int128(p.exponent)+__int128(p.first_output_limb)*64;
    require(exponent<=INT64_MAX,SBN3_FATAL_SIZE,"stored linear output exponent");
    out.exponent2=int64_t(exponent);out.negative=nonzero&&negative;out.extent=extent;out.extent_known=true;
    if(p.precision)require(limit_stored_integer(out,p.precision),SBN3_FATAL_MATH,"stored sum limit");return {};
}
IoError run_stored_sum(const StoredSumPlan&p,Transport&t,ValueStore&s,const StoredInteger&x,const StoredInteger&y,
                      StoredInteger&out,std::span<uint8_t>w) noexcept {
    if(!x.value||!y.value)return {IoCode::invalid,"stored sum input ownership"};
    return run_stored_linear(p,t,s,linear_input(x),linear_input(y),out,w);
}
} // namespace sbn::v3::external
