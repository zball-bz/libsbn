// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/carry_output.hpp"
#include "common/checked.hpp"
#include "common/m61.hpp"
#include "common/m64.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
namespace sbn::v3::external {
namespace {
static_assert(std::endian::native==std::endian::little);
bool zero(std::span<const uint8_t>b){for(uint8_t c:b)if(c)return false;return true;}
bool increment(product::CrtCarry&v){for(auto &word:v.limb)if(++word)return true;return false;}
}
bool query_carry_output(uint64_t count,uint64_t length,uint64_t output,CarryOutputPlan&out,bool cyclic,uint64_t stream) noexcept {
    if(!count||length<64||length%8||output%8)return false;
    CarryOutputPlan p{};p.segments=count;p.segment_bytes=length;p.output_bytes=output;
    if(__builtin_mul_overflow(count,length,&p.coefficient_bytes)||p.coefficient_bytes>INT64_MAX||output>INT64_MAX ||
       count>SIZE_MAX/sizeof(CarrySegment))return false;
    if(cyclic&&p.coefficient_bytes!=output)return false;p.cyclic=cyclic;
    if(stream&&(count%stream||length%chunk_bytes))return false;p.stream_segments=stream;
    p.visible_segments=std::min(count,output/length+(output%length!=0));
    p.slots_per_segment=length%chunk_bytes?2:1;
    p.extra_last_slot=p.slots_per_segment==1&&output<p.coefficient_bytes&&output%chunk_bytes;
    uint64_t slots;
    const uint64_t groups=stream?p.visible_segments/stream+(p.visible_segments%stream!=0):p.visible_segments;
    if(__builtin_mul_overflow(groups,uint64_t(p.slots_per_segment),&slots)||
       __builtin_add_overflow(slots,uint64_t(p.extra_last_slot),&slots)||slots>SIZE_MAX/chunk_bytes)return false;
    p.descriptor_bytes=size_t(count)*sizeof(CarrySegment);p.fragment_bytes=size_t(slots)*chunk_bytes;
    out=p;return true;
}
IoError CarryOutput::fail(IoError e) noexcept {if(!error_)error_=e;if(writer_)writer_->abandon();return error_;}
std::span<uint8_t> CarryOutput::fragment(uint64_t sid,unsigned slot) noexcept {
    require(sid<plan_.visible_segments&&slot<2,SBN3_FATAL_WORKSPACE,"carry fragment id");
    uint64_t index=(plan_.stream_segments?sid/plan_.stream_segments:sid)*plan_.slots_per_segment+slot;
    if(slot>=plan_.slots_per_segment){
        require(plan_.extra_last_slot&&sid+1==plan_.visible_segments,SBN3_FATAL_WORKSPACE,"carry last fragment");
        index=plan_.stream_segments?plan_.visible_segments/plan_.stream_segments+(plan_.visible_segments%plan_.stream_segments!=0):plan_.visible_segments;
    }
    return fragments_.subspan(size_t(index)*chunk_bytes,chunk_bytes);
}
IoError CarryOutput::open(const CarryOutputPlan&p,ChunkWriter&w,std::span<CarrySegment>d,
                          std::span<uint8_t>f,std::span<uint8_t>a) noexcept {
    CarryOutputPlan check{};
    if(!query_carry_output(p.segments,p.segment_bytes,p.output_bytes,check,p.cyclic,p.stream_segments)||p.descriptor_bytes!=check.descriptor_bytes ||
       p.fragment_bytes!=check.fragment_bytes || !w.region() ||w.logical_bytes()!=p.output_bytes||w.written_chunks()||
       d.size()<p.segments||f.size()<p.fragment_bytes||a.size()<chunk_bytes||!d.data()||!a.data() ||
       (p.fragment_bytes&&!f.data())||overlaps(d.data(),check.descriptor_bytes,f.data(),f.size())||
       overlaps(d.data(),check.descriptor_bytes,a.data(),a.size())||overlaps(f.data(),f.size(),a.data(),a.size()))
        return {IoCode::invalid,"carry output grants"};
    plan_=check;writer_=&w;segments_=d.first(size_t(p.segments));fragments_=f.first(p.fragment_bytes);assembly_=a.first(chunk_bytes);
    for(auto &s:segments_)s={};submitted_=position_=0;fill_=0;finished_=false;error_={};m61_=0;full_extent_={};complement_extent_={};
    middle_=gap_nonzero_=false;gap_first_=gap_end_=0;return {};
}
void CarryOutput::inspect_middle(uint64_t first,uint64_t end) noexcept {
    require(!submitted_&&plan_.cyclic&&first<=end&&end<=plan_.output_bytes&&!(first%8)&&!(end%8),SBN3_FATAL_ARGUMENT,"middle guard interval");
    middle_=true;gap_first_=first;gap_end_=end;
}
void CarryOutput::observe(uint64_t at,std::span<const uint8_t>bytes) noexcept {
    if(middle_){
        m64::add_at(m61_,bytes.data(),bytes.size()/8,at/8);
        const uint64_t lo=std::max(at,gap_first_),hi=std::min(at+bytes.size(),gap_end_);
        if(!gap_nonzero_&&lo<hi)gap_nonzero_=!zero(bytes.subspan(size_t(lo-at),size_t(hi-lo)));
    }else m61::add_at(m61_,bytes.data(),bytes.size()/8,at/8);
    full_extent_.observe(at/8,bytes,false);
    if(plan_.cyclic)complement_extent_.observe(at/8,bytes,false,true);
}
IoError CarryOutput::submit(uint64_t sid,std::span<uint8_t>raw,const product::CrtCarry&tail) noexcept {
    if(error_)return error_;
    if(!writer_||finished_||sid>=plan_.segments||raw.size()!=plan_.segment_bytes||!raw.data()||segments_[size_t(sid)].present ||
       overlaps(raw.data(),raw.size(),fragments_.data(),fragments_.size())||overlaps(raw.data(),raw.size(),segments_.data(),plan_.descriptor_bytes))
        return fail({IoCode::invalid,"carry segment submission",sid});
    auto &s=segments_[size_t(sid)];const uint64_t start=sid*plan_.segment_bytes,end=start+plan_.segment_bytes;
    if(plan_.stream_segments){
        const bool first=sid%plan_.stream_segments==0;
        if(!first&&!segments_[size_t(sid-1)].present)return fail({IoCode::invalid,"carry row order",sid});
        const CarrySegment*previous=first?nullptr:&segments_[size_t(sid-1)];
        // A non-0xff endpoint absorbs any incoming overflow. Its outgoing
        // carry is therefore its independent tail, even before that endpoint
        // itself can be finalized. At most one arbitrary fragment per row is
        // left unresolved; earlier all-0xff segments have no such fragments.
        if((sid==0&&!plan_.cyclic)||(previous&&(previous->resolved||previous->endpoint!=256))){
            const product::CrtCarry incoming=previous?previous->tail:product::CrtCarry{};uint64_t cy=0;
            for(unsigned j=0;j<8;++j){uint64_t word;std::memcpy(&word,raw.data()+8*j,8);
                const __uint128_t x=__uint128_t(word)+incoming.limb[j]+cy;word=uint64_t(x);cy=uint64_t(x>>64);
                std::memcpy(raw.data()+8*j,&word,8);}
            for(size_t j=64;cy&&j<raw.size();++j){const unsigned x=unsigned(raw[j])+unsigned(cy);raw[j]=uint8_t(x);cy=x>>8;}
            s.tail=tail;if(cy)require(increment(s.tail),SBN3_FATAL_MATH,"ordered carry width");
            if(end>plan_.output_bytes)require(zero(raw.subspan(size_t(start>=plan_.output_bytes?0:plan_.output_bytes-start))),
                SBN3_FATAL_MATH,"ordered carry output overflow");
            if(start<plan_.output_bytes){const auto bytes=raw.first(size_t(std::min(end,plan_.output_bytes)-start));
                if(auto e=writer_->write(start/chunk_bytes,bytes))return fail(e);observe(start,bytes);}
            s.present=s.resolved=true;++submitted_;return {};
        }
    }
    if(end>plan_.output_bytes){const size_t first=size_t(start>=plan_.output_bytes?0:plan_.output_bytes-start);
        require(zero(raw.subspan(first)),SBN3_FATAL_MATH,"nonzero independent segment beyond output");}
    s.tail=tail;std::memcpy(s.guard,raw.data(),64);
    size_t at=64;
    while(at+8<=raw.size()){uint64_t v;std::memcpy(&v,raw.data()+at,8);if(v!=UINT64_MAX)break;at+=8;}
    while(at<raw.size()&&raw[at]==255)++at;
    s.ff_bytes=at-64;
    if(at<raw.size())s.endpoint=raw[at++];
    uint64_t cursor=start+at,limit=std::min(end,plan_.output_bytes);
    auto keep=[&](unsigned slot,uint64_t begin,uint64_t stop){
        if(stop<=begin)return;
        auto target=fragment(sid,slot);const size_t bytes=size_t(stop-begin);
        require(bytes<=target.size(),SBN3_FATAL_WORKSPACE,"carry retained fragment");
        std::memcpy(target.data(),raw.data()+size_t(begin-start),bytes);s.fragments[slot]={begin,bytes};
    };
    if(cursor<limit){
        const uint64_t aligned=(cursor+chunk_bytes-1)&~uint64_t(chunk_bytes-1);
        const uint64_t edge=std::min(limit,aligned);keep(0,cursor,edge);cursor=edge;
        const uint64_t full=limit&~uint64_t(chunk_bytes-1);
        if(full>cursor){auto bytes=raw.subspan(size_t(cursor-start),size_t(full-cursor));
            if(auto e=writer_->write(cursor/chunk_bytes,bytes))return fail(e);
            observe(cursor,bytes);cursor=full;}
        keep(1,cursor,limit);
    }
    s.present=true;++submitted_;return {};
}
IoError CarryOutput::publish() noexcept {
    const uint64_t start=position_-fill_;
    if(auto e=writer_->write(start/chunk_bytes,assembly_.first(fill_)))return fail(e);
    observe(start,assembly_.first(fill_));
    fill_=0;return {};
}
IoError CarryOutput::append(uint64_t at,std::span<const uint8_t>data) noexcept {
    while(!data.empty()){
        if(at>=plan_.output_bytes){require(zero(data),SBN3_FATAL_MATH,"carry output overflow");return {};}
        if(at!=position_){require(!fill_&&at>=position_&&!(at%chunk_bytes)&&!(position_%chunk_bytes),SBN3_FATAL_MATH,"carry late gap");position_=at;}
        if(!fill_)require(!(position_%chunk_bytes),SBN3_FATAL_MATH,"carry chunk start");
        const size_t n=std::min({data.size(),chunk_bytes-fill_,size_t(plan_.output_bytes-at)});
        std::memcpy(assembly_.data()+fill_,data.data(),n);fill_+=n;position_+=n;at+=n;data=data.subspan(n);
        if(fill_==chunk_bytes||position_==plan_.output_bytes)if(auto e=publish())return e;
    }
    return {};
}
IoError CarryOutput::fill(uint64_t at,uint8_t value,uint64_t count) noexcept {
    while(count){
        if(at>=plan_.output_bytes){require(!value,SBN3_FATAL_MATH,"carry uniform run overflow");return {};}
        if(at!=position_){require(!fill_&&at>=position_&&!(at%chunk_bytes)&&!(position_%chunk_bytes),SBN3_FATAL_MATH,"carry fill gap");position_=at;}
        if(!fill_)require(!(position_%chunk_bytes),SBN3_FATAL_MATH,"carry fill chunk start");
        const size_t n=size_t(std::min({count,uint64_t(chunk_bytes-fill_),plan_.output_bytes-at}));
        std::memset(assembly_.data()+fill_,value,n);fill_+=n;position_+=n;at+=n;count-=n;
        if(fill_==chunk_bytes||position_==plan_.output_bytes)if(auto e=publish())return e;
    }
    return {};
}
IoError CarryOutput::finish() noexcept {
    if(error_)return error_;
    if(!writer_||finished_||submitted_!=plan_.segments)return fail({IoCode::incomplete,"carry segments incomplete",0,plan_.segments,submitted_});
    product::CrtCarry carry{};
    if(plan_.cyclic){
        // Each segment's final carry is its independent tail plus at most one.
        // The monotone composed map has image {F(0),F(0)+1}. Starting at zero
        // reaches a fixed point in at most three descriptor-only traversals.
        bool closed=false;
        for(unsigned pass=0;pass<3;++pass){auto next=carry;
            for(const auto&s:segments_){if(s.resolved){next=s.tail;continue;}uint64_t cy=0;
                for(unsigned j=0;j<8;++j){uint64_t word;std::memcpy(&word,s.guard+8*j,8);
                    cy=uint64_t((__uint128_t(word)+next.limb[j]+cy)>>64);}
                next=s.tail;if(s.endpoint==256&&cy)require(increment(next),SBN3_FATAL_MATH,"cyclic carry width");
            }
            if(!std::memcmp(next.limb,carry.limb,sizeof carry.limb)){closed=true;break;}carry=next;
        }
        require(closed,SBN3_FATAL_MATH,"cyclic carry fixed point");
    }
    const auto incoming=carry;
    for(uint64_t sid=0;sid<plan_.segments;++sid){auto &s=segments_[size_t(sid)];
        if(s.resolved){require(!fill_,SBN3_FATAL_MATH,"ordered carry edge alignment");carry=s.tail;continue;}uint64_t cy=0;
        for(unsigned j=0;j<8;++j){uint64_t old;std::memcpy(&old,s.guard+8*j,8);const __uint128_t sum=__uint128_t(old)+carry.limb[j]+cy;
            const uint64_t low=uint64_t(sum);std::memcpy(s.guard+8*j,&low,8);cy=uint64_t(sum>>64);}
        uint64_t at=sid*plan_.segment_bytes;
        if(auto e=append(at,s.guard))return e;at+=64;
        if(auto e=fill(at,cy?0:255,s.ff_bytes))return e;at+=s.ff_bytes;
        if(s.endpoint!=256){const uint8_t endpoint=uint8_t(s.endpoint+cy);if(auto e=append(at,{&endpoint,1}))return e;}
        for(unsigned k=0;k<2;++k)if(s.fragments[k].bytes){
            const auto &f=s.fragments[k];if(auto e=append(f.at,fragment(sid,k).first(f.bytes)))return e;}
        carry=s.tail;if(s.endpoint==256&&cy)require(increment(carry),SBN3_FATAL_MATH,"carry summary width");
    }
    const uint64_t end=plan_.coefficient_bytes;
    if(plan_.cyclic){
        require(!std::memcmp(carry.limb,incoming.limb,sizeof carry.limb)&&!fill_,SBN3_FATAL_MATH,"cyclic carry closure");
        finished_=true;return {};
    }
    if(auto e=append(end,{reinterpret_cast<const uint8_t *>(carry.limb),sizeof carry.limb}))return e;
    if(plan_.output_bytes>end && plan_.output_bytes-end>sizeof carry.limb)
        if(auto e=fill(end+sizeof carry.limb,0,plan_.output_bytes-end-sizeof carry.limb))return e;
    require(!fill_,SBN3_FATAL_MATH,"carry unfinished output chunk");finished_=true;return {};
}
} // namespace sbn::v3::external
