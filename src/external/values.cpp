// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/values.hpp"
#include "common/checked.hpp"
#include <algorithm>
#include <bit>
#include <new>
#include <utility>
namespace sbn::v3::external {
bool query_value_store(size_t slots, ValueStoreStorage &out) noexcept {
    if (!slots || slots == SIZE_MAX) return false;
    ValueStoreStorage p{}; p.slots = slots;
    size_t at = 0, bytes;
    auto add = [&](size_t n, size_t width, size_t alignment, size_t &offset) {
        return mul_size(n, width, bytes) && align_size(at, alignment, offset) && add_size(offset, bytes, at);
    };
    if (!add(slots, sizeof(Region), alignof(Region), p.regions) ||
        !add(slots, sizeof(uint64_t), alignof(uint64_t), p.references) ||
        !add(slots+1, sizeof(ValueStore::Extent), alignof(ValueStore::Extent), p.extents) ||
        !align_size(at, p.alignment, p.bytes)) return false;
    out = p; return true;
}
bool query_slot_store(size_t slots,uint64_t limit,SlotStorePlan&out) noexcept {
    SlotStorePlan p{};if(!limit||!query_value_store(slots,p.metadata))return false;
    p.slot_shift=unsigned(std::bit_width(std::max(limit,extent_alignment)-1));
    if(p.slot_shift>=63)return false;p.slot_bytes=uint64_t(1)<<p.slot_shift;p.live_byte_limit=limit;
    if(__builtin_mul_overflow(p.slot_bytes,uint64_t(slots),&p.address_bytes)||p.address_bytes>INT64_MAX)return false;
    out=p;return true;
}
ValueStore::~ValueStore() { require(!usage_.live_regions, SBN3_FATAL_LIFETIME, "stored values outlive owner"); }
IoError ValueStore::open(uint64_t lo, uint64_t hi, uint64_t offset, uint64_t bytes,
                         std::span<uint8_t> memory, size_t slots) noexcept {
    ValueStoreStorage p{}; uint64_t end;
    if (regions_ || !(lo|hi) || !query_value_store(slots,p) || !bytes ||
        ((offset|bytes)%extent_alignment) || __builtin_add_overflow(offset,bytes,&end) || end>INT64_MAX ||
        !memory.data() || (uintptr_t(memory.data())&(p.alignment-1)) || memory.size()<p.bytes)
        return {IoCode::invalid,"value store backing/metadata",0,p.bytes,memory.size()};
    regions_=reinterpret_cast<Region *>(memory.data()+p.regions);
    references_=reinterpret_cast<uint64_t *>(memory.data()+p.references);
    free_=reinterpret_cast<Extent *>(memory.data()+p.extents);slots_=slots;
    for(size_t i=0;i<slots;++i){::new(regions_+i)Region{};::new(references_+i)uint64_t(0);}
    for(size_t i=0;i<=slots;++i)::new(free_+i)Extent{};
    free_[0]={offset,bytes};free_count_=1;capacity_=bytes;backing_offset_=offset;uuid_low_=lo;uuid_high_=hi;
    return {};
}
IoError ValueStore::open_slots(uint64_t lo,uint64_t hi,const SlotStorePlan&p,std::span<uint8_t>memory,const RegionLifecycle&lifecycle) noexcept {
    SlotStorePlan q{};
    if(!query_slot_store(p.metadata.slots,p.live_byte_limit,q)||p.slot_bytes!=q.slot_bytes||p.address_bytes!=q.address_bytes||
       p.slot_shift!=q.slot_shift||p.metadata.bytes!=q.metadata.bytes||bool(lifecycle.create)!=bool(lifecycle.retire))
        return {IoCode::invalid,"slotted value store layout"};
    // Share the checked metadata initializer; the temporary contiguous range
    // is discarded without any file/page allocation.
    if(auto e=open(lo,hi,0,q.slot_bytes,memory,q.metadata.slots))return e;
    slot_bytes_=q.slot_bytes;capacity_=q.live_byte_limit;free_count_=0;lifecycle_=lifecycle;return {};
}
Region &ValueStore::at(size_t slot) const noexcept {
    require(slot<slots_ && references_[slot],SBN3_FATAL_LIFETIME,"stored region reference");
    return regions_[slot];
}
IoError ValueStore::reserve(uint64_t bytes,RegionKind kind,RegionLease &out) noexcept {
    if(!regions_ || out || serial_==UINT64_MAX)return {IoCode::invalid,"value reservation ownership"};
    RegionLayout shape{};if(auto e=region_layout(bytes,0,shape))return e;
    size_t slot=0;while(slot<slots_ && references_[slot])++slot;
    if(slot==slots_)return {IoCode::capacity,"value metadata slots",0,slots_+1,slots_};
    size_t extent=0;uint64_t offset=0;
    if(slot_bytes_){
        if(shape.allocation_bytes>capacity_-usage_.live_bytes||shape.allocation_bytes>slot_bytes_)
            return {IoCode::capacity,"value live backing allowance",0,shape.allocation_bytes,capacity_-usage_.live_bytes};
        offset=uint64_t(slot)*slot_bytes_;
    }else{
        while(extent<free_count_ && free_[extent].bytes<shape.allocation_bytes)++extent;
        if(shape.allocation_bytes && extent==free_count_)return {IoCode::capacity,"value backing extent",0,shape.allocation_bytes,usage().largest_free};
        offset=shape.allocation_bytes?free_[extent].offset:backing_offset_;
    }
    if(auto e=region_layout(bytes,offset,shape))return e;
    const uint64_t serial=serial_+1;
    Region candidate;if(auto e=begin_region(candidate,{uuid_low_,uuid_high_,serial},serial,shape,kind))return e;
    serial_=serial; // A failed creation must not reuse its checksum incarnation.
    if(lifecycle_.create)if(auto e=lifecycle_.create(lifecycle_.context,candidate))return e;
    regions_[slot]=candidate;references_[slot]=1;
    if(shape.allocation_bytes&&!slot_bytes_){
        free_[extent].offset+=shape.allocation_bytes;free_[extent].bytes-=shape.allocation_bytes;
        if(!free_[extent].bytes){for(size_t i=extent+1;i<free_count_;++i)free_[i-1]=free_[i];--free_count_;}
    }
    usage_.live_bytes+=shape.allocation_bytes;usage_.peak_bytes=std::max(usage_.peak_bytes,usage_.live_bytes);
    ++usage_.live_regions;usage_.peak_regions=std::max(usage_.peak_regions,usage_.live_regions);
    out.owner_=this;out.slot_=slot;return {};
}
void ValueStore::retain(size_t slot) noexcept {
    require(at(slot).state==RegionState::sealed && references_[slot]!=UINT64_MAX,SBN3_FATAL_LIFETIME,"retain immutable value");
    ++references_[slot];
}
IoError ValueStore::import(const Region&source,StoredValue&out) noexcept {
    if(!slot_bytes_||!lifecycle_.attach||out||source.state!=RegionState::sealed||source.kind!=RegionKind::value||
       !source.id.incarnation||!source.index_instance)return {IoCode::invalid,"stored value import"};
    size_t slot=0;for(;slot<slots_&&references_[slot];++slot){}
    if(slot==slots_)return {IoCode::capacity,"import metadata slots",0,slots_+1,slots_};
    for(size_t i=0;i<slots_;++i)if(references_[i]&&regions_[i].id==source.id)return {IoCode::invalid,"duplicate imported identity"};
    Region r=source;if(auto e=region_layout(source.layout.logical_bytes,uint64_t(slot)*slot_bytes_,r.layout))return e;
    if(r.layout.allocation_bytes>capacity_-usage_.live_bytes||r.layout.allocation_bytes>slot_bytes_)
        return {IoCode::capacity,"import live byte allowance",0,r.layout.allocation_bytes,capacity_-usage_.live_bytes};
    if(auto e=lifecycle_.attach(lifecycle_.context,r))return e;
    regions_[slot]=r;references_[slot]=1;usage_.live_bytes+=r.layout.allocation_bytes;
    usage_.peak_bytes=std::max(usage_.peak_bytes,usage_.live_bytes);++usage_.live_regions;usage_.peak_regions=std::max(usage_.peak_regions,usage_.live_regions);
    if(r.id.uuid_low==uuid_low_&&r.id.uuid_high==uuid_high_)serial_=std::max(serial_,r.id.incarnation);
    out.owner_=this;out.slot_=slot;return {};
}
void ValueStore::drop(size_t slot) noexcept {
    auto &r=at(slot);if(--references_[slot])return;
    const Extent returned{r.layout.data_offset,r.layout.allocation_bytes};r.state=RegionState::invalid;
    usage_.live_bytes-=returned.bytes;--usage_.live_regions;
    if(lifecycle_.retire)lifecycle_.retire(lifecycle_.context,r);
    if(slot_bytes_)return;
    if(!returned.bytes)return;
    size_t at=0;while(at<free_count_ && free_[at].offset<returned.offset)++at;
    require(free_count_<=slots_,SBN3_FATAL_LIFETIME,"value free-extent bound");
    for(size_t i=free_count_;i>at;--i)free_[i]=free_[i-1];free_[at]=returned;++free_count_;
    if(at && free_[at-1].offset+free_[at-1].bytes==free_[at].offset){
        free_[at-1].bytes+=free_[at].bytes;
        for(size_t i=at+1;i<free_count_;++i)free_[i-1]=free_[i];--free_count_;--at;
    }
    if(at+1<free_count_ && free_[at].offset+free_[at].bytes==free_[at+1].offset){
        free_[at].bytes+=free_[at+1].bytes;
        for(size_t i=at+2;i<free_count_;++i)free_[i-1]=free_[i];--free_count_;
    }
}
ValueStoreUsage ValueStore::usage() const noexcept {
    auto u=usage_;u.free_bytes=capacity_-u.live_bytes;
    if(slot_bytes_){u.largest_free=available_regions()?std::min(slot_bytes_,u.free_bytes):0;return u;}
    for(size_t i=0;i<free_count_;++i)u.largest_free=std::max(u.largest_free,free_[i].bytes);
    return u;
}
RegionLease::RegionLease(RegionLease &&x) noexcept:owner_(std::exchange(x.owner_,nullptr)),slot_(x.slot_){}
RegionLease &RegionLease::operator=(RegionLease &&x) noexcept {
    if(this!=&x){reset();owner_=std::exchange(x.owner_,nullptr);slot_=x.slot_;}return *this;
}
RegionLease::~RegionLease(){reset();}
void RegionLease::reset() noexcept {if(owner_){auto *o=std::exchange(owner_,nullptr);o->drop(slot_);}}
Region &RegionLease::region() const noexcept {require(owner_,SBN3_FATAL_LIFETIME,"empty region reservation");return owner_->at(slot_);}
IoError RegionLease::publish(StoredValue &out) noexcept {
    if(!owner_ || out || region().kind!=RegionKind::value || region().state!=RegionState::sealed)
        return {IoCode::invalid,"publish immutable value"};
    if(owner_->lifecycle_.publish)owner_->lifecycle_.publish(owner_->lifecycle_.context,region());
    out.owner_=std::exchange(owner_,nullptr);out.slot_=slot_;return {};
}
StoredValue::StoredValue(StoredValue &&x) noexcept:owner_(std::exchange(x.owner_,nullptr)),slot_(x.slot_){}
StoredValue &StoredValue::operator=(StoredValue &&x) noexcept {
    if(this!=&x){reset();owner_=std::exchange(x.owner_,nullptr);slot_=x.slot_;}return *this;
}
StoredValue::~StoredValue(){reset();}
void StoredValue::reset() noexcept {if(owner_){auto *o=std::exchange(owner_,nullptr);o->drop(slot_);}}
const Region &StoredValue::region() const noexcept {
    require(owner_,SBN3_FATAL_LIFETIME,"empty stored value");const auto &r=owner_->at(slot_);
    require(r.state==RegionState::sealed && r.kind==RegionKind::value,SBN3_FATAL_LIFETIME,"unsealed stored value");return r;
}
void StoredValue::share(StoredValue &out) const noexcept {
    require(owner_ && !out,SBN3_FATAL_LIFETIME,"stored value share destination");owner_->retain(slot_);out.owner_=owner_;out.slot_=slot_;
}
bool StoredValue::unique() const noexcept {return owner_ && owner_->references_[slot_]==1;}
} // namespace sbn::v3::external
