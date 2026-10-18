// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/scratch_plane.hpp"
#include "common/checked.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cstring>

namespace sbn::v3::external {
IoError plane_index_storage(const RegionLayout &l, PlaneIndexStorage &out) noexcept {
    RegionLayout expected{};
    if (auto e = region_layout(l.logical_bytes, l.data_offset, expected)) return e;
    if (!(expected == l) || l.index_bytes > SIZE_MAX) return {IoCode::invalid, "plane index layout"};
    const uint64_t pages = l.index_bytes/index_page_bytes;
    out = {size_t(l.index_bytes), size_t(pages/8+(pages%8 != 0)), index_page_bytes}; return {};
}
IoError PlaneChunkReader::read(uint64_t offset,std::span<uint8_t>data) noexcept {
    if(transport_&&transport_->read_batch&&data.size()>transport_->max_transfer){ReadRange range{offset,data};return read_many({&range,1});}
    if(!transport_||offset%chunk_bytes||data.size()%chunk_bytes||!data.data()||
       (uintptr_t(data.data())&(transport_->alignment-1))||offset>layout_.logical_bytes||data.size()>layout_.logical_bytes-offset||
       overlaps(data.data(),data.size(),image_.data(),image_.size())||overlaps(data.data(),data.size(),retained_.data(),retained_.size()))
        return {IoCode::invalid,"borrowed plane full chunks",offset};
    while(!data.empty()){
        if(transport_->cancel.test())return {IoCode::cancelled,"borrowed plane read",offset};
        const size_t bytes=std::min(data.size(),transport_->max_transfer/chunk_bytes*chunk_bytes);
        if(!bytes)return {IoCode::invalid,"borrowed plane transfer limit"};
        auto batch=data.first(bytes);const size_t held=offset<retained_.size()?std::min(bytes,retained_.size()-size_t(offset)):0;
        if(held)std::memcpy(batch.data(),retained_.data()+offset,held);
        if(held<bytes)if(auto e=read_exact(*transport_,layout_.data_offset+offset+held,batch.subspan(held)))return e;
        for(size_t j=0;j<bytes;j+=chunk_bytes){const uint64_t chunk=(offset+j)/chunk_bytes;
            const uint64_t expected=get64(image_.data()+(chunk/records_per_page)*index_page_bytes+index_header_bytes+8*(chunk%records_per_page));
            auto piece=batch.subspan(j,chunk_bytes);
            if(chunk_checksum(id_,chunk,piece)!=expected){
                if(offset+j<retained_.size())std::memcpy(piece.data(),retained_.data()+offset+j,chunk_bytes);
                else if(auto e=read_exact(*transport_,layout_.data_offset+offset+j,piece))return e;
                if(chunk_checksum(id_,chunk,piece)!=expected)return {IoCode::corrupt,"borrowed plane data chunk",layout_.data_offset+offset+j};
            }
        }
        offset+=bytes;data=data.subspan(bytes);
    }
    return {};
}
IoError PlaneChunkReader::read_many(std::span<const ReadRange> ranges) noexcept {
    if(!transport_||ranges.size()>io_batch_max)return {IoCode::invalid,"plane batch reader"};
    if(transport_->cancel.test())return {IoCode::cancelled,"plane batch cancellation"};
    if(!transport_->read_batch){for(const auto&r:ranges)if(auto e=read(r.offset,r.data))return e;return {};}
    for(size_t i=0;i<ranges.size();++i){const auto&r=ranges[i];
        if(r.offset%chunk_bytes||r.data.size()%chunk_bytes||!r.data.data()||
           (uintptr_t(r.data.data())&(transport_->alignment-1))||r.offset>layout_.logical_bytes||r.data.size()>layout_.logical_bytes-r.offset||
           overlaps(r.data.data(),r.data.size(),image_.data(),image_.size())||overlaps(r.data.data(),r.data.size(),retained_.data(),retained_.size()))
            return {IoCode::invalid,"plane batch chunks",r.offset};
        for(size_t k=0;k<i;++k)if(overlaps(r.data.data(),r.data.size(),ranges[k].data.data(),ranges[k].data.size()))return {IoCode::invalid,"plane batch alias"};
    }
    ReadRequest requests[io_batch_max]{};size_t used=0;const size_t limit=std::min(size_t(transport_->batch_limit),io_batch_max);
    const size_t transfer=transport_->max_transfer/chunk_bytes*chunk_bytes;if(!limit||!transfer)return {IoCode::invalid,"plane batch capacity"};
    auto flush=[&](){auto e=read_batch_exact(*transport_,{requests,used});used=0;return e;};
    for(const auto&r:ranges){
        const size_t held=r.offset<retained_.size()?std::min(r.data.size(),retained_.size()-size_t(r.offset)):0;
        if(held)std::memcpy(r.data.data(),retained_.data()+r.offset,held);
        for(size_t at=held;at<r.data.size();){const size_t n=std::min(transfer,r.data.size()-at);
            requests[used++]={layout_.data_offset+r.offset+at,r.data.subspan(at,n),{}};at+=n;
            if(used==limit)if(auto e=flush())return e;
        }
    }
    if(used)if(auto e=flush())return e;
    // All data requests are joined before integrity checks or slot publication.
    // The same one-reread policy as scalar IO applies to each bad chunk.
    for(const auto&r:ranges)for(size_t j=0;j<r.data.size();j+=chunk_bytes){
        const uint64_t chunk=(r.offset+j)/chunk_bytes;
        const uint64_t expected=get64(image_.data()+(chunk/records_per_page)*index_page_bytes+index_header_bytes+8*(chunk%records_per_page));
        auto piece=r.data.subspan(j,chunk_bytes);
        if(chunk_checksum(id_,chunk,piece)!=expected){
            if(r.offset+j<retained_.size())std::memcpy(piece.data(),retained_.data()+r.offset+j,chunk_bytes);
            else if(auto e=read_exact(*transport_,layout_.data_offset+r.offset+j,piece))return e;
            if(chunk_checksum(id_,chunk,piece)!=expected)return {IoCode::corrupt,"plane batch data chunk",layout_.data_offset+r.offset+j};
        }
    }
    return {};
}
IoError ScratchPlane::borrow_chunks(PlaneView v,Transport&t,PlaneChunkReader&out) const noexcept {
    if(error_)return error_;
    if(!region_||editing_||resident_busy_||v.owner_!=this||v.epoch_!=epoch_||
       t.context!=transport_->context||t.read!=transport_->read||t.alignment!=transport_->alignment||t.max_transfer!=transport_->max_transfer)
        return {IoCode::invalid,"plane read loan"};
    out.id_=region_->id;out.layout_=region_->layout;out.transport_=&t;out.image_=image_;out.retained_=retained_;return {};
}
IoError ScratchPlane::borrow_chunks(const PlaneUpdate&u,Transport&t,PlaneChunkReader&out) const noexcept {
    if(error_)return error_;
    if(!region_||!editing_||resident_busy_||u.owner_!=this||u.epoch_!=epoch_||
       t.context!=transport_->context||t.read!=transport_->read||t.alignment!=transport_->alignment||t.max_transfer!=transport_->max_transfer)
        return {IoCode::invalid,"plane update read loan"};
    out.id_=region_->id;out.layout_=region_->layout;out.transport_=&t;out.image_=image_;out.retained_=retained_;return {};
}
IoError ScratchPlane::fail(IoError e) noexcept {
    if (!error_) error_ = e;
    if (region_) region_->state = RegionState::invalid;
    return error_;
}
uint64_t ScratchPlane::expectation(uint64_t chunk) const noexcept {
    return get64(image_.data()+(chunk/records_per_page)*index_page_bytes+index_header_bytes+8*(chunk%records_per_page));
}
void ScratchPlane::record(uint64_t chunk, std::span<const uint8_t> data) noexcept {
    const uint64_t page = chunk/records_per_page;
    put64(image_.data()+page*index_page_bytes+index_header_bytes+8*(chunk%records_per_page), chunk_checksum(region_->id, chunk, data));
    dirty_[size_t(page/8)] |= uint8_t(1u << (page%8));
}
IoError ScratchPlane::take(Transport &t, Region &r, StoreBuffers io, std::span<uint8_t> image,
                           std::span<uint8_t> dirty) noexcept {
    return attach(t,r,io,image,dirty,false);
}
IoError ScratchPlane::attach(Transport &t,Region &r,StoreBuffers io,std::span<uint8_t> image,
                             std::span<uint8_t> dirty,bool created,std::span<uint8_t> retained) noexcept {
    if (region_ || epoch_ == UINT64_MAX || r.revision == UINT64_MAX || r.kind != RegionKind::scratch ||
        r.state != (created?RegionState::writing:RegionState::sealed))
        return {IoCode::invalid, "scratch exclusive take"};
    if (auto e = validate_store_region(t, r, io)) return e;
    if(!retained.empty()&&(!created||!retained.data()||(uintptr_t(retained.data())&(index_page_bytes-1))||retained.size()%chunk_bytes||
       retained.size()>r.layout.data_bytes||overlaps(retained.data(),retained.size(),image.data(),image.size())||
       overlaps(retained.data(),retained.size(),dirty.data(),dirty.size())||overlaps(retained.data(),retained.size(),io.data.data(),io.data.size())||
       overlaps(retained.data(),retained.size(),io.index.data(),io.index.size())))return {IoCode::invalid,"scratch retained prefix"};
    PlaneIndexStorage need{};
    if (auto e = plane_index_storage(r.layout, need)) return e;
    if (image.size() < need.image_bytes || dirty.size() < need.dirty_bytes ||
        (need.image_bytes && (!image.data() || (uintptr_t(image.data()) & (index_page_bytes-1)))) ||
        (need.dirty_bytes && !dirty.data()) || overlaps(image.data(), image.size(), dirty.data(), dirty.size()) ||
        overlaps(image.data(), image.size(), io.data.data(), io.data.size()) ||
        overlaps(image.data(), image.size(), io.index.data(), io.index.size()) ||
        overlaps(dirty.data(), dirty.size(), io.data.data(), io.data.size()) ||
        overlaps(dirty.data(), dirty.size(), io.index.data(), io.index.size()))
        return {IoCode::capacity, "scratch index storage", 0, need.image_bytes, image.size()};
    for (size_t at = 0; !created && at < need.image_bytes;) {
        const size_t n = std::min(need.image_bytes-at, t.max_transfer/index_page_bytes*index_page_bytes);
        if (auto e = read_exact(t, r.layout.index_offset+at, image.subspan(at,n))) return e;
        for (size_t j = 0; j < n; j += index_page_bytes) {
            auto page = image.subspan(at+j,index_page_bytes);
            const uint64_t number = (at+j)/index_page_bytes;
            if (!valid_index_page(r, number, page)) {
                if (auto e = read_exact(t, r.layout.index_offset+at+j, page)) return e;
                if (!valid_index_page(r, number, page)) return {IoCode::corrupt, "scratch initial index", r.layout.index_offset+at+j};
            }
        }
        at += n;
    }
    region_ = &r; transport_ = &t; io_ = io; image_ = image.first(need.image_bytes); dirty_ = dirty.first(need.dirty_bytes);retained_=retained;
    if (!dirty_.empty()) std::memset(dirty_.data(), created?0xff:0, dirty_.size());
    editing_ = false; error_ = {}; ++r.revision; r.state = RegionState::scratch_owned;
    return {};
}
IoError ScratchPlane::view(PlaneView &out) const noexcept {
    if (error_) return error_;
    if (!region_ || editing_) return {IoCode::stale, "scratch read view"};
    out.owner_ = this; out.epoch_ = epoch_; return {};
}
IoError ScratchPlane::read_chunks(uint64_t first, std::span<uint8_t> data) noexcept {
    if(transport_->cancel.test())return fail({IoCode::cancelled,"scratch retained read"});
    const uint64_t byte=first*chunk_bytes;
    const size_t cached=byte<retained_.size()?std::min(data.size(),retained_.size()-size_t(byte)):0;
    if(cached)std::memcpy(data.data(),retained_.data()+byte,cached);
    if(cached<data.size())if(auto e=read_exact(*transport_,region_->layout.data_offset+byte+cached,data.subspan(cached)))return fail(e);
    for (size_t j = 0; j < data.size()/chunk_bytes; ++j) {
        const uint64_t chunk = first+j;
        const size_t valid = size_t(std::min(uint64_t(chunk_bytes), region_->layout.logical_bytes-chunk*chunk_bytes));
        auto bytes = data.subspan(j*chunk_bytes,chunk_bytes);
        if (chunk_checksum(region_->id,chunk,bytes.first(valid)) != expectation(chunk)) {
            // RAM expectations are authoritative: never reload a stale disk index.
            if(chunk*chunk_bytes<retained_.size())std::memcpy(bytes.data(),retained_.data()+chunk*chunk_bytes,chunk_bytes);
            else if (auto e = read_exact(*transport_, region_->layout.data_offset+chunk*chunk_bytes, bytes)) return fail(e);
            if (chunk_checksum(region_->id,chunk,bytes.first(valid)) != expectation(chunk))
                return fail({IoCode::corrupt,"scratch data chunk",region_->layout.data_offset+chunk*chunk_bytes});
        }
    }
    return {};
}
IoError ScratchPlane::read_current(uint64_t offset, std::span<uint8_t> out) noexcept {
    if (offset > region_->layout.logical_bytes || out.size() > region_->layout.logical_bytes-offset ||
        (!out.empty() && !out.data()) || overlaps(out.data(),out.size(),io_.data.data(),io_.data.size()) ||
        overlaps(out.data(),out.size(),io_.index.data(),io_.index.size()) ||
        overlaps(out.data(),out.size(),image_.data(),image_.size()) || overlaps(out.data(),out.size(),dirty_.data(),dirty_.size()))
        return {IoCode::invalid,"scratch read range",offset};
    if(overlaps(out.data(),out.size(),retained_.data(),retained_.size()))return {IoCode::invalid,"scratch retained read alias"};
    while (!out.empty()) {
        const uint64_t first = offset/chunk_bytes;
        const size_t skip = size_t(offset%chunk_bytes);
        if(!skip && out.size()>=chunk_bytes && !(uintptr_t(out.data())&(transport_->alignment-1))){
            const size_t count=std::min(out.size()/chunk_bytes,std::min(io_.data.size(),transport_->max_transfer)/chunk_bytes);
            auto data=out.first(count*chunk_bytes);
            if(auto e=read_chunks(first,data))return e;
            out=out.subspan(data.size());offset+=data.size();continue;
        }
        const size_t wanted = out.size()/chunk_bytes+(out.size()%chunk_bytes+skip+chunk_bytes-1)/chunk_bytes;
        const size_t count = size_t(std::min(uint64_t(std::min(wanted,std::min(io_.data.size(),transport_->max_transfer)/chunk_bytes)),region_->layout.chunks-first));
        auto data = io_.data.first(count*chunk_bytes);
        if (auto e = read_chunks(first,data)) return e;
        const size_t n = std::min(out.size(),data.size()-skip);
        std::memcpy(out.data(),data.data()+skip,n); out=out.subspan(n); offset+=n;
    }
    return {};
}
IoError ScratchPlane::read(PlaneView v, uint64_t offset, std::span<uint8_t> out) noexcept {
    if (error_) return error_;
    if (!region_ || editing_ || v.owner_ != this || v.epoch_ != epoch_) return {IoCode::stale,"scratch view epoch"};
    return read_current(offset,out);
}
IoError ScratchPlane::read_many(PlaneView v,std::span<const ReadRange> ranges) noexcept {
    if(error_)return error_;
    if(!region_||editing_||v.owner_!=this||v.epoch_!=epoch_)return {IoCode::stale,"scratch batch view"};
    bool whole=transport_->read_batch!=nullptr;
    for(const auto&r:ranges){
        if(overlaps(r.data.data(),r.data.size(),io_.data.data(),io_.data.size())||overlaps(r.data.data(),r.data.size(),io_.index.data(),io_.index.size())||
           overlaps(r.data.data(),r.data.size(),dirty_.data(),dirty_.size()))return {IoCode::invalid,"scratch batch workspace alias"};
        whole&=r.offset%chunk_bytes==0&&r.data.size()%chunk_bytes==0&&!(uintptr_t(r.data.data())&(transport_->alignment-1));
    }
    if(!whole){for(const auto&r:ranges)if(auto e=read(v,r.offset,r.data))return e;return {};}
    PlaneChunkReader reader;if(auto e=borrow_chunks(v,*transport_,reader))return e;
    if(auto e=reader.read_many(ranges))return fail(e);return {};
}
IoError ScratchPlane::begin_update(PlaneUpdate &u) noexcept {
    if (error_) return error_;
    if (!region_ || editing_ || u.owner_ || epoch_ == UINT64_MAX) return {IoCode::invalid,"scratch update ownership"};
    ++epoch_; editing_=true; u.owner_=this; u.epoch_=epoch_; return {};
}
IoError ScratchPlane::read(const PlaneUpdate &u, uint64_t offset, std::span<uint8_t> out) noexcept {
    if (error_) return error_;
    if (!region_ || !editing_ || resident_busy_ || u.owner_!=this || u.epoch_!=epoch_) return {IoCode::stale,"scratch update read"};
    return read_current(offset,out);
}
IoError ScratchPlane::patch(const PlaneUpdate &u, uint64_t offset, std::span<const uint8_t> input) noexcept {
    if (error_) return error_;
    if (!region_ || !editing_ || resident_busy_ || u.owner_!=this || u.epoch_!=epoch_) return {IoCode::stale,"scratch update write"};
    if (offset > region_->layout.logical_bytes || input.size() > region_->layout.logical_bytes-offset ||
        (!input.empty() && !input.data()) || overlaps(input.data(),input.size(),io_.data.data(),io_.data.size()) ||
        overlaps(input.data(),input.size(),io_.index.data(),io_.index.size()) ||
        overlaps(input.data(),input.size(),image_.data(),image_.size()) || overlaps(input.data(),input.size(),dirty_.data(),dirty_.size()))
        return {IoCode::invalid,"scratch patch range",offset};
    if(overlaps(input.data(),input.size(),retained_.data(),retained_.size()))return {IoCode::invalid,"scratch retained patch alias"};
    while (!input.empty()) {
        if(transport_->cancel.test())return fail({IoCode::cancelled,"scratch retained patch"});
        const uint64_t chunk = offset/chunk_bytes;
        const size_t skip = size_t(offset%chunk_bytes);
        const size_t valid = size_t(std::min(uint64_t(chunk_bytes),region_->layout.logical_bytes-chunk*chunk_bytes));
        const size_t unit=std::min(io_.data.size(),transport_->max_transfer)/chunk_bytes*chunk_bytes;
        if(transport_->write_batch&&!skip&&offset>=retained_.size()&&input.size()>unit&&
           !(uintptr_t(input.data())&(transport_->alignment-1))){
            WriteRequest requests[io_batch_max];const size_t limit=std::min(size_t(transport_->batch_limit),io_batch_max);
            if(!unit||!limit)return fail({IoCode::invalid,"scratch write batch capacity"});
            size_t n=0,bytes=0;const size_t full=input.size()/chunk_bytes*chunk_bytes;
            while(n<limit&&bytes<full){const size_t count=std::min(unit,full-bytes);
                requests[n++]={region_->layout.data_offset+offset+bytes,input.subspan(bytes,count),{}};bytes+=count;}
            if(auto e=write_batch_exact(*transport_,{requests,n}))return fail(e);
            for(size_t j=0;j<bytes/chunk_bytes;++j)record(chunk+j,input.subspan(j*chunk_bytes,chunk_bytes));
            offset+=bytes;input=input.subspan(bytes);continue;
        }
        // A complete chunk replacement needs neither its old bytes nor a copy
        // through io_.data. A retained destination likewise needs just one copy.
        if(!skip && input.size()>=chunk_bytes &&
           (offset<retained_.size() || !(uintptr_t(input.data())&(transport_->alignment-1)))){
            size_t chunks=std::min(input.size()/chunk_bytes,std::min(io_.data.size(),transport_->max_transfer)/chunk_bytes);
            if(offset<retained_.size())chunks=std::min(chunks,(retained_.size()-size_t(offset))/chunk_bytes);
            auto data=input.first(chunks*chunk_bytes);
            if(offset<retained_.size())std::memcpy(retained_.data()+offset,data.data(),data.size());
            else if(auto e=write_exact(*transport_,region_->layout.data_offset+offset,data))return fail(e);
            for(size_t j=0;j<chunks;++j)record(chunk+j,data.subspan(j*chunk_bytes,chunk_bytes));
            offset+=data.size();input=input.subspan(data.size());continue;
        }
        size_t written, chunks;
        if (skip || input.size()<valid) {
            auto buffer = io_.data.first(chunk_bytes);
            if (auto e = read_chunks(chunk,buffer)) return e;
            written=std::min(input.size(),valid-skip); chunks=1;
            std::memcpy(buffer.data()+skip,input.data(),written);
        } else {
            const size_t capacity=std::min(io_.data.size(),transport_->max_transfer)/chunk_bytes;
            chunks=std::min(capacity,input.size()/chunk_bytes);
            if(chunk<retained_.size()/chunk_bytes)chunks=std::min(chunks,retained_.size()/chunk_bytes-size_t(chunk));
            if (!chunks) chunks=1; // complete replacement of the final partial chunk
            written=std::min(input.size(),chunks*chunk_bytes);
            std::memcpy(io_.data.data(),input.data(),written);
        }
        const uint64_t last = chunk+chunks-1;
        const size_t final_valid=size_t(std::min(uint64_t(chunk_bytes),region_->layout.logical_bytes-last*chunk_bytes));
        if(final_valid<chunk_bytes)std::memset(io_.data.data()+(chunks-1)*chunk_bytes+final_valid,0,chunk_bytes-final_valid);
        if(chunk*chunk_bytes<retained_.size())std::memcpy(retained_.data()+chunk*chunk_bytes,io_.data.data(),chunks*chunk_bytes);
        else if (auto e = write_exact(*transport_,region_->layout.data_offset+chunk*chunk_bytes,io_.data.first(chunks*chunk_bytes))) return fail(e);
        for(size_t j=0;j<chunks;++j){
            const size_t len=j+1==chunks?final_valid:chunk_bytes;
            record(chunk+j,io_.data.subspan(j*chunk_bytes,len));
        }
        offset+=written; input=input.subspan(written);
    }
    return {};
}
struct ScratchPlane::ResidentCall {
    ScratchPlane *plane;
    ResidentAction action;
    void *argument;
    uint64_t offset=0,first=0,last=0;
    size_t bytes=0;
    uint32_t corrupt=0;
    static void verify(void*ptr,uint64_t lo,uint64_t hi,unsigned){
        auto&c=*static_cast<ResidentCall *>(ptr);auto&p=*c.plane;const auto&r=*p.region_;
        for(uint64_t chunk=lo;chunk<hi;++chunk){
            const size_t n=size_t(std::min(uint64_t(chunk_bytes),r.layout.logical_bytes-chunk*chunk_bytes));
            auto data=p.retained_.subspan(size_t(chunk)*chunk_bytes,n);
            const uint64_t expected=p.expectation(chunk);
            if(chunk_checksum(r.id,chunk,data)!=expected&&chunk_checksum(r.id,chunk,data)!=expected)
                __atomic_store_n(&c.corrupt,1,__ATOMIC_RELAXED);
        }
    }
    static void refresh(void*ptr,uint64_t lo,uint64_t hi,unsigned){
        auto&c=*static_cast<ResidentCall *>(ptr);auto&p=*c.plane;const auto&r=*p.region_;
        for(uint64_t chunk=lo;chunk<hi;++chunk){
            const size_t n=size_t(std::min(uint64_t(chunk_bytes),r.layout.logical_bytes-chunk*chunk_bytes));
            put64(p.image_.data()+(chunk/records_per_page)*index_page_bytes+index_header_bytes+8*(chunk%records_per_page),
                  chunk_checksum(r.id,chunk,p.retained_.subspan(size_t(chunk)*chunk_bytes,n)));
        }
    }
    static void run(void*ptr,sbn3_team_scope*scope){
        auto&c=*static_cast<ResidentCall *>(ptr);
        sbn3_team_for(scope,c.first,c.last,1,SBN3_STATIC,verify,&c);
        if(__atomic_load_n(&c.corrupt,__ATOMIC_RELAXED))return;
        c.action(c.argument,c.plane->retained_.subspan(size_t(c.offset),c.bytes),scope);
        sbn3_team_for(scope,c.first,c.last,1,SBN3_STATIC,refresh,&c);
    }
};
IoError ScratchPlane::edit_resident(PlaneUpdate&u,uint64_t offset,size_t bytes,sbn3_team&team,ResidentAction action,void*argument) noexcept {
    if(error_)return error_;
    if(!region_||!editing_||resident_busy_||u.owner_!=this||u.epoch_!=epoch_||!action||team.busy||
       !bytes||offset>retained_.size()||bytes>retained_.size()-offset||offset>region_->layout.logical_bytes||bytes>region_->layout.logical_bytes-offset)
        return {IoCode::invalid,"resident scratch update"};
    if(transport_->cancel.test())return fail({IoCode::cancelled,"resident scratch before compute"});
    resident_busy_=true;ResidentCall call{this,action,argument,offset,offset/chunk_bytes,(offset+bytes+chunk_bytes-1)/chunk_bytes,bytes};
    sbn3_team_run(&team,ResidentCall::run,&call);resident_busy_=false;
    if(call.corrupt)return fail({IoCode::corrupt,"resident scratch checksum"});
    // Worker stores touch distinct index words; the controller owns bitmap bits.
    const uint64_t first_page=call.first/records_per_page,last_page=(call.last-1)/records_per_page;
    for(uint64_t page=first_page;page<=last_page;++page)dirty_[size_t(page/8)]|=uint8_t(1u<<(page%8));
    return {};
}
IoError ScratchPlane::flush() noexcept {
    const size_t pages=image_.size()/index_page_bytes;
    auto dirty=[&](size_t p){return (dirty_[p/8]>>(p%8))&1;};
    for(size_t p=0;p<pages;){
        if(!dirty(p)){++p;continue;}
        size_t end=p+1;
        while(end<pages && dirty(end) && end-p<transport_->max_transfer/index_page_bytes)++end;
        for(size_t j=p;j<end;++j)finalize_index_page(*region_,j*records_per_page,image_.subspan(j*index_page_bytes,index_page_bytes));
        if(auto e=write_exact(*transport_,region_->layout.index_offset+p*index_page_bytes,image_.subspan(p*index_page_bytes,(end-p)*index_page_bytes)))return fail(e);
        for(size_t j=p;j<end;++j)dirty_[j/8]&=uint8_t(~(1u<<(j%8)));
        p=end;
    }
    return {};
}
IoError ScratchPlane::materialize() noexcept {
    for(size_t at=0;at<retained_.size();){
        const size_t n=std::min(retained_.size()-at,transport_->max_transfer/chunk_bytes*chunk_bytes);
        if(auto e=write_exact(*transport_,region_->layout.data_offset+at,retained_.subspan(at,n)))return fail(e);at+=n;
    }
    return flush();
}
IoError ScratchPlane::finish(PlaneUpdate &u, bool persist) noexcept {
    if(error_)return error_;
    if(!region_ || !editing_ || resident_busy_ || u.owner_!=this || u.epoch_!=epoch_)return {IoCode::stale,"scratch finish"};
    if(transport_->cancel.test())return fail({IoCode::cancelled,"scratch finish cancellation"});
    if(persist)if(auto e=materialize())return e;
    editing_=false;u.owner_=nullptr;u.epoch_=0;return {};
}
IoError ScratchPlane::release() noexcept {
    if(error_)return error_;
    if(!region_ || editing_)return {IoCode::invalid,"scratch release"};
    if(transport_->cancel.test())return fail({IoCode::cancelled,"scratch release cancellation"});
    if(auto e=materialize())return e;
    region_->state=RegionState::sealed;region_=nullptr;transport_=nullptr;image_={};dirty_={};retained_={};io_={};
    if(epoch_!=UINT64_MAX)++epoch_;return {};
}
void ScratchPlane::discard() noexcept {
    require(!resident_busy_,SBN3_FATAL_WORKSPACE,"discard during resident scratch callback");
    if(region_)region_->state=RegionState::invalid;
    region_=nullptr;transport_=nullptr;image_={};dirty_={};retained_={};io_={};editing_=false;
    if(epoch_!=UINT64_MAX)++epoch_;
}
bool ScratchPlane::overlaps_backing(const ScratchPlane &other) const noexcept {
    if(!region_ || !other.region_ || transport_!=other.transport_)return false;
    const auto &a=region_->layout,&b=other.region_->layout;
    return a.data_offset<=b.data_offset ? a.allocation_bytes>b.data_offset-a.data_offset
                                       : b.allocation_bytes>a.data_offset-b.data_offset;
}
} // namespace sbn::v3::external
