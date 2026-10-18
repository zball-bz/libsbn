// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/store.hpp"
#include "common/checked.hpp"
#include "common/m61.hpp"
#include <algorithm>
#include <cstring>

namespace sbn::v3::external {
void WordExtent::observe(uint64_t at,std::span<const uint8_t> data,bool checksum,bool complement) noexcept {
    const size_t n=data.size()/8;size_t lo=0,hi=n;
    if(checksum){
        if(!complement)m61::add_at(residue_m61,data.data(),n,at);
        else{
            const uint64_t full=(uint64_t(1)<<((3*(n%61))%61))-1,r=m61::words(data.data(),n);
            const uint64_t value=full>=r?full-r:full+m61::modulus-r;
            residue_m61+=m61::multiply(value,uint64_t(1)<<((3*(at%61))%61));
            if(residue_m61>=m61::modulus)residue_m61-=m61::modulus;
        }
    }
    auto word=[&](size_t i){uint64_t v;std::memcpy(&v,data.data()+8*i,8);return complement?~v:v;};
    while(lo<hi&&!word(lo))++lo;
    if(lo==hi)return;
    while(!word(hi-1))--hi;
    first=std::min(first,at+lo);
    if(at+hi>end){end=at+hi;top_word=word(hi-1);}
}
namespace {
bool align64(uint64_t n, uint64_t alignment, uint64_t &out) {
    uint64_t value;
    if (__builtin_add_overflow(n, alignment-1, &value)) return false;
    out = value & ~(alignment-1); return true;
}
}
IoError validate_store_region(const Transport &t, const Region &r, StoreBuffers b) noexcept {
    RegionLayout expected{};
    if (auto error = region_layout(r.layout.logical_bytes, r.layout.data_offset, expected)) return error;
    if (!(expected == r.layout) || !r.id.incarnation || !r.index_instance)
        return {IoCode::invalid, "region descriptor"};
    if (!t.alignment || (t.alignment & (t.alignment-1)) || t.alignment > index_page_bytes ||
        t.max_transfer < chunk_bytes || b.data.size() < chunk_bytes || b.index.size() < index_page_bytes ||
        !b.data.data() || !b.index.data() || (uintptr_t(b.data.data()) & (t.alignment-1)) ||
        (uintptr_t(b.index.data()) & (t.alignment-1)) ||
        overlaps(b.data.data(), b.data.size(), b.index.data(), b.index.size()))
        return {IoCode::invalid, "Store IO buffers"};
    if (r.layout.data_offset > t.capacity || r.layout.allocation_bytes > t.capacity-r.layout.data_offset)
        return {IoCode::capacity, "Store region allocation", r.layout.data_offset, r.layout.allocation_bytes, t.capacity};
    return {};
}
void finalize_index_page(const Region &r, uint64_t first_record, std::span<uint8_t> page) noexcept {
    put64(page.data(), index_magic); put64(page.data()+8, r.index_instance);
    put64(page.data()+16, first_record); put64(page.data()+24, 0);
    put64(page.data()+24, metadata_checksum(page));
}
bool valid_index_page(const Region &r, uint64_t number, std::span<uint8_t> page) noexcept {
    const uint64_t expected = get64(page.data()+24);
    put64(page.data()+24, 0);
    const uint64_t actual = metadata_checksum(page);
    put64(page.data()+24, expected);
    return expected == actual && get64(page.data()) == index_magic &&
           get64(page.data()+8) == r.index_instance && get64(page.data()+16) == number*records_per_page;
}
namespace {
size_t covered(const Region &r, uint64_t chunk) {
    return size_t(std::min(uint64_t(chunk_bytes), r.layout.logical_bytes-chunk*chunk_bytes));
}
}
IoError region_layout(uint64_t logical, uint64_t offset, RegionLayout &out) noexcept {
    if (offset & (extent_alignment-1)) return {IoCode::invalid, "region extent alignment", offset};
    RegionLayout l{};
    l.logical_bytes = logical; l.data_offset = offset;
    l.chunks = logical/chunk_bytes + (logical%chunk_bytes != 0);
    if (__builtin_mul_overflow(l.chunks, uint64_t(chunk_bytes), &l.data_bytes)) return {IoCode::overflow, "region data"};
    uint64_t data_extent, index_extent, pages = l.chunks/records_per_page + (l.chunks%records_per_page != 0), end;
    if (!align64(l.data_bytes, extent_alignment, data_extent) ||
        __builtin_mul_overflow(pages, uint64_t(index_page_bytes), &l.index_bytes) ||
        !align64(l.index_bytes, extent_alignment, index_extent) ||
        __builtin_add_overflow(offset, data_extent, &l.index_offset) ||
        __builtin_add_overflow(data_extent, index_extent, &l.allocation_bytes) ||
        __builtin_add_overflow(offset, l.allocation_bytes, &end) || end > INT64_MAX)
        return {IoCode::overflow, "region layout"};
    out = l; return {};
}
IoError begin_region(Region &r, RegionId id, uint64_t index, const RegionLayout &l, RegionKind kind) noexcept {
    RegionLayout expected{};
    if (auto error = region_layout(l.logical_bytes, l.data_offset, expected)) return error;
    if (!(l == expected) || !id.incarnation || !index || (r.state != RegionState::empty && r.state != RegionState::invalid) ||
        (r.state != RegionState::empty && r.id == id)) return {IoCode::invalid, "region begin/reuse"};
    r = {id, index, l, RegionState::building, kind}; return {};
}
IoError ValueWriter::fail(IoError error) noexcept {
    if (!error_) error_ = error;
    if (region_ && region_->state == RegionState::writing) region_->state = RegionState::invalid;
    return error_;
}
size_t ValueWriter::batch_capacity() const noexcept {
    const size_t chunks = std::min({buffers_.data.size()/chunk_bytes, transport_->max_transfer/chunk_bytes,
                                   records_per_page-records_});
    return chunks*chunk_bytes;
}
IoError ValueWriter::open(Transport &t, Region &r, StoreBuffers b,WordExtent *extent) noexcept {
    if (region_ && region_->state == RegionState::writing) return {IoCode::invalid, "writer already active"};
    if (r.state != RegionState::building) return {IoCode::invalid, "writer requires building region"};
    if (auto e = validate_store_region(t, r, b)) return e;
    if(extent&&r.layout.logical_bytes%8)return {IoCode::invalid,"word extent requires integral limbs"};
    extent_=extent;if(extent_)*extent_={};
    transport_ = &t; region_ = &r; buffers_ = b;
    accepted_ = written_chunks_ = first_record_ = 0; buffered_ = records_ = 0; error_ = {};
    r.state = RegionState::writing;
    std::memset(buffers_.index.data(), 0, index_page_bytes);
    return {};
}
IoError ValueWriter::flush_index() noexcept {
    if (!records_) return {};
    auto page = buffers_.index.first(index_page_bytes);
    finalize_index_page(*region_, first_record_, page);
    const uint64_t at = region_->layout.index_offset + (first_record_/records_per_page)*index_page_bytes;
    if (auto e = write_exact(*transport_, at, page)) return fail(e);
    first_record_ += records_; records_ = 0;
    std::memset(page.data(), 0, page.size());
    return {};
}
IoError ValueWriter::flush_data(size_t n) noexcept {
    return write_data(buffers_.data.first(n));
}
IoError ValueWriter::write_data(std::span<const uint8_t> data) noexcept {
    const size_t n=data.size();
    const size_t chunks = n/chunk_bytes;
    if (n%chunk_bytes || chunks > records_per_page-records_ || written_chunks_+chunks > region_->layout.chunks)
        return fail({IoCode::invalid, "writer batch shape"});
    const uint64_t at = region_->layout.data_offset + written_chunks_*chunk_bytes;
    if (auto e = write_exact(*transport_, at, data)) return fail(e);
    for (size_t j = 0; j < chunks; ++j) {
        const uint64_t index = written_chunks_+j;
        const auto sum = chunk_checksum(region_->id, index, data.subspan(j*chunk_bytes, covered(*region_, index)));
        if(extent_)extent_->observe(index*(chunk_bytes/8),data.subspan(j*chunk_bytes,covered(*region_,index)));
        put64(buffers_.index.data()+index_header_bytes+8*records_++, sum);
    }
    written_chunks_ += chunks; buffered_ = 0;
    return records_ == records_per_page ? flush_index() : IoError{};
}
IoError ValueWriter::append(std::span<const uint8_t> input) noexcept {
    if (error_) return error_;
    if (!region_ || region_->state != RegionState::writing) return {IoCode::invalid, "writer state"};
    if (input.size() > region_->layout.logical_bytes-accepted_ || (!input.empty() && !input.data()))
        return fail({IoCode::invalid, "writer length", accepted_, input.size(), region_->layout.logical_bytes-accepted_});
    if (overlaps(input.data(), input.size(), buffers_.data.data(), buffers_.data.size()) ||
        overlaps(input.data(), input.size(), buffers_.index.data(), buffers_.index.size()))
        return fail({IoCode::invalid, "writer source/IO buffer alias"});
    while (!input.empty()) {
        if (transport_->cancel.test()) return fail({IoCode::cancelled, "writer cancellation"});
        if(!buffered_&&input.size()>=chunk_bytes&&!(uintptr_t(input.data())&(transport_->alignment-1))){
            const size_t chunks=std::min({input.size()/chunk_bytes,transport_->max_transfer/chunk_bytes,records_per_page-records_});
            const size_t bytes=chunks*chunk_bytes;accepted_+=bytes;
            if(auto e=write_data(input.first(bytes)))return e;input=input.subspan(bytes);continue;
        }
        const size_t cap = batch_capacity(), n = std::min(input.size(), cap-buffered_);
        std::memmove(buffers_.data.data()+buffered_, input.data(), n);
        buffered_ += n; accepted_ += n; input = input.subspan(n);
        if (buffered_ == cap) if (auto e = flush_data(cap)) return e;
    }
    return {};
}
IoError ValueWriter::seal() noexcept {
    if (error_) return error_;
    if (!region_ || region_->state != RegionState::writing) return {IoCode::invalid, "writer seal state"};
    if (transport_->cancel.test()) return fail({IoCode::cancelled, "writer seal cancellation"});
    if (accepted_ != region_->layout.logical_bytes)
        return fail({IoCode::incomplete, "writer incomplete value", accepted_, region_->layout.logical_bytes, accepted_});
    if (buffered_) {
        const size_t n = (buffered_+chunk_bytes-1)/chunk_bytes*chunk_bytes;
        std::memset(buffers_.data.data()+buffered_, 0, n-buffered_);
        if (auto e = flush_data(n)) return e;
    }
    if (auto e = flush_index()) return e;
    if (written_chunks_ != region_->layout.chunks) return fail({IoCode::incomplete, "writer chunk coverage"});
    region_->state = RegionState::sealed; return {};
}
void ValueWriter::abandon() noexcept {
    if (region_ && region_->state == RegionState::writing) region_->state = RegionState::invalid;
    region_ = nullptr; transport_ = nullptr; buffers_ = {};
}
IoError ValueReader::open(Transport &t, const Region &r, StoreBuffers b) noexcept {
    if (r.state != RegionState::sealed) return {IoCode::incomplete, "reader requires sealed value"};
    if (auto e = validate_store_region(t, r, b)) return e;
    transport_ = &t; region_ = &r; buffers_ = b; cached_page_ = UINT64_MAX;
    pinned_index_ = {};memory_={};
    opened_id_ = r.id; opened_index_ = r.index_instance; opened_revision_ = r.revision;
    return {};
}
IoError ValueReader::open_memory(std::span<const uint8_t>data) noexcept {
    if(!data.data())return {IoCode::invalid,"resident reader null storage"};
    memory_=data;transport_=nullptr;region_=nullptr;buffers_={};pinned_index_={};return {};
}
IoError ValueReader::open_pinned(Transport &t,const Region &r,StoreBuffers b,std::span<uint8_t> image) noexcept {
    if(image.size()<r.layout.index_bytes || (r.layout.index_bytes && (!image.data() || (uintptr_t(image.data())&(index_page_bytes-1)))) ||
       overlaps(image.data(),image.size(),b.data.data(),b.data.size()) || overlaps(image.data(),image.size(),b.index.data(),b.index.size()))
        return {IoCode::capacity,"pinned value index",0,r.layout.index_bytes,image.size()};
    if(auto e=open(t,r,b))return e;
    for(size_t at=0;at<r.layout.index_bytes;){
        const size_t n=std::min(size_t(r.layout.index_bytes)-at,t.max_transfer/index_page_bytes*index_page_bytes);
        if(auto e=read_exact(t,r.layout.index_offset+at,image.subspan(at,n))){region_=nullptr;return e;}
        for(size_t j=0;j<n;j+=index_page_bytes){
            auto page=image.subspan(at+j,index_page_bytes);const uint64_t number=(at+j)/index_page_bytes;
            if(!valid_index_page(r,number,page)){
                if(auto e=read_exact(t,r.layout.index_offset+at+j,page)){region_=nullptr;return e;}
                if(!valid_index_page(r,number,page)){region_=nullptr;return {IoCode::corrupt,"pinned value index page",r.layout.index_offset+at+j};}
            }
        }
        at+=n;
    }
    pinned_index_=image.first(size_t(r.layout.index_bytes));return {};
}
IoError ValueReader::index_page(uint64_t page_number, bool fresh) noexcept {
    if(!pinned_index_.empty())return {};
    if (!fresh && cached_page_ == page_number) return {};
    cached_page_ = UINT64_MAX;
    const uint64_t at = region_->layout.index_offset + page_number*index_page_bytes;
    auto page = buffers_.index.first(index_page_bytes);
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
        if (auto e = read_exact(*transport_, at, page)) return e;
        if (valid_index_page(*region_, page_number, page)) {
            cached_page_ = page_number; return {};
        }
    }
    return {IoCode::corrupt, "checksum index page", at};
}
IoError ValueReader::verify(uint64_t chunk, std::span<uint8_t> data) noexcept {
    const uint64_t page = chunk/records_per_page;
    if (auto e = index_page(page)) return e;
    const uint8_t *records=pinned_index_.empty()?buffers_.index.data():pinned_index_.data()+page*index_page_bytes;
    auto expected = get64(records+index_header_bytes+8*(chunk%records_per_page));
    if (chunk_checksum(region_->id, chunk, data.first(covered(*region_, chunk))) == expected) return {};
    // Reread this chunk and its expectation once. The caller sees no partial success.
    if (auto e = index_page(page, true)) return e;
    if (auto e = read_exact(*transport_, region_->layout.data_offset+chunk*chunk_bytes, data)) return e;
    expected = get64(records+index_header_bytes+8*(chunk%records_per_page));
    if (chunk_checksum(region_->id, chunk, data.first(covered(*region_, chunk))) == expected) return {};
    return {IoCode::corrupt, "data chunk", region_->layout.data_offset+chunk*chunk_bytes};
}
IoError ValueReader::read(uint64_t offset, std::span<uint8_t> output) noexcept {
    if(memory_.data()){
        if(offset>memory_.size()||output.size()>memory_.size()-size_t(offset)||(!output.empty()&&!output.data())||
           overlaps(output.data(),output.size(),memory_.data(),memory_.size()))return {IoCode::invalid,"resident reader range/alias",offset};
        if(!output.empty())std::memcpy(output.data(),memory_.data()+size_t(offset),output.size());return {};
    }
    if (!region_ || region_->state != RegionState::sealed) return {IoCode::incomplete, "reader state"};
    if (!(opened_id_ == region_->id) || opened_index_ != region_->index_instance || opened_revision_ != region_->revision)
        return {IoCode::stale, "reader version"};
    if (offset > region_->layout.logical_bytes || output.size() > region_->layout.logical_bytes-offset ||
        (!output.empty() && !output.data()) || overlaps(output.data(), output.size(), buffers_.data.data(), buffers_.data.size()) ||
        overlaps(output.data(), output.size(), buffers_.index.data(), buffers_.index.size()))
        return {IoCode::invalid, "reader range/buffer alias", offset, output.size(), region_->layout.logical_bytes};
    if(overlaps(output.data(),output.size(),pinned_index_.data(),pinned_index_.size()))return {IoCode::invalid,"reader pinned-index alias"};
    while (!output.empty()) {
        const uint64_t first = offset/chunk_bytes;
        const size_t skip = size_t(offset%chunk_bytes);
        const size_t capacity = std::min(buffers_.data.size(), transport_->max_transfer)/chunk_bytes;
        // Whole aligned chunks can be verified in the caller's discardable
        // destination. Keep bounce storage only for unreadable edge padding.
        if (!skip && output.size()>=chunk_bytes && !(uintptr_t(output.data())&(transport_->alignment-1))) {
            const size_t count=std::min(capacity,output.size()/chunk_bytes);
            auto data=output.first(count*chunk_bytes);
            if(auto e=read_exact(*transport_,region_->layout.data_offset+offset,data))return e;
            for(size_t j=0;j<count;++j)if(auto e=verify(first+j,data.subspan(j*chunk_bytes,chunk_bytes)))return e;
            output=output.subspan(data.size());offset+=data.size();continue;
        }
        const size_t wanted = output.size()/chunk_bytes + ((output.size()%chunk_bytes + skip + chunk_bytes-1)/chunk_bytes);
        const size_t count = size_t(std::min(uint64_t(std::min(capacity, wanted)), region_->layout.chunks-first));
        auto data = buffers_.data.first(count*chunk_bytes);
        if (auto e = read_exact(*transport_, region_->layout.data_offset+first*chunk_bytes, data)) return e;
        for (size_t j = 0; j < count; ++j) if (auto e = verify(first+j, data.subspan(j*chunk_bytes, chunk_bytes))) return e;
        const size_t n = std::min(output.size(), data.size()-skip);
        std::memcpy(output.data(), data.data()+skip, n);
        output = output.subspan(n); offset += n;
    }
    return {};
}
} // namespace sbn::v3::external
