// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/chunk_writer.hpp"
#include "common/checked.hpp"
#include <algorithm>
#include <cstring>
namespace sbn::v3::external {
IoError chunk_writer_storage(const RegionLayout &l,ChunkWriterStorage &out) noexcept {
    PlaneIndexStorage index{};if(auto e=plane_index_storage(l,index))return e;
    const uint64_t bits=l.chunks/8+(l.chunks%8!=0);
    if(bits>SIZE_MAX)return {IoCode::overflow,"chunk coverage"};
    out={index.image_bytes,size_t(bits)};return {};
}
IoError ChunkWriter::fail(IoError e) noexcept {
    if(!error_)error_=e;
    if(region_ && region_->state==RegionState::writing)region_->state=RegionState::invalid;
    return error_;
}
IoError ChunkWriter::open(Transport&t,Region&r,StoreBuffers io,std::span<uint8_t> image,std::span<uint8_t> coverage,std::span<uint8_t> retained,WordExtent *extent) noexcept {
    if(region_ || r.state!=RegionState::building)return {IoCode::invalid,"chunk writer ownership"};
    if(auto e=validate_store_region(t,r,io))return e;
    if(extent&&r.layout.logical_bytes%8)return {IoCode::invalid,"chunk word extent requires limbs"};
    if(!retained.empty()&&(r.kind!=RegionKind::scratch||!retained.data()||(uintptr_t(retained.data())&(index_page_bytes-1))||
        retained.size()%chunk_bytes||retained.size()>r.layout.data_bytes||
        overlaps(retained.data(),retained.size(),image.data(),image.size())||overlaps(retained.data(),retained.size(),coverage.data(),coverage.size())||
        overlaps(retained.data(),retained.size(),io.data.data(),io.data.size())||overlaps(retained.data(),retained.size(),io.index.data(),io.index.size())))
        return {IoCode::invalid,"chunk retained prefix"};
    ChunkWriterStorage need{};if(auto e=chunk_writer_storage(r.layout,need))return e;
    if(image.size()<need.index_bytes || coverage.size()<need.coverage_bytes ||
       (need.index_bytes && (!image.data() || (uintptr_t(image.data())&(index_page_bytes-1)))) ||
       (need.coverage_bytes && !coverage.data()) || overlaps(image.data(),image.size(),coverage.data(),coverage.size()) ||
       overlaps(image.data(),image.size(),io.data.data(),io.data.size()) || overlaps(image.data(),image.size(),io.index.data(),io.index.size()) ||
       overlaps(coverage.data(),coverage.size(),io.data.data(),io.data.size()) || overlaps(coverage.data(),coverage.size(),io.index.data(),io.index.size()))
        return {IoCode::capacity,"chunk writer metadata",0,need.index_bytes,image.size()};
    transport_=&t;region_=&r;io_=io;image_=image.first(need.index_bytes);coverage_=coverage.first(need.coverage_bytes);retained_=retained;
    extent_=extent;if(extent_)*extent_={};
    logical_bytes_=r.layout.logical_bytes;first_byte_=0;dispatched_=false;
    written_=0;error_={};if(!image_.empty())std::memset(image_.data(),0,image_.size());
    if(!coverage_.empty())std::memset(coverage_.data(),0,coverage_.size());r.state=RegionState::writing;return {};
}
IoError ChunkWriter::project(uint64_t logical,uint64_t first) noexcept {
    if(!region_||region_->state!=RegionState::writing||region_->kind!=RegionKind::value||dispatched_||
       !retained_.empty()||logical>INT64_MAX||first%chunk_bytes||first>logical||region_->layout.logical_bytes>logical-first)
        return {IoCode::invalid,"chunk output projection"};
    logical_bytes_=logical;first_byte_=first;return {};
}
IoError ChunkWriter::write(uint64_t first,std::span<const uint8_t> input) noexcept {
    if(error_)return error_;
    if(!region_ || region_->state!=RegionState::writing)return {IoCode::invalid,"chunk writer state"};
    if(input.empty())return {};
    const uint64_t logical_chunks=logical_bytes_/chunk_bytes+(logical_bytes_%chunk_bytes!=0);
    if(!input.data() || first>=logical_chunks)return fail({IoCode::invalid,"chunk writer range"});
    const uint64_t remaining=logical_bytes_-first*chunk_bytes;
    if(input.size()>remaining || (input.size()%chunk_bytes && input.size()!=remaining) ||
       overlaps(input.data(),input.size(),image_.data(),image_.size()) || overlaps(input.data(),input.size(),coverage_.data(),coverage_.size()) ||
       overlaps(input.data(),input.size(),io_.data.data(),io_.data.size()) || overlaps(input.data(),input.size(),io_.index.data(),io_.index.size()))
        return fail({IoCode::invalid,"chunk writer finalized range",first*chunk_bytes,input.size(),remaining});
    if(overlaps(input.data(),input.size(),retained_.data(),retained_.size()))return fail({IoCode::invalid,"chunk retained source alias"});
    if(transport_->cancel.test())return fail({IoCode::cancelled,"projected output cancellation"});
    dispatched_=true;
    const uint64_t begin=first*chunk_bytes,stop=begin+input.size();
    const uint64_t kept_begin=std::max(begin,first_byte_),kept_end=std::min(stop,first_byte_+region_->layout.logical_bytes);
    if(kept_begin>=kept_end)return {};
    input=input.subspan(size_t(kept_begin-begin),size_t(kept_end-kept_begin));
    first=(kept_begin-first_byte_)/chunk_bytes;
    const size_t total=input.size()/chunk_bytes+(input.size()%chunk_bytes!=0);
    for(size_t j=0;j<total;++j){const uint64_t k=first+j;
        if((coverage_[size_t(k/8)]>>(k%8))&1)return fail({IoCode::invalid,"chunk written twice",k});}
    if(transport_->write_batch&&first_byte_==0&&logical_bytes_==region_->layout.logical_bytes&&
       input.size()>transport_->max_transfer&&input.size()%chunk_bytes==0&&!(uintptr_t(input.data())&(transport_->alignment-1))){
        ChunkWrite range{this,first,input};return write_many({&range,1});}
    while(!input.empty()){
        if(transport_->cancel.test())return fail({IoCode::cancelled,"chunk retained cancellation"});
        const size_t cached=retained_.size()/chunk_bytes;
        const size_t limit=first<cached?size_t(cached-first):SIZE_MAX;
        size_t chunks=std::min(input.size()/chunk_bytes,transport_->max_transfer/chunk_bytes);
        chunks=std::min(chunks,limit);
        const bool direct=chunks && !(uintptr_t(input.data())&(transport_->alignment-1));
        std::span<const uint8_t> bytes;
        size_t valid;
        if(direct){valid=chunks*chunk_bytes;bytes=input.first(valid);}
        else{
            chunks=std::min(std::min(io_.data.size(),transport_->max_transfer)/chunk_bytes,
                            input.size()/chunk_bytes+(input.size()%chunk_bytes!=0));
            chunks=std::min(chunks,limit);
            valid=std::min(input.size(),chunks*chunk_bytes);
            std::memcpy(io_.data.data(),input.data(),valid);
            std::memset(io_.data.data()+valid,0,chunks*chunk_bytes-valid);bytes=io_.data.first(chunks*chunk_bytes);
        }
        if(first<cached)std::memcpy(retained_.data()+first*chunk_bytes,bytes.data(),bytes.size());
        else if(auto e=write_exact(*transport_,region_->layout.data_offset+first*chunk_bytes,bytes))return fail(e);
        for(size_t j=0;j<chunks;++j){
            const uint64_t k=first+j;const size_t covered=size_t(std::min(uint64_t(chunk_bytes),region_->layout.logical_bytes-k*chunk_bytes));
            if(extent_)extent_->observe(k*(chunk_bytes/8),bytes.subspan(j*chunk_bytes,covered));
            put64(image_.data()+(k/records_per_page)*index_page_bytes+index_header_bytes+8*(k%records_per_page),
                  chunk_checksum(region_->id,k,bytes.subspan(j*chunk_bytes,covered)));
            coverage_[size_t(k/8)]|=uint8_t(1u<<(k%8));++written_;
        }
        first+=chunks;input=input.subspan(valid);
    }
    return {};
}
IoError ChunkWriter::write_many(std::span<const ChunkWrite> ranges) noexcept {
    if(ranges.empty())return {};
    if(ranges.size()>io_batch_max||!ranges[0].writer)return {IoCode::invalid,"chunk batch descriptors"};
    auto*t=ranges[0].writer->transport_;bool whole=t&&t->write_batch;
    if(!t)return {IoCode::invalid,"chunk batch unopened writer"};
    for(const auto&r:ranges){auto*w=r.writer;
        if(!w||w->transport_!=t)return {IoCode::invalid,"chunk batch transport"};
        whole&=w->region_&&w->first_byte_==0&&w->logical_bytes_==w->region_->layout.logical_bytes&&!r.data.empty()&&r.data.size()%chunk_bytes==0&&
            r.data.data()&&!(uintptr_t(r.data.data())&(t->alignment-1));
    }
    if(!whole){for(const auto&r:ranges)if(auto e=r.writer->write(r.first_chunk,r.data))return e;return {};}
    auto failure=[&](IoError e){for(const auto&r:ranges)r.writer->fail(e);return e;};
    for(size_t i=0;i<ranges.size();++i){const auto&r=ranges[i];auto&w=*r.writer;
        if(w.error_)return failure(w.error_);
        if(w.region_->state!=RegionState::writing||r.first_chunk>=w.region_->layout.chunks||
           r.data.size()>w.region_->layout.logical_bytes-r.first_chunk*chunk_bytes)return failure({IoCode::invalid,"chunk batch range"});
        for(size_t k=0;k<ranges.size();++k){const auto&s=ranges[k];const auto&other=*s.writer;
            if(overlaps(r.data.data(),r.data.size(),other.image_.data(),other.image_.size())||
               overlaps(r.data.data(),r.data.size(),other.coverage_.data(),other.coverage_.size())||
               overlaps(r.data.data(),r.data.size(),other.retained_.data(),other.retained_.size())||
               overlaps(r.data.data(),r.data.size(),other.io_.data.data(),other.io_.data.size())||
               overlaps(r.data.data(),r.data.size(),other.io_.index.data(),other.io_.index.size()))return failure({IoCode::invalid,"chunk batch workspace alias"});
            if(k<i&&r.writer==s.writer&&r.first_chunk<s.first_chunk+s.data.size()/chunk_bytes&&s.first_chunk<r.first_chunk+r.data.size()/chunk_bytes)
                return failure({IoCode::invalid,"chunk batch duplicate coverage"});
        }
        for(size_t k=0;k<r.data.size()/chunk_bytes;++k){const uint64_t c=r.first_chunk+k;
            if((w.coverage_[size_t(c/8)]>>(c%8))&1)return failure({IoCode::invalid,"chunk batch written twice",c});}
    }
    if(t->cancel.test())return failure({IoCode::cancelled,"chunk batch cancellation"});
    const size_t limit=std::min(size_t(t->batch_limit),io_batch_max),unit=t->max_transfer/chunk_bytes*chunk_bytes;
    if(!limit||!unit)return failure({IoCode::invalid,"chunk batch capacity"});
    WriteRequest requests[io_batch_max];size_t used=0;
    auto flush=[&](){auto e=write_batch_exact(*t,{requests,used});used=0;return e;};
    for(const auto&r:ranges){auto&w=*r.writer;w.dispatched_=true;const uint64_t start=r.first_chunk*chunk_bytes;
        const size_t held=start<w.retained_.size()?std::min(r.data.size(),w.retained_.size()-size_t(start)):0;
        if(held)std::memcpy(w.retained_.data()+start,r.data.data(),held);
        for(size_t at=held;at<r.data.size();){const size_t n=std::min(unit,r.data.size()-at);
            requests[used++]={w.region_->layout.data_offset+start+at,r.data.subspan(at,n),{}};at+=n;
            if(used==limit)if(auto e=flush())return failure(e);
        }
    }
    if(used)if(auto e=flush())return failure(e);
    for(const auto&r:ranges){auto&w=*r.writer;
        for(size_t j=0;j<r.data.size()/chunk_bytes;++j){const uint64_t k=r.first_chunk+j;const auto bytes=r.data.subspan(j*chunk_bytes,chunk_bytes);
            if(w.extent_)w.extent_->observe(k*(chunk_bytes/8),bytes);
            put64(w.image_.data()+(k/records_per_page)*index_page_bytes+index_header_bytes+8*(k%records_per_page),chunk_checksum(w.region_->id,k,bytes));
            w.coverage_[size_t(k/8)]|=uint8_t(1u<<(k%8));++w.written_;
        }
    }
    return {};
}
IoError ChunkWriter::complete() noexcept {
    if(error_)return error_;
    if(!region_ || region_->state!=RegionState::writing)return {IoCode::invalid,"chunk completion state"};
    if(transport_->cancel.test())return fail({IoCode::cancelled,"chunk completion cancellation"});
    if(written_!=region_->layout.chunks)return fail({IoCode::incomplete,"chunk coverage incomplete",0,region_->layout.chunks,written_});
    for(size_t at=0;at<image_.size();at+=index_page_bytes)
        finalize_index_page(*region_,(at/index_page_bytes)*records_per_page,image_.subspan(at,index_page_bytes));
    return {};
}
IoError ChunkWriter::seal() noexcept {
    if(!retained_.empty())return {IoCode::invalid,"retained scratch requires plane handoff"};
    if(auto e=complete())return e;
    for(size_t at=0;at<image_.size();){
        const size_t n=std::min(image_.size()-at,transport_->max_transfer/index_page_bytes*index_page_bytes);
        if(auto e=write_exact(*transport_,region_->layout.index_offset+at,image_.subspan(at,n)))return fail(e);at+=n;
    }
    region_->state=RegionState::sealed;region_=nullptr;return {};
}
IoError ChunkWriter::finish_plane(ScratchPlane&plane,std::span<uint8_t>dirty) noexcept {
    if(auto e=complete())return e;
    if(region_->kind!=RegionKind::scratch)return {IoCode::invalid,"chunk handoff requires scratch"};
    if(auto e=plane.attach(*transport_,*region_,io_,image_,dirty,true,retained_))return e;
    region_=nullptr;return {};
}
void ChunkWriter::abandon() noexcept {
    if(region_ && region_->state==RegionState::writing)region_->state=RegionState::invalid;
    region_=nullptr;
}
} // namespace sbn::v3::external
