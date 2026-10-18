// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/store.hpp"
#include "external/resident_product.hpp"
#include "common/m61.hpp"
#include <array>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <memory>
using namespace sbn::v3;
using namespace sbn::v3::external;

static void m61_checks(){
    Aligned bytes(8*8193+8);
    for(unsigned mode=0;mode<3;++mode){
        if(mode==0)fill(bytes.span());else std::memset(bytes.data,mode==1?0:255,bytes.bytes);
        for(size_t n:{size_t(0),size_t(1),size_t(7),size_t(8),size_t(9),size_t(63),size_t(64),size_t(65),
                      size_t(511),size_t(512),size_t(513),size_t(2048),size_t(8193)})for(size_t shift=0;shift<8;++shift){
            uint64_t expected=0;
            for(size_t i=n;i-- >0;){uint64_t x;std::memcpy(&x,bytes.data+shift+8*i,8);
                expected=uint64_t(((__uint128_t(expected)<<64)+x)%m61::modulus);}
            assert(m61::words_scalar(bytes.data+shift,n)==expected);
            assert(m61::words_bulk(bytes.data+shift,n)==expected);
            assert(m61::words(bytes.data+shift,n)==expected);
        }
    }
    puts("m61 bulk: independent Horner reference, every tail/alignment, zero/all-ones/random PASS");
}

static void checksum_checks() {
    assert(metadata_checksum({}) == 0x2d06800538d394c2ULL);
    std::array<uint8_t, 48+91> raw{};
    RegionId id{31, 47, 3};
    put64(raw.data(), 0x314b48434e4253ULL); put64(raw.data()+8, id.uuid_low); put64(raw.data()+16, id.uuid_high);
    put64(raw.data()+24, id.incarnation); put64(raw.data()+32, 17); put64(raw.data()+40, 91);
    fill(std::span(raw).subspan(48));
    const auto value = chunk_checksum(id, 17, std::span(raw).subspan(48));
    assert(value == metadata_checksum(raw));
    assert(value != chunk_checksum({31,47,4}, 17, std::span(raw).subspan(48)));
    assert(value != chunk_checksum(id, 18, std::span(raw).subspan(48)));
    assert(value != chunk_checksum(id, 17, std::span(raw).subspan(48,90)));
    RegionLayout l{}; assert(region_layout(UINT64_MAX, 0, l).code == IoCode::overflow);
    assert(region_layout(17, 3, l).code == IoCode::invalid);
}
static void roundtrip(size_t n) {
    auto r = fresh(n, extent_alignment);
    Device disk(size_t(r.layout.data_offset+r.layout.allocation_bytes)); auto io = disk.transport();
    Aligned payload(n), data(16*chunk_bytes), index(index_page_bytes), output(n);
    fill(payload.span()); StoreBuffers buffers{data.span(), index.span()};
    write_value(io, r, buffers, payload.span());
    assert(r.state == RegionState::sealed);
    assert(io.counters.write_submitted == r.layout.data_bytes+r.layout.index_bytes);
    assert(io.counters.write_completed == io.counters.write_submitted);
    assert(std::all_of(disk.bytes.begin(), disk.bytes.begin()+extent_alignment, [](uint8_t x){return x==0xa5;}));
    for (uint64_t i = n; i < r.layout.data_bytes; ++i) assert(disk.bytes[size_t(r.layout.data_offset+i)] == 0);
    ValueReader reader;
    allocation_watch_start();
    assert(!reader.open(io, r, buffers)); assert(!reader.read(0, output.span()));
    assert(!allocation_watch_stop());
    assert(!memcmp(payload.data, output.data, n));
    assert(io.counters.read_submitted == r.layout.data_bytes+r.layout.index_bytes);
    if (r.layout.chunks > records_per_page) {
        // A correctly checksummed page from the same index is still wrong when
        // delivered for another logical record range.
        std::array<uint8_t, index_page_bytes> saved{};
        std::memcpy(saved.data(), disk.bytes.data()+r.layout.index_offset, saved.size());
        std::memcpy(disk.bytes.data()+r.layout.index_offset,
                    disk.bytes.data()+r.layout.index_offset+index_page_bytes, index_page_bytes);
        ValueReader wrong_page; assert(!wrong_page.open(io, r, buffers));
        assert(wrong_page.read(0, output.span().first(1)).code == IoCode::corrupt);
        std::memcpy(disk.bytes.data()+r.layout.index_offset, saved.data(), saved.size());
        assert(!reader.open(io, r, buffers)); // reacquire the cache after another reader used these buffers
    }
    if (n > 33) {
        std::array<uint8_t, 33> small{};
        for (size_t at : {size_t(1), n/3, n-33}) {
            assert(!reader.read(at, small)); assert(!memcmp(small.data(), payload.data+at, small.size()));
        }
        auto error = reader.read(n-32, small); assert(error.code == IoCode::invalid);
    }
    if(n>5*chunk_bytes){
        // Large edge reads cross batches, with each combination of source and
        // destination alignment; short final chunks remain unreadable padding.
        for(size_t start:{size_t(0),size_t(1),chunk_bytes-1})for(size_t shift:{size_t(0),size_t(1)}){
            const size_t count=5*chunk_bytes-17;
            assert(!reader.read(start,output.span().subspan(shift,count)));
            assert(!memcmp(output.data+shift,payload.data+start,count));
        }
    }
    assert(io.counters.writes < r.layout.chunks+3); // IO is batched, not one request per checksum.
    printf("Store roundtrip: bytes=%zu chunks=%llu exact R/W=%llu/%llu PASS\n", n,
           (unsigned long long)r.layout.chunks, (unsigned long long)(r.layout.data_bytes+r.layout.index_bytes),
           (unsigned long long)io.counters.write_submitted);
}
static void failures() {
    constexpr size_t n = 3*chunk_bytes+19;
    auto r = fresh(n); Device disk(size_t(r.layout.allocation_bytes)); auto t = disk.transport();
    Aligned input(n), data(4*chunk_bytes), index(index_page_bytes), output(n);
    fill(input.span()); StoreBuffers buffers{data.span(), index.span()}; write_value(t, r, buffers, input.span());
    auto read = [&] { ValueReader v; assert(!v.open(t, r, buffers)); return v.read(0, output.span()); };
    disk.corrupt_at = 31; disk.corrupt_reads = 1;
    assert(!read() && !memcmp(input.data, output.data, n));
    disk.corrupt_reads = -1; auto before = disk.reads;
    assert(read().code == IoCode::corrupt && disk.reads-before <= 5);
    disk.corrupt_at = r.layout.index_offset+64; disk.corrupt_reads = 1;
    assert(!read());
    disk.corrupt_reads = -1; before = disk.reads;
    assert(read().code == IoCode::corrupt && disk.reads-before == 3); // data plus two index reads
    disk.corrupt_reads = 0;
    Region wrong = r; ++wrong.id.incarnation;
    ValueReader v; assert(!v.open(t, wrong, buffers)); assert(v.read(0, output.span()).code == IoCode::corrupt);
    wrong = r; ++wrong.index_instance;
    assert(!v.open(t, wrong, buffers)); assert(v.read(0, output.span()).code == IoCode::corrupt);
    disk.short_read = disk.reads+1;
    assert(read().code == IoCode::short_io); disk.short_read = 0;
    disk.fail_read = disk.reads+1;
    assert(read().code == IoCode::io); disk.fail_read = 0;
    disk.cancelled = true; before = disk.reads;
    assert(read().code == IoCode::cancelled && disk.reads == before); disk.cancelled = false;
    assert(r.state == RegionState::sealed);

    for (unsigned mode = 0; mode < 6; ++mode) {
        auto out = fresh(n, 0, 100+mode); Device dest(size_t(out.layout.allocation_bytes)); auto io = dest.transport();
        if (mode == 0) dest.fail_write = 1;
        if (mode == 1) dest.fail_write = 2; // index after all data
        if (mode == 2) dest.short_write = 1;
        if (mode == 3) io.write_limit = out.layout.data_bytes;
        if (mode == 4) dest.cancel_after_write = 1;
        ValueWriter w; assert(!w.open(io, out, buffers));
        ValueWriter duplicate; assert(duplicate.open(io, out, buffers).code == IoCode::invalid);
        auto error = w.append(mode == 5 ? input.span().first(n-1) : input.span());
        if (!error) error = w.seal();
        assert(error && out.state == RegionState::invalid);
        assert(w.seal().code == error.code); // first error is sticky
        assert(io.counters.write_submitted <= io.write_limit);
        if (mode < 3) assert(io.counters.write_submitted > io.counters.write_completed);
        if (mode == 4) assert(error.code == IoCode::cancelled);
        if (mode == 5) assert(error.code == IoCode::incomplete);
        ValueReader bad; assert(bad.open(io, out, buffers).code == IoCode::incomplete);
        assert(begin_region(out, out.id, out.index_instance, out.layout).code == IoCode::invalid);
        auto newer = out.id; ++newer.incarnation;
        assert(!begin_region(out, newer, out.index_instance+1, out.layout));
    }
    puts("Store failures: corruption/retry, wrong identity, short IO, write cap, cancellation and unpublished output PASS");
}
static void product(size_t an, size_t bn, unsigned workers) {
    auto a = fresh(an*8, 0, 10);
    auto b = fresh(bn*8, a.layout.allocation_bytes, 20);
    auto result = fresh((an+bn)*8, b.layout.data_offset+b.layout.allocation_bytes, 30);
    auto failed = fresh((an+bn)*8, result.layout.data_offset+result.layout.allocation_bytes, 40);
    Device disk(size_t(failed.layout.data_offset+failed.layout.allocation_bytes)); auto io = disk.transport();
    Aligned av(an*8), bv(bn*8), staging(16*chunk_bytes), index(index_page_bytes), output((an+bn)*8);
    fill(av.span()); fill(bv.span());
    StoreBuffers buffers{staging.span(), index.span()}; write_value(io, a, buffers, av.span()); write_value(io, b, buffers, bv.span());
    const std::vector<uint8_t> preserved(disk.bytes.begin(), disk.bytes.begin()+result.layout.data_offset);
    ResidentProductPlan plan{}; sbn3_mul_options options{}; options.workers = workers;
    allocation_watch_start();
    assert(query_resident_product(an, bn, options, staging.bytes, plan) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    assert(plan.storage_bytes < (size_t(32)<<20));
    Fixture f(workers, false, size_t(64)<<20);
    const auto workspace = f.allocate(plan.storage_bytes, plan.alignment);
    const auto before = io.counters;
    allocation_watch_start();
    auto error = run_resident_product(plan, io, a, b, result,
                                     {static_cast<uint8_t *>(workspace.data), workspace.bytes}, *f.team);
    assert(!error && !allocation_watch_stop());
    assert(io.counters.read_submitted-before.read_submitted == a.layout.data_bytes+a.layout.index_bytes+b.layout.data_bytes+b.layout.index_bytes);
    assert(io.counters.write_submitted-before.write_submitted == result.layout.data_bytes+result.layout.index_bytes);
    assert(result.state == RegionState::sealed && a.state == RegionState::sealed && b.state == RegionState::sealed);
    assert(!memcmp(preserved.data(), disk.bytes.data(), preserved.size()));
    ValueReader reader; assert(!reader.open(io, result, buffers)); assert(!reader.read(0, output.span()));
    assert(ref_product_equal(reinterpret_cast<uint64_t *>(av.data), an, reinterpret_cast<uint64_t *>(bv.data), bn,
                             reinterpret_cast<uint64_t *>(output.data), an+bn));
    // The consumer may reuse the entire prefix after the product joins. Test
    // a prefix beyond the old operation, forcing the result-after-dead-work
    // layout; overwriting that prefix must leave the exact borrowed result.
    auto retained=plan;const size_t prefix=plan.storage_bytes+4096;
    assert(reserve_resident_result_after(retained,prefix)&&retained.result>=prefix);
    const auto held=f.allocate(retained.storage_bytes,retained.alignment);ResidentProductValue borrowed;
    const auto writes=io.counters.write_submitted;allocation_watch_start();
    assert(!run_resident_product_borrowed(retained,io,a,b,{static_cast<uint8_t *>(held.data),held.bytes},*f.team,borrowed));
    assert(!allocation_watch_stop()&&io.counters.write_submitted==writes);
    memset(held.data,0xcc,prefix);
    assert(borrowed.words.size()==an+bn&&ref_product_equal(reinterpret_cast<uint64_t *>(av.data),an,
        reinterpret_cast<uint64_t *>(bv.data),bn,borrowed.words.data(),borrowed.words.size()));
    if(plan.product.contract&program_bounded_inputs)for(unsigned mask:{1u,2u,3u}){
        auto direct=plan;assert(borrow_resident_inputs(direct,mask)&&direct.storage_bytes<=plan.storage_bytes);
        const size_t ac=an-((mask&1)?2:0),bc=bn-((mask&2)?2:0);
        assert(rebind_resident_sources(direct,{an,(mask&1)?1u:0u,ac},{bn,(mask&2)?1u:0u,bc}));
        const auto scratch=f.allocate(direct.storage_bytes,direct.alignment);
        memset(scratch.data,0xa5,scratch.bytes);const auto counters=io.counters;ResidentProductValue value;
        const ResidentProductInput x{mask&1?nullptr:&a,mask&1?std::span(reinterpret_cast<const uint64_t *>(av.data),an):std::span<const uint64_t>{}},
                                   y{mask&2?nullptr:&b,mask&2?std::span(reinterpret_cast<const uint64_t *>(bv.data),bn):std::span<const uint64_t>{}};
        allocation_watch_start();assert(!run_resident_product_borrowed(direct,io,x,y,{static_cast<uint8_t *>(scratch.data),scratch.bytes},*f.team,value));assert(!allocation_watch_stop());
        assert(io.counters.write_submitted==counters.write_submitted);
        const uint64_t expected_reads=(mask&1?0:a.layout.data_bytes+a.layout.index_bytes)+(mask&2?0:b.layout.data_bytes+b.layout.index_bytes);
        assert(io.counters.read_submitted-counters.read_submitted==expected_reads);
        assert(ref_product_equal(reinterpret_cast<const uint64_t *>(av.data)+((mask&1)?1:0),ac,
            reinterpret_cast<const uint64_t *>(bv.data)+((mask&2)?1:0),bc,value.words.data(),ac+bc));
        for(size_t i=ac+bc;i<value.words.size();++i)assert(!value.words[i]);
    }
    if(plan.product.contract&program_bounded_inputs){
        f.guard_pages=true;auto*ga=f.guarded(an),*gb=f.guarded(bn);f.guard_pages=false;
        memcpy(ga,av.data,an*8);memcpy(gb,bv.data,bn*8);
        auto direct=plan;assert(borrow_resident_inputs(direct,3));const auto memory=f.allocate(direct.storage_bytes,direct.alignment);
        const auto counts=io.counters;ResidentProductValue got;
        allocation_watch_start();assert(!run_resident_product_borrowed(direct,io,{nullptr,{ga,an}},{nullptr,{gb,bn}},
            {static_cast<uint8_t *>(memory.data),memory.bytes},*f.team,got));assert(!allocation_watch_stop());
        assert(io.counters.read_submitted==counts.read_submitted&&io.counters.write_submitted==counts.write_submitted);
        assert(ref_product_equal(ga,an,gb,bn,got.words.data(),an+bn));
        assert(!memcmp(ga,av.data,an*8)&&!memcmp(gb,bv.data,bn*8));
    }
    // Failed stored-boundary execution does not publish any output or mutate sources.
    disk.fail_read = disk.reads+1;
    error = run_resident_product(plan, io, a, b, failed,
                                {static_cast<uint8_t *>(workspace.data), workspace.bytes}, *f.team);
    assert(error.code == IoCode::io && failed.state == RegionState::invalid);
    assert(!memcmp(preserved.data(), disk.bytes.data(), preserved.size()));
    printf("Stored-boundary product: %zux%zu W%u resident grant=%zu bytes PASS\n", an, bn, workers, plan.storage_bytes);
}
int main() {
    m61_checks();
    checksum_checks();
    for (size_t n : {size_t(0), size_t(1), chunk_bytes-1, chunk_bytes, chunk_bytes+11, records_per_page*chunk_bytes+97}) roundtrip(n);
    failures();
    for (unsigned w : {1u,4u}) for (auto pair : {std::pair<size_t,size_t>{17,9}, {2053,1301}, {32769,12977}})
        product(pair.first, pair.second, w);
    puts("OOC Store: pinned XXH3-64, bounded batched IO, native promotion, exact byte ledger and allocation gates PASS");
}
