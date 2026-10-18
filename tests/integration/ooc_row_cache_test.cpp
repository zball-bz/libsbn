// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/row_cache.hpp"
#include "external/owned_product.hpp"
#include <utility>

static StoredValue upload(ValueStore&s,Transport&t,StoreBuffers io,const std::vector<uint64_t>&v){
    RegionLease lease;StoredValue out;assert(!s.reserve(v.size()*8,RegionKind::value,lease));
    write_value(t,lease.region(),io,{reinterpret_cast<const uint8_t *>(v.data()),v.size()*8});assert(!lease.publish(out));return out;
}
static void one(RowCacheKind kind){
    const size_t an=kind==RowCacheKind::outer_rows?6001:30013,bn=kind==RowCacheKind::outer_rows?7003:17011,cn=kind==RowCacheKind::outer_rows?4507:101;
    std::vector<uint64_t>a(an),b(bn),c(cn);for(auto*v:{&a,&b,&c})for(auto&word:*v)word=random_word();
    ValueStoreStorage metadata_plan{};assert(query_value_store(48,metadata_plan));Aligned metadata(metadata_plan.bytes),io_data(4*chunk_bytes),page(index_page_bytes);
    Device disk(32u<<20);auto t=disk.transport();StoreBuffers io{io_data.span(),page.span()};ValueStore store;
    assert(!store.open(79,89,0,disk.bytes.size(),metadata.span(),48));
    auto av=upload(store,t,io,a),bv=upload(store,t,io,b),cv=upload(store,t,io,c);
    SpilledProductPlan full{};assert(query_spilled_product({an,bn,4,2048,1024,6,128,3,4,false,12*chunk_bytes},full));
    RowCachePlan cache_plan{};assert(query_row_cache(full.forward[0],cache_plan,kind));
    auto apply=full;assert(use_cached_a(apply));assert(apply.plane_count==6&&apply.storage_bytes<full.storage_bytes);
    if(kind==RowCacheKind::completed_columns)assert(use_completed_cached_a(apply));
    const size_t work_bytes=std::max(apply.storage_bytes,cache_plan.build_bytes);
    Fixture fixture(4,false,32u<<20);auto memory=fixture.allocate(cache_plan.storage_bytes+work_bytes,4096);
    auto cache_memory=std::span(static_cast<uint8_t *>(memory.data),cache_plan.storage_bytes);
    auto work=std::span(static_cast<uint8_t *>(memory.data)+cache_plan.storage_bytes,work_bytes);
    RowCacheBinding cache{};const auto held=store.usage().live_bytes;
    disk.fail_write=disk.writes+1;allocation_watch_start();
    assert(build_row_cache(cache_plan,t,store,av,full.spec.a_source,cache_memory,work,*fixture.team,cache).code==IoCode::io);
    assert(!allocation_watch_stop()&&!cache.control&&store.usage().live_bytes==held);disk.fail_write=0;
    auto before=t.counters;allocation_watch_start();
    assert(!build_row_cache(cache_plan,t,store,av,full.spec.a_source,cache_memory,work,*fixture.team,cache));assert(!allocation_watch_stop());
    assert(t.counters.read_submitted-before.read_submitted==cache_plan.normal_read_bytes);
    assert(t.counters.write_submitted-before.write_submitted==cache_plan.normal_write_bytes);
    uint64_t reads=cache_plan.normal_read_bytes,writes=cache_plan.normal_write_bytes,baseline_reads=0,baseline_writes=0,saved_family=0;
    const auto old_offset=av.region().layout.data_offset;av.reset();
    // Reuse and overwrite the dead raw source. Cached consumers must not read it.
    std::vector<uint64_t> poison(an,0x8383838383838383ULL);auto replacement=upload(store,t,io,poison);
    assert(replacement.region().layout.data_offset==old_offset);
    StoredValue result[2];WordExtent extent[2];
    for(unsigned i=0;i<2;++i){const auto&source=i?cv:bv;const auto&original=i?c:b;
        auto p=apply;auto reference=full;
        if(i&&kind==RowCacheKind::completed_columns){auto spec=full.spec;spec.bn=original.size();spec.b_source={original.size(),0,original.size()};
            assert(query_spilled_product(spec,reference));p=reference;assert(use_cached_a(p)&&use_completed_cached_a(p));
            assert(p.forward[0].spec.live_rows<cache_plan.forward.spec.live_rows);
        }
        const SourceWindow view{original.size(),0,original.size()};
        assert(rebind_spilled_sources(p,row_cache_source(cache),view));
        assert(rebind_spilled_sources(reference,full.spec.a_source,view));
        if(i)saved_family=reference.spec.primes*reference.plane_bytes;
        baseline_reads+=reference.normal_read_bytes;baseline_writes+=reference.normal_write_bytes;
        before=t.counters;allocation_watch_start();assert(!run_owned_cached_product(p,t,store,cache,source,result[i],work,*fixture.team,&extent[i]));assert(!allocation_watch_stop());
        const auto rd=t.counters.read_submitted-before.read_submitted,wr=t.counters.write_submitted-before.write_submitted;
        assert(rd==p.normal_read_bytes&&wr==p.normal_write_bytes);reads+=rd;writes+=wr;
        std::vector<uint64_t> output(p.output_bytes/8);ValueReader reader;assert(!reader.open(t,result[i].region(),io));
        assert(!reader.read(0,{reinterpret_cast<uint8_t *>(output.data()),output.size()*8}));
        assert(ref_product_equal(a.data(),a.size(),original.data(),original.size(),output.data(),a.size()+original.size()));
        for(size_t j=a.size()+original.size();j<output.size();++j)assert(!output[j]);
    }
    assert(baseline_reads-reads==full.forward[0].source_read_bytes);
    assert(baseline_writes-writes==saved_family);
    auto wrong=apply;assert(rebind_spilled_sources(wrong,{an,1,an-1},full.spec.b_source));StoredValue rejected;
    before=t.counters;const auto active=store.usage().live_bytes;
    assert(run_owned_cached_product(wrong,t,store,cache,bv,rejected,work,*fixture.team).code==IoCode::invalid);
    assert(!rejected&&store.usage().live_bytes==active&&before.write_submitted==t.counters.write_submitted);
    wrong=apply;wrong.middle.spec.a_transformed=!wrong.middle.spec.a_transformed;
    assert(run_owned_cached_product(wrong,t,store,cache,bv,rejected,work,*fixture.team).code==IoCode::invalid);
    assert(!rejected&&store.usage().live_bytes==active&&before.write_submitted==t.counters.write_submitted);
    disk.fail_write=disk.writes+1;
    assert(run_owned_cached_product(apply,t,store,cache,bv,rejected,work,*fixture.team).code==IoCode::io);
    assert(!rejected&&store.usage().live_bytes==active);disk.fail_write=0;
    assert(!run_owned_cached_product(apply,t,store,cache,bv,rejected,work,*fixture.team));
    rejected.reset();
    allocation_watch_start();release_row_cache(cache);assert(!allocation_watch_stop());assert(!cache.control);
    assert(run_owned_cached_product(apply,t,store,cache,bv,rejected,work,*fixture.team).code==IoCode::invalid);
    printf("Cached state %u: two unequal products, original source retired/reused; saved R=%llu W=%llu bytes; cache RAM=%zu, apply RAM=%zu PASS\n",unsigned(kind),
           (unsigned long long)(baseline_reads-reads),(unsigned long long)(baseline_writes-writes),cache_plan.storage_bytes,apply.storage_bytes);
    puts("Cached rows: exact IO, full product checks, allocation watch, failed build and incompatible/dead cache rejection PASS");
}
int main(){one(RowCacheKind::outer_rows);one(RowCacheKind::completed_columns);}
