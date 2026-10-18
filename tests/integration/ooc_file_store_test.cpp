// SPDX-License-Identifier: LGPL-3.0-or-later
#include "file_store_support.hpp"
#include "runtime/team.hpp"
#include <sys/stat.h>
static StoredValue value(FileDevice&d,Transport&t,StoreBuffers io,std::span<uint8_t>bytes){
    RegionLease region;StoredValue result;assert(!d.store.reserve(bytes.size(),RegionKind::value,region));
    ValueWriter writer;assert(!writer.open(t,region.region(),io));assert(!writer.append(bytes));assert(!writer.seal());assert(!region.publish(result));return result;
}
static void roundtrip(bool named){
    FileDevice disk(24,8u<<20,named);auto t=disk.transport();Aligned data(4*chunk_bytes),page(index_page_bytes),input(9*chunk_bytes+53),output(9*chunk_bytes+53);
    fill(input.span());StoreBuffers io{data.span(),page.span()};allocation_watch_start();auto saved=value(disk,t,io,input.span());assert(!allocation_watch_stop());
    const auto region=saved.region();const int fd=disk.files.descriptor(region);assert(fd>=0);struct stat st{};assert(!fstat(fd,&st));
    assert(uint64_t(st.st_size)==region.layout.allocation_bytes&&uint64_t(st.st_size)<disk.plan.slot_bytes);
    ValueReader reader;allocation_watch_start();assert(!reader.open(t,region,io));assert(!reader.read(0,output.span()));assert(!allocation_watch_stop());assert(!memcmp(input.data,output.data,input.bytes));
    assert(t.counters.read_submitted==region.layout.data_bytes+region.layout.index_bytes&&t.counters.write_submitted==t.counters.read_submitted);
    assert(write_exact(t,region.layout.data_offset,data.span().first(chunk_bytes)).code==IoCode::invalid); // published values are read-only
    std::memcpy(data.data,input.data,chunk_bytes);data.data[17]^=0x80;assert(pwrite(fd,data.data,chunk_bytes,0)==ssize_t(chunk_bytes));assert(!reader.open(t,region,io));
    assert(reader.read(0,output.span()).code==IoCode::corrupt);assert(pwrite(fd,input.data,chunk_bytes,0)==ssize_t(chunk_bytes));
    assert(!reader.open(t,region,io));assert(!reader.read(0,output.span()));assert(!disk.files.sync(region));
    StoredValue pin;saved.share(pin);saved.reset();assert(disk.files.usage().live_files==1);const auto before=disk.files.usage();pin.reset();
    assert(disk.files.usage().open_files==before.open_files&&disk.files.usage().retired_files==1);assert(fcntl(fd,F_GETFD)>=0);
    assert(read_exact(t,region.layout.data_offset,data.span().first(chunk_bytes)).code==IoCode::stale);
    allocation_watch_start();assert(!disk.files.collect());assert(!allocation_watch_stop());assert(fcntl(fd,F_GETFD)<0&&errno==EBADF);
    assert(!disk.files.usage().open_files&&!disk.files.usage().claimed_bytes&&!disk.store.usage().live_bytes);
    disk.close();printf("Object files %s: exact bytes, compact lengths, readonly publication, corruption, sync, pins/deferred cleanup PASS\n",named?"tmpfs DIRECT":"memfd");
}
static void fragmentation(){
    constexpr uint64_t q=extent_alignment;FileDevice disk(16,12*q);auto t=disk.transport();(void)t;RegionLease r[5];
    for(unsigned i=0;i<4;++i){const auto e=disk.store.reserve(2*q-2*chunk_bytes,RegionKind::scratch,r[i]);
        if(e)fprintf(stderr,"slot reserve %u code%u %s need%llu have%llu errno%d live%llu files%llu\n",i,unsigned(e.code),e.where,
            (unsigned long long)e.need,(unsigned long long)e.have,e.system_error,(unsigned long long)disk.store.usage().live_bytes,(unsigned long long)disk.files.usage().claimed_bytes);assert(!e);}
    assert(disk.store.usage().live_bytes==12*q);const auto old=r[0].region().id;const uint64_t address=r[0].region().layout.data_offset;
    r[0].reset();r[2].reset();assert(disk.store.usage().free_bytes==6*q&&disk.files.usage().retired_bytes==6*q);
    allocation_watch_start();assert(!disk.store.reserve(4*q-2*chunk_bytes,RegionKind::scratch,r[4]));assert(!allocation_watch_stop());
    assert(r[4].region().layout.data_offset==address&&r[4].region().id!=old&&disk.store.usage().live_bytes==11*q);
    assert(disk.files.usage().claimed_bytes==11*q&&disk.files.usage().peak_claimed_bytes<=12*q);
    RegionLease refused;assert(disk.store.reserve(q-2*chunk_bytes,RegionKind::scratch,refused).code==IoCode::capacity&&!refused);
    for(auto&v:r)v.reset();assert(!disk.files.collect());disk.close();
    puts("Virtual slots: 5Q allocation fits fragmented 3Q+3Q free budget, fresh incarnation and no compaction/extra IO PASS");
}
struct FailureHooks {
    RegionLifecycle next{};uint64_t failed_incarnation=0;bool fail=true;
    static IoError create(void*p,const Region&r) noexcept{auto&f=*static_cast<FailureHooks *>(p);if(f.fail){f.failed_incarnation=r.id.incarnation;return {IoCode::io,"injected creation",0,0,0,ENOMEM};}return f.next.create(f.next.context,r);}
    static void retire(void*p,const Region&r) noexcept{auto&f=*static_cast<FailureHooks *>(p);f.next.retire(f.next.context,r);}
    static void publish(void*p,const Region&r) noexcept{auto&f=*static_cast<FailureHooks *>(p);f.next.publish(f.next.context,r);}
};
static void creation_failure(){
    SlotStorePlan p;assert(query_slot_store(8,1u<<20,p));FileStoreStorage q;assert(query_file_store(8,q));Aligned metadata(p.metadata.bytes),registry(q.bytes);FileStore files;
    assert(!files.open_memory(p,registry.span()));FailureHooks hooks{files.lifecycle()};ValueStore store;
    assert(!store.open_slots(51,61,p,metadata.span(),{&hooks,FailureHooks::create,FailureHooks::retire,FailureHooks::publish}));RegionLease r;
    assert(store.reserve(17,RegionKind::value,r).code==IoCode::io&&!r&&!store.usage().live_bytes&&!files.usage().open_files);
    hooks.fail=false;assert(!store.reserve(17,RegionKind::value,r));assert(r.region().id.incarnation>hooks.failed_incarnation);
    r.reset();assert(!files.close());puts("File creation refusal: metadata unchanged, failed identity consumed, subsequent reserve/cleanup PASS");
}
static void simultaneous_writes(){
    Fixture team(4,false,8u<<20);FileDevice disk(8,2u<<20);auto t=disk.transport();assert(t.parallel_writes>=4);
    RegionLease values[4];for(auto&value:values)assert(!disk.store.reserve(4*chunk_bytes,RegionKind::scratch,value));
    Aligned input(4*chunk_bytes),output(input.bytes);fill(input.span());
    struct Job {Transport*t;uint64_t at[4];std::span<const uint8_t>data;IoError errors[4]{};} job{&t,{},input.span(),{}};
    for(unsigned k=0;k<4;++k)job.at[k]=values[k].region().layout.data_offset;
    auto worker=[](void*p,uint64_t lo,uint64_t hi,unsigned){auto&j=*static_cast<Job*>(p);for(auto i=lo;i<hi;++i)j.errors[i]=write_exact(*j.t,j.at[i],j.data);};
    struct Call {Job*j;sbn3_for_fn f;} call{&job,worker};
    auto run=[](void*p,sbn3_team_scope*s){auto&c=*static_cast<Call*>(p);sbn3_team_for(s,0,4,1,SBN3_STATIC,c.f,c.j);};
    t.write_limit=6*chunk_bytes;allocation_watch_start();sbn3_team_run(team.team,run,&call);assert(!allocation_watch_stop());
    unsigned success=0;for(auto e:job.errors){success+=!e;if(e)assert(e.code==IoCode::capacity);}assert(success==1);
    assert(t.counters.write_submitted==4*chunk_bytes&&t.counters.write_completed==4*chunk_bytes);
    t.write_limit+=16*chunk_bytes;sbn3_team_run(team.team,run,&call);for(auto e:job.errors)assert(!e);
    assert(disk.files.usage().write_bytes==t.counters.write_completed&&t.counters.write_completed==20*chunk_bytes);
    for(auto at:job.at){assert(!read_exact(t,at,output.span()));assert(!memcmp(input.data,output.data,input.bytes));}
    for(auto&value:values)value.reset();disk.close();puts("Four concurrent file writers: exact bytes, atomic budget admission and joined ownership PASS");
}
int main(){setvbuf(stdout,nullptr,_IOLBF,0);roundtrip(false);roundtrip(true);fragmentation();creation_failure();simultaneous_writes();
    puts("Linux object-file Store: sum-of-live-byte admission, compact virtual slots, real syscalls and zero SSD operand writes PASS");}
