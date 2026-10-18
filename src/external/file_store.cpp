// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/file_store.hpp"
#include "common/checked.hpp"
#include "runtime/linux/sync.hpp"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <linux/memfd.h>
#include <new>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
namespace sbn::v3::external {
namespace {
IoError failed(const char*where,int code=errno){return {IoCode::io,where,0,0,0,code};}
struct RingLock {
    uint32_t *word;
    explicit RingLock(uint32_t&p):word(&p){uint32_t free=0;
        if(__atomic_compare_exchange_n(word,&free,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED))return;
        while(__atomic_exchange_n(word,2,__ATOMIC_ACQUIRE))os::wait_changed(word,2);}
    ~RingLock(){if(__atomic_exchange_n(word,0,__ATOMIC_RELEASE)==2)os::wake(word);}
};
void file_name(char(&out)[64],RegionId id){
    static constexpr char hex[]="0123456789abcdef";size_t at=0;out[at++]='s';out[at++]='b';out[at++]='n';
    for(uint64_t word:{id.uuid_high,id.uuid_low,id.incarnation}){out[at++]='-';for(unsigned i=16;i-- >0;)out[at++]=hex[(word>>(i*4))&15];}
    out[at]=0;
}
}
bool query_file_store(size_t slots,FileStoreStorage&out) noexcept {
    FileStoreStorage p{};if(!slots||!mul_size(slots,sizeof(FileStore::Entry),p.bytes)||!align_size(p.bytes,p.alignment,p.bytes))return false;out=p;return true;
}
FileStore::~FileStore(){require(!entries_&&directory_<0&&!usage_.open_files,SBN3_FATAL_LIFETIME,"file Store requires explicit close");}
IoError FileStore::open(const SlotStorePlan&p,std::span<uint8_t>memory,int directory,bool direct) noexcept {
    SlotStorePlan q{};FileStoreStorage storage{};
    if(entries_||!query_slot_store(p.metadata.slots,p.live_byte_limit,q)||p.slot_bytes!=q.slot_bytes||p.address_bytes!=q.address_bytes||
       p.slot_shift!=q.slot_shift||!query_file_store(p.metadata.slots,storage)||!memory.data()||(uintptr_t(memory.data())&(storage.alignment-1))||memory.size()<storage.bytes)
        return {IoCode::invalid,"file Store layout/metadata"};
    int owned=-1;
    if(directory>=0){struct stat st{};if(fstat(directory,&st))return failed("file Store directory stat");
        if(!S_ISDIR(st.st_mode))return {IoCode::invalid,"file Store requires directory"};
        owned=fcntl(directory,F_DUPFD_CLOEXEC,0);if(owned<0)return failed("file Store directory duplicate");
    }else if(direct)return {IoCode::invalid,"memory file direct mode"};
    entries_=reinterpret_cast<Entry *>(memory.data());plan_=q;directory_=owned;memory_=directory<0;direct_=direct;detaching_=suspended_=false;usage_={};
    for(size_t i=0;i<q.metadata.slots;++i)::new(entries_+i)Entry{};
    return {};
}
IoError FileStore::open_memory(const SlotStorePlan&p,std::span<uint8_t>w) noexcept {return open(p,w,-1,false);}
IoError FileStore::open_directory(const SlotStorePlan&p,std::span<uint8_t>w,int directory,bool direct) noexcept {
    if(directory<0)return {IoCode::invalid,"file Store directory descriptor"};return open(p,w,directory,direct);
}
FileStore::Entry*FileStore::entry(const Region&r) const noexcept {
    if(!entries_||r.layout.data_offset%plan_.slot_bytes)return nullptr;
    const auto slot=r.layout.data_offset>>plan_.slot_shift;if(slot>=plan_.metadata.slots)return nullptr;
    auto*e=entries_+slot;return e->live&&!e->retired&&e->id==r.id&&e->bytes==r.layout.allocation_bytes?e:nullptr;
}
IoError FileStore::create(const Region&r) noexcept {
    if(suspended_)return {IoCode::commit_unknown,"file creation blocked by checkpoint state"};
    if(!entries_||detaching_||r.layout.data_offset%plan_.slot_bytes||r.layout.allocation_bytes>plan_.slot_bytes||
       r.layout.allocation_bytes>plan_.live_byte_limit)return {IoCode::invalid,"file object layout"};
    const size_t slot=size_t(r.layout.data_offset>>plan_.slot_shift);
    if(slot>=plan_.metadata.slots)return {IoCode::invalid,"file object slot"};
    if(auto e=collect())return e;
    if(detaching_)return {IoCode::invalid,"file Store preserving detached root"};
    auto&e=entries_[slot];if(e.fd>=0||e.live)return {IoCode::invalid,"file object slot already owned"};
    if(r.layout.allocation_bytes>plan_.live_byte_limit-usage_.claimed_bytes)
        return {IoCode::capacity,"file object live byte cap",0,r.layout.allocation_bytes,plan_.live_byte_limit-usage_.claimed_bytes};
    char name[64]{};file_name(name,r.id);
    const int fd=memory_?int(syscall(SYS_memfd_create,name,MFD_CLOEXEC)):
        openat(directory_,name,O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC|O_RDWR|(direct_?O_DIRECT:0),0600);
    if(fd<0)return failed("file object create");
    e={};e.fd=fd;e.id=r.id;e.bytes=r.layout.allocation_bytes;e.live=true;std::copy_n(name,64,e.name);
    ++usage_.created;++usage_.open_files;++usage_.live_files;usage_.claimed_bytes+=e.bytes;
    usage_.peak_claimed_bytes=std::max(usage_.peak_claimed_bytes,usage_.claimed_bytes);
    if(ftruncate(fd,off_t(e.bytes))){const auto error=failed("file object resize");retire(r);return error;}
    return {};
}
void FileStore::retire(const Region&r) noexcept {
    auto*e=entry(r);require(e,SBN3_FATAL_LIFETIME,"file object retirement identity");e->retired=true;
    if(e->keep)detaching_=true;
    --usage_.live_files;++usage_.retired_files;usage_.retired_bytes+=e->bytes;
}
void FileStore::published(const Region&r) noexcept {
    auto*e=entry(r);require(e&&r.kind==RegionKind::value&&r.state==RegionState::sealed,SBN3_FATAL_LIFETIME,"file value publication");e->readonly=true;
}
IoError FileStore::collect() noexcept {
    if(!entries_)return {IoCode::invalid,"file Store closed"};
    for(size_t i=0;i<plan_.metadata.slots;++i){auto&e=entries_[i];if(!e.retired)continue;
        if(!memory_&&!e.keep&&unlinkat(directory_,e.name,0)&&errno!=ENOENT)return failed("file object unlink");
        const int fd=std::exchange(e.fd,-1);const int rc=::close(fd);const int error=rc?errno:0;
        usage_.claimed_bytes-=e.bytes;usage_.retired_bytes-=e.bytes;--usage_.open_files;--usage_.retired_files;
        if(e.keep)usage_.detached_bytes+=e.bytes;else ++usage_.removed;e={};
        // On Linux never retry close: the descriptor may already be reused.
        if(error)return failed("file object close",error);
    }
    return {};
}
Completion FileStore::transfer(uint64_t offset,void*data,size_t bytes,bool write) noexcept {
    if(!entries_||!data||!bytes||bytes>SSIZE_MAX)return {IoCode::invalid,0,EINVAL};
    const size_t slot=size_t(offset>>plan_.slot_shift);const uint64_t local=offset&(plan_.slot_bytes-1);
    if(slot>=plan_.metadata.slots)return {IoCode::capacity,0,EINVAL};auto&e=entries_[slot];
    if(!e.live||e.retired||e.fd<0)return {IoCode::stale,0,EBADF};
    if(local>e.bytes||bytes>e.bytes-local)return {IoCode::capacity,0,EINVAL};
    if(write&&e.readonly)return {IoCode::invalid,0,EROFS};
    if(write)__atomic_store_n(&e.synced,false,__ATOMIC_RELAXED);
    if(write)__atomic_fetch_add(&usage_.write_calls,uint64_t(1),__ATOMIC_RELAXED);else __atomic_fetch_add(&usage_.read_calls,uint64_t(1),__ATOMIC_RELAXED);
    const ssize_t n=write?pwrite(e.fd,data,bytes,off_t(local)):pread(e.fd,data,bytes,off_t(local));
    if(n<0)return {IoCode::io,0,errno};if(write)__atomic_fetch_add(&usage_.write_bytes,uint64_t(n),__ATOMIC_RELAXED);else __atomic_fetch_add(&usage_.read_bytes,uint64_t(n),__ATOMIC_RELAXED);
    return {IoCode::ok,size_t(n),0};
}
IoError FileStore::sync(const Region&r) noexcept {
    auto*e=entry(r);if(!e)return {IoCode::stale,"file sync identity"};if(e->synced)return {};++usage_.sync_calls;
    if(fsync(e->fd))return failed("file object sync");e->synced=true;return {};
}
IoError FileStore::attach(const Region&r) noexcept {
    if(!entries_||memory_||detaching_||r.layout.data_offset%plan_.slot_bytes||r.state!=RegionState::sealed||r.kind!=RegionKind::value)
        return {IoCode::invalid,"file import mode/layout"};
    const size_t slot=size_t(r.layout.data_offset>>plan_.slot_shift);
    if(slot>=plan_.metadata.slots)return {IoCode::invalid,"file import slot"};
    if(auto e=collect())return e;
    if(detaching_)return {IoCode::invalid,"file import detached root"};
    auto&e=entries_[slot];if(e.live||e.fd>=0)return {IoCode::invalid,"file import occupied slot"};
    if(r.layout.allocation_bytes>plan_.live_byte_limit-usage_.claimed_bytes)return {IoCode::capacity,"file import byte cap"};
    char name[64]{};file_name(name,r.id);const int fd=openat(directory_,name,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|(direct_?O_DIRECT:0));
    if(fd<0)return failed("file import open");
    struct stat st{};const int rc=fstat(fd,&st),error=rc?errno:0;
    if(rc||!S_ISREG(st.st_mode)||st.st_size<0||uint64_t(st.st_size)!=r.layout.allocation_bytes){::close(fd);
        return rc?failed("file import stat",error):IoError{IoCode::corrupt,"file import length/type"};}
    e={};e.fd=fd;e.id=r.id;e.bytes=r.layout.allocation_bytes;e.live=e.readonly=e.keep=true;std::copy_n(name,64,e.name);
    ++usage_.open_files;++usage_.live_files;++usage_.imported;usage_.claimed_bytes+=e.bytes;usage_.peak_claimed_bytes=std::max(usage_.peak_claimed_bytes,usage_.claimed_bytes);return {};
}
IoError FileStore::adopt(const Region&r) noexcept {auto*e=entry(r);if(!e||!e->readonly)return {IoCode::invalid,"file import adoption"};
    e->keep=false;e->synced=true;return {};}
IoError FileStore::preserve(const Region&r) noexcept {auto*e=entry(r);if(memory_||!e||!e->readonly)return {IoCode::invalid,"file root preservation"};
    e->keep=true;detaching_=true;return {};}
IoError FileStore::collect_orphans(std::span<uint8_t>w,size_t max_files,bool drafts) noexcept {
    size_t names;if(!entries_||directory_<0||!mul_size(max_files,64,names)||names>w.size()||w.size()-names<4096)return {IoCode::capacity,"orphan listing storage"};
    if(auto e=collect())return e;
    const int dir=openat(directory_,".",O_RDONLY|O_DIRECTORY|O_CLOEXEC);if(dir<0)return failed("orphan directory open");
    struct End {int fd;~End(){::close(fd);}} end{dir};size_t count=0;auto buffer=w.subspan(names);
    while(true){const long got=syscall(SYS_getdents64,dir,buffer.data(),std::min(buffer.size(),size_t(65536)));
        if(got<0)return failed("orphan directory read");if(!got)break;
        for(size_t at=0;at<size_t(got);){
            if(size_t(got)-at<20)return {IoCode::corrupt,"directory entry header"};uint16_t length;std::memcpy(&length,buffer.data()+at+16,2);
            if(length<20||length>size_t(got)-at)return {IoCode::corrupt,"directory entry length"};
            const char*name=reinterpret_cast<const char *>(buffer.data()+at+19);const void*zero=std::memchr(name,0,length-19);
            if(!zero)return {IoCode::corrupt,"directory entry name"};const size_t len=static_cast<const char *>(zero)-name;at+=length;
            bool valid=len==54&&!std::memcmp(name,"sbn-",4);
            if(valid)for(size_t j=4;j<len;++j)valid&=(j==20||j==37)?name[j]=='-':(name[j]>='0'&&name[j]<='9')||(name[j]>='a'&&name[j]<='f');
            if(!valid&&drafts&&len==47&&!std::memcmp(name,"sbn-ckpt-",9)&&!std::memcmp(name+42,".next",5)){
                valid=true;for(size_t j=9;j<42;++j)valid&=j==25?name[j]=='-':(name[j]>='0'&&name[j]<='9')||(name[j]>='a'&&name[j]<='f');}
            if(!valid)continue;bool live=false;for(size_t i=0;i<plan_.metadata.slots;++i)if(entries_[i].live&&!entries_[i].retired&&!std::strcmp(entries_[i].name,name)){live=true;break;}
            if(live)continue;if(count==max_files)return {IoCode::capacity,"orphan name capacity",0,count+1,max_files};
            std::memcpy(w.data()+64*count,name,len+1);++count;
        }
    }
    // Directory is unchanged while enumerating. Remove the bounded snapshot
    // afterward, so deletion cannot invalidate the enumeration cookies.
    for(size_t i=0;i<count;++i)if(unlinkat(directory_,reinterpret_cast<const char *>(w.data()+64*i),0)&&errno!=ENOENT)return failed("orphan object unlink");
    if(count&&fsync(directory_))return failed("orphan directory sync");return {};
}
IoError FileStore::close() noexcept {
    if(!entries_)return {};
    if(usage_.live_files)return {IoCode::invalid,"file Store still owns live values",0,0,usage_.live_files};
    if(auto e=collect())return e;
    read_ring_.close();write_ring_.close();
    const int dir=std::exchange(directory_,-1);entries_=nullptr;
    if(dir>=0&&::close(dir))return failed("file Store directory close");return {};
}
int FileStore::descriptor(const Region&r) const noexcept {const auto*e=entry(r);return e?e->fd:-1;}
IoError FileStore::create_thunk(void*p,const Region&r) noexcept{return static_cast<FileStore *>(p)->create(r);}
void FileStore::retire_thunk(void*p,const Region&r) noexcept{static_cast<FileStore *>(p)->retire(r);}
void FileStore::publish_thunk(void*p,const Region&r) noexcept{static_cast<FileStore *>(p)->published(r);}
IoError FileStore::attach_thunk(void*p,const Region&r) noexcept{return static_cast<FileStore *>(p)->attach(r);}
Completion FileStore::read_thunk(void*p,uint64_t at,std::span<uint8_t>w) noexcept{return static_cast<FileStore *>(p)->transfer(at,w.data(),w.size(),false);}
Completion FileStore::write_thunk(void*p,uint64_t at,std::span<const uint8_t>w) noexcept{return static_cast<FileStore *>(p)->transfer(at,const_cast<uint8_t *>(w.data()),w.size(),true);}
IoError FileStore::enable_uring(unsigned depth,unsigned workers) noexcept {
    if(!entries_||read_ring_.entries()||write_ring_.entries()||!os::Uring::memory_bound(depth))return {IoCode::invalid,"file io_uring preparation"};
    if(const int e=read_ring_.open(depth,workers))return failed("file read ring preparation",-e);
    if(const int e=write_ring_.open(depth,workers)){read_ring_.close();return failed("file write ring preparation",-e);}
    return {};
}
void FileStore::read_batch_thunk(void*ptr,std::span<ReadRequest> requests) noexcept {
    auto&f=*static_cast<FileStore*>(ptr);RingLock owner(f.read_ring_lock_);
    if(!f.entries_||requests.size()>f.read_ring_.entries()){for(auto&r:requests)r.result={IoCode::invalid,0,EINVAL};return;}
    os::UringRequest batch[io_batch_max]{};unsigned count=0,index[io_batch_max]{};
    for(unsigned i=0;i<requests.size();++i){auto&r=requests[i];r.result={IoCode::invalid,0,EINVAL};
        const uint64_t local=r.offset&(f.plan_.slot_bytes-1),slot=r.offset>>f.plan_.slot_shift;
        if(slot>=f.plan_.metadata.slots)continue;auto&e=f.entries_[slot];
        if(!e.live||e.retired||e.fd<0){r.result={IoCode::stale,0,EBADF};continue;}
        if(local>e.bytes||r.data.size()>e.bytes-local||r.data.size()>INT32_MAX)continue;
        index[count]=i;batch[count++]={e.fd,r.data.data(),local,unsigned(r.data.size()),false,0};
    }
    (void)f.read_ring_.transfer({batch,count});
    for(unsigned k=0;k<count;++k){auto&r=requests[index[k]];int n=batch[k].result;
        r.result=n<0?Completion{IoCode::io,0,-n}:Completion{IoCode::ok,size_t(n),0};
        __atomic_fetch_add(&f.usage_.read_calls,uint64_t(1),__ATOMIC_RELAXED);
        if(n>0)__atomic_fetch_add(&f.usage_.read_bytes,uint64_t(n),__ATOMIC_RELAXED);
    }
}
void FileStore::write_batch_thunk(void*ptr,std::span<WriteRequest> requests) noexcept {
    auto&f=*static_cast<FileStore*>(ptr);RingLock owner(f.write_ring_lock_);
    if(!f.entries_||requests.size()>f.write_ring_.entries()){for(auto&r:requests)r.result={IoCode::invalid,0,EINVAL};return;}
    os::UringRequest batch[io_batch_max]{};unsigned count=0,index[io_batch_max]{};
    for(unsigned i=0;i<requests.size();++i){auto&r=requests[i];r.result={IoCode::invalid,0,EINVAL};
        const uint64_t local=r.offset&(f.plan_.slot_bytes-1),slot=r.offset>>f.plan_.slot_shift;
        if(slot>=f.plan_.metadata.slots)continue;auto&e=f.entries_[slot];
        if(!e.live||e.retired||e.fd<0){r.result={IoCode::stale,0,EBADF};continue;}
        if(e.readonly){r.result={IoCode::invalid,0,EROFS};continue;}
        if(local>e.bytes||r.data.size()>e.bytes-local||r.data.size()>INT32_MAX)continue;
        __atomic_store_n(&e.synced,false,__ATOMIC_RELAXED);
        index[count]=i;batch[count++]={e.fd,const_cast<uint8_t*>(r.data.data()),local,unsigned(r.data.size()),true,0};
    }
    (void)f.write_ring_.transfer({batch,count});
    for(unsigned k=0;k<count;++k){auto&r=requests[index[k]];int n=batch[k].result;
        r.result=n<0?Completion{IoCode::io,0,-n}:Completion{IoCode::ok,size_t(n),0};
        __atomic_fetch_add(&f.usage_.write_calls,uint64_t(1),__ATOMIC_RELAXED);
        if(n>0)__atomic_fetch_add(&f.usage_.write_bytes,uint64_t(n),__ATOMIC_RELAXED);
    }
}
RegionLifecycle FileStore::lifecycle() noexcept{return {this,create_thunk,retire_thunk,publish_thunk,attach_thunk};}
Transport FileStore::transport(Cancel cancel) noexcept {Transport t{};t.context=this;t.read=read_thunk;t.write=write_thunk;
    t.capacity=plan_.address_bytes;t.alignment=4096;t.max_transfer=1u<<20;t.cancel=cancel;t.duplex=true;t.parallel_writes=t.parallel_reads=4;
    if(read_ring_.entries()&&write_ring_.entries()){t.read_batch=read_batch_thunk;t.write_batch=write_batch_thunk;t.batch_limit=read_ring_.entries();}
    return t;}
} // namespace sbn::v3::external
