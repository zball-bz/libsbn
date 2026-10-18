// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "ooc_store_support.hpp"
#include "external/file_store.hpp"
#include <fcntl.h>
#include <linux/magic.h>
#include <sys/statfs.h>
struct FileDevice {
    static SlotStorePlan layout(size_t slots,uint64_t bytes){SlotStorePlan p;assert(query_slot_store(slots,bytes,p));return p;}
    static FileStoreStorage storage(size_t slots){FileStoreStorage p;assert(query_file_store(slots,p));return p;}
    SlotStorePlan plan;
    Aligned values,registry;
    FileStore files;
    ValueStore store;
    Transport inner{};
    uint64_t writes=0,reads=0,fail_write=0,fail_read=0,cancel_after_write=0;
    bool cancelled=false,closed=false;
    char directory[128]{};
    FileDevice(size_t slots,uint64_t bytes,bool named=false,bool direct=true):
        plan(layout(slots,bytes)),values(plan.metadata.bytes),registry(storage(slots).bytes){
        if(named){
            const char*root=nullptr;struct statfs fs{};
            for(const char*p:{"/tmp","/dev/shm"})if(!statfs(p,&fs)&&fs.f_type==TMPFS_MAGIC&&!access(p,W_OK|X_OK)){root=p;break;}
            assert(root&&"file tests require tmpfs; never fall back to SSD");
            const int count=snprintf(directory,sizeof directory,"%s/libsbn-file-test-XXXXXX",root);assert(count>0&&size_t(count)<sizeof directory);
            assert(mkdtemp(directory));const int fd=open(directory,O_RDONLY|O_DIRECTORY|O_CLOEXEC);assert(fd>=0);
            const auto e=files.open_directory(plan,registry.span(),fd,direct);assert(!e);assert(!::close(fd));
        }else assert(!files.open_memory(plan,registry.span()));
        assert(!store.open_slots(0x711,0x912,plan,values.span(),files.lifecycle()));inner=files.transport();
    }
    ~FileDevice(){assert(closed);}
    static Completion read(void*p,uint64_t at,std::span<uint8_t>w) noexcept {
        auto&d=*static_cast<FileDevice *>(p);if(__atomic_add_fetch(&d.reads,uint64_t(1),__ATOMIC_RELAXED)==d.fail_read)return {IoCode::io,0,EIO};return d.inner.read(d.inner.context,at,w);
    }
    static Completion write(void*p,uint64_t at,std::span<const uint8_t>w) noexcept {
        auto&d=*static_cast<FileDevice *>(p);const auto ticket=__atomic_add_fetch(&d.writes,uint64_t(1),__ATOMIC_RELAXED);
        if(ticket==d.fail_write)return {IoCode::io,0,ENOSPC};
        const auto c=d.inner.write(d.inner.context,at,w);if(ticket==d.cancel_after_write)__atomic_store_n(&d.cancelled,true,__ATOMIC_RELEASE);return c;
    }
    static bool cancel(void*p) noexcept{return __atomic_load_n(&static_cast<FileDevice *>(p)->cancelled,__ATOMIC_ACQUIRE);}
    static void read_many(void*p,std::span<ReadRequest>rs) noexcept {
        auto&d=*static_cast<FileDevice*>(p);
        if(d.fail_read){for(auto&r:rs)r.result=read(p,r.offset,r.data);return;}
        __atomic_fetch_add(&d.reads,uint64_t(rs.size()),__ATOMIC_RELAXED);d.inner.read_batch(d.inner.context,rs);
    }
    static void write_many(void*p,std::span<WriteRequest>rs) noexcept {
        auto&d=*static_cast<FileDevice*>(p);
        if(d.fail_write||d.cancel_after_write){for(auto&r:rs)r.result=write(p,r.offset,r.data);return;}
        __atomic_fetch_add(&d.writes,uint64_t(rs.size()),__ATOMIC_RELAXED);d.inner.write_batch(d.inner.context,rs);
    }
    Transport transport(){auto t=inner;t.context=this;t.read=read;t.write=write;t.cancel={this,cancel};
        t.read_batch=inner.read_batch?read_many:nullptr;t.write_batch=inner.write_batch?write_many:nullptr;return t;}
    void close(){assert(!store.usage().live_regions);assert(!files.close());if(directory[0])assert(!rmdir(directory));closed=true;}
};
