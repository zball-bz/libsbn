// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/base.h"
#include <cstddef>
#include <cstdint>
#include <span>
namespace sbn {
namespace detail {
[[noreturn]] void fail(sbn3_fatal_kind,const char*,std::uint64_t need=0,std::uint64_t have=0) noexcept;
inline void require(bool ok,sbn3_fatal_kind kind,const char*where,std::uint64_t need=0,std::uint64_t have=0) noexcept {
    if(!ok)[[unlikely]]fail(kind,where,need,have);
}
}
enum class resource_errc {none,capacity,no_memory,system,busy};
struct resource_error {
    resource_errc code=resource_errc::none;
    std::size_t need=0,have=0;int system_error=0;const char*where="";
    explicit operator bool()const noexcept{return code!=resource_errc::none;}
};
struct allocation {void*data=nullptr;std::size_t bytes=0,alignment=0;};
struct resource_statistics {std::size_t live_bytes=0,live_allocations=0,resident_bytes=0,peak_resident_bytes=0;};
// Explicit allocation boundary. Implementations do not get called from limb
// loops or admitted *_into operations. A supplied resource must outlive every
// value/work object referring to it, including inline values with no allocation.
class memory_resource {
public:
    memory_resource()=default;memory_resource(const memory_resource&)=delete;
    memory_resource&operator=(const memory_resource&)=delete;
    virtual ~memory_resource()=default;
    virtual resource_error try_allocate(std::size_t bytes,std::size_t alignment,allocation&) noexcept=0;
    virtual void deallocate(allocation) noexcept=0;
    virtual resource_statistics statistics()const noexcept=0;
    virtual resource_error release_unused()noexcept{return {};}
    allocation allocate(std::size_t bytes,std::size_t alignment=64)noexcept {
        allocation a;const auto e=try_allocate(bytes,alignment,a);
        detail::require(!e,SBN3_FATAL_WORKSPACE,e.where,e.need,e.have);return a;
    }
};
// Process-lifetime, thread-safe, serial convenience provider. Value pages are
// prefaulted, not locked. At most one empty 128 KiB slab per size class is
// retained. release_unused() drops those slabs. No implicit Team or plan cache.
memory_resource& default_resource() noexcept;
// An explicit fixed span; constructor/destructor and calls are single-owner.
// No OS calls, growth, or heap allocation. Arbitrary frees coalesce. Values
// must die before this resource and the caller's backing storage/lease.
class buffer_resource final:public memory_resource {
    void*base_=nullptr;std::size_t bytes_=0,live_=0,count_=0,peak_=0;
public:
    explicit buffer_resource(std::span<std::byte>)noexcept;
    ~buffer_resource()override;
    resource_error try_allocate(std::size_t,std::size_t,allocation&)noexcept override;
    void deallocate(allocation)noexcept override;
    resource_statistics statistics()const noexcept override{return {live_,count_,bytes_,peak_};}
};
}
