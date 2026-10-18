// Original libsbn code; LGPL-3.0-or-later. Allocation/OS boundary, not arithmetic.
#include "sbn3/memory.hpp"
#include "common/checked.hpp"
#include "cpp/asan.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <new>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
namespace sbn {
namespace detail {[[noreturn]]void fail(sbn3_fatal_kind k,const char*w,std::uint64_t n,std::uint64_t h)noexcept{sbn::v3::fatal(k,w,n,h);}}
namespace {
using U=std::uintptr_t;
constexpr std::size_t slab_bytes=128u<<10,small_limit=32u<<10;
constexpr unsigned classes=10;
struct alignas(64) Slab {
    Slab*next;Slab*prev;void*free;std::size_t live,block;unsigned kind;std::uint64_t magic;
};
static_assert(sizeof(Slab)==64);
constexpr std::uint64_t slab_magic=0x73626e33706f6f6cULL;
std::size_t rounded(std::size_t n,std::size_t a){return (n+a-1)&~(a-1);}
resource_error bad(resource_errc c,const char*w,std::size_t n=0,std::size_t h=0,int e=0){return {c,n,h,e,w};}
bool alignment_ok(std::size_t a){return a&&!(a&(a-1))&&a<=(std::size_t(1)<<30);}
void*map_aligned(std::size_t bytes,std::size_t align,int&error){
    const std::size_t page=std::size_t(sysconf(_SC_PAGESIZE));align=std::max(align,page);
    if(bytes>SIZE_MAX-align){error=ENOMEM;return nullptr;}
    const std::size_t total=bytes+align;void*raw=mmap(nullptr,total,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(raw==MAP_FAILED){error=errno;return nullptr;}
    const U begin=U(raw),at=rounded(begin,align);const auto prefix=std::size_t(at-begin),suffix=total-prefix-bytes;
    if(prefix)detail::require(!munmap(raw,prefix),SBN3_FATAL_LIFETIME,"value resource leading trim");
    if(suffix)detail::require(!munmap(reinterpret_cast<void*>(at+bytes),suffix),SBN3_FATAL_LIFETIME,"value resource trailing trim");
    auto*p=reinterpret_cast<unsigned char*>(at);
    if(mprotect(p,bytes,PROT_READ|PROT_WRITE)){error=errno;munmap(p,bytes);return nullptr;}
    // An allocation boundary performs first touch; hot arithmetic never does
    // explicit mapping or page preparation. Caller can instead supply pinned storage.
    for(std::size_t i=0;i<bytes;i+=page)p[i]=0;
    return p;
}
struct Lock {pthread_mutex_t&m;explicit Lock(pthread_mutex_t&x):m(x){detail::require(!pthread_mutex_lock(&m),SBN3_FATAL_LIFETIME,"value resource lock");}~Lock(){detail::require(!pthread_mutex_unlock(&m),SBN3_FATAL_LIFETIME,"value resource unlock");}};
class Pool final:public memory_resource {
    mutable pthread_mutex_t mutex_=PTHREAD_MUTEX_INITIALIZER;
    Slab*available_[classes]{};Slab*empty_[classes]{};resource_statistics stats_{};
    void unlink(Slab*s){if(s->prev)s->prev->next=s->next;else available_[s->kind]=s->next;if(s->next)s->next->prev=s->prev;s->next=s->prev=nullptr;}
    void prepend(Slab*s){auto*&head=available_[s->kind];s->prev=nullptr;s->next=head;if(head)head->prev=s;head=s;}
public:
    constexpr Pool()=default;
    resource_error try_allocate(std::size_t bytes,std::size_t align,allocation&out)noexcept override {
        out={};if(!alignment_ok(align))return bad(resource_errc::capacity,"value alignment",align);
        if(!bytes)return {};
        Lock lock(mutex_);std::size_t actual=bytes;void*p=nullptr;
        if(bytes<=small_limit&&align<=small_limit){
            actual=64;unsigned k=0;while(actual<std::max(bytes,align)){actual*=2;++k;}
            auto*s=available_[k];
            if(!s){int error=0;void*memory=map_aligned(slab_bytes,slab_bytes,error);if(!memory)return bad(resource_errc::no_memory,"value slab mapping",slab_bytes,0,error);
                s=::new(memory)Slab{};s->block=actual;s->kind=k;s->magic=slab_magic;
                const auto first=rounded(sizeof(Slab),actual);auto*b=static_cast<unsigned char*>(memory);
                for(std::size_t at=slab_bytes;at>first;){at-=actual;*reinterpret_cast<void**>(b+at)=s->free;s->free=b+at;SBN_CPP_POISON(b+at,actual);}
                prepend(s);stats_.resident_bytes+=slab_bytes;stats_.peak_resident_bytes=std::max(stats_.peak_resident_bytes,stats_.resident_bytes);
            }
            if(empty_[k]==s)empty_[k]=nullptr;
            p=s->free;SBN_CPP_UNPOISON(p,actual);s->free=*static_cast<void**>(p);++s->live;if(!s->free)unlink(s);
        }else{
            const std::size_t page=std::size_t(sysconf(_SC_PAGESIZE));if(bytes>SIZE_MAX-page)return bad(resource_errc::capacity,"value mapping extent",bytes);
            actual=rounded(bytes,page);int error=0;p=map_aligned(actual,align,error);if(!p)return bad(resource_errc::no_memory,"value mapping",actual,0,error);
            stats_.resident_bytes+=actual;stats_.peak_resident_bytes=std::max(stats_.peak_resident_bytes,stats_.resident_bytes);
        }
        stats_.live_bytes+=actual;++stats_.live_allocations;out={p,actual,align};return {};
    }
    void deallocate(allocation a)noexcept override {
        if(!a.data)return;Lock lock(mutex_);stats_.live_bytes-=a.bytes;--stats_.live_allocations;
        if(a.bytes<=small_limit&&a.alignment<=small_limit){
            auto*s=reinterpret_cast<Slab*>(U(a.data)&~U(slab_bytes-1));
            detail::require(s->magic==slab_magic&&s->block==a.bytes&&s->live,SBN3_FATAL_LIFETIME,"value pool allocation provenance");
            if(!s->free)prepend(s);*static_cast<void**>(a.data)=s->free;s->free=a.data;SBN_CPP_POISON(a.data,a.bytes);--s->live;
            if(!s->live){if(empty_[s->kind]){unlink(s);detail::require(!munmap(s,slab_bytes),SBN3_FATAL_LIFETIME,"value slab release");stats_.resident_bytes-=slab_bytes;}else empty_[s->kind]=s;}
        }else{detail::require(!munmap(a.data,a.bytes),SBN3_FATAL_LIFETIME,"value mapping release");stats_.resident_bytes-=a.bytes;}
    }
    resource_statistics statistics()const noexcept override{Lock lock(mutex_);return stats_;}
    resource_error release_unused()noexcept override {
        Lock lock(mutex_);for(unsigned k=0;k<classes;++k)if(auto*s=empty_[k]){unlink(s);empty_[k]=nullptr;detail::require(!munmap(s,slab_bytes),SBN3_FATAL_LIFETIME,"unused value slab release");stats_.resident_bytes-=slab_bytes;}return {};
    }
};
// Do not destroy an allocator before global integers in another translation
// unit. OS mappings are reclaimed by the process; release_unused is explicit.
[[clang::no_destroy]] constinit Pool default_pool;
struct alignas(64) Block {std::size_t bytes;Block*prev;Block*next;bool free;};
static_assert(sizeof(Block)==64);
}
memory_resource&default_resource()noexcept{return default_pool;}
buffer_resource::buffer_resource(std::span<std::byte>b)noexcept {
    const U at=rounded(U(b.data()),64);const auto lost=std::size_t(at-U(b.data()));
    if(lost>b.size()||b.size()-lost<sizeof(Block))return;
    base_=reinterpret_cast<void*>(at);bytes_=(b.size()-lost)&~std::size_t(63);auto*h=::new(base_)Block{};h->bytes=bytes_;h->free=true;SBN_CPP_POISON(static_cast<unsigned char*>(base_)+sizeof(Block),bytes_-sizeof(Block));
}
buffer_resource::~buffer_resource(){detail::require(!count_,SBN3_FATAL_LIFETIME,"buffer resource destroyed with live values",count_);if(base_)SBN_CPP_UNPOISON(base_,bytes_);}
resource_error buffer_resource::try_allocate(std::size_t n,std::size_t align,allocation&out)noexcept {
    out={};if(!n)return {};if(!alignment_ok(align)||n>SIZE_MAX-128)return bad(resource_errc::capacity,"buffer value request",n,bytes_);
    align=std::max(align,std::size_t(64));n=rounded(n,64);
    std::size_t best=0;
    for(auto*h=static_cast<Block*>(base_);h;h=h->next)if(h->free){
        const U start=U(h),payload=rounded(start+sizeof(Block),align);const auto front=std::size_t(payload-sizeof(Block)-start);
        if(front>h->bytes||h->bytes-front<sizeof(Block))continue;
        best=std::max(best,h->bytes-front-sizeof(Block));
        if(h->bytes-front<sizeof(Block)+n)continue;
        if(front){SBN_CPP_UNPOISON(reinterpret_cast<void*>(start+front),sizeof(Block));auto*chosen=::new(reinterpret_cast<void*>(start+front))Block{h->bytes-front,h,h->next,true};if(h->next)h->next->prev=chosen;h->next=chosen;h->bytes=front;h=chosen;}
        const std::size_t take=sizeof(Block)+n,left=h->bytes-take;
        if(left>=2*sizeof(Block)){SBN_CPP_UNPOISON(reinterpret_cast<void*>(U(h)+take),sizeof(Block));auto*tail=::new(reinterpret_cast<void*>(U(h)+take))Block{left,h,h->next,true};if(h->next)h->next->prev=tail;h->next=tail;h->bytes=take;}
        h->free=false;const auto size=h->bytes-sizeof(Block);live_+=size;++count_;peak_=std::max(peak_,live_);out={reinterpret_cast<void*>(U(h)+sizeof(Block)),size,align};SBN_CPP_UNPOISON(out.data,out.bytes);return {};
    }
    return bad(resource_errc::capacity,"buffer resource exhausted",n,best);
}
void buffer_resource::deallocate(allocation a)noexcept {
    if(!a.data)return;auto*h=reinterpret_cast<Block*>(U(a.data)-sizeof(Block));
    detail::require(U(h)>=U(base_)&&U(h)-U(base_)<bytes_&&!h->free&&h->bytes-sizeof(Block)==a.bytes,SBN3_FATAL_LIFETIME,"buffer allocation provenance");
    live_-=a.bytes;--count_;h->free=true;
    if(h->next&&h->next->free){auto*next=h->next;h->bytes+=next->bytes;h->next=next->next;if(h->next)h->next->prev=h;}
    if(h->prev&&h->prev->free){auto*prev=h->prev;prev->bytes+=h->bytes;prev->next=h->next;if(h->next)h->next->prev=prev;h=prev;}
    SBN_CPP_POISON(reinterpret_cast<void*>(U(h)+sizeof(Block)),h->bytes-sizeof(Block));
}
}
