#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
static unsigned enabled;
static uint64_t calls;
static void observe(){if(__atomic_load_n(&enabled,__ATOMIC_ACQUIRE))__atomic_add_fetch(&calls,1,__ATOMIC_RELAXED);}
extern "C" void allocation_watch_start(){__atomic_store_n(&calls,0,__ATOMIC_RELAXED);__atomic_store_n(&enabled,1,__ATOMIC_RELEASE);}
extern "C" uint64_t allocation_watch_stop(){__atomic_store_n(&enabled,0,__ATOMIC_RELEASE);return __atomic_load_n(&calls,__ATOMIC_ACQUIRE);}
extern "C" {
void *__real_malloc(size_t);void *__real_calloc(size_t,size_t);void *__real_realloc(void*,size_t);void __real_free(void*);
void *__real_aligned_alloc(size_t,size_t);int __real_posix_memalign(void**,size_t,size_t);
void *__real_mmap(void*,size_t,int,int,int,off_t);int __real_mprotect(void*,size_t,int);
int __real_mlock(const void*,size_t);int __real_munlock(const void*,size_t);int __real_madvise(void*,size_t,int);
void *__wrap_malloc(size_t n){observe();return __real_malloc(n);}
void *__wrap_calloc(size_t n,size_t s){observe();return __real_calloc(n,s);}
void *__wrap_realloc(void*p,size_t n){observe();return __real_realloc(p,n);}
void __wrap_free(void*p){observe();__real_free(p);}
void *__wrap_aligned_alloc(size_t a,size_t n){observe();return __real_aligned_alloc(a,n);}
int __wrap_posix_memalign(void **p,size_t a,size_t n){observe();return __real_posix_memalign(p,a,n);}
void *__wrap_mmap(void*p,size_t n,int prot,int flags,int fd,off_t o){observe();return __real_mmap(p,n,prot,flags,fd,o);}
int __wrap_mprotect(void*p,size_t n,int prot){observe();return __real_mprotect(p,n,prot);}
int __wrap_mlock(const void*p,size_t n){observe();return __real_mlock(p,n);}
int __wrap_munlock(const void*p,size_t n){observe();return __real_munlock(p,n);}
int __wrap_madvise(void*p,size_t n,int advice){observe();return __real_madvise(p,n,advice);}
}
