#pragma once
#include <stdint.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
namespace sbn::v3::os {
inline void pause() noexcept {__asm__ volatile("pause" ::: "memory");}
inline void publish_stores() noexcept {__asm__ volatile("sfence" ::: "memory");}
inline void wake(uint32_t *p,int n=1) noexcept {(void)syscall(SYS_futex,p,FUTEX_WAKE_PRIVATE,n,nullptr,nullptr,0);}
inline void wait_changed(uint32_t *p,uint32_t value) noexcept {
    for(unsigned i=0;i<512;++i) {
        if(__atomic_load_n(p,__ATOMIC_ACQUIRE)!=value)return;
        pause();
    }
    while(__atomic_load_n(p,__ATOMIC_ACQUIRE)==value)
        (void)syscall(SYS_futex,p,FUTEX_WAIT_PRIVATE,value,nullptr,nullptr,0);
}
inline void wait_equal(uint32_t *p,uint32_t value) noexcept {
    for(;;) {uint32_t now=__atomic_load_n(p,__ATOMIC_ACQUIRE);if(now==value)return;wait_changed(p,now);}
}
/* Paired epoch/park flag protocol from the v2 persistent pool. The two
 * sequentially consistent operations prevent the producer and consumer
 * from both missing the other's publication before FUTEX_WAIT. */
inline uint64_t now_ns() noexcept {timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
inline void wait_epoch(uint32_t *p,uint32_t value,uint32_t *parked) noexcept {
    if(__atomic_load_n(p,__ATOMIC_ACQUIRE)!=value)return;
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
    constexpr uint64_t spin_ns=2000;
#else
    constexpr uint64_t spin_ns=100000;
#endif
#else
    constexpr uint64_t spin_ns=100000;
#endif
    const uint64_t begin=now_ns();
    do {
        for(unsigned i=0;i<512;++i){if(__atomic_load_n(p,__ATOMIC_ACQUIRE)!=value)return;pause();}
    } while(now_ns()-begin<spin_ns);
    __atomic_store_n(parked,1,__ATOMIC_SEQ_CST);
    while(__atomic_load_n(p,__ATOMIC_SEQ_CST)==value)
        (void)syscall(SYS_futex,p,FUTEX_WAIT_PRIVATE,value,nullptr,nullptr,0);
    __atomic_store_n(parked,0,__ATOMIC_RELAXED);
}
inline void publish_epoch(uint32_t *p,uint32_t value,uint32_t *parked) noexcept {
    __atomic_store_n(p,value,__ATOMIC_SEQ_CST);
    if(__atomic_load_n(parked,__ATOMIC_SEQ_CST))wake(p);
}
}
