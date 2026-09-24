#pragma once
#include <new>
#include <stddef.h>
#include <type_traits>
namespace sbn::v3::radix {
// Fixed-capacity storage for plans whose live counts are held by the owner.
// Only emplaced entries may be read. No allocation and no initialization of
// unused capacity; all element types must have trivial destruction.
template<class T,size_t N> class PlanSlots {
    static_assert(std::is_trivially_destructible_v<T>);
    alignas(T) unsigned char bytes_[sizeof(T)*N];
public:
    PlanSlots() noexcept {}
    PlanSlots(const PlanSlots &)=delete;
    PlanSlots &operator=(const PlanSlots &)=delete;
    T &emplace(size_t i) noexcept { return *::new(bytes_+i*sizeof(T)) T{}; }
    T &emplace(size_t i,const T &value) noexcept { return *::new(bytes_+i*sizeof(T)) T(value); }
    T &operator[](size_t i) noexcept { return *std::launder(reinterpret_cast<T *>(bytes_+i*sizeof(T))); }
    const T &operator[](size_t i) const noexcept { return *std::launder(reinterpret_cast<const T *>(bytes_+i*sizeof(T))); }
    size_t index_of(const T *p) const noexcept {
        return size_t(reinterpret_cast<const unsigned char *>(p)-bytes_)/sizeof(T);
    }
};
}
