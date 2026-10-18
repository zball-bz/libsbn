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
    void *storage() noexcept { return bytes_; }
    T &emplace(size_t i) noexcept { return *::new(bytes_+i*sizeof(T)) T{}; }
    T &emplace(size_t i,const T &value) noexcept { return *::new(bytes_+i*sizeof(T)) T(value); }
    T &operator[](size_t i) noexcept { return *std::launder(reinterpret_cast<T *>(bytes_+i*sizeof(T))); }
    const T &operator[](size_t i) const noexcept { return *std::launder(reinterpret_cast<const T *>(bytes_+i*sizeof(T))); }
    size_t index_of(const T *p) const noexcept {
        return size_t(reinterpret_cast<const unsigned char *>(p)-bytes_)/sizeof(T);
    }
};
// Non-owning directory into caller-provided plan storage. Query may point it
// at bounded stack capacity; a binding keeps only its live entries.
template<class T,size_t N> class PlanView {
    static_assert(std::is_trivially_destructible_v<T>);
    unsigned char *bytes_=nullptr;
public:
    void reset(void *p) noexcept { bytes_=static_cast<unsigned char *>(p); }
    T &emplace(size_t i) noexcept { return *::new(bytes_+i*sizeof(T)) T{}; }
    T &emplace(size_t i,const T &v) noexcept { return *::new(bytes_+i*sizeof(T)) T(v); }
    T &operator[](size_t i) noexcept {return *std::launder(reinterpret_cast<T *>(bytes_+i*sizeof(T)));}
    const T &operator[](size_t i)const noexcept {return *std::launder(reinterpret_cast<const T *>(bytes_+i*sizeof(T)));}
    size_t index_of(const T *p)const noexcept {return size_t(reinterpret_cast<const unsigned char *>(p)-bytes_)/sizeof(T);}
};
}
