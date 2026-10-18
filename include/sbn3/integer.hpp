// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/memory.hpp"
#include "sbn3/bitwise.h"
#include "sbn3/execution.hpp"
#include "sbn3/detail/bitwise.hpp"
#include "sbn3/detail/checks.hpp"
#include <algorithm>
#include <concepts>
#include <compare>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
namespace sbn {
class integer;class integer_buffer;class integer_ref;class borrowed_integer;
namespace detail {
template<class T>concept builtin_integer=std::integral<T>||std::same_as<T,__int128_t>||std::same_as<T,__uint128_t>;
template<class T>inline constexpr bool signed_integer=std::is_signed_v<T>||std::same_as<T,__int128_t>;
template<builtin_integer I>inline std::size_t checked_count(I v)noexcept{
    if constexpr(signed_integer<I>)require(v>=0,SBN3_FATAL_ARGUMENT,"negative bit count/index");
    if constexpr(sizeof(I)>sizeof(std::size_t))require(__uint128_t(v)<=SIZE_MAX,SBN3_FATAL_SIZE,"bit count/index overflow");
    return std::size_t(v);
}
struct canonical {};
constexpr std::uint64_t borrow_mask=0xffff,version_step=0x10000;
inline void writable(std::uint64_t*state)noexcept {if(state)require(!(__atomic_load_n(state,__ATOMIC_ACQUIRE)&borrow_mask),SBN3_FATAL_LIFETIME,"write borrowed integer");}
// Writers are externally exclusive, as for ordinary C++ containers. Version
// publication is a store, not a locked read-modify-write on every small op.
inline void changed(std::uint64_t*state)noexcept {if(state)__atomic_store_n(state,__atomic_load_n(state,__ATOMIC_RELAXED)+version_step,__ATOMIC_RELEASE);}
}
class integer_view {
    sbn3_int_view value_{};
    constexpr integer_view(sbn3_int_view v,detail::canonical)noexcept:value_(v){}
    friend class integer;friend class integer_buffer;friend class integer_ref;
public:
    constexpr integer_view()noexcept=default;
    explicit integer_view(std::span<const std::uint64_t> words,bool negative=false)noexcept {
        std::size_t n=words.size();while(n&&!words[n-1])--n;value_={words.data(),n,unsigned(n&&negative)};
    }
    constexpr const std::uint64_t*data()const noexcept{return value_.data;}
    constexpr std::size_t size()const noexcept{return value_.size;}
    constexpr bool negative()const noexcept{return value_.negative;}
    constexpr bool zero()const noexcept{return !value_.size;}
    constexpr std::span<const std::uint64_t>limbs()const noexcept{return {value_.data,value_.size};}
    constexpr sbn3_int_view c_view()const noexcept{return value_;}
};
class integer_ref {
    sbn3_int*value_;std::uint64_t*state_;
    friend class integer;friend class integer_buffer;
    constexpr integer_ref(sbn3_int*v,std::uint64_t*s)noexcept:value_(v),state_(s){}
public:
    constexpr std::size_t capacity()const noexcept{return value_->capacity;}
    constexpr bool same_object(integer_ref other)const noexcept{return value_==other.value_;}
    integer_view view()const noexcept{return {{value_->data,value_->size,value_->negative},detail::canonical{}};}
    // Interop is deliberately scoped by begin/finish; ordinary applications
    // use the *_into functions below. No value allocation is performed.
    sbn3_int*begin_write()const noexcept{detail::writable(state_);return value_;}
    void finish_write()const noexcept{detail::changed(state_);}
};
class integer_buffer {
    sbn3_int value_{};
public:
    explicit integer_buffer(std::span<std::uint64_t> storage)noexcept:value_{storage.data(),storage.size(),0,0}{}
    integer_buffer(std::span<std::uint64_t> storage,std::size_t used,bool sign=false)noexcept:value_{storage.data(),storage.size(),used,unsigned(sign)}{
        detail::require(used<=storage.size(),SBN3_FATAL_ARGUMENT,"integer buffer length",used,storage.size());sbn3_int_normalize(&value_);
    }
    integer_ref ref()&noexcept{return {&value_,nullptr};}
    integer_ref ref()&&=delete;
    integer_view view()const&noexcept{return {{value_.data,value_.size,value_.negative},detail::canonical{}};}
    integer_view view()const&&=delete;
    std::size_t size()const noexcept{return value_.size;}
    bool negative()const noexcept{return value_.negative;}
};
template<std::size_t N>class fixed_integer {
    std::uint64_t words_[N?N:1]{};integer_buffer buffer_;
public:
    fixed_integer()noexcept:buffer_(std::span<std::uint64_t>{words_,N}){}
    template<detail::builtin_integer I>fixed_integer(I x)noexcept:fixed_integer(){assign(x);}
    fixed_integer(const fixed_integer&other)noexcept:fixed_integer(){assign(other.view());}
    fixed_integer&operator=(const fixed_integer&other)noexcept{if(this!=&other)assign(other.view());return *this;}
    void assign(integer_view v)noexcept {
        detail::require(v.size()<=N,SBN3_FATAL_WORKSPACE,"fixed integer capacity",v.size(),N);
        auto*r=buffer_.ref().begin_write();if(v.size())std::memmove(r->data,v.data(),v.size()*8);r->size=v.size();r->negative=v.negative();
    }
    template<detail::builtin_integer I>void assign(I x)noexcept{
        bool sign=false;if constexpr(detail::signed_integer<I>)sign=x<0;
        __uint128_t u=__uint128_t(x);if(sign)u=0-u;
        const std::uint64_t words[2]{std::uint64_t(u),std::uint64_t(u>>64)};assign(integer_view{words,sign});
    }
    template<detail::builtin_integer I>fixed_integer&operator=(I x)noexcept{assign(x);return *this;}
    integer_ref ref()&noexcept{return buffer_.ref();}integer_ref ref()&&=delete;
    integer_view view()const&noexcept{return buffer_.view();}integer_view view()const&&=delete;
};
class integer {
    friend class borrowed_integer;
    sbn3_int value_;
    std::uint64_t small_[2]{};
    memory_resource*resource_=nullptr;
    mutable std::uint64_t state_=0; // lower 16 bits: managed readers; upper: generation
    void dispose()noexcept;
    void move_from(integer&)noexcept;
    void empty_inline()noexcept{value_={small_,2,0,0};}
public:
    integer()noexcept:value_{small_,2,0,0}{}
    explicit integer(memory_resource&r)noexcept:integer(){resource_=&r;}
    explicit integer(integer_view v,memory_resource&r=default_resource())noexcept:integer(r){assign(v);}
    template<detail::builtin_integer I>integer(I x)noexcept:integer(){
        bool sign=false;if constexpr(detail::signed_integer<I>)sign=x<0;
        __uint128_t v=__uint128_t(x);if(sign)v=__uint128_t(0)-v;
        small_[0]=std::uint64_t(v);small_[1]=std::uint64_t(v>>64);
        value_.size=small_[1]?2:bool(small_[0]);value_.negative=sign&&value_.size;
    }
    integer(const integer&x)noexcept:integer(){resource_=x.resource_;assign(x.view());}
    integer(integer&&x)noexcept:integer(){resource_=x.resource_;move_from(x);}
    ~integer(){detail::writable(&state_);dispose();}
    integer&operator=(const integer&)noexcept;
    integer&operator=(integer&&)noexcept;
    template<detail::builtin_integer I>integer&operator=(I x)noexcept{integer t{x};assign(t.view());return *this;}
    integer_view view()const&noexcept{return {{value_.data,value_.size,value_.negative},detail::canonical{}};}
    integer_view view()const&&=delete;
    integer_ref ref()&noexcept{detail::writable(&state_);return {&value_,&state_};}integer_ref ref()&&=delete;
    borrowed_integer borrow()const&noexcept;
    borrowed_integer borrow()const&&=delete;
    const std::uint64_t*data()const&noexcept{return value_.data;}
    const std::uint64_t*data()const&&=delete;
    std::span<const std::uint64_t>limbs()const&noexcept{return {data(),size()};}
    std::span<const std::uint64_t>limbs()const&&=delete;
    std::size_t size()const noexcept{return value_.size;}
    std::size_t capacity()const noexcept{return value_.capacity;}
    bool negative()const noexcept{return value_.negative;}
    bool zero()const noexcept{return !value_.size;}
    int sign()const noexcept{return zero()?0:negative()?-1:1;}
    explicit operator bool()const noexcept{return !zero();}
    memory_resource&resource()const noexcept{return resource_?*resource_:default_resource();}
    resource_error try_reserve(std::size_t limbs)noexcept;
    void reserve(std::size_t n)noexcept{auto e=try_reserve(n);detail::require(!e,SBN3_FATAL_WORKSPACE,e.where,e.need,e.have);}
    void assign(integer_view)noexcept;
    void clear()noexcept{detail::writable(&state_);value_.size=value_.negative=0;detail::changed(&state_);}
    void reset()noexcept{detail::writable(&state_);dispose();empty_inline();detail::changed(&state_);}
    void negate()noexcept{detail::writable(&state_);if(size())value_.negative^=1;detail::changed(&state_);}
    void swap(integer&)noexcept;
    template<detail::builtin_integer I>std::optional<I>try_to()const noexcept{
        if(size()>2)return std::nullopt;
        __uint128_t m=size()?data()[0]:0;if(size()>1)m|=__uint128_t(data()[1])<<64;
        constexpr unsigned width=std::same_as<I,bool>?1:sizeof(I)*8;
        if constexpr(detail::signed_integer<I>){const __uint128_t high=__uint128_t(1)<<(width-1);
            if(m>high-unsigned(!negative()))return std::nullopt;
            return I(negative()?__uint128_t(0)-m:m);
        }else{if(negative())return std::nullopt;
            if constexpr(width<128)if(m>=(__uint128_t(1)<<width))return std::nullopt;
            return I(m);}
    }
    std::size_t bit_length()const noexcept{return sbn3_int_bit_length(view().c_view());}
    std::size_t magnitude_popcount()const noexcept{return detail::word_population(data(),size());}
    bool test_bit(std::size_t b)const noexcept{return sbn3_int_test_bit(view().c_view(),b);}
    std::size_t scan_bit(std::size_t first,bool value)const noexcept{return sbn3_int_scan_bit(view().c_view(),first,value);}
    std::size_t count_ones(std::size_t first,std::size_t count)const noexcept{return sbn3_int_popcount_range(view().c_view(),first,count);}
    integer&set_bit(std::size_t,bool value=true)noexcept;
    integer&clear_bit(std::size_t b)noexcept{return set_bit(b,false);}
    integer&flip_bit(std::size_t b)noexcept{return set_bit(b,!test_bit(b));}
    template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
    bool test_bit(I b)const noexcept{return test_bit(detail::checked_count(b));}
    template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
    std::size_t scan_bit(I b,bool value)const noexcept{return scan_bit(detail::checked_count(b),value);}
    template<detail::builtin_integer I,detail::builtin_integer J>requires(!std::same_as<I,std::size_t>||!std::same_as<J,std::size_t>)
    std::size_t count_ones(I b,J n)const noexcept{return count_ones(detail::checked_count(b),detail::checked_count(n));}
    template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
    integer&set_bit(I b,bool value=true)noexcept{return set_bit(detail::checked_count(b),value);}
    template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
    integer&clear_bit(I b)noexcept{return clear_bit(detail::checked_count(b));}
    template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
    integer&flip_bit(I b)noexcept{return flip_bit(detail::checked_count(b));}
    integer&operator+=(const integer&)noexcept;
    integer&operator-=(const integer&)noexcept;
    integer&operator*=(const integer&)noexcept;
    integer&operator/=(const integer&)noexcept;
    integer&operator%=(const integer&)noexcept;
    integer&operator&=(const integer&)noexcept;
    integer&operator|=(const integer&)noexcept;
    integer&operator^=(const integer&)noexcept;
    integer&operator<<=(std::size_t)noexcept;
    integer&operator>>=(std::size_t)noexcept;
    template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
    integer&operator<<=(I b)noexcept{return operator<<=(detail::checked_count(b));}
    template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
    integer&operator>>=(I b)noexcept{return operator>>=(detail::checked_count(b));}
    integer&operator++()noexcept{*this+=1;return *this;}
    integer&operator--()noexcept{*this-=1;return *this;}
    integer operator++(int)noexcept{auto old=*this;++*this;return old;}
    integer operator--(int)noexcept{auto old=*this;--*this;return old;}
    // Internal value factory: resource preference survives eager expressions;
    // no default resource lookup occurs for inline values.
    static integer with_resource_of(const integer&x)noexcept{integer y;y.resource_=x.resource_;return y;}
    static integer from_parts(std::uint64_t lo,std::uint64_t hi,std::uint64_t top,bool neg,const integer&model)noexcept{
        integer r=with_resource_of(model);if(top)r.reserve(3);
        r.value_.data[0]=lo;r.value_.data[1]=hi;if(top)r.value_.data[2]=top;
        r.value_.size=top?3:hi?2:bool(lo);r.value_.negative=r.size()&&neg;return r;
    }
};
inline void swap(integer&a,integer&b)noexcept{a.swap(b);}
inline int compare(integer_view a,integer_view b)noexcept{return sbn3_int_compare(a.c_view(),b.c_view());}
inline std::size_t bit_length(integer_view a)noexcept{return sbn3_int_bit_length(a.c_view());}
inline std::size_t magnitude_popcount(integer_view a)noexcept{return detail::word_population(a.data(),a.size());}
template<detail::builtin_integer I>inline bool test_bit(integer_view a,I i)noexcept{return sbn3_int_test_bit(a.c_view(),detail::checked_count(i));}
template<detail::builtin_integer I>inline std::size_t scan_bit(integer_view a,I first,bool bit)noexcept{return sbn3_int_scan_bit(a.c_view(),detail::checked_count(first),bit);}
template<detail::builtin_integer I,detail::builtin_integer J>inline std::size_t count_ones(integer_view a,I first,J count)noexcept{return sbn3_int_popcount_range(a.c_view(),detail::checked_count(first),detail::checked_count(count));}
inline bool operator==(const integer&a,const integer&b)noexcept{return compare(a.view(),b.view())==0;}
inline std::strong_ordering operator<=>(const integer&a,const integer&b)noexcept{return compare(a.view(),b.view())<=>0;}
inline void add_into(integer_ref r,integer_view a,integer_view b)noexcept{auto*p=r.begin_write();sbn3_int_add(p,a.c_view(),b.c_view());r.finish_write();}
inline void subtract_into(integer_ref r,integer_view a,integer_view b)noexcept{auto*p=r.begin_write();sbn3_int_sub(p,a.c_view(),b.c_view());r.finish_write();}
inline void bitwise_into(integer_ref r,sbn3_bit_operation op,integer_view a,integer_view b)noexcept{auto*p=r.begin_write();sbn3_int_bitwise(p,op,a.c_view(),b.c_view());r.finish_write();}
namespace detail {
struct bit_parts {std::uint64_t lo,hi,top;bool negative;};
inline std::size_t parts_size(bit_parts x)noexcept{return x.top?3:x.hi?2:bool(x.lo);}
template<sbn3_bit_operation Op>inline bit_parts small_bits(integer_view a,integer_view b)noexcept{
    const bool negative=Op==SBN3_BIT_AND?a.negative()&&b.negative():Op==SBN3_BIT_OR?a.negative()||b.negative():a.negative()!=b.negative();
    if(a.size()<=1&&b.size()<=1){std::uint64_t x=a.size()?a.data()[0]:0,y=b.size()?b.data()[0]:0;if(a.negative())x=0-x;if(b.negative())y=0-y;
        const auto t=Op==SBN3_BIT_AND?x&y:Op==SBN3_BIT_OR?x|y:x^y;const auto z=negative?0-t:t;return {z,std::uint64_t(negative&&!z),0,negative};}
    auto load=[](integer_view x){return x.size()?__uint128_t(x.data()[0])|(x.size()>1?__uint128_t(x.data()[1])<<64:0):0;};
    auto x=load(a),y=load(b);if(a.negative())x=__uint128_t(0)-x;if(b.negative())y=__uint128_t(0)-y;
    const auto t=Op==SBN3_BIT_AND?x&y:Op==SBN3_BIT_OR?x|y:x^y;const auto z=negative?__uint128_t(0)-t:t;return {std::uint64_t(z),std::uint64_t(z>>64),std::uint64_t(negative&&!z),negative};
}
inline void check_bit_alias(sbn3_int*r,integer_view a,integer_view b,std::size_t n)noexcept{
    if constexpr(checked){auto overlap=[](const void*x,std::size_t nx,const void*y,std::size_t ny){if(!nx||!ny)return false;auto a=std::uintptr_t(x),b=std::uintptr_t(y);return a<=b?b-a<nx:a-b<ny;};
        require(!n||r->data,SBN3_FATAL_ARGUMENT,"null bit destination");
        for(auto v:{a,b})require((r->data==v.data()||!overlap(r->data,n*8,v.data(),v.size()*8))&&!overlap(r,sizeof*r,v.data(),v.size()*8),SBN3_FATAL_ARGUMENT,"C++ bit partial alias");
        require(!overlap(r,sizeof*r,r->data,n*8),SBN3_FATAL_ARGUMENT,"C++ bit descriptor alias");}
}
template<sbn3_bit_operation Op>inline void canonical_bitwise_into(integer_ref out,integer_view a,integer_view b)noexcept{
    auto*r=out.begin_write();
    if(a.size()<=2&&b.size()<=2){const auto x=small_bits<Op>(a,b);const auto n=parts_size(x);require(r->capacity>=n,SBN3_FATAL_WORKSPACE,"small bit result capacity",n,r->capacity);check_bit_alias(r,a,b,n);
        if(n)r->data[0]=x.lo;if(n>1)r->data[1]=x.hi;if(n>2)r->data[2]=x.top;r->size=n;r->negative=n&&x.negative;
    }else{
        const bool negative=Op==SBN3_BIT_AND?a.negative()&&b.negative():Op==SBN3_BIT_OR?a.negative()||b.negative():a.negative()!=b.negative();
        const auto n=Op==SBN3_BIT_XOR?std::max(a.size(),b.size()):Op==SBN3_BIT_AND?(a.negative()?(b.negative()?std::max(a.size(),b.size()):b.size()):b.negative()?a.size():std::min(a.size(),b.size())):
            a.negative()?(b.negative()?std::min(a.size(),b.size()):a.size()):b.negative()?b.size():std::max(a.size(),b.size());
        require(r->capacity>=n+unsigned(negative),SBN3_FATAL_WORKSPACE,"bit result capacity",n+unsigned(negative),r->capacity);check_bit_alias(r,a,b,n+unsigned(negative));
        constexpr auto high=std::size_t(1)<<63;const auto as=a.size()|(a.negative()?high:0),bs=b.size()|(b.negative()?high:0);
        std::size_t size;if constexpr(Op==SBN3_BIT_AND)size=bit_and(r->data,a.data(),as,b.data(),bs);else if constexpr(Op==SBN3_BIT_OR)size=bit_or(r->data,a.data(),as,b.data(),bs);else size=bit_xor(r->data,a.data(),as,b.data(),bs);
        r->size=size&~high;r->negative=size>>63;
    }
    out.finish_write();
}
}
inline void and_into(integer_ref r,integer_view a,integer_view b)noexcept{detail::canonical_bitwise_into<SBN3_BIT_AND>(r,a,b);}
inline void or_into(integer_ref r,integer_view a,integer_view b)noexcept{detail::canonical_bitwise_into<SBN3_BIT_OR>(r,a,b);}
inline void xor_into(integer_ref r,integer_view a,integer_view b)noexcept{detail::canonical_bitwise_into<SBN3_BIT_XOR>(r,a,b);}
inline void assign_into(integer_ref r,integer_view a)noexcept{auto*p=r.begin_write();detail::require(p->capacity>=a.size(),SBN3_FATAL_WORKSPACE,"integer destination capacity",a.size(),p->capacity);if(a.size())std::memmove(p->data,a.data(),a.size()*8);p->size=a.size();p->negative=a.negative();r.finish_write();}
inline void negate_into(integer_ref r,integer_view a)noexcept{assign_into(r,a);auto*p=r.begin_write();if(p->size)p->negative^=1;r.finish_write();}
inline void abs_into(integer_ref r,integer_view a)noexcept{assign_into(r,a);r.begin_write()->negative=0;r.finish_write();}
inline void complement_into(integer_ref r,integer_view a)noexcept{auto*p=r.begin_write();sbn3_int_complement(p,a.c_view());r.finish_write();}
inline void lshift_into(integer_ref r,integer_view a,std::size_t bits)noexcept{auto*p=r.begin_write();sbn3_int_lshift(p,a.c_view(),bits);r.finish_write();}
inline void rshift_into(integer_ref r,integer_view a,std::size_t bits)noexcept{auto*p=r.begin_write();sbn3_int_arshift(p,a.c_view(),bits);r.finish_write();}
inline void trunc_shift_into(integer_ref r,integer_view a,std::size_t bits)noexcept{auto*p=r.begin_write();sbn3_int_rshift(p,a.c_view(),bits);r.finish_write();}
template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
inline void lshift_into(integer_ref r,integer_view a,I bits)noexcept{lshift_into(r,a,detail::checked_count(bits));}
template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
inline void rshift_into(integer_ref r,integer_view a,I bits)noexcept{rshift_into(r,a,detail::checked_count(bits));}
template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
inline void trunc_shift_into(integer_ref r,integer_view a,I bits)noexcept{trunc_shift_into(r,a,detail::checked_count(bits));}
namespace detail {
inline __uint128_t magnitude128(const integer&a)noexcept{return a.size()?__uint128_t(a.data()[0])|(a.size()>1?__uint128_t(a.data()[1])<<64:0):0;}
integer large_sum(const integer&,const integer&,bool subtract)noexcept;
integer large_bitwise(const integer&,const integer&,sbn3_bit_operation)noexcept;
inline integer sum(const integer&a,const integer&b,bool sub)noexcept{
    if(a.size()>2||b.size()>2)return large_sum(a,b,sub);
    const __uint128_t x=magnitude128(a),y=magnitude128(b);const bool sa=a.negative(),sb=b.negative()!=(sub&&!b.zero());
    __uint128_t z;bool sign;std::uint64_t top=0;
    if(sa==sb){z=x+y;top=z<x;sign=sa;}else{sign=x>=y?sa:sb;z=x>=y?x-y:y-x;}
    return integer::from_parts(std::uint64_t(z),std::uint64_t(z>>64),top,sign,a);
}
template<sbn3_bit_operation Op>inline integer bitwise(const integer&a,const integer&b)noexcept{
    if(&a==&b){if constexpr(Op==SBN3_BIT_XOR)return integer::with_resource_of(a);else return integer(a);}
    if(a.size()>2||b.size()>2)return large_bitwise(a,b,Op);
    const auto p=small_bits<Op>(a.view(),b.view());return integer::from_parts(p.lo,p.hi,p.top,p.negative,a);
}
}
requirements multiply_requirements(integer_ref,integer_view,integer_view,unsigned workers=1)noexcept;
requirements square_requirements(integer_ref,integer_view,unsigned workers=1)noexcept;
void mul_into(integer_ref,integer_view,integer_view,execution={})noexcept;
void square_into(integer_ref,integer_view,execution={})noexcept;
integer multiply(const integer&,const integer&)noexcept;
integer square(const integer&)noexcept;
inline integer operator*(const integer&a,const integer&b)noexcept{
    if(a.zero()||b.zero())return integer::with_resource_of(a);
    if(a.size()==1&&b.size()==1){const auto z=__uint128_t(a.data()[0])*b.data()[0];return integer::from_parts(std::uint64_t(z),std::uint64_t(z>>64),0,a.negative()!=b.negative(),a);}
    return multiply(a,b);
}
inline integer operator+(const integer&a,const integer&b)noexcept{return detail::sum(a,b,false);}
inline integer operator-(const integer&a,const integer&b)noexcept{return detail::sum(a,b,true);}
inline integer operator&(const integer&a,const integer&b)noexcept{return detail::bitwise<SBN3_BIT_AND>(a,b);}
inline integer operator|(const integer&a,const integer&b)noexcept{return detail::bitwise<SBN3_BIT_OR>(a,b);}
inline integer operator^(const integer&a,const integer&b)noexcept{return detail::bitwise<SBN3_BIT_XOR>(a,b);}
inline integer operator-(integer a)noexcept{a.negate();return a;}
inline integer operator+(integer a)noexcept{return a;}
inline integer abs(integer a)noexcept{if(a.negative())a.negate();return a;}
inline integer operator~(const integer&a)noexcept{
    if(a.size()<=2){const auto x=detail::magnitude128(a);const auto z=a.negative()?x-1:x+1;
        return integer::from_parts(std::uint64_t(z),std::uint64_t(z>>64),!a.negative()&&!z,!a.negative(),a);}
    auto r=integer::with_resource_of(a);r.reserve(a.size()+1);complement_into(r.ref(),a.view());return r;
}
inline integer operator<<(integer a,std::size_t bits)noexcept{a<<=bits;return a;}
inline integer operator>>(integer a,std::size_t bits)noexcept{a>>=bits;return a;}
template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
inline integer operator<<(integer a,I bits)noexcept{a<<=detail::checked_count(bits);return a;}
template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
inline integer operator>>(integer a,I bits)noexcept{a>>=detail::checked_count(bits);return a;}
// / and % truncate toward zero, as for built-in signed integers. Named
// rounding modes, paired results and retained divisors are in division.hpp.
integer operator/(const integer&,const integer&)noexcept;
integer operator%(const integer&,const integer&)noexcept;
inline integer trunc_shift(const integer&a,std::size_t bits)noexcept{auto r=integer::with_resource_of(a);r.reserve(a.size());trunc_shift_into(r.ref(),a.view(),bits);return r;}
template<detail::builtin_integer I>requires(!std::same_as<I,std::size_t>)
inline integer trunc_shift(const integer&a,I bits)noexcept{return trunc_shift(a,detail::checked_count(bits));}
}
