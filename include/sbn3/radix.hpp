// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/integer.hpp"
#include <string_view>
namespace sbn {
// Default alphabet matches radix.h: 0-9 a-z through base 36; above it,
// 0-9 A-Z a-z + /. A custom alphabet is borrowed for the duration of a call.
struct radix {
    unsigned base=10;std::string_view alphabet{};
    constexpr radix(unsigned b=10)noexcept:base(b){}
    explicit constexpr radix(std::string_view a)noexcept:base(unsigned(a.size())),alphabet(a){}
};
enum class conversion_errc {none,empty,invalid_digit,invalid_radix,unsupported,capacity};
struct conversion_error {
    conversion_errc code=conversion_errc::none;std::size_t index=0;
    explicit operator bool()const noexcept{return code!=conversion_errc::none;}
};
struct text_requirements : requirements {std::size_t characters=0;};
struct format_result {std::string_view text{};conversion_error error{};explicit operator bool()const noexcept{return !error;}};
struct integer_parse_result {integer value;conversion_error error{};explicit operator bool()const noexcept{return !error;}};
// Owns text through the same resource model as integer. Small outputs need
// no allocation. view/data/c_str are lvalue-only borrows; the value is copyable.
class integer_text {
    alignas(64) char small_[192];
    memory_resource*resource_=nullptr;allocation block_{};std::size_t offset_=0,size_=0;
    char*storage()noexcept{return block_.data?static_cast<char*>(block_.data):small_;}
    const char*storage()const noexcept{return block_.data?static_cast<const char*>(block_.data):small_;}
    void reserve(std::size_t)noexcept;
    friend integer_text to_string(const integer&,radix)noexcept;
public:
    explicit integer_text(memory_resource&r=default_resource())noexcept:resource_(&r){small_[0]=0;}
    integer_text(const integer_text&)noexcept;
    integer_text(integer_text&&)noexcept;
    integer_text&operator=(const integer_text&)noexcept;
    integer_text&operator=(integer_text&&)noexcept;
    ~integer_text(){if(resource_)resource_->deallocate(block_);}
    std::size_t size()const noexcept{return size_;}
    bool empty()const noexcept{return !size_;}
    std::string_view view()const&noexcept{return {storage()+offset_,size_};}
    std::string_view view()const&&=delete;
    const char*data()const&noexcept{return storage()+offset_;}const char*data()const&&=delete;
    const char*c_str()const&noexcept{return data();}const char*c_str()const&&=delete;
    operator std::string_view()const&noexcept{return view();}
    operator std::string_view()const&&=delete;
};
// Includes NUL and backend output padding. The returned view can start inside
// the span, so formatting need not copy a large digit array to remove padding.
text_requirements format_requirements(integer_view,radix={},unsigned workers=1)noexcept;
format_result format_into(std::span<char>,integer_view,radix={},execution={})noexcept;
integer_text to_string(const integer&,radix={})noexcept;
requirements parse_requirements(integer_ref,std::string_view,radix={},unsigned workers=1)noexcept;
// Entire input is consumed; no whitespace or radix prefix is skipped. '-' is
// a sign; '+' is a sign only when it is not a digit of the selected alphabet.
// Empty/sign-only text is invalid. Invalid input leaves the destination intact.
conversion_error parse_integer_into(integer_ref,std::string_view,radix={},execution={})noexcept;
integer_parse_result parse_integer(std::string_view,radix={},memory_resource& =default_resource())noexcept;
}
