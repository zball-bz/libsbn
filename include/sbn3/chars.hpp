// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/radix.hpp"
#include <charconv>
namespace sbn {
// Character adapters consume/emit the numeric prefix, without a NUL or skipped
// whitespace. For bases 2..36 the default alphabet parses both letter cases;
// custom alphabets and bases 37..64 are case-sensitive. '+' is a digit only
// where the alphabet says so, never an optional sign in from_chars.
requirements to_chars_requirements(std::span<char>,integer_view,radix={},unsigned workers=1)noexcept;
std::to_chars_result to_chars(char*first,char*last,integer_view,radix,execution)noexcept;
std::to_chars_result to_chars(char*first,char*last,const integer&,radix={})noexcept;
requirements from_chars_requirements(integer_ref,std::string_view,radix={},unsigned workers=1)noexcept;
std::from_chars_result from_chars(const char*first,const char*last,integer_ref,radix,execution)noexcept;
std::from_chars_result from_chars(const char*first,const char*last,integer&,radix={})noexcept;
}
