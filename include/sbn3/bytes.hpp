// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/integer.hpp"
#include <system_error>
namespace sbn {
enum class byte_order {big,little};
enum class integer_encoding {twos_complement,magnitude};
struct byte_format {byte_order order=byte_order::big;integer_encoding encoding=integer_encoding::twos_complement;};
struct byte_result {std::byte*end=nullptr;std::errc error{};explicit operator bool()const noexcept{return error==std::errc{};}};
// Canonical minimum-length encoding, with zero encoded as one zero byte.
// The default round-trips signed values; magnitude explicitly ignores sign.
std::size_t export_bytes_size(integer_view,integer_encoding=integer_encoding::twos_complement)noexcept;
// No allocation. On insufficient capacity, output is unchanged. Output must
// be disjoint from the magnitude. No alignment or native-endian assumption.
byte_result export_bytes(std::span<std::byte>,integer_view,byte_format={})noexcept;
// All byte strings are valid, including redundant sign/zero prefixes. Empty
// input denotes zero. Capacity ceil(input.size()/8) always suffices. Destination
// and bytes must be disjoint; no scratch or allocation in the into entry.
void import_bytes_into(integer_ref,std::span<const std::byte>,byte_format={})noexcept;
integer from_bytes(std::span<const std::byte>,byte_format={},memory_resource& resource=default_resource())noexcept;
}
