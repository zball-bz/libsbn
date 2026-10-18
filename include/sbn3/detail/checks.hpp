// Original libsbn code; LGPL-3.0-or-later.
#pragma once
namespace sbn::detail {
#if defined(SBN3_CHECK_SMALL) && SBN3_CHECK_SMALL
inline constexpr bool checked=true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
inline constexpr bool checked=true;
#else
inline constexpr bool checked=false;
#endif
#else
inline constexpr bool checked=false;
#endif
}
