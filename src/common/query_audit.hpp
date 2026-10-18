// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "common/small_checks.h"

// Private checked-build probe. It is deliberately absent from public headers
// and from ordinary object code: disabled sites do not evaluate their tag.
// A diagnostic executable may supply this observer; other checked clients
// leave the weak symbol undefined. The callback must be allocation-free and
// thread-safe, and may only observe (never re-enter a library query).
#if SBN3_CHECK_SMALL
extern "C" void sbn3_query_audit_observer(const char* tag) noexcept __attribute__((weak));
namespace sbn::v3::query_audit {
inline void observe(const char* tag) noexcept {
    if(sbn3_query_audit_observer)sbn3_query_audit_observer(tag);
}
}
#define SBN3_QUERY_AUDIT(tag) ::sbn::v3::query_audit::observe(tag)
#else
#define SBN3_QUERY_AUDIT(tag) ((void)0)
#endif
