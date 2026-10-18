#pragma once
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define SBN_CPP_POISON(p,n) __asan_poison_memory_region((p),(n))
#define SBN_CPP_UNPOISON(p,n) __asan_unpoison_memory_region((p),(n))
#endif
#endif
#ifndef SBN_CPP_POISON
#define SBN_CPP_POISON(p,n) ((void)(p),(void)(n))
#define SBN_CPP_UNPOISON(p,n) ((void)(p),(void)(n))
#endif
