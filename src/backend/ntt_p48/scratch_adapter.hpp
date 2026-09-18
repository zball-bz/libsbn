#pragma once
#ifndef SBN3_P48_NS
#define SBN3_P48_NS p48_np6
#endif
#include "runtime/scratch.hpp"
namespace sbn::v3::SBN3_P48_NS {
using scratch=::sbn::v3::Frame;
using FrameMark=::sbn::v3::FrameMark;
}
/* Private migration aliases; never included from the public headers. */
#define SALLOC(s,T,n) ((s)->alloc<T>(n))
#define SALLOC0(s,T,n) ((s)->alloc_zero<T>(n))
