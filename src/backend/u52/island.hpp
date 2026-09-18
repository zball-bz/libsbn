#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include "runtime/scratch.hpp"
#include "backend/u52/kernels.hpp"
#include "backend/u52/lane.hpp"
namespace sbn::v3::u52 {using scratch=::sbn::v3::Frame;}
#define SBN3_U52_CAT0(a,b) a##b
#define SBN3_U52_CAT(a,b) SBN3_U52_CAT0(a,b)
#define SCRATCH(s) ::sbn::v3::AssumedFrameMark SBN3_U52_CAT(mark_,__LINE__)(*(s))
#define SALLOC(s,T,n) (s)->alloc_assumed<T>(n)
#include "backend/u52/mparam.hpp"
#include "backend/u52/canon.hpp"
#include "backend/u52/interp.hpp"
#include "backend/u52/mul.hpp"
#include "backend/u52/cvt.hpp"
#undef SCRATCH
#undef SALLOC
