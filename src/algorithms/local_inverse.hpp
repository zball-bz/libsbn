#pragma once
#include "tuning/native_policy.hpp"
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3 {
class Frame;
namespace product {struct LocalWindowProgram;struct LocalWindowReplay;}
inline constexpr size_t local_inverse_max_limbs=native_policy::window_inline_words;
// Small fixed-point quotient shares the exact word/u52 division kernels.
// Selection reuses the ordinary balanced division crossover, without an inverse.
bool local_dyadic_divide_supported(size_t n) noexcept;
bool local_dyadic_divide_native(size_t n) noexcept;
size_t local_dyadic_divide_bytes(size_t n,bool native) noexcept;
void local_dyadic_divide(uint64_t *q,const uint64_t *a,const uint64_t *d,size_t n,
                         bool native,void *scratch,size_t bytes) noexcept;
// Normalized n-limb D, 1<=n<=local_inverse_max_limbs. Exact floor((B^(2n)-1)/D),
// n+1 limbs with the high limb 1. Disjoint inputs/output/scratch. Local
// u52/FFT products follow the ordinary Newton residual/correction proof.
// Length-based local recipes need no operation-level product search or
// spectrum. The byte query includes all temporary FFT tables and work;
// arithmetic performs no ordinary heap allocation.
size_t local_inverse_bytes(size_t n) noexcept;
void local_inverse(uint64_t *out,const uint64_t *d,size_t n,Frame &) noexcept;
// Same framing, but only |U-B^(2n)/D|<3 is required. Skips the full
// multiply-back that makes the all-ones numerator quotient exact.
// This is the contract consumed by block Barrett and Newton scaling.
// A larger W1 refinement may fit the product provider's compact cyclic
// domain. This does not increase the ordinary local seed/exact inverse limit.
bool local_refinement_supported(size_t n,bool quotient) noexcept;
size_t local_inverse_approximate_bytes(size_t n) noexcept;
void local_inverse_approximate(uint64_t *out,const uint64_t *d,size_t n,Frame &) noexcept;
// Public local services may keep the query's product descriptors in their
// caller-owned binding. Compilation is still part of the fresh query.
size_t local_refinement_compile(size_t n,bool quotient,product::LocalWindowProgram &) noexcept;
void local_inverse_replay(uint64_t *out,const uint64_t *d,size_t n,Frame &,
                          product::LocalWindowReplay &) noexcept;
void local_divide_replay(uint64_t *out,const uint64_t *a,const uint64_t *d,size_t n,Frame &,
                         product::LocalWindowReplay &) noexcept;
// Local fused quotient, same fixed-point contract as DivideTerminal:
// normalized n-word D, n+1-word A with A[n]<=1; |Q-B^n*A/D|<3.
// Q may equal A. The terminal consumes an already prepared half reciprocal
// (m=n/2+1, U[m]=1, <3 ulps); the complete call prepares it in this frame.
// Larger admitted compact FFT groups share this interface. Their retained
// form keeps U across the uncached residual, with its full scratch quoted.
inline constexpr size_t local_divide_max_limbs=2*local_inverse_max_limbs-2;
enum class LocalTerminal { Ordinary, Compact, CompactRetained };
size_t local_divide_terminal_bytes(size_t n,LocalTerminal=LocalTerminal::Ordinary) noexcept;
void local_divide_terminal(uint64_t *q,const uint64_t *a,const uint64_t *d,
                           const uint64_t *u,size_t n,Frame &,LocalTerminal=LocalTerminal::Ordinary) noexcept;
size_t local_divide_bytes(size_t n) noexcept;
void local_divide(uint64_t *q,const uint64_t *a,const uint64_t *d,size_t n,Frame &) noexcept;
}
