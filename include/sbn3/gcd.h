#ifndef SBN3_GCD_H
#define SBN3_GCD_H
#include <stddef.h>
#include <stdint.h>
#include "sbn3/mul.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { SBN3_GCD_SMALL_LIMBS = 1024 };
/* Ordinary variable-time GCD. gcd(0,0)=0. No context or allocation. */
uint64_t sbn3_gcd_u64(uint64_t a,uint64_t b);

/* Pure capacity query for the small/middle implementation. Returns SIZE_MAX
 * if either declared input exceeds SBN3_GCD_SMALL_LIMBS; zero if either
 * declared operand has <=2 limbs. No plan, arena or table is constructed. */
size_t sbn3_gcd_small_scratch_bytes(size_t an,size_t bn);

/* Positive little-endian magnitudes; leading zero words and empty inputs
 * are allowed. Inputs are preserved. For two nonzero values, min(an,bn)
 * output words suffice. If one is zero, output must fit the other;
 * max(an,bn) is the unconditional bound. Returns the normalized GCD length
 * and writes only those words. Null output is allowed for a zero GCD.
 * scratch has the queried bytes and 8-byte alignment. Output and scratch
 * are mutually disjoint and disjoint from both inputs. The two read-only
 * inputs may overlap. Null input/scratch spans need zero lengths/bytes.
 * No allocation, page operation or retained state. Not a constant-time API. */
size_t sbn3_gcd_small(uint64_t *out,const uint64_t *a,size_t an,
                     const uint64_t *b,size_t bn,void *scratch);

/* Independent pairs, out[i]=gcd(a[i],b[i]). SIMD throughput entry, not a
 * different single-pair algorithm. n=0 permits null pointers. out may equal
 * a or b exactly; other partial overlap with output is not supported.
 * No scratch, context, allocation, or hidden prepared state. */
void sbn3_gcd_u64_batch(uint64_t *out,const uint64_t *a,const uint64_t *b,size_t n);

/* General in-RAM GCD. Values, output and scratch are caller-owned. The
 * preserving entry dispatches directly to gcd_small in its small domain.
 * workers=0 means one. A multiworker call uses an existing idle caller team;
 * a one-worker call permits NULL team. A zero GCD permits NULL output.
 * No worker creation or allocation. */
typedef struct sbn3_gcd_options { unsigned workers; } sbn3_gcd_options;
enum { SBN3_EXGCD_X=1, SBN3_EXGCD_Y=2 };
typedef struct sbn3_gcd_info {
    size_t scratch_bytes,scratch_alignment;
    size_t value_bytes,matrix_bytes,temporary_bytes,arithmetic_bytes;
    size_t gcd_capacity,x_capacity,y_capacity;
    size_t published_constants_bytes; /* process-wide, not private scratch */
    unsigned workers,cofactor_mask;
} sbn3_gcd_info;
/* Current general domain: each declared input <=2^20 limbs. SIZE_MAX from
 * these thin capacity queries means unsupported. workers=0 means one.
 * Full *_query additionally reports the memory categories and output bounds. */
size_t sbn3_gcd_scratch_bytes(size_t an,size_t bn,unsigned workers);
size_t sbn3_exgcd_scratch_bytes(size_t an,size_t bn,unsigned cofactor_mask,unsigned workers);
sbn3_query_result sbn3_gcd_query(size_t an,size_t bn,const sbn3_gcd_options *,sbn3_gcd_info *);
sbn3_query_result sbn3_exgcd_query(size_t an,size_t bn,unsigned cofactor_mask,
                                const sbn3_gcd_options *,sbn3_gcd_info *);
size_t sbn3_gcd(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,
                void *scratch,size_t scratch_bytes,sbn3_team *team);
/* g>=0 and x*a+y*b=g. x and y may independently be NULL; query with the
 * corresponding output mask. Magnitudes and signs are normalized. g is
 * required. No minimal/unique coefficient convention is promised.
 * Both zero inputs produce g=x=y=0. A single nonzero input uses its sign
 * as its coefficient and zero for the other. Capacity is given by query.
 * All writable spans are mutually disjoint and disjoint from inputs; the
 * two read-only inputs may overlap. Leading zeros/negative zero are allowed.
 * Inputs are preserved. This is variable-time, not a cryptographic API. */
void sbn3_int_exgcd(sbn3_int *g,sbn3_int *x,sbn3_int *y,sbn3_int_view a,sbn3_int_view b,
                    void *scratch,size_t scratch_bytes,sbn3_team *team);

#ifdef __cplusplus
}
#endif
#endif
