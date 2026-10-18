#ifndef SBN3_JACOBI_H
#define SBN3_JACOBI_H
#include "sbn3/mul.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Jacobi (a/n), n positive and odd, result in {-1,0,1}. (a/1)=1,
 * including (0/1). Variable time; not a cryptographic constant-time API. */
int sbn3_jacobi_u64(uint64_t a,uint64_t n);

typedef struct sbn3_jacobi_options {unsigned workers;} sbn3_jacobi_options;
typedef struct sbn3_jacobi_info {
    size_t scratch_bytes,scratch_alignment;
    size_t value_bytes,temporary_bytes,arithmetic_bytes;
    size_t published_constants_bytes;
    unsigned workers;
} sbn3_jacobi_info;
/* Pure queries, no plan construction or allocation. Each declared length
 * <=2^20 limbs, nn>0; workers 0 means 1. Unsupported => SIZE_MAX.
 * Scratch equals ordinary GCD's bound; <=2-limb shorter input needs none. */
size_t sbn3_jacobi_scratch_bytes(size_t an,size_t nn,unsigned workers);
sbn3_query_result sbn3_jacobi_query(size_t an,size_t nn,
                                  const sbn3_jacobi_options *,sbn3_jacobi_info *);
/* Unsigned magnitudes, little endian; leading zeros and empty numerator
 * allowed. Input spans may overlap each other, but neither may overlap the
 * queried scratch span. Values are preserved. Scratch alignment is queried.
 * No allocation, page operations, implicit context, or retained private
 * state. An existing idle team is optional; no worker is created here. */
int sbn3_jacobi(const uint64_t *a,size_t an,const uint64_t *n,size_t nn,
                void *scratch,size_t scratch_bytes,sbn3_team *team);
/* Signed numerator; denominator must still be positive odd. Negative zero
 * is accepted. All storage/alias rules match sbn3_jacobi. */
int sbn3_int_jacobi(sbn3_int_view a,sbn3_int_view n,
                    void *scratch,size_t scratch_bytes,sbn3_team *team);
#ifdef __cplusplus
}
#endif
#endif
