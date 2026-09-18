#ifndef SBN3_BASE_H
#define SBN3_BASE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Development ABI: only implemented capabilities are declared here. */
typedef enum sbn3_status {
    SBN3_OK = 0,
    SBN3_ENOMEM = 1,
    SBN3_ELOCK = 2,
    SBN3_EOS = 3,
    SBN3_EBUSY = 4,
    SBN3_ECAPACITY = 5
} sbn3_status;

typedef enum sbn3_fatal_kind {
    SBN3_FATAL_ARGUMENT = 1,
    SBN3_FATAL_SIZE = 2,
    SBN3_FATAL_WORKSPACE = 3,
    SBN3_FATAL_LIFETIME = 4,
    SBN3_FATAL_TEAM = 5,
    SBN3_FATAL_MATH = 6
} sbn3_fatal_kind;

typedef struct sbn3_fatal_info {
    sbn3_fatal_kind kind;
    const char *where;
    uint64_t need, have;
} sbn3_fatal_info;
typedef void (*sbn3_fatal_hook)(const sbn3_fatal_info *, void *);

/* Configure before creating workers; the hook must not return by longjmp. */
void sbn3_set_fatal_hook(sbn3_fatal_hook hook, void *argument);
const char *sbn3_build_version(void);

/* Exact alias with either input is allowed; partial overlap is not.
 * n==0 permits NULL pointers and returns zero. */
uint64_t sbn3_add_n(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n);
uint64_t sbn3_sub_n(uint64_t *r, const uint64_t *a, const uint64_t *b, size_t n);

/* Writes exactly an+bn limbs, including high zero limbs. Output must be
 * disjoint from both inputs. Either zero length produces an+bn zero limbs.
 * Explicit basecase entry; no implicit allocation or algorithm selection.
 * Normal builds trust length/capacity/alias preconditions; --checked and
 * sanitizer builds diagnose violations. */
void sbn3_mul_basecase(uint64_t *r, size_t capacity,
                       const uint64_t *a, size_t an,
                       const uint64_t *b, size_t bn);

typedef struct sbn3_int_view {const uint64_t *data;size_t size;unsigned negative;} sbn3_int_view;
typedef struct sbn3_int {uint64_t *data;size_t capacity,size;unsigned negative;} sbn3_int;
/* Caller-owned magnitude storage; normalizes the result including negative zero. */
void sbn3_int_mul_basecase(sbn3_int *,sbn3_int_view,sbn3_int_view);

#ifdef __cplusplus
}
#endif
#endif
