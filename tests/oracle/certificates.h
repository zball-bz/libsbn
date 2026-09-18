#ifndef SBN3_TEST_CERTIFICATES_H
#define SBN3_TEST_CERTIFICATES_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int ref_mac2_equal(const uint64_t *, size_t, const uint64_t *, size_t, const uint64_t *, size_t,
                   const uint64_t *, size_t, const uint64_t *, size_t);
#ifdef __cplusplus
}
#endif
#endif
