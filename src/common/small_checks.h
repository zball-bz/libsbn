#ifndef SBN3_SMALL_CHECKS_H
#define SBN3_SMALL_CHECKS_H
/* Small execution trusts the bound-plan/caller contract in normal builds.
 * Query support decisions, explicit resource preparation and numerical
 * certificates remain unconditional. Compact borrowed bindings additionally
 * audit their caller-granted range and lease lifetime in checked builds.
 * Enable explicitly for diagnostics; sanitizers enable it by default. */
#ifndef SBN3_CHECK_SMALL
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define SBN3_CHECK_SMALL 1
#endif
#endif
#ifndef SBN3_CHECK_SMALL
#define SBN3_CHECK_SMALL 0
#endif
#endif
#endif
