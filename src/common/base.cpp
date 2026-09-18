#include "common/checked.hpp"
#include <stdlib.h>
#include <unistd.h>

namespace {
sbn3_fatal_hook hook;
void *hook_argument;
void print(const char *s) noexcept {
    if (!s) return;
    size_t n = 0;
    while (s[n]) ++n;
    while (n) {
        const ssize_t k = write(STDERR_FILENO, s, n);
        if (k <= 0) return;
        s += k; n -= static_cast<size_t>(k);
    }
}
void number(uint64_t x) noexcept {
    char b[24]; size_t n = sizeof b;
    do { b[--n] = static_cast<char>('0' + x % 10); x /= 10; } while (x);
    (void)write(STDERR_FILENO, b + n, sizeof b - n);
}
}
namespace sbn::v3 {
[[noreturn]] void fatal(sbn3_fatal_kind kind, const char *where,
                        uint64_t need, uint64_t have) noexcept {
    const sbn3_fatal_info info{kind, where, need, have};
    print("libsbn_v3 fatal: "); print(where); print(" kind="); number(kind);
    print(" need="); number(need); print(" have="); number(have); print("\n");
    if (hook) hook(&info, hook_argument);
    abort();
}
}
extern "C" void sbn3_set_fatal_hook(sbn3_fatal_hook fn, void *arg) {
    hook = fn; hook_argument = arg;
}
extern "C" const char *sbn3_build_version(void) {
    return "libsbn_v3 development / clang " __clang_version__;
}
