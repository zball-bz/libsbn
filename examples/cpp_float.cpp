// Original libsbn example; LGPL-3.0-or-later.
#include <sbn3/float.hpp>
#include <array>
#include <cstdint>
#include <cstdio>

// Display the stored dyadic exactly; no decimal float formatter is needed.
static void print_point(const char* label, sbn::float_view value) {
    std::printf("%s = %s0x", label, value.negative() ? "-" : "");
    const auto limbs = value.magnitude().limbs();
    if (limbs.empty()) std::printf("0");
    for (std::size_t i = limbs.size(); i-- > 0;)
        std::printf("%016llx", static_cast<unsigned long long>(limbs[i]));
    std::printf(" * 2^(%lld)\n", static_cast<long long>(value.exponent2()));
}

int main() {
    // Eager calls own their storage, preserve unary precision and discard
    // local reports. The result is a dyadic approximation, not an error ball.
    const auto two = sbn::bigfloat::from(2, sbn::bits{256});
    const auto one = sbn::bigfloat::from(1, sbn::bits{256});
    const auto eager = sbn::sqrt(two) + one;
    print_point("eager sqrt(2) + 1", eager.view());

    // Controlled evaluation: output limbs belong to the caller. Output
    // precision is independent of the input precision and limb capacity.
    constexpr std::size_t precision = 513;
    std::array<std::uint64_t, (precision + 63) / 64> limbs{};
    sbn::float_buffer result(limbs, sbn::bits{precision});
    const auto need = sbn::sqrt_requirements(result.ref(), two.view());
    if (!need.supported()) {
        std::fprintf(stderr, "sqrt unsupported: %s\n", need.reason);
        return 1;
    }
    sbn::workspace work;
    if (const auto error = work.try_reserve(need.scratch_bytes, need.scratch_alignment); error) {
        std::fprintf(stderr, "workspace reservation failed: %s\n", error.where);
        return 2;
    }
    const auto report = sbn::sqrt_into(result.ref(), two.view(), work.on());
    if (!result.view().finite()) return 3;
    print_point("controlled sqrt(2)", result.view());
    switch (report.error.kind) {
    case sbn::error_kind::exact:
        std::puts("local absolute error: 0");
        break;
    case sbn::error_kind::finite:
        std::printf("local absolute error <= 2^(%lld)\n",
                    static_cast<long long>(report.error.exponent2));
        break;
    case sbn::error_kind::unbounded:
        std::puts("no finite absolute-error bound");
        return 4;
    }
    std::printf("numerical flags: %u\n", static_cast<unsigned>(report.flags));
    return 0;
}
