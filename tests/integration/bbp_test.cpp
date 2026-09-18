#include "product_support.hpp"
#include "sbn3/constants.h"
#include "sbn3/verify.h"
#include "verify/bbp.hpp"
using namespace sbn::v3;
static void kernels() {
    for (unsigned n : {4u, 8u}) {
        for (uint64_t offset : {uint64_t(0), uint64_t(128), uint64_t(4096), uint64_t(16609640384),
                                (uint64_t(1) << 40) - 1, (uint64_t(1) << 48) - 3}) {
            for (auto t : bbp::formula) {
                const uint64_t count = bbp::main_terms(offset, t);
                for (unsigned trial = 0; trial < 12 && count; ++trial) {
                    const uint64_t start = trial == 0   ? 0
                                           : trial == 1 ? (count > 97 ? count - 97 : 0)
                                                        : random_word() % count;
                    const uint64_t end = std::min(count, start + uint64_t(97));
                    bbp::Fixed got{}, want{};
                    bbp::native_sum(got, offset, t, start, end, n);
                    for (uint64_t k = start; k < end; ++k)
                        want.add(bbp::scalar_term(offset, t, k, n), n, t.negative != bool(k & 1));
                    if (memcmp(&want, &got, sizeof got))
                        fprintf(stderr, "BBP kernel mismatch W%u offset=%llu a=%u b=%u start=%llu\n",
                                n * bbp::radix_bits, (unsigned long long)offset, t.a, t.b,
                                (unsigned long long)start);
                    assert(!memcmp(&want, &got, sizeof got));
                }
            }
        }
    }
}
static void known(Fixture &f, uint64_t offset, uint64_t hi, uint64_t lo) {
    for (unsigned window : {192u, 384u}) {
        sbn3_bbp_result result{};
        allocation_watch_start();
        sbn3_pi_bbp(f.team, offset, window, &result);
        assert(!allocation_watch_stop());
        assert(result.stable && result.bits[0] == hi && result.bits[1] == lo);
    }
}
static void tail(Fixture &f, size_t n) {
    sbn3_pi_options options{{sbn3_team_workers(f.team), 8, 0, 0, 0}, 0, 128, 0, 0};
    sbn3_pi_plan plan{};
    sbn3_pi_info info{};
    assert(sbn3_pi_query(n, &options, &plan, &info) == SBN3_SUPPORTED);
    const size_t offset = up(f.base + f.cursor, info.storage_alignment) - f.base;
    f.cursor = up(offset + info.storage_bytes, 4096) + 4096;
    sbn3_error error{};
    assert(sbn3_arena_prepare(f.arena, offset, info.storage_bytes, &error) == SBN3_OK);
    sbn3_pi_binding *binding = nullptr;
    sbn3_pi_bind(&plan, f.arena, offset, f.team, &binding);
    auto value = sbn3_pi_execute_inplace(binding);
    uint64_t sample[3] = {value.data[0], value.data[1], value.data[n]};
    sbn3_bbp_result result{};
    for (unsigned bits : {192u, 384u}) {
        allocation_watch_start();
        const int match = sbn3_pi_bbp_check_tail(f.team, value, bits, &result);
        assert(!allocation_watch_stop());
        assert(match);
        auto *mutable_value = const_cast<uint64_t *>(value.data);
        for (unsigned bit : {0u, 63u, 64u, 127u}) {
            mutable_value[bit / 64] ^= uint64_t(1) << (bit % 64);
            assert(!sbn3_pi_bbp_check_tail(f.team, value, bits, &result));
            mutable_value[bit / 64] ^= uint64_t(1) << (bit % 64);
        }
    }
    assert(value.data[0] == sample[0] && value.data[1] == sample[1] && value.data[n] == sample[2]);
    sbn3_pi_unbind(binding);
}
int main() {
    assert(sbn3_pi_bbp_supported(0, 192));
    assert(!sbn3_pi_bbp_supported(uint64_t(1) << 48, 192));
    assert(!sbn3_pi_bbp_supported(0, 128));
    assert(!sbn3_pi_bbp_supported(0, 224));
    assert(!sbn3_pi_bbp_supported(0, 448));
    kernels();
    sbn3_bbp_result parallel_reference{};
    {
        Fixture f(1, false);
        sbn3_pi_bbp(f.team, 4000003, 384, &parallel_reference);
        assert(parallel_reference.stable);
    }
    for (unsigned workers : {1u, 3u, 16u}) {
        Fixture f(workers, false);
        known(f, 0, 0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL);
        known(f, 1, 0x487ed5110b4611a6ULL, 0x2633145c06e0e689ULL);
        known(f, 64, 0x13198a2e03707344ULL, 0xa4093822299f31d0ULL);
        // More than 16 full dynamic chunks: actually exercise concurrent
        // rank-local accumulators, not just a multi-worker idle Team.
        known(f, 4000003, parallel_reference.bits[0], parallel_reference.bits[1]);
        for (size_t n : {2u, 7u, 64u})
            tail(f, n);
    }
    puts("BBP exact SIMD/scalar, 192/384-bit, offsets up to 2^48, thread/rebind, tail corruption, no "
         "allocation PASS");
}
