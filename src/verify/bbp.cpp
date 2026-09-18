#include "sbn3/verify.h"
#include "common/checked.hpp"
#include "verify/bbp_terms.hpp"
#include <fenv.h>
using namespace sbn::v3;
namespace {
struct alignas(64) Partial {
    bbp::Fixed sum;
};
struct Job {
    uint64_t offset;
    unsigned digits;
    Partial partial[32]{};
};
void chunk(void *arg, uint64_t begin, uint64_t end, unsigned rank) {
    auto &j = *static_cast<Job *>(arg);
    require(fegetround() == FE_TONEAREST, SBN3_FATAL_ARGUMENT, "BBP rounding mode");
    for (const auto t : bbp::formula) {
        const uint64_t count = bbp::main_terms(j.offset, t);
        if (begin < count) {
            const bbp::WordStream source{t.a, t.b, 1,    1,         int64_t(j.offset) + t.shift,
                                         10,  1,   true, t.negative};
            bbp::native_word_sum(j.partial[rank].sum, source, begin, end < count ? end : count, j.digits);
        }
    }
}
void run(void *arg, sbn3_team_scope *scope) {
    const auto &j = *static_cast<Job *>(arg);
    sbn3_team_for(scope, 0, bbp::main_terms(j.offset, bbp::formula[2]), 16384, SBN3_DYNAMIC, chunk, arg);
}
} // namespace
extern "C" int sbn3_pi_bbp_supported(uint64_t offset, unsigned bits) {
    return offset <= (uint64_t(1) << 48) - 3 && (bits == 192 || bits == 384);
}
extern "C" void sbn3_pi_bbp(sbn3_team *team, uint64_t offset, unsigned bits, sbn3_bbp_result *out) {
    require(team && out && sbn3_pi_bbp_supported(offset, bits), SBN3_FATAL_ARGUMENT, "BBP arguments");
    Job job{offset, bits / bbp::radix_bits, {}};
    sbn3_team_run(team, run, &job);
    bbp::Fixed sum{};
    for (unsigned w = 0; w < sbn3_team_workers(team); ++w)
        sum.add(job.partial[w].sum, job.digits);
    uint64_t terms = 0;
    for (const auto t : bbp::formula) {
        uint64_t k = bbp::main_terms(offset, t);
        terms += k;
        for (; int64_t(offset) + t.shift - int64_t(10 * k) + bits >= 0; ++k) {
            sum.add(bbp::scalar_term(offset, t, k, job.digits), job.digits, t.negative != bool(k & 1));
            ++terms;
        }
    }
    // Each included term incurs <1 ulp. Each of the seven omitted geometric
    // tails is <1 ulp (ratio <= 1/1024). No floating-point summation error.
    const uint64_t radius = terms + 7;
    auto low = sum, high = sum;
    low.adjust(radius, job.digits, true);
    high.adjust(radius, job.digits, false);
    *out = {offset, {sum.extract(bits - 64), sum.extract(bits - 128)}, terms, radius, bits, 1};
    for (unsigned i = 0; i < 2; ++i)
        out->stable &= low.extract(bits - 64 * (i + 1)) == high.extract(bits - 64 * (i + 1));
}
extern "C" int sbn3_pi_bbp_check_tail(sbn3_team *team, sbn3_const_limbs value, unsigned bits,
                                      sbn3_bbp_result *out) {
    require(value.data && value.count >= 3 && value.count - 1 <= (uint64_t(1) << 48) / 64,
            SBN3_FATAL_ARGUMENT, "BBP value shape");
    sbn3_pi_bbp(team, 64 * (value.count - 1) - 128, bits, out);
    return out->stable && value.data[value.count - 1] == 3 && value.data[1] == out->bits[0] &&
           value.data[0] == out->bits[1];
}
