#include "sbn3/bitwindow.h"
#include "common/checked.hpp"
#include "verify/bbp_terms.hpp"
#include <fenv.h>

using namespace sbn::v3;
namespace {
constexpr unsigned max_streams = 32;
constexpr sbn3_bbp_stream huvent[] = {
    {3, 12, 1, 1, -1, 6, 2, 1},  {-3, 12, 5, 1, -3, 6, 2, 1}, {-3, 12, 7, 1, -4, 6, 2, 1},
    {3, 12, 11, 1, -6, 6, 2, 1}, {-3, 6, 1, 1, -3, 6, 2, 1},  {-3, 6, 5, 1, -7, 6, 2, 1},
    {-1, 4, 1, 3, -1, 6, 2, 1},  {-1, 4, 3, 3, -4, 6, 2, 1},  {-1, 2, 1, 3, -4, 6, 2, 1}};
uint64_t magnitude(int64_t x) {
    return x < 0 ? uint64_t(0) - uint64_t(x) : uint64_t(x);
}

bool denominator(const sbn3_bbp_stream &s, uint64_t k, bbp::WideModulus &m) {
    const __uint128_t x = __uint128_t(s.a) * k + s.b;
    if (x > UINT64_MAX)
        return false;
    m = {};
    m.d[0] = s.denominator_scale;
    for (unsigned j = 0; j < s.power; ++j)
        if (!m.multiply(uint64_t(x)))
            return false;
    return true;
}

bool query(const sbn3_bbp_stream *streams, size_t count, uint64_t offset, unsigned width,
           sbn3_bbp_window_info &info, uint64_t *ends) {
    if (!streams || !count || count > max_streams || offset > (uint64_t(1) << 62) ||
        (width != 192 && width != 384))
        return false;
    info = {};
    for (size_t j = 0; j < count; ++j) {
        const auto &s = streams[j];
        if (!s.numerator || !s.b || !s.denominator_scale || !s.stride || !s.power || s.power > 8 ||
            s.alternating > 1)
            return false;
        const unsigned amplitude_bits = 64 - __builtin_clzll(magnitude(s.numerator));
        // First omitted E+W+ceil_log2(P+1)<0: its absolute contribution is
        // <1/2 ulp. stride>=1 and a>=0 bound its whole tail by <1 ulp.
        const int64_t limit = int64_t(offset) + s.shift + width + amplitude_bits;
        const uint64_t terms = limit < 0 ? 0 : uint64_t(limit) / s.stride + 1;
        bbp::WideModulus m;
        if (!denominator(s, terms ? terms - 1 : 0, m))
            return false;
        if (__builtin_add_overflow(info.terms, terms, &info.terms))
            return false;
        if (m.bits() > info.max_modulus_bits)
            info.max_modulus_bits = m.bits();
        if (ends)
            ends[j] = info.terms;
    }
    if (__builtin_add_overflow(info.terms, uint64_t(count), &info.error_ulps))
        return false;
    info.max_native_digits = (info.max_modulus_bits + 51) / 52;
    return true;
}

struct alignas(64) Partial {
    bbp::Fixed value;
};
struct Job {
    const sbn3_bbp_stream *streams;
    uint64_t offset, terms;
    unsigned digits;
    uint64_t ends[max_streams]{};
    bool word_stream[max_streams]{};
    Partial partial[32]{};
};
void chunk(void *arg, uint64_t begin, uint64_t end, unsigned rank) {
    auto &job = *static_cast<Job *>(arg);
    require(fegetround() == FE_TONEAREST, SBN3_FATAL_ARGUMENT, "BBP rounding mode");
    unsigned stream = 0;
    while (begin >= job.ends[stream])
        ++stream;
    bbp::RationalTerm terms[64];
    while (begin < end) {
        while (begin >= job.ends[stream])
            ++stream;
        if (job.word_stream[stream]) {
            const auto &s = job.streams[stream];
            const uint64_t stream_begin = stream ? job.ends[stream - 1] : 0;
            const uint64_t k = begin - stream_begin;
            const int64_t e = int64_t(job.offset) + s.shift;
            const uint64_t main_end = e < 0 ? 0 : uint64_t(e) / s.stride + 1;
            uint64_t limit = (end < job.ends[stream] ? end : job.ends[stream]) - stream_begin;
            if (limit > main_end)
                limit = main_end;
            if (k < limit) {
                const bbp::WordStream source{
                    s.a,      s.b,     s.denominator_scale, magnitude(s.numerator), e,
                    s.stride, s.power, bool(s.alternating), s.numerator < 0};
                bbp::native_word_sum(job.partial[rank].value, source, k, limit, job.digits);
                begin = stream_begin + limit;
                continue;
            }
        }
        unsigned filled = 0;
        for (; begin < end && filled < 64; ++begin, ++filled) {
            while (begin >= job.ends[stream])
                ++stream;
            const auto &s = job.streams[stream];
            const uint64_t k = begin - (stream ? job.ends[stream - 1] : 0);
            bbp::WideModulus m;
            require(denominator(s, k, m), SBN3_FATAL_MATH, "BBP denominator exceeds query");
            auto &t = terms[filled];
            for (unsigned j = 0; j < bbp::modulus_words; ++j)
                t.modulus[j] = m.d[j];
            t.numerator = magnitude(s.numerator);
            t.exponent = int64_t(job.offset) + s.shift - int64_t(uint64_t(s.stride) * k);
            t.negative = (s.numerator < 0) != bool(s.alternating && (k & 1));
        }
        bbp::native_rational_sum(job.partial[rank].value, terms, filled, job.digits);
    }
}
void run(void *arg, sbn3_team_scope *scope) {
    auto &j = *static_cast<Job *>(arg);
    sbn3_team_for(scope, 0, j.terms, 16384, SBN3_DYNAMIC, chunk, arg);
}
} // namespace

extern "C" int sbn3_bbp_window_supported(const sbn3_bbp_stream *streams, size_t count, uint64_t offset,
                                         unsigned bits, sbn3_bbp_window_info *out) {
    sbn3_bbp_window_info info;
    if (!query(streams, count, offset, bits, info, nullptr))
        return 0;
    if (out)
        *out = info;
    return 1;
}
extern "C" void sbn3_bbp_window(sbn3_team *team, const sbn3_bbp_stream *streams, size_t count,
                                uint64_t offset, unsigned bits, sbn3_bbp_result *out) {
    Job job{streams, offset, 0, bits / bbp::radix_bits, {}, {}, {}};
    sbn3_bbp_window_info info;
    require(team && out && query(streams, count, offset, bits, info, job.ends), SBN3_FATAL_ARGUMENT,
            "BBP window arguments");
    job.terms = info.terms;
    for (size_t j = 0; j < count; ++j) {
        const auto &s = streams[j];
        const uint64_t terms = job.ends[j] - (j ? job.ends[j - 1] : 0);
        bbp::WideModulus maximum{};
        const bool valid = denominator(s, terms ? terms - 1 : 0, maximum);
        job.word_stream[j] = valid && maximum.bits() <= 104 && magnitude(s.numerator) < (uint64_t(1) << 52) &&
                             !(s.a & 1) && (s.b & 1) && (s.denominator_scale & 1);
    }
    if (job.terms)
        sbn3_team_run(team, run, &job);
    bbp::Fixed sum{};
    for (unsigned w = 0; w < sbn3_team_workers(team); ++w)
        sum.add(job.partial[w].value, job.digits);
    auto low = sum, high = sum;
    low.adjust(info.error_ulps, job.digits, true);
    high.adjust(info.error_ulps, job.digits, false);
    *out = {offset, {sum.extract(bits - 64), sum.extract(bits - 128)}, info.terms, info.error_ulps, bits, 1};
    for (unsigned j = 0; j < 2; ++j)
        out->stable &= low.extract(bits - 64 * (j + 1)) == high.extract(bits - 64 * (j + 1));
}
extern "C" int sbn3_catalan_bbp_supported(uint64_t offset, unsigned bits) {
    return sbn3_bbp_window_supported(huvent, 9, offset, bits, nullptr);
}
extern "C" void sbn3_catalan_bbp(sbn3_team *team, uint64_t offset, unsigned bits, sbn3_bbp_result *out) {
    sbn3_bbp_window(team, huvent, 9, offset, bits, out);
}
