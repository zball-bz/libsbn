// Parse service gate: digit strings -> floor(value * 2^bits) against exact integer arithmetic, all base
// classes, exactly representable inputs (the multiply-back path), invalid bytes, and the round trip
// through the format service.
#include "product_support.hpp"
#include "sbn3/radix.h"
#include "radix/parse_tree.hpp"
#include <initializer_list>
#include <string>
#include <math.h>
namespace {
struct Pool {
    Fixture &f;
    size_t offset = 0, bytes = 0;
    Pool(Fixture &fixture, size_t size) : f(fixture), bytes(size) {
        auto block = f.allocate(size, size_t(1) << 21);
        offset = size_t(reinterpret_cast<uintptr_t>(block.data) - reinterpret_cast<uintptr_t>(f.arena->base));
        sbn3_arena_release(f.arena, &f.leases.back());
        f.leases.pop_back();
    }
};
struct Big {
    ref_int v;
    Big() { ref_init(v); }
    ~Big() { ref_clear(v); }
    Big(const Big &) = delete;
};
void power(ref_number *out, unsigned base, uint64_t e) {
    Big acc;
    ref_set_ui(out, 1);
    ref_set_ui(acc.v, base);
    for (; e; e >>= 1) {
        if (e & 1)
            ref_mul(out, out, acc.v);
        if (e > 1)
            ref_mul(acc.v, acc.v, acc.v);
    }
}
// floor(N 2^bits / base^fraction) for the digit values `d`, as limbs
std::vector<uint64_t> reference(const std::vector<uint8_t> &d, unsigned base, uint64_t fraction, uint64_t bits, size_t limbs) {
    Big n, scale, chunk;
    // Horner in chunks of base^8 keeps the reference quadratic but quick
    ref_set_ui(n.v, 0);
    size_t at = 0;
    const size_t head = d.size() % 8;
    auto take = [&](size_t count) {
        uint64_t w = 0, scale_word = 1;
        for (size_t j = 0; j < count; ++j) {
            w = w * base + d[at++];
            scale_word *= base;
        }
        ref_mul_ui(n.v, n.v, scale_word);
        ref_add_ui(n.v, n.v, w);
    };
    if (head)
        take(head);
    while (at < d.size())
        take(8);
    ref_mul_2exp(n.v, n.v, bits);
    power(scale.v, base, fraction);
    ref_fdiv_q(n.v, n.v, scale.v);
    std::vector<uint64_t> out(limbs, 0);
    size_t count = 0;
    if (ref_sgn(n.v))
        ref_export(out.data(), &count, -1, 8, 0, 0, n.v);
    assert(count <= limbs);
    return out;
}
struct Stats {
    uint64_t cases = 0, fallbacks = 0, invalid = 0, round_trips = 0;
} stats;
bool canonical_decimal_only=false,once_mode=false;
void check(Pool &pool, unsigned workers, unsigned base, const std::vector<uint8_t> &values, uint64_t integer,
           uint64_t bits, bool alphabet, int expect_fallback = -1) {
    Fixture &f = pool.f;
    const bool use_once=once_mode&&expect_fallback<0; // explicit recipe witnesses retain the planned interface
    const uint64_t fraction = values.size() - integer;
    sbn3_parse_spec spec{base, integer, fraction, bits};
    sbn3_radix_options options{};
    options.workers = workers;
    options.repeated=unsigned(random_word()&1);
    unsigned char table[64];
    sbn3_radix_alphabet(base, table);
    if(alphabet&&!canonical_decimal_only&&(random_word()&1))for(unsigned j=0;j<64;++j)table[j]=uint8_t(73*j+151);
    if (alphabet) {
        options.use_alphabet = 1;
        memcpy(options.alphabet, table, 64);
    }
    std::vector<unsigned char> text(values.size());
    for (size_t j = 0; j < values.size(); ++j)
        text[j] = alphabet ? table[values[j]] : values[j];
    sbn3_parse_plan plan{};
    sbn3_parse_info info{};
    allocation_watch_start();
    const auto rc = use_once?sbn3_parse_once_query(&spec,&options,&info):sbn3_parse_query(&spec, &options, &plan, &info);
    assert(allocation_watch_stop() == 0);
    assert(rc == SBN3_SUPPORTED && info.storage_bytes <= pool.bytes && info.exponent2 == -int64_t(bits));
    sbn3_parse_binding *binding = nullptr;
    allocation_watch_start();
    if(!use_once)sbn3_parse_bind(&plan, f.arena, pool.offset, f.team, &binding);
    assert(allocation_watch_stop() == 0);
    auto different=values;
    for(size_t j=0;j<different.size();++j)different[j]=uint8_t((different[j]+1+j)%base);
    const size_t padded = up(info.limbs * 8 + 64, 64);
    auto *out = static_cast<uint64_t *>(aligned_alloc(64, padded));
    auto execute=[&](sbn3_parse_result *result){
        if(use_once){const sbn3_radix_workspace w{f.arena,f.team,pool.offset,pool.bytes};
            sbn3_parse_once_result r;sbn3_parse_once(&spec,&options,text.data(),out,info.storage_bytes?&w:nullptr,&r);
            assert(r.limbs==(r.valid?info.limbs:0));*result={r.valid,r.invalid_index,r.exact_fallback};}
        else sbn3_parse_execute(binding,text.data(),{out,info.limbs},result);
    };
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        const auto &current=repeat==1?different:values;
        for(size_t j=0;j<text.size();++j)text[j]=alphabet?table[current[j]]:current[j];
        const auto expect=reference(current,base,fraction,bits,info.limbs);
        memset(out, 0xee, padded);
        sbn3_parse_result result{};
        allocation_watch_start();
        execute(&result);
        assert(allocation_watch_stop() == 0);
        assert(result.valid == 1);
        for (size_t j = 0; j < info.limbs; ++j)
            if (out[j] != expect[j]) {
                fprintf(stderr, "parse mismatch: base %u digits %zu integer %llu bits %llu limb %zu of %zu fallback %u\n",
                        base, values.size(), (unsigned long long)integer, (unsigned long long)bits, j, info.limbs,
                        result.exact_fallback);
                assert(false);
            }
        assert(out[info.limbs] == 0xeeeeeeeeeeeeeeeeULL);
        if (expect_fallback >= 0&&repeat!=1)
            assert(int(result.exact_fallback) == expect_fallback);
        stats.fallbacks += result.exact_fallback;
        ++stats.cases;
    }
    // an invalid byte is reported with its position
    if (!values.empty()) {
        const size_t at = random_word() % values.size();
        const unsigned char saved = text[at];
        unsigned invalid=0;
        if(alphabet){for(;;++invalid){bool used=false;for(unsigned j=0;j<base;++j)used|=table[j]==invalid;if(!used)break;}}
        text[at] = alphabet ? uint8_t(invalid) : uint8_t(base);
        sbn3_parse_result result{};
        execute(&result);
        assert(result.valid == 0 && result.invalid_index == at);
        text[at] = saved;
        // The public contract leaves output unspecified on invalid input.
        // Restore it before the integer round trip; also exercise reuse
        // after a rejected digit sequence.
        allocation_watch_start();
        execute(&result);
        assert(allocation_watch_stop()==0&&result.valid==1);
        ++stats.invalid;
    }
    allocation_watch_start();
    if(!use_once)sbn3_parse_unbind(binding);
    assert(allocation_watch_stop() == 0);
    // Round trip: formatting the parsed number (exactly) must give back digits that parse to it again;
    // for integers it must give back the digit string itself.
    if (!fraction && !bits) {
        sbn3_format_spec fs{base, info.limbs, 0, 0, SBN3_RADIX_EXACT};
        sbn3_format_plan fplan{};
        sbn3_format_info finfo{};
        assert(sbn3_format_query(&fs, &options, &fplan, &finfo) == SBN3_SUPPORTED && finfo.storage_bytes <= pool.bytes);
        sbn3_format_binding *fb = nullptr;
        sbn3_format_bind(&fplan, f.arena, pool.offset, f.team, &fb);
        std::vector<unsigned char> digits(finfo.digit_bytes);
        sbn3_format_result fr{};
        sbn3_format_execute(fb, {out, info.limbs}, digits.data(), &fr);
        sbn3_format_unbind(fb);
        size_t lead = 0;
        while (lead < values.size() && values[lead] == 0)
            ++lead;
        assert(finfo.integer_digits - fr.integer_first == values.size() - lead);
        assert(!memcmp(digits.data() + fr.integer_first, text.data() + lead, values.size() - lead));
        ++stats.round_trips;
    }
    free(out);
}
std::vector<uint8_t> random_digits(unsigned base, size_t n) {
    std::vector<uint8_t> d(n);
    for (auto &v : d)
        v = uint8_t(random_word() % base);
    return d;
}
} // namespace
void large_storage_gate(){
    Fixture f(16,false);Pool pool(f,size_t(192)<<20);
    constexpr size_t count=1000003;
    for(unsigned base:{3u,10u,63u})for(unsigned workers:{1u,16u})for(bool fraction:{false,true}){
        std::vector<uint8_t> input(count,fraction?uint8_t(base-1):uint8_t(0));
        if(!fraction)input.back()=1;
        sbn3_parse_spec spec{base,fraction?0:count,fraction?count:0,fraction?128u:0u};
        sbn3_radix_options options{};options.workers=workers;
        sbn3_parse_plan plan;sbn3_parse_info info{};
        assert(sbn3_parse_query(&spec,&options,&plan,&info)==SBN3_SUPPORTED&&info.storage_bytes<=pool.bytes);
        auto *out=static_cast<uint64_t *>(aligned_alloc(64,up((info.limbs+8)*8,64)));
        sbn3_parse_binding *binding;sbn3_parse_bind(&plan,f.arena,pool.offset,f.team,&binding);
        for(unsigned repeat=0;repeat<2;++repeat){
            if(repeat){std::fill(input.begin(),input.end(),0);input.back()=1;}
            std::fill_n(out,info.limbs+8,UINT64_C(0xcccccccccccccccc));sbn3_parse_result result;
            allocation_watch_start();sbn3_parse_execute(binding,input.data(),{out,info.limbs},&result);assert(!allocation_watch_stop()&&result.valid);
            for(size_t j=0;j<info.limbs;++j)assert(out[j]==(fraction?(!repeat&&j<2?UINT64_MAX:0):j==0?1:0));
            for(size_t j=info.limbs;j<info.limbs+8;++j)assert(out[j]==UINT64_C(0xcccccccccccccccc));
        }
        sbn3_parse_unbind(binding);free(out);
    }
    puts("large parse storage: million-digit integer/fraction, bases3/10/63, W1/W16 on shared team, exact boundaries and changed values PASS");
}
int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--once-small")){once_mode=true;argv[1]=const_cast<char*>("--decimal-small");}
    if(argc==2&&!strcmp(argv[1],"--once")){once_mode=true;argc=1;}
    if(argc==2&&!strcmp(argv[1],"--decimal-small")){
        canonical_decimal_only=true;Fixture f(1,false,size_t(16)<<20);Pool pool(f,size_t(4)<<20);
        for(size_t n=1;n<=40;++n){
            for(size_t count:{size_t(1),size_t(18),size_t(19),size_t(20),size_t(38),size_t(39),size_t(40),19*n,19*n+1,64*n}){
                auto digits=random_digits(10,count);
                check(pool,1,10,digits,count,0,true);
                check(pool,1,10,digits,0,64*n,true);
                std::fill(digits.begin(),digits.end(),uint8_t(9));check(pool,1,10,digits,0,64*n,true);
            }
            auto digits=random_digits(10,19*n);
            check(pool,1,10,digits,0,64*n-1,true);check(pool,1,10,digits,0,64*n+1,true);
            check(pool,1,10,digits,n,64*n,true);
        }
        puts("decimal service parse: dense 1..40 limbs, block/crossover boundaries, invalid/reuse, partial exponents and mixed values PASS");return 0;
    }
    if(argc>1){assert(argc==2&&!strcmp(argv[1],"--storage-only"));large_storage_gate();return 0;}
    {
        Fixture f(16,false);Pool pool(f,size_t(128)<<20);
        for(unsigned workers:{1u,4u,8u})
            check(pool,workers,10,random_digits(10,8193),4097,4099,false);
    }
    for (unsigned workers : {1u, 4u}) {
        Fixture f(workers, false);
        Pool pool(f, size_t(192) << 20);
        for(unsigned base=2;base<=64;++base){
            // Stress the compact recursion's live-storage bound at its
            // output-limb limit, including the handoff to the general tree.
            for(unsigned limbs:{255u,256u,257u}){
                const size_t digits=size_t(64.0*limbs/log2(double(base)));
                check(pool,workers,base,std::vector<uint8_t>(digits,uint8_t(base-1)),digits,0,false);
            }
            for(size_t n:{64u,65u,511u,512u,513u}){
                const auto d=random_digits(base,n);
                check(pool,workers,base,d,n/3,129,true);
                check(pool,workers,base,d,0,64*sbn::v3::radix::parse_leaf_limbs+1,false);
            }
        }
        const unsigned bases[] = {2, 8, 16, 32, 64, 3, 7, 10, 12, 36, 62, 63};
        for (unsigned base : bases) {
            for (size_t n : {size_t(1), size_t(2), size_t(9), size_t(63), size_t(64), size_t(65), size_t(200), size_t(513),
                             size_t(1000), size_t(5000)}) {
                const auto d = random_digits(base, n);
                const bool alphabet = base == 36 || base == 62 || base == 16;
                check(pool, workers, base, d, n, 0, alphabet);                               // integer
                check(pool, workers, base, d, n, 1 + random_word() % 200, alphabet);         // integer, scaled
                check(pool, workers, base, d, 0, uint64_t(n * 6.1) + 70, alphabet);          // fraction, more bits than digits
                check(pool, workers, base, d, 0, uint64_t(n * 0.9) + 1, alphabet);           // fraction, fewer bits
                check(pool, workers, base, d, n / 3, uint64_t(n * 3.4) + 5, alphabet);       // mixed
                check(pool, workers, base, d, n / 2, 0, alphabet);                           // floor of a mixed number
            }
            // structured strings: zeros, all b-1, one digit then zeros
            for (size_t n : {size_t(7), size_t(64), size_t(300)}) {
                std::vector<uint8_t> zeros(n, 0), top(n, uint8_t(base - 1)), one(n, 0);
                one[0] = 1;
                for (const auto &d : {zeros, top, one}) {
                    check(pool, workers, base, d, 0, 500, false);
                    check(pool, workers, base, d, n, 0, false);
                    check(pool, workers, base, d, 1, 64, false);
                }
            }
        }
        // Exactly representable fractions: a short leaf divides exactly; larger
        // requests exercise the approximate quotient and multiply-back witness.
        for(unsigned base:{3u,10u,63u})for(uint64_t bits:{63u,64u,65u,127u,128u,129u})
            for(size_t digits:{300u,513u,20000u})
                check(pool,workers,base,std::vector<uint8_t>(digits,uint8_t(base-1)),0,bits,false);
        for(unsigned base:{3u,10u,63u})for(size_t digits:{1025u,4095u,4096u,4097u,8193u,18001u}){
            const auto d=random_digits(base,digits);
            check(pool,workers,base,d,digits,0,true);
            check(pool,workers,base,d,0,64*448,false);
            check(pool,workers,base,d,digits/2,64*448+1,true);
        }
        const std::vector<uint8_t> half{5}, eighths{3, 7, 5}, padded{0, 6, 2, 5, 0, 0, 0, 0, 0, 0};
        for (uint64_t bits : {uint64_t(1), uint64_t(4), uint64_t(64), uint64_t(1000), uint64_t(100000)}) {
            check(pool, workers, 10, half, 0, bits, false, bits>64*sbn::v3::radix::parse_leaf_limbs);
            check(pool, workers, 10, eighths, 0, bits, false, bits>64*sbn::v3::radix::parse_leaf_limbs);
            check(pool, workers, 10, padded, 0, bits, true, bits>64*sbn::v3::radix::parse_leaf_limbs);
            check(pool, workers, 12, std::vector<uint8_t>{6}, 0, bits, false, bits>64*sbn::v3::radix::parse_leaf_limbs); // 6/12
        }
        {
            auto d = random_digits(10, 3000);
            for (size_t j = 1500; j < 3000; ++j)
                d[j] = 0;
            check(pool, workers, 10, d, 1000, 9000, false);
        }
        for (unsigned base : {10u, 7u}) {
            const auto d = random_digits(base, 60000);
            check(pool, workers, base, d, 60000, 0, false);
            check(pool, workers, base, d, 0, uint64_t(60000 * log2(double(base))) + 40, base == 10);
            check(pool, workers, base, d, 100, 150000, false);
        }
    }
    printf("radix parse: %llu conversions (%llu exact fallbacks, %llu invalid inputs located, %llu integer round trips) "
           "against exact arithmetic PASS\n",
           (unsigned long long)stats.cases, (unsigned long long)stats.fallbacks, (unsigned long long)stats.invalid,
           (unsigned long long)stats.round_trips);
}
