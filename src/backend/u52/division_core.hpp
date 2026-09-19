// Imported from libsbn v2 detail/div/div2b.h. Core and attribution retained;
// upper-band estimates have the additional bound in division_estimate.hpp.
// Includes are centralized in the private IFMA island. The caller has
// at least 17 divisor limbs, so the top-window loads stay within the input.
#pragma once
namespace sbn::v3::u52 {
extern "C" void sbn3i_recip_u1024(const uint64_t *,uint64_t *);
// The division stack (u52 block domain, beta = 2^416 = 8 digits = one zmm).
//
// One core: extended-precision Barrett long division producing TWO quotient
// blocks per step (one 1-block parity step when the count is odd).  There is
// no div3by2 and no 3/2 block reciprocal: the estimate is a 18x18-digit
// product of the remainder window (top 936 bits of the frame) against an
// 18-digit reciprocal built once per division from recip_u1024 on the top
// 1024 bits of the divisor.  Estimate slack is ~2^-104 (under) / ~2^-186
// (over) quotient ULPs, so corrections are a COLD +-1 branch (validated in
// bench/recip_2b_seed.c; fuzzed against GMP in bench/div2b_bench.c and
// bench/dc2b_bench.c).  The exponent algebra is grid-independent, so the same
// core is the flat wrapper's engine and dc_div.h's recursion leaf.
//
// The rank-1 update is a fused bn=16 streaming submul: b0*dp_i and b1*dp_{i-1}
// land at identical output digits, so both accumulate into one diagonal array
// per block -- 32 madds per block for one harvest, one canonize, one subtract.

#define INLINE static inline __attribute__((always_inline))

// canon.h defines `canonize`, but mul_vec.h (pulled in via div.h) #undef's it;
// redefine the positive streaming canonicalize here (one SWAR carry pass).
#ifndef canonize
#define canonize(vec, carry, sigcarry) do {                                    \
    const sb_vec M = SB_MASK52();                                               \
    sb_vec hi  = sb_srli(vec, 52);                                              \
    sb_vec his = sb_alignr64(hi, carry, 7);                                     \
    sb_vec _ct = sb_add(sb_and(vec, M), his);                                   \
    carry    = hi;                                                              \
    unsigned g = (unsigned)sb_gtu(_ct, M);                                      \
    unsigned p = (unsigned)sb_eq (_ct, M);                                      \
    unsigned chain = p + ((g << 1) | sigcarry);                                \
    sigcarry = chain >> 8;                                                      \
    __mmask8 cin = (__mmask8)(p ^ chain);                                      \
    vec = sb_and(sb_sub(_ct, M, cin, _ct), M);                                 \
} while(0)
#endif

// ===================== block primitives (base beta = 2^416) =====================

typedef struct _u832{ sb_vec lo, hi; } _u832;

// canonical block a + b + (*c) -> canonical block; *c = carry-out bit
INLINE sb_vec block_addc(sb_vec a, sb_vec b, unsigned* c){
    const sb_vec M = SB_MASK52(); const sb_vec ONE = sb_set1_64(1);
    sb_vec res = sb_add(a, b);                       // lanes < 2^53
    unsigned p = (unsigned)sb_eq(res, M);
    unsigned g = (unsigned)sb_gtu(res, M);
    unsigned cf = (g << 1) | (*c);
    if(!p){                                          // no propagate lane (common)
        *c = cf >> 8;
        return sb_and(sb_add(res, ONE, (__mmask8)cf, res), M);
    }
    unsigned ch = p + cf; *c = ch >> 8;
    __mmask8 ci = (__mmask8)(p ^ ch);
    return sb_and(sb_add(res, ONE, ci, res), M);     // +1 (mod 2^52) on carry lanes
}
// canonical block a - b - (*bw) mod beta -> canonical block; *bw = borrow-out bit
INLINE sb_vec block_subb(sb_vec a, sb_vec b, unsigned* bw){
    const sb_vec M = SB_MASK52();
    sb_vec res = sb_sub(a, b);                        // a-b lanewise (mod 2^64)
    unsigned p = (unsigned)sb_eq(res, sb_zero());     // ==0 propagates borrow
    unsigned g = (unsigned)sb_gtu(res, M);            // underflow -> borrow generate
    unsigned bf = (g << 1) | (*bw);
    if(!p){                                          // no zero lane -> no cascade (common)
        *bw = bf >> 8;
        return sb_and(sb_add(res, M, (__mmask8)bf, res), M);
    }
    unsigned ch = p + bf; *bw = ch >> 8;
    __mmask8 bi = (__mmask8)(p ^ ch);
    return sb_and(sb_add(res, M, bi, res), M);        // -1 (mod 2^52) on borrow lanes
}
// canonical compares (a>=b)
INLINE int block_ge(sb_vec a, sb_vec b){
    uint8_t ne = (uint8_t)sb_neq(a, b); if(!ne) return 1;
    uint8_t lt = (uint8_t)sb_ltu(a, b); return (uint8_t)(ne ^ lt) > lt;
}
INLINE int block_ge2(sb_vec a1, sb_vec a0, sb_vec b1, sb_vec b0){
    uint8_t ne = (uint8_t)sb_neq(a1, b1);
    if(ne){ uint8_t lt = (uint8_t)sb_ltu(a1, b1); return (uint8_t)(ne ^ lt) > lt; }
    return block_ge(a0, b0);
}


// all 8 digits equal?
INLINE int block_eq(sb_vec a, sb_vec b){ return (uint8_t)sb_neq(a, b) == 0; }

// np[0..len) += dp[0..len)  (len blocks); returns carry-out bit
INLINE unsigned block_add_n(sb_limb* np, const sb_limb* dp, uint64_t len){
    unsigned c = 0;
    for(uint64_t i = 0; i < len; i++){
        sb_vec x = sb_load((sb_cpvec)(np + 8*i));
        x = block_addc(x, sb_load((sb_cpvec)(dp + 8*i)), &c);
        sb_store((sb_pvec)(np + 8*i), x);
    }
    return c;
}

// np[0..len) -= dp[0..len)  (blocks); returns borrow-out bit
INLINE unsigned block_sub_n(sb_limb* np, const sb_limb* dp, uint64_t len){
    unsigned b = 0;
    for(uint64_t i = 0; i < len; i++){
        sb_vec x = sb_load((sb_cpvec)(np + 8*i));
        x = block_subb(x, sb_load((sb_cpvec)(dp + 8*i)), &b);
        sb_store((sb_pvec)(np + 8*i), x);
    }
    return b;
}

// compare canonical len-block arrays: 1 if a>b, -1 if a<b, 0 if equal.
INLINE int block_cmp(const sb_limb* a, const sb_limb* b, uint64_t len){
    for(uint64_t i = len; i-- > 0; ){
        sb_vec av = sb_load((sb_cpvec)(a + 8*i)), bv = sb_load((sb_cpvec)(b + 8*i));
        if(!block_eq(av, bv)) return block_ge(av, bv) ? 1 : -1;
    }
    return 0;
}

// canonical multi-block compare: a[0..len) >= b[0..len) ? (top-down early exit)
INLINE int block_ge_n(const sb_limb* a, const sb_limb* b, uint64_t len){
    for(uint64_t i = len; i-- > 0; ){
        sb_vec x = sb_load((sb_cpvec)(a + 8*i)), y = sb_load((sb_cpvec)(b + 8*i));
        unsigned g = (unsigned)sb_gtu(x, y), l = (unsigned)sb_ltu(x, y);
        if(g != l) return g > l;
    }
    return 1;
}

// np[0..len) -= qhat * dp[0..len)  (len blocks, len >= 1); returns the high
// carry block cy such that (np_old - qhat*dp) = np_new - cy*beta^len.
//
// Fused streaming submul, modelled on mul_u52_vec: it accumulates the product
// with deferred carry (the t[] diagonal), and per output block does exactly one
// canonize (resolve the product block) + one borrow_prop (np[i] -= block),
// threading a single borrow.  That is 2 canon-class passes per block instead of
// the 5 of a per-block mul_zmm + canon2 + 3 carry ops.
INLINE sb_vec block_submul_vec(sb_limb* np, const sb_limb* dp, uint64_t len, sb_vec b){
    sb_vec t[17];
#define mulacc_s(ind) {                                                 \
        t[(ind)+8] = sb_madd52lo(t[(ind)+8], b, sb_splat_load(dp, ind)); \
        t[(ind)+9] = sb_zero();                                         \
        t[(ind)+9] = sb_madd52hi(t[(ind)+9], b, sb_splat_load(dp, ind)); \
    }
    t[8]=sb_zero(); t[1]=sb_zero(); t[2]=sb_zero(); t[3]=sb_zero();
    t[4]=sb_zero(); t[5]=sb_zero(); t[6]=sb_zero(); t[7]=sb_zero();
    t[0]=sb_zero();                       // t[0] = canonize carry (product overflow)
    int sig = 0;                          // SWAR carry for the product canonize
    unsigned bw = 0;                      // borrow chain for the subtract
    sb_pvec rp = (sb_pvec)np;
    for(uint64_t n = len; n; --n, dp += 8){
        mulacc_s(0); mulacc_s(1); mulacc_s(2); mulacc_s(3);
        mulacc_s(4); mulacc_s(5); mulacc_s(6); mulacc_s(7);
        sb_vec k1,k2,k3,k4,k12,k34;
        k1 = sb_add(t[8],                        sb_alignr64(t[9],  t[1], 7)); t[1]=t[9];  t[8]=t[16];
        k2 = sb_add(sb_alignr64(t[10], t[2], 6), sb_alignr64(t[11], t[3], 5)); t[2]=t[10]; t[3]=t[11];
        k3 = sb_add(sb_alignr64(t[12], t[4], 4), sb_alignr64(t[13], t[5], 3)); t[4]=t[12]; t[5]=t[13];
        k4 = sb_add(sb_alignr64(t[14], t[6], 2), sb_alignr64(t[15], t[7], 1)); t[6]=t[14]; t[7]=t[15];
        k12 = sb_add(k1, k2); k34 = sb_add(k3, k4); k12 = sb_add(k12, k34);
        canonize(k12, t[0], sig);                  // canonical product block
        sb_vec npv = sb_load((sb_cpvec)rp);
        borrow_prop(rp, npv, k12, bw);             // *rp = npv - k12 ; rp++ ; thread bw
    }
    // the multiply's high spill-over block (= product block `len`)
    t[9]=sb_zero(); t[10]=sb_zero(); t[11]=sb_zero(); t[12]=sb_zero();
    t[13]=sb_zero(); t[14]=sb_zero(); t[15]=sb_zero();
    sb_vec k1 = sb_add(t[8],                        sb_alignr64(t[9],  t[1], 7));
    sb_vec k2 = sb_add(sb_alignr64(t[10], t[2], 6), sb_alignr64(t[11], t[3], 5));
    sb_vec k3 = sb_add(sb_alignr64(t[12], t[4], 4), sb_alignr64(t[13], t[5], 3));
    sb_vec k4 = sb_add(sb_alignr64(t[14], t[6], 2), sb_alignr64(t[15], t[7], 1));
    sb_vec top = sb_add(sb_add(k1,k2), sb_add(k3,k4));
    canonize(top, t[0], sig);
    unsigned c = bw;                               // cy = top + final borrow (top <= beta-2)
    return block_addc(top, sb_zero(), &c);
#undef mulacc_s
}

// Fused decode + left-shift: r = (value of ap[0..an)) << s, as u52 digits.
// One pass over ap -- no intermediate shifted-u64 buffer.  The whole-digit part
// of the shift (kd = s/52) is the output digit offset; the sub-digit part
// (kb = s%52) is a u52 funnel across digits during the decode.  r must be
// pre-zeroed for out_blocks blocks (caller does this; the low kd digits and any
// padding stay zero).
static inline void u52_from_u64_lsh(sb_pvec r, const uint64_t* ap, uint64_t an, uint64_t s){
    const uint64_t kd = s / 52; const unsigned kb = (unsigned)(s % 52);
    const uint64_t bits = u64_bit_length(ap, an);
    const uint64_t n52 = (bits + 51) / 52;
    const uint8_t* p = (const uint8_t*)ap;
    int64_t rem = (int64_t)(an * 8);
    const sb_vec perm = sb_load((sb_cpvec)u52_dec_perm);
    const sb_vec sh = sb_setr_64(0, 4, 0, 4, 0, 4, 0, 4);
    const sb_vec kbv = sb_set1_64(kb), k52 = sb_set1_64(52 - kb), M = SB_MASK52();
    sb_limb* out = (sb_limb*)r + kd;
    uint64_t blocks = (n52 + 7) >> 3, bi = 0;
    sb_vec prev = sb_zero();
    for(; blocks; --blocks, p += 52, rem -= 52, ++bi){
        const sb_vec w = sb__fn(maskz_loadu_epi8)(
            rem >= 64 ? ~0ull : (rem > 0 ? (~0ull >> (64 - rem)) : 0), p);
        sb_vec dec = sb_and(sb_srlv(sb_permb(perm, w), sh), M);          // 8 u52 digits
        sb_vec o = kb ? sb_and(sb_or(sb_sllv(dec, kbv), sb_srlv(sb_alignr64(dec, prev, 7), k52)), M)
                    : dec;                                      // funnel by kb
        sb_store((sb_pvec)(out + 8*bi), o);
        prev = dec;
    }
    if(kb)  // top digit's spill-out -> one more digit
        sb_store((sb_pvec)(out + 8*bi), sb_and(sb_srlv(sb_alignr64(sb_zero(), prev, 7), k52), M));
}

// out[0..blocks) = (src as u52 number) >> s, in u52 digits.  Mirror of the
// fused decode-shift: digit offset kd = s/52, sub-digit funnel kb = s%52.
// Reads src[kd .. kd+8*blocks] (one digit past the last block -> caller
// zero-pads that block).  In-place safe (out == src).
static inline void u52_rshift(sb_limb* out, const sb_limb* src, uint64_t blocks, uint64_t s){
    const uint64_t kd = s / 52; const unsigned kb = (unsigned)(s % 52);
    const sb_vec kbv = sb_set1_64(kb), k52 = sb_set1_64(52 - kb), M = SB_MASK52();
    const sb_limb* sp = src + kd;
    for(uint64_t bk = 0; bk < blocks; ++bk){
        sb_vec cur = sb_load((sb_cpvec)(sp + 8*bk));
        sb_vec o = kb ? sb_and(sb_or(sb_srlv(cur, kbv),
                                 sb_sllv(sb_load((sb_cpvec)(sp + 8*bk + 1)), k52)), M)
                    : cur;                            // (cur>>kb)|(next<<(52-kb))
        sb_store((sb_pvec)(out + 8*bk), o);
    }
}

// out[kd .. kd+8*blocks] = (src as 8*blocks u52 digits) << s, in u52 digits
// (kd = s/52 whole-digit offset, kb = s%52 sub-digit funnel); out[0..kd) is
// zeroed and ONE spill digit is always written at out[kd+8*blocks] (zero when
// kb == 0 or nothing shifts out), so out must hold 8*blocks + kd + 1 digits.
// Mirror of u52_rshift. blocks >= 1. In-place safe (out == src): descending
// sweep, every store lands at or above the highest not-yet-consumed read.
static inline void u52_lshift(sb_limb* out, const sb_limb* src, uint64_t blocks, uint64_t s){
    const uint64_t kd = s / 52; const unsigned kb = (unsigned)(s % 52);
    const sb_vec kbv = sb_set1_64(kb), k52 = sb_set1_64(52 - kb), M = SB_MASK52();
    sb_limb* op = out + kd;
    op[8*blocks] = kb ? (src[8*blocks - 1] >> (52 - kb)) : 0;
    sb_vec cur = sb_load((sb_cpvec)(src + 8*(blocks - 1)));
    for(uint64_t bk = blocks; bk-- > 0; ){
        const sb_vec prev = bk ? sb_load((sb_cpvec)(src + 8*bk - 8)) : sb_zero();
        // (cur<<kb)|(digit-below>>(52-kb)); kb==0 degenerates to cur (52-bit
        // digits shifted right by 52 vanish), so one path serves all kb
        sb_store((sb_pvec)(op + 8*bk),
                  sb_and(sb_or(sb_sllv(cur, kbv), sb_srlv(sb_alignr64(cur, prev, 7), k52)), M));
        cur = prev;
    }
    for(uint64_t i = 0; i < kd; ++i) out[i] = 0;
}

// Build the 18-digit reciprocal V18 (+6 zero pad digits) from the divisor.
// dp/dn64: the raw (unshifted) divisor limbs (dn64 >= 14, i.e. bitlen > 832).
// V18 = floor(recip_u1024(top 1024 bits of D) / 2^88): ~2^1959 / Dtop.
// Vectorized: the 1024-bit top window is two masked loads + one funnel shift
// per vector (fault-suppressed masked loads cover dn64 < 17), and the
// digit extraction reuses the fused decoder: V18 = digits 2..19 of the u52
// decode of R<<16 (digit i+2 of R<<16 = bits [88+52i, 88+52i+52) of R).
static inline void div2b_recip(sb_limb* V18 /*24 digits*/,
                               const uint64_t* dp, uint64_t dn64){
    uint64_t T16[16] __attribute__((aligned(64)));
    uint64_t R[16];
    const unsigned cl = (unsigned)__builtin_clzll(dp[dn64 - 1]);
    const sb_vec cnt = sb_set1_64(cl);
    for(int k = 0; k < 2; k++){
        const int64_t idx = (int64_t)dn64 - 16 + 8*k;
        const __mmask8 ma = (__mmask8)(idx     >= 0 ? 0xFF : 0xFF << -idx);
        const __mmask8 mb = (__mmask8)(idx - 1 >= 0 ? 0xFF : 0xFF << -(idx - 1));
        sb_vec a = sb_load((sb_cpvec)(dp + idx),     ma);
        sb_vec b = sb_load((sb_cpvec)(dp + idx - 1), mb);
        sb_store((sb_pvec)(T16 + 8*k), sb_fshl64v(a, b, cnt));
    }
    int pow2 = (T16[15] == (1ull << 63));
    if(pow2) for(int i = 0; i < 15 && pow2; i++) pow2 = (T16[i] == 0);
    if(pow2){                       // Dtop = 2^1023: true recip 2^1024 overflows;
        for(int i = 0; i < 16; i++) R[i] = ~0ull;   // use 2^1024-1 (1 ULP under)
    } else {
        sbn3i_recip_u1024(T16, R);
    }
    sb_limb V32[40] __attribute__((aligned(64)));
    memset(V32, 0, sizeof V32);
    u52_from_u64_lsh((sb_pvec)V32, R, 16, 16);
    for(int k = 0; k < 3; k++)
        sb_store((sb_pvec)(V18 + 8*k), sb_load((sb_cpvec)(V32 + 2 + 8*k)));
}

// ===================== quotient estimate =====================

// Two-block estimate: compute only the upper product band, normalize its
// guarded window and funnel to 16 quotient digits. qh[16]!=0 means clamp.
static inline void div2b_est3(const sb_limb* V18, const sb_limb* uw,
                              sb_limb* qh, sb_limb* S /*48 digits*/){
    const sb_vec M = SB_MASK52();
    dc_estimate_upper<16>(S,V18,uw);
    u52_canon_pos((sb_pvec)(S + 16), (sb_cpvec)(S + 16), 24);
    sb_vec a0 = sb_load((sb_cpvec)(S + 19)), b0 = sb_load((sb_cpvec)(S + 20));
    sb_vec a1 = sb_load((sb_cpvec)(S + 27)), b1 = sb_load((sb_cpvec)(S + 28));
    sb_store((sb_pvec)qh,       sb_and(sb_or(sb_srli(a0, 51), sb_slli(b0, 1)), M));
    sb_store((sb_pvec)(qh + 8), sb_and(sb_or(sb_srli(a1, 51), sb_slli(b1, 1)), M));
    qh[16] = S[35] >> 51;                   // (P[35]>>51 | P[36]<<1), P[36]=0
}

// Parity-step estimate (one 1-block quotient step, used by dc_div's leaf when
// the quotient block count is odd). The same upper-product method, with a
// narrower guarded band and a different
// funnel anchor differs: q^ = (U^*V18) >> (27 digits + 51 bits), derived from
// the grid-independent exponent algebra with a 416-bit step (E = 2895-1024-416).
// qh[0..7] = q^ digits; qh[8] != 0 => clamp to B-1.
static inline void div2b_est1b(const sb_limb* V18, const sb_limb* uw,
                               sb_limb* qh, sb_limb* S /*48 digits*/){
    const sb_vec M = SB_MASK52();
    dc_estimate_upper<24>(S,V18,uw);
    u52_canon_pos((sb_pvec)(S + 24), (sb_cpvec)(S + 24), 16);
    sb_vec a = sb_load((sb_cpvec)(S + 27)), b = sb_load((sb_cpvec)(S + 28));
    sb_store((sb_pvec)qh, sb_and(sb_or(sb_srli(a, 51), sb_slli(b, 1)), M));
    qh[8] = S[35] >> 51;
}

// ===================== fused bn=16 streaming submul =====================

// np[0..len blocks) -= <b1,b0> * dp[0..len blocks); returns the 2-block high
// carry cy so that np_old - q*dp = np_new - cy*beta^len.  Two phase-shifted
// diagonal pipelines (b1's products land one block higher), one shared
// canonize+borrow stream: 32 madds per block for one canonize + one subtract.
#define D2B_ACC(t, b, dp) do{                                                  \
    t[8]  = sb_madd52lo(t[8],  b, sb_splat_load(dp, 0));                       \
    t[9]  = sb_zero();  t[9]  = sb_madd52hi(t[9],  b, sb_splat_load(dp, 0));   \
    t[9]  = sb_madd52lo(t[9],  b, sb_splat_load(dp, 1));                       \
    t[10] = sb_zero();  t[10] = sb_madd52hi(t[10], b, sb_splat_load(dp, 1));   \
    t[10] = sb_madd52lo(t[10], b, sb_splat_load(dp, 2));                       \
    t[11] = sb_zero();  t[11] = sb_madd52hi(t[11], b, sb_splat_load(dp, 2));   \
    t[11] = sb_madd52lo(t[11], b, sb_splat_load(dp, 3));                       \
    t[12] = sb_zero();  t[12] = sb_madd52hi(t[12], b, sb_splat_load(dp, 3));   \
    t[12] = sb_madd52lo(t[12], b, sb_splat_load(dp, 4));                       \
    t[13] = sb_zero();  t[13] = sb_madd52hi(t[13], b, sb_splat_load(dp, 4));   \
    t[13] = sb_madd52lo(t[13], b, sb_splat_load(dp, 5));                       \
    t[14] = sb_zero();  t[14] = sb_madd52hi(t[14], b, sb_splat_load(dp, 5));   \
    t[14] = sb_madd52lo(t[14], b, sb_splat_load(dp, 6));                       \
    t[15] = sb_zero();  t[15] = sb_madd52hi(t[15], b, sb_splat_load(dp, 6));   \
    t[15] = sb_madd52lo(t[15], b, sb_splat_load(dp, 7));                       \
    t[16] = sb_zero();  t[16] = sb_madd52hi(t[16], b, sb_splat_load(dp, 7));   \
}while(0)

#define D2B_HARV(t, out) do{                                                   \
    sb_vec k1_ = sb_add(t[8],                        sb_alignr64(t[9],  t[1], 7)); \
    t[1] = t[9];  t[8] = t[16];                                                \
    sb_vec k2_ = sb_add(sb_alignr64(t[10], t[2], 6), sb_alignr64(t[11], t[3], 5)); \
    t[2] = t[10]; t[3] = t[11];                                                \
    sb_vec k3_ = sb_add(sb_alignr64(t[12], t[4], 4), sb_alignr64(t[13], t[5], 3)); \
    t[4] = t[12]; t[5] = t[13];                                                \
    sb_vec k4_ = sb_add(sb_alignr64(t[14], t[6], 2), sb_alignr64(t[15], t[7], 1)); \
    t[6] = t[14]; t[7] = t[15];                                                \
    out = sb_add(sb_add(k1_, k2_), sb_add(k3_, k4_));                          \
}while(0)

#define D2B_ZERO_HI(t) do{ t[9]=sb_zero(); t[10]=sb_zero(); t[11]=sb_zero();   \
    t[12]=sb_zero(); t[13]=sb_zero(); t[14]=sb_zero(); t[15]=sb_zero(); t[16]=sb_zero(); }while(0)


// accumulate-only variant (no zero-reset): used to fold b1*dp_{i-1} into the
// SAME diagonal array as b0*dp_i (their products land at identical digits).
#define D2B_ACCB(t, b, dp) do{                                                 \
    t[8]  = sb_madd52lo(t[8],  b, sb_splat_load(dp, 0));                       \
    t[9]  = sb_madd52hi(t[9],  b, sb_splat_load(dp, 0));                       \
    t[9]  = sb_madd52lo(t[9],  b, sb_splat_load(dp, 1));                       \
    t[10] = sb_madd52hi(t[10], b, sb_splat_load(dp, 1));                       \
    t[10] = sb_madd52lo(t[10], b, sb_splat_load(dp, 2));                       \
    t[11] = sb_madd52hi(t[11], b, sb_splat_load(dp, 2));                       \
    t[11] = sb_madd52lo(t[11], b, sb_splat_load(dp, 3));                       \
    t[12] = sb_madd52hi(t[12], b, sb_splat_load(dp, 3));                       \
    t[12] = sb_madd52lo(t[12], b, sb_splat_load(dp, 4));                       \
    t[13] = sb_madd52hi(t[13], b, sb_splat_load(dp, 4));                       \
    t[13] = sb_madd52lo(t[13], b, sb_splat_load(dp, 5));                       \
    t[14] = sb_madd52hi(t[14], b, sb_splat_load(dp, 5));                       \
    t[14] = sb_madd52lo(t[14], b, sb_splat_load(dp, 6));                       \
    t[15] = sb_madd52hi(t[15], b, sb_splat_load(dp, 6));                       \
    t[15] = sb_madd52lo(t[15], b, sb_splat_load(dp, 7));                       \
    t[16] = sb_madd52hi(t[16], b, sb_splat_load(dp, 7));                       \
}while(0)

INLINE _u832 blk_submul16(sb_limb* np, const sb_limb* dp, uint64_t len,
                          sb_vec b0, sb_vec b1){
    // Merged single-pipeline: per block i, accumulate b0*dp_i (with the
    // zero-reset ACC) and b1*dp_{i-1} (accumulate-only) into ONE diagonal
    // array -- their products land at identical output digits.  One harvest
    // (7 alignr) per block, ~24 live vectors (the two-pipeline version
    // spilled at 34+).  Lane bound: <= 4 madd terms per entry < 2^54, well
    // under the canonize limit.
    sb_vec t[17];
    for(int i = 0; i < 17; i++) t[i] = sb_zero();
    sb_vec cv = sb_zero(); int sig = 0; unsigned bw = 0;
    sb_pvec rp = (sb_pvec)np;
    for(uint64_t i = 0; i < len; i++, dp += 8){
        D2B_ACC(t, b0, dp);
        if(i) D2B_ACCB(t, b1, dp - 8);
        sb_vec k;
        D2B_HARV(t, k);
        canonize(k, cv, sig);
        sb_vec npv = sb_load((sb_cpvec)rp);
        borrow_prop(rp, npv, k, bw);
    }
    sb_vec cy0, cy1;
    D2B_ACC(t, b1, dp - 8);           // block len: b1*dp_{len-1} low part
    D2B_HARV(t, cy0);                 //   (+ rotated highs already in t[1..7])
    canonize(cy0, cv, sig);
    D2B_ZERO_HI(t);
    D2B_HARV(t, cy1);                 // block len+1: spill
    canonize(cy1, cv, sig);
    unsigned c = bw;
    cy0 = block_addc(cy0, sb_zero(), &c);
    cy1 = block_addc(cy1, sb_zero(), &c);
    _u832 ret_; ret_.lo = cy0; ret_.hi = cy1; return ret_;
}

// Specialized estimate kernel (est=2).  NOTE: measured SLOWER than est=3 in
// C (the ~34 live vector accumulators spill; the generic basecase's schedule
// is already near-optimal for 18x18) -- kept as the blueprint for an asm
// version, where register-residency makes the fused form win.
// Computes the full 18x18 product
// P = U^*V with the blk_submul16 diagonal-pipeline structure -- v0/v1 (V18's
// two blocks) are the stationary vectors, the 18 window digits are broadcast
// straight from the dividend (no staging buffer); the 2-digit reciprocal
// extension rides a role-swapped mini-pipeline (V[16],V[17] broadcast x the
// window vectors, two rolling alignrs).  Canonize is fused into the harvest
// chain, product blocks 2..4 stay in registers, and q^ is extracted with a
// vector funnel (q^[i] = (P[19+i]>>51 | P[20+i]<<1) & M52).
// Bit-identical to div2b_est(est=0) by construction.
// uw = pointer to the window's low digit (np + top - 2); digits [18..) above
// the window must read as zero (the loop invariant guarantees this).
static inline void div2b_est2(const sb_limb* V18, const sb_limb* uw, sb_limb* qh){
    const sb_vec M   = SB_MASK52();
    const sb_vec v0  = sb_load((sb_cpvec)V18);
    const sb_vec v1  = sb_load((sb_cpvec)(V18 + 8));
    sb_vec t[17], s[17];
    for(int i = 0; i < 17; i++){ t[i] = sb_zero(); s[i] = sb_zero(); }
    sb_vec cv = sb_zero(); int sig = 0;
    sb_vec Yp = sb_zero(), Zp = sb_zero();
    sb_vec P2r = sb_zero(), P3r = sb_zero(), P4r = sb_zero();
    for(int k = 0; k <= 4; k++){
        sb_vec k0 = sb_zero(), k1 = sb_zero(), km = sb_zero();
        if(k <= 1){
            D2B_ACC(t, v0, uw + 8*k);
            D2B_HARV(t, k0);
        } else if(k == 2){                         // window digits 16,17 only
            t[8]  = sb_madd52lo(t[8], v0, sb_splat_load(uw, 16));
            t[9]  = sb_zero(); t[9] = sb_madd52hi(t[9], v0, sb_splat_load(uw, 16));
            t[9]  = sb_madd52lo(t[9], v0, sb_splat_load(uw, 17));
            t[10] = sb_zero(); t[10] = sb_madd52hi(t[10], v0, sb_splat_load(uw, 17));
            t[11] = sb_zero(); t[12] = sb_zero(); t[13] = sb_zero();
            t[14] = sb_zero(); t[15] = sb_zero(); t[16] = sb_zero();
            D2B_HARV(t, k0);
        } else if(k == 3){
            D2B_ZERO_HI(t);
            D2B_HARV(t, k0);                       // v0 spill
        }
        if(k >= 1 && k <= 3){
            D2B_HARV(s, k1);                       // v1: one block behind
        } else if(k == 4){
            D2B_ZERO_HI(s);
            D2B_HARV(s, k1);                       // v1 spill
        }
        if(k <= 1){
            D2B_ACC(s, v1, uw + 8*k);
        } else if(k == 2){
            s[8]  = sb_madd52lo(s[8], v1, sb_splat_load(uw, 16));
            s[9]  = sb_zero(); s[9] = sb_madd52hi(s[9], v1, sb_splat_load(uw, 16));
            s[9]  = sb_madd52lo(s[9], v1, sb_splat_load(uw, 17));
            s[10] = sb_zero(); s[10] = sb_madd52hi(s[10], v1, sb_splat_load(uw, 17));
            s[11] = sb_zero(); s[12] = sb_zero(); s[13] = sb_zero();
            s[14] = sb_zero(); s[15] = sb_zero(); s[16] = sb_zero();
        }
        if(k >= 2){                                // extension mini-pipeline
            sb_vec uv = (k == 4) ? sb_load((sb_cpvec)(uw + 16), (__mmask8)0x03)
                               : sb_load((sb_cpvec)(uw + 8*(k - 2)));
            sb_vec X  = sb_madd52lo(sb_zero(), uv, sb_splat_load(V18, 16));
            sb_vec Y  = sb_madd52hi(sb_zero(), uv, sb_splat_load(V18, 16));
            Y         = sb_madd52lo(Y,      uv, sb_splat_load(V18, 17));
            sb_vec Z  = sb_madd52hi(sb_zero(), uv, sb_splat_load(V18, 17));
            km = sb_add(X, sb_add(sb_alignr64(Y, Yp, 7), sb_alignr64(Z, Zp, 6)));
            Yp = Y; Zp = Z;
        }
        sb_vec P = sb_add(sb_add(k0, k1), km);
        canonize(P, cv, sig);
        if(k == 2) P2r = P; else if(k == 3) P3r = P; else if(k == 4) P4r = P;
    }
    sb_vec a0 = sb_alignr64(P3r, P2r, 3), b0 = sb_alignr64(P3r, P2r, 4);
    sb_vec a1 = sb_alignr64(P4r, P3r, 3), b1 = sb_alignr64(P4r, P3r, 4);
    sb_store((sb_pvec)qh,       sb_and(sb_or(sb_srli(a0, 51), sb_slli(b0, 1)), M));
    sb_store((sb_pvec)(qh + 8), sb_and(sb_or(sb_srli(a1, 51), sb_slli(b1, 1)), M));
    sb_limb p4[8] __attribute__((aligned(64)));
    sb_store((sb_pvec)p4, P4r);
    qh[16] = p4[3] >> 51;                          // (P[35]>>51 | P[36]<<1), P[36]=0
}

// ===================== the division core (416-bit block grid) =====================
// np: nn blocks (modified -> remainder in the low dn blocks, consumed tops
// zeroed); dp: dn >= 2 blocks, normalized (top digit bit 51 set); quotient to
// qp[0..nn-dn).  Returns qh (the implicit top quotient block: 1 iff the top
// dn dividend blocks were >= d; pre-subtracted).  V18 = div2b_recip of the
// divisor's top 1024 bits -- in the D&C recursion every sub-divisor is a top
// slice of D, so one V18 serves the whole tree (leaves span >= 1024 bits for
// dc_thresh >= 8; shorter divisors only cost warm corrections).  Quotient is
// produced 2 blocks per step; odd counts take one 1-block parity step first
// (same estimate product, funnel base 27 instead of 19).  `est`: 2 = fused
// blueprint estimate kernel, else div2b_est3.
static int div2b_core(sb_limb* qp, sb_limb* np, uint64_t nn, const sb_limb* dp, uint64_t dn,
                      const sb_limb* V18, int est){
    const uint64_t qn = nn - dn;
    int qh = block_cmp(np + 8*qn, dp, dn) >= 0;
    if(qh) block_sub_n(np + 8*qn, dp, dn);
    sb_limb S[48]  __attribute__((aligned(64)));
    sb_limb qd[17] __attribute__((aligned(64)));
    uint64_t j = qn;
    if(j & 1){                                     // parity: one 1-block step
        j--;
        sb_limb* base = np + 8*j;
        const uint64_t top = 8*(j + dn);
        div2b_est1b(V18, np + top - 10, qd, S);
        sb_vec qhat = __builtin_expect(qd[8] != 0, 0) ? SB_MASK52()
                                                    : sb_load((sb_cpvec)qd);
        sb_vec cy = block_submul_vec(base, dp, dn, qhat);
        unsigned b = 0;
        sb_vec t0 = block_subb(sb_load((sb_cpvec)(np + top)), cy, &b);
        sb_store((sb_pvec)(np + top), t0);
        while(__builtin_expect(b != 0, 0)){
            unsigned bb = 1;
            qhat = block_subb(qhat, sb_zero(), &bb);
            unsigned c = block_add_n(base, dp, dn);
            t0 = block_addc(sb_load((sb_cpvec)(np + top)), sb_zero(), &c);
            sb_store((sb_pvec)(np + top), t0);
            b -= c;
        }
        while(__builtin_expect(
                (uint8_t)sb_neq(sb_load((sb_cpvec)(np + top)), sb_zero()) ||
                block_ge_n(base, dp, dn), 0)){
            unsigned c = 1;
            qhat = block_addc(qhat, sb_zero(), &c);
            unsigned bb = block_sub_n(base, dp, dn);
            t0 = block_subb(sb_load((sb_cpvec)(np + top)), sb_zero(), &bb);
            sb_store((sb_pvec)(np + top), t0);
        }
        sb_store((sb_pvec)(qp + 8*j), qhat);
    }
    while(j){                                      // 2-block steps
        j -= 2;
        sb_limb* base = np + 8*j;
        const uint64_t top = 8*(j + dn);
        if(__builtin_expect(est == 2, 0))
            div2b_est2(V18, np + top - 2, qd);
        else
            div2b_est3(V18, np + top - 2, qd, S);
        sb_vec q0, q1;
        if(__builtin_expect(qd[16] != 0, 0)){
            q0 = q1 = SB_MASK52();
        } else {
            q0 = sb_load((sb_cpvec)qd);
            q1 = sb_load((sb_cpvec)(qd + 8));
        }
        _u832 cy = blk_submul16(base, dp, dn, q0, q1);
        unsigned b = 0;
        sb_vec t0 = block_subb(sb_load((sb_cpvec)(np + top)),     cy.lo, &b);
        sb_vec t1 = block_subb(sb_load((sb_cpvec)(np + top + 8)), cy.hi, &b);
        sb_store((sb_pvec)(np + top),     t0);
        sb_store((sb_pvec)(np + top + 8), t1);
        while(__builtin_expect(b != 0, 0)){
            unsigned bb = 1;
            q0 = block_subb(q0, sb_zero(), &bb);
            q1 = block_subb(q1, sb_zero(), &bb);
            unsigned c = block_add_n(base, dp, dn);
            t0 = block_addc(sb_load((sb_cpvec)(np + top)),     sb_zero(), &c);
            t1 = block_addc(sb_load((sb_cpvec)(np + top + 8)), sb_zero(), &c);
            sb_store((sb_pvec)(np + top),     t0);
            sb_store((sb_pvec)(np + top + 8), t1);
            b -= c;
        }
        while(__builtin_expect(
                (uint8_t)sb_neq(sb_load((sb_cpvec)(np + top)), sb_zero()) ||
                (uint8_t)sb_neq(sb_load((sb_cpvec)(np + top + 8)), sb_zero()) ||
                block_ge_n(base, dp, dn), 0)){
            unsigned c = 1;
            q0 = block_addc(q0, sb_zero(), &c);
            q1 = block_addc(q1, sb_zero(), &c);
            unsigned bb = block_sub_n(base, dp, dn);
            unsigned b2 = bb;
            t0 = block_subb(sb_load((sb_cpvec)(np + top)),     sb_zero(), &b2);
            t1 = block_subb(sb_load((sb_cpvec)(np + top + 8)), sb_zero(), &b2);
            sb_store((sb_pvec)(np + top),     t0);
            sb_store((sb_pvec)(np + top + 8), t1);
        }
        sb_store((sb_pvec)(qp + 8*j),     q0);
        sb_store((sb_pvec)(qp + 8*j + 8), q1);
    }
    return qh;
}


} // namespace sbn::v3::u52
