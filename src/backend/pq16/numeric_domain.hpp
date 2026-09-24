#pragma once
#include "backend/pq16/kernels.hpp"
namespace sbn::v3::pq16 {
// 16-bit CT cyclic envelope: full-ring x bounded short support, odd radix 3/5/7.
// Adversarial raw-coefficient gate: fft_cyclic_precision, 2026-09-23.
// The narrow pair carry proof additionally requires min digits <= 2^16.
inline constexpr unsigned ct16_cyclic_points=32768;
// Independently measured cyclic envelope: full long support, short support
// below 9/16 period (5/8 for balanced 16-bit). Coefficient and integer gates
// cover opposed signs, maximum/preceding sizes, and factored CT cross roots.
// The signed emitter counts actual digit support in its carry-bias bound.
// These entries intentionally do not inherit changes to the linear envelope.
inline constexpr unsigned cyclic_balanced_cap(unsigned bits,unsigned radix) noexcept {
    const unsigned column=radix==1?0:radix==3?1:radix==5?2:3;
    constexpr unsigned caps[5][4]={{131072,196608,163840,229376},
        {32768,32768,32768,32768},{16384,12288,20480,14336},
        {4096,6144,5120,3584},{2048,1536,1280,1792}};
    return bits>=16&&bits<=20?caps[bits-16][column]:0;
}
inline constexpr size_t cyclic_max_words=229376ul*16/32;
// Independent bounded plus-ring envelope: unsigned 16-bit / balanced 17–18, long support <r
// and short support <9r/16. The 18-bit M3/N12288 probe reaches 0.25 and is
// excluded. These caps are not inherited from ordinary linear multiplication.
inline constexpr unsigned bounded_plus_cap(unsigned bits,unsigned radix) noexcept {
    if(radix!=3&&radix!=5&&radix!=7)return 0;
    const unsigned col=radix==3?0:radix==5?1:2;
    constexpr unsigned caps[3][3]={{12288,20480,28672},{12288,20480,14336},{6144,5120,3584}};
    return bits>=16&&bits<=18?caps[bits-16][col]:0;
}
// Balanced-digit envelope (2026-09-08, results/balanced_digits_2026-09-08/signed-sweep.jsonl): largest N whose
// worst adversarial coefficient error with the production signed kernels stays below 0.25 (shapes measuring
// exactly 0.25 are excluded). Not universal FP proofs.
inline unsigned balanced_cap(Recipe r,unsigned bits,unsigned radix) noexcept {
    const bool rac=r==Recipe::RightAngle;
    // 16-bit balanced (2026-09-09, results/small_wide_2026-09-09/envelope-large16{,b}.jsonl; the 32-bit pair slots take a
    // two-bit carry from N 2^17 on): CT/PQ M1 262144 and M3 393216 / M5 327680 measure 0.25, M7 458752 0.375; right-angle
    // M5 163840 0.28, M7 229376 0.31.
    if(bits==16)return rac?(radix==3?196608:radix==5?81920:114688):(radix==1?131072:radix==3?196608:radix==5?163840:229376);
    if(bits==17)return rac?(radix==3?24576:radix==5?20480:28672):32768;
    if(bits==18)return rac?(radix==3?12288:radix==5?10240:14336):(radix==1?16384:radix==3?12288:radix==5?20480:14336);
    if(bits==19)return rac?(radix==3?3072:radix==5?2560:1792):(radix==1?4096:radix==3?6144:radix==5?5120:3584);
    // 20-bit (2026-09-09, results/small_wide_2026-09-09/envelope-small.jsonl): the next shape up measures 0.31-0.5.
    // 21-bit digits are out: 0.25 already at the smallest shapes (N 512 / 768), 0.375-0.5 elsewhere.
    if(bits==20)return rac?(radix==3?768:radix==5?640:896):(radix==1?2048:radix==3?1536:radix==5?1280:1792);
    return 0;
}
struct VariableInput {
    size_t limbs=0,unsigned_required=0,balanced_required=0;
    unsigned bits=0;
    bool nonzero=false;
};
inline VariableInput variable_input(size_t an,size_t bn,unsigned bits) noexcept {
    const size_t a=64*an,b=64*bn,qa=a/bits,qb=b/bits;
    // Unsigned: ceil(a/bits) digits. Balanced: floor(a/bits)+1,
    // including the possible carry digit. Compute each division once for
    // all radix/recipe candidates at this width.
    const size_t total=qa+qb+unsigned(a%bits!=0)+unsigned(b%bits!=0);
    return {an+bn,total?total-1:0,qa+qb+1,bits,an!=0 && bn!=0};
}
inline bool variable_supported(Shape s,const VariableInput &input) noexcept {
    // 2026-09-09: the codec opens at 128-limb operands (an+bn >= 256) and takes 20-bit digits (small band; 21-bit
    // measured out of the envelope at every shape). 16-bit balanced digits run the same kernels for the large
    // band (an+bn up to 2^18 limbs), replacing the centered PFA path for odd radix at W1.
    if(s.centered||s.bits<16||s.bits>20||s.bits!=input.bits||!input.nonzero||input.limbs<256)return false;
    if(s.bits==16){if(!s.balanced||input.limbs>262144)return false;}else if(input.limbs>16384)return false;
    if(s.recipe!=Recipe::CooleyTukeyPQ&&!(s.recipe==Recipe::RightAngle&&s.radix!=1))return false;
    if(s.branch<128&&s.recipe!=Recipe::RightAngle)return false;   // branch 64 (M5 320 / M7 448): the PQ pair pointwise needs >= 2 groups; right-angle only
    // Empirical error-envelope caps (worst adversarial sample error < 0.25; shapes measuring exactly 0.25 are
    // excluded, as for the balanced codec), not universal FP proofs. Re-validated 2026-09-08 with the odd-radix
    // branch plan (results/odd_radix_2026-09-08/envelopes).
    unsigned cap;
    if(s.balanced)cap=balanced_cap(s.recipe,s.bits,s.radix);
    else if(s.recipe==Recipe::CooleyTukeyPQ)cap=s.bits==17?32768:s.bits==18?(s.radix==5?5120:8192):s.bits==19?2048:(s.radix==1?512:0); // 20b unsigned: pow2 512 only (odd radix 0.25 at the smallest shapes)
    else{ // right-angle: 1.3-3.6x the CT/PQ error at equal shapes (results/rac_2026-09-08/gate-wide.jsonl)
        if(s.bits==17)cap=s.radix==3?12288:s.radix==5?10240:14336;
        else if(s.bits==18)cap=s.radix==3?3072:s.radix==5?2560:3584;   // M5 5120 measures exactly 0.25
        else if(s.bits==19)cap=s.radix==3?768:s.radix==5?1280:896;     // M3 1536 measures exactly 0.25; M7 896 0.156, 1792 0.31 (2026-09-09)
        else cap=0;                                                    // 20b unsigned right-angle: >= 0.5 everywhere
    }
    // digits per operand; the balanced codec may append a carry digit when the top digit is complete
    return s.nfull<=cap&&(s.balanced?input.balanced_required:input.unsigned_required)<=2*size_t(s.nfull);
}
inline bool variable_supported(Shape s,size_t an,size_t bn) noexcept {
    if(s.centered||s.bits<16||s.bits>20||!an||!bn||an+bn<256)return false;
    if(s.bits==16){if(!s.balanced||an+bn>262144)return false;}else if(an+bn>16384)return false;
    return variable_supported(s,variable_input(an,bn,s.bits));
}
}
