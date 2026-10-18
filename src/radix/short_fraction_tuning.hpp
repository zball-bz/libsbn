#pragma once
#include "radix/geometry.hpp"
#include "radix/word_base.hpp"
namespace sbn::v3::radix {
// Direct extraction performs one multiply-by-word per radix word. Its work
// follows input precision times the requested output rounds, less the known
// low zeros introduced by an even radix. The tree alternative pays a fixed
// setup plus linear input and digit-emission work. Both sides are expressed in
// scalar limb-product equivalents (native fresh crossover in bases 3/10/63).
// No tree classes or candidate plans are built to choose this small path.
// 0: SRT; 1: native u64 word extraction; 2: carry-deferred IFMA extraction.
inline unsigned short_fraction_kind(const BaseInfo &base,size_t limbs,uint64_t digits,bool enclosed,bool repeated=false) noexcept {
    const uint64_t total=digits+(enclosed?word_digits:0);
    const double fragments=double((total+fragment_digits-1)/fragment_digits);
    if(limbs>=32){
        const auto b=fraction_bases[base.base];
        const double rounds=double((total+b.digits-1)/b.digits);
        const double rectangle=double((64*limbs+407)/408)*rounds;
        const double zeros=double(base.twos*b.digits)*rounds*(rounds-1)/816.;
        const double work=rectangle>zeros?rectangle-zeros:0.;
        // Whole fresh calls in bases 3/10/63; alternate 1/16-octave points
        // held out. One work inequality, no candidate plans or length table.
        // A retained tree avoids setup on subsequent values. Its measured
        // crossover is lower, but short direct conversions still win. The
        // same work estimate covers bases 3/10/63 and partial output lengths.
        // Decimal's published emitter makes the fresh serial leaf cheaper;
        // price the same vector-work expression against the measured tree
        // startup cost. No transform size or individual length exceptions.
        const double startup=base.base==10?5600.:4000.;
        return work<=(repeated?2000.+5.*fragments:startup+15.*fragments)?2:0;
    }
    const auto wb=word_bases[base.base];
    const double rounds=double((total+wb.digits-1)/wb.digits);
    const double rectangular=double(limbs)*rounds;
    const double zeros=double(base.twos*wb.digits)*rounds*(rounds-1)/128.;
    const double steps=rectangular>zeros?rectangular-zeros:0.;
    return steps<=4800.+23.*double(limbs)+78.*fragments?1:0;
}
}
