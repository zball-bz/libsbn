#pragma once
namespace sbn::v3::u52 {
inline uint64_t prefix_word(const sb_limb *a,size_t bit) {
    const size_t at=bit/52;const unsigned s=unsigned(bit%52);
    uint64_t word=(a[at]>>s)|(a[at+1]<<(52-s));
    if(s>40)word|=a[at+2]<<(104-s);
    return word;
}
inline unsigned __int128 prefix_mulhi128(uint64_t x0,uint64_t x1,uint64_t y0,uint64_t y1) {
    using Wide=unsigned __int128;
    const Wide lo=Wide(x0)*y0;
    const Wide a=Wide(x1)*y0+(lo>>64);
    const Wide b=Wide(x0)*y1+uint64_t(a);
    return Wide(x1)*y1+(a>>64)+(b>>64);
}
// A short initial quotient uses few scalar digits against each stationary
// divisor vector. Keep the merged two-vector update for the full steps.
template<unsigned J,unsigned K>
inline sb_vec prefix_harvest(sb_vec (&cur)[K+1],sb_vec (&prev)[K+1]) {
    if constexpr(J==0)return cur[0];
    else{
        const sb_vec part=sb_alignr64(cur[J],prev[J],8-J);prev[J]=cur[J];
        return sb_add(prefix_harvest<J-1,K>(cur,prev),part);
    }
}
template<unsigned K>
inline sb_vec prefix_submul(sb_limb *np,const sb_limb *dp,size_t blocks,const sb_limb *q) {
    static_assert(K>=1 && K<=4);
    sb_vec prev[K+1]{},cur[K+1]{},quot[K];
    for(unsigned j=0;j<K;++j)quot[j]=sb_set1_64(q[j]);
    sb_vec carry=sb_zero();int sig=0;unsigned borrow=0;
    sb_pvec out=(sb_pvec)np;
    for(size_t i=0;i<blocks;++i){
        const sb_vec d=sb_load((sb_cpvec)(dp+8*i));
        cur[0]=sb_madd52lo(sb_zero(),d,quot[0]);
        for(unsigned j=1;j<K;++j)cur[j]=sb_madd52lo(sb_madd52hi(sb_zero(),d,quot[j-1]),d,quot[j]);
        cur[K]=sb_madd52hi(sb_zero(),d,quot[K-1]);
        sb_vec product=prefix_harvest<K,K>(cur,prev);
        canonize(product,carry,sig);
        const sb_vec original=sb_load((sb_cpvec)out);borrow_prop(out,original,product,borrow);
    }
    for(unsigned j=0;j<=K;++j)cur[j]=sb_zero();
    sb_vec high=prefix_harvest<K,K>(cur,prev);canonize(high,carry,sig);
    return block_addc(high,sb_zero(),&borrow);
}
template<unsigned Tail,unsigned J=0>
inline void prefix_tail_acc(sb_vec (&t)[17],sb_vec d,const sb_vec (&q)[Tail]) {
    t[8+J]=sb_madd52lo(t[8+J],d,q[J]);t[9+J]=sb_madd52hi(t[9+J],d,q[J]);
    if constexpr(J+1<Tail)prefix_tail_acc<Tail,J+1>(t,d,q);
}
template<unsigned Tail>
inline _u832 prefix_submul_wide(sb_limb *np,const sb_limb *dp,size_t blocks,const sb_limb *q) {
    static_assert(Tail>=1 && Tail<=4);
    const sb_vec first=sb_load((sb_cpvec)q);sb_vec tail[Tail];
    for(unsigned j=0;j<Tail;++j)tail[j]=sb_set1_64(q[8+j]);
    sb_vec t[17]{},cv=sb_zero();int sig=0;unsigned borrow=0;sb_pvec out=(sb_pvec)np;
    for(size_t i=0;i<blocks;++i,dp+=8){
        D2B_ACC(t,first,dp);
        if(i)prefix_tail_acc<Tail>(t,sb_load((sb_cpvec)(dp-8)),tail);
        sb_vec value;D2B_HARV(t,value);canonize(value,cv,sig);
        const sb_vec original=sb_load((sb_cpvec)out);borrow_prop(out,original,value,borrow);
    }
    D2B_ZERO_HI(t);prefix_tail_acc<Tail>(t,sb_load((sb_cpvec)(dp-8)),tail);
    sb_vec spill0,spill1;D2B_HARV(t,spill0);canonize(spill0,cv,sig);
    D2B_ZERO_HI(t);D2B_HARV(t,spill1);canonize(spill1,cv,sig);
    spill0=block_addc(spill0,sb_zero(),&borrow);spill1=block_addc(spill1,sb_zero(),&borrow);return {spill0,spill1};
}

// Divide the high window by D, producing at most bits+1 quotient bits.
// bits is in [1,831]; j is a multiple of 16 u52 digits. The invariant after
// this step is np[j..j+8*dn) < D, so all lower steps can remain two-vector.
// V18 and the estimate exponent are the same as the existing div2b leaf:
// quotient = (top_936_bits * V18) >> (1871-bits), up to the cold correction.
inline void divide_prefix(sb_limb *qp,sb_limb *np,const sb_limb *dp,size_t dn,size_t j,
                          unsigned bits,const sb_limb *inverse) {
    alignas(64) sb_limb product[48]{},q[24]{};
    constexpr uint64_t mask=(uint64_t(1)<<52)-1;
    const unsigned digits=(bits+52)/52;
    if(bits<=96){
        // Two 128-bit prefixes suffice for a <=96-bit initial quotient.
        // Dropping their tails adds <2^(bits-126) quotient ulps of
        // under-estimate; the same exact correction handles the boundary.
        const size_t at=52*(j+8*dn)+bits-128;
        auto value=prefix_mulhi128(prefix_word(np,at),prefix_word(np,at+64),
                                  prefix_word(inverse,808),prefix_word(inverse,872));
        value>>=127-bits;q[0]=uint64_t(value)&mask;q[1]=uint64_t(value>>52)&mask;
    }else{
        const size_t origin=52*(j+8*dn)+bits-936;
        const size_t digit=origin/52;const unsigned shift=unsigned(origin%52);
        // Keep the input in its existing digit grid. One extra digit and a
        // shifted output funnel replace an 18-digit bit-shifted copy.
        const sb_limb *input=np+digit;
        if(bits<=255){dc_estimate_upper<28,19>(product,inverse,input);u52_canon_pos((sb_pvec)(product+28),(sb_cpvec)(product+28),16);}
        else if(bits<=448){dc_estimate_upper<24,19>(product,inverse,input);u52_canon_pos((sb_pvec)(product+24),(sb_cpvec)(product+24),16);}
        else{dc_estimate_upper<16,19>(product,inverse,input);u52_canon_pos((sb_pvec)(product+16),(sb_cpvec)(product+16),24);}
        const unsigned power=1871-bits+shift,start=power/52,offset=power%52;
        for(unsigned i=0;i<digits;++i)q[i]=((product[start+i]>>offset)|(product[start+i+1]<<(52-offset)))&mask;
    }
    sb_vec q0=sb_load((sb_cpvec)q),q1=sb_load((sb_cpvec)(q+8));
    sb_limb *base=np+j;const size_t top=j+8*dn;
    _u832 cy;
    switch(digits){
        case 1:cy={prefix_submul<1>(base,dp,dn,q),sb_zero()};break;
        case 2:cy={prefix_submul<2>(base,dp,dn,q),sb_zero()};break;
        case 3:cy={prefix_submul<3>(base,dp,dn,q),sb_zero()};break;
        case 4:cy={prefix_submul<4>(base,dp,dn,q),sb_zero()};break;
        case 9:cy=prefix_submul_wide<1>(base,dp,dn,q);break;
        case 10:cy=prefix_submul_wide<2>(base,dp,dn,q);break;
        case 11:cy=prefix_submul_wide<3>(base,dp,dn,q);break;
        case 12:cy=prefix_submul_wide<4>(base,dp,dn,q);break;
        default:cy=digits<=8?_u832{block_submul_vec(base,dp,dn,q0),sb_zero()}:blk_submul16(base,dp,dn,q0,q1);break;
    }
    unsigned borrow=0;
    sb_vec t0=block_subb(sb_load((sb_cpvec)(np+top)),cy.lo,&borrow);
    sb_vec t1=block_subb(sb_load((sb_cpvec)(np+top+8)),cy.hi,&borrow);
    sb_store((sb_pvec)(np+top),t0);sb_store((sb_pvec)(np+top+8),t1);
    while(__builtin_expect(borrow!=0,0)){
        unsigned b=1;q0=block_subb(q0,sb_zero(),&b);q1=block_subb(q1,sb_zero(),&b);
        unsigned carry=block_add_n(base,dp,dn);
        t0=block_addc(sb_load((sb_cpvec)(np+top)),sb_zero(),&carry);
        t1=block_addc(sb_load((sb_cpvec)(np+top+8)),sb_zero(),&carry);
        sb_store((sb_pvec)(np+top),t0);sb_store((sb_pvec)(np+top+8),t1);borrow-=carry;
    }
    while(__builtin_expect((uint8_t)sb_neq(sb_load((sb_cpvec)(np+top)),sb_zero()) ||
                           (uint8_t)sb_neq(sb_load((sb_cpvec)(np+top+8)),sb_zero()) || block_ge_n(base,dp,dn),0)){
        unsigned carry=1;q0=block_addc(q0,sb_zero(),&carry);q1=block_addc(q1,sb_zero(),&carry);
        unsigned b=block_sub_n(base,dp,dn);
        t0=block_subb(sb_load((sb_cpvec)(np+top)),sb_zero(),&b);
        t1=block_subb(sb_load((sb_cpvec)(np+top+8)),sb_zero(),&b);
        sb_store((sb_pvec)(np+top),t0);sb_store((sb_pvec)(np+top+8),t1);
    }
    sb_store((sb_pvec)(qp+j),q0);sb_store((sb_pvec)(qp+j+8),q1);
}
}
