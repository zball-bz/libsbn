// SPDX-License-Identifier: LGPL-3.0-or-later
// Included once by each existing p48 instance. Reuses its optimized digit,
// vector lag-chain and packing kernels; no second CRT algorithm.
#include "product/crt.hpp"
namespace {
struct alignas(64) ExternalCrt {
    uint64_t marker=0x53424e3343525431ULL;
    p::EmitK digits;
    p::Pack8 pack;
};
size_t external_crt_bytes(unsigned T) noexcept {
    if(T<64 || T>232 || T%8 || 2*T>=48*PN || 48*PN-2*T>64 || (T+63)/64>p::WDMAX)return 0;
    return (sizeof(ExternalCrt)+63)&~size_t(63);
}
bool external_crt_capacity(unsigned T,uint64_t terms) noexcept {
    return external_crt_bytes(T) && coefficient_bound::fits(p::crt_product,T,terms);
}
void external_crt_prepare_scaled(unsigned T,bool preconditioned,std::span<uint8_t> bytes) noexcept {
    require(external_crt_bytes(T) && bytes.data() && !(uintptr_t(bytes.data())&63) && bytes.size()>=external_crt_bytes(T),
            SBN3_FATAL_WORKSPACE,"external CRT preparation");
    auto &c=*::new(bytes.data())ExternalCrt{};
    p::Primes primes{};uint8_t dummy;auto frame=Frame::external(&dummy,0);primes.init(frame,1);
    p::Plan plan{};plan.T=int(T);plan.C=1;plan.rowscale=1;plan.Lg=1;
    const unsigned mode=default_crt(PN);
    p::Garner garner{};garner.init(primes,plan,int(mode));
    // Row/outer inverses normalize and cancel REDC. A product may also fold
    // input_scale(q) into its existing middle normalization multiply. The
    // prepared descriptor records which of these two representations arrives.
    garner.nosc=mode==1||preconditioned;
    for(unsigned q=0;q<PN;++q)garner.sc[q]=p::cc_of(mode==2&&!preconditioned?p::cofactor_inverse(int(q)):1,p::PR[q]);
    c.digits.init(garner,primes,int(T));c.digits.gen=1;
    c.pack.init(int(T),int((T+63)/64));
}
void external_crt_prepare(unsigned T,std::span<uint8_t>bytes) noexcept {external_crt_prepare_scaled(T,false,bytes);}
uint64_t external_crt_input_scale(unsigned q) noexcept {
    require(q<PN,SBN3_FATAL_ARGUMENT,"external CRT prime");return default_crt(PN)==2?p::cofactor_inverse(int(q)):1;
}
template<int WD>
void external_crt_emit_words(const ExternalCrt&c,const uint64_t*input,size_t stride,size_t cells,uint8_t*out,product::CrtCarry&tail){
    const auto &K=c.digits;const p::V zero=_mm512_setzero_si512();
    p::V previous[WD],previous2=zero;for(int k=0;k<WD;++k)previous[k]=zero;
    unsigned carry=0;
    for(size_t i=0;i<cells;++i){
        p::V x[PN];for(unsigned q=0;q<PN;++q)x[q]=_mm512_loadu_si512(input+q*stride+i*8);
        const auto d=p::digits_of<WD>(K,x);p::V w[p::WDMAX]{},a1[WD];
        for(int k=0;k<WD;++k){w[k]=d.b0[k];a1[k]=_mm512_alignr_epi64(d.b1[k],previous[k],7);}
        const p::V a2=_mm512_alignr_epi64(d.b2,previous2,6);
        carry=p::lag_add<WD>(w,a1,a2,carry,K.LIM,K.HM,K.HB);
        c.pack.run(out+i*size_t(K.T),w);
        for(int k=0;k<WD;++k)previous[k]=d.b1[k];previous2=d.b2;
    }
    p::V w[p::WDMAX]{},a1[WD];for(int k=0;k<WD;++k)a1[k]=_mm512_alignr_epi64(zero,previous[k],7);
    const p::V a2=_mm512_alignr_epi64(zero,previous2,6);
    require(p::lag_add<WD>(w,a1,a2,carry,K.LIM,K.HM,K.HB)==0,SBN3_FATAL_MATH,"external CRT tail width");
    alignas(64) uint8_t packed[256]{};c.pack.run(packed,w);memcpy(tail.limb,packed,sizeof tail.limb);
    for(size_t i=sizeof tail.limb;i<size_t(K.T);++i)require(!packed[i],SBN3_FATAL_MATH,"external CRT carry capacity");
}
void external_crt_emit(std::span<const uint8_t> storage,const uint64_t*input,size_t stride,size_t cells,
                       std::span<uint8_t>output,product::CrtCarry&tail) noexcept {
    require(storage.data() && !(uintptr_t(storage.data())&63) && storage.size()>=sizeof(ExternalCrt),SBN3_FATAL_ARGUMENT,"external CRT context");
    const auto &c=*reinterpret_cast<const ExternalCrt *>(storage.data());size_t bytes=0,words=0,source=0;
    require(c.marker==0x53424e3343525431ULL && input && cells && mul_size(cells,8,words) && stride>=words &&
        mul_size(cells,size_t(c.digits.T),bytes)&&add_size(bytes,256,bytes)&&output.size()>=bytes&&output.data()&&
        mul_size(stride,PN-1,source)&&add_size(source,words,source)&&mul_size(source,8,source)&&
        !overlaps(input,source,output.data(),bytes)&&!overlaps(storage.data(),storage.size(),output.data(),bytes),
        SBN3_FATAL_WORKSPACE,"external CRT spans");
    if constexpr(p::WDMAX>=4)if(c.digits.WD==4){external_crt_emit_words<4>(c,input,stride,cells,output.data(),tail);return;}
    if constexpr(p::WDMAX>=3)if(c.digits.WD==3){external_crt_emit_words<3>(c,input,stride,cells,output.data(),tail);return;}
    if(c.digits.WD==2)external_crt_emit_words<2>(c,input,stride,cells,output.data(),tail);
    else external_crt_emit_words<1>(c,input,stride,cells,output.data(),tail);
}
}
#define SBN3_CRT_FN_(N) p48_crt_##N
#define SBN3_CRT_FN(N) SBN3_CRT_FN_(N)
namespace sbn::v3::product {
const CrtBackend &SBN3_CRT_FN(CR_NP)() noexcept {
    static const CrtBackend instance{CR_NP,external_crt_bytes,external_crt_capacity,external_crt_prepare,external_crt_emit,
        external_crt_input_scale,external_crt_prepare_scaled};return instance;
}
}
#undef SBN3_CRT_FN
#undef SBN3_CRT_FN_
