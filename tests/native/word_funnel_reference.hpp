#pragma once
// Rejected word-funnel experiment: intentionally absent from production Dec.
struct WordFunnelReference {
    p::V wa_idx[4],wb_idx[4],piece_mask[4];__mmask32 wb_msk[4]{};__mmask8 odd_byte[4]{};
    explicit WordFunnelReference(int T){
        const int tb=T/8,npc=(T+47)/48;
        for(int k=0;k<npc;++k){const int len=(k+1)*6<=tb?6:tb-6*k;
        alignas(64) uint16_t wa[32]{},wb[32]{};
        for(int l=0;l<8;++l){
            const unsigned offset=l*tb+6*k, first=offset/2;
            const bool second=first+4>64;
            for(int w=0;w<4;++w){wa[4*l+w]=uint16_t(first+w);wb[4*l+w]=uint16_t(first+w-32);}
            if(second)wb_msk[k]|=__mmask32(15u<<(4*l));
            if(offset&1)odd_byte[k]|=__mmask8(1u<<l);
        }
        wa_idx[k]=_mm512_load_si512(wa);wb_idx[k]=_mm512_load_si512(wb);
        piece_mask[k]=p::vset((uint64_t(1)<<(8*len))-1);

        }
    }
    p::V piece(int k,p::V L0,p::V L1,p::V L2) const {
        p::V pc=_mm512_permutex2var_epi16(L0,wa_idx[k],L1);
        if(wb_msk[k])pc=_mm512_mask_mov_epi16(pc,wb_msk[k],_mm512_permutex2var_epi16(L1,wb_idx[k],L2));
        pc=_mm512_mask_srli_epi64(pc,odd_byte[k],pc,8);
        return _mm512_and_si512(pc,piece_mask[k]);

    }
};
