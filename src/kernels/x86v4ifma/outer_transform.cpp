// SPDX-License-Identifier: LGPL-3.0-or-later
#define CR_NP 10
#define SBN3_P48_NS p48_external_outer
#include "backend/ntt_p48/rows.hpp"
#include "product/outer_transform.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <new>
namespace sbn::v3::product {
namespace {
namespace p=::sbn::v3::p48_external_outer;
constexpr uint64_t magic=0x53424e334f555431ULL;
struct alignas(64) Prepared {
    uint64_t marker=magic;
    OuterTransformSpec spec{};
    OuterTransformInfo info{};
    p::Primes primes{};
    p::Dec decoder{};
    p::Pk52 codec{};
};
bool power2(size_t n){return n && !(n&(n-1));}
const Prepared &get(std::span<const uint8_t>b){
    require(b.data() && !(uintptr_t(b.data())&63) && b.size()>=sizeof(Prepared),SBN3_FATAL_ARGUMENT,"outer prepared span");
    const auto &p=*reinterpret_cast<const Prepared *>(b.data());require(p.marker==magic,SBN3_FATAL_ARGUMENT,"outer prepared state");return p;
}
struct Call {const Prepared*c;uint64_t column;unsigned first,count;const uint8_t*raw;uint8_t*bank,*scratch;
    size_t raw_stride=0,raw_column=0,packed_stride=0,packed_column=0;
    const uint8_t*prefix=nullptr;size_t prefix_cells=0,raw_first_row=0;};
void store_four(uint8_t*dst,p::V*const columns[4],size_t row,const p::Pk52&codec,const p::PrimeV&pv){
    p::V x[4];for(unsigned k=0;k<4;++k)x[k]=_mm512_permutexvar_epi8(codec.s48,p::shp(p::sh2(columns[k][row],pv.p2),pv.p));
    // Four 48-byte cells are exactly three lines. Complete lines avoid the
    // partial-line revisits caused by stepping through many page-strided rows.
    const p::V lines[]{
        _mm512_permutex2var_epi64(x[0],_mm512_setr_epi64(0,1,2,3,4,5,8,9),x[1]),
        _mm512_permutex2var_epi64(x[1],_mm512_setr_epi64(2,3,4,5,8,9,10,11),x[2]),
        _mm512_permutex2var_epi64(x[2],_mm512_setr_epi64(4,5,8,9,10,11,12,13),x[3])};
    for(unsigned k=0;k<3;++k){
        if(!(uintptr_t(dst)&63))_mm512_stream_si512(reinterpret_cast<__m512i *>(dst+64*k),lines[k]);
        else _mm512_storeu_si512(dst+64*k,lines[k]);
    }
}
template<unsigned Pieces,unsigned Windows>
__attribute__((always_inline)) inline void decode_four(const p::Dec&d,const uint8_t*const source[4],unsigned first,unsigned count,
                                                       p::V*x,size_t stride,size_t row){
    p::V pieces[4][Pieces];
    #pragma unroll
    for(unsigned k=0;k<4;++k){
        const p::V l0=_mm512_loadu_si512(source[k]),l1=_mm512_loadu_si512(source[k]+64);
        p::V l2=l0,l3=l0;
        if constexpr(Windows>=3)l2=_mm512_loadu_si512(source[k]+128);
        if constexpr(Windows>=4)l3=_mm512_loadu_si512(source[k]+192);
        #pragma unroll
        for(unsigned i=0;i<Pieces;++i){
            p::V v=_mm512_maskz_permutex2var_epi8(d.pa_msk[i],l0,d.pa_idx[i],l1);
            if constexpr(Windows>=3)if(d.pb_msk[i])v=_mm512_or_si512(v,_mm512_maskz_permutex2var_epi8(d.pb_msk[i],l1,d.pb_idx[i],l2));
            if constexpr(Windows>=4)if(d.fourth.mask[i])v=_mm512_or_si512(v,_mm512_maskz_permutex2var_epi8(d.fourth.mask[i],l2,d.fourth.index[i],l3));
            pieces[k][i]=v;
        }
    }
    for(unsigned q=0;q<count;++q){
        p::V lo[4]{},hi[4]{};
        #pragma unroll
        for(unsigned i=0;i<Pieces;++i){const p::V factor=d.vCw[i][first+q];
            #pragma unroll
            for(unsigned k=0;k<4;++k){lo[k]=_mm512_madd52lo_epu64(lo[k],pieces[k][i],factor);hi[k]=_mm512_madd52hi_epu64(hi[k],pieces[k][i],factor);}
        }
        const p::V J=d.vJ[first+q],P=d.vP[first+q];
        #pragma unroll
        for(unsigned k=0;k<4;++k)x[(4*q+k)*stride+row]=p::redc(lo[k],hi[k],J,P,p::vset(p::M52));
    }
}
template<bool Cached=false,unsigned Pieces=0,unsigned Windows=0>
void forward_columns(void *ptr,uint64_t lo,uint64_t hi,unsigned rank){
    auto &a=*static_cast<Call *>(ptr);const auto &c=*a.c;const auto &s=c.spec;
    const size_t stride=s.outer_cells+1; // odd vector stride keeps adjacent columns out of one cache set
    auto *x=reinterpret_cast<p::V *>(a.scratch+rank*c.info.worker_bytes);
    for(size_t j=4*size_t(lo);j<std::min(s.stripe_cells,4*size_t(hi));j+=4){
        const unsigned columns=unsigned(std::min(size_t(4),s.stripe_cells-j));size_t live[4]{};
        for(unsigned k=0;k<columns;++k){const uint64_t at=a.column+j+k;
            live[k]=at<s.input_cells?size_t(std::min(uint64_t(s.outer_cells),1+(s.input_cells-1-at)/s.inner_cells)):0;}
        const size_t input_rows=*std::max_element(live,live+columns);
        const size_t common_rows=*std::min_element(live,live+columns);
        for(size_t r=0;r<input_rows;++r){const uint8_t*source_batch[4]{};
          for(unsigned k=0;k<columns;++k){
            if(r>=live[k])continue;
            // The next row is often many pages away. Request a bounded future
            // row while decoding this one; four adjacent cells share its line
            // streams. This uses no new staging/retention memory.
            constexpr size_t ahead=8;
            if(r+ahead<live[k]){
                const uint8_t*future;
                if constexpr(Cached){const size_t cell=(r+ahead)*s.inner_cells+a.column+j+k;
                    future=cell<a.prefix_cells?a.prefix+cell*s.trunk_bits:
                        a.raw+(r+ahead-a.raw_first_row)*a.raw_stride+(j+k)*s.trunk_bits;
                }else future=a.raw+(r+ahead)*a.raw_stride+a.raw_column+(j+k)*s.trunk_bits;
                _mm_prefetch(reinterpret_cast<const char*>(future),_MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(future+64),_MM_HINT_T0);
                if(s.trunk_bits>128)_mm_prefetch(reinterpret_cast<const char*>(future+128),_MM_HINT_T0);
                if(s.trunk_bits>192)_mm_prefetch(reinterpret_cast<const char*>(future+192),_MM_HINT_T0);
            }
            const uint8_t*source;
            if constexpr(Cached){const size_t cell=r*s.inner_cells+a.column+j+k;
                source=cell<a.prefix_cells?a.prefix+cell*s.trunk_bits:
                    a.raw+(r-a.raw_first_row)*a.raw_stride+(j+k)*s.trunk_bits;
            }else source=a.raw+r*a.raw_stride+a.raw_column+(j+k)*s.trunk_bits;
            if constexpr(Pieces){if(columns==4&&r<common_rows){source_batch[k]=source;continue;}}
            if(c.decoder.t8){
                const p::V l0=_mm512_loadu_si512(source),l1=_mm512_loadu_si512(source+64);
                const p::V l2=c.decoder.ld3?_mm512_loadu_si512(source+128):l0;
                const p::V l3=c.decoder.fourth.load?_mm512_loadu_si512(source+192):l0;
                p::V pieces[p::DECODE_PIECES];for(int k=0;k<c.decoder.npc;++k)pieces[k]=p::dec8_piece(c.decoder,k,l0,l1,l2,l3);
                for(unsigned q=0;q<a.count;++q)x[(4*q+k)*stride+r]=p::dec8_prepared(c.decoder,pieces,int(a.first+q));
            }else for(unsigned q=0;q<a.count;++q)x[(4*q+k)*stride+r]=p::dec1q(c.decoder,reinterpret_cast<const uint64_t *>(source),int(a.first+q),0);
          }
          if constexpr(Pieces){if(columns==4&&r<common_rows)decode_four<Pieces,Windows>(c.decoder,source_batch,a.first,a.count,x,stride,r);}
        }
        for(unsigned q=0;q<a.count;++q){
            p::V*data[4];for(unsigned k=0;k<columns;++k){data[k]=x+(4*q+k)*stride;
                // Blocked TFT never loads complete known-zero input blocks.
                // Only the incomplete last input block needs physical zeros;
                // the small vdH fallback still needs its whole zero tail.
                const size_t block=s.outer_cells>CR_BLK_MIN?s.outer_cells/8:s.outer_cells;
                const size_t zero_end=live[k]?(s.outer_cells>CR_BLK_MIN?
                    (live[k]+block-1)/block*block:s.outer_cells):0;
                for(size_t r=live[k];r<zero_end;++r)data[k][r]=_mm512_setzero_si512();
                p::tft_blk(data[k],s.outer_cells,0,s.live_rows,live[k],c.primes.P[a.first+q],c.primes.V[a.first+q]);}
            for(size_t r=0;r<s.live_rows;++r){auto*dst=a.bank+(q*s.live_rows+r)*a.packed_stride+a.packed_column+48*j;
                if(columns==4)store_four(dst,data,r,c.codec,c.primes.V[a.first+q]);
                else for(unsigned k=0;k<columns;++k)p::stslot<48>(dst+48*k,data[k][r],c.codec,c.primes.V[a.first+q]);}
        }
    }
}
template<bool Cached>
void forward_dispatch(sbn3_team_scope*scope,const Prepared&c,Call&a){
    const size_t tasks=(c.spec.stripe_cells+3)/4;
    switch(c.decoder.npc){
    case 2:sbn3_team_for(scope,0,tasks,1,SBN3_STATIC,forward_columns<Cached,2,2>,&a);break;
    case 3:if(c.decoder.ld3)sbn3_team_for(scope,0,tasks,1,SBN3_STATIC,forward_columns<Cached,3,3>,&a);
           else sbn3_team_for(scope,0,tasks,1,SBN3_STATIC,forward_columns<Cached,3,2>,&a);break;
    case 4:sbn3_team_for(scope,0,tasks,1,SBN3_STATIC,forward_columns<Cached,4,3>,&a);break;
    case 5:sbn3_team_for(scope,0,tasks,1,SBN3_STATIC,forward_columns<Cached,5,4>,&a);break;
    default:sbn3_team_for(scope,0,tasks,1,SBN3_STATIC,forward_columns<Cached>,&a);break;
    }
}
void expand_rows(void *ptr,uint64_t lo,uint64_t hi,unsigned){
    auto &a=*static_cast<Call *>(ptr);const auto &c=*a.c;const size_t n=c.spec.stripe_cells;
    for(size_t r=lo;r<hi;++r){
        auto *base=a.bank+r*n*64;
        for(size_t j=n;j-- >0;)_mm512_store_si512(base+j*64,p::ldslot<48>(base+j*48,c.codec));
    }
}
void inverse_columns(void *ptr,uint64_t lo,uint64_t hi,unsigned rank){
    auto &a=*static_cast<Call *>(ptr);const auto &c=*a.c;const auto &s=c.spec;
    const size_t stride=s.outer_cells+1;
    auto *x=reinterpret_cast<p::V *>(a.scratch+rank*c.info.worker_bytes);
    for(size_t j=4*size_t(lo);j<std::min(s.stripe_cells,4*size_t(hi));j+=4)for(unsigned q=0;q<a.count;++q){
        const unsigned columns=unsigned(std::min(size_t(4),s.stripe_cells-j));
        auto *bank=a.bank+q*s.live_rows*s.stripe_cells*64;
        for(size_t r=0;r<s.outer_cells;++r)for(unsigned k=0;k<columns;++k)
            x[k*stride+r]=r<s.live_rows?_mm512_load_si512(bank+(r*s.stripe_cells+j+k)*64):_mm512_setzero_si512();
        for(unsigned k=0;k<columns;++k)p::itft_blk<true>(x+k*stride,s.outer_cells,0,s.live_rows,c.primes.P[a.first+q],c.primes.V[a.first+q]);
        for(size_t r=0;r<s.live_rows;++r)for(unsigned k=0;k<columns;++k)
            _mm512_stream_si512(reinterpret_cast<__m512i *>(bank+(r*s.stripe_cells+j+k)*64),x[k*stride+r]);
    }
}
void validate(const Prepared &c,unsigned first,unsigned count,std::span<uint8_t>w,sbn3_team_scope *scope){
    require(count && count<=c.spec.prime_batch && first<c.spec.prime_count && count<=c.spec.prime_count-first &&
            w.data() && !(uintptr_t(w.data())&63) && w.size()>=c.info.workspace_bytes && scope && sbn3_team_width(scope)>=c.info.workers,
            SBN3_FATAL_WORKSPACE,"outer transform grants");require_scope_leader(scope);
}
}
bool p48_outer_query(const OuterTransformSpec&s,OuterTransformInfo&out) noexcept {
    if(!power2(s.outer_cells)||s.outer_cells>8192||!power2(s.inner_cells)||!power2(s.stripe_cells)||s.stripe_cells>s.inner_cells ||
       !s.live_rows||s.live_rows>s.outer_cells||s.trunk_bits<64||s.trunk_bits>232||s.trunk_bits%8 ||
       !s.prime_count||s.prime_count>10||!s.prime_batch||s.prime_batch>s.prime_count||!s.workers||s.workers>32)return false;
    size_t cells;if(!mul_size(s.outer_cells,s.inner_cells,cells)||cells>(uint64_t(1)<<32)||s.input_cells>cells)return false;
    OuterTransformInfo i{};i.workers=unsigned(std::min(size_t(s.workers),(s.stripe_cells+3)/4));
    size_t raw,rows;
    if(!mul_size(s.stripe_cells,s.trunk_bits,raw))return false;
    if(s.stripe_cells==s.inner_cells)i.raw_stride=raw;
    else if(!add_size(raw,64,raw)||!align_size(raw,64,i.raw_stride))return false;
    // The TFT never loads rows beyond input_cells (forward_columns' live[k]
    // guard). Known-zero rows need no physical staging. A full-width stripe
    // is contiguous and can also omit the unused tail of the final input row.
    size_t tail=0;raw=0;
    if(s.input_cells){
        const size_t last=size_t(s.input_cells-1),columns=std::min(s.stripe_cells,last%s.inner_cells+1);
        if(!mul_size(last/s.inner_cells,i.raw_stride,raw)||!mul_size(columns,s.trunk_bits,tail)||!add_size(raw,tail,raw))return false;
    }
    if(!add_size(raw,64,i.raw_bytes) ||
       !mul_size(s.stripe_cells,48,raw)||!align_size(raw,16384,i.fragment_bytes) ||
       !mul_size(s.live_rows,s.prime_batch,rows)||!mul_size(rows,i.fragment_bytes,i.packed_bytes)||
       !mul_size(rows,s.stripe_cells,raw)||!mul_size(raw,64,i.coefficient_bytes)||
       !mul_size(s.outer_cells+1,s.prime_batch,raw)||!mul_size(raw,4*64,i.worker_bytes)||
       !mul_size(i.worker_bytes,i.workers,i.workspace_bytes)||!align_size(sizeof(Prepared),64,i.prepared_bytes))return false;
    i.shared_bytes=p48_root_bank::bytes();out=i;return true;
}
void p48_outer_prepare(const OuterTransformSpec&s,const OuterTransformInfo&i,std::span<uint8_t>b) noexcept {
    OuterTransformInfo expected{};
    require(p48_outer_query(s,expected)&&expected.prepared_bytes==i.prepared_bytes&&expected.workspace_bytes==i.workspace_bytes&&
        b.data()&&!(uintptr_t(b.data())&63)&&b.size()>=expected.prepared_bytes,SBN3_FATAL_ARGUMENT,"outer preparation");
    auto &c=*::new(b.data())Prepared{};c.spec=s;c.info=expected;
    uint8_t dummy;auto frame=Frame::external(&dummy,0);
    c.primes.init(frame,size_t(__builtin_ctzll(s.outer_cells))+1);
    p::Plan geometry{};geometry.T=int(s.trunk_bits);geometry.LW=int(s.trunk_bits)-52;
    p::dec_init(c.decoder,nullptr,geometry,c.primes);c.codec=p::pk52_mk();
}
void p48_outer_forward(std::span<const uint8_t>b,uint64_t column,unsigned first,unsigned count,std::span<const uint8_t>raw,
                        std::span<uint8_t>packed,std::span<uint8_t>w,sbn3_team_scope *scope) noexcept {
    const auto &c=get(b);validate(c,first,count,w,scope);
    require(column<=c.spec.inner_cells-c.spec.stripe_cells&&raw.size()>=c.info.raw_bytes&&raw.data()&&
        packed.data()&&packed.size()>=c.info.packed_bytes&&!overlaps(raw.data(),raw.size(),packed.data(),packed.size())&&
        !overlaps(w.data(),w.size(),raw.data(),raw.size())&&!overlaps(w.data(),w.size(),packed.data(),packed.size())&&
        !overlaps(b.data(),b.size(),w.data(),w.size())&&!overlaps(b.data(),b.size(),packed.data(),packed.size()),SBN3_FATAL_ARGUMENT,"outer forward spans");
    Call a{&c,column,first,count,raw.data(),packed.data(),w.data()};
    a.raw_stride=c.info.raw_stride;a.packed_stride=c.info.fragment_bytes;
    sbn3_team_scope part{scope->team,scope->first,c.info.workers,false,scope->epoch};
    forward_dispatch<false>(&part,c,a);
    _mm_sfence();
}
void p48_outer_forward_resident(std::span<const uint8_t>b,uint64_t column,unsigned prime,
                        std::span<const uint8_t>raw,std::span<uint8_t>plane,size_t row_stride,
                        std::span<uint8_t>w,sbn3_team_scope*scope) noexcept {
    const auto&c=get(b);const auto&s=c.spec;validate(c,prime,1,w,scope);
    require(column<=s.inner_cells-s.stripe_cells&&raw.data()&&raw.size()>=s.input_cells*s.trunk_bits+64&&
        row_stride>=s.inner_cells*48&&plane.data()&&plane.size()>=s.live_rows*row_stride&&
        !overlaps(raw.data(),raw.size(),plane.data(),plane.size())&&
        !overlaps(w.data(),w.size(),raw.data(),raw.size())&&!overlaps(w.data(),w.size(),plane.data(),plane.size())&&
        !overlaps(b.data(),b.size(),plane.data(),plane.size()),SBN3_FATAL_ARGUMENT,"resident outer forward spans");
    Call a{&c,column,prime,1,raw.data(),plane.data(),w.data()};
    a.raw_stride=s.inner_cells*s.trunk_bits;a.raw_column=column*s.trunk_bits;
    a.packed_stride=row_stride;a.packed_column=column*48;
    sbn3_team_scope part{scope->team,scope->first,c.info.workers,false,scope->epoch};
    forward_dispatch<false>(&part,c,a);
    _mm_sfence();
}
void p48_outer_inverse(std::span<const uint8_t>b,unsigned first,unsigned count,std::span<uint8_t>bank,
                        std::span<uint8_t>w,sbn3_team_scope *scope) noexcept {
    const auto &c=get(b);validate(c,first,count,w,scope);
    require(bank.data()&&!(uintptr_t(bank.data())&63)&&bank.size()>=size_t(count)*c.spec.live_rows*c.spec.stripe_cells*64&&
        !overlaps(bank.data(),bank.size(),w.data(),w.size())&&!overlaps(bank.data(),bank.size(),b.data(),b.size())&&
        !overlaps(w.data(),w.size(),b.data(),b.size()),SBN3_FATAL_ARGUMENT,"outer inverse spans");
    Call a{&c,0,first,count,nullptr,bank.data(),w.data()};
    sbn3_team_scope part{scope->team,scope->first,c.info.workers,false,scope->epoch};
    sbn3_team_for(&part,0,count*c.spec.live_rows,1,SBN3_STATIC,expand_rows,&a);
    sbn3_team_for(&part,0,(c.spec.stripe_cells+3)/4,1,SBN3_STATIC,inverse_columns,&a);
    _mm_sfence();
}
void p48_outer_forward_cached(std::span<const uint8_t>b,uint64_t column,unsigned prime,
    std::span<const uint8_t>prefix,size_t prefix_cells,std::span<const uint8_t>suffix,
    std::span<uint8_t>plane,size_t row_stride,std::span<uint8_t>w,sbn3_team_scope*scope) noexcept {
    const auto&c=get(b);const auto&s=c.spec;validate(c,prime,1,w,scope);
    const size_t first=prefix_cells/s.inner_cells;
    const size_t skipped=first*c.info.raw_stride;
    const size_t needed=c.info.raw_bytes>skipped?c.info.raw_bytes-skipped:64;
    require(column<=s.inner_cells-s.stripe_cells&&prefix_cells<=s.input_cells&&prefix.data()&&
        prefix.size()>=prefix_cells*s.trunk_bits+64&&suffix.data()&&suffix.size()>=needed&&
        row_stride>=s.inner_cells*48&&plane.data()&&plane.size()>=s.live_rows*row_stride&&
        !overlaps(prefix.data(),prefix.size(),plane.data(),plane.size())&&
        !overlaps(suffix.data(),suffix.size(),plane.data(),plane.size())&&
        !overlaps(w.data(),w.size(),prefix.data(),prefix.size())&&
        !overlaps(w.data(),w.size(),suffix.data(),suffix.size())&&
        !overlaps(w.data(),w.size(),plane.data(),plane.size()),SBN3_FATAL_WORKSPACE,"cached outer forward spans");
    Call a{&c,column,prime,1,suffix.data(),plane.data(),w.data()};
    a.raw_stride=c.info.raw_stride;a.packed_stride=row_stride;a.packed_column=column*48;
    a.prefix=prefix.data();a.prefix_cells=prefix_cells;a.raw_first_row=first;
    sbn3_team_scope part{scope->team,scope->first,c.info.workers,false,scope->epoch};
    forward_dispatch<true>(&part,c,a);_mm_sfence();
}
bool p48_outer_emit_query(const OuterTransformSpec&s,OuterEmitInfo&out) noexcept {
    OuterTransformInfo base;if(!p48_outer_query(s,base))return false;
    OuterEmitInfo i;size_t cell;
    if(!mul_size(s.prime_count,s.live_rows,cell)||!mul_size(cell,64,cell))return false;
    i.tile_cells=std::min(s.stripe_cells,std::max(size_t(1),std::min(size_t(32),std::bit_floor((2u<<20)/cell))));
    i.tiles=s.stripe_cells/i.tile_cells;i.workers=unsigned(std::min(size_t(s.workers),i.tiles));
    size_t cursor=0,bytes;
    if(!mul_size(s.outer_cells+1,4*64,cursor))return false;i.coefficients=cursor;
    if(!mul_size(cell,i.tile_cells,bytes)||!add_size(cursor,bytes,cursor))return false;i.packed=cursor;
    if(!mul_size(i.tile_cells,s.trunk_bits,bytes)||!add_size(bytes,256,bytes)||!add_size(cursor,bytes,cursor)||
       !align_size(cursor,64,i.worker_bytes)||!mul_size(i.worker_bytes,i.workers,i.tails)||
       !mul_size(i.tiles,s.live_rows,bytes)||!mul_size(bytes,sizeof(CrtCarry),bytes)||
       !add_size(i.tails,bytes,i.workspace_bytes))return false;
    out=i;return true;
}
namespace {
struct EmitCall {
    const Prepared*c;OuterEmitInfo info;const uint8_t*bank;const CrtBackend*crt;std::span<const uint8_t>crt_prepared;
    uint8_t*output;size_t row_stride;CrtCarry*row_tails;uint8_t*work;
};
void inverse_emit_tiles(void*ptr,uint64_t lo,uint64_t hi,unsigned rank){
    auto&a=*static_cast<EmitCall*>(ptr);const auto&c=*a.c;const auto&s=c.spec;const auto&i=a.info;
    auto*work=a.work+rank*i.worker_bytes;auto*x=reinterpret_cast<p::V*>(work);
    auto*values=reinterpret_cast<uint64_t*>(work+i.coefficients);auto*out=work+i.packed;
    auto*tails=reinterpret_cast<CrtCarry*>(a.work+i.tails);const size_t B=i.tile_cells,stride=s.outer_cells+1;
    for(size_t tile=lo;tile<hi;++tile){const size_t column=tile*B;
        for(unsigned q=0;q<s.prime_count;++q)for(size_t j=0;j<B;j+=4){
            const size_t columns=std::min(size_t(4),B-j);const auto&pv=c.primes.V[q];
            for(size_t r=0;r<s.outer_cells;++r)for(size_t k=0;k<columns;++k)
                x[k*stride+r]=r<s.live_rows?p::ldslot<48>(a.bank+((q*s.live_rows+r)*s.stripe_cells+column+j+k)*48,c.codec):_mm512_setzero_si512();
            for(size_t k=0;k<columns;++k)p::itft_blk<true>(x+k*stride,s.outer_cells,0,s.live_rows,c.primes.P[q],pv);
            for(size_t r=0;r<s.live_rows;++r)for(size_t k=0;k<columns;++k)
                _mm512_store_si512(values+((q*s.live_rows+r)*B+j+k)*8,x[k*stride+r]);
        }
        for(size_t r=0;r<s.live_rows;++r){
            a.crt->emit(a.crt_prepared,values+r*B*8,s.live_rows*B*8,B,{out,B*s.trunk_bits+256},tails[tile*s.live_rows+r]);
            std::memcpy(a.output+r*a.row_stride+column*s.trunk_bits,out,B*s.trunk_bits);
        }
    }
}
void join_emit_tiles(void*ptr,uint64_t lo,uint64_t hi,unsigned){
    auto&a=*static_cast<EmitCall*>(ptr);const auto&s=a.c->spec;const auto&i=a.info;
    const auto*tails=reinterpret_cast<const CrtCarry*>(a.work+i.tails);const size_t bytes=i.tile_cells*s.trunk_bits;
    for(size_t r=lo;r<hi;++r){CrtCarry carry{};
        for(size_t tile=0;tile<i.tiles;++tile){auto*out=a.output+r*a.row_stride+tile*bytes;uint64_t cy=0;
            for(unsigned k=0;k<8;++k){uint64_t word;std::memcpy(&word,out+k*8,8);
                const __uint128_t sum=__uint128_t(word)+carry.limb[k]+cy;word=uint64_t(sum);cy=uint64_t(sum>>64);std::memcpy(out+k*8,&word,8);}
            for(size_t at=64;cy&&at<bytes;++at){const unsigned sum=out[at]+unsigned(cy);out[at]=uint8_t(sum);cy=sum>>8;}
            carry=tails[tile*s.live_rows+r];
            for(unsigned k=0;cy&&k<8;++k){const __uint128_t sum=__uint128_t(carry.limb[k])+cy;carry.limb[k]=uint64_t(sum);cy=uint64_t(sum>>64);}
            require(!cy,SBN3_FATAL_MATH,"outer tile carry width");
        }
        a.row_tails[r]=carry;
    }
}
}
void p48_outer_inverse_emit(std::span<const uint8_t>b,std::span<const uint8_t>bank,const CrtBackend&crt,
    std::span<const uint8_t>crt_prepared,std::span<uint8_t>output,size_t stride,std::span<CrtCarry>tails,
    std::span<uint8_t>workspace,sbn3_team_scope*scope) noexcept {
    const auto&c=get(b);const auto&s=c.spec;OuterEmitInfo i;require_scope_leader(scope);
    require(p48_outer_emit_query(s,i)&&scope->width>=i.workers&&crt.primes==s.prime_count&&
        bank.data()&&bank.size()>=s.prime_count*s.live_rows*s.stripe_cells*48+64&&
        output.data()&&stride>=s.stripe_cells*s.trunk_bits&&output.size()/stride>=s.live_rows&&
        tails.size()>=s.live_rows&&workspace.data()&&!(uintptr_t(workspace.data())&63)&&workspace.size()>=i.workspace_bytes&&
        !overlaps(bank.data(),bank.size(),workspace.data(),workspace.size())&&
        !overlaps(output.data(),output.size(),workspace.data(),workspace.size())&&
        !overlaps(output.data(),output.size(),bank.data(),bank.size()),SBN3_FATAL_WORKSPACE,"fused outer emission");
    EmitCall a{&c,i,bank.data(),&crt,crt_prepared,output.data(),stride,tails.data(),workspace.data()};
    sbn3_team_scope part{scope->team,scope->first,i.workers,false,scope->epoch};
    sbn3_team_for(&part,0,i.tiles,1,SBN3_STATIC,inverse_emit_tiles,&a);
    sbn3_team_for(&part,0,s.live_rows,1,SBN3_STATIC,join_emit_tiles,&a);
}
} // namespace sbn::v3::product
