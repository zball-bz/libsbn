// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/carry_output.hpp"
#include "external/integer_emit.hpp"
#include "runtime/team.hpp"
#include "product/crt.hpp"
#include "product/modular_row.hpp"
#include "common/m61.hpp"
using namespace sbn::v3::product;
static void crt(unsigned np,unsigned T,size_t cells,bool weighted=false){
    const auto *backend=p48_crt_backend(np);assert(backend&&backend->prepared_bytes(T));
    Aligned prepared(backend->prepared_bytes(T)),out(cells*T+256);
    std::vector<uint64_t>residues(np*cells*8);
    ref_int P,x,y,sum,actual,tail_ref;ref_inits(P,x,y,sum,actual,tail_ref,nullptr);ref_set_ui(P,1);ref_set_ui(sum,0);
    for(unsigned q=0;q<np;++q)ref_mul_ui(P,P,p48_row_prime(q));
    for(size_t j=0;j<cells*8;++j){
        uint64_t words[8];for(auto &word:words)word=random_word();ref_import(x,8,-1,8,0,0,words);ref_mod(x,x,P);
        if(j%9==0)ref_sub_ui(x,P,1);if(j%9==1)ref_set_ui(x,0);
        for(unsigned q=0;q<np;++q){const uint64_t p=p48_row_prime(q);uint64_t v=ref_fdiv_ui(x,p);
            if(weighted)v=uint64_t(__uint128_t(v)*backend->input_scale(q)%p);residues[q*cells*8+j]=v+(j%4)*p;}
        ref_mul_2exp(y,x,j*T);ref_add(sum,sum,y);
    }
    CrtCarry tail{};allocation_watch_start();backend->prepare_scaled(T,weighted,prepared.span());
    backend->emit(prepared.span(),residues.data(),cells*8,cells,out.span(),tail);assert(!allocation_watch_stop());
    ref_import(actual,cells*T,-1,1,0,0,out.data);ref_import(tail_ref,8,-1,8,0,0,tail.limb);
    ref_mul_2exp(tail_ref,tail_ref,cells*T*8);ref_add(actual,actual,tail_ref);assert(ref_cmp(sum,actual)==0);
    ref_clears(P,x,y,sum,actual,tail_ref,nullptr);
    printf("External vector CRT NP%u T%u cells%zu weighted%d: exact normalized stream and carry PASS\n",np,T,cells,weighted);
}
static void carry(size_t length,size_t count,size_t output,bool runs,size_t first=0,bool cyclic=false,bool zero_ring=false,size_t stream=0){
    CarryOutputPlan p{};assert(query_carry_output(count,length,output,p,cyclic,stream));
    if(stream)assert(p.fragment_bytes<=((count/stream)+1)*chunk_bytes);
    const size_t stored=output-first;
    auto r=fresh(stored);Device disk(size_t(r.layout.allocation_bytes));auto t=disk.transport();
    Aligned io_data(4*chunk_bytes),page(index_page_bytes),raw(length),fragments(p.fragment_bytes),assembly(chunk_bytes),result(stored);
    ChunkWriterStorage cw{};assert(!chunk_writer_storage(r.layout,cw));Aligned image(cw.index_bytes),bits(cw.coverage_bytes);
    ChunkWriter writer;assert(!writer.open(t,r,{io_data.span(),page.span()},image.span(),bits.span()));
    assert(!writer.project(output,first));
    auto segments=std::make_unique<CarrySegment[]>(count);CarryOutput emit;
    assert(!emit.open(p,writer,{segments.get(),count},fragments.span(),assembly.span()));
    ref_int sum,value,tail_value,shifted,got,modulus;ref_inits(sum,value,tail_value,shifted,got,modulus,nullptr);ref_set_ui(sum,0);
    for(size_t arrival=0;arrival<count;++arrival){
        const size_t sid=stream?(count/stream-1-arrival%(count/stream))*stream+arrival/(count/stream):count-1-arrival;
        fill(raw.span());for(size_t j=0;j<length;++j)raw.data[j]^=uint8_t(31*sid);
        if(runs&&sid==1)std::memset(raw.data,255,length);
        if(runs&&sid==2){std::memset(raw.data,255,std::min(length,size_t(2*chunk_bytes+123)));if(length>2*chunk_bytes+123)raw.data[2*chunk_bytes+123]=37;}
        if(stream&&runs&&sid%stream<2)std::memset(raw.data,255,length);
        if(cyclic&&(runs||zero_ring))std::memset(raw.data,255,length);
        const size_t begin=sid*length;
        if(begin+length>output)std::memset(raw.data+std::min(length,output>begin?output-begin:0),0,length-std::min(length,output>begin?output-begin:0));
        CrtCarry tail{};tail.limb[0]=sid+1<count?sid+1:0;
        if(count==1&&output>length)tail.limb[0]=0x123456789aULL;
        if(cyclic)tail.limb[0]=zero_ring?0:sid+1;
        ref_import(value,length,-1,1,0,0,raw.data);ref_import(tail_value,8,-1,8,0,0,tail.limb);
        ref_mul_2exp(tail_value,tail_value,length*8);ref_add(value,value,tail_value);
        ref_mul_2exp(shifted,value,begin*8);ref_add(sum,sum,shifted);
        allocation_watch_start();assert(!emit.submit(sid,raw.span(),tail));assert(!allocation_watch_stop());
    }
    allocation_watch_start();assert(!emit.finish());assert(!writer.seal());assert(!allocation_watch_stop());
    if(cyclic){ref_set_ui(modulus,1);ref_mul_2exp(modulus,modulus,8*output);ref_sub_ui(modulus,modulus,1);
        ref_mod(sum,sum,modulus);if(zero_ring){assert(ref_cmp_ui(sum,0)==0);ref_set(sum,modulus);}}
    assert(t.counters.read_submitted==0&&t.counters.write_submitted==r.layout.data_bytes+r.layout.index_bytes);
    assert(emit.residue()==ref_fdiv_ui(sum,m61::modulus));
    ValueReader reader;assert(!reader.open(t,r,{io_data.span(),page.span()}));assert(!reader.read(0,result.span()));
    ref_import(got,stored,-1,1,0,0,result.data);ref_fdiv_q_2exp(value,sum,first*8);assert(ref_cmp(got,value)==0);
    if(length%chunk_bytes==0)assert(p.fragment_bytes<2*count*chunk_bytes);
    ref_clears(sum,value,tail_value,shifted,got,modulus,nullptr);
    printf("Carry output segments=%zu bytes=%zu output=%zu stored=%zu retained=%zu stream=%zu: interleaved/reverse arrival, symbolic runs, write once PASS\n",count,length,output,stored,p.fragment_bytes,stream);
}
static void faults(){
    auto r=fresh(4*chunk_bytes);Device disk(size_t(r.layout.allocation_bytes));auto t=disk.transport();
    CarryOutputPlan p{};assert(query_carry_output(2,2*chunk_bytes,4*chunk_bytes,p));
    Aligned io_data(chunk_bytes),page(index_page_bytes),raw(2*chunk_bytes),frags(p.fragment_bytes),assembly(chunk_bytes);fill(raw.span());
    ChunkWriterStorage q{};assert(!chunk_writer_storage(r.layout,q));Aligned image(q.index_bytes),bits(q.coverage_bytes);
    ChunkWriter writer;assert(!writer.open(t,r,{io_data.span(),page.span()},image.span(),bits.span()));
    CarrySegment segments[2];CarryOutput output;assert(!output.open(p,writer,segments,frags.span(),assembly.span()));
    disk.fail_write=1;assert(output.submit(1,raw.span(),{}).code==IoCode::io);assert(r.state==RegionState::invalid);assert(output.finish().code==IoCode::io);
}
static void projection_faults(){
    for(unsigned mode=0;mode<3;++mode){
        auto r=fresh(2*chunk_bytes);Device disk(size_t(r.layout.allocation_bytes));auto t=disk.transport();
        Aligned data(chunk_bytes),page(index_page_bytes),source(chunk_bytes);fill(source.span());
        ChunkWriterStorage q{};assert(!chunk_writer_storage(r.layout,q));Aligned image(q.index_bytes),bits(q.coverage_bytes);
        ChunkWriter w;assert(!w.open(t,r,{data.span(),page.span()},image.span(),bits.span()));
        assert(w.project(4*chunk_bytes,1).code==IoCode::invalid);assert(!w.project(4*chunk_bytes,2*chunk_bytes));
        if(mode==0)disk.cancelled=true;
        const auto skipped=w.write(0,source.span());
        assert(t.counters.write_submitted==0);
        if(mode==0){assert(skipped.code==IoCode::cancelled&&r.state==RegionState::invalid);continue;}
        assert(!skipped);assert(w.project(4*chunk_bytes,chunk_bytes).code==IoCode::invalid);
        if(mode==1){assert(w.seal().code==IoCode::incomplete&&r.state==RegionState::invalid);continue;}
        disk.fail_write=1;assert(w.write(2,source.span()).code==IoCode::io&&r.state==RegionState::invalid);
    }
    puts("Projected chunks: immutable mapping after dispatch, kept coverage, cancellation and IO failures PASS");
}
static void streamed_rows(unsigned workers,unsigned fault=0){
    constexpr size_t rows=7,inner=512,stripe=256,primes=6,T=128,digits=rows*inner*8;
    IntegerEmitPlan plan;assert(query_integer_emit(primes,T,rows,inner,stripe,rows*inner*T+64,workers,plan,{},0,true));plan.scalar=3;
    auto serial=plan;serial.pipeline=false;assert(plan.pipeline&&serial.storage_bytes==plan.storage_bytes);
    std::vector<uint64_t>coeff(primes*digits),expected(digits*2+8);uint64_t carry=0;
    for(size_t i=0;i<digits;++i){const uint64_t lo=i%17<9?UINT64_MAX:random_word(),hi=i%17<9?UINT64_MAX:random_word(),extra=random_word()&0xffffffff;
        __uint128_t v=__uint128_t(lo)+carry;expected[2*i]=uint64_t(v);v=__uint128_t(hi)+uint64_t(v>>64);expected[2*i+1]=uint64_t(v);carry=extra+uint64_t(v>>64);
        for(unsigned q=0;q<primes;++q){const uint64_t p=p48_row_prime(q);uint64_t x=uint64_t((__uint128_t(extra)<<64|hi)%p);
            x=uint64_t((__uint128_t(x)<<64|lo)%p);coeff[q*digits+i]=x+(i%4)*p;}}
    expected[digits*2]=carry;carry=0;
    for(auto&word:expected){const __uint128_t x=__uint128_t(word)*3+carry;word=uint64_t(x);carry=uint64_t(x>>64);}assert(!carry);
    const auto residue=m61::words(expected.data(),expected.size());Fixture f(workers,false,64u<<20);
    auto grant=f.allocate(plan.storage_bytes,4096);auto work=std::span(static_cast<uint8_t *>(grant.data),grant.bytes);
    std::vector<uint64_t>bank(primes*rows*stripe*8);uint64_t writes=0;
    for(unsigned variant=0;variant<2;++variant){
        auto r=fresh(expected.size()*8);Device disk(size_t(r.layout.allocation_bytes));auto t=disk.transport();
        Aligned io(4*chunk_bytes),page(index_page_bytes);StoreBuffers buffers{io.span(),page.span()};
        ChunkWriterStorage q;assert(!chunk_writer_storage(r.layout,q));Aligned index(q.index_bytes),bits(q.coverage_bytes),result(expected.size()*8);
        ChunkWriter writer;assert(!writer.open(t,r,buffers,index.span(),bits.span()));IntegerEmitBinding binding;
        assert(!prepare_integer_emit(variant?plan:serial,work,writer,*f.team,binding));
        if(variant&&fault==1)disk.fail_write=2;if(variant&&fault==2)disk.cancel_after_write=1;
        IoError error{};
        for(size_t column=0;column<inner;column+=stripe){
            for(unsigned p=0;p<primes;++p)for(size_t row=0;row<rows;++row)
                std::copy_n(coeff.data()+p*digits+(row*inner+column)*8,stripe*8,bank.data()+(p*rows+row)*stripe*8);
            CoefficientStripe view{bank,primes,rows,stripe,column};allocation_watch_start();error=emit_integer_stripe(binding,view);assert(!allocation_watch_stop());if(error)break;
        }
        if(variant&&fault){assert(error&&!f.team->busy&&r.state==RegionState::invalid);writer.abandon();continue;}
        assert(!error);allocation_watch_start();assert(!finish_integer_emit(binding,residue));assert(!allocation_watch_stop());
        if(!variant)writes=t.counters.write_submitted;else assert(t.counters.write_submitted==writes);
        assert(!t.counters.read_submitted);ValueReader reader;assert(!reader.open(t,r,buffers));assert(!reader.read(0,result.span()));
        assert(!memcmp(result.data,expected.data(),result.bytes));
    }
    printf("Streamed CRT W%u fault%u: independent carry model, scaling, exact IO/scratch and joined failure PASS\n",workers,fault);
}
int main(){
    for(unsigned np=4;np<=10;++np){const unsigned T=np==4?80:24*np-16;crt(np,T,1);crt(np,T,13);crt(np,T,13,true);}
    crt(4,72,5);crt(10,232,7);crt(10,232,7,true);
    carry(80,5,5*80+64,false);carry(5*chunk_bytes,4,20*chunk_bytes,true);
    carry(chunk_bytes+80,3,3*(chunk_bytes+80)+64,true);carry(3*chunk_bytes,4,9*chunk_bytes+104,false);carry(80,1,88,false);
    carry(5*chunk_bytes,4,20*chunk_bytes,true,6*chunk_bytes);
    carry(chunk_bytes+80,3,3*(chunk_bytes+80)+64,true,chunk_bytes);
    carry(5*chunk_bytes,4,20*chunk_bytes,true,20*chunk_bytes);
    carry(80,1,80,true,0,true); // fixed point needs the third descriptor traversal
    carry(80,1,80,true,0,true,true); // redundant all-ones representation of zero
    carry(80,5,400,true,0,true);carry(chunk_bytes+80,3,3*(chunk_bytes+80),false,0,true);
    carry(5*chunk_bytes,4,20*chunk_bytes,true,6*chunk_bytes,true);
    carry(3*chunk_bytes,15,45*chunk_bytes+64,false,0,false,false,5);
    carry(3*chunk_bytes,15,45*chunk_bytes+64,true,6*chunk_bytes,false,false,5);
    carry(3*chunk_bytes,15,43*chunk_bytes+104,false,0,false,false,5);
    carry(3*chunk_bytes,15,45*chunk_bytes,false,0,true,false,5);
    carry(3*chunk_bytes,15,45*chunk_bytes,true,0,true,false,5);
    carry(3*chunk_bytes,15,45*chunk_bytes,true,0,true,true,5);
    faults();projection_faults();streamed_rows(2);streamed_rows(4);streamed_rows(8);streamed_rows(4,1);streamed_rows(4,2);
    puts("OOC CRT/carry: vector reconstruction, arbitrary segment arrival, bounded fragments, m61 and IO failure PASS");
}
