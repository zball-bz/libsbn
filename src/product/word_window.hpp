#pragma once
#include "product/local_product.hpp"
#include "product/program.hpp"
#include "value/limbs.hpp"
#include "sbn3/base.h"
#include "tuning/native_policy.hpp"

namespace sbn::v3::product {
// Exact 64-bit window of A*B. Four equal prefix blocks give at most ten
// multiplications; a shorter operand caps the block width, giving the usual
// rectangular O(n/b*M(b)) strip product instead. Completed low blocks are
// discarded after their carry enters the next block. Space is one small
// product episode; work remains O(M(n)), including the exceptional case.
struct WordWindowPlan {
    size_t block=0,output_words=0,table_bytes=0,work_bytes=0,work_alignment=64,bytes=0;
    unsigned workers=1;
    ProductChoice choice=0;
    bool local=false,scalar=false;
    LocalProductPlan kernel{};
};
inline size_t word_window_align(size_t n) noexcept {return (n+63)&~size_t(63);}
inline sbn3_query_result word_window_query(size_t an,size_t bn,uint64_t bit,unsigned workers,
                                          WordWindowPlan &p,ProductChoice chosen=0) noexcept {
    p={};p.workers=workers;
    if(!an||!bn||bit/64>=an+bn)return SBN3_SUPPORTED;
    const size_t end=size_t(bit/64)+1+unsigned(bit%64!=0);
    p.block=std::min({(end+3)/4,an,bn});
    p.output_words=2*p.block;
    p.scalar=p.block<=8;
    // A small exceptional carry witness uses the same serial local kernel
    // regardless of the enclosing team's width. Reserving a parallel NTT
    // here would make a rarely taken branch dominate the whole service's RAM.
    p.local=!p.scalar&&2*p.block<=native_policy::window_inline_words;
    if(p.local)p.workers=1;
    if(p.local){p.kernel=local_product_query(p.block,p.block,false,false);
        p.table_bytes=p.kernel.persistent_bytes+128;p.work_bytes=p.kernel.work_bytes;}
    else if(!p.scalar){
        ProductProgramPlan q{};sbn3_mul_options o{};o.workers=workers;
        const auto rc=chosen?product_program_chosen(p.block,p.block,o,chosen,q):
            product_program_choose(p.block,p.block,o,q,chosen);
        if(rc!=SBN3_SUPPORTED)return rc;
        p.choice=chosen;p.output_words=sbn3_mul_output_capacity(&q.info);
        p.table_bytes=(q.prepared_bytes+127)&~size_t(127);p.table_bytes+=256;
        p.work_bytes=q.info.workspace_bytes;p.work_alignment=std::max<size_t>(64,q.info.workspace_alignment);
    }
    p.bytes=word_window_align((2*p.block+1)*8)+2*word_window_align(p.block*8)+
        word_window_align(p.output_words*8)+p.table_bytes+p.work_alignment+p.work_bytes+256;
    return SBN3_SUPPORTED;
}
inline uint64_t word_window_execute(const WordWindowPlan &p,Span a,Span b,uint64_t bit,
                                    Frame &space,sbn3_team *team,uint64_t a_bits=UINT64_MAX,
                                    sbn3_team_scope *scope=nullptr) noexcept {
    if(!p.block)return 0;
    FrameMark lifetime(space);
    const size_t block=p.block,word=size_t(bit/64),last=word+unsigned(bit%64!=0);
    auto *acc=space.alloc_zero<uint64_t>(2*block+1),*x=space.alloc<uint64_t>(block),
         *y=space.alloc<uint64_t>(block),*z=space.alloc<uint64_t>(p.output_words);
    auto tables=space.subframe(p.table_bytes,128);
    auto work=space.subframe(p.work_bytes,p.work_alignment);
    PreparedLocalProduct local{};ProductProgram program{};
    if(p.local)local=local_product_prepare(p.kernel,x,tables);
    else if(!p.scalar){
        ProductProgramPlan q{};sbn3_mul_options o{};o.workers=p.workers;
        require(product_program_chosen(block,block,o,p.choice,q)==SBN3_SUPPORTED&&
                q.info.workspace_bytes==p.work_bytes&&sbn3_mul_output_capacity(&q.info)==p.output_words,
                SBN3_FATAL_MATH,"word-window product replay");
        program=product_program_prepare(q,tables);
    }
    auto copy=[&](uint64_t *out,Span value,size_t start,uint64_t bits){
        size_t count=start<value.words?std::min(block,value.words-start):0;
        if(bits!=UINT64_MAX){const size_t limit=size_t((bits+63)/64);count=start<limit?std::min(count,limit-start):0;}
        if(count)memcpy(out,value.data+start,count*8);
        memset(out+count,0,(block-count)*8);
        if(bits!=UINT64_MAX&&bits%64&&bits/64>=start&&bits/64-start<block)
            out[bits/64-start]&=(uint64_t(1)<<(bits%64))-1;
    };
    uint64_t lo=0,hi=0;
    for(size_t column=0;column<=last/block;++column){
        const size_t ablocks=(a.words+block-1)/block,bblocks=(b.words+block-1)/block;
        const size_t first=column>=bblocks?column-bblocks+1:0,stop=std::min(column+1,ablocks);
        for(size_t i=first;i<stop;++i){const size_t j=column-i;
            copy(x,a,i*block,a_bits);copy(y,b,j*block,UINT64_MAX);
            if(p.scalar)sbn3_mul_basecase(z,2*block,x,block,y,block);
            else if(p.local)local_product_apply(p.kernel,local,z,y,block,work);
            else if(scope)product_program_execute(program,work,scope,{x,block},{y,block},{z,p.output_words});
            else{
                struct Job {const ProductProgram *program;Frame *work;uint64_t *a,*b,*out;size_t block,words;};
                Job job{&program,&work,x,y,z,block,p.output_words};
                sbn3_team_run(team,[](void *ptr,sbn3_team_scope *scope){auto &j=*static_cast<Job *>(ptr);
                    product_program_execute(*j.program,*j.work,scope,{j.a,j.block},{j.b,j.block},{j.out,j.words});},&job);
            }
            require(!limbs::add_to(acc,2*block+1,z,2*block),SBN3_FATAL_MATH,"word-window carry bound");
        }
        const size_t start=column*block;
        if(word>=start&&word-start<block)lo=acc[word-start];
        if(last>=start&&last-start<block)hi=acc[last-start];
        memmove(acc,acc+block,(block+1)*8);memset(acc+block+1,0,block*8);
    }
    const unsigned shift=unsigned(bit%64);
    return shift?(lo>>shift)|(hi<<(64-shift)):lo;
}
}
