// Original libsbn code; LGPL-3.0-or-later.
#include "gcd/internal.hpp"
#include "product/cyclic_program.hpp"
#include "product/matrix_program.hpp"
namespace sbn::v3::gcd {
static void fold(Nat &a,size_t period) {
    // All original high words have been consumed by the matrix computation.
    size_t offset=period;
    while(offset<a.n) {
        const size_t count=std::min(period,a.n-offset);
        U carry=limbs::add_to(a.p,period,a.p+offset,count);
        while(carry)carry=limbs::add_to(a.p,period,&carry,1);
        offset+=count;
    }
    if(a.n<period)memset(a.p+a.n,0,(period-a.n)*8);
    bool all_ones=true;for(size_t j=0;j<period;++j)all_ones&=a.p[j]==UINT64_MAX;
    if(all_ones)memset(a.p,0,period*8);a.n=period;
}
static void modular_subtract(Nat &a,const Nat &b,size_t period) {
    U borrow=sbn3i_sub_n(a.p,a.p,b.p,long(period));
    // Integer underflow added B^r; the modulus adds only B^r-1.
    if(borrow){U one=1;require(!limbs::sub_from(a.p,period,&one,1),SBN3_FATAL_MATH,"cyclic subtraction wrap");}
    bool all_ones=true;for(size_t j=0;j<period;++j)all_ones&=a.p[j]==UINT64_MAX;
    if(all_ones)memset(a.p,0,period*8);a.n=period;a.trim();
}
static bool joint_vector_apply(Pair &p,const Matrix &m,Engine&e,size_t minimum,size_t mn,const size_t *bounds){
    size_t period=0;
    product::MatrixProgramPlan plan{};auto &q=plan.product;
    const unsigned workers=p.size()>=e.policy.parallel?e.scope->width:1;
    if(!product::matrix_vector_query(minimum,p.size()-1,mn,workers,e.policy.fft_vector,e.policy.joint_vector,plan,period)||
       period>std::min(p.a.cap,p.b.cap))return false;
    const size_t need=16*(period+2)+q.prepared_bytes+plan.workspace_bytes+512;
    if(need>e.space.capacity()-e.space.used())return false;
    FrameMark mark(e.space);auto u=allocate(e.space,period+2),v=allocate(e.space,period+2);
    auto tables=e.space.subframe(q.prepared_bytes+128,128);const auto program=product_program_prepare(q,tables);
    auto work=e.space.subframe(plan.workspace_bytes,64);
    fold(p.a,period);fold(p.b,period);
    MatrixProductInputs in{};in.compact_forward=e.policy.compact_forward;in.vector=true;in.negative_left=6;
    in.left[0][0]={m.e[1][1].p,m.e[1][1].n};in.left[0][1]={m.e[0][1].p,m.e[0][1].n};
    in.left[1][0]={m.e[1][0].p,m.e[1][0].n};in.left[1][1]={m.e[0][0].p,m.e[0][0].n};
    in.right[0][0]={p.a.p,period};in.right[1][0]={p.b.p,period};
    in.out[0][0]={u.p,period};in.out[1][0]={v.p,period};
    product::matrix_program_execute(program,work,e.scope,in);u.n=v.n=period;u.trim();v.trim();
    require(u.n<=bounds[0]&&v.n<=bounds[1],SBN3_FATAL_MATH,"joint cyclic residual bounds");
    assign(p.a,u);assign(p.b,v);if(e.metrics){++e.metrics->cyclic;++e.metrics->cyclic_sums;e.metrics->products+=4;}
    return true;
}
bool cyclic_apply(Pair &p,const Matrix &m,Engine &e) {
    PhaseMeter phase(e.metrics,Phase::cyclic,p.size());
    if(!m.e[0][1].n||!m.e[1][0].n)return false;
    const size_t n=p.size();size_t bounds[2]{SIZE_MAX,SIZE_MAX},mn=0;
    for(unsigned row=0;row<2;++row)for(unsigned col=0;col<2;++col) {
        const size_t words=m.e[row][col].n;
        mn=std::max(mn,words);const size_t input=row?p.b.n:p.a.n;
        if(words){require(words<=input,SBN3_FATAL_MATH,"HGCD coefficient size bound");bounds[col]=std::min(bounds[col],input-words+1);}
    }
    const size_t minimum=std::max(bounds[0],bounds[1])+1;
    if(minimum>=n)return false;
    if((e.policy.fft_vector||e.policy.joint_vector)&&joint_vector_apply(p,m,e,minimum,mn,bounds))return true;
    product::CyclicProgramPlan plan{};
    const unsigned workers=n>=e.policy.parallel?e.scope->width:1;
    auto admit=[&](unsigned width){
        bool ok=product::cyclic_program_query(minimum,mn,width,plan,e.policy.cyclic_sum,e.policy.legacy_capacity);
        if(!ok&&e.policy.cyclic_sum)ok=product::cyclic_program_query(minimum,mn,width,plan);
        if(!ok||plan.period>=n||plan.period>std::min(p.a.cap,p.b.cap))return false;
        const auto &q=plan.product;
        const size_t need=16*(plan.period+2)+8*(1+unsigned(plan.sum2))*(mn+1)+q.prepared_bytes+q.info.workspace_bytes+q.info.workspace_alignment+512;
        return need<=e.space.capacity()-e.space.used();
    };
    // Emission scratch can dominate a small parallel ring. A serial episode
    // retains the cyclic reduction without forcing four longer linear
    // products when the requested worker width does not fit the arena.
    if(!admit(workers)&&(workers==1||!admit(1)))return false;
    const auto &q=plan.product;const size_t period=plan.period;
    FrameMark mark(e.space);auto t0=allocate(e.space,period+2),t1=allocate(e.space,period+2),coef=allocate(e.space,mn+1);
    auto tables=e.space.subframe(q.prepared_bytes+128,128);const auto program=product_program_prepare(q,tables);
    auto work=e.space.subframe(q.info.workspace_bytes,q.info.workspace_alignment);
    fold(p.a,period);fold(p.b,period);
    if(plan.sum2) {
        auto coef1=allocate(e.space,mn+1);
        auto complement=[&](Nat &a){for(size_t j=0;j<period;++j)a.p[j]=~a.p[j];};
        auto sum=[&](Nat &out,Span a,Span b,Span c,Span d){
            if(b.n)memcpy(coef.p,b.p,b.n*8);if(b.n<mn)memset(coef.p+b.n,0,(mn-b.n)*8);
            if(d.n)memcpy(coef1.p,d.p,d.n*8);if(d.n<mn)memset(coef1.p+d.n,0,(mn-d.n)*8);
            product_program_sum(program,work,e.scope,{{a.p,period},{coef.p,mn},{c.p,period},{coef1.p,mn}},{out.p,out.cap});out.n=period;out.trim();
            if(e.metrics)e.metrics->products+=2;
        };
        // P-B is the limbwise complement in the minus ring. Both terms of
        // the MAC remain positive, so existing unsigned CRT bounds apply;
        // no unproved negative-coefficient CRT interpretation is introduced.
        complement(p.b);sum(t0,p.a,m.e[1][1],p.b,m.e[0][1]);complement(p.b);
        complement(p.a);sum(t1,p.b,m.e[0][0],p.a,m.e[1][0]);
        require(t0.n<=bounds[0]&&t1.n<=bounds[1],SBN3_FATAL_MATH,"HGCD cyclic MAC bounds");
        assign(p.a,t0);assign(p.b,t1);if(e.metrics){++e.metrics->cyclic;++e.metrics->cyclic_sums;}return true;
    }
    auto multiply=[&](Nat &out,Span a,Span b){
        if(b.n)memcpy(coef.p,b.p,b.n*8);if(b.n<mn)memset(coef.p+b.n,0,(mn-b.n)*8);
        product_program_execute(program,work,e.scope,{a.p,period},{coef.p,mn},{out.p,out.cap});
        out.n=period;if(e.metrics)++e.metrics->products;
    };
    multiply(t0,p.a,m.e[1][1]);multiply(t1,p.b,m.e[0][1]);modular_subtract(t0,t1,period);
    require(t0.n<=bounds[0],SBN3_FATAL_MATH,"HGCD cyclic A bound");
    multiply(t1,p.a,m.e[1][0]);assign(p.a,t0);
    multiply(t0,p.b,m.e[0][0]);modular_subtract(t0,t1,period);
    require(t0.n<=bounds[1],SBN3_FATAL_MATH,"HGCD cyclic B bound");assign(p.b,t0);
    if(e.metrics)++e.metrics->cyclic;return true;
}
void apply_matrix(Pair &p,const Matrix &m,Engine &e){
    FrameMark lifetime(e.space);size_t mn=0;
    for(const auto &row:m.e)for(const auto &v:row)mn=std::max(mn,v.n);
    const size_t n=p.size();
    if(mn==1&&std::min(p.a.cap,p.b.cap)>=n){
        gcd_small::Matrix words{};U mask=0;
        for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
            const auto &v=m.e[i][j];words.u[i][j]=v.n?v.p[0]:0;mask|=words.u[i][j];
        }
        if(!(mask>>63)){
            // The same exact inverse word-matrix kernel used by a Lehmer
            // leaf. A short reduction does not require four full products.
            memset(p.a.p+p.a.n,0,(n-p.a.n)*8);memset(p.b.p+p.b.n,0,(n-p.b.n)*8);
            if(e.policy.word32)apply_head_word32(p.a.p,p.b.p,n,words);
            else apply_head(p.a.p,p.b.p,n,words);
            p.a.n=p.b.n=n;p.a.trim();p.b.trim();return;
        }
    }
    auto x=allocate(e.space,p.size()+mn+2),y=allocate(e.space,p.size()+mn+2);
    e.multiply(x,p.a,m.e[1][1]);e.multiply(y,p.b,m.e[0][1]);subtract(x,x,y);
    e.multiply(y,p.a,m.e[1][0]);assign(p.a,x);
    e.multiply(x,p.b,m.e[0][0]);subtract(x,x,y);assign(p.b,x);
}
template<class E>static bool reduce_high_impl(Pair &p,size_t cut,Matrix &m,E &e) {
    auto source=high(p,cut);
    // These are the reducer's existing stop/equality conditions. Test the
    // read-only views before copying prefixes or querying a middle product.
    if(std::min(source.a.n,source.b.n)<=source.size()/2+e.policy.stop_guard||
       !compare(source.a,source.b))return false;
    // Full cyclic application needs original operands, so the optional
    // schedule computes the high reduction on independent, bounded copies.
    const bool matrix_only=e.policy.projected||(e.policy.middle_recursive&&p.size()-cut>=4096);
    if((e.policy.cyclic||matrix_only)&&p.size()>=4096) {
        bool ok=false,used=false;
        // Matrix storage belongs to the caller and survives head reclamation.
        {
            FrameMark mark(e.space);
            const size_t capacity=source.size()+3;
            Pair h{allocate(e.space,capacity),allocate(e.space,capacity)};
            assign(h.a,source.a);assign(h.b,source.b);
            ok=matrix_only?hgcd_matrix_only(h,m,e):hgcd(h,m,e);
            if(!ok)return false;
            if(!matrix_only){
                used=cyclic_apply(p,m,e);
                if(!used){
                    if(h.a.n)memcpy(p.a.p+cut,h.a.p,h.a.n*8);
                    if(h.b.n)memcpy(p.b.p+cut,h.b.p,h.b.n*8);
                    Pair heads{{p.a.p+cut,h.a.n,p.a.cap-cut},{p.b.p+cut,h.b.n,p.b.cap-cut}};
                    adjust(p,heads,m,cut,e);
                }
            }
        }
        // Matrix-only scratch has no remaining value consumer. Reclaim it
        // before the exact application, so it cannot force a wider budget
        // or a slower product just by overlapping the next phase.
        if(matrix_only&&!cyclic_apply(p,m,e))apply_matrix(p,m,e);
        if(matrix_only&&e.metrics)++e.metrics->projected;
        return ok;
    }
    if(!hgcd(source,m,e))return false;adjust(p,source,m,cut,e);return true;
}
bool reduce_high(Pair&p,size_t c,Matrix&m,Engine&e){return reduce_high_impl(p,c,m,e);}
bool reduce_high(Pair&p,size_t c,Matrix&m,SymbolEngine&e){return reduce_high_impl(p,c,m,e);}
}
