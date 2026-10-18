// Original libsbn schedules; LGPL-3.0-or-later.
#include "gcd/internal.hpp"
#include "product/matrix_program.hpp"
#include "product/independent.hpp"
namespace sbn::v3::gcd {
Matrix make_matrix(Frame &f,size_t c0,size_t c1,unsigned rows) {
    Matrix m{};m.rows=rows;
    for(unsigned i=0;i<2;++i)if(rows&(1u<<i))for(unsigned j=0;j<2;++j) {
        m.e[i][j]=allocate(f,i?c1:c0);word(m.e[i][j],i==j);
    }
    return m;
}
void matrix_word(Matrix &m,const gcd_small::Matrix &r,Engine &e) {
    (void)e;
    require(std::max({r.u[0][0],r.u[0][1],r.u[1][0],r.u[1][1]})<(U(1)<<63),SBN3_FATAL_MATH,"HGCD matrix word bound");
    for(unsigned i=0;i<2;++i)if(m.rows&(1u<<i)) {
        auto &a=m.e[i][0],&b=m.e[i][1];const size_t n=std::max(a.n,b.n);
        require(n<a.cap&&n<b.cap,SBN3_FATAL_WORKSPACE,"HGCD matrix word output");
        if(e.policy.word_ifma&&n>=16){
            memset(a.p+a.n,0,(n+1-a.n)*8);memset(b.p+b.n,0,(n+1-b.n)*8);
            if(e.policy.word32)compose_head_word32(a.p,b.p,n+1,r);
            else compose_head(a.p,b.p,n+1,r);a.n=b.n=n+1;a.trim();b.trim();continue;
        }
        W ca=0,cb=0;
        for(size_t j=0;j<n;++j) {
            const U x=j<a.n?a.p[j]:0,y=j<b.n?b.p[j]:0;
            const W u=W(x)*r.u[0][0]+W(y)*r.u[1][0]+ca;
            const W v=W(x)*r.u[0][1]+W(y)*r.u[1][1]+cb;
            a.p[j]=U(u);b.p[j]=U(v);ca=u>>64;cb=v>>64;
        }
        a.n=b.n=n;if(ca)a.p[a.n++]=U(ca);if(cb)b.p[b.n++]=U(cb);a.trim();b.trim();
    }
}
void matrix_shear(Matrix &m,unsigned column,Span q,Engine &e) {
    if(!q.n)return;
    if(q.n==1&&q.p[0]<(U(1)<<63)) {
        gcd_small::Matrix w{{{1,0},{0,1}}};w.u[1-column][column]=q.p[0];matrix_word(m,w,e);return;
    }
    FrameMark mark(e.space);
    size_t cap=0;for(unsigned row=0;row<2;++row)if(m.rows&(1u<<row))cap=std::max(cap,m.e[row][1-column].n+q.n+1);
    auto t=allocate(e.space,cap);
    for(unsigned row=0;row<2;++row)if(m.rows&(1u<<row)) {
        e.multiply(t,m.e[row][1-column],q);add(m.e[row][column],m.e[row][column],t);
    }
}
static bool compose_fft(Matrix &m,const Matrix &r,Engine &e) {
    size_t an=0,bn=0;
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
        if(m.rows&(1u<<i))an=std::max(an,m.e[i][j].n);
        bn=std::max(bn,r.e[i][j].n);
    }
    if(std::min(an,bn)<512||std::max(an,bn)>4*std::min(an,bn))return false;
    product::MatrixProgramPlan plan{};
    // Matrix jobs expose several independent spectra, just like the
    // existing seven-product batch; they are not one scalar multiply.
    const size_t grain=e.policy.fft_matrix?e.policy.batch_min:e.policy.parallel;
    const unsigned workers=an+bn>=grain?e.scope->width:1;
    if(!product::matrix_program_query(an,bn,workers,m.rows,plan,e.policy.fft_matrix))return false;
    const auto &q=plan.product;
    if(!e.policy.matrix_fft&&q.info.algorithm!=SBN3_MUL_PQ16)return false;
    if(q.prepared_bytes+plan.workspace_bytes+512>e.space.capacity()-e.space.used())return false;
    FrameMark mark(e.space);auto tables=e.space.subframe(q.prepared_bytes+128,128);
    const auto program=product_program_prepare(q,tables);auto work=e.space.subframe(plan.workspace_bytes,64);
    MatrixProductInputs input{};input.compact_forward=e.policy.compact_forward;input.rows=m.rows;
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
        input.right[i][j]={r.e[i][j].p,r.e[i][j].n};
        if(m.rows&(1u<<i)){
            input.left[i][j]={m.e[i][j].p,m.e[i][j].n};
            input.out[i][j]={m.e[i][j].p,m.e[i][j].cap};
        }
    }
    product::matrix_program_execute(program,work,e.scope,input);
    for(unsigned i=0;i<2;++i)if(m.rows&(1u<<i))for(unsigned j=0;j<2;++j){
        m.e[i][j].n=std::min(m.e[i][j].cap,q.info.output_limbs);m.e[i][j].trim();
    }
    if(e.metrics){++e.metrics->matrix_episodes;e.metrics->products+=4*__builtin_popcount(m.rows);}
    return true;
}
void compose(Matrix &m,const Matrix &r,Engine &e) {
    PhaseMeter phase(e.metrics,Phase::compose,std::max(m.e[0][0].n,m.e[1][1].n)+std::max(r.e[0][0].n,r.e[1][1].n));
    if(e.metrics)++e.metrics->compositions;
    if((e.policy.matrix_fft||e.policy.fft_matrix)&&compose_fft(m,r,e))return;
    if(e.policy.parallel_seven&&compose_seven_parallel(m,r,e))return;
    if(e.policy.parallel_rows&&m.rows==3&&e.scope&&e.scope->width>1) {
        size_t rn=0,mn=0;for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){rn=std::max(rn,m.e[i][j].n);mn=std::max(mn,r.e[i][j].n);}
        if(rn+mn>=e.policy.parallel&&std::min(rn,mn)>=128) {
            const unsigned left_workers=e.scope->width/2;
            const size_t need=16*(rn+mn+4)+arithmetic_envelope(std::max(rn,mn),e.scope->width-left_workers)+256;
            if(2*need+128<=e.space.capacity()-e.space.used()) {
                FrameMark lifetime(e.space);auto left=e.space.subframe(need),right=e.space.subframe(need);
                struct Task {Matrix matrix;const Matrix *right;Frame *space;Policy policy;Metrics metrics;};
                Task l{m,&r,&left,e.policy,{}},rr{m,&r,&right,e.policy,{}};l.matrix.rows=1;rr.matrix.rows=2;
                auto body=[](void *p,sbn3_team_scope *scope){auto &t=*static_cast<Task*>(p);Engine child{*t.space,scope,t.policy,&t.metrics};compose(t.matrix,*t.right,child);};
                sbn3_team_invoke2(e.scope,SBN3_PARALLEL_CHILDREN,left_workers,body,&l,body,&rr);
                for(unsigned j=0;j<2;++j){m.e[0][j]=l.matrix.e[0][j];m.e[1][j]=rr.matrix.e[1][j];}
                if(e.metrics){e.metrics->products+=l.metrics.products+rr.metrics.products;e.metrics->pairs+=l.metrics.pairs+rr.metrics.pairs;
                    e.metrics->pair_preparations+=l.metrics.pair_preparations+rr.metrics.pair_preparations;}
                return;
            }
        }
    }
    if(e.policy.strassen&&compose_seven(m,r,e))return;
    FrameMark mark(e.space);size_t cap=2;
    for(unsigned i=0;i<2;++i)if(m.rows&(1u<<i))for(unsigned j=0;j<2;++j)for(unsigned k=0;k<2;++k)
        if(m.e[i][k].n&&r.e[k][j].n)cap=std::max(cap,m.e[i][k].n+r.e[k][j].n+1);
    auto t0=allocate(e.space,cap),t1=allocate(e.space,cap);
    for(unsigned i=0;i<2;++i)if(m.rows&(1u<<i)) {
        auto &a=m.e[i][0],&b=m.e[i][1];
        e.multiply(t0,a,r.e[0][0]);e.multiply(t1,b,r.e[1][0]);add(t0,t0,t1);
        e.multiply(t1,a,r.e[0][1]);assign(a,t1);
        e.multiply(t1,b,r.e[1][1]);add(b,a,t1);assign(a,t0);
    }
}
static void combine_high(Nat &a,size_t cut,const Nat &head,Span low) {
    require(cut+head.n<a.cap,SBN3_FATAL_WORKSPACE,"HGCD lifted head capacity");
    const size_t first=std::min(cut,low.n);if(first)memcpy(a.p,low.p,first*8);
    if(first<cut)memset(a.p+first,0,(cut-first)*8);
    a.n=cut+head.n;
    if(low.n>cut) {
        const size_t n=std::max(head.n,low.n-cut);
        require(cut+n<a.cap,SBN3_FATAL_WORKSPACE,"HGCD lifted product capacity");
        if(n>head.n)memset(a.p+cut+head.n,0,(n-head.n)*8);
        const U cy=limbs::add_to(a.p+cut,n,low.p+cut,low.n-cut);
        a.n=cut+n;if(cy)a.p[a.n++]=cy;
    }
    a.trim();
}
void adjust(Pair &p,const Pair &heads,const Matrix &m,size_t cut,Engine &e) {
    PhaseMeter phase(e.metrics,Phase::adjust,p.size());
    FrameMark mark(e.space);size_t mn=0;
    for(const auto &row:m.e)for(const auto &x:row)mn=std::max(mn,x.n);
    auto t0=allocate(e.space,cut+mn+2),t1=allocate(e.space,cut+mn+2);
    const Span a0{p.a.p,cut},b0{p.b.p,cut};
    if(e.policy.parallel_adjust&&cut+mn>=e.policy.batch_min&&e.scope&&e.scope->width>1&&
       16*(cut+mn+2)+256<=e.space.capacity()-e.space.used()){
        FrameMark parallel(e.space);
        // Every source is consumed before either original tail is changed.
        auto t2=allocate(e.space,cut+mn+2),t3=allocate(e.space,cut+mn+2);
        product::IndependentProduct jobs[4]{
            {{a0.p,a0.n},{m.e[1][1].p,m.e[1][1].n},{t0.p,t0.cap}},
            {{b0.p,b0.n},{m.e[0][1].p,m.e[0][1].n},{t1.p,t1.cap}},
            {{b0.p,b0.n},{m.e[0][0].p,m.e[0][0].n},{t2.p,t2.cap}},
            {{a0.p,a0.n},{m.e[1][0].p,m.e[1][0].n},{t3.p,t3.cap}}};
        if(product::independent_products(jobs,4,e.space,e.scope)){
            Nat *out[]{&t0,&t1,&t2,&t3};for(unsigned j=0;j<4;++j){out[j]->n=jobs[j].a.count+jobs[j].b.count;out[j]->trim();}
            combine_high(p.a,cut,heads.a,t0);subtract(p.a,p.a,t1);
            combine_high(p.b,cut,heads.b,t2);subtract(p.b,p.b,t3);
            if(e.metrics){e.metrics->products+=4;++e.metrics->batch_adjustments;}return;
        }
    }
    e.multiply_pair(t0,t1,a0,m.e[1][1],m.e[1][0]);
    combine_high(p.a,cut,heads.a,t0);
    e.multiply(t0,b0,m.e[0][1]);subtract(p.a,p.a,t0);
    e.multiply(t0,b0,m.e[0][0]);combine_high(p.b,cut,heads.b,t0);subtract(p.b,p.b,t1);
}
}
