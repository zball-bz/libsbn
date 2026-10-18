// Original libsbn code; LGPL-3.0-or-later.
// Integer adaptation of matrix middle products: coefficients stay uncarried
// until both adjacent windows have been joined. Only then is the bounded low
// carry omitted. A guard test certifies the returned integer reduction matrix.
#include "gcd/internal.hpp"
#include "product/matrix_program.hpp"
namespace sbn::v3::gcd {
static size_t digit_sum(U (&sum)[5],Span a,unsigned bits){
    memset(sum,0,sizeof sum);const unsigned words=(bits+63)/64,high=bits-64*(words-1);
    for(size_t bit=0;bit<64*a.n;bit+=bits){
        const size_t at=bit/64;const unsigned shift=bit%64;U carry=0;
        for(unsigned j=0;j<words;++j){
            U digit=at+j<a.n?a.p[at+j]>>shift:0;
            if(shift&&at+j+1<a.n)digit|=a.p[at+j+1]<<(64-shift);
            if(j+1==words&&high<64)digit&=(U(1)<<high)-1;
            const W x=W(sum[j])+digit+carry;sum[j]=U(x);carry=U(x>>64);
        }
        for(unsigned j=words;carry&&j<5;++j){const W x=W(sum[j])+carry;sum[j]=U(x);carry=U(x>>64);}
        require(!carry,SBN3_FATAL_MATH,"middle digit-sum bound");
    }
    size_t n=5;while(n&&!sum[n-1])--n;return n;
}
bool middle_matrix(Pair &p,Matrix &m,Engine &e){
    const size_t n=p.size();if(n<4096||!e.scope)return false;
    const unsigned workers=n>=e.policy.parallel?e.scope->width:1;
    product::MatrixMiddlePlan plan{};if(!product::matrix_middle_query(n,workers,plan))return false;
    const auto &q=plan.matrix.product;const size_t d=plan.half_words,cut=n-2*d;
    // Align the FIRST high reduction to the middle-product grid: its input
    // has 2d limbs, coefficients have <d limbs, and d<=cut<=2d. Rounding
    // the discard UP after an ordinary n/2 split would lose Theta(n) useful
    // bits and force quadratic correction steps at the parent.
    require(d<=cut&&cut<=2*d,SBN3_FATAL_MATH,"middle split coverage");
    const size_t prefix_words=q.info.emit_limbs;
    const size_t need=(2*prefix_words+3*d)*8+q.prepared_bytes+plan.matrix.workspace_bytes+1024;
    if(need>e.space.capacity()-e.space.used())return false;
    auto heads=high(p,cut);
    if(!hgcd(heads,m,e))return false;
    FrameMark lifetime(e.space);
    Pair prefix{allocate(e.space,prefix_words),allocate(e.space,prefix_words)};
    {
        PhaseMeter phase(e.metrics,Phase::middle,n);
        FrameMark episode(e.space);auto *negative=e.space.template alloc<U>(3*d);
        for(size_t j=0;j<3*d;++j)negative[j]=j<cut?~p.b.p[j]:UINT64_MAX;
        auto tables=e.space.subframe(q.prepared_bytes+128,128);const auto program=product_program_prepare(q,tables);
        auto work=e.space.subframe(plan.matrix.workspace_bytes,64);
        MatrixProductInputs in{};in.compact_forward=e.policy.compact_forward;in.middle=true;
        in.left[0][0]={m.e[1][1].p,m.e[1][1].n};in.left[0][1]={m.e[0][1].p,m.e[0][1].n};
        in.left[1][0]={m.e[1][0].p,m.e[1][0].n};in.left[1][1]={m.e[0][0].p,m.e[0][0].n};
        in.right[0][0]={p.a.p,cut};in.right[0][1]={p.a.p+d,cut-d};
        in.right[1][0]={negative,2*d};in.right[1][1]={negative+d,2*d};
        in.out[0][0]={prefix.a.p,prefix.a.cap};in.out[1][0]={prefix.b.p,prefix.b.cap};
        product::matrix_program_execute(program,work,e.scope,in);
    }
    // For the middle range, complementing the right operand adds the
    // constant coefficient (beta-1)*S, S=sum(coefficient digits). Across
    // L=2m joined digits this is S*(beta^L-1). Subtract that bias exactly;
    // the second row is then negated. All operations below use a fixed
    // unsigned width, with ample sign/carry padding, followed by a bound.
    for(unsigned row=0;row<2;++row){
        auto &out=row?prefix.b:prefix.a;const auto &head=row?heads.b:heads.a;
        U sum[5];const size_t sn=digit_sum(sum,m.e[0][row?0:1],plan.digit_bits);
        (void)limbs::add_to(out.p,prefix_words,sum,sn);
        (void)limbs::sub_from(out.p+2*d,prefix_words-2*d,sum,sn);
        if(row){for(size_t j=0;j<prefix_words;++j)out.p[j]=~out.p[j];U one=1;(void)limbs::add_to(out.p,prefix_words,&one,1);}
        require(cut-d+head.n<prefix_words,SBN3_FATAL_MATH,"middle head coverage");
        (void)limbs::add_to(out.p+cut-d,prefix_words-(cut-d),head.p,head.n);
        out.n=prefix_words;out.trim();
        require(out.n+1<prefix_words,SBN3_FATAL_MATH,"middle signed prefix bound");
    }
    if(e.metrics){++e.metrics->middle_windows;e.metrics->products+=8;}
    auto second=make_matrix(e.space,prefix.size());
    Engine guarded{e.space,e.scope,e.policy,e.metrics};
    const size_t error_words=(plan.error_bits+63)/64;
    guarded.policy.stop_guard=1+(error_words+1)/2;
    // Exact reduction of the approximate prefixes. If the interval after
    // applying its inverse straddles zero, keep only the already-certified
    // first matrix. This is a mathematical boundary, not a timing heuristic.
    if(hgcd(prefix,second,guarded)){
        bool safe=true;
        for(unsigned row=0;row<2;++row){
            const size_t coefficient=std::max(second.e[1-row][1-row].n,second.e[row][1-row].n);
            const size_t remaining=row?prefix.b.n:prefix.a.n;
            safe&=remaining>=coefficient+error_words+2;
        }
        if(safe)compose(m,second,e);
        else if(e.metrics)++e.metrics->middle_guard_stops;
    }
    return true;
}

template<class E>static bool recursive_middle_impl(Pair &p,Matrix &m,E &e,ResidualBound &bound){
    const size_t n=p.size();if(n<4096||!e.scope)return false;
    const unsigned workers=n>=e.policy.parallel?e.scope->width:1;
    product::MatrixMiddlePlan plan{};
    // Reserve a few integer guard words at the high end. The actual
    // reconstructed-value bound below, not this guard, admits the window.
    if(!product::matrix_middle_query(n,workers,plan,64,e.policy.legacy_capacity,e.policy.fft_matrix)){
        if(e.metrics)++e.metrics->middle_declined;return false;
    }
    const auto &q=plan.matrix.product;const size_t d=plan.half_words,cut=n-2*d;
    const size_t need=(2*q.info.emit_limbs+3*d)*8+q.prepared_bytes+plan.matrix.workspace_bytes+1024;
    if(d>cut||cut>2*d||need>e.space.capacity()-e.space.used()){
        if(e.metrics)++e.metrics->middle_declined;return false;
    }
    ResidualBound child{};
    {
        FrameMark first(e.space);auto src=high(p,cut);
        Pair heads{allocate(e.space,2*d+3),allocate(e.space,2*d+3)};
        assign(heads.a,src.a);assign(heads.b,src.b);
        if(!hgcd_matrix_only(heads,m,e,&child))return false;
    }
    require(lift_bound(child,m,64*cut,0,bound),SBN3_FATAL_MATH,"middle first-child lift certificate");

    // original=M*residual, with positive entries/residuals. For each
    // column, any nonzero coefficient gives an upper bound on that
    // residual. The two recovered middle blocks cover [d,3d) limbs.
    // Admit only if the true prefix stays below half their modulus and
    // above its possible incoming carry; thus modular lifting is unique.
    size_t upper[2]{SIZE_MAX,SIZE_MAX},coefficient=0;
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
        const auto c=m.e[i][j];coefficient=std::max(coefficient,c.n);
        if(c.n){const size_t ib=bit_length(i?p.b:p.a),cb=bit_length(c);
            require(cb<=ib,SBN3_FATAL_MATH,"middle positive reconstruction bound");
            upper[j]=std::min(upper[j],ib-cb+1);}
    }
    auto decline=[&](){for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)word(m.e[i][j],i==j);return false;};
    const bool fft_balanced=q.info.algorithm==SBN3_MUL_PQ16&&(q.info.codec_mode&8);
    if(coefficient+unsigned(fft_balanced)>d||std::max(upper[0],upper[1])>64*3*d-1||
       bound.lower_bits<64*d+plan.error_bits+2){
        if(e.metrics)++e.metrics->middle_range_stops;return decline();
    }
    // If the first child stopped short by delta words, its matrix has
    // delta fewer digits. Those extra low coefficients of the cyclic
    // product are uncontaminated. Retaining them compensates for the
    // deficient first reduction instead of propagating it up the tree.
    const size_t quantum=8u/(1u<<std::min(3u,unsigned(__builtin_ctz(plan.digit_bits))));
    const size_t origin0=(64*coefficient+unsigned(fft_balanced)*plan.digit_bits+8*plan.digit_bits-1)/(8*plan.digit_bits);
    const size_t origin=(origin0+quantum-1)/quantum*quantum;
    const size_t discard=origin*plan.digit_bits/8,kept=3*d-discard;
    const size_t prefix_words=product::matrix_middle_output_words(plan.matrix,origin);
    plan.matrix.workspace_bytes=product::matrix_program_workspace(plan.matrix,3,origin);
    const size_t basic=(2*prefix_words+3*d)*8+q.prepared_bytes+plan.matrix.workspace_bytes+2048;
    if(basic>e.space.capacity()-e.space.used())return decline();
    const size_t linear_work=product::matrix_program_workspace(plan.matrix,3,SIZE_MAX,true);
    const size_t cached_middle_work=product::matrix_program_workspace(plan.matrix,3,origin,true);
    const size_t cache_bytes=product::matrix_spectrum_bytes(plan.matrix);
    bool cached=e.policy.middle_cache;
    if(cached){
        sbn3_gcd_info child_info{};policy_query(prefix_words,prefix_words,workers,child_info);
        const size_t second_matrix=32*matrix_capacity(prefix_words);
        const size_t peak=cache_bytes+q.prepared_bytes+16*prefix_words+second_matrix+
            std::max({24*d+cached_middle_work,linear_work,child_info.scratch_bytes})+4096;
        cached=peak<=e.space.capacity()-e.space.used();
        if(e.metrics&&!cached)++e.metrics->cache_declined;
    }
    FrameMark lifetime(e.space);
    if(cached)plan.matrix.workspace_bytes=cached_middle_work;
    MatrixSpectrumCache cache{};
    if(cached){cache.bytes=cache_bytes;cache.data=e.space.template alloc<uint8_t>(cache_bytes);}
    // Immutable preparation is retained across the two consumers. This
    // external view does not pin a child Frame while recursive marks rewind
    // later storage. Its underlying range remains owned by this lifetime.
    auto *table_memory=e.space.allocate(q.prepared_bytes+128,128);
    auto tables=Frame::external(table_memory,q.prepared_bytes+128);
    const auto program=product_program_prepare(q,tables);
    Pair prefix{allocate(e.space,prefix_words),allocate(e.space,prefix_words)};
    {
        PhaseMeter phase(e.metrics,Phase::middle,n);
        FrameMark episode(e.space);auto *negative=e.space.template alloc<U>(3*d);
        for(size_t j=0;j<3*d;++j)negative[j]=j<p.b.n?~p.b.p[j]:UINT64_MAX;
        auto work=e.space.subframe(plan.matrix.workspace_bytes,64);
        MatrixProductInputs in{};in.compact_forward=e.policy.compact_forward;in.middle=true;in.middle_first_vector=origin;
        in.left_cache=cached?&cache:nullptr;
        in.left[0][0]={m.e[1][1].p,m.e[1][1].n};in.left[0][1]={m.e[0][1].p,m.e[0][1].n};
        in.left[1][0]={m.e[1][0].p,m.e[1][0].n};in.left[1][1]={m.e[0][0].p,m.e[0][0].n};
        in.right[0][0]={p.a.p,std::min(p.a.n,2*d)};
        in.right[0][1]={p.a.p+d,p.a.n>d?std::min(p.a.n-d,2*d):0};
        in.right[1][0]={negative,2*d};in.right[1][1]={negative+d,2*d};
        in.out[0][0]={prefix.a.p,prefix.a.cap};in.out[1][0]={prefix.b.p,prefix.b.cap};
        product::matrix_program_execute(program,work,e.scope,in);
    }
    // Work modulo B^(3d-discard). Coefficients above the middle window and the
    // high part of the complement bias are multiples of this modulus.
    // There is no exact reduced high child to add back. The positivity
    // and size certificate above resolves the modular representative.
    for(unsigned row=0;row<2;++row){
        auto &out=row?prefix.b:prefix.a;U sum[5];
        const size_t sn=digit_sum(sum,m.e[0][row?0:1],plan.digit_bits);
        (void)limbs::add_to(out.p,kept,sum,sn);
        if(row){for(size_t j=0;j<kept;++j)out.p[j]=~out.p[j];U one=1;(void)limbs::add_to(out.p,kept,&one,1);}
        out.n=kept;out.trim();
        require(out.n,SBN3_FATAL_MATH,"middle certified nonzero prefix");
    }
    if(e.metrics){++e.metrics->middle_windows;++e.metrics->middle_recursive_windows;e.metrics->products+=8;
        ++e.metrics->matrix_basis[q.info.np][q.info.workers];}
    auto second=make_matrix(e.space,prefix.size());
    auto guarded=rebind(e,e.space,e.scope);
    const size_t error_words=(plan.error_bits+63)/64;
    guarded.policy.stop_guard=std::max(e.policy.stop_guard,1+(error_words+1)/2);
    ResidualBound second_bound{},combined{};
    if(hgcd_matrix_only(prefix,second,guarded,&second_bound)){
        if(lift_bound(second_bound,second,64*discard,plan.error_bits,combined)&&
           combined.lower_bits>=64*(n/2+e.policy.stop_guard)){
            bool linear=cached&&linear_work+256<=e.space.capacity()-e.space.used();
            for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)for(unsigned k=0;k<2;++k)
                linear&=m.e[i][k].n+second.e[k][j].n+1<=2*d;
            if(linear){
                PhaseMeter phase(e.metrics,Phase::compose,n/2);
                FrameMark episode(e.space);auto work=e.space.subframe(linear_work,64);
                MatrixProductInputs in{};in.compact_forward=e.policy.compact_forward;in.left_cache=&cache;in.use_cached_left=true;
                for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j){
                    in.left[i][j]={m.e[i][j].p,m.e[i][j].n};
                    in.right[i][j]={second.e[i][j].p,second.e[i][j].n};
                    in.out[i][j]={m.e[i][j].p,m.e[i][j].cap};
                }
                product::matrix_program_execute(program,work,e.scope,in);
                for(auto &row:m.e)for(auto &v:row){v.n=std::min(v.cap,q.info.output_limbs);v.trim();}
                if(e.metrics){++e.metrics->compositions;++e.metrics->matrix_episodes;
                    ++e.metrics->middle_cache_compositions;e.metrics->products+=8;}
            }else compose(m,second,e);
            bound=combined;
        }else {if(e.metrics)++e.metrics->middle_bound_stops;return decline();}
    }
    return true;
}
bool recursive_middle_matrix(Pair&p,Matrix&m,Engine&e,ResidualBound&b){return recursive_middle_impl(p,m,e,b);}
bool recursive_middle_matrix(Pair&p,Matrix&m,SymbolEngine&e,ResidualBound&b){SymbolTransaction tx(e);bool ok=recursive_middle_impl(p,m,e,b);if(ok)tx.commit();return ok;}
}
