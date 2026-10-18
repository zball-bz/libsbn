// Original libsbn schedules; LGPL-3.0-or-later.
#include "gcd/internal.hpp"
#include "product/matrix_program.hpp"
#include <cstddef>
#include <new>
namespace sbn::v3::gcd {
// This is the existing eager suffix's complete bound, excluding the two
// operand buffers which the caller already owns. A deferred child is entered
// only with this much space still available. Otherwise the eager continuation
// uses the same matrix and values, without retrying or changing storage.
size_t deferred_suffix_bytes(size_t n,unsigned workers,unsigned rows){
    return 16*unsigned(__builtin_popcount(rows))*round_words(n+3)+8*(4*n+4096)+arithmetic_envelope(n,workers)+256;
}
static size_t packed_bytes(const Matrix&m){
    size_t words=0;for(const auto&row:m.e)for(const auto&v:row)words+=round_words(v.n+1);return words*8;
}
static void pack_matrix(Matrix&m,Frame::Mark mark,Engine&e){
    U *const base=m.e[0][0].p;size_t words=0;
    for(unsigned row=0;row<2;++row)if(m.rows&(1u<<row))for(auto&v:m.e[row]){const size_t cap=round_words(v.n+1);U *dest=base+words;
        require(dest<=v.p,SBN3_FATAL_LIFETIME,"GCD history pack order");
        if(v.n)memmove(dest,v.p,v.n*8);v.p=dest;v.cap=cap;words+=cap;
    }
    for(unsigned row=0;row<2;++row)if(!(m.rows&(1u<<row)))for(auto&v:m.e[row])v={};
    e.space.rewind(mark);require(e.space.allocate(words*8,64)==base,SBN3_FATAL_LIFETIME,"GCD history pack base");
}
static void grow_matrix(Matrix&m,size_t cap,Engine&e){
    Nat *slots[4]{};unsigned count=0;for(unsigned row=0;row<2;++row)if(m.rows&(1u<<row))for(auto&v:m.e[row])slots[count++]=&v;
    U *const base=slots[0]->p;size_t old_words=0;
    for(unsigned i=0;i<count;++i){require(slots[i]->p==base+old_words&&slots[i]->cap<=cap,SBN3_FATAL_LIFETIME,"GCD history grow layout");old_words+=slots[i]->cap;}
    const size_t words=count*cap;
    if(words>old_words)require(e.space.allocate((words-old_words)*8,64)==base+old_words,SBN3_FATAL_LIFETIME,"GCD history grow extent");
    for(unsigned i=count;i-->0;){auto&v=*slots[i];U *dest=base+i*cap;if(v.n)memmove(dest,v.p,v.n*8);v.p=dest;v.cap=cap;}
}
static bool column_program(Pair&v,const Matrix&m,size_t mn,Engine&e){
    if(std::min(mn,v.size())<512)return false;
    product::MatrixProgramPlan plan{};const unsigned workers=mn+v.size()>=e.policy.batch_min?e.scope->width:1;
    if(!product::matrix_program_query(mn,v.size(),workers,3,plan,true,true))return false;
    const auto&q=plan.product;
    if(q.info.output_limbs>v.a.cap||q.info.output_limbs>v.b.cap||q.prepared_bytes+plan.workspace_bytes+512>e.space.capacity()-e.space.used())return false;
    FrameMark mark(e.space);auto tables=e.space.subframe(q.prepared_bytes+128,128);const auto program=product_program_prepare(q,tables);auto work=e.space.subframe(plan.workspace_bytes,64);
    MatrixProductInputs input{};input.compact_forward=e.policy.compact_forward;input.vector=true;
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)input.left[i][j]={m.e[i][j].p,m.e[i][j].n};
    input.right[0][0]={v.a.p,v.a.n};input.right[1][0]={v.b.p,v.b.n};input.out[0][0]={v.a.p,v.a.cap};input.out[1][0]={v.b.p,v.b.cap};
    product::matrix_program_execute(program,work,e.scope,input);v.a.n=v.b.n=q.info.output_limbs;v.a.trim();v.b.trim();
    if(e.metrics){++e.metrics->column_episodes;e.metrics->products+=4;}return true;
}
static void column_update(Pair&v,const Matrix&m,Engine&e){
    PhaseMeter phase(e.metrics,Phase::compose,std::max(v.size(),std::max(m.e[0][0].n,m.e[1][1].n)));
    FrameMark mark(e.space);size_t mn=0;for(const auto&row:m.e)for(const auto&a:row)mn=std::max(mn,a.n);
    if(e.policy.column_fft&&column_program(v,m,mn,e)){if(e.metrics)++e.metrics->deferred_updates;return;}
    // Bound actual participating terms, not unrelated maxima from opposite
    // columns. A zero term needs no product-sized output reservation.
    size_t cap=1;const Span input[2]{v.a,v.b};
    for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)
        if(m.e[i][j].n&&input[j].n)cap=std::max(cap,m.e[i][j].n+input[j].n+1);
    auto x=allocate(e.space,cap),y=allocate(e.space,cap),t=allocate(e.space,cap),u=allocate(e.space,cap);
    e.multiply_pair(x,y,v.a,m.e[0][0],m.e[1][0]);
    e.multiply_pair(t,u,v.b,m.e[0][1],m.e[1][1]);
    add(x,x,t);add(y,y,u);assign(v.a,x);assign(v.b,y);
    if(e.metrics)++e.metrics->deferred_updates;
}
void deferred_column(Nat&g,Pair&p,Engine&e,unsigned rows){
    constexpr size_t limit=64;
    struct Node {Matrix matrix;Frame::Mark mark;};
    // Fixed, bounded metadata; only constructed entries are touched. No heap
    // or recursive C++ call chain for the outer coefficient history.
    alignas(Node) std::byte storage[limit*sizeof(Node)];
    static_assert(sizeof(storage)<=16384);
    size_t count=0,round=0;const size_t origin=e.space.used();
    const unsigned workers=e.scope?e.scope->width:1;
    require(e.space.capacity()-e.space.used()>=deferred_suffix_bytes(p.size(),workers,rows),SBN3_FATAL_WORKSPACE,"GCD deferred root admission");
    if(e.metrics)++e.metrics->deferred_calls;
    // For equal limb lengths, one exact quotient fits one word; a certified
    // Lehmer block also has word coefficients. Seed the reduction before
    // entering HGCD, so a short Euclidean chain need not build a large
    // prefix recipe. Keep its matrix on the stack: even if this step removes
    // no whole limb, it must not consume the eager-suffix storage guarantee.
    U seed_words[4][2]{{1,0},{0,0},{0,0},{1,0}};Matrix seed{};
    for(unsigned i=0;i<4;++i)seed.e[i/2][i%2]={seed_words[i],size_t(i==0||i==3),2};
    // A one-row fallback must not discard a component needed by this seed.
    bool seeded=false;
    if(rows==3&&p.a.n==p.b.n){
        if(!compare(p.a,p.b)){assign(g,p.a);word(p.a,0);word(p.b,1);return;}
        seeded=step(p,0,&seed,e);
    }
    for(;;){
        if(!compare(p.a,p.b)){
            assign(g,p.a);word(p.a,0);word(p.b,1);break;
        }
        const size_t n=p.size();const auto mark=e.space.mark();
        const bool terminal=n<=e.policy.leaf;
        const bool first=round++==0;
        const size_t prefix=terminal?n:outer_prefix(n,first&&rows!=3?1:e.policy.outer,first);
        auto m=make_matrix(e.space,round_words(prefix+3));
        const size_t before=e.metrics?std::max(bit_length(p.a),bit_length(p.b)):0;
        const bool advanced=!terminal&&reduce_high(p,n-prefix,m,e);
        if(e.metrics&&!terminal){++e.metrics->outer_calls;e.metrics->outer_input_words+=n;e.metrics->outer_prefix_words+=prefix;
            const size_t after=std::max(bit_length(p.a),bit_length(p.b));e.metrics->outer_drop_bits+=before>after?before-after:0;}
        const size_t keep=packed_bytes(m),base=size_t(reinterpret_cast<uint8_t*>(m.e[0][0].p)-static_cast<uint8_t*>(e.space.data()));
        const size_t child=std::max(deferred_suffix_bytes(p.size(),workers),deferred_suffix_bytes(n,workers,0));
        const bool admit=advanced&&compare(p.a,p.b)&&count<limit&&keep<=e.space.capacity()-base&&child<=e.space.capacity()-base-keep;
        if(admit){
            pack_matrix(m,mark,e);::new(static_cast<void*>(storage+count*sizeof(Node)))Node{m,mark};++count;
            if(e.metrics){e.metrics->history_peak_bytes=std::max(e.metrics->history_peak_bytes,e.space.used()-origin);e.metrics->history_peak_nodes=std::max(e.metrics->history_peak_nodes,count);}
            continue;
        }
        // Complete the suffix using the already-admitted eager implementation.
        // m may already describe its first successful reduction.
        if(!terminal&&compare(p.a,p.b)&&e.metrics)++e.metrics->deferred_fallbacks;
        if(!count&&rows!=3){m.rows=rows;pack_matrix(m,mark,e);}
        grow_matrix(m,round_words(n+3),e);
        Engine eager{e.space,e.scope,e.policy,e.metrics};eager.policy.deferred=false;
        if(eager.policy.outer==4)eager.policy.outer=2;
        // A rejected high-prefix call has already established that an exact
        // step is needed. Do not run the same unsuccessful call twice.
        if(!terminal&&!advanced)(void)step(p,0,&m,eager);
        if(!compare(p.a,p.b))assign(g,p.a);else execution(g,&m,p,eager);
        // The GCD has been emitted. Reuse the two operand buffers for the
        // second column; the remaining computation no longer needs residues.
        if(m.rows&1)assign(p.a,m.e[0][1]);else word(p.a,0);
        if(m.rows&2)assign(p.b,m.e[1][1]);else word(p.b,0);e.space.rewind(mark);break;
    }
    while(count){
        auto*node=std::launder(reinterpret_cast<Node*>(storage+(--count)*sizeof(Node)));
        column_update(p,node->matrix,e);const auto mark=node->mark;node->~Node();e.space.rewind(mark);
    }
    if(seeded)column_update(p,seed,e);
}
}
