// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/gcd.h"
#include "gcd/jacobi_state.hpp"
#include "runtime/scratch.hpp"
#include "runtime/team.hpp"
#include "value/limbs.hpp"
#include "value/divrem_words.hpp"
#include "kernels/x86v4ifma/gcd_head.hpp"
#include <algorithm>
#include <cstring>

namespace sbn::v3::gcd {
using U=uint64_t;using W=__uint128_t;
inline constexpr size_t direct_product_limit=96;
struct Span {const U *p=nullptr;size_t n=0;};
struct Nat {
    U *p=nullptr;size_t n=0,cap=0;
    operator Span() const noexcept {return {p,n};}
    void trim() noexcept {while(n&&!p[n-1])--n;}
};
struct Pair {Nat a,b;size_t size()const{return std::max(a.n,b.n);}};
struct Matrix {Nat e[2][2];unsigned rows=3;};
// A matrix-only reduction need not return the reduced integers. It does
// return a lower bound on both of them: each is >= 2^lower_bits. The bound
// survives truncation/lifting and lets a parent certify an approximate
// prefix without materializing the child's two exact remainders.
struct ResidualBound {size_t lower_bits=0;};
enum class Phase:unsigned {leaf,adjust,cyclic,compose,middle,finish,count};
struct Metrics {
    void (*leaf_observer)(void *,const Pair &,const Matrix *,size_t,const gcd_small::Matrix &)=nullptr;
    void *leaf_context=nullptr;
    uint64_t heads=0,divisions=0,recursive_calls=0,products=0,pairs=0,cyclic=0,compositions=0,cyclic_sums=0,pair_preparations=0,matrix_episodes=0,projected=0,middle_windows=0,middle_guard_stops=0;
    size_t peak=0;
    uint64_t middle_recursive_windows=0,middle_range_stops=0,middle_bound_stops=0;
    uint64_t middle_cache_compositions=0,batch_compositions=0,batch_adjustments=0;
    uint64_t product_width[33]{},product_fallback=0,middle_declined=0,cache_declined=0;
    uint64_t matrix_basis[11][33]{};
    uint64_t leaf_streams=0,leaf_stream_wraps=0,leaf_stream_flushes=0;
    uint64_t outer_calls=0,outer_input_words=0,outer_prefix_words=0,outer_drop_bits=0;
    uint64_t deferred_calls=0,deferred_updates=0,deferred_fallbacks=0,column_episodes=0;size_t history_peak_bytes=0,history_peak_nodes=0;
    uint64_t (*clock)()=nullptr; // optional research clock; no clock in normal calls
    uint64_t phase_ns[unsigned(Phase::count)][21]{},phase_calls[unsigned(Phase::count)][21]{};
};
class PhaseMeter {
    Metrics *m_;unsigned phase_,bin_;uint64_t start_;
public:
    PhaseMeter(Metrics *m,Phase p,size_t n):m_(m&&m->clock?m:nullptr),phase_(unsigned(p)),
        bin_(n?std::min(20u,63u-unsigned(__builtin_clzll(n))):0),start_(m_?m_->clock():0){}
    ~PhaseMeter(){if(m_){m_->phase_ns[phase_][bin_]+=m_->clock()-start_;++m_->phase_calls[phase_][bin_];}}
};
struct Policy {
    bool word32=false,leaf_pipeline=false,deferred=false,column_fft=false,compact_forward=false;
    bool fft_matrix=false,fft_vector=false,joint_vector=false;
    size_t leaf=96,finish=768,parallel=8192,stop_guard=1,batch_min=256;
    unsigned outer=0; // 0 value-only (plain) / half (extended), 1 high half, 2 high two-thirds, 3 high third, 4 half then two-thirds
    bool pairs=true,cyclic=false,strassen=true,parallel_rows=false,cyclic_sum=true,retain_pair_cache=false;
    bool matrix_fft=false,projected=false,middle=false,word_ifma=false,partial_tail=false;
    bool middle_recursive=false;
    bool middle_cache=false;
    bool legacy_capacity=false;
    bool parallel_seven=false,parallel_adjust=false;
};
// Shared arithmetic baseline for controlled experiments and default selection.
Policy baseline_policy(unsigned workers,bool extended=false);
Policy default_policy(unsigned workers,bool extended=false);
struct HeadStream;
void head_stream_push(HeadStream *,const gcd_small::Matrix &);
void head_stream_flush(HeadStream *);
struct Engine {
    static constexpr bool symbolic=false;
    Frame &space;sbn3_team_scope *scope=nullptr;Policy policy{};Metrics *metrics=nullptr;HeadStream *stream=nullptr;
    void multiply(Nat &,Span,Span);
    void multiply_pair(Nat &,Nat &,Span,Span,Span);
    void divide(Nat &q,Nat &r,Span a,Span b);
};
// A distinct control instantiation keeps ordinary GCD's quotient loop free
// of symbol tests. Arithmetic/matrix kernels are shared unchanged.
struct SymbolEngine : Engine {static constexpr bool symbolic=true;JacobiState *symbol;};
inline Engine rebind(Engine &e,Frame &f,sbn3_team_scope *scope) {return {f,scope,e.policy,e.metrics};}
inline SymbolEngine rebind(SymbolEngine &e,Frame &f,sbn3_team_scope *scope) {return {{f,scope,e.policy,e.metrics},e.symbol};}
inline NoSymbol observer(Engine &) {return {};}
struct SymbolObserver {JacobiState *p;void operator()(unsigned d,U q)const {(*p)(d,q);}};
inline SymbolObserver observer(SymbolEngine &e) {return {e.symbol};}
template<class E>struct SymbolTransaction {explicit SymbolTransaction(E&){} void commit(){} };
template<>struct SymbolTransaction<SymbolEngine> {
    JacobiState *p,saved;bool keep=false;
    explicit SymbolTransaction(SymbolEngine &e):p(e.symbol),saved(*p){}
    void commit(){keep=true;}~SymbolTransaction(){if(!keep)*p=saved;}
};
int symbol_finish(const U *,size_t,const U *,size_t,void *,JacobiState);
int symbol_execution(Pair &,SymbolEngine &);
int symbol_small(const U *,size_t,const U *,size_t,void *);
inline Nat allocate(Frame &f,size_t cap){return {f.alloc<U>(cap),0,cap};}
inline size_t round_words(size_t n){return (n+7)&~size_t(7);}
inline Span normalized(Span a){while(a.n&&!a.p[a.n-1])--a.n;return a;}
inline size_t bit_length(Span a){a=normalized(a);return a.n?64*a.n-__builtin_clzll(a.p[a.n-1]):0;}
inline ResidualBound residual_bound(const Pair &p){
    const size_t bits=std::min(bit_length(p.a),bit_length(p.b));
    require(bits,SBN3_FATAL_MATH,"HGCD positive residual certificate");return {bits-1};
}
inline bool lift_bound(ResidualBound child,const Matrix &m,size_t shift,
                       unsigned error_bits,ResidualBound &out){
    size_t bits=0;for(const auto &row:m.e)for(const auto &v:row)bits=std::max(bits,bit_length(v));
    // A row sum is <2^(bits+1). If each input error is <2^error_bits,
    // reserve a factor of two between that row error and the child bound.
    if(child.lower_bits<bits+error_bits+2)return false;
    out.lower_bits=child.lower_bits-1+shift;return true;
}
inline int compare(Span a,Span b){a=normalized(a);b=normalized(b);
    if(a.n!=b.n)return a.n>b.n?1:-1;
    for(size_t i=a.n;i-->0;)if(a.p[i]!=b.p[i])return a.p[i]>b.p[i]?1:-1;return 0;}
inline void assign(Nat &a,Span b){require(b.n<=a.cap,SBN3_FATAL_WORKSPACE,"GCD value capacity",b.n,a.cap);
    if(b.n&&a.p!=b.p)memmove(a.p,b.p,b.n*8);a.n=b.n;}
inline void word(Nat &a,U v){require(a.cap,SBN3_FATAL_WORKSPACE,"GCD word capacity");a.p[0]=v;a.n=bool(v);}
inline void add(Nat &r,Span a,Span b){
    if(a.n<b.n)std::swap(a,b);require(a.n<r.cap,SBN3_FATAL_WORKSPACE,"GCD sum capacity",a.n+1,r.cap);
    U c=b.n?sbn3i_add_n(r.p,a.p,b.p,long(b.n)):0;
    if(a.n>b.n){if(r.p+b.n!=a.p+b.n)memmove(r.p+b.n,a.p+b.n,(a.n-b.n)*8);
        c=limbs::add_to(r.p+b.n,a.n-b.n,&c,bool(c));}
    r.n=a.n;if(c)r.p[r.n++]=c;
}
inline void subtract(Nat &r,Span a,Span b){
    require(a.n<=r.cap&&a.n>=b.n,SBN3_FATAL_MATH,"GCD subtraction shape");
    U c=b.n?sbn3i_sub_n(r.p,a.p,b.p,long(b.n)):0;
    if(a.n>b.n){if(r.p+b.n!=a.p+b.n)memmove(r.p+b.n,a.p+b.n,(a.n-b.n)*8);
        c=limbs::sub_from(r.p+b.n,a.n-b.n,&c,bool(c));}
    require(!c,SBN3_FATAL_MATH,"GCD negative residual");r.n=a.n;r.trim();
}
inline Pair high(Pair p,size_t cut){
    require(cut<p.a.cap&&cut<p.b.cap,SBN3_FATAL_WORKSPACE,"HGCD high slice");
    return {{p.a.p+cut,p.a.n>cut?p.a.n-cut:0,p.a.cap-cut},
            {p.b.p+cut,p.b.n>cut?p.b.n-cut:0,p.b.cap-cut}};
}
inline size_t matrix_capacity(size_t n){return round_words((n+1)/2+2);}
Matrix make_matrix(Frame &,size_t row0,size_t row1,unsigned rows=3);
inline Matrix make_matrix(Frame &f,size_t n){return make_matrix(f,matrix_capacity(n),matrix_capacity(n),3);}
void matrix_word(Matrix &,const gcd_small::Matrix &,Engine &);
void matrix_shear(Matrix &,unsigned column,Span q,Engine &);
void compose(Matrix &,const Matrix &,Engine &);
bool compose_seven(Matrix &,const Matrix &,Engine &);
bool compose_seven_parallel(Matrix &,const Matrix &,Engine &);
void adjust(Pair &,const Pair &head,const Matrix &,size_t cut,Engine &);
bool step(Pair &,size_t stop,Matrix *,Engine &);
U word_matrix(U,U,gcd_small::Matrix &);
bool hgcd(Pair &,Matrix &,Engine &);
// Matrix-only half reduction: inputs are scratch and have no returned value
// meaning. The positive unimodular result can be lifted/applied to originals.
bool hgcd_matrix_only(Pair &,Matrix &,Engine &,ResidualBound * =nullptr);
bool middle_matrix(Pair &,Matrix &,Engine &);
bool recursive_middle_matrix(Pair &,Matrix &,Engine &,ResidualBound &);
void apply_matrix(Pair &,const Matrix &,Engine &);
bool reduce_high(Pair &,size_t cut,Matrix &,Engine &);
bool cyclic_apply(Pair &,const Matrix &,Engine &);
size_t outer_prefix(size_t,unsigned policy,bool initial);
size_t deferred_suffix_bytes(size_t,unsigned workers,unsigned rows=3);
void deferred_column(Nat &,Pair &,Engine &,unsigned rows=3);
bool reduce(Pair &,Matrix *accumulate,Engine &,bool initial=true);
void stream_remainder(Nat &,Span,Span,Engine &,Nat *quotient=nullptr);
size_t arithmetic_envelope(size_t n,unsigned workers);
size_t execution(Nat &out,Matrix *cofactors,Pair &,Engine &);
// Diagnostic/algorithm consumers can supply a fixed policy. This is not a
// public tuning API and never performs online measurements or grid search.
void policy_query(size_t,size_t,unsigned workers,sbn3_gcd_info &);
size_t policy_gcd(U *,Span,Span,void *,size_t,const Policy &,Metrics *,sbn3_team * =nullptr,bool public_storage=false);
void policy_exgcd(sbn3_int *,sbn3_int *,sbn3_int *,sbn3_int_view,sbn3_int_view,
                  void *,size_t,sbn3_team *,const Policy &,Metrics * =nullptr);
bool step(Pair &,size_t,Matrix *,SymbolEngine &);
bool hgcd(Pair &,Matrix &,SymbolEngine &);
bool hgcd_matrix_only(Pair &,Matrix &,SymbolEngine &,ResidualBound * =nullptr);
bool reduce_high(Pair &,size_t,Matrix &,SymbolEngine &);
bool reduce(Pair &,Matrix *,SymbolEngine &,bool initial=true);
bool recursive_middle_matrix(Pair &,Matrix &,SymbolEngine &,ResidualBound &);
// Private native bridge: exact in-place application, buffers padded to n.
void apply_head(U *,U *,size_t,const gcd_small::Matrix &);
void compose_head(U *,U *,size_t,const gcd_small::Matrix &);
void apply_head_word32(U *,U *,size_t,const gcd_small::Matrix &);
void compose_head_word32(U *,U *,size_t,const gcd_small::Matrix &);
size_t finish(U *,const U *,size_t,const U *,size_t,void *);
inline size_t finish(U *out,Span a,Span b,void *scratch){return finish(out,a.p,a.n,b.p,b.n,scratch);}
}
