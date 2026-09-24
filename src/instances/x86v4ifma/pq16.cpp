#include "backend/pq16/island.hpp"
#include <new>
#include <initializer_list>
#include <algorithm>
#include "backend/pq16/ct.hpp"
#include "backend/pq16/rac.hpp"
#include "backend/pq16/variable.hpp"
#include "backend/pq16/rac_wide.hpp"
#include "backend/pq16/rac_window.hpp"
#include "backend/pq16/cost_model.hpp"
#include "backend/pq16/numeric_domain.hpp"
namespace sbn::v3::pq16 {
struct Tables {pq16_plan plan{};Shape shape{};const CtTables *ct=nullptr;const RacTables *rac=nullptr;};
Shape query(size_t an,size_t bn,unsigned minimum_pow2) noexcept {
    if(an>(1u<<20)||bn>(1u<<20)||(minimum_pow2!=128&&minimum_pow2!=256&&minimum_pow2!=512))return {};int c=0;const auto s=pq16_lin_choose(an,bn,&c,minimum_pow2);return {s.nfull,s.branch,s.M,bool(c)};
}
Shape select(size_t an,size_t bn,unsigned workers,unsigned bits,size_t budget,bool square,double preparation) noexcept {
    auto price=[&](Shape s){const double run=native_cost(s,an,bn);
        return preparation ? run+preparation*double(table_bytes(s)) : run;};
    Shape old=execution_shape(query(an,bn),workers);
    if(!old.nfull|| (bits&&(bits<16||bits>20)))return {};
    if(workers!=1||an+bn<256||an+bn>262144){if(bits&&bits!=16)return {};return old;}
    if(an+bn>16384){ // large band (2026-09-09): the classic 16-bit plan against 16-bit balanced CT/PQ and right-angle shapes,
        // which only pay off where the classic codec would be centered (N > 2^17: -12..-29 % measured at M5 163840, M3 196608,
        // M7 229376); below that the balanced kernels are 5-11 % slower than PFA/PQ at equal shapes.
        if(bits&&bits!=16)return {};
        Shape best=old;double lowest=price(old);
        const size_t need=((64*an+15)/16+(64*bn+15)/16)/2;
        const auto input=variable_input(an,bn,16);
        for(unsigned m:{3u,5u,7u}){unsigned n=256;while(size_t(m)*n<need)n*=2;
            if(size_t(m)*n<=131072||size_t(m)*n>524288)continue;
            for(Recipe r:{Recipe::CooleyTukeyPQ,Recipe::RightAngle}){if(m==1&&r==Recipe::RightAngle)continue;
                Shape s{m*n,n,m,false,r,16,true};unsigned nb=n;while(!variable_supported(s,input)&&size_t(m)*nb<2*need){nb*=2;s=Shape{m*nb,nb,m,false,r,16,true};}
                if(!variable_supported(s,input)||(budget&&scratch_bytes(s,an,bn,workers,square)>budget))continue;
                const double c=price(s);if(c<lowest){lowest=c;best=s;}}}
        return best;
    }
    Shape candidates[64]{};double costs[64]{};unsigned count=0;double lowest=INFINITY; // up to 4 bits x (1 + 3x4) recipes
    auto consider=[&](Shape s,const VariableInput *input=nullptr){
        if(s.nfull>32768||(s.bits>16&&!(input?variable_supported(s,*input):variable_supported(s,an,bn)))||(budget&&scratch_bytes(s,an,bn,workers,square)>budget))return;
        const double c=price(s);candidates[count]=s;costs[count++]=c;if(c<lowest)lowest=c;
    };
    if(!bits||bits==16)consider(old);
    for(unsigned b=bits?bits:16;b<=(bits?bits:20);++b){
        const auto input=variable_input(an,bn,b);
        const size_t need=(input.unsigned_required+1)/2;
        for(unsigned m:{1u,3u,5u,7u}){unsigned n=m==1?512:m==3?256:b==16?128:64;while(size_t(m)*n<need)n*=2;   // branch >= 64 for M5/M7 (2026-09-09: 20b M5 320 / M7 448 in the small band)
            // The larger wide-M7 tower did not beat the 16-bit alternative
            // in its local confirmation; explicit bit queries retain it.
            if(!bits&&b>16&&m==7&&n>2048)continue;
            Shape s{m*n,n,m,false,b==16&&m==1?Recipe::PfaPQ:Recipe::CooleyTukeyPQ,b};
            if(b==16&&m!=1){
                // 16-bit odd radix: the right-angle band follows execution_shape; beyond it CT/PQ and PFA/PQ compete as before.
                const Shape e=execution_shape(s,1);
                if(e.recipe==Recipe::RightAngle)consider(e);else{consider(s);s.recipe=Recipe::PfaPQ;consider(s);}
            }else{
                consider(s,&input);if(b>16&&m!=1){s.recipe=Recipe::RightAngle;consider(s,&input);}
                if(b>16){ // balanced digits: same geometry unless the carry digits need the next size; CT/PQ and right-angle, each within its own envelope
                    unsigned nb=n;while(!variable_supported(Shape{m*nb,nb,m,false,Recipe::CooleyTukeyPQ,b,true},input)&&size_t(m)*nb<2*need)nb*=2;
                    Shape t{m*nb,nb,m,false,Recipe::CooleyTukeyPQ,b,true};consider(t,&input);if(m!=1){t.recipe=Recipe::RightAngle;consider(t,&input);}
                }
            }
        }
    }
    // Near ties do not justify a much larger root table and setup cost.
    Shape best{};size_t bytes=SIZE_MAX;double best_cost=INFINITY;
    for(unsigned j=0;j<count;++j)if(costs[j]<best_cost){best=candidates[j];best_cost=costs[j];bytes=table_bytes(best);}
    const size_t worthwhile=bytes-bytes/8;
    for(unsigned j=0;j<count;++j)if(costs[j]<=1.01*lowest){const size_t tb=table_bytes(candidates[j]);
        if(tb<=worthwhile&&tb<bytes){best=candidates[j];bytes=tb;}}
    return best;
}
size_t table_bytes(Shape s) noexcept {
    if(!s.nfull)return 0;const auto shape=pq16_shape_of(s.branch);size_t bytes=((sizeof(Tables)+127)&~size_t(127))+127;
    // tw22[lg n] + the classic r8 stages + whatever the odd-radix branch plan adds (each (kind, length) once)
    auto stage_bytes=[](uint32_t n,bool radix8){return root_bank::has_stage(unsigned(__builtin_ctz(n)),pq16_twc(n),radix8)?size_t(0):size_t(n)*(pq16_twc(n)?4:8);};
    uint64_t seen22=1ull<<__builtin_ctz(s.branch),seen8=0;
    if(s.branch>root_bank::pq_branch)bytes+=4*s.branch;
    bytes+=stage_bytes(s.branch,false);
    for(unsigned i=0;i<shape.cnt;++i){seen8|=1ull<<__builtin_ctz(shape.len[i]);bytes+=stage_bytes(shape.len[i],true);}
    const auto odd=pq16_odd_plan_of(s.branch);
    if(s.radix!=1)for(unsigned i=0;i<odd.cnt;++i){const unsigned lg=__builtin_ctz(odd.len[i]);uint64_t &seen=odd.radix[i]==8?seen8:seen22;
        if(!(seen>>lg&1)){seen|=1ull<<lg;bytes+=stage_bytes(odd.len[i],odd.radix[i]==8);}}
    return bytes+(s.recipe==Recipe::CooleyTukeyPQ?ct_table_bytes(s):s.recipe==Recipe::RightAngle?rac_table_bytes(s):0);
}
bool tables_published(Shape s) noexcept {
    if(!s.nfull||s.branch<128||s.branch>root_bank::pq_branch)return false;
    if(s.branch>root_bank::branch&&s.radix!=1)return false;
    for(unsigned n=s.branch;n>=64;n/=2)
        if(!root_bank::has_stage(unsigned(__builtin_ctz(n)),pq16_twc(n)))return false;
    // All power-of-two stages in this branch are covered by the bank. CT
    // and RAC additionally need their own odd-radix cross-root prefix.
    if(s.recipe==Recipe::CooleyTukeyPQ)return s.radix==1||root_bank::ct_twiddle(s.radix,s.branch)||root_bank::ct_fine_twiddle(s.radix,s.branch);
    if(s.recipe==Recipe::RightAngle)return root_bank::rac_twiddle(s.radix,s.branch);
    return s.recipe==Recipe::PfaPQ;
}
bool compact_table_setup(Shape s) noexcept {
    if(tables_published(s))return true;
    return s.recipe==Recipe::CooleyTukeyPQ&&s.branch<=root_bank::branch&&ct_factored(s);
}
size_t scratch_bytes(Shape s,size_t an,size_t bn,unsigned workers,bool square) noexcept {
    if(!s.nfull)return 0;
    const bool one=square && (!s.centered || s.radix==1);
    return (one?16ul:32ul)*s.nfull+1024+(s.balanced?8*(s.recipe==Recipe::RightAngle?vx_rac_tail_limbs(s):vx_tail_limbs(s))+128:0)+(s.centered?
        (pq16_direct_input(workers)?0:(square?4ul:8ul)*s.nfull)+
        (pq16_direct_output(workers)?0:8*(an+bn)):0);
}
Tables *prepare(Frame &frame,Shape s) noexcept {
    require(s.nfull,SBN3_FATAL_ARGUMENT,"pq16 supported shape");auto *p=new(frame.allocate(sizeof(Tables),128)) Tables{};p->shape=s;p->plan.builder=&frame;pq16_plan_ensure(&p->plan,s.branch,s.radix!=1);
    if(s.recipe==Recipe::CooleyTukeyPQ)p->ct=ct_prepare(frame,p->plan,s);
    else if(s.recipe==Recipe::RightAngle)p->rac=rac_prepare(frame,p->plan,s);
    return p;
}
template<unsigned B,bool S>static bool multiply_variable(uint64_t *r,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const CtTables &t,Frame &f,const double *cached=nullptr,bool square=false){
    switch(t.shape.radix){case 1:return vx_multiply<1,B,S>(r,a,an,b,bn,t,f,cached,square);case 3:return vx_multiply<3,B,S>(r,a,an,b,bn,t,f,cached,square);case 5:return vx_multiply<5,B,S>(r,a,an,b,bn,t,f,cached,square);case 7:return vx_multiply<7,B,S>(r,a,an,b,bn,t,f,cached,square);}return false;
}
template<unsigned B,bool S>static bool multiply_variable_rac(uint64_t *r,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const RacTables &t,Frame &f,const double *cached=nullptr,bool square=false){
    switch(t.shape.radix){case 3:return vx_rac_multiply<3,B,S>(r,a,an,b,bn,t,f,cached,square);case 5:return vx_rac_multiply<5,B,S>(r,a,an,b,bn,t,f,cached,square);case 7:return vx_rac_multiply<7,B,S>(r,a,an,b,bn,t,f,cached,square);}return false;
}
// wide codec dispatch: bits 17..20 unsigned or balanced, 16-bit balanced (large band); 21-bit measured out of the envelope (2026-09-09)
static bool multiply_wide(uint64_t *r,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const Tables &t,Frame &f,const double *cached=nullptr,bool square=false){
    const bool s=t.shape.balanced;
    if(t.rac)switch(t.shape.bits){
        case 16:return s&&multiply_variable_rac<16,true>(r,a,an,b,bn,*t.rac,f,cached,square);
        case 17:return s?multiply_variable_rac<17,true>(r,a,an,b,bn,*t.rac,f,cached,square):multiply_variable_rac<17,false>(r,a,an,b,bn,*t.rac,f,cached,square);
        case 18:return s?multiply_variable_rac<18,true>(r,a,an,b,bn,*t.rac,f,cached,square):multiply_variable_rac<18,false>(r,a,an,b,bn,*t.rac,f,cached,square);
        case 19:return s?multiply_variable_rac<19,true>(r,a,an,b,bn,*t.rac,f,cached,square):multiply_variable_rac<19,false>(r,a,an,b,bn,*t.rac,f,cached,square);
        case 20:return s?multiply_variable_rac<20,true>(r,a,an,b,bn,*t.rac,f,cached,square):multiply_variable_rac<20,false>(r,a,an,b,bn,*t.rac,f,cached,square);}
    else switch(t.shape.bits){
        case 16:return s&&multiply_variable<16,true>(r,a,an,b,bn,*t.ct,f,cached,square);
        case 17:return s?multiply_variable<17,true>(r,a,an,b,bn,*t.ct,f,cached,square):multiply_variable<17,false>(r,a,an,b,bn,*t.ct,f,cached,square);
        case 18:return s?multiply_variable<18,true>(r,a,an,b,bn,*t.ct,f,cached,square):multiply_variable<18,false>(r,a,an,b,bn,*t.ct,f,cached,square);
        case 19:return s?multiply_variable<19,true>(r,a,an,b,bn,*t.ct,f,cached,square):multiply_variable<19,false>(r,a,an,b,bn,*t.ct,f,cached,square);
        case 20:return s?multiply_variable<20,true>(r,a,an,b,bn,*t.ct,f,cached,square):multiply_variable<20,false>(r,a,an,b,bn,*t.ct,f,cached,square);}
    return false;
}
void multiply(uint64_t *r,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const Tables &tables,Frame &space,sbn3_team_scope *scope) noexcept {
    sbn_team adapter{scope};if(scope)require_scope_leader(scope);
    if(tables.rac){
        if constexpr(SBN3_CHECK_SMALL)require(!scope||scope->width==1,SBN3_FATAL_ARGUMENT,"RAC planned execution width");bool ok=false;
        if(tables.shape.bits>16||tables.shape.balanced){require(multiply_wide(r,a,an,b,bn,tables,space),SBN3_FATAL_MATH,"RAC wide numerical emit invariant");return;}
        switch(tables.shape.radix){case 3:ok=rac_multiply<3>(r,a,an,b,bn,*tables.rac,space);break;case 5:ok=rac_multiply<5>(r,a,an,b,bn,*tables.rac,space);break;case 7:ok=rac_multiply<7>(r,a,an,b,bn,*tables.rac,space);break;}
        require(ok,SBN3_FATAL_MATH,"RAC numerical emit invariant");return;
    }
    if(tables.ct){
        if constexpr(SBN3_CHECK_SMALL)require(!scope||scope->width==1,SBN3_FATAL_ARGUMENT,"CT planned execution width");bool ok=false;
        if(tables.shape.bits>16||tables.shape.balanced){require(multiply_wide(r,a,an,b,bn,tables,space),SBN3_FATAL_MATH,"variable FFT numerical emit invariant");return;}
        switch(tables.shape.radix){case 3:ok=ct_multiply<3>(r,a,an,b,bn,*tables.ct,space);break;case 5:ok=ct_multiply<5>(r,a,an,b,bn,*tables.ct,space);break;case 7:ok=ct_multiply<7>(r,a,an,b,bn,*tables.ct,space);break;}
        require(ok,SBN3_FATAL_MATH,"CT numerical emit invariant");return;
    }
    int ok=pq16_mul_core(r,a,an,b,bn,&space,scope?&adapter:nullptr,scope?int(scope->width):1,const_cast<pq16_plan *>(&tables.plan),{tables.shape.nfull,tables.shape.branch,tables.shape.radix},tables.shape.centered);
    require(ok,SBN3_FATAL_MATH,"pq16 numerical emit/shape invariant");
}
bool supported(Shape s,size_t an,size_t bn,unsigned workers) noexcept {
    if(!an||!bn||an>(1u<<20)||bn>(1u<<20)||!workers||workers>32||!s.branch||(s.branch&(s.branch-1))||
       s.nfull!=s.branch*s.radix||(s.radix!=1&&s.radix!=3&&s.radix!=5&&s.radix!=7))return false;
    if(s.recipe!=Recipe::PfaPQ && workers!=1)return false;
    if(s.balanced||s.bits>16)return variable_supported(s,an,bn);
    if(s.bits!=16||s.branch<128||2*(an+bn)>s.nfull)return false;
    if(s.recipe==Recipe::RightAngle)return !s.centered&&s.radix!=1&&s.nfull<=20480;
    if(s.recipe==Recipe::CooleyTukeyPQ)return !s.centered&&s.radix!=1&&s.nfull<=32768;
    if(unsigned(s.recipe)>2||s.nfull>(s.centered?PQ16_MAX_N_C:PQ16_MAX_N))return false;
    if(!s.centered&&s.radix==7&&s.nfull>PQ16_PFA7_MAX_N)return false;
    return !(s.centered&&s.radix==1&&s.nfull==PQ16_MAX_N_C&&4*std::min(an,bn)>3*(s.nfull/4));
}
size_t cached_scratch_bytes(Shape s,size_t an,size_t bn) noexcept {
    return 16ul*s.nfull+1024+(s.centered?(s.radix>1?16ul*s.nfull:0)+8*(an+bn+8):0)+
        (s.balanced?8*(s.recipe==Recipe::RightAngle?vx_rac_tail_limbs(s):vx_tail_limbs(s)):0);
}
template<unsigned B,bool S>static void forward_wide(double *out,const uint64_t *a,size_t n,const Tables &t){
    if(t.rac){switch(t.shape.radix){
        case 3:vx_rac_forward<3,B,S>(out,a,n,*t.rac);return;
        case 5:vx_rac_forward<5,B,S>(out,a,n,*t.rac);return;
        case 7:vx_rac_forward<7,B,S>(out,a,n,*t.rac);return;}}
    else{switch(t.shape.radix){
        case 1:vx_forward<1,B,S>(out,a,n,*t.ct);return;
        case 3:vx_forward<3,B,S>(out,a,n,*t.ct);return;
        case 5:vx_forward<5,B,S>(out,a,n,*t.ct);return;
        case 7:vx_forward<7,B,S>(out,a,n,*t.ct);return;}}
    fatal(SBN3_FATAL_ARGUMENT,"FFT spectrum radix");
}
void forward_spectrum(double *out,const uint64_t *a,size_t count,const Tables &t,sbn3_team_scope *scope) noexcept {
    const auto s=t.shape;const bool balanced=s.balanced;
    if(s.bits>16||balanced){
        require(!scope||scope->width==1,SBN3_FATAL_ARGUMENT,"wide FFT spectrum width");
        switch(s.bits){
            case 16:forward_wide<16,true>(out,a,count,t);return;
            case 17:if(balanced)forward_wide<17,true>(out,a,count,t);else forward_wide<17,false>(out,a,count,t);return;
            case 18:if(balanced)forward_wide<18,true>(out,a,count,t);else forward_wide<18,false>(out,a,count,t);return;
            case 19:if(balanced)forward_wide<19,true>(out,a,count,t);else forward_wide<19,false>(out,a,count,t);return;
            case 20:if(balanced)forward_wide<20,true>(out,a,count,t);else forward_wide<20,false>(out,a,count,t);return;
        }
    }
    if(t.ct){switch(s.radix){case 3:ct_forward<3>(out,a,count,*t.ct);return;case 5:ct_forward<5>(out,a,count,*t.ct);return;case 7:ct_forward<7>(out,a,count,*t.ct);return;}}
    if(t.rac){switch(s.radix){case 3:rac_plus_forward<3>(out,a,count,*t.rac);return;case 5:rac_plus_forward<5>(out,a,count,*t.rac);return;case 7:rac_plus_forward<7>(out,a,count,*t.rac);return;}}
    sbn_team adapter{scope};auto *team=scope?&adapter:nullptr;const int width=scope?int(scope->width):1;
    if(s.radix==1){pq16_input_stage_w(out,a,count,s.branch,&t.plan,s.centered?2:0,team,width);pq16_fwd_core_w(out,s.branch,&t.plan,team,width);}
    else pq16_pfa_fwd_w(out,a,count,s.branch,s.radix,&t.plan,s.centered?2:0,team,width);
}
void apply_spectrum(uint64_t *out,const double *cached,const uint64_t *original,size_t an,const uint64_t *fresh,size_t bn,bool square,const Tables &t,Frame &f,sbn3_team_scope *scope) noexcept {
    const auto s=t.shape;bool ok=false;
    if(t.ct||t.rac){
        require(!scope||scope->width==1,SBN3_FATAL_ARGUMENT,"FFT cached execution width");
        if(s.bits>16||s.balanced)ok=multiply_wide(out,original,an,fresh,bn,t,f,cached,square);
        else if(t.ct){switch(s.radix){case 3:ok=ct_multiply<3>(out,original,an,fresh,bn,*t.ct,f,cached,square);break;case 5:ok=ct_multiply<5>(out,original,an,fresh,bn,*t.ct,f,cached,square);break;case 7:ok=ct_multiply<7>(out,original,an,fresh,bn,*t.ct,f,cached,square);break;}}
        else{switch(s.radix){case 3:ok=rac_multiply<3>(out,original,an,fresh,bn,*t.rac,f,cached,square);break;case 5:ok=rac_multiply<5>(out,original,an,fresh,bn,*t.rac,f,cached,square);break;case 7:ok=rac_multiply<7>(out,original,an,fresh,bn,*t.rac,f,cached,square);break;}}
    }else{
        FrameMark mark(f);double *work=f.alloc<double>(2*size_t(s.nfull)+32),*natural=nullptr;
        if(s.centered&&s.radix>1)natural=f.alloc<double>(2*size_t(s.nfull)+32);
        if(square)memcpy(work,cached,16*size_t(s.nfull));else forward_spectrum(work,fresh,bn,t,scope);
        uint64_t *sink=s.centered?f.alloc<uint64_t>(an+bn+8):out;
        q_cctx cx{reinterpret_cast<const uint16_t *>(square?original:fresh),reinterpret_cast<const uint16_t *>(original),4*int64_t(square?an:bn),4*int64_t(an),0};
        sbn_team adapter{scope};ok=pq16_conv_emit_w(sink,an+bn,work,cached,s.branch,s.radix,s.nfull,s.centered,0,&cx,natural,const_cast<pq16_plan *>(&t.plan),scope?&adapter:nullptr,scope?int(scope->width):1);
        if(ok&&s.centered)memcpy(out,sink,(an+bn)*8);
    }
    require(ok,SBN3_FATAL_MATH,"FFT cached numerical emit invariant");
}
Shape plus_shape(size_t minimum_limbs) noexcept {
    Shape best{};for(unsigned m:{3u,5u,7u}){unsigned n=128;while(size_t(m)*n<2*minimum_limbs&&n<8192)n*=2;
        Shape s{m*n,n,m,false,Recipe::RightAngle,16,false};
        if(plus_supported(s,1,1)&&(!best.nfull||s.nfull<best.nfull))best=s;}return best;
}
bool bounded_plus_supported(Shape s,size_t an,size_t bn) noexcept {
    const size_t r=cyclic_period(s);
    return s.recipe==Recipe::RightAngle&&(s.bits==16?!s.balanced:s.balanced)&&!s.centered&&s.branch>=128&&!(s.branch&(s.branch-1))&&
        s.nfull==s.radix*s.branch&&s.nfull<=bounded_plus_cap(s.bits,s.radix)&&an&&bn&&
        std::max(an,bn)<r&&std::min(an,bn)<9*r/16;
}
template<unsigned B>static void bounded_plus_bits(uint64_t *out,const uint64_t*a,size_t an,const uint64_t*b,size_t bn,
                                                const double *cache,const Tables&t,Frame&f){
    switch(t.shape.radix){
        case 3:rac_window_multiply<3,B,(B!=16)>(out,a,an,b,bn,cache,*t.rac,f);return;
        case 5:rac_window_multiply<5,B,(B!=16)>(out,a,an,b,bn,cache,*t.rac,f);return;
        case 7:rac_window_multiply<7,B,(B!=16)>(out,a,an,b,bn,cache,*t.rac,f);return;
    }
    fatal(SBN3_FATAL_ARGUMENT,"bounded plus radix");
}
void bounded_plus_multiply(uint64_t*out,const uint64_t*a,size_t an,const uint64_t*b,size_t bn,
                           const double *cache,const Tables&t,Frame&f) noexcept {
    require(bounded_plus_supported(t.shape,an,bn),SBN3_FATAL_ARGUMENT,"bounded plus input support");
    switch(t.shape.bits){
        case 16:bounded_plus_bits<16>(out,a,an,b,bn,cache,t,f);return;
        case 17:bounded_plus_bits<17>(out,a,an,b,bn,cache,t,f);return;
        case 18:bounded_plus_bits<18>(out,a,an,b,bn,cache,t,f);return;
    }
    fatal(SBN3_FATAL_ARGUMENT,"bounded plus codec");
}
bool plus_supported(Shape s,size_t an,size_t bn) noexcept {
    // Half of the existing right-angle linear band. Both operands may fill
    // the ring; this keeps the product spectral magnitude within that band.
    return s.recipe==Recipe::RightAngle&&s.bits==16&&!s.centered&&!s.balanced&&
        (s.radix==3||s.radix==5||s.radix==7)&&s.branch>=128&&!(s.branch&(s.branch-1))&&
        s.nfull==s.branch*s.radix&&s.nfull<=10240&&an&&bn&&an<=s.nfull/2+1&&bn<=s.nfull/2+1;
}
void plus_multiply(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,bool square,const double *cached,const Tables &t,Frame &f) noexcept {
    switch(t.shape.radix){case 3:rac_plus_multiply<3>(out,a,an,b,bn,square,cached,*t.rac,f);return;
        case 5:rac_plus_multiply<5>(out,a,an,b,bn,square,cached,*t.rac,f);return;
        case 7:rac_plus_multiply<7>(out,a,an,b,bn,square,cached,*t.rac,f);return;}
    fatal(SBN3_FATAL_ARGUMENT,"plus ring radix");
}
static unsigned wide_cyclic_cap(Shape s) noexcept {
    if(s.balanced)return cyclic_balanced_cap(s.bits,s.radix);
    const unsigned cap=s.bits==17?32768:s.bits==18?(s.radix==5?5120:8192):s.bits==19?2048:s.bits==20&&s.radix==1?512:0;
    return cap;
}
size_t wide_cyclic_minimum_words(size_t an,size_t bn) noexcept {
    // min(an,bn) < floor(9*r/16) iff r >= ceil(16*(min+1)/9).
    return std::max({an,bn,(16*(std::min(an,bn)+1)+8)/9});
}
Shape cyclic_shape(size_t minimum_limbs,unsigned bits,unsigned workers,bool allow_balanced) noexcept {
    if(minimum_limbs>cyclic_max_words)return {};
    if(bits!=16){
        if(bits<17||bits>20)return {};
        Shape best{};
        for(unsigned m:{1u,3u,5u,7u}){unsigned branch=128;while(size_t(m)*branch*bits/32<minimum_limbs)branch*=2;
            Shape s{m*branch,branch,m,false,Recipe::CooleyTukeyPQ,bits,false};
            if(!cyclic_supported(s,1,1))s.balanced=true;
            if(cyclic_supported(s,1,1)&&(!best.nfull||s.nfull<best.nfull))best=s;
        }return best;
    }
    Shape best{};
    for(unsigned m:{1u,3u,5u,7u}){unsigned branch=m==3?256:128;while(size_t(m)*branch<2*minimum_limbs && branch<(1u<<19))branch*=2;
        Shape s{m*branch,branch,m,false,Recipe::PfaPQ,16,false};
        if(cyclic_supported(s,1,1) && (!best.nfull||s.nfull<best.nfull))best=s;
    }
    // The odd-radix CT kernel has the same 16-bit ring and carry contract.
    // Keep the established narrow-pair domain and the measured root prefix;
    // larger shapes and parallel teams retain their existing PFA recipe.
    if(workers==1&&best.radix!=1&&root_bank::ct_twiddle(best.radix,best.branch)){
        auto ct=best;ct.recipe=Recipe::CooleyTukeyPQ;
        const size_t ring=cyclic_period(best);
        if(cyclic_supported(ct,1,1)&&native_cost(ct,ring/2,ring)<native_cost(best,ring/2,ring))best=ct;
    }
    if(workers==1&&allow_balanced)for(unsigned m:{1u,3u,5u,7u}){
        unsigned branch=128;while(size_t(m)*branch/2<minimum_limbs)branch*=2;
        Shape candidate{m*branch,branch,m,false,Recipe::CooleyTukeyPQ,16,true};
        if(cyclic_supported(candidate,1,1)&&(!best.nfull||candidate.nfull<best.nfull))best=candidate;
    }
    return best;
}
bool cyclic_supported(Shape s,size_t an,size_t bn) noexcept {
    if(s.bits>16||s.balanced){const size_t r=cyclic_period(s);
        return s.bits>=16&&s.bits<=20&&!s.centered&&s.recipe==Recipe::CooleyTukeyPQ&&s.branch>=128&&!(s.branch&(s.branch-1))&&
            (s.radix==1||s.radix==3||s.radix==5||s.radix==7)&&s.nfull==s.radix*s.branch&&
            s.nfull<=wide_cyclic_cap(s)&&an&&bn&&std::max(an,bn)<=r&&std::min(an,bn)<(s.bits==16&&s.balanced?10:9)*r/16;
    }
    if(s.bits!=16||s.centered||s.balanced||s.branch<128||(s.branch&(s.branch-1))||
       (s.radix!=1&&s.radix!=3&&s.radix!=5&&s.radix!=7)||s.nfull!=s.radix*s.branch||!an||!bn)return false;
    if(s.recipe==Recipe::CooleyTukeyPQ)
        return s.radix!=1&&s.nfull<=ct16_cyclic_points&&std::max(an,bn)<=s.nfull/2&&std::min(an,bn)<9*(s.nfull/2)/16;
    if(s.recipe!=Recipe::PfaPQ)return false;
    // Half the established uncentered transform cap, and at most half-ring
    // on one operand: coefficient peak and spectral magnitude retain margin.
    const size_t cap=s.radix==7?PQ16_PFA7_MAX_N/2:PQ16_MAX_N/2,ring=s.nfull/2;
    return s.nfull<=cap && std::max(an,bn)<=ring &&
        (std::min(an,bn)<=ring/2||(s.nfull<=ct16_cyclic_points&&std::min(an,bn)<9*ring/16));
}
template<unsigned B,bool S,bool Prefix=false,bool Tail=false>static bool cyclic_wide(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,bool square,const double *cached,const Tables &t,Frame &f,size_t prefix=0,size_t origin=0){
    switch(t.shape.radix){
        case 1:return vx_multiply<1,B,S,true,Prefix,Tail>(out,a,an,b,bn,*t.ct,f,cached,square,prefix,origin);
        case 3:return vx_multiply<3,B,S,true,Prefix,Tail>(out,a,an,b,bn,*t.ct,f,cached,square,prefix,origin);
        case 5:return vx_multiply<5,B,S,true,Prefix,Tail>(out,a,an,b,bn,*t.ct,f,cached,square,prefix,origin);
        case 7:return vx_multiply<7,B,S,true,Prefix,Tail>(out,a,an,b,bn,*t.ct,f,cached,square,prefix,origin);
    }return false;
}
void cyclic_tail_multiply(uint64_t*out,const uint64_t*a,size_t an,const uint64_t*b,size_t bn,
                         const double*cached,const Tables&t,Frame&f,size_t origin) noexcept {
    const auto s=t.shape;
    require(s.recipe==Recipe::CooleyTukeyPQ&&origin&&origin<cyclic_period(s),SBN3_FATAL_ARGUMENT,"cyclic tail geometry");
    bool ok=false;
#define WIDE_TAIL(B) case B:ok=s.balanced?cyclic_wide<B,true,false,true>(out,a,an,b,bn,false,cached,t,f,0,origin):cyclic_wide<B,false,false,true>(out,a,an,b,bn,false,cached,t,f,0,origin);break
    switch(s.bits){WIDE_TAIL(17);WIDE_TAIL(18);WIDE_TAIL(19);WIDE_TAIL(20);}
#undef WIDE_TAIL
    require(ok,SBN3_FATAL_MATH,"cyclic tail emit");
}
void cyclic_multiply(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,bool square,const double *cached,const Tables &t,Frame &f,sbn3_team_scope *scope,size_t prefix) noexcept {
    const auto shape=t.shape;
    if(shape.bits==16&&!shape.balanced&&shape.recipe==Recipe::CooleyTukeyPQ){
        require(!scope||scope->width==1,SBN3_FATAL_ARGUMENT,"CT cyclic width");bool ok=false;
#define CT_CYCLE(M) case M:ok=prefix?ct_multiply<M,true,true>(out,a,an,b,bn,*t.ct,f,cached,square,prefix):ct_multiply<M,true>(out,a,an,b,bn,*t.ct,f,cached,square);break
        switch(shape.radix){CT_CYCLE(3);CT_CYCLE(5);CT_CYCLE(7);}
#undef CT_CYCLE
        require(ok,SBN3_FATAL_MATH,"CT cyclic emit");if(prefix)return;
        uint64_t all=UINT64_MAX;for(size_t j=0;j<cyclic_period(shape);++j)all&=out[j];
        if(all==UINT64_MAX)memset(out,0,8*cyclic_period(shape));return;
    }
    if(shape.bits>16||shape.balanced){
        require(!scope||scope->width==1,SBN3_FATAL_ARGUMENT,"wide cyclic width");bool ok=false;
#define WIDE_CYCLE(B) case B:if(prefix)ok=shape.balanced?cyclic_wide<B,true,true>(out,a,an,b,bn,square,cached,t,f,prefix):cyclic_wide<B,false,true>(out,a,an,b,bn,square,cached,t,f,prefix);else ok=shape.balanced?cyclic_wide<B,true>(out,a,an,b,bn,square,cached,t,f):cyclic_wide<B,false>(out,a,an,b,bn,square,cached,t,f);break
        switch(shape.bits){WIDE_CYCLE(16);WIDE_CYCLE(17);WIDE_CYCLE(18);WIDE_CYCLE(19);WIDE_CYCLE(20);}
#undef WIDE_CYCLE
        require(ok,SBN3_FATAL_MATH,"wide cyclic emit");if(prefix)return;uint64_t all=UINT64_MAX;
        for(size_t j=0;j<cyclic_period(shape);++j)all&=out[j];if(all==UINT64_MAX)memset(out,0,8*cyclic_period(shape));return;
    }
    FrameMark mark(f);const auto s=t.shape;double *work=f.alloc<double>(2*size_t(s.nfull)+32);const double *other=cached;
    if(cached){if(square)memcpy(work,cached,16*size_t(s.nfull));else forward_spectrum(work,b,bn,t,scope);}
    else{if(square)other=work;else{auto *fresh=f.alloc<double>(2*size_t(s.nfull)+32);forward_spectrum(fresh,b,bn,t,scope);other=fresh;}forward_spectrum(work,a,an,t,scope);}
    sbn_team adapter{scope};q_cctx unused{};
    const bool ok=pq16_conv_emit_w(out,prefix?prefix:s.nfull/2,work,other,s.branch,s.radix,s.nfull,0,prefix?2:1,&unused,nullptr,const_cast<pq16_plan *>(&t.plan),scope?&adapter:nullptr,scope?int(scope->width):1);
    require(ok,SBN3_FATAL_MATH,"FFT cyclic emit");if(prefix)return;
    uint64_t all=UINT64_MAX;for(size_t k=0;k<s.nfull/2;++k)all&=out[k];if(all==UINT64_MAX)memset(out,0,4*size_t(s.nfull));
}
}
