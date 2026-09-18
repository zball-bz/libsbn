// Native test/probe only. Production setup and arithmetic are used verbatim.
#include "backend/ntt_p48/engine.hpp"
#include "../oracle/oracle.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include <time.h>
#include <stdlib.h>
namespace {
namespace p=sbn::v3::SBN3_P48_NS;
using p::V;
#include "word_funnel_reference.hpp"
static uint64_t seed=0x83597251u;
static uint64_t rnd(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static uint64_t tick(){unsigned aux;_mm_lfence();auto t=__rdtscp(&aux);_mm_lfence();return t;}
static double probe_temp(){
    static char sensor[128]{};
    if(!sensor[0])for(unsigned i=0;i<128;++i){char path[128],name[64]{};snprintf(path,sizeof path,"/sys/class/hwmon/hwmon%u/name",i);
        FILE *f=fopen(path,"r");if(!f)continue;(void)fgets(name,sizeof name,f);fclose(f);
        if(!strncmp(name,"k10temp",7)){snprintf(sensor,sizeof sensor,"/sys/class/hwmon/hwmon%u/temp1_input",i);break;}}
    FILE *f=fopen(sensor,"r");long n=0;assert(f && fscanf(f,"%ld",&n)==1);fclose(f);return n/1000.;
}
static double probe_cool(){for(unsigned n=0;n<600;++n){const double t=probe_temp();if(t<=62)return t;timespec wait{0,200000000};nanosleep(&wait,nullptr);}abort();}
static void primes(p::Primes &ps){
    for(int q=0;q<p::NP;++q){auto &v=ps.P[q];v.p=p::PR[q];v.p2=2*v.p;
        uint64_t inv=1;for(int k=0;k<6;++k)inv*=2-v.p*inv;
        v.J=(-inv)&p::M52;v.r1=(uint64_t(1)<<52)/v.p;v.inv2=p::cc_of((v.p+1)/2,v.p);ps.V[q].init(v);}
}
constexpr size_t nv=64;
__attribute__((noinline)) static void crt(V *out,const V *in,const p::EmitK &k,bool direct){
    for(size_t v=0;v<nv;++v){V x[p::NP],c[p::NC];for(int q=0;q<p::NP;++q)x[q]=in[v*p::NP+q];
        if(direct)p::direct_compose(k,x,c);else p::garner_compose(k,x,c);
        for(int j=0;j<p::NC;++j)out[v*p::NC+j]=c[j];}
}
__attribute__((noinline)) static void codec(V *out,const uint64_t *in,const p::Dec &d,const WordFunnelReference *ref=nullptr){
    for(int q=0;q<p::NP;++q)for(size_t v=0;v<nv;++v){
        if(ref && d.t8 && d.TB8/8<=184){const V L0=_mm512_load_si512(in+v*32),L1=_mm512_load_si512(in+v*32+8),L2=_mm512_load_si512(in+v*32+16);V pcs[4];
            for(int k=0;k<d.npc;++k)pcs[k]=ref->piece(k,L0,L1,L2);out[q*nv+v]=p::dec8_prepared(d,pcs,q);}
        else out[q*nv+v]=p::dec1q(d,in+v*32,q,int((d.TB8&63)?v&1:0));
    }
}
}
extern "C" void SBN3_ENDPOINT_NAME(int timing){
    p::Primes ps{};primes(ps);p::Plan plan{};assert(p::plan_codec(plan,p::plan_T_max()));plan.C=64;plan.rowscale=1;plan.Lg=1;
    for(unsigned q=0;q<p::NP;++q){p::cc scale[33];const uint64_t mod=p::PR[q];p::inverse_powers_build(scale,mod);
        for(unsigned k=0;k<=32;++k){const auto s=scale[k];assert(p::mulm(s.c,uint64_t(1)<<k,mod)==1);
            const p::u128 target=p::u128(s.c)<<52;assert(p::u128(s.rec)*mod<=target && p::u128(s.rec+1)*mod>target);}}
    // Independent leaf congruence/range oracle, including A==Y and sum of two REDCs.
    for(int q=0;q<p::NP;++q){const uint64_t mod=p::PR[q];const auto &pv=ps.V[q];
        const uint64_t invR=p::invm(uint64_t((p::u128(1)<<52)%mod),mod);
        for(unsigned trial=0;trial<16;++trial){alignas(64) uint64_t aa[8][8],bb[8][8],got[8][8],ref[8][8];
            const uint64_t kc=trial==0?1:1+rnd()%(mod-1),wr=1+rnd()%(mod-1),wi=p::invm(wr,mod);
            for(unsigned j=0;j<8;++j)for(unsigned l=0;l<8;++l){aa[j][l]=trial==0?4*mod-1:rnd()%(4*mod);bb[j][l]=rnd()%(4*mod);}
            for(unsigned mid=0;mid<2;++mid){memcpy(got,bb,sizeof got);
                if(mid)p::convLT<8,true>((V*)got,1,(V*)aa,1,p::vset(wr),p::vset(p::cc_of(wr,mod).rec),p::vset(wi),p::vset(p::cc_of(wi,mod).rec),pv,p::vset(kc),p::vset(p::cc_of(kc,mod).rec));
                else p::convL<8,true>((V*)got,1,(V*)aa,1,p::vset(wr),p::vset(p::cc_of(wr,mod).rec),pv,p::vset(kc),p::vset(p::cc_of(kc,mod).rec));
                for(unsigned k=0;k<8;++k)for(unsigned l=0;l<8;++l){uint64_t sum=0;
                    for(unsigned j=0;j<8;++j){unsigned ix=(k+8-j)%8;uint64_t a=aa[mid && ix?8-ix:ix][l]%mod;
                        if(mid && ix)a=p::mulm(a,wr,mod);if(k<j)a=p::mulm(a,mid?wi:wr,mod);
                        sum=(sum+p::mulm(a,bb[j][l]%mod,mod))%mod;}
                    ref[k][l]=p::mulm(p::mulm(sum,kc,mod),invR,mod);
                    assert(got[k][l]<2*mod && got[k][l]%mod==ref[k][l]);
                }
            }
            memcpy(got,aa,sizeof got);p::convL<8,true>((V*)got,1,(V*)got,1,p::vset(wr),p::vset(p::cc_of(wr,mod).rec),pv,p::vset(kc),p::vset(p::cc_of(kc,mod).rec));
            memcpy(ref,aa,sizeof ref);p::convL<8,true>((V*)ref,1,(V*)aa,1,p::vset(wr),p::vset(p::cc_of(wr,mod).rec),pv,p::vset(kc),p::vset(p::cc_of(kc,mod).rec));
            assert(!memcmp(got,ref,sizeof got));
        }
    }
    p::Garner g1{},g2{};g1.init(ps,plan,1);g2.init(ps,plan,2);
    p::EmitK k1{},k2{};k1.init(g1,ps,plan.T);k2.init(g2,ps,plan.T);k1.nosc=k2.nosc=1;
    uint64_t inverses[p::NP];for(int q=0;q<p::NP;++q)inverses[q]=p::cofactor_inverse(q);
    alignas(64) V a[nv*p::NP],b[nv*p::NP],out[nv*p::NC],expected[nv*p::NC];
    ref_int P,X,tmp,lim;ref_inits(P,X,tmp,lim,nullptr);ref_set_ui(P,1);for(int q=0;q<p::NP;++q)ref_mul_ui(P,P,p::PR[q]);ref_fdiv_q_2exp(lim,P,2);
    for(unsigned iter=0;iter<24;++iter){
        for(size_t v=0;v<nv;++v)for(unsigned l=0;l<8;++l){
            uint64_t bits[8];for(auto &w:bits)w=rnd();ref_import(X,8,-1,8,0,0,bits);ref_mod(X,X,iter<16?P:lim);
            if(iter==0){switch(l){case 0:ref_set_ui(X,0);break;case 1:ref_set_ui(X,1);break;case 2:ref_sub_ui(X,P,1);break;case 3:ref_sub_ui(X,P,2);break;
                default:ref_fdiv_q_2exp(X,P,1);if(l&1)ref_add_ui(X,X,1);else ref_sub_ui(X,X,1);}}
            if(iter==1 && l==0){ref_set_ui(X,0);for(int q=0;q<p::NP;++q){ref_divexact_ui(tmp,P,p::PR[q]);ref_sub(X,X,tmp);}ref_mod(X,X,P);}
            for(int q=0;q<p::NP;++q){const auto x=ref_fdiv_ui(X,p::PR[q]);
                reinterpret_cast<uint64_t *>(a+v*p::NP+q)[l]=x+(rnd()%4)*p::PR[q];
                reinterpret_cast<uint64_t *>(b+v*p::NP+q)[l]=p::mulm(x,inverses[q],p::PR[q])+((iter==1 && l==0)?3:rnd()%4)*p::PR[q];}
            ref_set(tmp,X);for(int j=0;j<p::NC;++j){reinterpret_cast<uint64_t *>(expected+v*p::NC+j)[l]=ref_get_ui(tmp)&p::M52;ref_fdiv_q_2exp(tmp,tmp,52);}
        }
        // MXCSR rounding modes must not invalidate the quotient correction.
        const unsigned saved=_mm_getcsr();_mm_setcsr((saved&~_MM_ROUND_MASK)|((iter%4)<<13));
        crt(out,b,k2,true);assert(!memcmp(out,expected,sizeof out));_mm_setcsr(saved);
        crt(out,a,k1,false);assert(!memcmp(out,expected,sizeof out));
    }
    if(timing)for(unsigned rep=0;rep<9;++rep)for(unsigned t=0;t<2;++t){const unsigned mode=(rep+t)&1;const double before=probe_cool();const uint64_t start=tick();
        for(unsigned n=0;n<1500;++n){crt(out,mode?b:a,mode?k2:k1,mode);asm volatile("":::"memory");}
        const auto elapsed=tick()-start;const auto after=probe_temp();
        printf("{\"temp_before\":%.3f,\"temp_after\":%.3f,\"kind\":\"crt_probe\",\"np\":%d,\"mode\":%u,\"round\":%u,\"tsc_per_vector\":%.4f}\n",before,after,p::NP,mode,rep,double(elapsed)/(1500*nv));}
    // Every candidate T, both narrow phases, folded and ordinary decode.
    alignas(64) uint64_t input[nv*32];for(auto &x:input)x=rnd();
    alignas(64) V decoded[nv*p::NP],other[nv*p::NP];
    for(int T=p::plan_T_max();T>=p::plan_T_min();T-=p::plan_T_step()){
        assert(p::plan_codec(plan,T));uint64_t scale[p::NP];for(auto &s:scale)s=rnd();
        for(unsigned folded=0;folded<2;++folded){p::Dec d1{},d2{};p::dec_init(d1,input,plan,ps,folded?scale:nullptr);p::dec_init(d2,input,plan,ps,folded?scale:nullptr);WordFunnelReference ref(T<=184?T:184);
            codec(decoded,input,d1);codec(other,input,d2,&ref);assert(!memcmp(decoded,other,sizeof decoded));
            for(int q=0;q<p::NP;++q)for(size_t v=0;v<nv;++v)for(unsigned l=0;l<8;++l){
                ref_import(X,32,-1,8,0,0,input+v*32);ref_fdiv_q_2exp(X,X,l*T+(T%8?32*(v&1):0));ref_fdiv_r_2exp(X,X,T);
                if(folded && T>88)ref_mul_ui(X,X,scale[q]%p::PR[q]);
                const auto got=reinterpret_cast<uint64_t *>(decoded+q*nv+v)[l];assert(got<4*p::PR[q]);assert(got%p::PR[q]==ref_fdiv_ui(X,p::PR[q]));
            }
            if(timing && folded && T>88 && T<=184)for(unsigned rep=0;rep<7;++rep)for(unsigned t=0;t<2;++t){const unsigned mode=(rep+t)&1;const double before=probe_cool();const uint64_t start=tick();
                for(unsigned n=0;n<700;++n){codec(other,input,mode?d2:d1,mode?&ref:nullptr);asm volatile("":::"memory");}
                const auto elapsed=tick()-start;const auto after=probe_temp();
                printf("{\"temp_before\":%.3f,\"temp_after\":%.3f,\"kind\":\"codec_probe\",\"np\":%d,\"T\":%d,\"mode\":%u,\"round\":%u,\"tsc_per_vector_prime\":%.4f}\n",before,after,p::NP,T,mode+1,rep,double(elapsed)/(700*nv*p::NP));}
        }
    }
#if CR_NP>8
    // Complete [0,P) digit extraction and four-word Pack8, independent reference/modular oracle oracle.
    auto digits_gate=[&]<int WD>(const p::EmitK &ek,int T){
        V x[p::NP];for(int q=0;q<p::NP;++q)x[q]=b[q];
        const auto dg=p::digits_of<WD>(ek,x);
        for(unsigned l=0;l<8;++l){
            ref_set_ui(X,0);for(int j=p::NC-1;j>=0;--j){ref_mul_2exp(X,X,52);ref_add_ui(X,X,reinterpret_cast<uint64_t *>(expected+j)[l]);}
            for(int digit=0;digit<2;++digit){ref_fdiv_q_2exp(tmp,X,digit*T);ref_fdiv_r_2exp(tmp,tmp,T);
                for(int w=0;w<WD;++w){alignas(64) uint64_t values[8];_mm512_store_si512(values,digit?dg.b1[w]:dg.b0[w]);assert(values[l]==ref_get_ui(tmp));ref_fdiv_q_2exp(tmp,tmp,64);}}
            ref_fdiv_q_2exp(tmp,X,2*T);alignas(64) uint64_t top[8];_mm512_store_si512(top,dg.b2);assert(top[l]==ref_get_ui(tmp));
        }
        p::Pack8 pack;pack.init(T,WD);alignas(64) uint8_t bytes[320]{};p::V words[p::WDMAX]{};for(int w=0;w<WD;++w)words[w]=dg.b0[w];pack.run(bytes,words);
        ref_set_ui(X,0);for(int l=7;l>=0;--l){ref_mul_2exp(X,X,T);ref_set_ui(tmp,0);
            for(int w=WD-1;w>=0;--w){alignas(64) uint64_t values[8];_mm512_store_si512(values,dg.b0[w]);ref_mul_2exp(tmp,tmp,64);ref_add_ui(tmp,tmp,values[l]);}ref_add(X,X,tmp);}
        ref_import(tmp,T,-1,1,0,0,bytes);assert(ref_cmp(tmp,X)==0);
    };
    for(int T=p::plan_T_max();T>=p::plan_T_min();T-=p::plan_T_step()){
        assert(p::plan_codec(plan,T));g2.init(ps,plan,2);k2.init(g2,ps,T);k2.nosc=1;
        for(int edges=0;edges<8;++edges){
            for(unsigned l=0;l<8;++l){
                uint64_t bits[8];for(auto &w:bits)w=rnd();ref_import(X,8,-1,8,0,0,bits);ref_mod(X,X,P);
                if(edges==0){if(l<2)ref_set_ui(X,l);else if(l<4)ref_sub_ui(X,P,l-1);else{ref_fdiv_q_2exp(X,P,1);if(l&1)ref_add_ui(X,X,1);}}
                for(int q=0;q<p::NP;++q)reinterpret_cast<uint64_t *>(b+q)[l]=p::mulm(ref_fdiv_ui(X,p::PR[q]),inverses[q],p::PR[q])+(rnd()%4)*p::PR[q];
                ref_set(tmp,X);for(int j=0;j<p::NC;++j){reinterpret_cast<uint64_t *>(expected+j)[l]=ref_get_ui(tmp)&p::M52;ref_fdiv_q_2exp(tmp,tmp,52);}
            }
            if((T+63)/64==4)digits_gate.template operator()<4>(k2,T);else digits_gate.template operator()<3>(k2,T);
        }
    }
#endif
    ref_clears(P,X,tmp,lim,nullptr);printf("{\"kind\":\"endpoint_gate\",\"np\":%d,\"crt_coefficients\":12288,\"gate\":true}\n",p::NP);fflush(stdout);
}
