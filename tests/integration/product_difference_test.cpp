#include "product_support.hpp"
#include "product/difference.hpp"
using namespace sbn::v3;
int main(){
    constexpr size_t dn=17,nn=23,capacity=32;constexpr uint64_t guard=0x19fab37c08de6241;
    ref_int N,H,X,D,Q,P,E,M,Z,T;ref_inits(N,H,X,D,Q,P,E,M,Z,T,nullptr);
    unsigned checked=0;
    for(unsigned shift:{0u,1u,31u,63u})for(bool split:{false,true})for(unsigned pattern=0;pattern<3;++pattern){
        uint64_t input[nn],high[dn],divisor[dn];
        for(auto &x:input)x=pattern==0?UINT64_MAX:random_word();
        for(auto &x:high)x=pattern==1?UINT64_MAX:random_word();
        for(auto &x:divisor)x=random_word();divisor[dn-1]|=uint64_t(1)<<63;
        const size_t origin=split?1:0,lo=split?5:nn+1;
        const product::DifferenceValue value{{input,nn},origin,lo,shift,split?product::Span{high,dn}:product::Span{}};
        ref_import(N,nn,-1,8,0,0,input);ref_mul_2exp(X,N,shift);ref_fdiv_q_2exp(X,X,64*origin);
        ref_fdiv_r_2exp(X,X,64*lo);
        if(split){ref_import(H,dn,-1,8,0,0,high);ref_mul_2exp(H,H,64*lo);ref_add(X,X,H);}
        ref_import(D,dn,-1,8,0,0,divisor);ref_fdiv_q(Q,X,D);
        for(int delta:{-3,-1,0,1,3}){
            ref_set(T,Q);if(delta<0)ref_sub_ui(T,T,unsigned(-delta));else ref_add_ui(T,T,unsigned(delta));
            ref_mul(P,T,D);ref_sub(E,X,P);
            uint64_t low=0;ref_fdiv_r_2exp(Z,P,64);ref_export(&low,nullptr,-1,8,0,0,Z);
            const bool negative=ref_sgn(E)<0;ref_abs(E,E);
            uint64_t expected[capacity]{};ref_export(expected,nullptr,-1,8,0,0,E);
            for(size_t period:{0ul,dn,dn+1,dn+7}){
                uint64_t data[capacity+2],fold[capacity+2];std::fill(std::begin(data),std::end(data),guard);std::fill(std::begin(fold),std::end(fold),guard);
                memset(data+1,0,capacity*8);
                if(period){ref_set_ui(M,1);ref_mul_2exp(M,M,64*period);ref_sub_ui(M,M,1);ref_mod(Z,P,M);}
                else ref_set(Z,P);
                ref_export(data+1,nullptr,-1,8,0,0,Z);
                allocation_watch_start();const auto result=product::product_difference(data+1,capacity,period,value,low,fold+1);
                assert(!allocation_watch_stop()&&result.negative==negative&&!result.error_bits);
                assert(!memcmp(result.data,expected,result.words*8)&&data[0]==guard&&data[capacity+1]==guard&&fold[0]==guard&&fold[capacity+1]==guard);++checked;
            }
        }
    }
    ref_clears(N,H,X,D,Q,P,E,M,Z,T,nullptr);
    printf("normalized/split/linear/cyclic product difference: %u independent cases PASS\n",checked);
}
