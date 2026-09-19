#include "product/short_query_bounds.hpp"
#include "sbn3/product.h"
#include <cassert>
#include <cstdio>
int main(){
    unsigned checked=0;
    for(size_t n:{1u,2u,7u,8u,17u,33u,63u,65u,127u,128u,129u,256u,257u,511u,512u,1023u,1024u,2047u,4097u,8192u,16384u,32768u,65536u,131072u,262144u,524288u,1048576u})
        for(unsigned ratio:{1u,3u,7u,64u})for(unsigned workers:{1u,2u,3u,4u,8u,16u,32u})for(int bits:{0,16,17,18,19,20}){
            sbn3_product_spec s{n,std::max<size_t>(1,n/ratio)};
            sbn3_mul_options o{};o.algorithm=SBN3_MUL_PQ16;o.workers=workers;o.trunk_bits=bits;
            sbn3_mul_plan p{};sbn3_mul_info i{};
            if(sbn3_mul_query(&s,&o,&p,&i)!=SBN3_SUPPORTED)continue;
            const double actual=sbn::v3::pq16_product_cost(i,s.a_limbs,s.b_limbs);
            const double lower=sbn::v3::query_bounds::fft_score(s.a_limbs,s.b_limbs,workers);
            if(!(lower<=actual))fprintf(stderr,"FFT floor %zu x %zu W%u bits%d: %.9g > %.9g\n",s.a_limbs,s.b_limbs,workers,bits,lower,actual);
            assert(lower<=actual);++checked;
        }
    printf("FFT score lower bound against %u actual plans, widths/worker counts/rectangular shapes PASS\n",checked);
}
