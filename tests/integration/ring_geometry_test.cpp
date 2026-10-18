#include "sbn3/product.h"
#include "product/backend.hpp"
#include "product/cost_model.hpp"
#include "product/root_prepare_cost.hpp"
#include "tuning/native_policy.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using namespace sbn::v3;
void same(const sbn3_mul_info &a,const sbn3_mul_info &b){
#define FIELD(f) assert(a.f==b.f)
    FIELD(np);FIELD(trunk_bits);FIELD(digit_words);FIELD(workers);FIELD(prime_batch);
    FIELD(fused);FIELD(row_major);FIELD(borrow_output);FIELD(full);FIELD(C);FIELD(M2);FIELD(lbv);FIELD(lbw);
    FIELD(nat);FIELD(nyt);FIELD(transform_trunks);FIELD(output_limbs);FIELD(output_alignment);
    FIELD(table_bytes);FIELD(workspace_bytes);FIELD(workspace_alignment);FIELD(per_worker_bytes);FIELD(plane_pitch);FIELD(pool_bytes);FIELD(transpose_bytes);
    FIELD(table_entries);FIELD(factor_levels);FIELD(root_order_log2);FIELD(block_stride);FIELD(product_row_stride);
    FIELD(emit_trunks);FIELD(emit_limbs);FIELD(emit_quantum_trunks);FIELD(emit_quantum_limbs);FIELD(rowscale);
    FIELD(row_task_grain);FIELD(fused_row_grain);FIELD(format_slot_bytes);FIELD(format_block_slots);FIELD(fused_items);
    FIELD(crt_mode);FIELD(codec_mode);FIELD(algorithm);FIELD(fused_start_skew_us);
#undef FIELD
}
int main(){unsigned checked=0;
    for(unsigned np=4;np<=10;++np)for(int T=24*int(np)-8;T>=24*int(np)-32;T-=8)
        for(unsigned k:{5u,9u,13u,16u})for(unsigned workers:{1u,4u,16u})
            for(unsigned algorithm:{unsigned(SBN3_MUL_FLAT),unsigned(SBN3_MUL_BAILEY)})for(unsigned batch:{0u,1u}){
                if(algorithm==SBN3_MUL_FLAT&&batch)continue;
                const size_t ring=2*size_t(T)*(size_t(1)<<k);
                sbn3_product_request r{};r.kind=SBN3_PRODUCT_MUL;r.a_limbs=ring/3+1;r.b_limbs=ring-17;r.cyclic_limbs=ring;
                sbn3_mul_options o{};o.prime_count=np;o.workers=workers;o.algorithm=algorithm;o.trunk_bits=T;o.borrow_output=2;o.prime_batch=batch;
                sbn3_mul_plan producer,consumer;sbn3_product_info pi{},ci{};sbn3_mul_info quoted{};sbn3_spectrum_desc future{};
                const auto *backend=backend_lookup(np);assert(backend&&backend->geometry_query_cached_a);
                allocation_watch_start();
                const auto qr=backend->geometry_query_cached_a(r,o,quoted);
                auto rc=sbn3_product_query(&r,&o,&producer,&pi);
                if(rc==SBN3_SUPPORTED)rc=sbn3_spectrum_query(&producer,SBN3_SPECTRUM_COLUMNS,7,&future);
                if(rc==SBN3_SUPPORTED){r.cached_a[0]=&future;rc=sbn3_product_query(&r,&o,&consumer,&ci);}
                assert(!allocation_watch_stop());
                assert(qr==rc);if(rc!=SBN3_SUPPORTED)continue;
                same(quoted,ci.mul);
                for(bool deep:{false,true})assert(cost_model::cyclic_product(quoted,deep).nanoseconds==cost_model::cyclic_product(ci.mul,deep).nanoseconds);
                assert(root_prepare_cost(quoted)==root_prepare_cost(pi.mul));++checked;
            }
    printf("future cached geometry: %u producer/consumer plans, all resource/cost fields identical, allocation-free PASS\n",checked);
}
