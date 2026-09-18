#pragma once
#include "product_support.hpp"
#include "algorithms/inverse_program.hpp"
#include "algorithms/divide_terminal.hpp"
// Test/client-side preparation; no allocation occurs in the library's execute.
struct PreparedInverse {
    Fixture &f;std::vector<sbn::v3::InverseRung> steps;std::vector<sbn3_spectrum *> handles;
    sbn::v3::InverseProgram program{};
    PreparedInverse(Fixture &fixture,size_t target):f(fixture){
        std::vector<size_t> sizes;size_t m=target;while(m>15){sizes.push_back(m);m=m/2+1;}
        program.target=target;program.seed_limbs=m;steps.reserve(sizes.size());
        for(size_t j=sizes.size();j-- >0;){const size_t n=sizes[j];
            sbn3_mul_options o{};o.workers=sbn3_team_workers(f.team);o.prime_count=6;o.borrow_output=1;o.algorithm=n<=8192?SBN3_MUL_FLAT:SBN3_MUL_BAILEY;
            sbn3_product_request req{};req.a_limbs=m+1;req.b_limbs=n;sbn3_mul_plan plan{};sbn3_product_info info{};bool found=false;
            for(int T:{136,128,120,112}){size_t ring=2*size_t(T);while(ring<n+4)ring*=2;o.trunk_bits=T;req.cyclic_limbs=ring;
                if(sbn3_product_query(&req,&o,&plan,&info)==SBN3_SUPPORTED){found=true;break;}}
            assert(found);sbn3_spectrum_desc future{};assert(sbn3_spectrum_query(&plan,SBN3_SPECTRUM_COLUMNS,n,&future)==SBN3_SUPPORTED);
            const auto first=f.leases.size();auto *producer=f.product(req,o,info,plan);auto storage=f.allocate(future.storage_bytes,64);sbn3_spectrum *cache=nullptr;
            sbn3_spectrum_reserve(producer,SBN3_SPECTRUM_COLUMNS,n,f.arena,&storage,&cache);req.cached_a[0]=&future;
            auto *product=f.product(req,o,info,plan,cache);f.unbind(producer);f.retire(first+1);f.retire(first);
            handles.push_back(cache);steps.push_back({m,n,req.cyclic_limbs,product,cache,f.guarded(req.cyclic_limbs),f.guarded(req.cyclic_limbs)});m=n;
        }
        program.rung_count=steps.size();program.rungs=steps.data();program.values[0]=f.guarded(target+1);program.values[1]=f.guarded(target+1);
    }
    ~PreparedInverse(){for(auto *h:handles)sbn3_spectrum_release(h);}
    void execute(const uint64_t *D,uint64_t *U){sbn::v3::inverse_program(program,D,U);}
};
struct PreparedDivision {
    sbn::v3::DivideTerminal terminal{};
    PreparedDivision(Fixture &f,size_t m,size_t n){
        sbn3_mul_options o{};o.workers=sbn3_team_workers(f.team);o.prime_count=6;o.borrow_output=1;o.algorithm=n<=8192?SBN3_MUL_FLAT:SBN3_MUL_BAILEY;
        sbn3_product_request req{};req.a_limbs=m+1;req.b_limbs=n+1;sbn3_mul_plan plan{};sbn3_product_info info{};bool found=false;
        for(int T:{136,128,120,112}){size_t ring=2*size_t(T);while(ring<2*m+4)ring*=2;o.trunk_bits=T;req.cyclic_limbs=ring;
            if(sbn3_product_query(&req,&o,&plan,&info)==SBN3_SUPPORTED){found=true;break;}}
        assert(found);sbn3_spectrum_desc future{};assert(sbn3_spectrum_query(&plan,SBN3_SPECTRUM_COLUMNS,n,&future)==SBN3_SUPPORTED);
        const auto first=f.leases.size();auto *producer=f.product(req,o,info,plan);auto storage=f.allocate(future.storage_bytes,64);sbn3_spectrum *cache=nullptr;
        sbn3_spectrum_reserve(producer,SBN3_SPECTRUM_COLUMNS,n,f.arena,&storage,&cache);req.cached_a[0]=&future;
        auto *cached=f.product(req,o,info,plan,cache);req.cached_a[0]=nullptr;req.b_limbs=n;auto *residual=f.product(req,o,info,plan);
        f.unbind(producer);f.retire(first+1);f.retire(first);
        terminal={m,n,req.cyclic_limbs,cached,residual,cache,{f.guarded(req.cyclic_limbs),f.guarded(req.cyclic_limbs)}};
    }
    ~PreparedDivision(){sbn3_spectrum_release(terminal.u);}
    void execute(const uint64_t *A,const uint64_t *D,const uint64_t *U,uint64_t *Q){sbn::v3::divide_terminal(terminal,A,D,U,Q);}
};
