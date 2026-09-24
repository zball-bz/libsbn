#include "divrem_support.hpp"
#include "runtime/arena.hpp"
#include "algorithms/divrem_prepared.hpp"

int main() {
    size_t cases=0;
    for(size_t dn:{513u,32768u})for(unsigned reuse:{0u,8u}){
        Fixture f(1,false);Service service(f,32*dn,dn,0,reuse);
        sbn3_divrem_request short_request{8*dn,dn};sbn3_divrem_options options{};
        options.workers=1;options.reuse_hint=reuse;options.timing=1;
        sbn3_divrem_plan plan;sbn3_divrem_info info;
        assert(sbn3_divrem_query(&short_request,&options,&plan,&info)==SBN3_SUPPORTED);
        // Both 8n/n and 32n/n use the long-quotient recipe: more blocks,
        // never a whole-numerator u52 copy or a longer inverse. A short
        // fresh call may choose a smaller reciprocal to reduce preparation.
        assert(info.storage_bytes==service.info.storage_bytes && info.block_limbs==service.info.block_limbs);
        short_request.numerator_limbs=4*dn;
        assert(sbn3_divrem_query(&short_request,&options,&plan,&info)==SBN3_SUPPORTED);
        assert(info.storage_bytes<=service.info.storage_bytes&&info.block_limbs<=service.info.block_limbs);
        std::vector<uint64_t>d(dn);for(auto&x:d)x=random_word();d.back()|=uint64_t(1)<<63;
        service.prepare(d);
        for(size_t nn:{dn+7,32*dn}){
            std::vector<uint64_t>n(nn),q(nn-dn+1),r(dn);for(auto&x:n)x=random_word();
            const auto result=service.execute(n,q,r);
            certify(n.data(),nn,d.data(),dn,q.data(),result.quotient_limbs,r.data(),result.remainder_limbs,true);
            ++cases;
        }
    }
    for(bool direct:{false,true})for(unsigned timed:{0u,1u})for(size_t dn:{1u,2u,3u,8u,31u,64u,127u,512u,1024u,4096u,16385u,32768u}) {
        for(unsigned algorithm:{0u,unsigned(SBN3_DIVREM_SCHOOLBOOK),unsigned(SBN3_DIVREM_DC)}) {
            if((algorithm==SBN3_DIVREM_DC&&dn<3)||(algorithm==SBN3_DIVREM_SCHOOLBOOK&&dn>127)||(algorithm&&dn>4096))continue;
            Fixture f(1,false);
            Service service(f,4*dn,dn,0,0,0,algorithm);
            assert(service.info.control_bytes<=1024);
            service.close();
            const size_t available=service.info.storage_bytes;
            const sbn3_divrem_request request{4*dn,dn};
            sbn3_divrem_options options{};options.workers=1;options.algorithm=algorithm;options.timing=timed;
            const auto *configuration=!timed&&!algorithm?nullptr:&options;
            sbn3_divrem_info retained_info;
            assert(sbn3_divrem_query(&request,configuration,&service.plan,&retained_info)==SBN3_SUPPORTED);
            if(!configuration){
                sbn3_divrem_plan explicit_plan;sbn3_divrem_info explicit_info;
                assert(sbn3_divrem_query(&request,&options,&explicit_plan,&explicit_info)==SBN3_SUPPORTED);
                assert(explicit_info.plan_id==retained_info.plan_id&&explicit_info.storage_bytes==retained_info.storage_bytes&&
                       explicit_info.algorithm==retained_info.algorithm&&explicit_info.block_limbs==retained_info.block_limbs);
            }
            assert(retained_info.storage_bytes<=available);service.info=retained_info;
            std::memset(f.arena->base+service.offset,0xa5,available);
            service.bind();
            unsigned preparations=0;
            auto *retained=f.arena->base+service.offset+retained_info.control_bytes;
            for(unsigned shift:{0u,1u,31u,63u}) {
                std::vector<uint64_t>d(dn);
                for(auto &x:d)x=random_word();
                d.back()=(d.back()>>shift)|(uint64_t(1)<<(63-shift));
                const auto saved=d;
                if(direct){
                    service.close();
                    allocation_watch_start();
                    service.bound=sbn3_divrem_prepare_into(&request,configuration,f.arena,service.offset,
                                                         service.info.storage_bytes,f.team,{d.data(),dn});
                    assert(!allocation_watch_stop());
                }else service.prepare(d);
                ++preparations;
                // No execution may depend on the original D, or mutate the
                // retained normalized words, inverse, tables or spectra.
                std::fill(d.begin(),d.end(),0);
                std::vector<unsigned char> immutable(retained,retained+retained_info.persistent_bytes);
                for(size_t nn:{size_t(0),dn-1,dn+1,4*dn}) {
                    const size_t qn=nn>=dn?nn-dn+1:0;
                    std::vector<uint64_t>n(nn),q(qn),r(dn);
                    for(auto &x:n)x=random_word();
                    const auto result=service.execute(n,q,r);
                    assert(result.quotient_limbs==trim(q.data(),qn)&&result.remainder_limbs==trim(r.data(),dn));
                    certify(n.data(),nn,saved.data(),dn,q.data(),result.quotient_limbs,r.data(),result.remainder_limbs,true);
                    assert(immutable.empty()||!memcmp(retained,immutable.data(),immutable.size()));
                    sbn3_int Q{q.data(),q.size(),0,0},R{r.data(),r.size(),0,0};
                    allocation_watch_start();
                    sbn3_int_divrem_execute(service.bound,&Q,&R,{n.data(),nn,1},{saved.data(),dn,1});
                    assert(!allocation_watch_stop()&&!Q.negative&&R.negative==(R.size!=0));
                    assert(immutable.empty()||!memcmp(retained,immutable.data(),immutable.size()));
                    ++cases;
                }
                sbn3_divrem_metrics metrics;
                sbn3_divrem_get_metrics(service.bound,&metrics);
                assert(metrics.prepares==(direct?1u:preparations));
                assert(metrics.executes==8*(direct?1u:preparations));
                if(!timed)assert(!metrics.prepare_ns&&!metrics.execute_ns);
            }
        }
    }
    printf("prepared division: %zu changing numerators, overwritten source D, immutable retained state, re-prepare/signed/zero-allocation PASS\n",cases);
}
