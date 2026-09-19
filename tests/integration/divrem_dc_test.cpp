#include "divrem_cases.hpp"
int main(){
    unsigned count=0;
    for(size_t dn:{17u,18u,26u,31u,32u,39u,52u,65u,78u,79u,80u,96u,127u,128u,257u,513u,1024u,2049u,4097u,8191u,16385u}){
        for(size_t nn:{size_t(0),dn-1,dn,dn+1,dn+7,2*dn-1,2*dn,2*dn+1,4*dn+3}){
            Fixture f(count%7==0?3:1,false);
            Service service(f,nn,dn,0,0,0,SBN3_DIVREM_DC);
            assert(service.info.algorithm==SBN3_DIVREM_DC && service.info.workers==1);
            for(const auto &d:divisors(dn)){
                service.prepare(d);
                auto values=dn<=257?numerators(d,nn):std::vector<std::vector<uint64_t>>{};
                if(dn>257){
                    std::vector<uint64_t> n(nn);for(auto &x:n)x=random_word();values.push_back(n);
                    std::fill(n.begin(),n.end(),UINT64_MAX);values.push_back(n);
                    std::fill(n.begin(),n.end(),0);values.push_back(n);
                    if(nn>=dn){std::copy(d.begin(),d.end(),n.begin());values.push_back(n);}
                }
                for(const auto &n:values){
                    std::vector<uint64_t> q(nn>=dn?nn-dn+1:0,UINT64_MAX),r(dn,UINT64_MAX);
                    auto result=service.execute(n,q,r);
                    assert(result.quotient_limbs==trim(q.data(),q.size()) && result.remainder_limbs==trim(r.data(),r.size()));
                    certify(n.data(),nn,d.data(),dn,q.data(),result.quotient_limbs,r.data(),result.remainder_limbs,dn<=513);
                    std::vector<uint64_t> qq(q.size()),rr(dn);
                    sbn3_int Q{qq.data(),qq.size(),0,0},R{rr.data(),rr.size(),0,0};
                    allocation_watch_start();
                    sbn3_int_divrem_execute(service.bound,&Q,&R,{n.data(),nn,1},{d.data(),dn,0});
                    assert(!allocation_watch_stop() && Q.size==result.quotient_limbs && R.size==result.remainder_limbs);
                    assert(Q.negative==(Q.size!=0) && R.negative==(R.size!=0));
                    assert(!memcmp(q.data(),qq.data(),q.size()*8) && !memcmp(r.data(),rr.data(),r.size()*8));
                    ++count;
                }
            }
            service.close();service.bind();
        }
    }
    printf("D&C division: %u cases, irregular blocks/high quotient, exact remainders, zero normalization, signed values, allocation/budget/rebind PASS\n",count);
}
