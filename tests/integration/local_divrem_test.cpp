#include "divrem_support.hpp"
#include "algorithms/local_divrem.hpp"
#include "product/cyclic_reconstruct.hpp"
#include "backend/u52/kernels.hpp"
#include "runtime/scratch.hpp"
using namespace sbn::v3;
int main(){
    for(size_t an=1;an<=256;an+=7)for(size_t bn=1;bn<=256;bn+=11){
        const size_t r=pq16::wide_cyclic_minimum_words(an,bn),small=std::min(an,bn),large=std::max(an,bn);
        assert(r>=large&&small<9*r/16);
        assert(r-1<large||small>=9*(r-1)/16);
    }
    // Capacity alone would choose 17 bits here, but the cached short
    // operand needs the 18-bit period to satisfy its coefficient support.
    {const auto p=product::local_product_query(4200,2560,true,true);
     assert(p.shape.bits==18&&p.ring==4608&&pq16::cyclic_supported(p.shape,4200,2560));}
    size_t count=0;
    for(size_t dn:{1024u,8193u,32768u})for(unsigned reuse:{0u,8u}){
        const size_t m=dn/4,nn=dn+m;const auto p=local_division_plan_for_block(nn,dn,m,reuse);
        assert(p.quotient.cached==(reuse>1)&&p.residual.cached==(reuse>1));
        std::vector<uint64_t>d(dn),saved,n(nn),q(m+1),r(dn),scratch((p.storage_bytes+7)/8+16);
        for(auto&x:d)x=random_word();d.back()|=uint64_t(1)<<63;saved=d;
        auto prepare=Frame::external(scratch.data(),p.storage_bytes);
        allocation_watch_start();const auto state=local_division_prepare(p,d.data(),prepare);assert(!allocation_watch_stop());
        std::fill(d.begin(),d.end(),0);
        for(unsigned j=0;j<2;++j){
            for(auto&x:n)x=random_word();
            auto work=Frame::external(reinterpret_cast<unsigned char*>(scratch.data())+p.persistent_bytes,p.work_bytes);
            allocation_watch_start();local_division_apply(p,state,q.data(),r.data(),n.data(),nn,work);
            assert(!allocation_watch_stop()&&!work.used()&&work.peak()<=p.work_bytes);
            certify(n.data(),nn,saved.data(),dn,q.data(),trim(q.data(),q.size()),r.data(),trim(r.data(),dn),true);
        }
    }
    for(size_t an:{512u,1024u,8192u})for(size_t bn:{1u,2u,63u,127u,128u,129u,192u,257u,510u})for(unsigned pattern=0;pattern<3;++pattern){
        std::vector<uint64_t>a(an),b(bn),out(bn+2,0x51429763),product(an+bn);
        for(auto&x:a)x=pattern==0?UINT64_MAX:random_word();
        for(auto&x:b)x=pattern==0?UINT64_MAX:random_word();
        if(pattern==1)std::fill(a.begin(),a.begin()+an-bn-2,0);
        const auto saved_a=a,saved_b=b;const size_t bytes=u52::high_prefix_scratch_bytes(bn);
        std::vector<uint64_t>scratch((bytes+7)/8+8,0xb4937852);
        {auto work=Frame::external(scratch.data()+1,bytes);allocation_watch_start();
         u52::high_prefix(out.data()+1,a.data(),an,b.data(),bn,work);
         assert(!allocation_watch_stop()&&work.peak()<=bytes&&!work.used());}
        sbn3_mul_basecase(product.data(),product.size(),a.data(),an,b.data(),bn);
        uint64_t borrow=0;
        for(size_t j=0;j<bn;++j){
            const unsigned __int128 sub=(unsigned __int128)out[j+1]+borrow;
            const uint64_t delta=product[an+j]-uint64_t(sub);borrow=(unsigned __int128)product[an+j]<sub;
            assert(j?delta==0:delta<=1);
        }
        assert(!borrow&&a==saved_a&&b==saved_b&&out.front()==0x51429763&&out.back()==0x51429763);
        assert(scratch.front()==0xb4937852&&scratch.back()==0xb4937852);
    }
    // Independent 128-bit oracle, including both representations of zero,
    // either sign and several whole-modulus multiples. This is precisely the
    // information that centered cyclic recovery cannot recover at a full ring.
    const __int128 modulus=static_cast<__int128>(UINT64_MAX);
    for(int wraps=-8;wraps<=8;++wraps)
      for(uint64_t residue:{uint64_t(0),uint64_t(1),uint64_t(8),uint64_t(1)<<63,UINT64_MAX-8,UINT64_MAX-1,UINT64_MAX}){
        const __int128 exact=residue+wraps*modulus,mag=exact<0?-exact:exact;
        uint64_t words[3]={residue,0,0x87654321};
        const uint64_t low[1]={static_cast<uint64_t>(exact)};
        const bool negative=reconstruct_cyclic(words,1,low);
        assert(negative==(exact<0)&&words[0]==static_cast<uint64_t>(mag)&&
               words[1]==static_cast<uint64_t>(mag>>64)&&words[2]==0x87654321);
      }
    constexpr uint64_t guard=0x1357aacf752194e8ULL;
    for(size_t dn:{3u,4u,7u,15u,16u,31u,32u,63u,64u,65u,127u,128u,129u,255u,256u,257u,319u,320u,383u,384u,447u,448u,510u,511u,512u,513u,640u,768u,769u,896u,1023u,1024u,1025u,2048u,4096u,4097u,8192u,16384u,16385u,22529u,32768u})
      for(unsigned pieces:{1u,2u,3u}){
        const auto plan=local_division_plan(4*dn,dn,pieces);
        if(plan.quotient.shape.nfull)assert(u52::high_prefix_scratch_bytes(std::min<size_t>(192,(plan.block_limbs+4)/2))<=plan.quotient.work_bytes);
        const size_t offset=dn%8+1,words=(plan.storage_bytes+7)/8;
        std::vector<uint64_t> scratch(words+offset+8,guard);
        for(size_t nn:{size_t(0),dn-1,dn,dn+1,dn+7,2*dn-1,2*dn,2*dn+1,2*dn+2,2*dn+3,2*dn+4,4*dn}){
            const size_t qn=nn>=dn?nn-dn+1:0;
            for(unsigned pattern=0;pattern<5;++pattern){
                std::vector<uint64_t> d(dn),n(nn),q(qn+2,guard),r(dn+2,guard);
                for(auto&x:d)x=random_word();for(auto&x:n)x=random_word();
                d.back()|=uint64_t(1)<<63;
                if(pattern==1)d.back()=(d.back()>>(1+dn%63))|1;
                if(pattern==2){std::fill(d.begin(),d.end(),0);d.back()=uint64_t(1)<<63;}
                if(pattern==3){std::fill(d.begin(),d.end(),UINT64_MAX);std::fill(n.begin(),n.end(),UINT64_MAX);}
                if(pattern==4)std::fill(n.begin(),n.end(),0);
                const auto nd=n,dd=d;
                {auto frame=Frame::external(scratch.data()+offset,plan.storage_bytes);allocation_watch_start();
                 local_division_execute(plan,q.data()+1,r.data()+1,n.data(),nn,d.data(),frame);
                 assert(!allocation_watch_stop()&&!frame.used()&&frame.peak()<=plan.storage_bytes);}
                certify(n.data(),nn,d.data(),dn,q.data()+1,trim(q.data()+1,qn),r.data()+1,trim(r.data()+1,dn),dn<=513||dn>16384);
                assert(n==nd&&d==dd&&q.front()==guard&&q.back()==guard&&r.front()==guard&&r.back()==guard);
                for(size_t j=0;j<offset;++j)assert(scratch[j]==guard);
                for(size_t j=offset+words;j<scratch.size();++j)assert(scratch[j]==guard);
                ++count;
            }
        }
      }
    printf("local Barrett: %zu cases, fixed plan with shorter/zero numerators, exact/residue certificates, readonly values, scratch offsets/guards and zero allocations PASS\n",count);
}
