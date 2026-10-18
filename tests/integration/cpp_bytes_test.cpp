// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/bytes.hpp"
#include <array>
#include <cassert>
#include <cstdio>
#include <vector>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::integer;using B=std::byte;
static void check(const integer&x){
    for(auto order:{sbn::byte_order::big,sbn::byte_order::little})for(auto encoding:{sbn::integer_encoding::twos_complement,sbn::integer_encoding::magnitude}){
        const auto count=sbn::export_bytes_size(x.view(),encoding);std::vector<B> buf(count+2,B{0x5a});
        allocation_watch_start();const auto written=sbn::export_bytes({buf.data()+1,count},x.view(),{order,encoding});assert(!allocation_watch_stop());
        assert(written&&written.end==buf.data()+1+count&&buf.front()==B{0x5a}&&buf.back()==B{0x5a});
        auto y=sbn::from_bytes({buf.data()+1,count},{order,encoding});assert(y==(encoding==sbn::integer_encoding::magnitude?sbn::abs(x):x));
        std::vector<std::uint64_t> dst((count+7)/8);sbn::integer_buffer target(dst);
        allocation_watch_start();sbn::import_bytes_into(target.ref(),{buf.data()+1,count},{order,encoding});assert(!allocation_watch_stop());assert(sbn::compare(target.view(),y.view())==0);
        auto saved=buf;const auto too_short=sbn::export_bytes({buf.data()+1,count-1},x.view(),{order,encoding});assert(!too_short&&buf==saved);
        // Add non-minimal sign extension on either end, without changing value.
        const B fill=encoding==sbn::integer_encoding::twos_complement&&x.negative()?B{255}:B{0};std::vector<B> padded(count+33,fill);
        if(order==sbn::byte_order::big)std::copy(buf.begin()+1,buf.end()-1,padded.begin()+33);else std::copy(buf.begin()+1,buf.end()-1,padded.begin());
        assert(sbn::from_bytes(padded,{order,encoding})==y);
    }
}
int main(){
    for(int n=-257;n<=257;++n)check(integer{n});
    const std::array<B,2> a{B{0xff},B{0x7f}},b{B{0x00},B{0x80}};
    assert(sbn::from_bytes(a)==-129&&sbn::from_bytes(b)==128);assert(sbn::from_bytes({})==0);
    for(std::size_t bit:{63u,64u,127u,128u,255u,256u,1023u,8192u}){integer n=1;n<<=bit;check(n);check(-n);check(n+1);check(-n-1);check(n-1);check(-n+1);}
    std::uint64_t seed=9821673;
    for(std::size_t size:{1u,7u,8u,9u,17u,65u,257u,1025u}){std::vector<B> bytes(size);
        for(auto&v:bytes){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;v=B(seed);}
        for(auto order:{sbn::byte_order::big,sbn::byte_order::little}){
            auto n=sbn::from_bytes(bytes,{order,sbn::integer_encoding::magnitude});
            // Independent base-256 Horner interpretation using owning arithmetic.
            integer expected=0;for(std::size_t j=0;j<size;++j){expected<<=8;expected+=std::to_integer<unsigned>(bytes[order==sbn::byte_order::big?j:size-1-j]);}assert(n==expected);check(n);check(-n);
        }
    }
    puts("C++ portable bytes: signs/endian/canonical padding/exact boundaries/capacity/allocation-free into PASS");
}
