// Original libsbn example; LGPL-3.0-or-later.
#include <sbn3/sbn.hpp>
#include <cstdio>
int main(){
    auto parsed=sbn::parse_integer("123456789012345678901234567890");
    if(!parsed){std::fprintf(stderr,"invalid input at byte %zu\n",parsed.error.index);return 1;}
    const sbn::integer divisor=97;
    const auto cube=sbn::pow(parsed.value,3);
    const auto [q,r]=sbn::divrem(cube,divisor);
    auto text=sbn::to_string(q),remainder=sbn::to_string(r);
    std::printf("quotient: %s\nremainder: %s\n",text.c_str(),remainder.c_str());

    // A retained divisor owns its normalization/inverse, independent of the
    // original value. The same handle may execute many bounded numerators.
    sbn::prepared_divisor prepared(divisor,cube.size(),100);
    const auto check=prepared.divrem(cube);
    if(check.quotient!=q||check.remainder!=r)return 2;

    // A strict call uses explicitly provisioned output and operation scratch.
    sbn::integer out;out.reserve(parsed.value.size()+divisor.size());
    const auto need=sbn::multiply_requirements(out.ref(),parsed.value.view(),divisor.view());
    if(!need.supported())return 3;
    sbn::workspace scratch;scratch.reserve(need.scratch_bytes,need.scratch_alignment);
    sbn::mul_into(out.ref(),parsed.value.view(),divisor.view(),scratch.on());
    return out==parsed.value*divisor?0:4;
}
