#include "backend/pq16/root_bank.hpp"
namespace sbn::v3::pq16::root_bank {
template<unsigned Lg,bool Radix8>
alignas(128) constinit const Stage<Lg> stage=make_stage<Lg,Radix8>();
#define SBN3_ROOT_STAGE_DEFINE(LG) \
    template const Stage<LG> stage<LG,false>; \
    template const Stage<LG> stage<LG,true>;
SBN3_ROOT_STAGE_DEFINE(6) SBN3_ROOT_STAGE_DEFINE(7) SBN3_ROOT_STAGE_DEFINE(8)
SBN3_ROOT_STAGE_DEFINE(9) SBN3_ROOT_STAGE_DEFINE(10) SBN3_ROOT_STAGE_DEFINE(11)
SBN3_ROOT_STAGE_DEFINE(12) SBN3_ROOT_STAGE_DEFINE(13) SBN3_ROOT_STAGE_DEFINE(14)
SBN3_ROOT_STAGE_DEFINE(15) SBN3_ROOT_STAGE_DEFINE(16) SBN3_ROOT_STAGE_DEFINE(17)
#undef SBN3_ROOT_STAGE_DEFINE
template const Stage<18> stage<18,false>;
alignas(128) constinit const std::array<double,pq_branch/2> pq=make_pq();
template<unsigned M,unsigned Lg>
alignas(128) constinit const std::array<double,2*M*(1u<<Lg)> rac=make_rac<M,Lg>();
#define SBN3_RAC_DEFINE(M,L) template const std::array<double,2*M*(1u<<L)> rac<M,L>;
#define SBN3_RAC_DEFINE_BAND(M) SBN3_RAC_DEFINE(M,6) SBN3_RAC_DEFINE(M,7) SBN3_RAC_DEFINE(M,8) SBN3_RAC_DEFINE(M,9) SBN3_RAC_DEFINE(M,10) SBN3_RAC_DEFINE(M,11)
SBN3_RAC_DEFINE_BAND(3) SBN3_RAC_DEFINE(3,12)
SBN3_RAC_DEFINE_BAND(5) SBN3_RAC_DEFINE(5,12)
SBN3_RAC_DEFINE_BAND(7)
#undef SBN3_RAC_DEFINE_BAND
#undef SBN3_RAC_DEFINE
template<unsigned M,unsigned Lg>
alignas(128) constinit const std::array<double,2*(M-1)*(1u<<Lg)> ct=make_ct<M,Lg>();
#define SBN3_CT_DEFINE(M,L) template const std::array<double,2*(M-1)*(1u<<L)> ct<M,L>;
#define SBN3_CT_DEFINE_BAND(M) SBN3_CT_DEFINE(M,7) SBN3_CT_DEFINE(M,8) SBN3_CT_DEFINE(M,9) SBN3_CT_DEFINE(M,10) SBN3_CT_DEFINE(M,11) SBN3_CT_DEFINE(M,12)
SBN3_CT_DEFINE_BAND(3) SBN3_CT_DEFINE_BAND(5) SBN3_CT_DEFINE_BAND(7)
#undef SBN3_CT_DEFINE_BAND
#undef SBN3_CT_DEFINE
template<unsigned M,unsigned Lg>
alignas(128) constinit const decltype(make_ct_fine<M,Lg>()) ct_fine=make_ct_fine<M,Lg>();
#define SBN3_CT_FINE_DEFINE(M,L) template const decltype(make_ct_fine<M,L>()) ct_fine<M,L>;
#define SBN3_CT_FINE_BAND(M) SBN3_CT_FINE_DEFINE(M,13) SBN3_CT_FINE_DEFINE(M,14) SBN3_CT_FINE_DEFINE(M,15) SBN3_CT_FINE_DEFINE(M,16) SBN3_CT_FINE_DEFINE(M,17)
SBN3_CT_FINE_BAND(3) SBN3_CT_FINE_BAND(5) SBN3_CT_FINE_BAND(7)
#undef SBN3_CT_FINE_BAND
#undef SBN3_CT_FINE_DEFINE
static_assert(rac_bytes==1491968&&ct_bytes==1548288&&ct_fine_bytes==13824&&bytes==7771648);
}
