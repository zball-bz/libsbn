// Original libsbn code; LGPL-3.0-or-later.
#include "cpp/float_constant_access.hpp"
#include "float/logarithm.hpp"
#include "float/exponential.hpp"
#include "float/elliptic.hpp"
#include "float/trig.hpp"
#include "float/inverse_trig.hpp"
#include "float/hyperbolic.hpp"
#include "float/derived_explog.hpp"
#include "float/real_power.hpp"

namespace sbn {
namespace f=v3::floating;
namespace {
constant_demand external_demand(f::ConstantBankDemand demand)noexcept{return {demand.pi_bits,demand.log2_bits};}
f::ConstantBankBinding binding_of(constant_view view)noexcept{return detail::constant_access::internal(view);}
f::EllipticParameter elliptic_kind(elliptic_argument kind)noexcept{
    detail::require(kind==elliptic_argument::parameter||kind==elliptic_argument::complement,
                    SBN3_FATAL_ARGUMENT,"elliptic parameter kind");
    return kind==elliptic_argument::parameter?f::EllipticParameter::parameter:f::EllipticParameter::complement;
}
}
constant_demand log_constants(bits precision)noexcept{return external_demand(f::logarithm_constant_demand(precision));}
constant_demand exp_constants(bits precision)noexcept{return external_demand(f::exponential_constant_demand(precision));}
constant_demand atan2_constants(bits precision)noexcept{return external_demand(f::inverse_trig_constant_demand(precision,f::InverseTrigKind::atan2));}
constant_demand elliptic_constants(bits precision)noexcept{return external_demand(f::elliptic_constant_demand(precision));}
#define SBN3_LOG_BANK(name,kind) \
requirements name##_requirements(float_ref out,float_view x,unsigned workers,constant_view bank)noexcept{ \
    return f::logarithm_requirements(out,x,f::LogarithmKind::kind,workers,binding_of(bank));} \
float_report name##_into(float_ref out,float_view x,execution ex,rounding mode,constant_view bank)noexcept{ \
    return f::logarithm_into(out,x,f::LogarithmKind::kind,ex,mode,binding_of(bank));}
SBN3_LOG_BANK(log,log)
SBN3_LOG_BANK(log1p,log1p)
#undef SBN3_LOG_BANK
#define SBN3_EXP_BANK(name,kind) \
requirements name##_requirements(float_ref out,float_view x,unsigned workers,constant_view bank)noexcept{ \
    return f::exponential_requirements(out,x,f::ExponentialKind::kind,workers,binding_of(bank));} \
float_report name##_into(float_ref out,float_view x,execution ex,rounding mode,constant_view bank)noexcept{ \
    return f::exponential_into(out,x,f::ExponentialKind::kind,ex,mode,binding_of(bank));}
SBN3_EXP_BANK(exp,exp)
SBN3_EXP_BANK(expm1,expm1)
#undef SBN3_EXP_BANK
#define SBN3_TRIG_BANK(name,cosine,kind) \
requirements name##_requirements(float_ref out,float_view x,unsigned workers,constant_view bank)noexcept{ \
    return f::unary_trig_requirements(out,x,workers,f::TrigKind::kind,binding_of(bank));} \
float_report name##_into(float_ref out,float_view x,execution ex,rounding mode,constant_view bank)noexcept{ \
    return f::unary_trig_into(out,x,cosine,f::TrigKind::kind,ex,mode,binding_of(bank));}
SBN3_TRIG_BANK(sin,false,radians)
SBN3_TRIG_BANK(cos,true,radians)
SBN3_TRIG_BANK(sinpi,false,pi_scaled)
SBN3_TRIG_BANK(cospi,true,pi_scaled)
#undef SBN3_TRIG_BANK
#define SBN3_INVERSE_BANK(name,kind) \
constant_demand name##_constants(bits precision)noexcept{ \
    return external_demand(f::inverse_trig_constant_demand(precision,f::InverseTrigKind::kind));} \
requirements name##_requirements(float_ref out,float_view x,unsigned workers,constant_view bank)noexcept{ \
    return f::inverse_trig_requirements(out,x,{},workers,f::InverseTrigKind::kind,binding_of(bank));} \
float_report name##_into(float_ref out,float_view x,execution ex,rounding mode,constant_view bank)noexcept{ \
    return f::inverse_trig_into(out,x,{},ex,mode,f::InverseTrigKind::kind,binding_of(bank));}
SBN3_INVERSE_BANK(atan,atan)
SBN3_INVERSE_BANK(asin,asin)
SBN3_INVERSE_BANK(acos,acos)
#undef SBN3_INVERSE_BANK
#define SBN3_HYPERBOLIC_BANK(name,kind) \
constant_demand name##_constants(bits precision)noexcept{ \
    return external_demand(f::hyperbolic_constant_demand(precision,f::HyperbolicKind::kind));} \
requirements name##_requirements(float_ref out,float_view x,unsigned workers,constant_view bank)noexcept{ \
    return f::hyperbolic_requirements(out,x,workers,f::HyperbolicKind::kind,binding_of(bank));} \
float_report name##_into(float_ref out,float_view x,execution ex,rounding mode,constant_view bank)noexcept{ \
    return f::hyperbolic_into(out,x,ex,mode,f::HyperbolicKind::kind,binding_of(bank));}
SBN3_HYPERBOLIC_BANK(sinh,sinh)
SBN3_HYPERBOLIC_BANK(cosh,cosh)
SBN3_HYPERBOLIC_BANK(tanh,tanh)
SBN3_HYPERBOLIC_BANK(asinh,asinh)
SBN3_HYPERBOLIC_BANK(acosh,acosh)
SBN3_HYPERBOLIC_BANK(atanh,atanh)
#undef SBN3_HYPERBOLIC_BANK
#define SBN3_DERIVED_BANK(name,kind) \
constant_demand name##_constants(bits precision)noexcept{ \
    return external_demand(f::derived_explog_constant_demand(precision,f::DerivedExpLogKind::kind));} \
requirements name##_requirements(float_ref out,float_view x,unsigned workers,constant_view bank)noexcept{ \
    return f::derived_explog_requirements(out,x,f::DerivedExpLogKind::kind,workers,binding_of(bank));} \
float_report name##_into(float_ref out,float_view x,execution ex,rounding mode,constant_view bank)noexcept{ \
    return f::derived_explog_into(out,x,f::DerivedExpLogKind::kind,ex,mode,binding_of(bank));}
SBN3_DERIVED_BANK(exp2,exp2)
SBN3_DERIVED_BANK(log2,log2)
SBN3_DERIVED_BANK(log10,log10)
#undef SBN3_DERIVED_BANK
constant_demand pow_constants(bits precision)noexcept{return external_demand(f::real_power_constant_demand(precision));}
requirements pow_requirements(float_ref out,float_view x,float_view y,unsigned workers,constant_view bank)noexcept{
    return f::real_power_requirements(out,x,y,workers,binding_of(bank));
}
float_report pow_into(float_ref out,float_view x,float_view y,execution ex,rounding mode,constant_view bank)noexcept{
    return f::real_power_into(out,x,y,ex,mode,binding_of(bank));
}
requirements atan2_requirements(float_ref out,float_view y,float_view x,unsigned workers,constant_view bank)noexcept{
    return f::inverse_trig_requirements(out,y,x,workers,f::InverseTrigKind::atan2,binding_of(bank));
}
float_report atan2_into(float_ref out,float_view y,float_view x,execution ex,rounding mode,constant_view bank)noexcept{
    return f::inverse_trig_into(out,y,x,ex,mode,f::InverseTrigKind::atan2,binding_of(bank));
}
#define SBN3_PAIRED_TRIG_BANK(name,kind) \
constant_demand name##_constants(bits sine,bits cosine,float_view x)noexcept{ \
    return external_demand(f::trig_constant_demand(sine,cosine,x,f::TrigKind::kind));} \
requirements name##_requirements(float_ref sine,float_ref cosine,float_view x,unsigned workers,constant_view bank)noexcept{ \
    return f::trig_requirements(sine,cosine,x,workers,f::TrigKind::kind,binding_of(bank));} \
sincos_report name##_into(float_ref sine,float_ref cosine,float_view x,execution ex,rounding mode,constant_view bank)noexcept{ \
    const auto report=f::trig_into(sine,cosine,x,ex,mode,f::TrigKind::kind,binding_of(bank));return {report.sine,report.cosine};}
SBN3_PAIRED_TRIG_BANK(sincos,radians)
SBN3_PAIRED_TRIG_BANK(sincospi,pi_scaled)
#undef SBN3_PAIRED_TRIG_BANK
requirements elliptic_ke_requirements(float_ref k,float_ref e,float_view x,elliptic_argument kind,unsigned workers,constant_view bank)noexcept{
    return f::elliptic_ke_requirements(k,e,x,elliptic_kind(kind),workers,binding_of(bank));
}
elliptic_report elliptic_ke_into(float_ref k,float_ref e,float_view x,elliptic_argument kind,execution ex,rounding mode,constant_view bank)noexcept{
    const auto result=f::elliptic_ke_into(k,e,x,elliptic_kind(kind),ex,mode,binding_of(bank));return {result.k,result.e};
}
}
