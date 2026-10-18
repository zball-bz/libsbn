// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/float.hpp"
#include <utility>

namespace sbn {
enum class constant_kind : unsigned char { pi, ln2 };
struct constant_demand {
    // Zero omits an entry; an all-zero request is a no-op.
    std::size_t pi_bits=0,ln2_bits=0;
};
constexpr constant_demand merge_constants(constant_demand a,constant_demand b)noexcept{
    return {a.pi_bits>b.pi_bits?a.pi_bits:b.pi_bits,a.ln2_bits>b.ln2_bits?a.ln2_bits:b.ln2_bits};
}
struct constant_storage_requirements {
    requirements work{};
    std::size_t value_bytes=0,value_alignment=128;
    constant_demand provided{};
    bool supported()const noexcept{return work.supported();}
};
struct constant_approximation {
    float_view point{};
    error_bound error{};
    bits precision{0};
};
namespace detail {struct constant_access;}

// Borrowed immutable values. The owner and its allocation must outlive every
// view and prepared consumer. Rebuilding the owner invalidates old borrows;
// moving it without relocating storage does not. Published views can be read
// by multiple workers, but reserve/build/destroy are single-owner boundaries.
class constant_view {
    const void* state_=nullptr;
    std::span<const std::byte> storage_{};
    constant_demand provided_{};
    friend struct detail::constant_access;
public:
    constexpr constant_view()noexcept=default;
    bool empty()const noexcept{return state_==nullptr;}
    constant_demand precision()const noexcept{return provided_;}
    bool supports(constant_demand)const noexcept;
    constant_approximation get(constant_kind)const noexcept;
};

// A ln2 request also admits the pi precision needed by its independent
// producer. Retained values and temporary construction scratch are separate.
constant_storage_requirements constants_requirements(constant_demand,unsigned workers=1)noexcept;
constant_view constants_into(std::span<std::byte> values,constant_demand,execution)noexcept;
// Nearest-even prefix copy with the producer and discarded-tail errors added.
// The output precision must not exceed the cached entry's precision.
float_report copy_constant_into(float_ref,constant_view,constant_kind)noexcept;

// Optional explicit-bank overloads. The bank is last so existing calls and
// execution{} arguments keep their meaning. Demand helpers quote the generic
// recipe; special/local paths may consume fewer constants. Merge demands with
// componentwise maxima before preparing one bank for several operations.
#define SBN3_CONSTANT_UNARY(name) \
requirements name##_requirements(float_ref,float_view,unsigned,constant_view)noexcept; \
float_report name##_into(float_ref,float_view,execution,rounding,constant_view)noexcept;
SBN3_CONSTANT_UNARY(log)
SBN3_CONSTANT_UNARY(log1p)
SBN3_CONSTANT_UNARY(exp)
SBN3_CONSTANT_UNARY(expm1)
SBN3_CONSTANT_UNARY(sin)
SBN3_CONSTANT_UNARY(cos)
SBN3_CONSTANT_UNARY(sinpi)
SBN3_CONSTANT_UNARY(cospi)
SBN3_CONSTANT_UNARY(atan)
SBN3_CONSTANT_UNARY(asin)
SBN3_CONSTANT_UNARY(acos)
SBN3_CONSTANT_UNARY(sinh)
SBN3_CONSTANT_UNARY(cosh)
SBN3_CONSTANT_UNARY(tanh)
SBN3_CONSTANT_UNARY(asinh)
SBN3_CONSTANT_UNARY(acosh)
SBN3_CONSTANT_UNARY(atanh)
SBN3_CONSTANT_UNARY(exp2)
SBN3_CONSTANT_UNARY(log2)
SBN3_CONSTANT_UNARY(log10)
#undef SBN3_CONSTANT_UNARY
constant_demand log_constants(bits)noexcept;
constant_demand exp_constants(bits)noexcept;
inline constant_demand log1p_constants(bits p)noexcept{return log_constants(p);}
inline constant_demand expm1_constants(bits p)noexcept{return exp_constants(p);}
constant_demand atan_constants(bits)noexcept;
constant_demand atan2_constants(bits)noexcept;
constant_demand asin_constants(bits)noexcept;
constant_demand acos_constants(bits)noexcept;
constant_demand sinh_constants(bits)noexcept;
constant_demand cosh_constants(bits)noexcept;
constant_demand tanh_constants(bits)noexcept;
constant_demand asinh_constants(bits)noexcept;
constant_demand acosh_constants(bits)noexcept;
constant_demand atanh_constants(bits)noexcept;
constant_demand exp2_constants(bits)noexcept;
constant_demand log2_constants(bits)noexcept;
constant_demand log10_constants(bits)noexcept;
constant_demand pow_constants(bits)noexcept;
constant_demand sincos_constants(bits sine,bits cosine,float_view)noexcept;
constant_demand sincospi_constants(bits sine,bits cosine,float_view)noexcept;
constant_demand elliptic_constants(bits)noexcept;
requirements atan2_requirements(float_ref,float_view,float_view,unsigned,constant_view)noexcept;
float_report atan2_into(float_ref,float_view,float_view,execution,rounding,constant_view)noexcept;
requirements pow_requirements(float_ref,float_view,float_view,unsigned,constant_view)noexcept;
float_report pow_into(float_ref,float_view,float_view,execution,rounding,constant_view)noexcept;
requirements sincos_requirements(float_ref,float_ref,float_view,unsigned,constant_view)noexcept;
requirements sincospi_requirements(float_ref,float_ref,float_view,unsigned,constant_view)noexcept;
sincos_report sincos_into(float_ref,float_ref,float_view,execution,rounding,constant_view)noexcept;
sincos_report sincospi_into(float_ref,float_ref,float_view,execution,rounding,constant_view)noexcept;
requirements elliptic_ke_requirements(float_ref,float_ref,float_view,elliptic_argument,unsigned,constant_view)noexcept;
elliptic_report elliptic_ke_into(float_ref,float_ref,float_view,elliptic_argument,execution,rounding,constant_view)noexcept;

// Explicit convenience owner. Construction allocates nothing. try_prepare
// may allocate value storage through the supplied resource; it never grows
// the caller's execution scratch. A compatible ready bank is retained.
class constant_bank {
    memory_resource* resource_;
    allocation values_{};
    constant_view ready_{};
public:
    explicit constant_bank(memory_resource& r=default_resource())noexcept:resource_(&r){}
    ~constant_bank(){if(values_.data)resource_->deallocate(values_);}
    constant_bank(const constant_bank&)=delete;
    constant_bank&operator=(const constant_bank&)=delete;
    constant_bank(constant_bank&& other)noexcept:resource_(other.resource_),
        values_(std::exchange(other.values_,{})),ready_(std::exchange(other.ready_,{})){}
    constant_bank&operator=(constant_bank&& other)noexcept{
        if(this!=&other){
            if(values_.data)resource_->deallocate(values_);
            resource_=other.resource_;values_=std::exchange(other.values_,{});ready_=std::exchange(other.ready_,{});
        }
        return *this;
    }
    resource_error try_prepare(constant_demand,execution)noexcept;
    void prepare(constant_demand demand,execution ex)noexcept{
        const auto error=try_prepare(demand,ex);
        detail::require(!error,SBN3_FATAL_WORKSPACE,error.where,error.need,error.have);
    }
    constant_view view()const&noexcept{return ready_;}
    constant_view view()const&&=delete;
    std::size_t capacity()const noexcept{return values_.bytes;}
    memory_resource& resource()const noexcept{return *resource_;}
};
}
