// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/integer.hpp"
#include <bit>
#include <compare>
#include <limits>

namespace sbn {
struct bits { std::size_t count; };
enum class float_class : unsigned char { finite, infinity, quiet_nan };
enum class rounding : unsigned char { nearest_even, toward_zero, upward, downward, away_zero };
enum class error_kind : unsigned char { exact, finite, unbounded };
// A finite bound means |computed - mathematical value| <= 2^exponent2.
// Its exponent is independent of the result's mantissa and allocation size.
struct error_bound {
    error_kind kind=error_kind::exact;
    std::int64_t exponent2=0;
    static constexpr error_bound power_of_two(std::int64_t e)noexcept{return {error_kind::finite,e};}
    static constexpr error_bound unbounded()noexcept{return {error_kind::unbounded,0};}
    constexpr bool exact()const noexcept{return kind==error_kind::exact;}
};
enum class float_flag : unsigned {
    none=0, rounded=1, invalid=2, divide_by_zero=4, overflow=8, underflow=16
};
constexpr float_flag operator|(float_flag a,float_flag b)noexcept{return float_flag(unsigned(a)|unsigned(b));}
constexpr bool has_flag(float_flag a,float_flag b)noexcept{return (unsigned(a)&unsigned(b))!=0;}
struct float_report {
    error_bound error{};
    float_flag flags=float_flag::none;
};
struct sincos_report {float_report sine{},cosine{};};
enum class elliptic_argument {parameter,complement};
struct elliptic_report {float_report k{},e{};};
// Domain limits leave checked exponent arithmetic ample headroom. For a
// nonzero finite result, exponent2 + bit_length(mantissa) is in this range.
inline constexpr std::int64_t float_exponent_limit=std::int64_t(1)<<60;
inline constexpr std::size_t float_precision_limit=std::size_t(1)<<40;

class float_view {
    integer_view magnitude_{};
    std::int64_t exponent2_=0;
    float_class class_=float_class::finite;
    bool negative_=false;
public:
    constexpr float_view()noexcept=default;
    // The input is an exact dyadic point, not an uncertain real approximation.
    explicit float_view(integer_view m,std::int64_t e=0,bool negative_zero=false)noexcept
        :magnitude_(m.limbs()),exponent2_(e),negative_(m.zero()?negative_zero:m.negative()){}
    static float_view from_magnitude(std::span<const std::uint64_t> m,std::int64_t e,bool negative=false)noexcept{
        float_view r;r.magnitude_=integer_view{m};r.exponent2_=e;r.negative_=negative;return r;
    }
    static float_view infinity(bool negative=false)noexcept{float_view r;r.class_=float_class::infinity;r.negative_=negative;return r;}
    static float_view nan()noexcept{float_view r;r.class_=float_class::quiet_nan;return r;}
    constexpr integer_view magnitude()const noexcept{return magnitude_;}
    constexpr std::int64_t exponent2()const noexcept{return exponent2_;}
    constexpr float_class classification()const noexcept{return class_;}
    constexpr bool finite()const noexcept{return class_==float_class::finite;}
    constexpr bool is_nan()const noexcept{return class_==float_class::quiet_nan;}
    constexpr bool zero()const noexcept{return finite()&&magnitude_.zero();}
    constexpr bool negative()const noexcept{return negative_;}
    float_view negated()const noexcept{auto r=*this;r.negative_=!r.negative_;return r;}
};
namespace detail {
struct float_metadata {std::int64_t exponent2=0;std::size_t precision=128;float_class classification=float_class::finite;bool negative=false;};
struct float_access;
inline void valid_precision(bits p)noexcept{require(p.count&&p.count<=float_precision_limit,SBN3_FATAL_ARGUMENT,"float precision",p.count,float_precision_limit);}
}
class bigfloat;class float_buffer;
class float_ref {
    integer_ref magnitude_;detail::float_metadata*metadata_;
    friend class bigfloat;friend class float_buffer;friend struct detail::float_access;
    float_ref(integer_ref m,detail::float_metadata*d)noexcept:magnitude_(m),metadata_(d){}
public:
    std::size_t capacity()const noexcept{return magnitude_.capacity();}
    bits precision()const noexcept{return {metadata_->precision};}
    float_view view()const noexcept{
        if(metadata_->classification==float_class::infinity)return float_view::infinity(metadata_->negative);
        if(metadata_->classification==float_class::quiet_nan)return float_view::nan();
        return float_view::from_magnitude(magnitude_.view().limbs(),metadata_->exponent2,metadata_->negative);
    }
};
class float_buffer {
    integer_buffer magnitude_;detail::float_metadata metadata_{};
public:
    explicit float_buffer(std::span<std::uint64_t> storage,bits precision)noexcept:magnitude_(storage){
        detail::valid_precision(precision);metadata_.precision=precision.count;
    }
    float_ref ref()&noexcept{return {magnitude_.ref(),&metadata_};}
    float_ref ref()&&=delete;
    float_view view()const&noexcept{
        if(metadata_.classification==float_class::infinity)return float_view::infinity(metadata_.negative);
        if(metadata_.classification==float_class::quiet_nan)return float_view::nan();
        return float_view::from_magnitude(magnitude_.view().limbs(),metadata_.exponent2,metadata_.negative);
    }
    float_view view()const&&=delete;
    bits precision()const noexcept{return {metadata_.precision};}
};

// Arithmetic never allocates value or scratch storage. Exact-base aliases are
// supported; overlapping but different value buffers are not. A report is local
// to these exact input points; callers explicitly compose earlier error bounds.
float_report round_into(float_ref,float_view,rounding=rounding::nearest_even)noexcept;
float_report scale2_into(float_ref,float_view,std::int64_t,rounding=rounding::nearest_even)noexcept;
requirements add_requirements(float_ref,float_view,float_view)noexcept;
float_report add_into(float_ref,float_view,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report subtract_into(float_ref,float_view,float_view,execution={},rounding=rounding::nearest_even)noexcept;
std::partial_ordering compare(float_view,float_view)noexcept;

// Retain at most p+128 input bits. Local error includes omitted input tails.
// The mode rounds the retained approximation, not the exact full-input product.
requirements multiply_requirements(float_ref,float_view,float_view,unsigned workers=1)noexcept;
requirements square_requirements(float_ref,float_view,unsigned workers=1)noexcept;
float_report mul_into(float_ref,float_view,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report square_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
// Full exact dyadic product plus addend, with one final rounding. Unlike mul,
// no multiplicand prefix is discarded before potentially severe cancellation.
requirements fma_requirements(float_ref,float_view,float_view,float_view,unsigned workers=1)noexcept;
float_report fma_into(float_ref,float_view,float_view,float_view,execution={},rounding=rounding::nearest_even)noexcept;

// General multiprecision radicands. The mode rounds the computed approximation,
// not the true root; the local report includes Newton and final-rounding errors.
requirements sqrt_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements rsqrt_requirements(float_ref,float_view,unsigned workers=1)noexcept;
float_report sqrt_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report rsqrt_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
requirements divide_requirements(float_ref,float_view,float_view,unsigned workers=1)noexcept;
requirements reciprocal_requirements(float_ref,float_view,unsigned workers=1)noexcept;
float_report divide_into(float_ref,float_view,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report reciprocal_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
requirements agm_requirements(float_ref,float_view,float_view,unsigned workers=1)noexcept;
float_report agm_into(float_ref,float_view,float_view,execution={},rounding=rounding::nearest_even)noexcept;
requirements powi_requirements(float_ref,float_view,std::int64_t,unsigned workers=1)noexcept;
float_report powi_into(float_ref,float_view,std::int64_t,execution={},rounding=rounding::nearest_even)noexcept;
requirements rootn_requirements(float_ref,float_view,std::uint64_t,unsigned workers=1)noexcept;
float_report rootn_into(float_ref,float_view,std::uint64_t,execution={},rounding=rounding::nearest_even)noexcept;
// Exact dyadic-to-integer rounding. Nonfinite inputs are outside this
// operation's domain; requirements reports unsupported before execution.
requirements to_integer_requirements(integer_ref,float_view,rounding=rounding::nearest_even)noexcept;
float_report to_integer_into(integer_ref,float_view,rounding=rounding::nearest_even)noexcept;
requirements log_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements log1p_requirements(float_ref,float_view,unsigned workers=1)noexcept;
float_report log_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report log1p_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
requirements exp_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements expm1_requirements(float_ref,float_view,unsigned workers=1)noexcept;
float_report exp_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report expm1_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
// Trig reports include absolute range-reduction error. A tiny component need
// not have p relative bits merely because a value has p-bit storage.
requirements sincos_requirements(float_ref,float_ref,float_view,unsigned workers=1)noexcept;
requirements sincospi_requirements(float_ref,float_ref,float_view,unsigned workers=1)noexcept;
sincos_report sincos_into(float_ref,float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
sincos_report sincospi_into(float_ref,float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
requirements sin_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements cos_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements sinpi_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements cospi_requirements(float_ref,float_view,unsigned workers=1)noexcept;
float_report sin_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report cos_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report sinpi_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report cospi_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
requirements atan_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements atan2_requirements(float_ref,float_view y,float_view x,unsigned workers=1)noexcept;
requirements asin_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements acos_requirements(float_ref,float_view,unsigned workers=1)noexcept;
float_report atan_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report atan2_into(float_ref,float_view y,float_view x,execution={},rounding=rounding::nearest_even)noexcept;
float_report asin_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report acos_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
requirements sinh_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements cosh_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements tanh_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements asinh_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements acosh_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements atanh_requirements(float_ref,float_view,unsigned workers=1)noexcept;
float_report sinh_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report cosh_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report tanh_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report asinh_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report acosh_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report atanh_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
requirements exp2_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements log2_requirements(float_ref,float_view,unsigned workers=1)noexcept;
requirements log10_requirements(float_ref,float_view,unsigned workers=1)noexcept;
float_report exp2_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report log2_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
float_report log10_into(float_ref,float_view,execution={},rounding=rounding::nearest_even)noexcept;
requirements pow_requirements(float_ref,float_view,float_view,unsigned workers=1)noexcept;
float_report pow_into(float_ref,float_view,float_view,execution={},rounding=rounding::nearest_even)noexcept;
// K/E use parameter m; complement means the supplied point is c=1-m.
// Both caller-owned outputs have the same precision and must be disjoint.
requirements elliptic_ke_requirements(float_ref,float_ref,float_view,
    elliptic_argument=elliptic_argument::parameter,unsigned workers=1)noexcept;
elliptic_report elliptic_ke_into(float_ref,float_ref,float_view,
    elliptic_argument=elliptic_argument::parameter,execution={},rounding=rounding::nearest_even)noexcept;

class bigfloat {
    integer magnitude_;detail::float_metadata metadata_{};
    friend struct detail::float_access;
    bigfloat(integer&&storage,bits p,detail::canonical)noexcept:magnitude_(std::move(storage)){
        detail::valid_precision(p);metadata_.precision=p.count;magnitude_.reserve((p.count+63)/64);
    }
public:
    explicit bigfloat(bits precision={128})noexcept{detail::valid_precision(precision);metadata_.precision=precision.count;magnitude_.reserve((precision.count+63)/64);}
    bigfloat(bits precision,memory_resource&r)noexcept:magnitude_(r){detail::valid_precision(precision);metadata_.precision=precision.count;magnitude_.reserve((precision.count+63)/64);}
    bigfloat(const bigfloat&x)noexcept:magnitude_(x.magnitude_),metadata_(x.metadata_){magnitude_.reserve((metadata_.precision+63)/64);}
    bigfloat(bigfloat&&x)noexcept:magnitude_(std::move(x.magnitude_)),metadata_(x.metadata_){x.metadata_={};}
    bigfloat&operator=(const bigfloat&x)noexcept{
        if(this!=&x){magnitude_=x.magnitude_;metadata_=x.metadata_;magnitude_.reserve((metadata_.precision+63)/64);}return *this;
    }
    bigfloat&operator=(bigfloat&&x)noexcept{
        if(this!=&x){magnitude_=std::move(x.magnitude_);metadata_=x.metadata_;magnitude_.reserve((metadata_.precision+63)/64);x.metadata_={};}return *this;
    }
    float_ref ref()&noexcept{return {magnitude_.ref(),&metadata_};}
    float_ref ref()&&=delete;
    float_view view()const&noexcept{
        if(metadata_.classification==float_class::infinity)return float_view::infinity(metadata_.negative);
        if(metadata_.classification==float_class::quiet_nan)return float_view::nan();
        return float_view::from_magnitude(magnitude_.view().limbs(),metadata_.exponent2,metadata_.negative);
    }
    float_view view()const&&=delete;
    bits precision()const noexcept{return {metadata_.precision};}
    std::size_t capacity()const noexcept{return magnitude_.capacity();}
    memory_resource&resource()const noexcept{return magnitude_.resource();}
    resource_error try_reserve_bits(bits p)noexcept{
        if(!p.count||p.count>float_precision_limit)return {resource_errc::capacity,p.count,float_precision_limit,0,"float precision"};
        return magnitude_.try_reserve((p.count+63)/64);
    }
    void reserve_bits(bits p)noexcept{auto e=try_reserve_bits(p);detail::require(!e,SBN3_FATAL_WORKSPACE,e.where,e.need,e.have);}
    float_report set_precision(bits p,rounding mode=rounding::nearest_even)noexcept{
        detail::valid_precision(p);reserve_bits(p);auto old=view();metadata_.precision=p.count;return round_into(ref(),old,mode);
    }
    float_report assign(float_view v,rounding mode=rounding::nearest_even)noexcept{return round_into(ref(),v,mode);}
    void negate()noexcept{metadata_.negative=!metadata_.negative;}
    static bigfloat with_resource_of(const bigfloat&model,bits p)noexcept{
        return bigfloat(integer::with_resource_of(model.magnitude_),p,detail::canonical{});
    }
    bigfloat&operator+=(const bigfloat&)noexcept;
    bigfloat&operator-=(const bigfloat&)noexcept;
    bigfloat&operator*=(const bigfloat&)noexcept;
    bigfloat&operator/=(const bigfloat&)noexcept;
    template<detail::builtin_integer I>static bigfloat from(I x,bits p={128})noexcept{
        integer i{x};bigfloat r(p);r.assign(float_view{i.view()});return r;
    }
    static bigfloat from(double x,bits p={128})noexcept{
        const auto raw=std::bit_cast<std::uint64_t>(x);const bool negative=raw>>63;
        const auto e=(raw>>52)&2047;std::uint64_t m=raw&((std::uint64_t(1)<<52)-1);
        bigfloat r(p);
        if(e==2047)r.assign(m?float_view::nan():float_view::infinity(negative));
        else{if(e)m|=std::uint64_t(1)<<52;
            r.assign(float_view::from_magnitude({&m,1},e?std::int64_t(e)-1075:-1074,negative));}
        return r;
    }
};
// Eager binary operations use max(a.precision(),b.precision()); unary
// operations preserve precision unless explicitly given a result precision.
// Use *_into to obtain an error report or control admitted workspace/Team.
bigfloat add(const bigfloat&,const bigfloat&)noexcept;
bigfloat subtract(const bigfloat&,const bigfloat&)noexcept;
bigfloat multiply(const bigfloat&,const bigfloat&)noexcept;
bigfloat square(const bigfloat&)noexcept;
bigfloat sqrt(const bigfloat&)noexcept;
bigfloat sqrt(const bigfloat&,bits)noexcept;
bigfloat rsqrt(const bigfloat&)noexcept;
bigfloat rsqrt(const bigfloat&,bits)noexcept;
bigfloat divide(const bigfloat&,const bigfloat&)noexcept;
bigfloat reciprocal(const bigfloat&)noexcept;
bigfloat reciprocal(const bigfloat&,bits)noexcept;
bigfloat agm(const bigfloat&,const bigfloat&)noexcept;
bigfloat powi(const bigfloat&,std::int64_t)noexcept;
bigfloat powi(const bigfloat&,std::int64_t,bits)noexcept;
bigfloat rootn(const bigfloat&,std::uint64_t)noexcept;
bigfloat rootn(const bigfloat&,std::uint64_t,bits)noexcept;
integer to_integer(const bigfloat&,rounding=rounding::nearest_even)noexcept;
bigfloat log(const bigfloat&)noexcept;
bigfloat log1p(const bigfloat&)noexcept;
bigfloat exp(const bigfloat&)noexcept;
bigfloat expm1(const bigfloat&)noexcept;
bigfloat sin(const bigfloat&)noexcept;
bigfloat cos(const bigfloat&)noexcept;
bigfloat sinpi(const bigfloat&)noexcept;
bigfloat cospi(const bigfloat&)noexcept;
bigfloat atan(const bigfloat&)noexcept;
bigfloat atan2(const bigfloat&,const bigfloat&)noexcept;
bigfloat asin(const bigfloat&)noexcept;
bigfloat acos(const bigfloat&)noexcept;
bigfloat sinh(const bigfloat&)noexcept;
bigfloat cosh(const bigfloat&)noexcept;
bigfloat tanh(const bigfloat&)noexcept;
bigfloat asinh(const bigfloat&)noexcept;
bigfloat acosh(const bigfloat&)noexcept;
bigfloat atanh(const bigfloat&)noexcept;
bigfloat exp2(const bigfloat&)noexcept;
bigfloat log2(const bigfloat&)noexcept;
bigfloat log10(const bigfloat&)noexcept;
bigfloat pow(const bigfloat&,const bigfloat&)noexcept;
struct sincos_result {bigfloat sine,cosine;};
struct elliptic_result {bigfloat k,e;};
sincos_result sincos(const bigfloat&)noexcept;
sincos_result sincospi(const bigfloat&)noexcept;
elliptic_result elliptic_ke(const bigfloat&,elliptic_argument=elliptic_argument::parameter)noexcept;
bigfloat operator+(const bigfloat&,const bigfloat&)noexcept;
bigfloat operator-(const bigfloat&,const bigfloat&)noexcept;
bigfloat operator*(const bigfloat&,const bigfloat&)noexcept;
bigfloat operator/(const bigfloat&,const bigfloat&)noexcept;
bigfloat operator+(const bigfloat&)noexcept;
bigfloat operator-(const bigfloat&)noexcept;
bool operator==(const bigfloat&,const bigfloat&)noexcept;
std::partial_ordering operator<=>(const bigfloat&,const bigfloat&)noexcept;
}
