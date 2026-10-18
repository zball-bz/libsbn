// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
#include <algorithm>
#include <utility>

namespace sbn {
namespace {
template<class Query, class Execute>
bigfloat binary_owned(const bigfloat& a, const bigfloat& b,
                      Query query, Execute execute) noexcept {
    const bits precision{std::max(a.precision().count, b.precision().count)};
    auto out = bigfloat::with_resource_of(a, precision);
    const auto need = query(out.ref(), a.view(), b.view());
    detail::require(need.supported(), SBN3_FATAL_ARGUMENT, need.reason);
    if (!need.scratch_bytes) {
        (void)execute(out.ref(), a.view(), b.view(), execution{});
        return out;
    }
    // This is an explicit owning convenience boundary. In particular, neither
    // this resource lookup nor workspace construction occurs on a zero-scratch
    // leaf; the strict *_into call below cannot grow either storage region.
    workspace work(out.resource());
    work.reserve(need.scratch_bytes, need.scratch_alignment);
    (void)execute(out.ref(), a.view(), b.view(), work.on());
    return out;
}
template<class Query, class Execute>
bigfloat unary_owned(const bigfloat& a, bits precision,
                     Query query, Execute execute) noexcept {
    auto out = bigfloat::with_resource_of(a, precision);
    const auto need = query(out.ref(), a.view());
    detail::require(need.supported(), SBN3_FATAL_ARGUMENT, need.reason);
    if (!need.scratch_bytes) {
        (void)execute(out.ref(), a.view(), execution{});
        return out;
    }
    workspace work(out.resource());
    work.reserve(need.scratch_bytes, need.scratch_alignment);
    (void)execute(out.ref(), a.view(), work.on());
    return out;
}
template<class Execute>
void execute_owned_in_place(bigfloat& out, const requirements& need,
                           Execute execute) noexcept {
    detail::require(need.supported(), SBN3_FATAL_ARGUMENT, need.reason);
    if (!need.scratch_bytes) {
        (void)execute(execution{});
        return;
    }
    workspace work(out.resource());
    work.reserve(need.scratch_bytes, need.scratch_alignment);
    (void)execute(work.on());
}
template<class Result,class Query,class Execute>
Result paired_owned(const bigfloat& a,Query query,Execute execute)noexcept{
    auto first=bigfloat::with_resource_of(a,a.precision());
    auto second=bigfloat::with_resource_of(a,a.precision());
    const auto need=query(first.ref(),second.ref(),a.view());
    detail::require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);
    if(!need.scratch_bytes)(void)execute(first.ref(),second.ref(),a.view(),execution{});
    else{
        workspace work(first.resource());work.reserve(need.scratch_bytes,need.scratch_alignment);
        (void)execute(first.ref(),second.ref(),a.view(),work.on());
    }
    return {std::move(first),std::move(second)};
}
} // namespace

bigfloat add(const bigfloat& a, const bigfloat& b) noexcept {
    return binary_owned(a, b,
        [](float_ref out, float_view x, float_view y) noexcept {
            return add_requirements(out, x, y);
        },
        [](float_ref out, float_view x, float_view y, execution ex) noexcept {
            return add_into(out, x, y, ex);
        });
}
bigfloat subtract(const bigfloat& a, const bigfloat& b) noexcept {
    return binary_owned(a, b,
        [](float_ref out, float_view x, float_view y) noexcept {
            return add_requirements(out, x, y.negated());
        },
        [](float_ref out, float_view x, float_view y, execution ex) noexcept {
            return subtract_into(out, x, y, ex);
        });
}
bigfloat multiply(const bigfloat& a, const bigfloat& b) noexcept {
    if (&a == &b) return square(a);
    return binary_owned(a, b,
        [](float_ref out, float_view x, float_view y) noexcept {
            return multiply_requirements(out, x, y, 1);
        },
        [](float_ref out, float_view x, float_view y, execution ex) noexcept {
            return mul_into(out, x, y, ex);
        });
}
bigfloat square(const bigfloat& a) noexcept {
    return unary_owned(a, a.precision(),
        [](float_ref out, float_view x) noexcept {
            return square_requirements(out, x, 1);
        },
        [](float_ref out, float_view x, execution ex) noexcept {
            return square_into(out, x, ex);
        });
}
bigfloat sqrt(const bigfloat& a, bits precision) noexcept {
    return unary_owned(a, precision,
        [](float_ref out, float_view x) noexcept {
            return sqrt_requirements(out, x, 1);
        },
        [](float_ref out, float_view x, execution ex) noexcept {
            return sqrt_into(out, x, ex);
        });
}
bigfloat sqrt(const bigfloat& a) noexcept {
    return sqrt(a, a.precision());
}
bigfloat rsqrt(const bigfloat& a, bits precision) noexcept {
    return unary_owned(a, precision,
        [](float_ref out, float_view x) noexcept {
            return rsqrt_requirements(out, x, 1);
        },
        [](float_ref out, float_view x, execution ex) noexcept {
            return rsqrt_into(out, x, ex);
        });
}
bigfloat rsqrt(const bigfloat& a) noexcept {
    return rsqrt(a, a.precision());
}
bigfloat divide(const bigfloat& a,const bigfloat& b)noexcept{
    return binary_owned(a,b,
        [](float_ref out,float_view x,float_view y)noexcept{return divide_requirements(out,x,y,1);},
        [](float_ref out,float_view x,float_view y,execution ex)noexcept{return divide_into(out,x,y,ex);});
}
bigfloat reciprocal(const bigfloat& a,bits precision)noexcept{
    return unary_owned(a,precision,
        [](float_ref out,float_view x)noexcept{return reciprocal_requirements(out,x,1);},
        [](float_ref out,float_view x,execution ex)noexcept{return reciprocal_into(out,x,ex);});
}
bigfloat reciprocal(const bigfloat& a)noexcept{return reciprocal(a,a.precision());}
bigfloat agm(const bigfloat& a,const bigfloat& b)noexcept{
    return binary_owned(a,b,
        [](float_ref out,float_view x,float_view y)noexcept{return agm_requirements(out,x,y,1);},
        [](float_ref out,float_view x,float_view y,execution ex)noexcept{return agm_into(out,x,y,ex);});
}
bigfloat powi(const bigfloat& a,std::int64_t exponent,bits precision)noexcept{
    return unary_owned(a,precision,
        [exponent](float_ref out,float_view x)noexcept{return powi_requirements(out,x,exponent,1);},
        [exponent](float_ref out,float_view x,execution ex)noexcept{return powi_into(out,x,exponent,ex);});
}
bigfloat powi(const bigfloat& a,std::int64_t exponent)noexcept{return powi(a,exponent,a.precision());}
bigfloat rootn(const bigfloat& a,std::uint64_t degree,bits precision)noexcept{
    return unary_owned(a,precision,
        [degree](float_ref out,float_view x)noexcept{return rootn_requirements(out,x,degree,1);},
        [degree](float_ref out,float_view x,execution ex)noexcept{return rootn_into(out,x,degree,ex);});
}
bigfloat rootn(const bigfloat& a,std::uint64_t degree)noexcept{return rootn(a,degree,a.precision());}
integer to_integer(const bigfloat& a,rounding mode)noexcept{
    integer out(a.resource());
    const auto need=to_integer_requirements(out.ref(),a.view(),mode);
    detail::require(need.supported()||need.status==support::capacity,SBN3_FATAL_ARGUMENT,need.reason);
    out.reserve(need.output_limbs);
    (void)to_integer_into(out.ref(),a.view(),mode);
    return out;
}
bigfloat log(const bigfloat& a)noexcept{
    return unary_owned(a,a.precision(),
        [](float_ref out,float_view x)noexcept{return log_requirements(out,x,1);},
        [](float_ref out,float_view x,execution ex)noexcept{return log_into(out,x,ex);});
}
bigfloat log1p(const bigfloat& a)noexcept{
    return unary_owned(a,a.precision(),
        [](float_ref out,float_view x)noexcept{return log1p_requirements(out,x,1);},
        [](float_ref out,float_view x,execution ex)noexcept{return log1p_into(out,x,ex);});
}
bigfloat exp(const bigfloat& a)noexcept{
    return unary_owned(a,a.precision(),
        [](float_ref out,float_view x)noexcept{return exp_requirements(out,x,1);},
        [](float_ref out,float_view x,execution ex)noexcept{return exp_into(out,x,ex);});
}
bigfloat expm1(const bigfloat& a)noexcept{
    return unary_owned(a,a.precision(),
        [](float_ref out,float_view x)noexcept{return expm1_requirements(out,x,1);},
        [](float_ref out,float_view x,execution ex)noexcept{return expm1_into(out,x,ex);});
}
sincos_result sincos(const bigfloat& a)noexcept{
    return paired_owned<sincos_result>(a,
        [](float_ref s,float_ref c,float_view x)noexcept{return sincos_requirements(s,c,x,1);},
        [](float_ref s,float_ref c,float_view x,execution ex)noexcept{return sincos_into(s,c,x,ex);});
}
sincos_result sincospi(const bigfloat& a)noexcept{
    return paired_owned<sincos_result>(a,
        [](float_ref s,float_ref c,float_view x)noexcept{return sincospi_requirements(s,c,x,1);},
        [](float_ref s,float_ref c,float_view x,execution ex)noexcept{return sincospi_into(s,c,x,ex);});
}
elliptic_result elliptic_ke(const bigfloat& a,elliptic_argument kind)noexcept{
    return paired_owned<elliptic_result>(a,
        [kind](float_ref k,float_ref e,float_view x)noexcept{return elliptic_ke_requirements(k,e,x,kind,1);},
        [kind](float_ref k,float_ref e,float_view x,execution ex)noexcept{return elliptic_ke_into(k,e,x,kind,ex);});
}
#define SBN_FLOAT_UNARY_OWNER(name) \
bigfloat name(const bigfloat& a)noexcept{ \
    return unary_owned(a,a.precision(), \
        [](float_ref out,float_view x)noexcept{return name##_requirements(out,x,1);}, \
        [](float_ref out,float_view x,execution ex)noexcept{return name##_into(out,x,ex);}); }
SBN_FLOAT_UNARY_OWNER(sin)
SBN_FLOAT_UNARY_OWNER(cos)
SBN_FLOAT_UNARY_OWNER(sinpi)
SBN_FLOAT_UNARY_OWNER(cospi)
SBN_FLOAT_UNARY_OWNER(atan)
SBN_FLOAT_UNARY_OWNER(asin)
SBN_FLOAT_UNARY_OWNER(acos)
SBN_FLOAT_UNARY_OWNER(sinh)
SBN_FLOAT_UNARY_OWNER(cosh)
SBN_FLOAT_UNARY_OWNER(tanh)
SBN_FLOAT_UNARY_OWNER(asinh)
SBN_FLOAT_UNARY_OWNER(acosh)
SBN_FLOAT_UNARY_OWNER(atanh)
SBN_FLOAT_UNARY_OWNER(exp2)
SBN_FLOAT_UNARY_OWNER(log2)
SBN_FLOAT_UNARY_OWNER(log10)
#undef SBN_FLOAT_UNARY_OWNER
bigfloat atan2(const bigfloat& y,const bigfloat& x)noexcept{
    return binary_owned(y,x,
        [](float_ref out,float_view a,float_view b)noexcept{return atan2_requirements(out,a,b,1);},
        [](float_ref out,float_view a,float_view b,execution ex)noexcept{return atan2_into(out,a,b,ex);});
}
bigfloat pow(const bigfloat& x,const bigfloat& y)noexcept{
    return binary_owned(x,y,
        [](float_ref out,float_view a,float_view b)noexcept{return pow_requirements(out,a,b,1);},
        [](float_ref out,float_view a,float_view b,execution ex)noexcept{return pow_into(out,a,b,ex);});
}

bigfloat operator+(const bigfloat& a, const bigfloat& b) noexcept { return add(a, b); }
bigfloat operator-(const bigfloat& a, const bigfloat& b) noexcept { return subtract(a, b); }
bigfloat operator*(const bigfloat& a, const bigfloat& b) noexcept { return multiply(a, b); }
bigfloat operator/(const bigfloat& a,const bigfloat& b)noexcept{return divide(a,b);}
bigfloat operator+(const bigfloat& a) noexcept { return bigfloat(a); }
bigfloat operator-(const bigfloat& a) noexcept {
    auto out = +a;
    out.negate();
    return out;
}
bool operator==(const bigfloat& a, const bigfloat& b) noexcept {
    return compare(a.view(), b.view()) == std::partial_ordering::equivalent;
}
std::partial_ordering operator<=>(const bigfloat& a, const bigfloat& b) noexcept {
    return compare(a.view(), b.view());
}
bigfloat& bigfloat::operator+=(const bigfloat& other) noexcept {
    const auto precision = std::max(metadata_.precision, other.metadata_.precision);
    magnitude_.reserve((precision + 63) / 64);
    // Reserve can move storage, so capture both views afterward. This also
    // handles other == *this without retaining an invalidated raw view.
    const auto a = view(), b = other.view();
    metadata_.precision = precision;
    const auto need = add_requirements(ref(), a, b);
    execute_owned_in_place(*this, need, [&](execution ex) noexcept {
        return add_into(ref(), a, b, ex);
    });
    return *this;
}
bigfloat& bigfloat::operator-=(const bigfloat& other) noexcept {
    const auto precision = std::max(metadata_.precision, other.metadata_.precision);
    magnitude_.reserve((precision + 63) / 64);
    const auto a = view(), b = other.view();
    metadata_.precision = precision;
    const auto need = add_requirements(ref(), a, b.negated());
    execute_owned_in_place(*this, need, [&](execution ex) noexcept {
        return subtract_into(ref(), a, b, ex);
    });
    return *this;
}
bigfloat& bigfloat::operator*=(const bigfloat& other) noexcept {
    const auto precision = std::max(metadata_.precision, other.metadata_.precision);
    magnitude_.reserve((precision + 63) / 64);
    const auto a = view(), b = other.view();
    metadata_.precision = precision;
    const bool same = this == &other;
    const auto need = same ? square_requirements(ref(), a, 1)
                           : multiply_requirements(ref(), a, b, 1);
    execute_owned_in_place(*this, need, [&](execution ex) noexcept {
        return same ? square_into(ref(), a, ex) : mul_into(ref(), a, b, ex);
    });
    return *this;
}
bigfloat& bigfloat::operator/=(const bigfloat& other)noexcept{
    const auto precision=std::max(metadata_.precision,other.metadata_.precision);
    magnitude_.reserve((precision+63)/64);
    const auto a=view(),b=other.view();metadata_.precision=precision;
    const auto need=divide_requirements(ref(),a,b,1);
    execute_owned_in_place(*this,need,[&](execution ex)noexcept{return divide_into(ref(),a,b,ex);});
    return *this;
}
} // namespace sbn
