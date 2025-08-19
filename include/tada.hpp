/**
 * tada
 *
 * Copyright (C) 2024  Lorenz Gillner
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TADA_HPP
#define TADA_HPP

#ifndef __CUDACC__
#define TADA_ENABLE_EXPERIMENTAL
#endif

#ifdef __CUDACC__
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/utility>
#define portable __host__ __device__
#define _std cuda::std
#else
#include <array>
#include <cmath>
#include <utility>
#ifdef TADA_ENABLE_EXPERIMENTAL
#include <tuple>
#endif
#define portable
#define _std std
#endif // __CUDACC__

#define TADA_ORDER 1

namespace tada {
template <typename T>
class Derivable {
private:
    T value = 0;
    T deriv = 0;
#if TADA_ORDER > 1
    T dderiv = 0;
#endif

public:
    portable Derivable();

    portable explicit Derivable(const T& v);

    portable Derivable(const T& v, const T& d);

    template <typename S>
    portable explicit Derivable(const S& v);

#if TADA_ORDER > 1
    portable Derivable(const T& v, const T& d, const T& dd);
    template <typename S>
    portable Derivable(const S& v, const S& d, const S& dd);
#endif

    const portable T& v() const;

    const portable T& d() const;
#if TADA_ORDER > 1
    const portable T& dd() const;
#endif

    portable Derivable derive() const;

    portable Derivable& operator=(const T& u);

    portable Derivable& operator=(const Derivable& u);

    template <typename S>
    portable Derivable& operator=(const S& u);

    portable Derivable& operator+=(const T& u);

    portable Derivable& operator+=(const Derivable& u);

    portable Derivable& operator-=(const T& u);

    portable Derivable& operator-=(const Derivable& u);

    portable Derivable& operator*=(const T& u);

    portable Derivable& operator*=(const Derivable& u);

    portable Derivable& operator/=(const T& u);

    portable Derivable& operator/=(const Derivable& u);
};

/** Default constructor */
template <typename T>
portable Derivable<T>::Derivable()
    = default;

/** Constructor for `Derivable`s from singletons */
template <typename T>
portable Derivable<T>::Derivable(const T& v)
    : value(v)
{
}

/** Constructor for `Derivable`s with specifier */
template <typename T>
portable Derivable<T>::Derivable(const T& v, const T& d)
    : value(v)
    , deriv(d)
{
}

/** Constructor for `Derivable`s with specifier */
#if TADA_ORDER > 1
template <typename T>
portable Derivable<T>::Derivable(const T& v, const T& d, const T& dd)
    : value(v)
    , deriv(d)
    , dderiv(dd)
{
}
#endif

/** Constructor for `Derivable`s from singletons of a different type */
template <typename T>
template <typename S>
portable Derivable<T>::Derivable(const S& v)
    : value(static_cast<T>(v))
{
}

/** Constructor for `Derivable`s with specifier and different types */
#if TADA_ORDER > 1
template <typename T>
template <typename S>
portable Derivable<T>::Derivable(const S& v, const S& d, const S& dd)
    : value(static_cast<T>(v))
    , deriv(static_cast<T>(d))
    , dderiv(static_cast<T>(dd))
{
}
#endif

template <typename T>
portable T zero()
{
    return static_cast<T>(0);
}

template <typename T>
portable T one()
{
    return static_cast<T>(1);
}

template <typename T>
portable T two()
{
    return static_cast<T>(2);
}

/** Access to the current value */
template <typename T>
portable const T& Derivable<T>::v() const
{
    return value;
}

/** Access to the first-order derivative */
template <typename T>
portable const T& Derivable<T>::d() const
{
    return deriv;
}

#if TADA_ORDER > 1
/** Access to the second-order derivative */
template <typename T>
portable const T& Derivable<T>::dd() const
{
    return dderiv;
}
#endif

/** Make a variable the independent variable */
template <typename T>
portable Derivable<T> Derivable<T>::derive() const
{
    return Derivable(value, 1);
}

/** Overloaded assignment operator for singletons */
template <typename T>
portable Derivable<T>& Derivable<T>::operator=(const T& u)
{
    value = u;
    deriv = zero<T>();
#if TADA_ORDER > 1
    dderiv = zero<T>();
#endif
    return *this;
}

/** Overloaded assignment operator for `Derivable`s */
template <typename T>
portable Derivable<T>& Derivable<T>::operator=(const Derivable& u)
{
    value = u.v();
    deriv = u.d();
#if TADA_ORDER > 1
    dderiv = u.dd();
#endif
    return *this;
}

/** Overloaded assignment operator for singletons of a different type */
template <typename T>
template <typename S>
portable Derivable<T>& Derivable<T>::operator=(const S& u)
{
    value = static_cast<T>(u);
    deriv = zero<T>();
#if TADA_ORDER > 1
    dderiv = zero<T>();
#endif
    return *this;
}

/** Overloaded compound assignment sum */
template <typename T>
portable Derivable<T>& Derivable<T>::operator+=(const Derivable& u)
{
    value += u.v();
    deriv += u.d();
#if TADA_ORDER > 1
    dderiv += u.dd();
#endif
    return *this;
}

/** Overloaded compound assignment sum for singletons */
template <typename T>
portable Derivable<T>& Derivable<T>::operator+=(const T& u)
{
    value += u;
    return *this;
}

/** Overloaded compound assignment difference */
template <typename T>
portable Derivable<T>& Derivable<T>::operator-=(const Derivable& u)
{
    value -= u.v();
    deriv -= u.d();
#if TADA_ORDER > 1
    dderiv -= u.dd();
#endif
    return *this;
}

/** Overloaded compound assignment difference for singletons*/
template <typename T>
portable Derivable<T>& Derivable<T>::operator-=(const T& u)
{
    value -= u;
    return *this;
}

/** Overloaded compound assignment product */
template <typename T>
portable Derivable<T>& Derivable<T>::operator*=(const Derivable& u)
{
#if TADA_ORDER > 1
    dderiv = value * u.dd() + (two<T>() * deriv * u.d()) + dderiv * u.v();
#endif
    deriv = value * u.d() + deriv * u.v();
    value *= u.v();
    return *this;
}

/** Overloaded compound assignment product for singletons */
template <typename T>
portable Derivable<T>& Derivable<T>::operator*=(const T& u)
{
#if TADA_ORDER > 1
    dderiv *= u;
#endif
    deriv *= u;
    value *= u;
    return *this;
}

/** Overloaded compound assignment division */
template <typename T>
portable Derivable<T>& Derivable<T>::operator/=(const Derivable& u)
{
    // TODO catch u.x() == 0
    T vdu = value / u.v();
#if TADA_ORDER > 1
    dderiv = (dderiv - two<T>() * vdu * u.d() - vdu * u.dd()) / u.v();
#endif
    deriv = (deriv - vdu * u.d()) / u.v();
    value = vdu;
    return *this;
}

/** Overloaded compound assignment division for singletons */
template <typename T>
portable Derivable<T>& Derivable<T>::operator/=(const T& u)
{
    // TODO catch u == 0
#if TADA_ORDER > 1
    dderiv = dderiv / u;
#endif
    deriv = deriv / u;
    value = value / u;
    return *this;
}

/** Overloaded addition operator */
template <typename T>
portable Derivable<T>& operator+(const Derivable<T>& x)
{
    return x;
}

template <typename T>
portable Derivable<T> operator+(const Derivable<T>& x, const T& y)
{
    return Derivable<T>(x.v() + y, x.d()
#if TADA_ORDER > 1
                                       ,
        x.dd()
#endif
    );
}

template <typename T>
portable Derivable<T> operator+(const Derivable<T>& x,
    const Derivable<T>& y)
{
    return Derivable<T>(x.v() + y.v(), x.d() + y.d()
#if TADA_ORDER > 1
                                           ,
        x.dd() + y.dd()
#endif
    );
}

template <typename T, typename S>
portable Derivable<T> operator+(const Derivable<T>& x, const S& y)
{
    return Derivable<T>(x.v() + static_cast<T>(y),
        x.d()
#if TADA_ORDER > 1
            ,
        x.dd()
#endif
    );
}

template <typename T, typename S>
portable Derivable<T> operator+(const S& x, const Derivable<T>& y)
{
    return y + x;
}

/** Overloaded subtraction operator */
template <typename T>
portable Derivable<T> operator-(const Derivable<T>& x)
{
    return Derivable<T>(-x.v(), -x.d()
#if TADA_ORDER > 1
                                    ,
        -x.dd()
#endif
    );
}

template <typename T>
portable Derivable<T> operator-(const Derivable<T>& x, const T& y)
{
    return Derivable<T>(x.v() - y, x.d()
#if TADA_ORDER > 1
                                       ,
        x.dd()
#endif
    );
}

template <typename T>
portable Derivable<T> operator-(const Derivable<T>& x,
    const Derivable<T>& y)
{
    return Derivable<T>(x.v() - y.v(), x.d() - y.d()
#if TADA_ORDER > 1
                                           ,
        x.dd() - y.dd()
#endif
    );
}

template <typename T, typename S>
portable Derivable<T> operator-(const Derivable<T>& x, const S& y)
{
    return Derivable<T>(x.v() - static_cast<T>(y),
        x.d()
#if TADA_ORDER > 1
            ,
        x.dd()
#endif
    );
}

template <typename T, typename S>
portable Derivable<T> operator-(const S& x, const Derivable<T>& y)
{
    return Derivable<T>(x - y.v(),
        -y.d()
#if TADA_ORDER > 1
            ,
        -y.dd()
#endif
    );
}

/** Overloaded multiplication operator */
template <typename T>
portable Derivable<T> operator*(const Derivable<T>& x, const T& y)
{
    return Derivable<T>(
        x.v() * y, x.d() * y
#if TADA_ORDER > 1
        ,
        x.dd() * y
#endif
    );
}

template <typename T>
portable Derivable<T> operator*(const Derivable<T>& x,
    const Derivable<T>& y)
{
    return Derivable<T>(
        x.v() * y.v(), x.v() * y.d() + x.d() * y.v()
#if TADA_ORDER > 1
                           ,
        x.v() * y.dd() + two<T>() * x.d() * y.d() + x.dd() * y.v()
#endif
    );
}

template <typename T, typename S>
portable Derivable<T> operator*(const Derivable<T>& x, const S& y)
{
    T _y = static_cast<T>(y);
    return Derivable<T>(
        x.v() * _y, x.d() * _y
#if TADA_ORDER > 1
        ,
        x.dd() * _y
#endif
    );
}

template <typename T, typename S>
portable Derivable<T> operator*(const S& x, const Derivable<T>& y)
{
    T _x = static_cast<T>(x);
    return Derivable<T>(
        _x * y.v(), _x * y.d()
#if TADA_ORDER > 1
                        ,
        _x * y.dd()
#endif
    );
}

/** Overloaded division operator */
template <typename T>
portable Derivable<T> operator/(const Derivable<T>& x, const T& y)
{
    T xdy = x.v() / y;
    return Derivable<T>(
        xdy, x.d() / y
#if TADA_ORDER > 1
        ,
        x.dd() / y
#endif
    );
}

template <typename T>
portable Derivable<T> operator/(const Derivable<T>& x,
    const Derivable<T>& y)
{
    T xdy = x.v() / y.v();
    return Derivable<T>(
        xdy, (x.d() - xdy * y.d()) / y.v()
#if TADA_ORDER > 1
                 ,
        (x.dd() - two<T>() * xdy * y.d() - xdy * y.dd()) / y.v()
#endif
    );
}

template <typename T, typename S>
portable Derivable<T> operator/(const Derivable<T>& x, const S& y)
{
    T _y = static_cast<T>(y);
    return Derivable<T>(
        x.v() / _y, x.d() / _y
#if TADA_ORDER > 1
        ,
        x.dd() / _y
#endif
    );
}

template <typename T, typename S>
portable Derivable<T> operator/(const S& x, const Derivable<T>& y)
{
    T xdy = x / y.v();
    return Derivable<T>(xdy, -(xdy * y.d()) / y.v()
#if TADA_ORDER > 1
                                 ,
        (-two<T>() * xdy * y.d() - xdy * y.dd()) / y.v()
#endif
    );
}

template <typename T>
portable T square(const T& x);

template <>
portable inline int square(const int& x) { return x * x; }

template <>
portable inline float square(const float& x)
{
    using _std::pow;
    return powf(x, 2);
}

template <>
portable inline double square(const double& x)
{
    using _std::pow;
    return pow(x, 2);
}

template <typename T>
portable T sqr(const T& x) { return square(x); }

template <typename T>
portable Derivable<T> square(const Derivable<T>& x)
{
    return Derivable<T>(square(x.v()), x.d() * 2 * x.v()
#if TADA_ORDER > 1
                                           ,
        two<T>() * (square(x.d()) + x.v() * x.dd())
#endif
    );
}

template <typename T, typename S>
portable Derivable<T> pow(const Derivable<T>& x, const S& p)
{
    // TODO catch p == 0
    using _std::pow;
    T _t = p * pow(x.v(), p - 1);
    return Derivable<T>(
        pow(x.v(), p), x.d() * _t
#if TADA_ORDER > 1
        ,
        x.dd() * _t + p * (p - 1) * pow(x.v(), p - 2) * sqr(x.d())
#endif
    );
}

template <typename T>
portable Derivable<T> sqrt(const Derivable<T>& x)
{
    using _std::sqrt;
    T sqrtx = sqrt(x.v());
    T _t = 0.5 / sqrtx;
    return Derivable<T>(sqrtx, x.d() * _t
#if TADA_ORDER > 1
        ,
        (x.dd() * _t) - 0.5 * x.d() / (x.v() * x.d() * _t)
#endif
    );
}

template <typename T>
portable Derivable<T> sin(const Derivable<T>& x)
{
    using _std::cos;
    using _std::sin;
    T _f0 = sin(x.v());
    T _f1 = cos(x.v());
    return Derivable<T>(_f0, x.d() * _f1
#if TADA_ORDER > 1
        ,
        x.dd() * _f1 - _f0 * sqr(x.d())
#endif
    );
}

template <typename T>
portable Derivable<T> cos(const Derivable<T>& x)
{
    using _std::cos;
    using _std::sin;
    T _f0 = cos(x.v());
    T _f1 = -sin(x.v());
    return Derivable<T>(_f0, x.d() * _f1
#if TADA_ORDER > 1
        ,
        x.dd() * _f1 - _f0 * sqr(x.d())
#endif
    );
}

template <typename T>
portable Derivable<T> exp(const Derivable<T>& x)
{
    using _std::exp;
    T _f0 = exp(x.v());
    return Derivable<T>(_f0, x.d() * _f0
#if TADA_ORDER > 1
        ,
        //_f0 * (x.dd() + sqr(x.d()))
        _f0 * x.dd() + _f0 * sqr(x.d())
#endif
    );
}

template <typename T>
portable Derivable<T> log(const Derivable<T>& x)
{
    using _std::log;
    T _f0 = log(x.v());
    return Derivable<T>(_f0, x.d() / x.v()
#if TADA_ORDER > 1
                                 ,
        (x.dd() / x.v()) - (_f0 * sqr(x.d()))
#endif
    );
}

/** Wrapper for calculating the derivative of a univariate function */
template <typename Function, typename T>
portable Derivable<T> derivative(Function f, const Derivable<T>& x)
{
    return f(x.derive());
}

/** Shortcut for declaring independent variables */
template <typename T>
portable Derivable<T> Ivar(const T& x)
{
    return Derivable<T>(x, one<T>());
}

template <typename T>
portable Derivable<T> Ivar(const Derivable<T>& x)
{
    return Derivable<T>(x.v(), one<T>());
}

template <typename T, typename S>
portable Derivable<T> Ivar(const S& x)
{
    return Derivable<T>(static_cast<T>(x), one<T>());
}

/** Wrapper for calculating partial derivatives of a multivariate function */
template <typename Function, typename T, size_t N>
portable _std::array<Derivable<T>, N>
gradient(Function f, const _std::array<Derivable<T>, N> xs)
{
    _std::array<Derivable<T>, N> rv = xs;
    for (auto n = 0; n < N; n++) {
        auto cv = xs;
        cv[n] = cv[n].derive();
        rv[n] = f(cv);
    }
    return rv;
}

#ifdef TADA_ENABLE_EXPERIMENTAL
template <typename Function, typename Tuple, size_t... N>
auto gradient_impl(Function f, Tuple args, std::index_sequence<N...>)
{
    return std::make_tuple([&] {
      auto cargs = args;
      std::get<N>(cargs) = std::get<N>(args).derive();
      return std::apply(f, cargs); }()...);
}

template <typename Function, typename... Ts>
auto gradient(Function f, Ts... args)
{
    return gradient_impl(f, std::make_tuple(args...),
        std::make_index_sequence<sizeof...(Ts)> {});
}
#else
template <typename Function, typename... Ts>
auto gradient(Function f, Ts&&... args)
{
    return _std::make_tuple([&] {
      auto cargs = args;
      _std::get<N>(cargs) = std::get<N>(args).derive();
      return std::apply(f, cargs); }()...);
}
#endif
} // namespace tada

#endif // TADA_HPP
