#ifndef TADA_HPP
#define TADA_HPP

#ifndef __CUDACC__
#define EXPERIMENTAL
#endif

#ifdef __NVCC__
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/utility>
#define __portable__ __host__ __device__
#define _std cuda::std
#else
#include <array>
#include <cmath>
#include <utility>
#ifdef EXPERIMENTAL
#include <tuple>
#endif
#define __portable__
#define _std std
#endif

namespace tada {

enum specifier { constant, independent };

template <typename T> class Derivable {
private:
  T value;
  T deriv;

public:
  __portable__ Derivable(const T &v);
  __portable__ Derivable(const T &v, const T &d);
  template <typename S> __portable__ Derivable(const S &v);
  template <typename S, typename U>
  __portable__ Derivable(const S &v, const U &d);

  const __portable__ T &v() const;
  const __portable__ T &d() const;

  const __portable__ Derivable derive() const;

  __portable__ Derivable &operator=(const T &u);
  __portable__ Derivable &operator=(const Derivable &u);
  template <typename S> __portable__ Derivable &operator=(const S &u);

  __portable__ Derivable &operator+=(const T &u);
  __portable__ Derivable &operator+=(const Derivable &u);
  __portable__ Derivable &operator-=(const T &u);
  __portable__ Derivable &operator-=(const Derivable &u);
  __portable__ Derivable &operator*=(const T &u);
  __portable__ Derivable &operator*=(const Derivable &u);
  __portable__ Derivable &operator/=(const T &u);
  __portable__ Derivable &operator/=(const Derivable &u);
};

/** Constructor for `Derivable`s from singletons */
template <typename T>
__portable__ Derivable<T>::Derivable(const T &v)
    : value(v), deriv(static_cast<T>(constant)) {}

/** Constructor for `Derivable`s with specifier */
template <typename T>
__portable__ Derivable<T>::Derivable(const T &v, const T &d)
    : value(v), deriv(d) {}

/** Constructor for `Derivable`s from singletons of a different type */
template <typename T>
template <typename S>
__portable__ Derivable<T>::Derivable(const S &v)
    : value(static_cast<T>(v)), deriv(static_cast<T>(constant)) {}

/** Constructor for `Derivable`s with specifier and different types */
template <typename T>
template <typename S, typename U>
__portable__ Derivable<T>::Derivable(const S &v, const U &d)
    : value(static_cast<T>(v)), deriv(static_cast<T>(d)) {}

/** Shortcut for declaring independent variables */
template <typename T>
__portable__ Derivable<T> Independent(const Derivable<T> &x) {
  return Derivable<T>(x.v(), static_cast<T>(independent));
}

template <typename T> __portable__ Derivable<T> Independent(const T &x) {
  return Derivable<T>(x, static_cast<T>(independent));
}

template <typename T, typename S>
__portable__ Derivable<T> Independent(const S &x) {
  return Derivable<T>(x, static_cast<T>(independent));
}

/** Shortcut for declaring constants */
template <typename T>
__portable__ Derivable<T> Constant(const Derivable<T> &x) {
  return Derivable<T>(x.v(), static_cast<T>(independent));
}

template <typename T> __portable__ Derivable<T> Constant(const T &x) {
  return Derivable<T>(x, static_cast<T>(independent));
}

template <typename T, typename S>
__portable__ Derivable<T> Constant(const S &x) {
  return Derivable<T>(x, static_cast<T>(independent));
}

/** Access to the current value */
template <typename T> __portable__ const T &Derivable<T>::v() const {
  return value;
}

/** Access to the derivative */
template <typename T> __portable__ const T &Derivable<T>::d() const {
  return deriv;
}

/** Make a variable the independent variable */
template <typename T>
__portable__ const Derivable<T> Derivable<T>::derive() const {
  return Derivable<T>(value, static_cast<T>(independent));
}

/** Overloaded assignment operator for singletons */
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator=(const T &u) {
  value = u;
  deriv = static_cast<T>(constant);
  return *this;
}

/** Overloaded assignment operator for `Derivable`s */
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator=(const Derivable &u) {
  value = u.v();
  deriv = u.d();
  return *this;
}

/** Overloaded assignment operator for singletons of a different type */
template <typename T>
template <typename S>
__portable__ Derivable<T> &Derivable<T>::operator=(const S &u) {
  value = static_cast<T>(u);
  deriv = static_cast<T>(constant);
  return *this;
}

/** Overloaded compound assignment sum */
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator+=(const Derivable<T> &u) {
  value += u.v();
  deriv += u.d();
  return *this;
}

/** Overloaded compound assignment sum for singletons */
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator+=(const T &u) {
  value += u;
  deriv += static_cast<T>(constant);
  return *this;
}

/** Overloaded compound assignment difference */
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator-=(const Derivable<T> &u) {
  value -= u.v();
  deriv -= u.d();
  return *this;
}

/** Overloaded compound assignment difference for singletons*/
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator-=(const T &u) {
  value -= u;
  deriv -= static_cast<T>(constant);
  return *this;
}

/** Overloaded compound assignment product */
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator*=(const Derivable<T> &u) {
  deriv = value * u.d() + deriv * u.v();
  value *= u.v();
  return *this;
}

/** Overloaded compound assignment product for singletons */
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator*=(const T &u) {
  deriv = value * static_cast<T>(constant) + deriv * u;
  value *= u;
  return *this;
}

/** Overloaded compound assignment division */
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator/=(const Derivable<T> &u) {
  // TODO catch u.x() == 0
  T vdu = value / u.v();
  deriv = (deriv - (vdu * u.d())) / u.v();
  value = vdu;
  return *this;
}

/** Overloaded compound assignment division for singletons */
template <typename T>
__portable__ Derivable<T> &Derivable<T>::operator/=(const T &u) {
  // TODO catch u == 0
  T vdu = value / u;
  deriv = (deriv - (vdu * static_cast<T>(constant))) / u;
  value = vdu;
  return *this;
}

/** Overloaded addition operator */
template <typename T>
__portable__ Derivable<T> &operator+(const Derivable<T> &x) {
  return x;
}

template <typename T>
__portable__ Derivable<T> operator+(const Derivable<T> &x, const T &y) {
  return Derivable<T>(x.v() + y, x.d() + static_cast<T>(constant));
}

template <typename T>
__portable__ Derivable<T> operator+(const Derivable<T> &x,
                                    const Derivable<T> &y) {
  return Derivable<T>(x.v() + y.v(), x.d() + y.d());
}

template <typename T, typename S>
__portable__ Derivable<T> operator+(const Derivable<T> &x, const S &y) {
  return Derivable<T>(x.v() + static_cast<T>(y),
                      x.d() + static_cast<T>(constant));
}

template <typename T, typename S>
__portable__ Derivable<T> operator+(const S &x, const Derivable<T> &y) {
  return y + x;
}

/** Overloaded subtraction operator */
template <typename T>
__portable__ Derivable<T> operator-(const Derivable<T> &x) {
  return Derivable<T>(-x.v(), -x.d());
}

template <typename T>
__portable__ Derivable<T> operator-(const Derivable<T> &x, const T &y) {
  return Derivable<T>(x.v() - y, x.d() - static_cast<T>(constant));
}

template <typename T>
__portable__ Derivable<T> operator-(const Derivable<T> &x,
                                    const Derivable<T> &y) {
  return Derivable<T>(x.v() - y.v(), x.d() - y.d());
}

template <typename T, typename S>
__portable__ Derivable<T> operator-(const Derivable<T> &x, const S &y) {
  return Derivable<T>(x.v() - static_cast<T>(y),
                      x.d() - static_cast<T>(constant));
}

template <typename T, typename S>
__portable__ Derivable<T> operator-(const S &x, const Derivable<T> &y) {
  return Derivable<T>(static_cast<T>(x) - y.v(),
                      static_cast<T>(constant) - y.d());
}

/** Overloaded multiplication operator */
template <typename T>
__portable__ Derivable<T> operator*(const Derivable<T> &x, const T &y) {
  return Derivable<T>(x.v() * y, x.v() * static_cast<T>(constant) + x.d() * y);
}

template <typename T>
__portable__ Derivable<T> operator*(const Derivable<T> &x,
                                    const Derivable<T> &y) {
  return Derivable<T>(x.v() * y.v(), x.v() * y.d() + x.d() * y.v());
}

template <typename T, typename S>
__portable__ Derivable<T> operator*(const Derivable<T> &x, const S &y) {
  return Derivable<T>(x.v() * static_cast<T>(y),
                      x.v() * static_cast<T>(constant) +
                          x.d() * static_cast<T>(y));
}

template <typename T, typename S>
__portable__ Derivable<T> operator*(const S &x, const Derivable<T> &y) {
  return Derivable<T>(static_cast<T>(x) * y.v(),
                      static_cast<T>(x) * y.d() +
                          static_cast<T>(constant) * y.v());
}

/** Overloaded division operator */
template <typename T>
__portable__ Derivable<T> operator/(const Derivable<T> &x, const T &y) {
  T xdy = x.v() / y;
  return Derivable<T>(xdy, (x.d() - (xdy * static_cast<T>(constant))) / y);
}

template <typename T>
__portable__ Derivable<T> operator/(const Derivable<T> &x,
                                    const Derivable<T> &y) {
  T xdy = x.v() / y.v();
  return Derivable<T>(xdy, (x.d() - (xdy * y.d())) / y.v());
}

template <typename T, typename S>
__portable__ Derivable<T> operator/(const Derivable<T> &x, const S &y) {
  T _y = static_cast<T>(y);
  T xdy = x.v() / _y;
  return Derivable<T>(xdy, (x.d() - (xdy * static_cast<T>(constant))) / _y);
}

template <typename T, typename S>
__portable__ Derivable<T> operator/(const S &x, const Derivable<T> &y) {
  T _x = static_cast<T>(x);
  T xdy = _x / y.v();
  return Derivable<T>(xdy, (static_cast<T>(constant) - (xdy * y.d())) / y.v());
}

template <typename T> __portable__ T square(const T &x);
template <> __portable__ int square(const int &x) { return x * x; }
template <> __portable__ float square(const float &x) {
  using _std::pow;
  return pow(x, 2);
}
template <> __portable__ double square(const double &x) {
  using _std::pow;
  return pow(x, 2);
}

template <typename T> __portable__ T sqr(const T &x) { return square(x); }

template <typename T> __portable__ Derivable<T> square(const Derivable<T> &x) {
  return Derivable<T>(square(x.v()), x.d() * 2 * x.v());
}

template <typename T>
__portable__ Derivable<T> pow(const Derivable<T> &x, const T &p) {
  // TODO catch p == 0
  using _std::pow;
  return Derivable<T>(pow(x.v(), p), x.d() * p * pow(x.v(), p - 1));
}

template <typename T> __portable__ Derivable<T> sqrt(const Derivable<T> &x) {
  using _std::sqrt;
  T sqrtx = sqrt(x.v());
  return Derivable<T>(sqrtx, x.d() / (2 * sqrtx));
}

template <typename T> __portable__ Derivable<T> sin(const Derivable<T> &x) {
  using _std::cos;
  using _std::sin;
  return Derivable<T>(sin(x.v()), x.d() * cos(x.v()));
}

template <typename T> __portable__ Derivable<T> cos(const Derivable<T> &x) {
  using _std::cos;
  using _std::sin;
  return Derivable<T>(cos(x.v()), -x.d() * sin(x.v()));
}

template <typename T> __portable__ Derivable<T> exp(const Derivable<T> &x) {
  using _std::exp;
  T expx = exp(x.v());
  return Derivable<T>(expx, x.d() * expx);
}

template <typename T> __portable__ Derivable<T> log(const Derivable<T> &x) {
  using _std::log;
  return Derivable<T>(log(x.v()), x.d() / x.v());
}

/** Wrapper for calculating the derivative of a univariate function */
template <typename Function, typename T>
__portable__ Derivable<T> derivative(Function f, const Derivable<T> &x) {
  return f(x.derive());
}

/** Wrapper for calculating partial derivatives of a multivariate function */
template <typename Function, typename T, size_t N>
_std::array<Derivable<T>, N> gradient(Function f,
                                      const _std::array<Derivable<T>, N> xs) {
  _std::array<Derivable<T>, N> rv = xs;
  for (auto n = 0; n < N; n++) {
    auto cv = xs;
    cv[n] = cv[n].derive();
    rv[n] = f(cv);
  }
  return rv;
}

#ifdef EXPERIMENTAL
template <typename Function, typename Tuple, size_t... N>
auto gradient_impl(Function f, Tuple args, std::index_sequence<N...>) {
  return std::make_tuple([&] {
    auto cargs = args;
    std::get<N>(cargs) = std::get<N>(args).derive();
    return std::apply(f, cargs);
  }()...);
}

template <typename Function, typename... Ts>
auto gradient(Function f, Ts... args) {
  return gradient_impl(f, std::make_tuple(args...),
                       std::make_index_sequence<sizeof...(Ts)>{});
}
#endif

} // namespace tada

#endif // TADA_HPP
