#ifndef TADA_HPP
#define TADA_HPP

#ifdef __NVCC__
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/utility>
#define __portable__ __host__ __device__
using namespace cuda;
#else
#include <array>
#include <cmath>
#include <utility>
#define __portable__
#endif

namespace tada {

enum specifier { constant, independent };

__portable__ template <typename T> class Derivable {
private:
  T value;
  T deriv;

public:
  Derivable(const T &v);
  Derivable(const T &v, const T &d);
  template <typename S> Derivable(const S &v);
  template <typename S, typename U> Derivable(const S &v, const U &d);

  const T &v() const;
  const T &d() const;

  const Derivable derive() const;

  Derivable &operator=(const T &u);
  Derivable &operator=(const Derivable &u);
  template <typename S> Derivable &operator=(const S &u);

  Derivable &operator+=(const T &u);
  Derivable &operator+=(const Derivable &u);
  Derivable &operator-=(const T &u);
  Derivable &operator-=(const Derivable &u);
  Derivable &operator*=(const T &u);
  Derivable &operator*=(const Derivable &u);
  Derivable &operator/=(const T &u);
  Derivable &operator/=(const Derivable &u);
};

/** Constructor for `Derivable`s from singletons */
__portable__ template <typename T>
Derivable<T>::Derivable(const T &v)
    : value(v), deriv(static_cast<T>(constant)) {}

/** Constructor for `Derivable`s with specifier */
__portable__ template <typename T>
Derivable<T>::Derivable(const T &v, const T &d) : value(v), deriv(d) {}

/** Constructor for `Derivable`s from singletons of a different type */
__portable__ template <typename T>
template <typename S>
Derivable<T>::Derivable(const S &v)
    : value(static_cast<T>(v)), deriv(static_cast<T>(constant)) {}

/** Constructor for `Derivable`s with specifier and different types */
__portable__ template <typename T>
template <typename S, typename U>
Derivable<T>::Derivable(const S &v, const U &d)
    : value(static_cast<T>(v)), deriv(static_cast<T>(d)) {}

/** Shortcut for declaring independent variables */
__portable__ template <typename T>
Derivable<T> Independent(const Derivable<T> &x) {
  return Derivable<T>(x.v(), static_cast<T>(independent));
}

__portable__ template <typename T> Derivable<T> Independent(const T &x) {
  return Derivable<T>(x, static_cast<T>(independent));
}

__portable__ template <typename T, typename S>
Derivable<T> Independent(const S &x) {
  return Derivable<T>(x, static_cast<T>(independent));
}

/** Shortcut for declaring constants */
__portable__ template <typename T>
Derivable<T> Constant(const Derivable<T> &x) {
  return Derivable<T>(x.v(), static_cast<T>(independent));
}

__portable__ template <typename T> Derivable<T> Constant(const T &x) {
  return Derivable<T>(x, static_cast<T>(independent));
}

__portable__ template <typename T, typename S>
Derivable<T> Constant(const S &x) {
  return Derivable<T>(x, static_cast<T>(independent));
}

/** Access to the current value */
__portable__ template <typename T> const T &Derivable<T>::v() const {
  return value;
}

/** Access to the derivative */
__portable__ template <typename T> const T &Derivable<T>::d() const {
  return deriv;
}

/** Make a variable the independent variable */
__portable__ template <typename T>
const Derivable<T> Derivable<T>::derive() const {
  return Derivable<T>(value, static_cast<T>(independent));
}

/** Overloaded assignment operator for singletons */
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator=(const T &u) {
  value = u;
  deriv = static_cast<T>(constant);
  return *this;
}

/** Overloaded assignment operator for `Derivable`s */
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator=(const Derivable &u) {
  value = u.v();
  deriv = u.d();
  return *this;
}

/** Overloaded assignment operator for singletons of a different type */
__portable__ template <typename T>
template <typename S>
Derivable<T> &Derivable<T>::operator=(const S &u) {
  value = static_cast<T>(u);
  deriv = static_cast<T>(constant);
  return *this;
}

/** Overloaded compound assignment sum */
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator+=(const Derivable<T> &u) {
  value += u.v();
  deriv += u.d();
  return *this;
}

/** Overloaded compound assignment sum for singletons */
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator+=(const T &u) {
  value += u;
  deriv += static_cast<T>(constant);
  return *this;
}

/** Overloaded compound assignment difference */
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator-=(const Derivable<T> &u) {
  value -= u.v();
  deriv -= u.d();
  return *this;
}

/** Overloaded compound assignment difference for singletons*/
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator-=(const T &u) {
  value -= u;
  deriv -= static_cast<T>(constant);
  return *this;
}

/** Overloaded compound assignment product */
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator*=(const Derivable<T> &u) {
  deriv = value * u.d() + deriv * u.v();
  value *= u.v();
  return *this;
}

/** Overloaded compound assignment product for singletons */
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator*=(const T &u) {
  deriv = value * static_cast<T>(constant) + deriv * u;
  value *= u;
  return *this;
}

/** Overloaded compound assignment division */
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator/=(const Derivable<T> &u) {
  // TODO catch u.x() == 0
  T vdu = value / u.v();
  deriv = (deriv - (vdu * u.d())) / u.v();
  value = vdu;
  return *this;
}

/** Overloaded compound assignment division for singletons */
__portable__ template <typename T>
Derivable<T> &Derivable<T>::operator/=(const T &u) {
  // TODO catch u == 0
  T vdu = value / u;
  deriv = (deriv - (vdu * static_cast<T>(constant))) / u;
  value = vdu;
  return *this;
}

/** Overloaded addition operator */
__portable__ template <typename T>
Derivable<T> &operator+(const Derivable<T> &x) {
  return x;
}

__portable__ template <typename T>
Derivable<T> operator+(const Derivable<T> &x, const T &y) {
  return Derivable<T>(x.v() + y, x.d() + static_cast<T>(constant));
}

__portable__ template <typename T>
Derivable<T> operator+(const Derivable<T> &x, const Derivable<T> &y) {
  return Derivable<T>(x.v() + y.v(), x.d() + y.d());
}

__portable__ template <typename T, typename S>
Derivable<T> operator+(const Derivable<T> &x, const S &y) {
  return Derivable<T>(x.v() + static_cast<T>(y),
                      x.d() + static_cast<T>(constant));
}

__portable__ template <typename T, typename S>
Derivable<T> operator+(const S &x, const Derivable<T> &y) {
  return y + x;
}

/** Overloaded subtraction operator */
__portable__ template <typename T>
Derivable<T> operator-(const Derivable<T> &x) {
  return Derivable<T>(-x.v(), -x.d());
}

__portable__ template <typename T>
Derivable<T> operator-(const Derivable<T> &x, const T &y) {
  return Derivable<T>(x.v() - y, x.d() - static_cast<T>(constant));
}

__portable__ template <typename T>
Derivable<T> operator-(const Derivable<T> &x, const Derivable<T> &y) {
  return Derivable<T>(x.v() - y.v(), x.d() - y.d());
}

__portable__ template <typename T, typename S>
Derivable<T> operator-(const Derivable<T> &x, const S &y) {
  return Derivable<T>(x.v() - static_cast<T>(y),
                      x.d() - static_cast<T>(constant));
}

__portable__ template <typename T, typename S>
Derivable<T> operator-(const S &x, const Derivable<T> &y) {
  return Derivable<T>(static_cast<T>(x) - y.v(),
                      static_cast<T>(constant) - y.d());
}

/** Overloaded multiplication operator */
__portable__ template <typename T>
Derivable<T> operator*(const Derivable<T> &x, const T &y) {
  return Derivable<T>(x.v() * y, x.v() * static_cast<T>(constant) + x.d() * y);
}

__portable__ template <typename T>
Derivable<T> operator*(const Derivable<T> &x, const Derivable<T> &y) {
  return Derivable<T>(x.v() * y.v(), x.v() * y.d() + x.d() * y.v());
}

__portable__ template <typename T, typename S>
Derivable<T> operator*(const Derivable<T> &x, const S &y) {
  return Derivable<T>(x.v() * static_cast<T>(y),
                      x.v() * static_cast<T>(constant) +
                          x.d() * static_cast<T>(y));
}

__portable__ template <typename T, typename S>
Derivable<T> operator*(const S &x, const Derivable<T> &y) {
  return Derivable<T>(static_cast<T>(x) * y.v(),
                      static_cast<T>(x) * y.d() +
                          static_cast<T>(constant) * y.v());
}

/** Overloaded division operator */
__portable__ template <typename T>
Derivable<T> operator/(const Derivable<T> &x, const T &y) {
  T xdy = x.v() / y;
  return Derivable<T>(xdy, (x.d() - (xdy * static_cast<T>(constant))) / y);
}

__portable__ template <typename T>
Derivable<T> operator/(const Derivable<T> &x, const Derivable<T> &y) {
  T xdy = x.v() / y.v();
  return Derivable<T>(xdy, (x.d() - (xdy * y.d())) / y.v());
}

__portable__ template <typename T, typename S>
Derivable<T> operator/(const Derivable<T> &x, const S &y) {
  T _y = static_cast<T>(y);
  T xdy = x.v() / _y;
  return Derivable<T>(xdy, (x.d() - (xdy * static_cast<T>(constant))) / _y);
}

__portable__ template <typename T, typename S>
Derivable<T> operator/(const S &x, const Derivable<T> &y) {
  T _x = static_cast<T>(x);
  T xdy = _x / y.v();
  return Derivable<T>(xdy, (static_cast<T>(constant) - (xdy * y.d())) / y.v());
}

__portable__ template <typename T> T square(const T &x);
__portable__ template <> int square(const int &x) { return x * x; }
__portable__ template <> float square(const float &x) {
  return ::std::pow(x, 2);
}
__portable__ template <> double square(const double &x) {
  return ::std::pow(x, 2);
}

__portable__ template <typename T> T sqr(const T &x) { return square(x); }

__portable__ template <typename T> Derivable<T> square(const Derivable<T> &x) {
  return Derivable<T>(square(x.v()), x.d() * 2 * x.v());
}

__portable__ template <typename T>
Derivable<T> pow(const Derivable<T> &x, const T &p) {
  // TODO catch p == 0
  using ::std::pow;
  return Derivable<T>(pow(x.v(), p), x.d() * p * pow(x.v(), p - 1));
}

__portable__ template <typename T> Derivable<T> sqrt(const Derivable<T> &x) {
  using ::std::sqrt;
  T sqrtx = sqrt(x.v());
  return Derivable<T>(sqrtx, x.d() / (2 * sqrtx));
}

__portable__ template <typename T> Derivable<T> sin(const Derivable<T> &x) {
  using ::std::cos;
  using ::std::sin;
  return Derivable<T>(sin(x.v()), x.d() * cos(x.v()));
}

__portable__ template <typename T> Derivable<T> cos(const Derivable<T> &x) {
  using ::std::cos;
  using ::std::sin;
  return Derivable<T>(cos(x.v()), -x.d() * sin(x.v()));
}

__portable__ template <typename T> Derivable<T> exp(const Derivable<T> &x) {
  using ::std::exp;
  T expx = exp(x.v());
  return Derivable<T>(expx, x.d() * expx);
}

__portable__ template <typename T> Derivable<T> log(const Derivable<T> &x) {
  using ::std::log;
  return Derivable<T>(log(x.v()), x.d() / x.v());
}

/** Wrapper for calculating the derivative of a univariate function */
__portable__ template <typename Function, typename T>
Derivable<T> derivative(Function f, const Derivable<T> &x) {
  return f(x.derive());
}

/** Wrapper for calculating partial derivatives of a multivariate function */
__portable__ template <typename Function, typename T, size_t N>
::std::array<Derivable<T>, N> gradient(Function f,
                                       const ::std::array<Derivable<T>, N> xs) {
  ::std::array<Derivable<T>, N> rv = xs;
  for (auto n = 0; n < N; n++) {
    auto cv = xs;
    cv[n] = cv[n].derive();
    rv[n] = f(cv);
  }
  return rv;
}

} // namespace tada

#endif // TADA_HPP
