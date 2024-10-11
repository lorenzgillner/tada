#ifndef TADA_HPP
#define TADA_HPP

#ifdef __NVCC__
#define PORTABLE __host__ __device__
#else
#define PORTABLE
#endif

#define VARIABLE 1
#define CONSTANT 0
#define INDEPEND 1

namespace tada {

enum specifier { constant, independent };

// XXX use template paramter for distinction/specifier instead of constructor argument?
template <typename T>
class Derivable
{
public:
    Derivable(const T &v);
    Derivable(const T &v, const T &d);
    template <typename S> Derivable(const S &v);
    template <typename S, typename U> Derivable(const S &v, const U &d);

    const T &v() const;
    const T &d() const;

    Derivable &operator=(const T &u);
    Derivable &operator=(const Derivable &u); // TODO casts between `Derivable`s of different types?
    template <typename S> Derivable &operator=(const S &u);

    Derivable &operator+=(const T &u);
    Derivable &operator+=(const Derivable &u);
    Derivable &operator-=(const T &u);
    Derivable &operator-=(const Derivable &u);
    Derivable &operator*=(const T &u);
    Derivable &operator*=(const Derivable &u);
    Derivable &operator/=(const T &u);
    Derivable &operator/=(const Derivable &u);

private:
    T value;
    T deriv;
};

// TODO variable and constant "shortcuts"

/** Constructor for `Derivable`s from singletons */
template <typename T>
Derivable<T>::Derivable(const T &v):
value(v), deriv(static_cast<T>(CONSTANT)) {}

/** Constructor for `Derivable`s with specifier */
template <typename T>
Derivable<T>::Derivable(const T &v, const T &d):
value(v), deriv(d) {} // TODO handle impossible specifiers

/** Constructor for `Derivable`s from singletons of a different type */
template <typename T> template <typename S>
Derivable<T>::Derivable(const S &v):
value(static_cast<T>(v)), deriv(static_cast<T>(CONSTANT)) {}

/** Constructor for `Derivable`s with specifier and different types */
template <typename T> template <typename S, typename U>
Derivable<T>::Derivable(const S &v, const U &d):
value(static_cast<T>(v)), deriv(static_cast<T>(d)) {} // TODO handle impossible specifiers

/** Access to the current value */
template <typename T>
const T& Derivable<T>::v() const
{
    return value;
}

/** Access to the derivative */
template <typename T>
const T& Derivable<T>::d() const
{
    return deriv;
}

/** Overloaded assignment operator for singletons */
template <typename T>
Derivable<T> &Derivable<T>::operator=(const T &u)
{
    value = u;
    deriv = static_cast<T>(CONSTANT);
    return *this;
}

/** Overloaded assignment operator for `Derivable`s */
template <typename T>
Derivable<T> &Derivable<T>::operator=(const Derivable &u)
{
    value = u.v();
    deriv = u.d();
    return *this;
}

/** Overloaded assignment operator for singletons of a different type */
template <typename T> template <typename S>
Derivable<T> &Derivable<T>::operator=(const S &u)
{
    value = static_cast<T>(u);
    deriv = static_cast<T>(CONSTANT);
    return *this;
}

/** Overloaded compound assignment sum */
template <typename T>
Derivable<T> &Derivable<T>::operator+=(const Derivable<T> &u)
{
    value += u.v();
    deriv += u.d();
    return *this;
}

/** Overloaded compound assignment sum for singletons */
template <typename T>
Derivable<T> &Derivable<T>::operator+=(const T &u)
{
    value += u;
    deriv += static_cast<T>(CONSTANT);
    return *this;
}

/** Overloaded compound assignment difference */
template <typename T>
Derivable<T> &Derivable<T>::operator-=(const Derivable<T> &u)
{
    value -= u.v();
    deriv -= u.d(); // TODO what if deriv < 0?
    return *this;
}

/** Overloaded compound assignment difference for singletons*/
template <typename T>
Derivable<T> &Derivable<T>::operator-=(const T &u)
{
    value -= u;
    deriv -= static_cast<T>(CONSTANT);
    return *this;
}

/** Overloaded compound assignment product */
template <typename T>
Derivable<T> &Derivable<T>::operator*=(const Derivable<T> &u)
{
    deriv = value * u.d() + deriv * u.v();
    value *= u.v();
    return *this;
}

/** Overloaded compound assignment product for singletons */
template <typename T>
Derivable<T> &Derivable<T>::operator*=(const T &u)
{
    deriv = value * static_cast<T>(CONSTANT) + deriv * u;
    value *= u;
    return *this;
}

/** Overloaded compound assignment division */
template <typename T>
Derivable<T> &Derivable<T>::operator/=(const Derivable<T> &u)
{
    // TODO catch u.x() == 0
    T vdu = value / u.v();
    deriv = (deriv - (vdu * u.d())) / u.v();
    value = vdu;
    return *this;
}

/** Overloaded compound assignment division for singletons */
template <typename T>
Derivable<T> &Derivable<T>::operator/=(const T &u)
{
    // TODO catch u == 0
    T vdu = value / u;
    deriv = (deriv - (vdu * static_cast<T>(CONSTANT))) / u;
    value = vdu;
    return *this;
}

/** Overloaded addition operator */
template <typename T>
Derivable<T> &operator+(const Derivable<T> &x)
{
    return x;
}

template <typename T>
Derivable<T> operator+(const Derivable<T> &x, const T &y)
{
    return Derivable<T>(x.v() + y, x.d() + static_cast<T>(CONSTANT));
}

template <typename T>
Derivable<T> operator+(const Derivable<T> &x, const Derivable<T> &y)
{
    return Derivable<T>(x.v() + y.v(), x.d() + y.d());
}

template <typename T, typename S>
Derivable<T> operator+(const Derivable<T> &x, const S &y)
{
    return Derivable<T>(x.v() + static_cast<T>(y), x.d() + static_cast<T>(CONSTANT));
}

template <typename T, typename S>
Derivable<T> operator+(const S &x, const Derivable<T> &y)
{
    return y + x;
}

/** Overloaded subtraction operator */
template <typename T>
Derivable<T> operator-(const Derivable<T> &x)
{
    return Derivable<T>(-x.v(), -x.d()); // XXX is this valid?
}

template <typename T>
Derivable<T> operator-(const Derivable<T> &x, const T &y)
{
    return Derivable<T>(x.v() - y, x.d() - static_cast<T>(CONSTANT));
}

template <typename T>
Derivable<T> operator-(const Derivable<T> &x, const Derivable<T> &y)
{
    return Derivable<T>(x.v() - y.v(), x.d() - y.d());
}

template <typename T, typename S>
Derivable<T> operator-(const Derivable<T> &x, const S &y)
{
    return Derivable<T>(x.v() - static_cast<T>(y), x.d() - static_cast<T>(CONSTANT));
}

template <typename T, typename S>
Derivable<T> operator-(const S &x, const Derivable<T> &y)
{
    return Derivable<T>(static_cast<T>(x) - y.v(), static_cast<T>(CONSTANT) - y.d());
}

/** Overloaded multiplication operator */
template <typename T>
Derivable<T> operator*(const Derivable<T> &x, const T &y)
{
    return Derivable<T>(x.v() * y, x.v() * static_cast<T>(CONSTANT) + x.d() * y);
}

template <typename T>
Derivable<T> operator*(const Derivable<T> &x, const Derivable<T> &y)
{
    return Derivable<T>(x.v() * y.v(), x.v() * y.d() + x.d() * y.v());
}

template <typename T, typename S>
Derivable<T> operator*(const Derivable<T> &x, const S &y)
{
    return Derivable<T>(x.v() * static_cast<T>(y), x.v() * static_cast<T>(CONSTANT) + x.d() * static_cast<T>(y));
}

template <typename T, typename S>
Derivable<T> operator*(const S &x, const Derivable<T> &y)
{
    return Derivable<T>(static_cast<T>(x) * y.v(), static_cast<T>(x) * y.d() + static_cast<T>(CONSTANT) * y.v());
}

/** Overloaded division operator */
template <typename T>
Derivable<T> operator/(const Derivable<T> &x, const T &y)
{
    T xdy = x.v() / y;
    return Derivable<T>(xdy, (x.d() - (xdy * static_cast<T>(CONSTANT))) / y);
}

template <typename T>
Derivable<T> operator/(const Derivable<T> &x, const Derivable<T> &y)
{
    T xdy = x.v() / y.v();
    return Derivable<T>(xdy, (x.d() - (xdy * y.d())) / y.v());
}

template <typename T, typename S>
Derivable<T> operator/(const Derivable<T> &x, const S &y)
{
    T _y = static_cast<T>(y);
    T xdy = x.v() / _y;
    return Derivable<T>(xdy, (x.d() - (xdy * static_cast<T>(CONSTANT))) / _y);
}

template <typename T, typename S>
Derivable<T> operator/(const S &x, const Derivable<T> &y)
{
    T _x = static_cast<T>(x);
    T xdy = _x / y.v();
    return Derivable<T>(xdy, (static_cast<T>(CONSTANT) - (xdy * y.d())) / y.v());
}

template <typename T> T square(const T &x);
template <> int square(const int &x) { return x * x; }
template <> float square(const float &x) { return ::std::pow(x, 2); }
template <> double square(const double &x) { return ::std::pow(x, 2); }

template <typename T>
Derivable<T> square(const Derivable<T> &x)
{
    return Derivable<T>(square(x.v()), x.d() * 2 * x.v());
}

template <typename T>
Derivable<T> pow(const Derivable<T> &x, const T &p)
{
    // TODO catch p == 0
    using ::std::pow;
    return Derivable<T>(pow(x.v(), p), x.d() * p * pow(x.v(), p - 1));
}

template <typename T>
Derivable<T> sqrt(const Derivable<T> &x)
{
    using ::std::sqrt;
    T sqrtx = sqrt(x.v());
    return Derivable<T>(sqrtx, x.d() / (2 * sqrtx));
}

template <typename T>
Derivable<T> sin(const Derivable<T> &x)
{
    using ::std::sin;
    using ::std::cos;
    return Derivable<T>(sin(x.v()), x.d() * cos(x.v()));
}

template <typename T>
Derivable<T> cos(const Derivable<T> &x)
{
    using ::std::sin;
    using ::std::cos;
    return Derivable<T>(cos(x.v()), -x.d() * sin(x.v()));
}

template <typename T>
Derivable<T> exp(const Derivable<T> &x)
{
    using ::std::exp;
    T expx = exp(x.v());
    return Derivable<T>(expx, x.d() * expx);
}

template <typename T>
Derivable<T> log(const Derivable<T> &x)
{
    using ::std::log;
    return Derivable<T>(log(x.v()), x.d() / x.v());
}

}

#endif // TADA_HPP
