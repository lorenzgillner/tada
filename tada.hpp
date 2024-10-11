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

 // XXX use template paramter for distinction/specifier instead of constructor argument?
template <typename T>
class Derivable
{
public:
    Derivable(const T &v);
    Derivable(const T &v, const T &d);
    template <typename S> Derivable(const S &v);
    template <typename S, typename U> Derivable(const S &v, const U &d);
    ~Derivable();

    T &v();
    T &d();

    Derivable &operator=(const T &u);
    Derivable &operator=(const Derivable &u); // TODO casts between `Derivable`s of different types
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

/** Constructor for `Derivable`s from singletons */
template <typename T>
Derivable<T>::Derivable(const T &v) : value(v), deriv(static_cast<T>(CONSTANT)) {}

/** Constructor for `Derivable`s with specifier */
template <typename T>
Derivable<T>::Derivable(const T &v, const T &d) : value(v), deriv(d) {} // TODO handle impossible specifiers

/** Constructor for `Derivable`s from singletons of a different type */
template <typename T> template <typename S>
Derivable<T>::Derivable(const S &v) : value(static_cast<T>(v)), deriv(static_cast<T>(CONSTANT)) {}

/** Constructor for `Derivable`s with specifier and different types */
template <typename T> template <typename S, typename U>
Derivable<T>::Derivable(const S &v, const U &d) : value(static_cast<T>(v)), deriv(static_cast<T>(d)) {} // TODO handle impossible specifiers

/** Access to the current value */
template <typename T>
T& Derivable<T>::v() { return value; }

/** Access to the derivative */
template <typename T>
T& Derivable<T>::d() { return deriv; }

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
    value = static_cast<T>(u.v());
    deriv = static_cast<T>(CONSTANT);
    return *this;
}

template <typename T>
Derivable<T> &Derivable<T>::operator+=(const Derivable<T> &u)
{
    value += u.v();
    deriv += u.d();
    return *this;
}

template <typename T>
Derivable<T> &Derivable<T>::operator-=(const Derivable<T> &u)
{
    value -= u.v();
    deriv -= u.d(); // TODO what if deriv < 0?
    return *this;
}

template <typename T>
Derivable<T> &Derivable<T>::operator*=(const Derivable<T> &u)
{
    deriv = value * u.d() + deriv * u.v();
    value *= u.v();
    return *this;
}

template <typename T>
Derivable<T> &Derivable<T>::operator/=(const Derivable<T> &u)
{
    // TODO catch v.x() = 0
    T vdu = value / u.v();
    deriv = (deriv - (vdu * u.d())) / u.v();
    value = udv;
    return *this;
}

// TODO plus and minus sign
// TODO operator overloading for singletons

// template <typename T, typename S>
// Derivable<T> operator+(const Derivable<T> &u, const S &v)
// {
//     Derivable<T> _v(v, CONSTANT);
//     return Derivable(u.x() + v.x(), u.d() + v.d());
// }

// template <typename T>
// Derivable<T> operator-(const Derivable<T> &u, const Derivable<T> &v)
// {
//     return Derivable(u.x() - v.x(), u.d() - v.d());
// }

// template <typename T>
// Derivable<T> operator*(const Derivable<T> &u, const Derivable<T> &v)
// {
//     return Derivable(u.x() * v.x(), u.x() * v.d() + u.d() * v.x());
// }

// template <typename T>
// Derivable<T> operator/(const Derivable<T> &u, const Derivable<T> &v)
// {
//     T udv = u.x() / v.x();
//     return Derivable(udv, (u.d() - (udv * v.d())) / v.x());
// }

// TODO instead of this:
// template <typename T>
// class Variable final : Derivable<T>
// {
//     Variable(const T &x) : value(x), deriv(T(1)) {};
// };
// template <typename T>
// class Constant final : Derivable<T>
// {
//     Constant(const T &x) : value(x), deriv(T(0)) {};
// };
// create "shortcuts" for Variable, Constant, DVar etc.

}

#endif // TADA_HPP