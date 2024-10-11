#include <iostream>
#include <cmath>

#include "tada.hpp"

using namespace tada;

typedef double itype;
typedef Derivable<itype> dtype;

dtype f_add(dtype x)    { return x + 2; }
dtype f_sub(dtype x)    { return x - 2; }
dtype f_mul(dtype x)    { return x * 2; }
dtype f_div(dtype x)    { return x / 2; }
dtype f_square(dtype x) { return square(x); }
dtype f_pow2(dtype x)   { return pow(x, itype(2)); }
dtype f_pow3(dtype x)   { return pow(x, itype(3)); }
dtype f_sqrt(dtype x)   { return sqrt(x); }
dtype f_sin(dtype x)    { return sin(x); }
dtype f_cos(dtype x)    { return cos(x); }
dtype f_exp(dtype x)    { return exp(x); }
dtype f_log(dtype x)    { return log(x); }

int main()
{
    dtype x(2, VARIABLE);

    dtype addx    = f_add(x);
    dtype subx    = f_sub(x);
    dtype mulx    = f_mul(x);
    dtype divx    = f_div(x);
    dtype squarex = f_square(x);
    dtype pow2x   = f_pow2(x);
    dtype pow3x   = f_pow3(x);
    dtype sqrtx   = f_sqrt(x);
    dtype sinx    = f_sin(x);
    dtype cosx    = f_cos(x);
    dtype expx    = f_exp(x);
    dtype logx    = f_log(x);

    std::cout << addx.v()    << "," << addx.d()    << "\n";
    std::cout << subx.v()    << "," << subx.d()    << "\n";
    std::cout << mulx.v()    << "," << mulx.d()    << "\n";
    std::cout << divx.v()    << "," << divx.d()    << "\n";
    std::cout << squarex.v() << "," << squarex.d() << "\n";
    std::cout << pow2x.v()   << "," << pow2x.d()   << "\n";
    std::cout << pow3x.v()   << "," << pow3x.d()   << "\n";
    std::cout << sqrtx.v()   << "," << sqrtx.d()   << "\n";
    std::cout << sinx.v()    << "," << sinx.d()    << "\n";
    std::cout << cosx.v()    << "," << cosx.d()    << "\n";
    std::cout << expx.v()    << "," << expx.d()    << "\n";
    std::cout << logx.v()    << "," << logx.d()    << "\n";
}