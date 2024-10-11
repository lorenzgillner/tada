#include <iostream>

#include "tada.hpp"

using namespace tada;

typedef double itype;
typedef Derivable<itype> dtype;

dtype f(dtype x, dtype y) { return x + y + x * y; }

int main()
{
    dtype x(2, VARIABLE);
    dtype y(3, CONSTANT);

    dtype fxy = f(x, y);

    std::cout << "f(x,y) = " << fxy.x() << "\n"
              << "df/dx(x, y) = " << fxy.d() << std::endl;
}