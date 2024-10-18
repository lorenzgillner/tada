#define BOOST_TEST_MODULE derivative
#include "config.hpp"

using namespace tada;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_derive, T, test_types) {
  using dtype = Derivable<T>;

  dtype x(2);

  auto f = [&](dtype x) { return 2 * x; };

  dtype dfdx = derivative(f, x);

  BOOST_TEST(dfdx.v() == 4.0);
  BOOST_TEST(dfdx.d() == 2.0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_gradient, T, test_types) {
  using dtype = Derivable<T>;
  using darray = std::array<dtype, 2>;

  dtype x0(2);
  dtype x1(3);

  darray x = {x0, x1};

  auto g = [&](darray x) { return (2 * x[0]) + (2 * x[1]) - (x[0] * x[1]); };

  darray df = gradient(g, x);

  BOOST_TEST(df[0].v() == 4.0);
  BOOST_TEST(df[0].d() == -1.0);
  BOOST_TEST(df[1].v() == 4.0);
  BOOST_TEST(df[1].d() == 0.0);
}
