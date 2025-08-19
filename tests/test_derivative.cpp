#define BOOST_TEST_MODULE derivative
#include "config.hpp"

using namespace tada;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_derive, T, test_types) {
  using dtype = Derivable<T>;

  dtype x0(2);

  auto f = [&](const dtype& x) { return 2 * x; };

  dtype dfdx = derivative(f, x0);

  BOOST_TEST(dfdx.v() == static_cast<T>(4));
  BOOST_TEST(dfdx.d() == static_cast<T>(2));
  BOOST_TEST(dfdx.dd() == static_cast<T>(0));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_gradient, T, test_types) {
  using dtype = Derivable<T>;
  using darray = std::array<dtype, 2>;

  dtype x0(2);
  dtype x1(3);

  darray X = {x0, x1};

  auto g = [&](darray x) { return 2 * x[0] + 2 * x[1] - x[0] * x[1]; };

  darray df = gradient(g, X);

  BOOST_TEST(df[0].v() == static_cast<T>(4));
  BOOST_TEST(df[0].d() == static_cast<T>(-1));
  BOOST_TEST(df[0].dd() == static_cast<T>(0));
  BOOST_TEST(df[1].v() == static_cast<T>(4));
  BOOST_TEST(df[1].d() == static_cast<T>(0));
  BOOST_TEST(df[1].dd() == static_cast<T>(0));
}
