#define BOOST_TEST_MODULE arithmetic
#include "config.hpp"

using namespace tada;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_add, T, test_types) {
  using dtype = Derivable<T>;
  auto f = [](const dtype& x) { return x + 2; };

  dtype x(2);
  dtype fx = f(x.derive());

  BOOST_TEST(fx.v() == 4.0);
  BOOST_TEST(fx.d() == 1.0);
  BOOST_TEST(fx.dd() == 0.0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_sub, T, test_types) {
  using dtype = Derivable<T>;
  auto f = [](const dtype& x) { return x - 2; };

  dtype x(2);
  dtype fx = f(x.derive());

  BOOST_TEST(fx.v() == 0.0);
  BOOST_TEST(fx.d() == 1.0);
  BOOST_TEST(fx.dd() == 0.0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_mul, T, test_types) {
  using dtype = Derivable<T>;
  auto f = [](const dtype& x) { return x * 2; };

  dtype x(2);
  dtype fx = f(x.derive());

  BOOST_TEST(fx.v() == 4.0);
  BOOST_TEST(fx.d() == 2.0);
  BOOST_TEST(fx.dd() == 0.0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_div, T, test_types) {
  using dtype = Derivable<T>;
  auto f = [](const dtype& x) { return x / 2; };

  dtype x(2);
  dtype fx = f(x.derive());

  BOOST_TEST(fx.v() == 1.0);
  BOOST_TEST(fx.d() == 0.5);
  BOOST_TEST(fx.dd() == 0.0);
}