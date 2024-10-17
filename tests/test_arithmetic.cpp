#define BOOST_TEST_MODULE arithmetic
#include "../tada.hpp"
#include "boost/test/included/unit_test.hpp"

using namespace tada;

typedef double itype;
typedef Derivable<itype> dtype;

BOOST_AUTO_TEST_CASE(test_add) {
  auto f = [](dtype x) { return x + 2; };

  dtype x(2);
  dtype fx = f(x.derive());

  BOOST_TEST(fx.v() == 4.0);
  BOOST_TEST(fx.d() == 1.0);
}

BOOST_AUTO_TEST_CASE(test_sub) {
  auto f = [](dtype x) { return x - 2; };

  dtype x(2);
  dtype fx = f(x.derive());

  BOOST_TEST(fx.v() == 0.0);
  BOOST_TEST(fx.d() == 1.0);
}

BOOST_AUTO_TEST_CASE(test_mul) {
  auto f = [](dtype x) { return x * 2; };

  dtype x(2);
  dtype fx = f(x.derive());

  BOOST_TEST(fx.v() == 4.0);
  BOOST_TEST(fx.d() == 2.0);
}

BOOST_AUTO_TEST_CASE(test_div) {
  auto f = [](dtype x) { return x / 2; };

  dtype x(2);
  dtype fx = f(x.derive());

  BOOST_TEST(fx.v() == 1.0);
  BOOST_TEST(fx.d() == 0.5);
}