#define BOOST_TEST_MODULE creation
#include "config.hpp"

using namespace tada;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_default, T, test_types) {
  Derivable<T> x;

  BOOST_TEST(x.v() == static_cast<T>(0));
  BOOST_TEST(x.d() == static_cast<T>(0));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_explicit, T, test_types) {
  Derivable<T> x(2, independent);

  BOOST_TEST(x.v() == static_cast<T>(2));
  BOOST_TEST(x.d() == static_cast<T>(1));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_Independent, T, test_types) {
  Derivable<T> x = Independent<T>(2);

  BOOST_TEST(x.v() == static_cast<T>(2));
  BOOST_TEST(x.d() == static_cast<T>(1));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_derive, T, test_types) {
  Derivable<T> x(2);
  Derivable<T> dx = x.derive();

  BOOST_TEST(dx.v() == static_cast<T>(2));
  BOOST_TEST(dx.d() == static_cast<T>(1));
}