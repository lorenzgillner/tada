#define BOOST_TEST_MODULE derivative
#include "../tada.hpp"
#include <boost/test/included/unit_test.hpp>

using namespace tada;

typedef Derivable<double> dtype;
typedef std::array<dtype, 2> darray;

dtype f(dtype x) { return 2 * x; }
dtype g(darray x) { return (2 * x[0]) + (2 * x[1]) - (x[0] * x[1]); }

BOOST_AUTO_TEST_CASE(test_derive) {
  dtype x(2);

  dtype dfdx = derivative(f, x);

  BOOST_TEST(dfdx.v() == 4.0);
  BOOST_TEST(dfdx.d() == 2.0);
}

BOOST_AUTO_TEST_CASE(test_gradient) {
  dtype x0(2);
  dtype x1(3);

  darray x = {x0, x1};

  darray df = gradient(g, x);

  BOOST_TEST(df[0].v() == 4.0);
  BOOST_TEST(df[0].d() == -1.0);
  BOOST_TEST(df[1].v() == 4.0);
  BOOST_TEST(df[1].d() == 0.0);
}
