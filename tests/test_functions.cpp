#define BOOST_TEST_MODULE functions
#include "../tada.hpp"
#include <boost/test/included/unit_test.hpp>

using namespace tada;

typedef Derivable<double> dtype;
typedef std::array<dtype, 2> darray;

dtype ring(darray x) { return sqr(x[0]) + sqr(x[1]) + x[0] * x[1]; }
dtype griewank(darray x) {
  return ((sqr(x[0]) / 4000) + (sqr(x[1]) / 4000)) -
         (cos(x[0]) * cos(x[1] / sqrt(2))) + 1;
}

double truncated(double x, int n) {
  double a = pow(10, n);
  return static_cast<int>(x * a) / a;
}

BOOST_AUTO_TEST_CASE(test_ring) {
  dtype x0(1.5);
  dtype x1(-0.5);

  darray x = {x0, x1};

  darray df = gradient(ring, x);

  BOOST_TEST(df[0].v() == df[1].v());
  BOOST_TEST(truncated(df[0].v(), 6) == 1.75);
  BOOST_TEST(truncated(df[0].d(), 6) == 2.5);
  BOOST_TEST(truncated(df[1].d(), 6) == 0.5);
}

BOOST_AUTO_TEST_CASE(test_griewank) {
  dtype x0(1.5);
  dtype x1(-0.5);

  darray x = {x0, x1};

  darray df = gradient(griewank, x);

  BOOST_TEST(df[0].v() == df[1].v());
  BOOST_TEST(truncated(df[0].v(), 6) == 0.934263);
  BOOST_TEST(truncated(df[0].d(), 6) == 0.936548);
  BOOST_TEST(truncated(df[1].d(), 6) == -0.017568);
}