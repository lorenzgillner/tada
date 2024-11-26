#define BOOST_TEST_MODULE functions
#include "tada.hpp"
#include <boost/test/included/unit_test.hpp>

using namespace tada;

typedef Derivable<double> dtype;
typedef std::array<dtype, 2> darray;
typedef std::tuple<dtype, dtype> gradtype;

dtype ring(darray x) { return sqr(x[0]) + sqr(x[1]) + x[0] * x[1]; }

dtype ring2(dtype x, dtype y) { return sqr(x) + sqr(y) + x * y; }

dtype griewank(darray x) {
  return ((sqr(x[0]) / 4000) + (sqr(x[1]) / 4000)) -
         (cos(x[0]) * cos(x[1] / sqrt(2))) + 1;
}

dtype griewank2(dtype x, dtype y) {
  return ((sqr(x) / 4000) + (sqr(y) / 4000)) - (cos(x) * cos(y / sqrt(2))) + 1;
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

BOOST_AUTO_TEST_CASE(test_ring2) {
  dtype x(1.5);
  dtype y(-0.5);

  auto df = gradient(ring2, x, y);

  BOOST_TEST(std::get<0>(df).v() == std::get<1>(df).v());
  BOOST_TEST(truncated(std::get<0>(df).v(), 6) == 1.75);
  BOOST_TEST(truncated(std::get<0>(df).d(), 6) == 2.5);
  BOOST_TEST(truncated(std::get<1>(df).d(), 6) == 0.5);
}

BOOST_AUTO_TEST_CASE(test_griewank) {
  dtype x0(1.5);
  dtype x1(-0.5);

  darray x = {x0, x1};

  darray df = gradient(griewank, x);

  BOOST_TEST(df[0].v() == df[1].v());
  BOOST_TEST(truncated(df[0].v(), 5) == 0.93426);
  BOOST_TEST(truncated(df[0].d(), 5) == 0.93654);
  BOOST_TEST(truncated(df[0].dd(), 5) == 0.06686);
  BOOST_TEST(truncated(df[1].d(), 5) == -0.01756);
  BOOST_TEST(truncated(df[1].dd(), 5) == 0.03368);
}

BOOST_AUTO_TEST_CASE(test_griewank2) {
  dtype x(1.5);
  dtype y(-0.5);

  auto df = gradient(griewank2, x, y);

  BOOST_TEST(std::get<0>(df).v() == std::get<1>(df).v());
  BOOST_TEST(truncated(std::get<0>(df).v(), 5) == 0.93426);
  BOOST_TEST(truncated(std::get<0>(df).d(), 5) == 0.93654);
  BOOST_TEST(truncated(std::get<0>(df).dd(), 5) == 0.06686);
  BOOST_TEST(truncated(std::get<1>(df).d(), 5) == -0.01756);
  BOOST_TEST(truncated(std::get<1>(df).dd(), 5) == 0.03368);
}