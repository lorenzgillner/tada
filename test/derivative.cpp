#define BOOST_TEST_MODULE derivative
#include <boost/test/included/unit_test.hpp>
#include "../tada.hpp"

using namespace tada;

typedef Derivable<double> dtype;

dtype f(dtype x) { return 2 * x; };

BOOST_AUTO_TEST_CASE( test_derive )
{
    dtype x(2);

    double dfdx = derivative(f, x);

    BOOST_TEST(dfdx == 2.0);
}
