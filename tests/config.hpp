#include "../tada.hpp"
#include <boost/mpl/list.hpp>
#include <boost/numeric/interval.hpp>
#include <boost/numeric/interval/io.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace boost;
using namespace numeric;
using namespace interval_lib;

typedef interval<double, policies<save_state<rounded_transc_std<double>>,
                                  checking_base<double>>>
    custom_interval;

typedef boost::mpl::list<float, double, interval<float>, interval<double>, custom_interval> test_types;