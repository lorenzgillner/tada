cmake_minimum_required(VERSION 3.10)
project(tada)

enable_testing()

set(TEST_DIR "${CMAKE_SOURCE_DIR}/test")

set(Boost_INCLUDE_DIR "/home/gillor/src/boost_1_86_0")

find_package(Boost 1.70 REQUIRED COMPONENTS unit_test_framework)

macro(setup_test target)
    add_executable(${target} ${TEST_DIR}/${target}.cpp)
    target_link_libraries(${target} PRIVATE Boost::unit_test_framework)
    add_test(NAME ${target} COMMAND ${target})
endmacro()

setup_test(creation)
setup_test(derivative)
setup_test(arithmetic)