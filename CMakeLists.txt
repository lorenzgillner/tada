cmake_minimum_required(VERSION 3.10)

project(tada)

enable_testing()

set(TEST_DIR "${CMAKE_SOURCE_DIR}/tests")

find_package(Boost 1.70 REQUIRED COMPONENTS unit_test_framework)
#set(CMAKE_CXX_STANDARD 11)

add_executable(example example/example.cpp)
target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR})

macro(setup_test target)
    add_executable(${target} ${TEST_DIR}/${target}.cpp)
    # set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
    target_link_libraries(${target} PRIVATE Boost::unit_test_framework)
    add_test(NAME ${target} COMMAND ${target})
endmacro()

setup_test(test_creation)
setup_test(test_derivative)
setup_test(test_arithmetic)
setup_test(test_functions)