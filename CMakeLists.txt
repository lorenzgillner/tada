cmake_minimum_required(VERSION 3.17)

project(tada LANGUAGES CXX)

find_package(Boost 1.82 REQUIRED COMPONENTS unit_test_framework)

enable_testing()
set(TEST_DIR "${CMAKE_SOURCE_DIR}/tests")

set(EXAMPLE_DIR "${CMAKE_SOURCE_DIR}/examples")

include_directories("${CMAKE_SOURCE_DIR}/include")

option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
    message(STATUS "CUDA support enabled")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    set(CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)

    add_executable(cuda_example example/example.cu)

    add_executable(cuda_interval_example example/interval_example.cu)
    target_include_directories(cuda_interval_example PRIVATE ${CUSTOM_BOOST_LIB})
endif()

add_executable(example ${EXAMPLE_DIR}/example.cpp)

macro(setup_test target)
    add_executable(${target} ${TEST_DIR}/${target}.cpp)
    target_link_libraries(${target} PRIVATE Boost::unit_test_framework)
    add_test(NAME ${target} COMMAND ${target})
endmacro()

setup_test(test_creation)
setup_test(test_derivative)
setup_test(test_arithmetic)
setup_test(test_functions)