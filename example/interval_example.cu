#include <cmath>
#include <iostream>

#include <boost/numeric/interval.hpp>
#include <boost/numeric/interval/io.hpp>

#include "tada.hpp"

#define newHostData(name, type, size) type *name = (type *)malloc(sizeof(type) * (size))
#define deleteHostData(name) free(name)
#define newDeviceData(name, type, size) type *name; cudaMalloc(&name, sizeof(type) * (size))
#define deleteDeviceData(name) cudaFree(&name)

using namespace tada;
using namespace boost;
using namespace numeric;
using namespace interval_lib;

typedef double itype;
typedef interval<itype, policies<save_state_nothing<rounded_transc_exact<itype>>,
                                 checking_base<itype>>>
    xinterval;
typedef Derivable<xinterval> dtype;

__device__ dtype f(dtype x, dtype y)
{
    return ((sqr(x) / 4000) + (sqr(y) / 4000)) - (cos(x) * cos(y / sqrt(2))) + 1;
}

__global__ void kernel(dtype *A)
{
    int i = threadIdx.x;

    dtype x(xinterval(1,2), independent);
    dtype y(xinterval(2,3));

    dtype fx = f(x, y);

    A[i] = fx;
}

int main()
{
    int N = 10;

    newHostData(h_A, dtype, N);
    newDeviceData(d_A, dtype, N);
    
    cudaMemcpy(d_A, h_A, sizeof(dtype) * N, cudaMemcpyHostToDevice);

    kernel<<<1, N>>>(d_A);
    cudaDeviceSynchronize();

    cudaMemcpy(h_A, d_A, sizeof(dtype) * N, cudaMemcpyDeviceToHost);

    for (int i(0); i < N; i++) std::cout << h_A[i].v() << "\t" << h_A[i].d() << std::endl;

    cudaFree(d_A);
    free(h_A);
}