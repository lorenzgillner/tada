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
typedef cuda::std::array<dtype, 2> boxtype;

__device__ dtype f(boxtype x)
{
    return ((sqr(x[0]) / 4000) + (sqr(x[1]) / 4000)) - (cos(x[0]) * cos(x[1] / sqrt(2))) + 1;
}

__global__ void kernel(boxtype *A)
{
    int i = threadIdx.x;

    dtype x(xinterval(1,2));
    dtype y(xinterval(2,3));

    boxtype _A = {x, y};

    A[i] = gradient(f, _A);
}

int main()
{
    int N = 10;

    newHostData(h_A, boxtype, N);
    newDeviceData(d_A, boxtype, N);
    
    cudaMemcpy(d_A, h_A, sizeof(boxtype) * N, cudaMemcpyHostToDevice);

    kernel<<<1, N>>>(d_A);
    cudaDeviceSynchronize();

    cudaMemcpy(h_A, d_A, sizeof(boxtype) * N, cudaMemcpyDeviceToHost);

    for (int i(0); i < N; i++) std::cout << h_A[i][0].v() << "\t" << h_A[i][0].d() << "\t" << h_A[i][1].d() << std::endl;

    cudaFree(d_A);
    free(h_A);
}