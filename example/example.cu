#include <cmath>
#include <iostream>

#include "tada.hpp"

using namespace tada;

typedef double itype;
typedef Derivable<itype> dtype;

__global__ void example_kernel(dtype *x)
{
  int i = threadIdx.x;

  switch (i)
  {
  case 0:
    x[i] = x[i] + 2;
    break;
  case 1:
    x[i] = x[i] - 2;
    break;
  case 2:
    x[i] = x[i] * 2;
    break;
  case 3:
    x[i] = x[i] / 2;
    break;
  case 4:
    x[i] = square(x[i]);
    break;
  case 5:
    x[i] = pow(x[i], static_cast<double>(2));
    break;
  case 6:
    x[i] = pow(x[i], static_cast<double>(3));
    break;
  case 7:
    x[i] = sqrt(x[i]);
    break;
  case 8:
    x[i] = sin(x[i]);
    break;
  case 9:
    x[i] = cos(x[i]);
    break;
  case 10:
    x[i] = exp(x[i]);
    break;
  case 11:
    x[i] = log(x[i]);
    break;
  }
}

int main()
{
  int n = 12;

  dtype *h_x = (dtype *)malloc(sizeof(dtype) * n);
  dtype x0(2, independent);

  for (int i = 0; i < n; i++)
  {
    h_x[i] = x0;
  }

  dtype *d_x;
  cudaMalloc(&d_x, sizeof(dtype) * n);

  cudaMemcpy(d_x, h_x, sizeof(dtype) * n, cudaMemcpyHostToDevice);

  example_kernel<<<1, n>>>(d_x);
  cudaDeviceSynchronize();

  cudaMemcpy(h_x, d_x, sizeof(dtype) * n, cudaMemcpyDeviceToHost);

  ::std::cout << "f;f(x);f'(x)\n";
  ::std::cout << "x + 2;" << h_x[0].v() << ";" << h_x[0].d() << "\n";
  ::std::cout << "x - 2;" << h_x[1].v() << ";" << h_x[1].d() << "\n";
  ::std::cout << "x * 2;" << h_x[2].v() << ";" << h_x[2].d() << "\n";
  ::std::cout << "x / 2;" << h_x[3].v() << ";" << h_x[3].d() << "\n";
  ::std::cout << "square(x);" << h_x[4].v() << ";" << h_x[4].d() << "\n";
  ::std::cout << "pow(x,2);" << h_x[5].v() << ";" << h_x[5].d() << "\n";
  ::std::cout << "pow(x,3);" << h_x[6].v() << ";" << h_x[6].d() << "\n";
  ::std::cout << "sqrt(x);" << h_x[7].v() << ";" << h_x[7].d() << "\n";
  ::std::cout << "sin(x);" << h_x[8].v() << ";" << h_x[8].d() << "\n";
  ::std::cout << "cos(x);" << h_x[9].v() << ";" << h_x[9].d() << "\n";
  ::std::cout << "exp(x);" << h_x[10].v() << ";" << h_x[10].d() << "\n";
  ::std::cout << "log(x);" << h_x[11].v() << ";" << h_x[11].d() << "\n";
  ::std::cout << "x = " << x0.v() << ::std::endl;
}