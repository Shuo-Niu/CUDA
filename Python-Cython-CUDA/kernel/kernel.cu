#include <stdio.h>

void __global__ kernel_matrix_sum(int *A, int *B, int *C, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy + ny * ix;

    if((ix < nx) && (iy < ny)) C[idx] = A[idx] + B[idx];
}