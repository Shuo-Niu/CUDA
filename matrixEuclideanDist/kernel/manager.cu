#include <assert.h>
#include <iostream>
#include "manager.hh"
#include "kernel.cu"
using namespace std;

distMatrix::distMatrix(float *_h_A, float *_h_B, float *_h_C, int _x1, int _x2, int _dim) {
    h_A = _h_A;
    h_B = _h_B;
    h_C = _h_C;
    x1 = _x1;
    x2 = _x2;
    dim = _dim;
}

void distMatrix::compute() {
    int bytes_A = x1 * dim * sizeof(float);
    int bytes_B = x2 * dim * sizeof(float);
    int bytes_C = x1 * x2 * sizeof(float);
    
    // alloc device memory
    cudaMalloc((void**)&d_A, bytes_A);
    cudaMalloc((void**)&d_B, bytes_B);
    cudaMalloc((void**)&d_C, bytes_C);
    cudaMemset(d_C, 0, bytes_C);
    
    // transfer from host to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // number of thread == x1 * x2 (C.shape)
    dim3 block(32, 32);
    dim3 grid((x1 + block.x - 1) / block.x,
                (x2 + block.y - 1) / block.y);
    kernel_matrix_dist<<<grid, block>>>(d_A, d_B, d_C, x1, x2, dim);
    cudaDeviceSynchronize();

    // copy back to host
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    assert(err == 0);
}

void distMatrix::compute_sharedMem() {
    int bytes_A = x1 * dim * sizeof(float);
    int bytes_B = x2 * dim * sizeof(float);
    int bytes_C = x1 * x2 * sizeof(float);
    
    // alloc device memory
    cudaMalloc((void**)&d_A, bytes_A);
    cudaMalloc((void**)&d_B, bytes_B);
    cudaMalloc((void**)&d_C, bytes_C);
    cudaMemset(d_C, 0, bytes_C);
    
    // transfer from host to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // number of thread == x1 * dim (A.shape)
    int block = 1024;
    int grid = (x1 * dim / block) + 1;
    kernel_matrix_dist_sharedMem<<<grid, block, block*sizeof(float)>>>(d_A, d_B, d_C, x1, x2, dim);
    cudaDeviceSynchronize();

    // copy back to host
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    assert(err == 0);
}

distMatrix::~distMatrix() {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}