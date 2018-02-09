#include <assert.h>
#include <iostream>
#include "manager.hh"
#include "kernel.cu"
using namespace std;

GPUMatAdd::GPUMatAdd(int *A, int *B, int *C, int _x, int _y) {
    // assign host address
    h_A = A;
    h_B = B;
    h_C = C;
    x = _x;
    y = _y;
    int noElements = x * y;
    int bytes = noElements * sizeof(int);
    // alloc device memory
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);
    // transfer from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    cudaError_t err = cudaGetLastError();
    assert(err == 0);
}

void GPUMatAdd::add() {
    int noElements = x * y;
    int bytes = noElements * sizeof(int);

    dim3 block(32, 32);
    dim3 grid((x + block.x - 1) / block.x,
                (y + block.y - 1) / block.y);
    kernel_matrix_sum<<<grid, block>>>(d_A, d_B, d_C, x, y);
    
    cudaDeviceSynchronize();
    // copy result memory back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    assert(err == 0);
}

GPUMatAdd::~GPUMatAdd() {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}