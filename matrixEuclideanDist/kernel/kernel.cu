#include <stdio.h>

__global__ void kernel_matrix_dist(float *A, float *B, float *C, const int nx1, const int nx2, const int dim) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy + nx2 * ix; // index in C

    if((ix < nx1) && (iy < nx2)) {
        for(int i = 0; i < dim; i++) {
            C[idx] += (A[ix * dim + i] - B[iy * dim + i]) * (A[ix * dim + i] - B[iy * dim + i]);
        }
        C[idx] = sqrtf(C[idx]);
    }
}

__global__ void kernel_matrix_dist_sharedMem(float *A, float *B, float *C, const int nx1, const int nx2, const int dim) {

    extern __shared__ float sharedPoints[]; // length == blockSize (i.e. blockDim.x here)

    int idx = threadIdx.x + blockIdx.x * blockDim.x; // index in A
    int numInA = idx / dim;
    int dimInA = idx % dim;

    for(int currentBlock = 0; currentBlock < (nx2*dim/blockDim.x)+1; currentBlock++) {

        // move a block of elements from B to shared memory in each iteration        
        if((threadIdx.x + currentBlock * blockDim.x) < (nx2 * dim)) {
            sharedPoints[threadIdx.x] = B[threadIdx.x + currentBlock * blockDim.x];
        }
        __syncthreads(); // wait for finishing moving to shared memory

        if(idx < (nx1 * dim)) {
            // compute distance in corresponding dimension between this A_point to all buffered B_points in shared memory
            for(int i = 0; i < blockDim.x; i++) {
                int idxInB = i + currentBlock * blockDim.x;
                if(idxInB >= (nx2 * dim)) break;
                int numInB = idxInB / dim;
                int dimInB = idxInB % dim;

                if(dimInA == dimInB) {
                    int idxInC = numInB + nx2 * numInA;
                    // necessary to have atomic operation here otherwise random errors introduced
                    atomicAdd(&C[idxInC], (A[idx] - sharedPoints[i]) * (A[idx] - sharedPoints[i])); 
                }
            }
        }
    }
    __syncthreads(); // wait for finishing adding all dimensions for all points in C array
    
    // thread with dimInA==0 do sqrtf() for the corresponding row in C
    if(idx < (nx1 * dim) && dimInA == 0) {
        for(int i = 0; i < nx2; i++) {
            C[i + numInA * nx2] = sqrtf(C[i + numInA * nx2]);
        }
    }
}