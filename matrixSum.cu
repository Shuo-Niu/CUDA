#include <stdio.h>

void matrixSumHost(float *A, float *B, float *C, const int nx, const int ny) {
    float *ia = A, *ib = B, *ic = C;
    for(int iy = 0; iy < ny; iy++) {
        for(int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void matrixSumDevice(float *A, float *B, float *C, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy + ny * ix;

    if((ix < nx) && (iy < ny)) C[idx] = A[idx] + B[idx];
}

int main() {
    clock_t c_begin, c_end, g_begin, g_end;
    double c_time, g_time;

    // set matrix size, make it power of 2
    int nx = 1 << 14;
    int ny = 1 << 14;
    int noElements = nx * ny;
    int bytes = noElements * sizeof(float);

    // alloc host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_cC = (float*)malloc(bytes); // result from CPU
    float *h_gC = (float*)malloc(bytes); // result from GPU

    // init data
    for(int i = 0; i < noElements; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    c_begin = clock();
    matrixSumHost(h_A, h_B, h_cC, nx, ny);
    c_end = clock();
    c_time = (double)(c_end - c_begin) / CLOCKS_PER_SEC;

    // alloc device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    g_begin = clock();
    // transfer data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // invoke kernel
    dim3 block(32,32);
    dim3 grid((nx + block.x - 1) / block.x, 
              (ny + block.y - 1) / block.y);

    matrixSumDevice<<<grid, block>>>(d_A, d_B, d_C, nx, ny);

    cudaDeviceSynchronize();

    // copy memory back
    cudaMemcpy(h_gC, d_C, bytes, cudaMemcpyDeviceToHost);

    g_end = clock();
    g_time = (double)(g_end - g_begin) / CLOCKS_PER_SEC;

    // compare results h_cC with h_gC
    printf("CPU\tGPU\tDiff\n");
    for(int i = 0; i < 10; i++) {
        printf("%.1f\t%.1f\t%.1f\n",h_cC[i],h_gC[i],h_cC[i]-h_gC[i]);
    }
    printf("CPU runtime: %lf\n", c_time);
    printf("GPU runtime: %lf\n", g_time);

    // free everything
    free(h_A);
    free(h_B);
    free(h_cC);
    free(h_gC);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}