#include <stdio.h>

void matrixMultiHost(float *A, int A_row, int A_col, 
                     float *B, int B_row, int B_col, float *C) {
    // A_col == B_row
    for(int i = 0; i < A_row; i++)
        for(int j = 0; j < B_col; j++)
            for(int k = 0; k < A_col; k++)
                C[i * B_col + j] += A[i * A_col + k] * B[k * B_col + j];
}

__global__ void matrixMultiDevice(float *A, int A_row, int A_col, 
                                  float *B, int B_row, int B_col, float *C) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x; // C_row == A_row
    int iy = threadIdx.y + blockIdx.y * blockDim.y; // C_col == B_col
    int idx = ix * B_col + iy; // index of C only affected by ix and iy
    int iz = threadIdx.z + blockIdx.z * blockDim.z; // A_col == B_row

    if((ix < A_row) && (iy < B_col) && (iz < A_col))
        atomicAdd(&C[idx], A[ix * A_col + iz] * B[iz * B_col + iy]);
}

float compareMatrices(float *A, float *B, int length) {
    float error = 0;
    for(int i = 0; i < length; i++) {
        if(A[i] != B[i]) error++;
    }
    return error / (float)length;
}

int main() {
    clock_t c_begin, c_end, g_begin, g_end;
    double c_time, g_time;

    // set matrix size
    int power = 10;
    int A_row = 1 << power;
    int A_col = 1 << power;
    int B_row = 1 << power;
    int B_col = 1 << power;
    int noAElements = A_row * A_col;
    int noBElements = B_row * B_col;
    int noCElements = A_row * B_col;
    int bytesA = noAElements * sizeof(float);
    int bytesB = noBElements * sizeof(float);
    int bytesC = noCElements * sizeof(float);

    // alloc host memory
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C_CPU = (float*)malloc(bytesC);
    float *h_C_GPU = (float*)malloc(bytesC);    

    // init data
    for(int i = 0; i < noAElements; i++) h_A[i] = i;
    for(int i = 0; i < noBElements; i++) h_B[i] = i;

    c_begin = clock();
    matrixMultiHost(h_A, A_row, A_col, h_B, B_row, B_col, h_C_CPU);
    c_end = clock();
    c_time = (double)(c_end - c_begin) / CLOCKS_PER_SEC;

    // alloc device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytesA);
    cudaMalloc((void**)&d_B, bytesB);
    cudaMalloc((void**)&d_C, bytesC);
    cudaMemset(d_C, 0, bytesC); // necessary here

    g_begin = clock();
    // transfer data to device
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    // invoke kernel
    dim3 block(32, 32, 1);
    dim3 grid((A_row + block.x - 1) / block.x, 
              (B_col + block.y - 1) / block.y,
              (A_col + block.z - 1) / block.z);

    matrixMultiDevice<<<grid, block>>>(d_A, A_row, A_col, d_B, B_row, B_col, d_C);

    cudaDeviceSynchronize();

    // copy memory back
    cudaMemcpy(h_C_GPU, d_C, bytesC, cudaMemcpyDeviceToHost);

    g_end = clock();
    g_time = (double)(g_end - g_begin) / CLOCKS_PER_SEC;

    printf("CPU runtime: %lf\n", c_time);
    printf("GPU runtime: %lf\n", g_time);
    printf("Error rate: %.2f\n", compareMatrices(h_C_CPU, h_C_GPU, noCElements));

    // free everything
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}