#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>

// define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

// define device functions
__device__ float sigmoid(float in) {
    return 1.f / (1.f + expf(-in));
}

// define pointwise functions for unfused elementwise LSTM unit
__global__ void pw_biasAdd(float *y, float *bias, int n, int nBias) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] += bias[i % nBias];
}

__global__ void pw_vecAdd(float *y, float *a,  float *b, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = a[i] + b[i];
}

__global__ void pw_vecMul(float *y, float *a,  float *b, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = a[i] * b[i];
}

__global__ void pw_tanh(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = tanh(a[i]);
}

__global__ void pw_sigmoid(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = sigmoid(a[i]);
}

int LSTM_unit_unfused(int hiddenSize,
                             int miniBatch,
                             float * __restrict__ h_in, // h(t-1) * R
                             float * __restrict__ x_in, // x(t) * W
                             float * __restrict__ bias,
                            //  float * __restrict__ linearGates,
                             float * __restrict__ h_out,// h(t)
                             float * __restrict__ c_in, // c(t-1)
                             float * __restrict__ c_out,// c(t)
                             cudaStream_t stream) {
    dim3 blockDim;
    dim3 gridDim;

    int numElements = hiddenSize * miniBatch;

    blockDim.x = 128;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

    // element wise calculations
    // x(t) = x(t) * W + h(t-1) * R + bias, as input to this unit
    for (int i = 0; i < 4; i++) {
        pw_vecAdd <<< gridDim, blockDim, 0, stream >>> (x_in + i * numElements,
                                                        x_in + i * numElements,
                                                        h_in + i * numElements,
                                                        numElements);
        cudaErrCheck(cudaGetLastError());

        pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (x_in + i * numElements,
                                                         bias + i * hiddenSize,
                                                         numElements,
                                                         hiddenSize);
        cudaErrCheck(cudaGetLastError());

        pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (x_in + i * numElements,
                                                         bias + (i + 4) * hiddenSize,
                                                         numElements,
                                                         hiddenSize);
        cudaErrCheck(cudaGetLastError());
    }

    // x(t) goes through 4 gates' activation
    pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (x_in + 0 * numElements, x_in + 0 * numElements, numElements);
    cudaErrCheck(cudaGetLastError());

    pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (x_in + 1 * numElements, x_in + 1 * numElements, numElements);
    cudaErrCheck(cudaGetLastError());

    pw_tanh <<< gridDim, blockDim, 0, stream >>> (x_in + 2 * numElements, x_in + 2 * numElements, numElements);
    cudaErrCheck(cudaGetLastError());

    pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (x_in + 3 * numElements, x_in + 3 * numElements, numElements);
    cudaErrCheck(cudaGetLastError());

    // assign location to 4 gates
    float *f_t = x_in + 0 * numElements;
    float *i_t = x_in + 1 * numElements;
    float *g_t = x_in + 2 * numElements;
    float *o_t = x_in + 3 * numElements;

    // f(t) *= c(t-1)
    pw_vecMul <<< gridDim, blockDim, 0, stream >>> (f_t, f_t, c_in, numElements);
    cudaErrCheck(cudaGetLastError());

    // i(t) *= g(t)
    pw_vecMul <<< gridDim, blockDim, 0, stream >>> (i_t, i_t, g_t, numElements);
    cudaErrCheck(cudaGetLastError());

    // i(t) += f(t)
    pw_vecAdd <<< gridDim, blockDim, 0, stream >>> (i_t, i_t, f_t, numElements);
    cudaErrCheck(cudaGetLastError());

    // c(t) = i(t), output cell state
    cudaErrCheck(cudaMemcpyAsync(c_out, i_t, numElements * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // i(t) = tanh(i(t)), i(t) === c(t) here, but we must not modify c(t)
    pw_tanh <<< gridDim, blockDim, 0, stream >>> (i_t, i_t, numElements);
    cudaErrCheck(cudaGetLastError());

    // h(t) = i(t) * o(t)
    pw_vecMul <<< gridDim, blockDim, 0, stream >>> (h_out, i_t, o_t, numElements);
    cudaErrCheck(cudaGetLastError());

    return 0;
}

float LSTMTest(int hiddenSize, int miniBatch, int seqLength, int numLayers) {
    int numElements = hiddenSize * miniBatch;
    // alloc device memory
    float *d_x, *d_h, *d_c;
    cudaErrCheck(cudaMalloc((void**)&d_x, (seqLength) * (numLayers + 1) * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&d_h, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&d_c, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));

    float *d_weight, *d_weight_T;
    cudaErrCheck(cudaMalloc((void**)&d_weight, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&d_weight_T, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));

    float *d_bias;
    cudaErrCheck(cudaMalloc((void**)&d_bias, numLayers * hiddenSize * 8 * sizeof(float)));

    float *d_h_in, *d_x_in;
    cudaErrCheck(cudaMalloc((void**)&d_h_in, numLayers * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&d_x_in, seqLength * numElements * sizeof(float)));

    // all use default NULL streams for now
    cudaStream_t *stream_x, *stream_h;
    stream_x = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));
    stream_h = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));
    // optimization 2 uses different streams for x and h
    // optimization 6 uses different streams for various layers
    for(int i = 0; i < numLayers; i++) {
        stream_x[i] = NULL;
        stream_h[i] = NULL;
    }

    // alloc events
    cudaEvent_t **events_x, **events_h;
    events_x = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
    events_h = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
    for(int i = 0; i < numLayers; i++) {
        events_x[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
        events_h[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
    }

    // initiate random inputs
    curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1782ULL));
    curandErrCheck(curandGenerateUniform(gen, d_x, (seqLength) * (numLayers + 1) * numElements));
    curandErrCheck(curandGenerateUniform(gen, d_h, (seqLength + 1) * (numLayers) * numElements));
    curandErrCheck(curandGenerateUniform(gen, d_c, (seqLength + 1) * (numLayers) * numElements));
    curandErrCheck(curandGenerateUniform(gen, d_weight, numLayers * hiddenSize * hiddenSize * 8));
    curandErrCheck(curandGenerateUniform(gen, d_bias, numLayers * hiddenSize * 8));
    curandErrCheck(curandDestroyGenerator(gen));

    // create cuBLAS handle
    cublasHandle_t handle;
    cublasErrCheck(cublasCreate(&handle));

    cudaErrCheck(cudaDeviceSynchronize());
    printf("--Done alloc device mem\n");

    //start timing
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));
    cudaErrCheck(cudaEventRecord(start));

    // LSTM
    
    const cublasOperation_t a_trans = CUBLAS_OP_T; // no pre-transpose for now
    const cublasOperation_t b_trans = CUBLAS_OP_N; // always N

    if (a_trans == CUBLAS_OP_N) {
        // optimization 4
    }
    else {
        d_weight_T = d_weight;
    }

    // cublasSgemm(): C = alpha * (A + B) + beta * C 
    float alpha = 1.f;
    float beta = 0.f;

    int lStart = 0;
    int lEnd = 0;
    int rStart = 0;
    int rEnd = 0;
    int recurBatchSize = 1; // optimization 5 will make it 2

    while(true) {
        // Many layer "scheduling".
        if (lEnd == 0) {
            lStart = 0;
            lEnd = 1;
            rStart = 0;
        }
        else {
            // Move "up" and "left"
            lStart++;
            lEnd++;
            
            rStart -= recurBatchSize;
            
            // Over the top or off the left, reset to layer 0
            if (lEnd > numLayers || rStart < 0) {
                rStart += (lStart + 1) * recurBatchSize;

                lStart = 0;
                lEnd = 1;
            }
            
            // Off the right, step up
            while (rStart >= seqLength && lEnd <= numLayers) {
                lStart++;
                lEnd++;
                
                rStart -= recurBatchSize;
            }
            
            // Over the top or off the left, done!
            if (lEnd > numLayers || rStart < 0) {
                break;
            }
        }
        rEnd = rStart + recurBatchSize;
        if (rEnd > seqLength) rEnd = seqLength;

        // lStart, lEnd always differ 1
        for(int layer = lStart; layer < lEnd; layer++) {
            cublasErrCheck(cublasSetStream(handle, stream_x[layer]));
            
            // rStart, rEnd differ recurBatchSize
            for(int i = rStart; i < rEnd; i++) {
                if(layer > 0) {
                    cudaErrCheck(cudaStreamWaitEvent(stream_x[layer], events_h[layer - 1][i], 0));
                    cudaErrCheck(cudaEventDestroy(events_h[layer - 1][i]));
                }
            }

            // x(t) *= [W_weight]
            // do optimization 1 here
            for(int igemm = 0; igemm < 4; igemm++) {
                cublasErrCheck(cublasSgemm(handle,
                                           a_trans, b_trans,
                                           hiddenSize, // #rows of A and C
                                           miniBatch * (rEnd - rStart), // #cols of B and C
                                           hiddenSize, // #cols of A and B
                                           &alpha,
                                           &d_weight_T[layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize], // A
                                           a_trans == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize, // leading dimension of A, where we can try different data layout
                                           d_x + rStart * numElements + layer * seqLength * numElements, // B
                                           hiddenSize, // leading dimension of B, where we can try different data layout
                                           &beta,
                                           d_x_in + 4 * rStart * numElements + igemm * hiddenSize, // C
                                           4 * hiddenSize // leading dimension of C
                                           ));
            }

            for(int i = rStart; i < rEnd; i++) {
                cudaErrCheck(cudaEventCreate(&events_x[layer][i], cudaEventDisableTiming));
                cudaErrCheck(cudaEventRecord(events_x[layer][i], stream_x[layer]));
            }

            for(int i = rStart; i < rEnd; i++) {
                cublasErrCheck(cublasSetStream(handle, stream_h[layer]));

                // h(t-1) *= [R_weight]
                // do optimization 1 here
                for(int igemm = 0; igemm < 4; igemm++) {
                    cublasErrCheck(cublasSgemm(handle,
                                               a_trans, b_trans,
                                               hiddenSize, miniBatch, hiddenSize,
                                               &alpha,
                                               &d_weight_T[4 * hiddenSize * hiddenSize + layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize],
                                               a_trans == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                                               d_h + i * numElements + layer * (seqLength + 1) * numElements,
                                               hiddenSize,
                                               &beta,
                                               d_h_in + 4 * layer * numElements + igemm * hiddenSize,
                                               4 * hiddenSize
                                               ));
                }

                cudaErrCheck(cudaStreamWaitEvent(stream_h[layer], events_x[layer][i], 0));
                cudaErrCheck(cudaEventDestroy(events_x[layer][i]));
                
                // do optimization 3 here
                LSTM_unit_unfused(hiddenSize, miniBatch,
                                  d_h_in + 4 * layer * numElements,
                                  d_x_in + 4 * i * numElements,
                                  d_bias + 8 * layer * hiddenSize,
                                  d_h + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                                  d_c + i * numElements + layer * (seqLength + 1) * numElements,
                                  d_c + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                                  stream_h[layer]
                                  );
                
                if(layer != numLayers - 1) {
                    cudaErrCheck(cudaEventCreate(&events_h[layer][i], cudaEventDisableTiming));
                    cudaErrCheck(cudaEventRecord(events_h[layer][i], stream_h[layer]));
                }
            }
        }
    }

    // stop timing
    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start, stop));
    cudaErrCheck(cudaDeviceSynchronize());

    cudaErrCheck(cudaDeviceSynchronize());
    printf("--Done LSTM\n");

    // free everything
    cudaErrCheck(cudaFree(d_x));
    cudaErrCheck(cudaFree(d_h));
    cudaErrCheck(cudaFree(d_c));
    cudaErrCheck(cudaFree(d_bias));
    cudaErrCheck(cudaFree(d_x_in));
    cudaErrCheck(cudaFree(d_h_in));
    
    for(int i = 0; i < numLayers; i++) {
        if(stream_x[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_x[i]));
        if(stream_h[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_h[i]));
    }
    free(stream_x);
    free(stream_h);

    for(int i = 0; i < numLayers; i++) {
        free(events_x[i]);
        free(events_h[i]);
    }
    free(events_x);
    free(events_h);
    printf("--Done free device mem\n");

    return elapsedTime;
}

int main(int argc, char* argv[]) {
    int seqLength;
    int numLayers;
    int hiddenSize;
    int miniBatch; 
    
    if (argc == 5) {
        seqLength = atoi(argv[1]);
        numLayers =  atoi(argv[2]);
        hiddenSize =  atoi(argv[3]);
        miniBatch =  atoi(argv[4]);   
    }
    else if (argc == 1) {
        printf("Running with default settings\n");
        seqLength = 100;
        numLayers = 4;
        hiddenSize = 512;
        miniBatch = 64;
    }
    else {
        printf("Usage: ./LSTM <seqLength> <numLayers> <hiddenSize> <miniBatch>\n");
        return 1;      
    }

    printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n", seqLength, numLayers, hiddenSize, miniBatch);

    int numRuns = 1;
   
    float totalTime = 0.f;
    for (int run = 0; run < numRuns; run++) {
        totalTime += LSTMTest(hiddenSize, miniBatch, seqLength, numLayers);
    }
    
    printf("Runtime %fms\n", totalTime / numRuns);
    
    return time < 0;
}