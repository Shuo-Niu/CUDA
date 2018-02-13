# Matrix Euclidean Distance

### Build and test
Reference to [Python-Cython-CUDA](https://github.com/Shuo-Niu/CUDA/tree/master/Python-Cython-CUDA#build-and-install)

---

Input matrices: A(N1 x D), B(N2 x D)

Output matrix: C(N1 x N2)

**Without shared memory ([```kernel_matrix_dist(args)```](https://github.com/Shuo-Niu/CUDA/blob/master/matrixEuclideanDist/kernel/kernel.cu))**

- Each thread is an element in C so the toal thread number is (N1 x N2). C[i][j] will be completely computed in the corresponding thread. Not necessarily using atomic addition.

**With shared memory ([```kernel_matrix_dist_sharedMem(args)```](https://github.com/Shuo-Niu/CUDA/blob/master/matrixEuclideanDist/kernel/kernel.cu))**

- Each thread is an element in A so the total thread number is (N1 x D). C[i][j] will be added up separately in different threads, so atomic addition is necessary (CUDA atomic operation is only available for type float32). 
- Elements in B will be moved to shared memory by blocking (so that B can be larger than 48KB - maximum size of shared memory). Each thread (i.e. each element in A) will compute with B in shared memory one block at a time iteratively, and write results to the corresponding position in C. At the end, each element in A can traverse all elements in B.
