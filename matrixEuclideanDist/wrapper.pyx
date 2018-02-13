import numpy as np
cimport numpy as np

assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "kernel/manager.hh":
    cdef cppclass C_distMatrix "distMatrix":
        C_distMatrix(np.float32_t*, np.float32_t*, np.float32_t*, int, int, int)
        void compute()
        void compute_sharedMem()

cdef class distMatrix:
    cdef C_distMatrix* g

    def __cinit__(self, \
                  np.ndarray[ndim=1, dtype=np.float32_t] A, \
                  np.ndarray[ndim=1, dtype=np.float32_t] B, \
                  np.ndarray[ndim=1, dtype=np.float32_t] C, \
                  int x1, int x2, int dim):
        self.g = new C_distMatrix(&A[0], &B[0], &C[0], x1, x2, dim)

    def compute(self):
        self.g.compute()

    def compute_sharedMem(self):
        self.g.compute_sharedMem()