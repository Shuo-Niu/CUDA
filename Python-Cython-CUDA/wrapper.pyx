import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "kernel/manager.hh":
    cdef cppclass C_GPUMatAdd "GPUMatAdd":
        C_GPUMatAdd(np.int32_t*, np.int32_t*, np.int32_t*, int, int)
        void add()

cdef class GPUMatAdd:
    cdef C_GPUMatAdd* g

    def __cinit__(self, \
                  np.ndarray[ndim=1, dtype=np.int32_t] A, \
                  np.ndarray[ndim=1, dtype=np.int32_t] B, \
                  np.ndarray[ndim=1, dtype=np.int32_t] C,
                  int x, int y):

        self.g = new C_GPUMatAdd(&A[0], &B[0], &C[0], x, y)

    def add(self):
        self.g.add()