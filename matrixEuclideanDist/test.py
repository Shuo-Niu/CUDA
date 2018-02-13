import numpy as np
import time
import distMatrix

def dist(x, y):
    return np.sqrt(np.sum((x - y)**2))

def test():
    # generate datasets
    A = np.random.rand(1000, 20)
    A = A.astype(np.float32) # CUDA atomic operation only available for type [float32]
    B = np.random.rand(1000, 20)
    B = B.astype(np.float32)

    # ----------- CPU compute distance matrix -----------
    C_CPU = np.zeros((np.shape(A)[0], np.shape(B)[0]), dtype=np.float32)
    CPU_start = time.time()
    for i, A_point in enumerate(A):
        for j, B_point in enumerate(B):
            C_CPU[i][j] = dist(A_point, B_point)
    CPU_end = time.time()

    print("CPU run time: %f" % (CPU_end - CPU_start))

    # ----------- GPU compute distance matrix -----------
    _A = np.reshape(A, [-1])
    _B = np.reshape(B, [-1])
    _C = np.zeros((np.shape(A)[0] * np.shape(B)[0]), dtype=np.float32)
    agent = distMatrix.distMatrix(_A, _B, _C, \
                                    np.shape(A)[0], \
                                    np.shape(B)[0], \
                                    np.shape(A)[1])
                                    
    # agent.compute() # why accelerate the second invoke???
    GPU_start = time.time()
    agent.compute_sharedMem()
    # agent.compute()
    GPU_end = time.time()
    
    C_GPU = np.reshape(_C, [np.shape(A)[0], np.shape(B)[0]])

    print("GPU run time: %f" % (GPU_end - GPU_start))

    # ----------- for sampling test -----------
    print("Sample:")
    for i in range(10):
        print("CPU: %f  GPU: %f  diff: %f" % (C_CPU[i*i][i*i], C_GPU[i*i][i*i], np.abs(C_CPU[i*i][i*i]-C_GPU[i*i][i*i])))

if __name__ == "__main__":
    test()