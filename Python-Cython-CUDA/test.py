import gpumatadd
import numpy as np

def test():
    A = [[1,2],[3,4]]
    B = [[10,10],[10,10]]
    x = np.shape(A)[0]
    y = np.shape(A)[1]

    # GPU only takes 1-D array
    _A = np.reshape(np.asarray(A, dtype=np.int32), [-1]) 
    _B = np.reshape(np.asarray(B, dtype=np.int32), [-1])
    
    print("Input:")
    print(A)
    print(B)

    _C = np.zeros((x * y), dtype=np.int32)
    adder = gpumatadd.GPUMatAdd(_A, _B, _C, x, y)
    adder.add()

    C = np.reshape(_C, [x, y])
    print("Output:")
    print(C)

if __name__ == "__main__":
    test()