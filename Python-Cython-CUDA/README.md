# Python-Cython-CUDA

A lightweigth example of adding matrices on GPU, invoking CUDA in Python via Cython. 

Check [matrixSum.cu](https://github.com/Shuo-Niu/CUDA/blob/master/matrixSum.cu) to see the pure CUDA version.
### Build and install
```python setup.py build_ext --inplace```

**Note**: setup.py is customized for running on UG(51-75) machines.

---
### Test
```python test.py```
