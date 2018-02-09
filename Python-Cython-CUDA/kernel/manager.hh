class GPUMatAdd {
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    int x, y;

public:
    GPUMatAdd(int *A, int *B, int *C, int _x, int _y); // constructor
    ~GPUMatAdd(); // destructor
    void add(); // add in-place to d_A on GPU
};