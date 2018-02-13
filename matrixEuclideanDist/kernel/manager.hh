class distMatrix {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int x1, x2, dim;

public:
    distMatrix(float *_h_A, float *_h_B, float *_h_C, int _x1, int _x2, int _dim);
    ~distMatrix();
    void compute();
    void compute_sharedMem();
};