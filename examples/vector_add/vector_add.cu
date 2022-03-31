template <typename T, int block_size_x>
__global__ void vector_add(T* c,  const T* a,  const T* b, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
