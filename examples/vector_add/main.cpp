#include <exception>
#include <iostream>
#include <cstdio>
#include <cuda.h>

#include "kernel_launcher.hpp"

namespace kl = kernel_launcher;

#define CUDA_CHECK(expr)     \
    if (expr != cudaSuccess) \
        throw std::runtime_error(#expr);


int main() {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

    // Initialize input
    int n = 100;
    std::vector<float> A(n);
    std::vector<float> B(n);
    std::vector<float> C(n);

    for (int i = 0; i < n; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
        C[i] = A[i] + B[i];
    }

    // Allocate memory
    kl::Memory<float> dev_A(A);
    kl::Memory<float> dev_B(B);
    kl::Memory<float> dev_C(C);

    // Builder kernel
    auto builder = kl::KernelBuilder("vector_add.cu", "vector_add");
    auto b = builder.tune("block_size_x", {64, 128, 256, 1024});
    builder.block_size(b);
    builder.template_args(kl::type_of<float>(), b);

    // Compile kernel
    auto config = builder.sample();
    auto kernel = kl::Kernel<float*,  const float*,  const float*, int>::compile(builder, config);

    // Run kernel
    kernel(n)(dev_C, dev_A, dev_B, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Validate output
    std::vector<float> output = dev_C.to_vector();

    for (int i = 0; i < n; i++) {
        if (output[i] != C[i]) {
           throw std::runtime_error("invalid output!");
        }
    }

    std::cout << "Everything ok!" << std::endl;
    return EXIT_SUCCESS;
}
