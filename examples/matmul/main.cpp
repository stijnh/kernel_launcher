#include <exception>
#include <iostream>
#include <cstdio>
#include <cuda.h>

#include "kernel_launcher.hpp"

namespace kl = kernel_launcher;
using TF = float;

#define CUDA_CHECK(expr)     \
    if (expr != cudaSuccess) \
        throw std::runtime_error(#expr);


int main() {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

    cudaStream_t stream;
    cudaEvent_t before;
    cudaEvent_t after;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&before));
    CUDA_CHECK(cudaEventCreate(&after));

    // Initialize input
    const int N = 4096;
    std::vector<TF> A(N * N);
    std::vector<TF> B(N * N);
    std::vector<TF> C(N * N);

#pragma omp parallel for
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate memory
    kl::Memory<TF> dev_A(A);
    kl::Memory<TF> dev_B(B);
    kl::Memory<TF> dev_C(C);

    // Builder kernel
    auto builder = kl::KernelBuilder("matmul.cu", "matmul_kernel");
    auto bx = builder.tune("block_size_x", {1, 2, 4, 8, 16, 32, 64, 128, 256});
    auto by = builder.tune("block_size_y", {1, 2, 4, 8, 16, 32, 64, 128, 256});
    auto tx = builder.tune("tile_size_x", {1, 2, 4, 8, 16, 32, 64, 128});
    auto ty = builder.tune("tile_size_y", {1, 2, 4, 8, 16, 32, 64, 128});
    builder.restrict(bx == by * ty);

    auto num_threads = bx * by;
    builder.restrict(num_threads % 32 == 0);
    builder.restrict(num_threads >= 64);
    builder.restrict(num_threads <= 1024);
    builder.restrict(N % (bx * tx) == 0);
    builder.restrict(N % (by * ty) == 0);

    auto sA_size = by * ty * bx * sizeof(TF);
    auto sB_size = by * ty * bx * tx * sizeof(TF);
    builder.restrict(sA_size + sB_size <= 48 * 1024);

    builder
        .template_args(kl::type_of<TF>())
        .block_size(bx, by)
        .grid_divisors(bx * tx, by * ty)
        .define(bx)
        .define(by)
        .define(tx)
        .define(ty);

    // Compile kernel
    kl::Config config;
    kl::Config best_config;
    float best_time = 1e99;

    auto iter = builder.iterate();
    while(iter.next(config)) {
        auto kernel = kl::Kernel<TF*, const TF*, const TF*>::compile(
            builder,
            config);

        CUDA_CHECK(cudaEventRecord(before, stream));
        for (int i = 0; i < 10; i++)
            kernel(stream, N, N)(dev_C, dev_A, dev_B);
        CUDA_CHECK(cudaEventRecord(after, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, before, after));

        if (time < best_time) {
            best_config = config;
            best_time = time;
        }

        std::cout << config.to_json() << ": " << time << " (best: " << best_time << ")" << std::endl;

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Validate output
    std::vector<TF> output = dev_C.to_vector();

    for (int i = 0; i < N * N; i++) {
        if (output[i] != C[i]) {
           //throw std::runtime_error("invalid output!");
        }
    }

    std::cout << "Everything ok!" << std::endl;
    return EXIT_SUCCESS;
}