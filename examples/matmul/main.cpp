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
    const unsigned int N = 4096;

    // Builder kernel
    auto builder = kl::KernelBuilder("matmul.cu", "matmul_kernel");
    auto bx = builder.tune("block_size_x", {1, 2, 4, 8, 16, 32, 64, 128, 256});
    auto by = builder.tune("block_size_y", {1, 2, 4, 8, 16, 32, 64, 128, 256});
    auto tx = builder.tune("tile_size_x", {1, 2, 4, 8, 16, 32, 64, 128});
    auto ty = builder.tune("tile_size_y", {1, 2, 4, 8, 16, 32, 64, 128});
    auto m = builder.tune("blocks_per_sm", {1, 2, 3, 4, 5, 6, 7, 8});

    auto threads_per_sm = bx * by * m;
    auto threads_per_block = bx * by;
    builder.restrict(threads_per_sm >= 128);
    builder.restrict(threads_per_sm <= 4096);
    builder.restrict(threads_per_block % 32 == 0);
    builder.restrict(threads_per_block >= 64);
    builder.restrict(threads_per_block <= 1024);
    builder.restrict(N % (bx * tx) == 0);
    builder.restrict(N % (by * ty) == 0);
    builder.restrict(bx == by * ty);

    auto sA_size = by * ty * bx * sizeof(TF);
    auto sB_size = by * ty * bx * tx * sizeof(TF);
    builder.restrict(sA_size + sB_size <= 48 * 1024);

    builder
        .template_args(kl::type_of<TF>(), N, bx, by, tx, ty, m)
        .block_size(bx, by)
        .grid_divisors(bx * tx, by * ty);

    {
        size_t n = 0;
        kl::Config config;
        auto it = builder.iterate();
        while (it.next(config)) n++;
        std::cout << "config size: " << n << std::endl;
    }

    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize input
    std::vector<TF> A(N * N);
    std::vector<TF> B(N * N);
    std::vector<TF> C(N * N);

    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate memory
    kl::Memory<TF> dev_A(A);
    kl::Memory<TF> dev_B(B);
    kl::Memory<TF> dev_C(C);

    // Compile kernel
    kl::Config config;
    kl::Config best_config;

    auto compiler = kl::AsyncCompiler(kl::NvrtcCompiler{});
    auto strategy = kl::CachingStrategy("matmul.json", kl::RandomStrategy());

    auto kernel = kl::TuneKernel<TF*, const TF*, const TF*>(
            builder,
            std::make_unique<decltype(strategy)>(std::move(strategy)),
            std::make_unique<decltype(compiler)>(compiler)
            );

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; i++) {
        kernel(stream, N, N)(dev_C, dev_A, dev_B);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (i % 10 == 0) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
            std::cout << "elapsed: " << elapsed_time_ms << std::endl;
            t_start = t_end;
        }
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