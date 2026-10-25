// Serial CPU vs NVIDIA CUDA Bitonic Sort
// Compile: nvcc -O3 bitonic_serial_vs_cuda.cu -o bitonic
// Run:     ./bitonic [output_csv]
// Output:  results.csv (for plotting with plot_results.py)
//
// Bitonic sort requires the array length to be a power of two.

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void bitonic_sort_cpu(std::vector<int>& a) {
    const int n = static_cast<int>(a.size());
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < n; ++i) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool ascending = ((i & k) == 0);
                    if ((ascending && a[i] > a[ixj]) ||
                        (!ascending && a[i] < a[ixj])) {
                        std::swap(a[i], a[ixj]);
                    }
                }
            }
        }
    }
}

// One CUDA kernel per compare-exchange stage.
// Each pair is processed by exactly one thread (ixj > i).
__global__ void bitonic_stage(int* a, int j, int k, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int ixj = i ^ j;
    if (ixj > i) {
        const bool ascending = ((i & k) == 0);
        const int x = a[i];
        const int y = a[ixj];
        if ((ascending && x > y) || (!ascending && x < y)) {
            a[i] = y;
            a[ixj] = x;
        }
    }
}

static bool is_sorted(const std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i - 1] > a[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    std::string csv_filename = "results.csv";
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            std::printf("Usage: %s [output_csv]\n", argv[0]);
            std::printf("Default output: results.csv\n");
            return 0;
        }
        csv_filename = arg;
    }

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::fprintf(stderr,
            "Error: No CUDA-capable NVIDIA GPU found. Please run this program on a CUDA-enabled system.\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("===============================================================================================\n");
    std::printf(" GPU Device: %s\n", prop.name);
    std::printf(" Compute Capability: %d.%d | Multiprocessors (SMs): %d | Global Memory: %.2f GB\n",
                prop.major, prop.minor, prop.multiProcessorCount,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    std::printf("===============================================================================================\n\n");

    // Warm up GPU to eliminate one-time driver context initialization overhead from benchmark timings
    {
        int dummy = 0;
        int* d_dummy = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy), sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dummy, &dummy, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFree(d_dummy));
    }

    const int sizes[] = {1 << 10, 1 << 14, 1 << 18, 1 << 20};
    const int threads_per_block = 256;
    std::ofstream csv(csv_filename);
    if (!csv) {
        std::fprintf(stderr, "Error: Could not create output file '%s'\n", csv_filename.c_str());
        return EXIT_FAILURE;
    }
    csv << "input_size,cpu_ms,cuda_kernel_ms,total_cuda_ms,speedup_kernel,speedup_total,correct\n";

    std::printf("%-12s %-12s %-16s %-16s %-15s %-14s %-9s\n",
        "Input size", "CPU ms", "CUDA kernel ms", "Total CUDA ms",
        "Kernel speedup", "Total speedup", "Correct");
    std::printf("-----------------------------------------------------------------------------------------------\n");

    std::srand(12345);
    for (int n : sizes) {
        std::vector<int> input(n), cpu(n), gpu(n);
        for (int i = 0; i < n; ++i) input[i] = std::rand();
        cpu = input;

        const auto cpu_start = Clock::now();
        bitonic_sort_cpu(cpu);
        const auto cpu_end = Clock::now();
        const double cpu_ms = elapsed_ms(cpu_start, cpu_end);

        int* d_a = nullptr;
        const size_t bytes = static_cast<size_t>(n) * sizeof(int);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));

        cudaEvent_t kernel_start, kernel_stop;
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));

        // Total CUDA time includes Host-to-Device transfer, all kernel stages,
        // synchronization, and Device-to-Host transfer.
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto total_start = Clock::now();
        CUDA_CHECK(cudaMemcpy(d_a, input.data(), bytes, cudaMemcpyHostToDevice));

        const int blocks = (n + threads_per_block - 1) / threads_per_block;
        CUDA_CHECK(cudaEventRecord(kernel_start));
        for (int k = 2; k <= n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                bitonic_stage<<<blocks, threads_per_block>>>(d_a, j, k, n);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        float kernel_ms_f = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_f, kernel_start, kernel_stop));
        CUDA_CHECK(cudaMemcpy(gpu.data(), d_a, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto total_end = Clock::now();
        const double total_cuda_ms = elapsed_ms(total_start, total_end);
        const double kernel_ms = static_cast<double>(kernel_ms_f);

        bool correct = is_sorted(gpu) && (gpu == cpu);
        const double kernel_speedup = kernel_ms > 0.0 ? cpu_ms / kernel_ms : 0.0;
        const double total_speedup = total_cuda_ms > 0.0 ? cpu_ms / total_cuda_ms : 0.0;

        std::printf("%-12d %-12.3f %-16.3f %-16.3f %-15.2fx %-14.2fx %-9s\n",
            n, cpu_ms, kernel_ms, total_cuda_ms, kernel_speedup,
            total_speedup, correct ? "YES" : "NO");

        csv << n << "," << cpu_ms << "," << kernel_ms << "," << total_cuda_ms
            << "," << kernel_speedup << "," << total_speedup << ","
            << (correct ? "YES" : "NO") << "\n";

        if (!correct) std::fprintf(stderr, "ERROR: correctness check failed for n=%d\n", n);

        CUDA_CHECK(cudaEventDestroy(kernel_start));
        CUDA_CHECK(cudaEventDestroy(kernel_stop));
        CUDA_CHECK(cudaFree(d_a));
    }
    csv.close();
    CUDA_CHECK(cudaDeviceReset());
    std::printf("\nMeasurements saved to %s\n", csv_filename.c_str());
    return 0;
}

