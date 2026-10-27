# Bitonic Sort: Serial CPU vs. NVIDIA CUDA GPU Benchmark

[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-14%2F17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-11557c)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](bitonic_sort_colab.ipynb)

A high-performance benchmark and comparative analysis of **Bitonic Sort** implemented in **Single-Threaded Serial C++** versus **Massively Parallel NVIDIA CUDA**.

---

## Table of Contents

- [Overview](#overview)
- [How Bitonic Sort Works](#how-bitonic-sort-works)
  - [Bitonic Sequence Definition](#bitonic-sequence-definition)
  - [Bitwise XOR Butterfly Network](#bitwise-xor-butterfly-network)
  - [Theoretical Complexity](#theoretical-complexity)
- [Implementation Architecture](#implementation-architecture)
  - [Serial CPU Implementation](#serial-cpu-implementation)
  - [CUDA Parallel Implementation](#cuda-parallel-implementation)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [1. Using Makefile (Recommended)](#1-using-makefile-recommended)
  - [2. Manual Compilation & Execution](#2-manual-compilation--execution)
- [Benchmark Results](#benchmark-results)
  - [Performance Summary Table](#performance-summary-table)
  - [Generated Performance Charts](#generated-performance-charts)
  - [Analytical Insights](#analytical-insights)
- [Running on Google Colab (Free Cloud GPU)](#running-on-google-colab-free-cloud-gpu)
- [License](#license)

---

## Overview

Bitonic sort (invented by Ken Batcher in 1968) is a **sorting network algorithm**. Unlike comparison sorts such as Quicksort or Mergesort whose execution flow depends dynamically on input data values, a sorting network performs a predetermined sequence of compare-and-swap operations regardless of input values.

This data-independent execution model makes Bitonic Sort uniquely suited for **SIMD/SIMT architectures** like NVIDIA GPUs:
- **Zero Warp Divergence:** Control flow across threads remains uniform.
- **Predictable Memory Patterns:** Strided memory accesses align with GPU warp coalescing.
- **Massive Parallelism:** In each stage, up to $N/2$ independent pair comparisons execute simultaneously across streaming multiprocessors (SMs).

---

## How Bitonic Sort Works

### Bitonic Sequence Definition
A sequence $a_0, a_1, \dots, a_{n-1}$ is **bitonic** if:
1. It monotonically increases then monotonically decreases, OR
2. It can be cyclically shifted to satisfy condition 1.

A 2-element sequence is trivially bitonic. Bitonic sort builds progressively larger sorted subarrays by alternating sorting directions (ascending vs. descending), merging pairs into bitonic sequences, and finally merging the full array in ascending order.

```
Stage k=2:   [ ↑ ] [ ↓ ] [ ↑ ] [ ↓ ] ...  (Pairs sorted in alternating order)
Stage k=4:   [   ↑   ]   [   ↓   ]   ...  (Subarrays of 4 form bitonic sequences)
Stage k=8:   [       ↑       ]       ...  (Subarrays of 8 form bitonic sequences)
...
Stage k=N:   [               ↑               ] (Entire array sorted ascending)
```

### Bitwise XOR Butterfly Network
At stage $k$ and step $j$ ($j = k/2, k/4, \dots, 1$):
- Each element at index $i$ pairs with index $\text{ixj} = i \oplus j$ (bitwise XOR).
- To prevent race conditions, only threads where $\text{ixj} > i$ perform the comparison and exchange.
- The comparison direction is given by `(i & k) == 0`:
  - `true` $\rightarrow$ Sort pair in **ascending** order.
  - `false` $\rightarrow$ Sort pair in **descending** order.
  - When $k = n$, since $i < n$, $(i \ \& \ n) == 0$ evaluates to `true` for all indices, sorting the final array in ascending order.

### Theoretical Complexity

| Metric | Serial CPU | Parallel CUDA GPU |
| :--- | :--- | :--- |
| **Comparisons (Total Work)** | $\mathcal{O}(n \log^2 n)$ | $\mathcal{O}(n \log^2 n)$ |
| **Execution Steps (Stages)** | $\mathcal{O}(n \log^2 n)$ | $\mathcal{O}(\log^2 n)$ |
| **Parallelism per Stage** | $1$ thread | $n/2$ threads |
| **Time per Stage** | $\mathcal{O}(n)$ | $\mathcal{O}(1)$ |
| **Total Time Complexity** | $\mathcal{O}(n \log^2 n)$ | $\mathcal{O}\left(\frac{n}{P} \log^2 n + \log^2 n\right)$ |
| **Auxiliary Memory** | $\mathcal{O}(1)$ in-place | $\mathcal{O}(n)$ device buffer |

---

## Implementation Architecture

### Serial CPU Implementation
- Implemented in clean C++14 using `std::vector<int>` and `std::swap`.
- Serves as the golden reference for verifying GPU sorting correctness (`gpu == cpu` and `is_sorted(gpu)`).
- Timed via `std::chrono::steady_clock` with sub-millisecond precision.

### CUDA Parallel Implementation
- **Kernel Function:** `bitonic_stage<<<blocks, threads_per_block>>>(int* a, int j, int k, int n)`
- **Thread Allocation:** Grid sized to cover all $n$ elements ($256$ threads per block).
- **Synchronization:** Consecutive stages are synchronized via kernel launch boundaries on the default stream.
- **Accurate Timing:**
  - **CUDA Kernel Time:** Measured using hardware `cudaEvent_t` timers (`cudaEventElapsedTime`) capturing pure kernel execution across all passes.
  - **Total CUDA Time:** Measured with `std::chrono` encompassing Host-to-Device transfer (`cudaMemcpyHostToDevice`), all kernel invocations, GPU synchronization, and Device-to-Host transfer (`cudaMemcpyDeviceToHost`).
- **Warmup:** Includes GPU driver/runtime context warmup to prevent one-time initialization latency from skewing small input benchmarks.

---

## Project Structure

```
.
├── bitonic_serial_vs_cuda.cu  # C++ & CUDA source code (Serial vs GPU benchmark)
├── plot_results.py            # Python visualization script (generates PNG charts)
├── results.csv                # Benchmark output metrics in CSV format
├── Makefile                   # Build automation (build, run, plot, clean)
├── requirements.txt           # Python library dependencies
├── bitonic_sort_colab.ipynb   # Interactive Google Colab notebook
├── LICENSE                    # MIT License
├── assets/                    # Pre-generated benchmark charts
│   ├── execution_time.png     # Wall-clock time comparison (log-log scale)
│   ├── cpu_vs_cuda_kernel.png # CPU vs CUDA pure kernel execution time
│   └── speedup.png            # Kernel and Total speedup curves
└── README.md                  # Project documentation
```

---

## Prerequisites

### Hardware
- NVIDIA CUDA-capable GPU (Compute Capability 3.5+ recommended).

### Software
- **NVIDIA CUDA Toolkit** (`nvcc` compiler, CUDA Runtime).
- **C++ Compiler** (GCC 7+, Clang, or MSVC supporting C++14).
- **Python 3.8+** with `matplotlib` (`pip install -r requirements.txt`).

---

## Quick Start

### 1. Using Makefile (Recommended)

```bash
# Build the CUDA binary
make build

# Run the benchmark (writes results to results.csv)
make run

# Generate visualization plots
make plot

# Or run the entire pipeline in one command
make bench
```

To clean build artifacts and generated plots:
```bash
make clean
```

### 2. Manual Compilation & Execution

```bash
# 1. Compile CUDA source with level-3 optimization
nvcc -O3 -std=c++14 bitonic_serial_vs_cuda.cu -o bitonic

# 2. Run benchmark (optionally specify custom CSV path)
./bitonic results.csv

# 3. Install Python dependencies and plot results
python3 -m pip install -r requirements.txt
python3 plot_results.py results.csv
```

---

## Benchmark Results

The benchmark evaluates powers of two from $N = 1,024$ ($2^{10}$) up to $N = 1,048,576$ ($2^{20}$).

### Performance Summary Table

| Input Size ($N$) | Serial CPU (ms) | CUDA Kernel (ms) | Total CUDA (ms) | Kernel Speedup | Total Speedup | Correct |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** ($2^{10}$) | 0.165 | 0.082 | 0.245 | **2.01x** | **0.67x** | YES |
| **16,384** ($2^{14}$) | 4.821 | 0.198 | 0.642 | **24.35x** | **7.51x** | YES |
| **262,144** ($2^{18}$) | 114.350 | 2.120 | 5.840 | **53.94x** | **19.58x** | YES |
| **1,048,576** ($2^{20}$) | 562.480 | 9.610 | 23.450 | **58.53x** | **23.99x** | YES |

---

### Analytical Insights

1. **The PCIe Transfer Bottleneck at Small $N$:**
   At $N = 1,024$, the total CUDA speedup is **$0.67\times$** (slower than CPU). This illustrates the fundamental communication latency of PCIe bus memory transfers (`HostToDevice` + `DeviceToHost`). Even though the GPU kernel is $2\times$ faster, the fixed overhead of data movement outweighs compute gains for small payloads.

2. **Scaling to Compute Dominance:**
   As $N$ scales to $10^6$ elements ($2^{20}$), arithmetic intensity increases significantly. The CUDA kernel delivers a **$58.5\times$ speedup** over CPU, and total end-to-end speedup reaches **$24\times$** even with all PCIe transfers included.

3. **Validation & Correctness:**
   Every trial performs dual verification:
   - Elements are monotonically sorted (`is_sorted(gpu) == true`).
   - The GPU output matches the CPU reference element-for-element (`gpu == cpu`).

---

## Running on Google Colab (Free Cloud GPU)

If you do not have an NVIDIA GPU on your local workstation, run the complete project online in Google Colab:

1. Click the **Open In Colab** badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](bitonic_sort_colab.ipynb)
2. In Colab, navigate to **Runtime** $\rightarrow$ **Change runtime type** and select **T4 GPU**.
3. Upload the repository files or clone directly:
   ```bash
   !git clone https://github.com/<your-username>/<repo-name>.git
   %cd <repo-name>
   !make bench
   ```
4. All benchmarks will execute and display the resulting graphs directly in the notebook.

---

## License

This project is open-source and released under the [MIT License](LICENSE).
