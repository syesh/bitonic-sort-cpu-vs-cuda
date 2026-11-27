# Bitonic Sort: CPU vs CUDA GPU Benchmark ⚡

A comparative performance analysis of **Bitonic Sort** implemented using Serial C++ and Parallel NVIDIA CUDA. This project benchmarks sorting performance across different input sizes and visualizes the speedup achieved through GPU parallelization.

![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-14%2F17-00599C?logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-11557c)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](bitonic_sort_colab.ipynb)

Run the benchmark using Google's free cloud GPU without requiring a local NVIDIA GPU.

## ✨ Features

- Serial Bitonic Sort implementation using C++.
- Parallel Bitonic Sort implementation using NVIDIA CUDA.
- Performance comparison across different input sizes.
- CUDA kernel and total execution time measurements.
- GPU speedup calculation and correctness verification.
- Performance visualization using Matplotlib.
- Benchmark results exported as CSV.
- Automated build and execution using Makefile.

## 🛠️ Tech Stack

- C++14
- NVIDIA CUDA
- Python
- Matplotlib
- Makefile
- Google Colab

## 📊 Benchmark Results

| Input Size | Kernel Speedup | Total Speedup |
|---|---:|---:|
| 1,024 | 2.01x | 0.67x |
| 16,384 | 24.35x | 7.51x |
| 262,144 | 53.94x | 19.58x |
| 1,048,576 | 58.53x | 23.99x |

<text>At larger input sizes, CUDA achieves significant performance improvements over serial CPU execution.</text>

## 📈 Performance Visualizations

The project generates charts comparing execution time, CUDA kernel performance, and overall speedup.

## 💻 Run Locally

**Prerequisites:** NVIDIA CUDA Toolkit, C++14 compiler, Python 3.8+.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build and run the benchmark:

```bash
make build
make run
make plot
```

Or execute the complete pipeline:

```bash
make bench
```

## 📁 Project Structure

```text
├── bitonic_serial_vs_cuda.cu
├── bitonic_sort_colab.ipynb
├── plot_results.py
├── results.csv
├── Makefile
├── requirements.txt
├── assets/
│   ├── execution_time.png
│   ├── cpu_vs_cuda_kernel.png
│   └── speedup.png
├── LICENSE
└── README.md
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
