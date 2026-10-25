"""Create the three benchmark comparison graphs for Bitonic Sort: Serial CPU vs CUDA.

Usage:
    python3 plot_results.py [results.csv] [--output-dir <dir>]

Requires:
    matplotlib (python3 -m pip install matplotlib)
"""

import csv
import os
import sys
import tempfile

# Ensure writable cache directory for Matplotlib in headless/restricted environments
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless servers & CI
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def format_elements(val, pos=None):
    """Format large numbers into human-readable K/M notation."""
    val = int(val)
    if val >= 1_048_576:
        return f"{val // 1_048_576}M\n(2^{val.bit_length() - 1})"
    elif val >= 1024:
        return f"{val // 1024}K\n(2^{val.bit_length() - 1})"
    return str(val)


def main():
    filename = "results.csv"
    output_dir = "."

    # Simple argument parsing
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        elif not args[i].startswith("-"):
            filename = args[i]
            i += 1
        else:
            i += 1

    if not os.path.exists(filename):
        sys.exit(
            f"Error: Results file '{filename}' was not found.\n"
            f"Please run the CUDA benchmark first to generate results:\n"
            f"    ./bitonic\n"
            f"or specify an existing CSV file:\n"
            f"    python3 plot_results.py path/to/results.csv"
        )

    with open(filename, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"Error: No result rows found in '{filename}'.")

    os.makedirs(output_dir, exist_ok=True)

    sizes = [int(r["input_size"]) for r in rows]
    cpu = [float(r["cpu_ms"]) for r in rows]
    kernel = [float(r["cuda_kernel_ms"]) for r in rows]
    total = [float(r["total_cuda_ms"]) for r in rows]
    speedup_kernel = [float(r["speedup_kernel"]) for r in rows]
    speedup_total = [float(r["speedup_total"]) for r in rows]

    # Chart 1: All execution times
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    ax.plot(sizes, cpu, marker="o", linewidth=2.2, markersize=7, color="#d9534f", label="Serial CPU")
    ax.plot(sizes, kernel, marker="s", linewidth=2.2, markersize=7, color="#0275d8", label="CUDA Kernel Only")
    ax.plot(sizes, total, marker="^", linewidth=2.2, markersize=7, color="#5cb85c", label="Total CUDA (Transfers Included)")
    ax.set_xlabel("Input Size (Elements)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Bitonic Sort: CPU vs CUDA Execution Time", fontsize=13, fontweight="bold", pad=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(format_elements))
    ax.set_xticks(sizes)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    chart1_path = os.path.join(output_dir, "execution_time.png")
    fig.savefig(chart1_path)
    plt.close(fig)

    # Chart 2: CPU vs CUDA kernel time
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    ax.plot(sizes, cpu, marker="o", linewidth=2.2, markersize=7, color="#d9534f", label="Serial CPU")
    ax.plot(sizes, kernel, marker="s", linewidth=2.2, markersize=7, color="#0275d8", label="CUDA Kernel Only")
    ax.set_xlabel("Input Size (Elements)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Bitonic Sort: CPU vs CUDA Kernel Time", fontsize=13, fontweight="bold", pad=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(format_elements))
    ax.set_xticks(sizes)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    chart2_path = os.path.join(output_dir, "cpu_vs_cuda_kernel.png")
    fig.savefig(chart2_path)
    plt.close(fig)

    # Chart 3: Speedup
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    ax.plot(sizes, speedup_kernel, marker="o", linewidth=2.2, markersize=7, color="#0275d8", label="Kernel Speedup (CPU / CUDA Kernel)")
    ax.plot(sizes, speedup_total, marker="s", linewidth=2.2, markersize=7, color="#5cb85c", label="Total Speedup (CPU / Total CUDA)")
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.5, label="Break-even (1.0x)")
    ax.set_xlabel("Input Size (Elements)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speedup (Factor)", fontsize=11, fontweight="bold")
    ax.set_title("Bitonic Sort: GPU Speedup over Serial CPU", fontsize=13, fontweight="bold", pad=12)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(format_elements))
    ax.set_xticks(sizes)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    fig.tight_layout()
    chart3_path = os.path.join(output_dir, "speedup.png")
    fig.savefig(chart3_path)
    plt.close(fig)

    print(f"Generated benchmark graphs in '{output_dir}':")
    print(f"  - {chart1_path}")
    print(f"  - {chart2_path}")
    print(f"  - {chart3_path}")


if __name__ == "__main__":
    main()

