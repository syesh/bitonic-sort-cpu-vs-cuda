# Makefile for Serial CPU vs NVIDIA CUDA Bitonic Sort Benchmark

NVCC        := nvcc
NVCC_FLAGS  := -O3 -std=c++14
TARGET      := bitonic
SRC         := bitonic_serial_vs_cuda.cu
CSV_OUT     := results.csv
PYTHON      := python3
PLOT_SCRIPT := plot_results.py

.PHONY: all build run plot bench clean help

all: build

## build: Compile the CUDA executable using nvcc
build: $(TARGET)

$(TARGET): $(SRC)
	@echo "==> Compiling $(SRC) with $(NVCC)..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "==> Build successful: $(TARGET)"

## run: Execute the benchmark binary (generates results.csv)
run: $(TARGET)
	@echo "==> Running $(TARGET)..."
	./$(TARGET) $(CSV_OUT)

## plot: Generate comparison graphs from results.csv
plot: $(PLOT_SCRIPT)
	@echo "==> Plotting results from $(CSV_OUT)..."
	$(PYTHON) $(PLOT_SCRIPT) $(CSV_OUT)

## bench: Build, execute benchmark, and plot graphs
bench: build run plot

## clean: Remove compiled binary and generated outputs
clean:
	@echo "==> Cleaning generated artifacts..."
	rm -f $(TARGET)
	rm -f execution_time.png cpu_vs_cuda_kernel.png speedup.png
	@echo "==> Clean complete."

## help: Display available make targets
help:
	@echo "Available targets:"
	@echo "  make build  - Compile $(TARGET) using $(NVCC)"
	@echo "  make run    - Run benchmark and produce $(CSV_OUT)"
	@echo "  make plot   - Generate plots from $(CSV_OUT)"
	@echo "  make bench  - Run complete workflow (build, run, plot)"
	@echo "  make clean  - Remove binary and generated chart files"
