#!/bin/bash

echo "Starting VaultX Experiments..."
echo "Current directory: $(pwd)"
echo "Disk space available: $(df -h . | awk 'NR==2 {print $4}')"

# Clean and build
make clean
make

if [ ! -f "./vaultx" ]; then
    echo "Error: vaultx binary not found!"
    exit 1
fi

echo "=== Phase 1: Small workload (K=26) ==="
echo "This will run 42 experiments (3 memory x 7 threads x 2 I/O threads)"
echo "Estimated time: 60-90 minutes"

# Run small experiments
make run-small

echo "=== Phase 2: Large workload (K=32) ==="
echo "This will run 3 experiments (3 memory sizes)"
echo "Estimated time: 5-8 hours"

# Run large experiments  
make run-large

echo "=== Phase 3: Search experiments ==="
echo "This will run 6 search configurations (2 K values x 3 difficulties)"
echo "Estimated time: 10-20 minutes"

# Run search experiments
make run-search

echo "=== Generating results and plots ==="

# Run Python analysis
if command -v python3 &> /dev/null; then
    python3 results.py
    echo "Results analysis completed. Check CSV files and PNG plots."
else
    echo "Python3 not available. Please run results.py manually to generate plots."
fi

echo "=== All experiments completed! ==="
echo "Summary files created:"
echo "  - results_small.txt (K=26 performance)"
echo "  - results_large.txt (K=32 performance)" 
echo "  - results_search.txt (Search performance)"
echo "  - Various CSV and PNG files for analysis"