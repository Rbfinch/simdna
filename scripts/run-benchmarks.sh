#!/usr/bin/env bash
# Copyright (c) 2025-present Nicholas D. Crosbie
# SPDX-License-Identifier: MIT

# Run benchmarks, convert to CSV, and generate plots
# All outputs are saved to the outputs/ directory with datetime prefixes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUTS_DIR="$PROJECT_DIR/outputs"

# Create outputs directory if it doesn't exist
mkdir -p "$OUTPUTS_DIR"

# Generate ISO 8601 datetime prefix (local time)
DATETIME=$(date +"%Y-%m-%dT%H-%M-%S")

# Output file paths
BENCHMARK_TXT="$OUTPUTS_DIR/${DATETIME}_benchmark.txt"
BENCHMARK_CSV="$OUTPUTS_DIR/${DATETIME}_benchmark.csv"
BENCHMARK_PLOT="$OUTPUTS_DIR/${DATETIME}_benchmark_plot.png"
THROUGHPUT_PLOT="$OUTPUTS_DIR/${DATETIME}_throughput_plot.png"

cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              simdna Benchmark Suite                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Datetime: $DATETIME"
echo "Output directory: $OUTPUTS_DIR"
echo ""

# Step 1: Run benchmarks
echo "════════════════════════════════════════════════════════════════"
echo "Step 1/3: Running benchmarks..."
echo "════════════════════════════════════════════════════════════════"
echo ""

cargo bench 2>&1 | tee "$BENCHMARK_TXT"

echo ""
echo "✓ Benchmark results saved to: $BENCHMARK_TXT"
echo ""

# Step 2: Convert to CSV
echo "════════════════════════════════════════════════════════════════"
echo "Step 2/3: Converting to CSV..."
echo "════════════════════════════════════════════════════════════════"
echo ""

python3 "$SCRIPT_DIR/convert_benchmark_to_csv.py" "$BENCHMARK_TXT" "$BENCHMARK_CSV"

echo ""
echo "✓ CSV saved to: $BENCHMARK_CSV"
echo ""

# Step 3: Generate plots
echo "════════════════════════════════════════════════════════════════"
echo "Step 3/3: Generating plots..."
echo "════════════════════════════════════════════════════════════════"
echo ""

python3 "$SCRIPT_DIR/plot_benchmarks.py" "$BENCHMARK_CSV" "$BENCHMARK_PLOT" "$THROUGHPUT_PLOT"

echo ""
echo "✓ Plots saved to:"
echo "  - $BENCHMARK_PLOT"
echo "  - $THROUGHPUT_PLOT"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Benchmark Suite Complete                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output files:"
echo "  📄 $BENCHMARK_TXT"
echo "  📊 $BENCHMARK_CSV"
echo "  📈 $BENCHMARK_PLOT"
echo "  📈 $THROUGHPUT_PLOT"
