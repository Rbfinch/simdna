#!/usr/bin/env python3
# Copyright (c) 2025-present Nicholas D. Crosbie
# SPDX-License-Identifier: MIT

"""
Plot quality encoding benchmark results.

This script parses the quality benchmark text output and creates
performance plots showing SIMD vs scalar comparison and throughput
across different sequence lengths.

Usage:
    plot_quality_benchmarks.py [input_txt] [output_plot]

If no arguments provided, looks for the most recent quality benchmark file
in the outputs/ directory.
"""

import csv
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Use non-interactive backend when running from script
if not sys.stdout.isatty():
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np


def format_throughput(value, pos):
    """Format throughput values with appropriate SI prefixes."""
    if value >= 1e9:
        return f"{value / 1e9:.1f} G"
    elif value >= 1e6:
        return f"{value / 1e6:.0f} M"
    elif value >= 1e3:
        return f"{value / 1e3:.0f} K"
    else:
        return f"{value:.0f}"


def parse_benchmark_output(input_file: Path) -> dict:
    """
    Parse criterion benchmark text output for quality benchmarks.

    Returns a nested dict: operation -> method -> seq_length -> {low, median, high}
    """
    results = defaultdict(lambda: defaultdict(dict))

    # Pattern to match benchmark result lines like:
    # quality_encode/simd/15  time:   [55.928 ns 55.999 ns 56.072 ns]
    benchmark_pattern = re.compile(
        r"^(quality_\w+)/(\w+)/(\d+)\s+time:\s+\[([0-9.]+)\s+(ns|µs|ms|us)\s+([0-9.]+)\s+(ns|µs|ms|us)\s+([0-9.]+)\s+(ns|µs|ms|us)\]",
        re.MULTILINE,
    )

    with open(input_file, "r") as f:
        content = f.read()

    for match in benchmark_pattern.finditer(content):
        operation = match.group(1)
        method = match.group(2)
        seq_length = int(match.group(3))

        # Parse time values and convert to nanoseconds
        time_low = convert_to_ns(float(match.group(4)), match.group(5))
        time_median = convert_to_ns(float(match.group(6)), match.group(7))
        time_high = convert_to_ns(float(match.group(8)), match.group(9))

        results[operation][method][seq_length] = {
            "low": time_low,
            "median": time_median,
            "high": time_high,
        }

    return results


def convert_to_ns(value: float, unit: str) -> float:
    """Convert time value to nanoseconds."""
    unit = unit.lower()
    if unit == "ns":
        return value
    elif unit in ("µs", "us"):
        return value * 1000
    elif unit == "ms":
        return value * 1_000_000
    elif unit == "s":
        return value * 1_000_000_000
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def plot_quality_benchmarks(results: dict, output_file: str):
    """
    Create benchmark plots for quality encoding results.
    """
    # Define the operations we expect to find
    operations = [
        "quality_encode",
        "quality_decode",
        "quality_roundtrip",
        "quality_binning",
        "quality_encode_into",
        "quality_decode_into",
        "quality_stats",
        "quality_compression",
    ]

    titles = {
        "quality_encode": "Quality Encode (SIMD vs Scalar)",
        "quality_decode": "Quality Decode",
        "quality_roundtrip": "Quality Roundtrip (Encode + Decode)",
        "quality_binning": "Quality Binning (SIMD vs Scalar)",
        "quality_encode_into": "Quality Encode (Alloc vs Zero-alloc)",
        "quality_decode_into": "Quality Decode (Alloc vs Zero-alloc)",
        "quality_stats": "Quality Statistics Computation",
        "quality_compression": "Quality Compression Profiles",
    }

    # Define colors and markers for different methods
    method_styles = {
        "simd": {"color": "#3498db", "marker": "s", "label": "SIMD"},
        "scalar": {"color": "#e74c3c", "marker": "^", "label": "Scalar"},
        "allocating": {"color": "#e74c3c", "marker": "^", "label": "Allocating"},
        "into_preallocated": {
            "color": "#2ecc71",
            "marker": "o",
            "label": "Zero-alloc (_into)",
        },
        "compute": {"color": "#9b59b6", "marker": "p", "label": "Compute"},
        "realistic": {"color": "#3498db", "marker": "s", "label": "Realistic Profile"},
        "uniform_high": {"color": "#2ecc71", "marker": "o", "label": "Uniform High-Q"},
    }

    # Filter to only operations that have data
    available_ops = [op for op in operations if op in results and results[op]]

    if not available_ops:
        print("No quality benchmark data found!")
        return

    # Create figure with subplots for each operation
    n_plots = len(available_ops)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 4 * n_plots))

    if n_plots == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle(
        "Quality Encoding Benchmark Results",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    for idx, operation in enumerate(available_ops):
        op_data = results[operation]
        ax_time = axes[idx, 0]
        ax_throughput = axes[idx, 1]

        for method, style in method_styles.items():
            if method not in op_data:
                continue

            method_data = op_data[method]

            # Sort by sequence length
            seq_lengths = sorted(method_data.keys())
            medians = [method_data[sl]["median"] for sl in seq_lengths]
            lows = [method_data[sl]["low"] for sl in seq_lengths]
            highs = [method_data[sl]["high"] for sl in seq_lengths]

            # Calculate error bars (distance from median)
            yerr_low = [medians[i] - lows[i] for i in range(len(medians))]
            yerr_high = [highs[i] - medians[i] for i in range(len(medians))]

            # Time plot (left)
            ax_time.errorbar(
                seq_lengths,
                medians,
                yerr=[yerr_low, yerr_high],
                fmt=style["marker"] + "-",
                color=style["color"],
                label=style["label"],
                capsize=4,
                capthick=1.5,
                markersize=8,
                linewidth=2,
                alpha=0.85,
            )

            # Throughput plot (right)
            throughputs = [sl / medians[i] * 1e9 for i, sl in enumerate(seq_lengths)]
            throughput_low = [sl / highs[i] * 1e9 for i, sl in enumerate(seq_lengths)]
            throughput_high = [sl / lows[i] * 1e9 for i, sl in enumerate(seq_lengths)]

            tp_yerr_low = [
                throughputs[i] - throughput_low[i] for i in range(len(throughputs))
            ]
            tp_yerr_high = [
                throughput_high[i] - throughputs[i] for i in range(len(throughputs))
            ]

            ax_throughput.errorbar(
                seq_lengths,
                throughputs,
                yerr=[tp_yerr_low, tp_yerr_high],
                fmt=style["marker"] + "-",
                color=style["color"],
                label=style["label"],
                capsize=4,
                capthick=1.5,
                markersize=8,
                linewidth=2,
                alpha=0.85,
            )

        # Configure time plot
        ax_time.set_xlabel("Sequence Length (bases)", fontsize=11)
        ax_time.set_ylabel("Time (ns)", fontsize=11)
        ax_time.set_title(
            f"{titles.get(operation, operation)} - Time",
            fontsize=13,
            fontweight="bold",
        )
        ax_time.set_xscale("log", base=10)
        ax_time.set_yscale("log")
        ax_time.tick_params(axis="both", which="major", labelsize=10)
        ax_time.grid(
            True, which="major", axis="y", alpha=0.5, linestyle="-", linewidth=0.8
        )
        ax_time.grid(
            True, which="minor", axis="y", alpha=0.2, linestyle="--", linewidth=0.5
        )
        ax_time.grid(
            True, which="major", axis="x", alpha=0.3, linestyle="--", linewidth=0.5
        )
        ax_time.legend(loc="upper left", fontsize=10, framealpha=0.9)

        # Configure throughput plot
        ax_throughput.set_xlabel("Sequence Length (bases)", fontsize=11)
        ax_throughput.set_ylabel("Throughput (bases/s)", fontsize=11)
        ax_throughput.set_title(
            f"{titles.get(operation, operation)} - Throughput",
            fontsize=13,
            fontweight="bold",
        )
        ax_throughput.set_xscale("log", base=10)
        ax_throughput.set_yscale("log")
        ax_throughput.yaxis.set_major_formatter(FuncFormatter(format_throughput))
        ax_throughput.tick_params(axis="both", which="major", labelsize=10)
        ax_throughput.grid(
            True, which="major", axis="y", alpha=0.5, linestyle="-", linewidth=0.8
        )
        ax_throughput.grid(
            True, which="minor", axis="y", alpha=0.2, linestyle="--", linewidth=0.5
        )
        ax_throughput.grid(
            True, which="major", axis="x", alpha=0.3, linestyle="--", linewidth=0.5
        )
        ax_throughput.legend(loc="lower right", fontsize=10, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot saved to {output_file}")
    plt.close(fig)


def print_summary(results: dict):
    """Print a summary table of the benchmark results."""
    print("\n" + "=" * 80)
    print("QUALITY ENCODING BENCHMARK SUMMARY")
    print("=" * 80)

    for operation in sorted(results.keys()):
        op_data = results[operation]
        if not op_data:
            continue

        print(f"\n{operation.upper()}")
        print("-" * 60)

        # Get all methods and sequence lengths
        methods = sorted(op_data.keys())
        all_lengths = set()
        for method_data in op_data.values():
            all_lengths.update(method_data.keys())

        # Print header
        print(f"  {'Length':>8}", end="")
        for method in methods:
            print(f"  {method:>15}", end="")
        print()

        # Print data rows
        for seq_len in sorted(all_lengths):
            print(f"  {seq_len:>8}", end="")
            for method in methods:
                if seq_len in op_data[method]:
                    median = op_data[method][seq_len]["median"]
                    if median >= 1000:
                        print(f"  {median / 1000:>12.2f} µs", end="")
                    else:
                        print(f"  {median:>12.2f} ns", end="")
                else:
                    print(f"  {'N/A':>15}", end="")
            print()

        # Print speedup comparison if both simd and scalar exist
        if "simd" in methods and "scalar" in methods:
            print("\n  Speedup (scalar/simd):")
            for seq_len in sorted(all_lengths):
                if seq_len in op_data["simd"] and seq_len in op_data["scalar"]:
                    simd_time = op_data["simd"][seq_len]["median"]
                    scalar_time = op_data["scalar"][seq_len]["median"]
                    speedup = scalar_time / simd_time
                    indicator = "✓" if speedup > 1 else "✗"
                    print(f"    {seq_len:>6}: {speedup:>6.2f}x {indicator}")


def find_latest_quality_benchmark(outputs_dir: Path) -> Path | None:
    """Find the most recent quality benchmark file in outputs directory."""
    pattern = "*quality*benchmark*.txt"
    files = list(outputs_dir.glob(pattern))

    if not files:
        return None

    # Sort by modification time, newest first
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0]


def main():
    """
    Generate quality encoding benchmark plots.

    Usage:
        plot_quality_benchmarks.py [input_txt] [output_plot]

    If no arguments provided, looks for the most recent quality benchmark file
    in the outputs/ directory.
    """
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    outputs_dir = project_dir / "outputs"

    # Default output filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    default_output = outputs_dir / f"{timestamp}_quality_benchmark_plot.png"

    # Accept command-line arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        # Find latest quality benchmark file
        input_file = find_latest_quality_benchmark(outputs_dir)
        if input_file is None:
            print(f"Error: No quality benchmark files found in {outputs_dir}")
            print(
                "Run: cargo bench -- quality_encode 2>&1 | tee outputs/quality_benchmark.txt"
            )
            return 1
        print(f"Using latest benchmark file: {input_file}")

    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    else:
        output_file = default_output

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    print(f"Parsing benchmark results from {input_file}")
    results = parse_benchmark_output(input_file)

    if not results:
        print("No quality benchmark results found in the input file!")
        return 1

    # Print summary
    total_entries = sum(sum(len(m) for m in op.values()) for op in results.values())
    print(f"Found {total_entries} benchmark entries across {len(results)} operations")
    print_summary(results)

    # Create plots
    plot_quality_benchmarks(results, str(output_file))

    return 0


if __name__ == "__main__":
    exit(main())
