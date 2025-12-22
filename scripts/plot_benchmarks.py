#!/usr/bin/env python3
# Copyright (c) 2025-present Nicholas D. Crosbie
# SPDX-License-Identifier: MIT

"""
Plot benchmark results from CSV file.

This script parses the benchmark CSV file and creates
box-and-whisker style plots showing performance across different
sequence lengths and encoding methods.

Usage:
    plot_benchmarks.py [csv_file] [benchmark_plot] [throughput_plot]

If no arguments provided, uses default paths in artefacts/ directory.
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

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


def parse_csv_results(csv_file: Path) -> dict:
    """
    Parse benchmark results from CSV file.

    Expected CSV columns: operation, method, seq_length, time_low_ns, time_median_ns, time_high_ns

    Returns a nested dict: operation -> method -> seq_length -> {low, median, high}
    """
    results = defaultdict(lambda: defaultdict(dict))

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            operation = row["operation"]
            method = row["method"]
            seq_length = int(row["seq_length"])

            results[operation][method][seq_length] = {
                "low": float(row["time_low_ns"]),
                "median": float(row["time_median_ns"]),
                "high": float(row["time_high_ns"]),
            }

    return results


def plot_benchmarks(results: dict, output_file: str = "benchmark_plot.png"):
    """
    Create box-and-whisker style plots for benchmark results.
    """
    operations = [
        "encode",
        "decode",
        "roundtrip",
        "reverse_complement",
        "encode_into",
        "decode_into",
        "reverse_complement_into",
        "reverse_complement_encoded_into",
    ]

    # Define colors and markers for different methods
    # Includes styles for both encode/decode/roundtrip and reverse_complement benchmarks
    method_styles = {
        # Encode/decode/roundtrip methods
        "simd_2bit": {"color": "#2ecc71", "marker": "o", "label": "SIMD 2-bit"},
        "simd_4bit": {"color": "#3498db", "marker": "s", "label": "SIMD 4-bit"},
        "scalar_2bit": {"color": "#e74c3c", "marker": "^", "label": "Scalar 2-bit"},
        "scalar_4bit": {"color": "#f39c12", "marker": "D", "label": "Scalar 4-bit"},
        # Reverse complement methods
        "simd": {"color": "#3498db", "marker": "s", "label": "SIMD (ASCII)"},
        "simd_high_level": {"color": "#3498db", "marker": "s", "label": "SIMD (ASCII)"},
        "simd_encoded": {"color": "#9b59b6", "marker": "p", "label": "SIMD (Encoded)"},
        "scalar": {"color": "#e74c3c", "marker": "^", "label": "Scalar"},
        # _into variants (allocation comparison)
        "allocating": {"color": "#e74c3c", "marker": "^", "label": "Allocating"},
        "into_preallocated": {
            "color": "#2ecc71",
            "marker": "o",
            "label": "Zero-alloc (_into)",
        },
    }

    # Responsive sizing: 8 subplots for all benchmark groups
    fig, axes = plt.subplots(8, 1, figsize=(10, 36))
    fig.suptitle(
        "DNA Encoding/Decoding Benchmark Results",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    for ax, operation in zip(axes, operations):
        op_data = results.get(operation, {})

        if not op_data:
            ax.set_title(f"{operation.capitalize()} (no data)", fontsize=14)
            continue

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

            # Plot with error bars - larger markers and thicker lines for visibility
            ax.errorbar(
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

        ax.set_xlabel("Sequence Length (bases)", fontsize=13)
        ax.set_ylabel("Time (ns)", fontsize=13)
        ax.set_title(
            f"{operation.replace('_', ' ').title()}", fontsize=15, fontweight="bold"
        )
        ax.set_xscale("log", base=10)
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.tick_params(axis="both", which="minor", labelsize=9)
        ax.grid(True, which="major", axis="y", alpha=0.5, linestyle="-", linewidth=0.8)
        ax.grid(True, which="minor", axis="y", alpha=0.2, linestyle="--", linewidth=0.5)
        ax.grid(True, which="major", axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot saved to {output_file}")
    plt.close(fig)  # Close to free memory


def plot_throughput(results: dict, output_file: str = "throughput_plot.png"):
    """
    Create throughput plots (bases per second) for benchmark results.
    """
    operations = [
        "encode",
        "decode",
        "roundtrip",
        "reverse_complement",
        "encode_into",
        "decode_into",
        "reverse_complement_into",
        "reverse_complement_encoded_into",
    ]
    titles = {
        "encode": "Encode",
        "decode": "Decode",
        "roundtrip": "Roundtrip (Total)",
        "reverse_complement": "Reverse Complement",
        "encode_into": "Encode (Alloc vs Zero-alloc)",
        "decode_into": "Decode (Alloc vs Zero-alloc)",
        "reverse_complement_into": "Reverse Complement (Alloc vs Zero-alloc)",
        "reverse_complement_encoded_into": "RC Encoded (Alloc vs Zero-alloc)",
    }

    # Includes styles for both encode/decode/roundtrip and reverse_complement benchmarks
    method_styles = {
        # Encode/decode/roundtrip methods
        "simd_2bit": {"color": "#2ecc71", "marker": "o", "label": "SIMD 2-bit"},
        "simd_4bit": {"color": "#3498db", "marker": "s", "label": "SIMD 4-bit"},
        "scalar_2bit": {"color": "#e74c3c", "marker": "^", "label": "Scalar 2-bit"},
        "scalar_4bit": {"color": "#f39c12", "marker": "D", "label": "Scalar 4-bit"},
        # Reverse complement methods
        "simd": {"color": "#3498db", "marker": "s", "label": "SIMD (ASCII)"},
        "simd_high_level": {"color": "#3498db", "marker": "s", "label": "SIMD (ASCII)"},
        "simd_encoded": {"color": "#9b59b6", "marker": "p", "label": "SIMD (Encoded)"},
        "scalar": {"color": "#e74c3c", "marker": "^", "label": "Scalar"},
        # _into variants (allocation comparison)
        "allocating": {"color": "#e74c3c", "marker": "^", "label": "Allocating"},
        "into_preallocated": {
            "color": "#2ecc71",
            "marker": "o",
            "label": "Zero-alloc (_into)",
        },
    }

    # Responsive sizing: 8 subplots for all benchmark groups
    fig, axes = plt.subplots(8, 1, figsize=(10, 36))
    fig.suptitle(
        "DNA Encoding/Decoding Throughput", fontsize=18, fontweight="bold", y=0.98
    )

    for ax, operation in zip(axes, operations):
        op_data = results.get(operation, {})

        if not op_data:
            ax.set_title(
                f"{titles.get(operation, operation.capitalize())} (no data)",
                fontsize=14,
            )
            continue

        for method, style in method_styles.items():
            if method not in op_data:
                continue

            method_data = op_data[method]

            # Sort by sequence length
            seq_lengths = sorted(method_data.keys())

            # Calculate throughput: bases / time_ns * 1e9 = bases per second
            throughputs = [sl / method_data[sl]["median"] * 1e9 for sl in seq_lengths]
            throughput_low = [
                sl / method_data[sl]["high"] * 1e9 for sl in seq_lengths
            ]  # Note: high time = low throughput
            throughput_high = [sl / method_data[sl]["low"] * 1e9 for sl in seq_lengths]

            yerr_low = [
                throughputs[i] - throughput_low[i] for i in range(len(throughputs))
            ]
            yerr_high = [
                throughput_high[i] - throughputs[i] for i in range(len(throughputs))
            ]

            # Larger markers and thicker lines for visibility
            ax.errorbar(
                seq_lengths,
                throughputs,
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

        ax.set_xlabel("Sequence Length (bases)", fontsize=13)
        ax.set_ylabel("Throughput (bases/s)", fontsize=13)
        ax.set_title(
            f"{titles.get(operation, operation.capitalize())} Throughput",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xscale("log", base=10)
        ax.set_yscale("log")
        # Set up tick locations - more labels for encode/decode/roundtrip, fewer for reverse_complement
        if operation == "reverse_complement":
            # Fewer ticks for reverse_complement (wider range)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=10))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, p: ""))
        else:
            # More ticks for encode/decode/roundtrip
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
            ax.yaxis.set_minor_locator(
                LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=15)
            )
            ax.yaxis.set_minor_formatter(FuncFormatter(format_throughput))
        ax.yaxis.set_major_formatter(FuncFormatter(format_throughput))
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=7)
        ax.grid(True, which="major", axis="y", alpha=0.5, linestyle="-", linewidth=0.8)
        ax.grid(True, which="minor", axis="y", alpha=0.2, linestyle="--", linewidth=0.5)
        ax.grid(True, which="major", axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)

        # Set Y-axis maximum for roundtrip to match encode scale
        if operation == "roundtrip":
            ax.set_ylim(top=4.0e9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Throughput plot saved to {output_file}")
    plt.close(fig)  # Close to free memory


def print_summary(results: dict):
    """Print a summary table of the benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for operation in [
        "encode",
        "decode",
        "roundtrip",
        "reverse_complement",
        "encode_into",
        "decode_into",
        "reverse_complement_into",
        "reverse_complement_encoded_into",
    ]:
        op_data = results.get(operation, {})
        if not op_data:
            continue

        print(f"\n{operation.upper()}")
        print("-" * 60)

        # Get all sequence lengths
        all_lengths = set()
        for method_data in op_data.values():
            all_lengths.update(method_data.keys())

        for seq_len in sorted(all_lengths):
            print(f"\n  Sequence Length: {seq_len}")
            for method in sorted(op_data.keys()):
                if seq_len in op_data[method]:
                    data = op_data[method][seq_len]
                    print(
                        f"    {method:15s}: {data['median']:10.2f} ns "
                        f"[{data['low']:10.2f} - {data['high']:10.2f}]"
                    )


def main():
    """
    Generate benchmark plots from CSV data.

    Usage:
        plot_benchmarks.py [csv_file] [benchmark_plot] [throughput_plot]

    If no arguments provided, uses default paths in artefacts/ directory.
    """
    import sys

    script_dir = Path(__file__).parent
    artefacts_dir = script_dir.parent / "artefacts"

    # Default paths (for backwards compatibility)
    csv_file = artefacts_dir / "benchmark_data_optimised.csv"
    benchmark_plot = artefacts_dir / "benchmark_plot.png"
    throughput_plot = artefacts_dir / "throughput_plot.png"

    # Accept command-line arguments
    if len(sys.argv) > 1:
        csv_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        benchmark_plot = Path(sys.argv[2])
    if len(sys.argv) > 3:
        throughput_plot = Path(sys.argv[3])

    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        print("Please run convert_benchmark_to_csv.py first")
        return 1

    print(f"Parsing benchmark results from {csv_file}")
    results = parse_csv_results(csv_file)

    if not results:
        print("No benchmark results found!")
        return 1

    # Print summary
    print_summary(results)

    # Create plots
    plot_benchmarks(results, str(benchmark_plot))
    plot_throughput(results, str(throughput_plot))

    return 0


if __name__ == "__main__":
    exit(main())
