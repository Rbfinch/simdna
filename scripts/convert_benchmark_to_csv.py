#!/usr/bin/env python3
"""
Convert benchmark output text to CSV format.

Parses the criterion benchmark text output and extracts timing data
into a structured CSV file.
"""

import re
import csv
import sys
from pathlib import Path


def parse_benchmark_output(input_file: Path) -> list[dict]:
    """
    Parse criterion benchmark text output.

    Returns a list of dicts with keys: operation, method, seq_length, time_low, time_median, time_high
    """
    results = []

    # Pattern to match benchmark result lines like:
    # encode/simd_4bit/15     time:   [101.13 ns 101.53 ns 102.21 ns]
    # or with different formatting:
    # encode/scalar_2bit/1023 time:   [801.77 ns 807.72 ns 819.67 ns]
    benchmark_pattern = re.compile(
        r"^(\w+)/(\w+)/(\d+)\s+time:\s+\[([0-9.]+)\s+(ns|µs|ms|us)\s+([0-9.]+)\s+(ns|µs|ms|us)\s+([0-9.]+)\s+(ns|µs|ms|us)\]",
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

        results.append(
            {
                "operation": operation,
                "method": method,
                "seq_length": seq_length,
                "time_low_ns": time_low,
                "time_median_ns": time_median,
                "time_high_ns": time_high,
            }
        )

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


def write_csv(results: list[dict], output_file: Path):
    """Write parsed results to CSV file."""
    if not results:
        print("No results to write!")
        return

    # Sort results for consistent ordering
    results.sort(key=lambda x: (x["operation"], x["method"], x["seq_length"]))

    fieldnames = [
        "operation",
        "method",
        "seq_length",
        "time_low_ns",
        "time_median_ns",
        "time_high_ns",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} benchmark results to {output_file}")


def main():
    script_dir = Path(__file__).parent
    artefacts_dir = script_dir.parent / "artefacts"

    input_file = artefacts_dir / "benchmark_output.txt"
    output_file = artefacts_dir / "benchmark_data.csv"

    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    print(f"Parsing {input_file}...")
    results = parse_benchmark_output(input_file)

    if not results:
        print("No benchmark results found in the input file!")
        return 1

    print(f"Found {len(results)} benchmark entries")

    # Print summary
    operations = set(r["operation"] for r in results)
    methods = set(r["method"] for r in results)
    seq_lengths = sorted(set(r["seq_length"] for r in results))

    print(f"  Operations: {', '.join(sorted(operations))}")
    print(f"  Methods: {', '.join(sorted(methods))}")
    print(
        f"  Sequence lengths: {seq_lengths[0]} to {seq_lengths[-1]} ({len(seq_lengths)} sizes)"
    )

    write_csv(results, output_file)

    return 0


if __name__ == "__main__":
    exit(main())
