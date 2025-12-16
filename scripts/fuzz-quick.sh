#!/usr/bin/env bash
# Quick fuzzing script for local development
# Runs each fuzz target for 60 seconds (5 minutes total)

set -euo pipefail

DURATION=60
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== simdna Quick Fuzz Testing ==="
echo "Duration: ${DURATION}s per target"
echo "Project: $PROJECT_DIR"
echo ""

TARGETS=(
    "roundtrip"
    "simd_scalar_equivalence"
    "valid_iupac"
    "decode_robust"
    "boundaries"
)

FAILED=()

for target in "${TARGETS[@]}"; do
    echo "----------------------------------------"
    echo "Fuzzing: $target (${DURATION}s)"
    echo "----------------------------------------"
    
    if cargo +nightly fuzz run "$target" -- -max_total_time="$DURATION" -rss_limit_mb=4096; then
        echo "✓ $target passed"
    else
        echo "✗ $target FAILED"
        FAILED+=("$target")
    fi
    echo ""
done

echo "========================================"
echo "Summary"
echo "========================================"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ All ${#TARGETS[@]} targets passed!"
    exit 0
else
    echo "✗ ${#FAILED[@]} target(s) failed:"
    for target in "${FAILED[@]}"; do
        echo "  - $target"
    done
    exit 1
fi
