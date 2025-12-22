#!/usr/bin/env bash
# Copyright (c) 2025-present Nicholas D. Crosbie
# SPDX-License-Identifier: MIT

# Nightly fuzzing script for thorough testing
# Runs each fuzz target for 10 minutes (50 minutes total)

set -euo pipefail

DURATION=600
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== simdna Nightly Fuzz Testing ==="
echo "Duration: ${DURATION}s (10 min) per target"
echo "Project: $PROJECT_DIR"
echo "Start time: $(date)"
echo ""

TARGETS=(
    "roundtrip"
    "simd_scalar_equivalence"
    "valid_iupac"
    "decode_robust"
    "boundaries"
    "bit_rotation"
    "reverse_complement"
    "revcomp_boundaries"
    "into_variants"
    "quality_roundtrip"
)

FAILED=()
START_TIME=$(date +%s)

for i in "${!TARGETS[@]}"; do
    target="${TARGETS[$i]}"
    remaining=$((${#TARGETS[@]} - i))
    
    echo "========================================"
    echo "[$((i + 1))/${#TARGETS[@]}] Fuzzing: $target"
    echo "Duration: ${DURATION}s | Remaining targets: $((remaining - 1))"
    echo "Started: $(date)"
    echo "========================================"
    
    TARGET_START=$(date +%s)
    
    if cargo +nightly fuzz run "$target" -- -max_total_time="$DURATION" -rss_limit_mb=4096; then
        TARGET_END=$(date +%s)
        TARGET_ELAPSED=$((TARGET_END - TARGET_START))
        echo "✓ $target passed (${TARGET_ELAPSED}s)"
    else
        echo "✗ $target FAILED"
        FAILED+=("$target")
        
        # Check for crash artifacts
        ARTIFACT_DIR="$PROJECT_DIR/fuzz/artifacts/$target"
        if [ -d "$ARTIFACT_DIR" ] && [ "$(ls -A "$ARTIFACT_DIR" 2>/dev/null)" ]; then
            echo "  Crash artifacts found in: $ARTIFACT_DIR"
            ls -la "$ARTIFACT_DIR"
        fi
    fi
    echo ""
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))

echo "========================================"
echo "Summary"
echo "========================================"
echo "End time: $(date)"
echo "Total duration: ${TOTAL_MINUTES} minutes (${TOTAL_ELAPSED}s)"
echo ""

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ All ${#TARGETS[@]} targets passed!"
    exit 0
else
    echo "✗ ${#FAILED[@]} target(s) failed:"
    for target in "${FAILED[@]}"; do
        echo "  - $target"
        echo "    Reproduce: cargo +nightly fuzz run $target fuzz/artifacts/$target/crash-*"
    done
    exit 1
fi
