# Fuzzing for simdna

This directory contains fuzz tests for the DNA encoding/decoding library using [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) and libFuzzer.

## Prerequisites

```bash
# Install cargo-fuzz (requires nightly Rust)
cargo install cargo-fuzz
```

## Fuzz Targets

| Target | Description |
|--------|-------------|
| `roundtrip` | Tests encode→decode produces original data (normalized). Most important for correctness. |
| `simd_scalar_equivalence` | Ensures SIMD and scalar implementations produce identical results for encode, decode, and reverse complement. |
| `valid_iupac` | Generates sequences with only valid IUPAC codes and verifies encoding success. |
| `decode_robust` | Tests decoder handles arbitrary/malformed input gracefully. |
| `boundaries` | Tests edge cases around SIMD width boundaries (16, 32, 64 bytes). |
| `bit_rotation` | Verifies bit rotation complement properties (involution, consistency, known pairs). |
| `reverse_complement` | Tests reverse complement correctness (double-rc = original, API consistency). |
| `revcomp_boundaries` | Tests reverse complement at SIMD threshold boundaries (32 bytes) and odd/even lengths. |
| `into_variants` | Verifies `_into` functions match allocating equivalents. |
| `quality_roundtrip` | Tests quality score encoding round-trip and SIMD/scalar equivalence. |

## Running Fuzz Tests

```bash
# List all targets
cargo fuzz list

# Run a specific target
cargo +nightly fuzz run roundtrip

# Run with time limit (seconds)
cargo +nightly fuzz run roundtrip -- -max_total_time=300

# Run with parallel jobs
cargo +nightly fuzz run roundtrip -- -jobs=8 -workers=8

# Run with memory limit
cargo +nightly fuzz run roundtrip -- -rss_limit_mb=4096

# Quick smoke test all targets
for target in roundtrip simd_scalar_equivalence valid_iupac decode_robust boundaries bit_rotation reverse_complement revcomp_boundaries; do
    echo "=== Testing $target ==="
    cargo +nightly fuzz run $target -- -max_total_time=30
done
```

## Understanding Output

- **NEW**: Found new coverage (good!)
- **REDUCE**: Minimized a corpus entry
- **pulse**: Progress heartbeat
- **DONE**: Completed without crashes

Example output:

```text
#12345  NEW    cov: 174 ft: 500 corp: 50/1000b lim: 128 exec/s: 12345 rss: 50Mb
```

- `#12345`: Number of iterations
- `cov: 174`: Edge coverage (higher = more code paths explored)
- `corp: 50/1000b`: Corpus size (50 inputs, 1000 bytes total)
- `exec/s: 12345`: Executions per second
- `rss: 50Mb`: Memory usage

## Reproducing Crashes

If a crash is found, it will be saved to `fuzz/artifacts/<target>/`:

```bash
# Reproduce a crash
cargo +nightly fuzz run roundtrip fuzz/artifacts/roundtrip/crash-<hash>

# Minimize the crash input
cargo +nightly fuzz tmin roundtrip fuzz/artifacts/roundtrip/crash-<hash>
```

## Corpus Management

The corpus (saved interesting inputs) is stored in `fuzz/corpus/<target>/`:

```bash
# Merge and minimize corpus
cargo +nightly fuzz cmin roundtrip

# Reset corpus (start fresh)
rm -rf fuzz/corpus/roundtrip
```

## CI Integration

Fuzzing runs automatically via GitHub Actions:

- **On PR/push**: Quick 60-second smoke tests
- **Nightly**: Extended 10-minute runs per target
- **Manual**: Trigger with custom duration via workflow dispatch

## Adding New Targets

1. Create a new file in `fuzz/fuzz_targets/your_target.rs`
2. Add to `fuzz/Cargo.toml`:

   ```toml
   [[bin]]
   name = "your_target"
   path = "fuzz_targets/your_target.rs"
   test = false
   doc = false
   bench = false
   ```

3. Run `cargo fuzz list` to verify it's recognized

## Coverage Analysis

```bash
# Generate coverage report (requires cargo-cov)
cargo +nightly fuzz coverage roundtrip
```
