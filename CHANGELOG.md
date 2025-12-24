# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2024-12-24

### Added

- **Hybrid 2-bit/4-bit encoder** (`hybrid_encoder` module):
  - Automatic bifurcation based on sequence content (clean ACGT vs IUPAC ambiguous)
  - 2-bit encoding for clean sequences: 4x compression (4 bases per byte)
  - 4-bit encoding for dirty sequences: 2x compression (2 bases per byte)
  - SIMD-accelerated purity check (~15+ GB/s throughput)
  - Zero-allocation `_into` variants for high-throughput pipelines
- **Binary serialization** (`serialization` module):
  - Efficient BLOB format for database storage (SQLite, etc.)
  - 14-byte self-describing header with magic bytes and version
  - Optional CRC32 checksum for data integrity
  - `to_blob` / `from_blob` functions for database integration
  - `validate_blob` for integrity checking without full deserialization
- **Tetranucleotide pattern matching** (`tetra_scanner` module):
  - `TetraLUT`: 256-entry lookup table for 4-mer pattern matching
  - Support for exact patterns and IUPAC ambiguity codes
  - Shift-And (Bitap) algorithm for patterns up to 64 bases
  - SIMD-accelerated scanning (~15+ GB/s for TetraLUT, ~8+ GB/s for Shift-And)

### Changed

- Updated documentation to cover all modules
- Improved README with comprehensive usage examples

## [1.0.2] - 2024-12-19

### Changed

- **Major reverse complement performance improvement**: SIMD now achieves consistent performance for both even and odd-length sequences
  - Odd-length sequences improved by up to 96% (e.g., length 9999: 12.5μs → 466ns)
  - Throughput now consistently ~20 GiB/s for encoded data regardless of sequence length
  - Eliminated "zig-zag" performance pattern between odd and even lengths
- New algorithm uses SIMD for all sequences ≥32 bytes, with efficient O(n) nibble shift for odd-length post-processing
- Updated documentation to reflect performance characteristics

## [1.0.1] - 2024-12-18

### Added

- Static lookup tables (256-byte encode, 16-byte decode) for branch-free encoding/decoding
- Prefetch hints for x86_64 (SSE) and ARM64 (NEON) to reduce cache misses on large sequences
- Optimized 4-at-a-time scalar path for remainder processing
- Direct SIMD encoding that processes 32 nucleotides per iteration (two 16-byte chunks)
- Cache-aligned memory structures (64-byte alignment for encode LUT)

### Changed

- Case handling now uses LUT directly instead of uppercase conversion (zero-allocation improvement)
- SIMD threshold increased from 16 to 32 nucleotides for better instruction-level parallelism
- Performance improved by 45-70% for encoding and 46-55% for decoding at larger sizes

### Fixed

- Invalid character encoding now correctly uses gap value (`0x0`) instead of `0xF`

## [1.0.0] - 2024-12-17

### Added

- Initial release of simdna
- 4-bit encoding supporting all 16 IUPAC nucleotide codes plus U (Uracil) for RNA
- SIMD acceleration using SSSE3 on x86_64 and NEON on ARM64
- Automatic fallback to optimized scalar implementation on unsupported platforms
- `encode_dna_prefer_simd` and `decode_dna_prefer_simd` public API functions
- Case-insensitive input handling
- Thread-safe pure functions with no global state
- Comprehensive unit test suite
- Fuzz testing targets for robustness verification
- CI/CD workflows for multi-platform testing (Linux x86_64, Linux ARM64, macOS ARM64)

[1.0.3]: https://github.com/Rbfinch/simdna/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/Rbfinch/simdna/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/Rbfinch/simdna/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Rbfinch/simdna/releases/tag/v1.0.0
