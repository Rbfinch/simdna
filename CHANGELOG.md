# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2024-12-20

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

[1.0.0]: https://github.com/Rbfinch/simdna/releases/tag/v1.0.0
