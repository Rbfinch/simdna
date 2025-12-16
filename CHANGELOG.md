# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
