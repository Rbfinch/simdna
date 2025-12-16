# simdna

High-performance DNA sequence encoding and decoding using SIMD instructions with automatic fallback to scalar implementations.

## Features

- **4-bit encoding** supporting all 16 IUPAC nucleotide codes
- **SIMD acceleration** on x86_64 (SSSE3) and ARM64 (NEON)
- **Automatic fallback** to optimized scalar implementation
- **Thread-safe** pure functions with no global state
- **2:1 compression** ratio compared to ASCII representation

## Encoding Scheme

Each nucleotide or IUPAC ambiguity code is encoded using 4 bits, with 2 nucleotides packed per byte:

| Code | Meaning              | Value | Code | Meaning              | Value |
|------|----------------------|-------|------|----------------------|-------|
| A    | Adenine              | 0x0   | K    | G or T (keto)        | 0x8   |
| C    | Cytosine             | 0x1   | M    | A or C (amino)       | 0x9   |
| G    | Guanine              | 0x2   | B    | C, G, or T (not A)   | 0xA   |
| T    | Thymine              | 0x3   | D    | A, G, or T (not C)   | 0xB   |
| R    | A or G (purine)      | 0x4   | H    | A, C, or T (not G)   | 0xC   |
| Y    | C or T (pyrimidine)  | 0x5   | V    | A, C, or G (not T)   | 0xD   |
| S    | G or C (strong)      | 0x6   | N    | Any base             | 0xE   |
| W    | A or T (weak)        | 0x7   | -    | Gap / Unknown        | 0xF   |

## Usage

```rust
use simdna::dna_simd_encoder::{encode_dna, decode_dna};

// Encode a DNA sequence with IUPAC codes
let sequence = b"ACGTNRYSWKMBDHV-";
let encoded = encode_dna(sequence);

// The encoded data is 2x smaller (2 nucleotides per byte)
assert_eq!(encoded.len(), sequence.len() / 2);

// Decode back to the original sequence
let decoded = decode_dna(&encoded, sequence.len());
assert_eq!(decoded, sequence);
```

## Platform Support

| Platform | SIMD | Fallback |
|----------|------|----------|
| x86_64   | SSSE3 | Scalar |
| ARM64    | NEON | Scalar |
| Other    | -    | Scalar |

## Performance

- SIMD processes 16 nucleotides per iteration
- 2:1 compression ratio (4 bits per nucleotide vs 8 bits ASCII)
- Expected speedup: 4-8x over scalar code on modern CPUs

## License

MIT
