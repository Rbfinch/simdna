# simdna

High-performance DNA/RNA sequence encoding and decoding using SIMD instructions with automatic fallback to scalar implementations.

## Features

- **4-bit encoding** supporting all IUPAC nucleotide codes (16 standard + U for RNA)
- **SIMD acceleration** on x86_64 (SSSE3) and ARM64 (NEON)
- **Automatic fallback** to optimized scalar implementation
- **Thread-safe** pure functions with no global state
- **2:1 compression** ratio compared to ASCII representation
- **RNA support** via U (Uracil) mapping to T

## IUPAC Nucleotide Codes

simdna supports the complete IUPAC nucleotide alphabet:

### Standard Nucleotides

| Code | Meaning              | Value |
|------|----------------------|-------|
| A    | Adenine              | 0x0   |
| C    | Cytosine             | 0x1   |
| G    | Guanine              | 0x2   |
| T    | Thymine              | 0x3   |
| U    | Uracil (RNA → T)     | 0x3   |

### Two-Base Ambiguity Codes

| Code | Meaning              | Value |
|------|----------------------|-------|
| R    | A or G (purine)      | 0x4   |
| Y    | C or T (pyrimidine)  | 0x5   |
| S    | G or C (strong)      | 0x6   |
| W    | A or T (weak)        | 0x7   |
| K    | G or T (keto)        | 0x8   |
| M    | A or C (amino)       | 0x9   |

### Three-Base Ambiguity Codes

| Code | Meaning              | Value |
|------|----------------------|-------|
| B    | C, G, or T (not A)   | 0xA   |
| D    | A, G, or T (not C)   | 0xB   |
| H    | A, C, or T (not G)   | 0xC   |
| V    | A, C, or G (not T)   | 0xD   |

### Wildcards and Gaps

| Code | Meaning              | Value |
|------|----------------------|-------|
| N    | Any base             | 0xE   |
| -    | Gap / deletion       | 0xF   |
| .    | Gap (alternative)    | 0xF   |

## Usage

```rust
use simdna::dna_simd_encoder::{encode_dna_prefer_simd, decode_dna_prefer_simd};

// Encode a DNA sequence with IUPAC codes
let sequence = b"ACGTNRYSWKMBDHV-";
let encoded = encode_dna_prefer_simd(sequence);

// The encoded data is 2x smaller (2 nucleotides per byte)
assert_eq!(encoded.len(), sequence.len() / 2);

// Decode back to the original sequence
let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
assert_eq!(decoded, sequence);

// RNA sequences work seamlessly (U maps to T)
let rna = b"ACGU";
let encoded_rna = encode_dna_prefer_simd(rna);
let decoded_rna = decode_dna_prefer_simd(&encoded_rna, rna.len());
assert_eq!(decoded_rna, b"ACGT"); // U decodes as T
```

## Input Handling

- **Case insensitive**: Both `"ACGT"` and `"acgt"` encode identically
- **Invalid characters**: Non-IUPAC characters (X, digits, etc.) encode as gap (0xF)
- **Decoding**: Always produces uppercase nucleotides

## Platform Support

| Platform | SIMD  | Fallback |
|----------|-------|----------|
| x86_64   | SSSE3 | Scalar   |
| ARM64    | NEON  | Scalar   |
| Other    | -     | Scalar   |

## Performance

- SIMD processes 16 nucleotides per iteration
- 2:1 compression ratio (4 bits per nucleotide vs 8 bits ASCII)
- Expected speedup: 4-8x over scalar code on modern CPUs

## License

MIT
