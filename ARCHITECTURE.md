# Simdna Architecture

This document provides a comprehensive overview of the simdna library architecture, focusing on high-performance SIMD-accelerated DNA sequence encoding and pattern matching.

## Overview

Simdna is a Rust-based library providing high-performance DNA/RNA sequence encoding and decoding using SIMD instructions with automatic fallback to scalar implementations. The library supports all IUPAC nucleotide codes and features bifurcated encoding for optimal storage efficiency.

## Project Structure

```
src/
├── lib.rs             # Library entry point and module exports
├── dna_simd_encoder.rs # Core 4-bit IUPAC encoding with SIMD
├── hybrid_encoder.rs   # Bifurcated 2-bit/4-bit encoding
├── tetra_scanner.rs    # Tetranucleotide pattern matching
├── quality_encoder.rs  # FASTQ quality score compression
└── serialization.rs    # Binary serialization for database storage
```

## Module Descriptions

---

### lib.rs - Library Entry Point

The main entry point that exports the public API and provides crate-level documentation.

#### Exported Modules

| Module             | Purpose                                                   |
|--------------------|-----------------------------------------------------------|
| `dna_simd_encoder` | Core 4-bit encoding/decoding with SIMD acceleration       |
| `hybrid_encoder`   | Bifurcated 2-bit/4-bit encoding based on sequence content |
| `tetra_scanner`    | TetraLUT and Shift-And pattern matching algorithms        |
| `quality_encoder`  | FASTQ quality score compression with binning and RLE      |
| `serialization`    | Binary serialization for SQLite BLOB storage              |

---

### dna_simd_encoder.rs - Core 4-bit Encoding

High-performance DNA sequence encoding and decoding using SIMD instructions with automatic fallback to scalar implementations.

#### Encoding Scheme (4-bit)

The encoding is designed to support efficient complement calculation via 2-bit rotation:

```
complement = ((bits << 2) | (bits >> 2)) & 0xF
```

**Standard Nucleotides:**

| Code | Meaning      | Binary | Complement |
|------|--------------|--------|------------|
| A    | Adenine      | 0x1    | T (0x4)    |
| C    | Cytosine     | 0x2    | G (0x8)    |
| T    | Thymine      | 0x4    | A (0x1)    |
| G    | Guanine      | 0x8    | C (0x2)    |
| U    | Uracil (RNA) | 0x4    | A (0x1)    |

**Two-Base Ambiguity Codes:**

| Code | Meaning             | Binary | Complement |
|------|---------------------|--------|------------|
| R    | A or G (purine)     | 0x9    | Y (0x6)    |
| Y    | C or T (pyrimidine) | 0x6    | R (0x9)    |
| S    | G or C (strong)     | 0xA    | S (0xA)    |
| W    | A or T (weak)       | 0x5    | W (0x5)    |
| K    | G or T (keto)       | 0xC    | M (0x3)    |
| M    | A or C (amino)      | 0x3    | K (0xC)    |

**Three-Base Ambiguity Codes:**

| Code | Meaning            | Binary | Complement |
|------|--------------------|--------|------------|
| B    | C, G, or T (not A) | 0xE    | V (0xB)    |
| D    | A, G, or T (not C) | 0xD    | H (0x7)    |
| H    | A, C, or T (not G) | 0x7    | D (0xD)    |
| V    | A, C, or G (not T) | 0xB    | B (0xE)    |

**Wildcards and Gaps:**

| Code | Meaning        | Binary | Complement |
|------|----------------|--------|------------|
| N    | Any base       | 0xF    | N (0xF)    |
| -    | Gap / deletion | 0x0    | - (0x0)    |

#### Key Functions

| Function                                | Description                          |
|-----------------------------------------|--------------------------------------|
| `encode_dna_prefer_simd(sequence)`      | Encodes DNA to 4-bit packed format   |
| `decode_dna_prefer_simd(encoded, len)`  | Decodes 4-bit packed to ASCII        |
| `encode_dna_into(sequence, buffer)`     | Zero-allocation encoding into buffer |
| `decode_dna_into(encoded, len, buffer)` | Zero-allocation decoding into buffer |
| `reverse_complement(sequence)`          | Computes reverse complement (SIMD)   |
| `reverse_complement_into(seq, buffer)`  | Zero-allocation reverse complement   |
| `required_encoded_len(len)`             | Returns bytes needed for encoding    |
| `required_decoded_len(len)`             | Returns bytes needed for decoding    |

#### Lookup Tables

| Table        | Size      | Alignment | Purpose                |
|--------------|-----------|-----------|------------------------|
| `ENCODE_LUT` | 256 bytes | 64 bytes  | ASCII → 4-bit encoding |
| `DECODE_LUT` | 16 bytes  | 16 bytes  | 4-bit → ASCII decoding |

#### Performance Optimizations

| Technique            | Description                                         |
|----------------------|-----------------------------------------------------|
| Static LUTs          | Pre-computed tables eliminate branch mispredictions |
| SIMD Processing      | 32 nucleotides per iteration                        |
| Prefetch Hints       | 128 bytes (2 cache lines) ahead                     |
| Direct Case Handling | LUT handles case-insensitivity                      |
| 4-at-a-time Scalar   | Optimized remainder processing                      |
| 64-byte Alignment    | Optimal cache line access                           |

---

### hybrid_encoder.rs - Bifurcated Encoding

Automatic selection between 2-bit and 4-bit encoding based on sequence content.

#### Design Rationale

In typical genomic datasets, 99%+ of sequences contain only standard ACGT bases. By using 2-bit encoding for these "clean" sequences:

- **4x memory density** compared to ASCII (vs 2x for 4-bit)
- **Reduced memory bandwidth** for SIMD scanning
- **Perfect alignment** with tetranucleotide lookup tables

#### EncodingType

```rust
#[repr(u8)]
pub enum EncodingType {
    Clean2Bit = 0,  // 4 bases per byte (ACGT only)
    Dirty4Bit = 1,  // 2 bases per byte (full IUPAC)
}
```

| Method                | Description                        |
|-----------------------|------------------------------------|
| `compression_ratio()` | Returns 4.0 (2-bit) or 2.0 (4-bit) |
| `bases_per_byte()`    | Returns 4 (2-bit) or 2 (4-bit)     |
| `bits_per_base()`     | Returns 2 or 4                     |
| `from_u8(value)`      | Converts from integer              |
| `from_i32(value)`     | Converts from database column      |

#### 2-bit Encoding Scheme

| Base | Binary | Decimal |
|------|--------|---------|
| A    | 00     | 0       |
| C    | 01     | 1       |
| G    | 10     | 2       |
| T    | 11     | 3       |

Packing: `(B0 << 6) | (B1 << 4) | (B2 << 2) | B3`

#### EncodedSequence

```rust
pub struct EncodedSequence {
    pub encoding: EncodingType,  // Which encoding was used
    pub data: Vec<u8>,           // Packed binary data
    pub original_len: usize,     // Original length in nucleotides
}
```

| Method                     | Description                         |
|----------------------------|-------------------------------------|
| `new(encoding, data, len)` | Creates a new encoded sequence      |
| `encoded_len()`            | Returns compressed data length      |
| `compression_ratio()`      | Returns actual compression achieved |

#### Key Functions

| Function                      | Description                     |
|-------------------------------|---------------------------------|
| `encode_bifurcated(sequence)` | Auto-selects optimal encoding   |
| `decode_bifurcated(encoded)`  | Decodes any encoding type       |
| `is_clean_sequence(sequence)` | Checks if ACGT-only (SIMD)      |
| `encode_2bit(sequence)`       | Forces 2-bit encoding           |
| `decode_2bit(encoded, len)`   | Decodes 2-bit data              |
| `encoding_2bit(base)`         | Encodes single base to 2-bit    |
| `required_2bit_len(len)`      | Bytes needed for 2-bit encoding |

---

### tetra_scanner.rs - Pattern Matching

High-performance pattern matching using tetranucleotide lookup tables and SIMD acceleration.

#### Design Rationale

Tetranucleotides (4-mers) are the natural unit for pattern matching in 2-bit encoded DNA:

- 4 bases × 2 bits = 8 bits = 1 byte exactly
- All 256 possible 4-mers map to byte values 0-255
- SIMD can process 16 tetranucleotides per instruction
- Pre-computed LUTs enable O(1) pattern matching per position

#### TetraLUT Structure

```rust
pub struct TetraLUT {
    lut: [u8; 256],        // 0xFF = match, 0x00 = no match
    pattern: [u8; 4],      // Original pattern
    match_count: usize,    // Number of matching tetranucleotides
}
```

| Method                         | Description                             |
|--------------------------------|-----------------------------------------|
| `from_literal(pattern)`        | Creates LUT for exact 4-mer (ACGT only) |
| `from_iupac(pattern)`          | Creates LUT with IUPAC expansion        |
| `matches_index(index)`         | Checks if index matches pattern         |
| `match_count()`                | Returns number of matching 4-mers       |
| `lut()`                        | Returns reference to internal table     |
| `scan_2bit(encoded, len)`      | Scans for matches (returns positions)   |
| `contains_match(encoded, len)` | Early-exit existence check              |

#### IUPAC Expansion

| Code | Expands To | Count |
|------|------------|-------|
| N    | ACGT       | 4     |
| R    | AG         | 2     |
| Y    | CT         | 2     |
| S    | GC         | 2     |
| W    | AT         | 2     |
| K    | GT         | 2     |
| M    | AC         | 2     |
| B    | CGT        | 3     |
| D    | AGT        | 3     |
| H    | ACT        | 3     |
| V    | ACG        | 3     |

#### Shift-And (Bitap) Algorithm

For patterns longer than 4 bases (up to 64 bases):

| Type              | Description                   |
|-------------------|-------------------------------|
| `ShiftAndMatcher` | Bit-parallel pattern matching |
| `ShiftAndPattern` | Pre-compiled pattern masks    |

| Method                                     | Description                   |
|--------------------------------------------|-------------------------------|
| `ShiftAndMatcher::new(pattern)`            | Compiles pattern for matching |
| `ShiftAndMatcher::scan_2bit(encoded, len)` | Scans 2-bit encoded data      |
| `ShiftAndMatcher::scan_4bit(encoded, len)` | Scans 4-bit encoded data      |
| `ShiftAndPattern::from_literal(pattern)`   | Compiles exact pattern        |
| `ShiftAndPattern::from_iupac(pattern)`     | Compiles with IUPAC support   |

---

### quality_encoder.rs - Quality Score Compression

High-performance FASTQ quality score encoding using SIMD with binning and run-length encoding.

#### Encoding Pipeline

```
ASCII Quality → Phred Conversion → Binning (4 levels) → RLE → Compressed
```

#### Binning Scheme

| Phred Range | Bin | Representative Q | Meaning          |
|-------------|-----|------------------|------------------|
| Q0-9        | 0   | Q6               | Very low quality |
| Q10-19      | 1   | Q15              | Low quality      |
| Q20-29      | 2   | Q25              | Medium quality   |
| Q30+        | 3   | Q37              | High quality     |

#### Phred Encoding Support

| Format   | ASCII Offset | Phred Range | Common Usage          |
|----------|--------------|-------------|-----------------------|
| Phred+33 | 33           | 0-41        | Illumina 1.8+, Sanger |
| Phred+64 | 64           | 0-40        | Illumina 1.3-1.7      |

#### RLE Packing Format

| Run Length | Format                             | Size    |
|------------|------------------------------------|---------|
| ≤63        | `[bin:2][length:6]`                | 1 byte  |
| >63        | `0xFF` + `[bin:8]` + `[length:16]` | 4 bytes |

#### Key Functions

| Function                                         | Description                 |
|--------------------------------------------------|-----------------------------|
| `encode_quality_scores(quality, encoding)`       | Compresses quality string   |
| `decode_quality_scores(encoded, encoding)`       | Decompresses to ASCII       |
| `encode_quality_into(quality, encoding, buffer)` | Zero-allocation encoding    |
| `decode_quality_into(encoded, encoding, buffer)` | Zero-allocation decoding    |
| `PhredEncoding::detect(quality)`                 | Auto-detects Phred encoding |

#### Compression Ratios

| Read Type          | Compression |
|--------------------|-------------|
| High-quality reads | 85-95%      |
| Mixed quality      | 70-85%      |
| Low-quality reads  | 60-75%      |

---

### serialization.rs - Binary Serialization

Efficient binary serialization for database BLOB storage in SQLite.

#### Wire Format (Version 1)

```
┌────────────────────────────────────────────────────────────────┐
│ Byte 0-1: Magic bytes (0x44, 0x4E = "DN")                      │
│ Byte 2: Format version (0x01)                                  │
│ Byte 3: Encoding type (0x00 = Clean2Bit, 0x01 = Dirty4Bit)    │
│ Byte 4: Flags (bit 0 = has_checksum)                          │
│ Byte 5: Reserved                                               │
│ Bytes 6-9: Original sequence length (little-endian u32)       │
│ Bytes 10-13: Encoded data length (little-endian u32)          │
│ Bytes 14+: Encoded sequence data                               │
│ [Optional] Last 4 bytes: CRC32 checksum                       │
└────────────────────────────────────────────────────────────────┘
```

#### Constants

| Constant            | Value          | Description            |
|---------------------|----------------|------------------------|
| `MAGIC_BYTES`       | `[0x44, 0x4E]` | "DN" identifier        |
| `FORMAT_VERSION`    | `1`            | Current format version |
| `HEADER_SIZE`       | `14`           | Bytes before data      |
| `FLAG_HAS_CHECKSUM` | `0x01`         | Checksum present flag  |
| `CHECKSUM_SIZE`     | `4`            | CRC32 size in bytes    |

#### Key Functions

| Function                          | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `to_bytes(encoded)`               | Serializes to bytes (no checksum)                |
| `from_bytes(bytes)`               | Deserializes from bytes                          |
| `to_blob(encoded, with_checksum)` | Serializes for database storage                  |
| `from_blob(blob)`                 | Deserializes with optional checksum verification |
| `crc32(data)`                     | Computes CRC32 checksum (IEEE polynomial)        |
| `crc32_update(current, data)`     | Incremental CRC32                                |

---

## Data Flow

### Encode Flow

```
┌─────────────────┐
│ ASCII Sequence  │ b"ACGTACGT" or b"ACNGTACGT"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ is_clean_       │ hybrid_encoder.rs
│ sequence()      │ SIMD purity check
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ 2-bit  │ │ 4-bit  │
│ encode │ │ encode │
│ (4:1)  │ │ (2:1)  │
└────┬───┘ └───┬────┘
     │         │
     └────┬────┘
          ▼
┌─────────────────┐
│ EncodedSequence │ encoding + data + original_len
└─────────────────┘
```

### Decode Flow

```
┌─────────────────┐
│ EncodedSequence │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check           │
│ encoding type   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ 2-bit  │ │ 4-bit  │
│ decode │ │ decode │
│ (SIMD) │ │ (SIMD) │
└────┬───┘ └───┬────┘
     │         │
     └────┬────┘
          ▼
┌─────────────────┐
│ Vec<u8>         │ ASCII DNA sequence
└─────────────────┘
```

### Pattern Search Flow

```
┌─────────────────┐
│ Pattern String  │ e.g., "ACGT" or "ACGN"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TetraLUT::      │ tetra_scanner.rs
│ from_iupac()    │ Build 256-entry LUT
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ scan_2bit()     │ NEON-accelerated
│ or scan_4bit()  │ 16 positions/iteration
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vec<usize>      │ Match positions
└─────────────────┘
```

---

## Memory Layout

### 4-bit Packed Format (2 nucleotides per byte)

```
Byte layout: [high_nibble:4][low_nibble:4]

Example: "ACGT" → [0x12, 0x84]
  Byte 0: A(0x1)<<4 | C(0x2) = 0x12
  Byte 1: G(0x8)<<4 | T(0x4) = 0x84

Compression: 2:1
```

### 2-bit Packed Format (4 nucleotides per byte)

```
Byte layout: [base0:2][base1:2][base2:2][base3:2]

Example: "ACGT" → [0x1B]
  A(00)<<6 | C(01)<<4 | G(10)<<2 | T(11) = 0b00011011 = 0x1B

Compression: 4:1
```

### Quality Score RLE Format

```
Short run (1-63 bases):
┌───────────────────┐
│ [bin:2][len:6]    │  1 byte
└───────────────────┘

Long run (64+ bases):
┌───────────────────────────────────────────┐
│ 0xFF │ [bin:8] │ [length:16 little-endian] │
└───────────────────────────────────────────┘  4 bytes
```

---

## Platform Support

### SIMD Capabilities

| Platform                | Instruction Set | Detection    | Usage             |
|-------------------------|-----------------|--------------|-------------------|
| aarch64 (Apple Silicon) | NEON            | Compile-time | All operations    |
| x86_64                  | SSSE3           | Runtime      | Encoding/decoding |
| x86_64                  | SSE2/SSE4.1     | Runtime      | Quality encoding  |
| Other                   | Scalar          | Default      | Fallback          |

### SIMD Processing Widths

| Operation          | aarch64 (NEON)      | x86_64 (SSE)        |
|--------------------|---------------------|---------------------|
| 4-bit encode       | 32 nucleotides/iter | 32 nucleotides/iter |
| 4-bit decode       | 32 nucleotides/iter | 32 nucleotides/iter |
| 2-bit scan         | 16 positions/iter   | Scalar              |
| Reverse complement | 16 bytes/iter       | 16 bytes/iter       |
| Quality binning    | 16 scores/iter      | 16 scores/iter      |

---

## Performance Considerations

1. **Static Lookup Tables** - Pre-computed 256-byte encode and 16-byte decode tables eliminate branch mispredictions (~15-20% faster than match statements)

2. **SIMD Processing** - Handles 32 nucleotides per iteration for improved instruction-level parallelism

3. **Prefetch Hints** - Prefetches data 128 bytes (2 cache lines) ahead to reduce cache misses on large sequences

4. **Direct Case Handling** - LUT handles case-insensitivity directly, avoiding O(n) uppercase conversion pass

5. **4-at-a-time Scalar** - Remainder processing uses optimized scalar path processing 4 nucleotides per iteration

6. **Aligned Memory** - LUTs use 64-byte alignment for optimal cache line access

7. **Zero-Allocation APIs** - `_into` variants avoid heap allocation for high-throughput pipelines

8. **Bifurcated Encoding** - 99%+ of genomic sequences use 4x compression instead of 2x

---

## Thread Safety

All public functions are thread-safe and can be safely called from multiple threads concurrently:

- No global or static mutable state
- Pure functions (output depends only on input)
- Thread-safe feature detection (atomic caching)

Suitable for use with `rayon`, `std::thread`, or any multi-threading framework.

---

## Dependencies

| Crate       | Category | Purpose                   |
|-------------|----------|---------------------------|
| (none)      | Runtime  | Zero runtime dependencies |
| `criterion` | Dev      | Benchmarking              |
| `proptest`  | Dev      | Property-based testing    |
| `chrono`    | Dev      | Timestamp formatting      |
| `seq_io`    | Dev      | FASTQ parsing in examples |

---

## Build Configuration

```toml
[profile.release]
opt-level = 3     # Maximum optimization
lto = true        # Link-time optimization
codegen-units = 1 # Single codegen unit for better optimization
strip = true      # Strip symbols for smaller binaries

[profile.bench]
lto = true
codegen-units = 1
opt-level = 3
```

### Recommended Compiler Flags (Apple Silicon)

```toml
# .cargo/config.toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=native"]
```

---

## Error Types

### BufferError (dna_simd_encoder)

```rust
pub enum BufferError {
    BufferTooSmall { needed: usize, actual: usize },
}
```

### HybridEncoderError (hybrid_encoder)

```rust
pub enum HybridEncoderError {
    BufferTooSmall { needed: usize, actual: usize },
    InvalidEncodingType(i32),
    InvalidEncodedData(String),
    InvalidBase(char),
}
```

### TetraLUTError (tetra_scanner)

```rust
pub enum TetraLUTError {
    InvalidPatternLength { expected: usize, actual: usize },
    InvalidBase { pattern: String },
}
```

### QualityError (quality_encoder)

```rust
pub enum QualityError {
    BufferTooSmall { needed: usize, actual: usize },
    InvalidEncodedData,
    InvalidQualityScore { score: u8, encoding: PhredEncoding },
}
```

### SerializationError (serialization)

```rust
pub enum SerializationError {
    InvalidMagic { expected: [u8; 2], found: [u8; 2] },
    UnsupportedVersion { version: u8, max_supported: u8 },
    InvalidEncodingType(u8),
    BufferTooSmall { needed: usize, actual: usize },
    ChecksumMismatch { expected: u32, computed: u32 },
    DataLengthMismatch { header: usize, actual: usize },
    InconsistentLength { original_len: usize, encoded_len: usize, encoding: EncodingType },
}
```
