// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

// Import SIMD-accelerated functions from the main crate
use simdna::dna_simd_encoder::{
    decode_dna_into, decode_dna_prefer_simd, encode_dna_into, encode_dna_prefer_simd,
    required_decoded_len, required_encoded_len, reverse_complement, reverse_complement_encoded,
    reverse_complement_encoded_into, reverse_complement_into,
};

// Import quality encoder functions
use simdna::quality_encoder::{
    PhredEncoding, bin_quality, decode_quality_scores, decode_quality_scores_into,
    encode_quality_scores, encode_quality_scores_into, quality_stats,
};

/// Package version from Cargo.toml
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Print benchmark header with version and timestamp
fn print_benchmark_header() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let now = chrono::Utc::now();
        eprintln!("\n╔════════════════════════════════════════════════════════════╗");
        eprintln!(
            "║ simdna benchmark v{}                                     ║",
            VERSION
        );
        eprintln!(
            "║ Run date: {}                              ║",
            now.format("%Y-%m-%d %H:%M:%S UTC")
        );
        eprintln!("║ Platform: {:<49} ║", std::env::consts::ARCH);
        eprintln!("╚════════════════════════════════════════════════════════════╝\n");
    });
}

// ============================================================================
// Scalar 4-bit Encoding Implementation (for comparison)
// ============================================================================
// Uses the NEW bit-rotation-compatible encoding scheme:
// - Gap=0x0, A=0x1, C=0x2, M=0x3, T=0x4, W=0x5, Y=0x6, H=0x7
// - G=0x8, R=0x9, S=0xA, V=0xB, K=0xC, D=0xD, B=0xE, N=0xF
// Complement is computed via: ((bits << 2) | (bits >> 2)) & 0xF

/// Encode DNA sequence to 4-bit representation (scalar)
/// Uses the bit-rotation-compatible encoding scheme.
/// 2 nucleotides packed per byte (high nibble first)
pub fn encode_dna_4bit_scalar(sequence: &[u8]) -> Vec<u8> {
    let output_len = sequence.len().div_ceil(2);
    let mut output = vec![0u8; output_len];

    for (i, chunk) in sequence.chunks(2).enumerate() {
        let high = char_to_4bit(chunk[0]);
        let low = if chunk.len() > 1 {
            char_to_4bit(chunk[1])
        } else {
            0x0 // Pad with gap
        };
        output[i] = (high << 4) | low;
    }

    output
}

/// Decode 4-bit DNA representation back to ASCII (scalar)
pub fn decode_dna_4bit_scalar(encoded: &[u8], length: usize) -> Vec<u8> {
    let mut output = vec![0u8; length];
    let mut out_idx = 0;

    for &byte in encoded {
        if out_idx >= length {
            break;
        }
        output[out_idx] = four_bit_to_char((byte >> 4) & 0x0F);
        out_idx += 1;

        if out_idx >= length {
            break;
        }
        output[out_idx] = four_bit_to_char(byte & 0x0F);
        out_idx += 1;
    }

    output
}

/// Convert ASCII character to 4-bit encoding (bit-rotation-compatible scheme)
#[inline]
fn char_to_4bit(c: u8) -> u8 {
    match c {
        b'-' | b'.' => 0x0,               // Gap (self-complement)
        b'A' | b'a' => 0x1,               // A ↔ T (0x4)
        b'C' | b'c' => 0x2,               // C ↔ G (0x8)
        b'M' | b'm' => 0x3,               // M (A|C) ↔ K (0xC)
        b'T' | b't' | b'U' | b'u' => 0x4, // T ↔ A (0x1)
        b'W' | b'w' => 0x5,               // W (A|T) (self-complement)
        b'Y' | b'y' => 0x6,               // Y (C|T) ↔ R (0x9)
        b'H' | b'h' => 0x7,               // H (A|C|T) ↔ D (0xD)
        b'G' | b'g' => 0x8,               // G ↔ C (0x2)
        b'R' | b'r' => 0x9,               // R (A|G) ↔ Y (0x6)
        b'S' | b's' => 0xA,               // S (G|C) (self-complement)
        b'V' | b'v' => 0xB,               // V (A|C|G) ↔ B (0xE)
        b'K' | b'k' => 0xC,               // K (G|T) ↔ M (0x3)
        b'D' | b'd' => 0xD,               // D (A|G|T) ↔ H (0x7)
        b'B' | b'b' => 0xE,               // B (C|G|T) ↔ V (0xB)
        b'N' | b'n' => 0xF,               // N (any) (self-complement)
        _ => 0x0,                         // Unknown → gap
    }
}

/// Convert 4-bit encoding to ASCII character (bit-rotation-compatible scheme)
#[inline]
fn four_bit_to_char(bits: u8) -> u8 {
    // Decode table indexed by 4-bit encoding value
    const DECODE_TABLE: [u8; 16] = [
        b'-', // 0x0 = Gap
        b'A', // 0x1 = Adenine
        b'C', // 0x2 = Cytosine
        b'M', // 0x3 = A or C (amino)
        b'T', // 0x4 = Thymine
        b'W', // 0x5 = A or T (weak)
        b'Y', // 0x6 = C or T (pyrimidine)
        b'H', // 0x7 = A, C, or T (not G)
        b'G', // 0x8 = Guanine
        b'R', // 0x9 = A or G (purine)
        b'S', // 0xA = G or C (strong)
        b'V', // 0xB = A, C, or G (not T)
        b'K', // 0xC = G or T (keto)
        b'D', // 0xD = A, G, or T (not C)
        b'B', // 0xE = C, G, or T (not A)
        b'N', // 0xF = Any base
    ];
    DECODE_TABLE[(bits & 0x0F) as usize]
}

// ============================================================================
// Scalar 2-bit Encoding Implementation (for comparison)
// ============================================================================
// Note: 2-bit encoding only supports A, C, G, T (no IUPAC ambiguity codes)
// This is kept for performance comparison purposes only.

/// Encode DNA sequence to 2-bit representation (pure scalar, no SIMD)
/// Each nucleotide: A=00, C=01, G=10, T=11
/// 4 nucleotides packed per byte
pub fn encode_dna_2bit_scalar(sequence: &[u8]) -> Vec<u8> {
    let padded_len = (sequence.len() + 3) & !3; // Pad to multiple of 4
    let mut padded = sequence.to_vec();
    padded.resize(padded_len, b'A');

    let mut output = vec![0u8; padded_len / 4];

    for (i, chunk) in padded.chunks_exact(4).enumerate() {
        let packed = (char_to_2bit(chunk[0]) << 6)
            | (char_to_2bit(chunk[1]) << 4)
            | (char_to_2bit(chunk[2]) << 2)
            | char_to_2bit(chunk[3]);
        output[i] = packed;
    }

    output
}

/// Decode 2-bit DNA representation back to ASCII (pure scalar, no SIMD)
pub fn decode_dna_2bit_scalar(encoded: &[u8], length: usize) -> Vec<u8> {
    let mut output = vec![0u8; length];
    let mut out_idx = 0;

    for &byte in encoded {
        if out_idx >= length {
            break;
        }
        output[out_idx] = two_bit_to_char((byte >> 6) & 0x3);
        out_idx += 1;

        if out_idx >= length {
            break;
        }
        output[out_idx] = two_bit_to_char((byte >> 4) & 0x3);
        out_idx += 1;

        if out_idx >= length {
            break;
        }
        output[out_idx] = two_bit_to_char((byte >> 2) & 0x3);
        out_idx += 1;

        if out_idx >= length {
            break;
        }
        output[out_idx] = two_bit_to_char(byte & 0x3);
        out_idx += 1;
    }

    output
}

#[inline]
fn char_to_2bit(c: u8) -> u8 {
    match c {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 0,
    }
}

#[inline]
fn two_bit_to_char(bits: u8) -> u8 {
    match bits & 0x3 {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        _ => unreachable!(),
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

fn generate_dna_sequence(len: usize) -> Vec<u8> {
    (0..len).map(|i| b"ACGT"[i % 4]).collect()
}

fn bench_encode(c: &mut Criterion) {
    print_benchmark_header();
    let mut group = c.benchmark_group("encode");

    // Test various sequence lengths to measure SIMD/scalar fallback interaction:
    // - 15/16/17: around SIMD block boundary (16 nucleotides per SIMD iteration)
    // - 32/33: two full SIMD blocks vs two + 1 scalar
    // - Powers of 2 and adjacent values to test alignment effects
    // - Larger sizes up to 10000 for throughput measurement
    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let sequence = generate_dna_sequence(size);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("simd_4bit", size), &sequence, |b, seq| {
            b.iter(|| encode_dna_prefer_simd(black_box(seq)));
        });

        group.bench_with_input(
            BenchmarkId::new("scalar_2bit", size),
            &sequence,
            |b, seq| {
                b.iter(|| encode_dna_2bit_scalar(black_box(seq)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_4bit", size),
            &sequence,
            |b, seq| {
                b.iter(|| encode_dna_4bit_scalar(black_box(seq)));
            },
        );
    }

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    // Test various sequence lengths to measure SIMD/scalar fallback interaction:
    // - 15/16/17: around SIMD block boundary (16 nucleotides per SIMD iteration)
    // - 32/33: two full SIMD blocks vs two + 1 scalar
    // - Powers of 2 and adjacent values to test alignment effects
    // - Larger sizes up to 10000 for throughput measurement
    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let sequence = generate_dna_sequence(size);

        // Pre-encode for decode benchmarks
        let encoded_simd = encode_dna_prefer_simd(&sequence);
        let encoded_scalar_2bit = encode_dna_2bit_scalar(&sequence);
        let encoded_scalar_4bit = encode_dna_4bit_scalar(&sequence);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("simd_4bit", size),
            &(&encoded_simd, size),
            |b, (encoded, len)| {
                b.iter(|| decode_dna_prefer_simd(black_box(encoded), black_box(*len)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_2bit", size),
            &(&encoded_scalar_2bit, size),
            |b, (encoded, len)| {
                b.iter(|| decode_dna_2bit_scalar(black_box(encoded), black_box(*len)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_4bit", size),
            &(&encoded_scalar_4bit, size),
            |b, (encoded, len)| {
                b.iter(|| decode_dna_4bit_scalar(black_box(encoded), black_box(*len)));
            },
        );
    }

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    // Test various sequence lengths to measure SIMD/scalar fallback interaction:
    // - 15/16/17: around SIMD block boundary (16 nucleotides per SIMD iteration)
    // - 32/33: two full SIMD blocks vs two + 1 scalar
    // - Powers of 2 and adjacent values to test alignment effects
    // - Larger sizes up to 10000 for throughput measurement
    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let sequence = generate_dna_sequence(size);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("simd_4bit", size), &sequence, |b, seq| {
            b.iter(|| {
                let encoded = encode_dna_prefer_simd(black_box(seq));
                decode_dna_prefer_simd(black_box(&encoded), seq.len())
            });
        });

        group.bench_with_input(
            BenchmarkId::new("scalar_2bit", size),
            &sequence,
            |b, seq| {
                b.iter(|| {
                    let encoded = encode_dna_2bit_scalar(black_box(seq));
                    decode_dna_2bit_scalar(black_box(&encoded), seq.len())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_4bit", size),
            &sequence,
            |b, seq| {
                b.iter(|| {
                    let encoded = encode_dna_4bit_scalar(black_box(seq));
                    decode_dna_4bit_scalar(black_box(&encoded), seq.len())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Scalar Reverse Complement Implementation (for comparison)
// ============================================================================

/// Compute the complement of a single nucleotide (scalar, lookup table)
#[inline]
fn complement_char_scalar(c: u8) -> u8 {
    match c {
        b'A' | b'a' => b'T',
        b'T' | b't' | b'U' | b'u' => b'A',
        b'C' | b'c' => b'G',
        b'G' | b'g' => b'C',
        b'R' | b'r' => b'Y',
        b'Y' | b'y' => b'R',
        b'S' | b's' => b'S',
        b'W' | b'w' => b'W',
        b'K' | b'k' => b'M',
        b'M' | b'm' => b'K',
        b'B' | b'b' => b'V',
        b'D' | b'd' => b'H',
        b'H' | b'h' => b'D',
        b'V' | b'v' => b'B',
        b'N' | b'n' => b'N',
        b'-' | b'.' => b'-',
        _ => b'-',
    }
}

/// Scalar reverse complement implementation using lookup table
pub fn reverse_complement_scalar(sequence: &[u8]) -> Vec<u8> {
    sequence
        .iter()
        .rev()
        .map(|&c| complement_char_scalar(c))
        .collect()
}

fn bench_reverse_complement(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_complement");

    // Test various sequence lengths to measure SIMD/scalar fallback interaction
    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let sequence = generate_dna_sequence(size);

        // Pre-encode for encoded reverse complement benchmark
        let encoded = encode_dna_prefer_simd(&sequence);

        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark SIMD reverse complement (high-level API)
        group.bench_with_input(
            BenchmarkId::new("simd_high_level", size),
            &sequence,
            |b, seq| {
                b.iter(|| reverse_complement(black_box(seq)));
            },
        );

        // Benchmark SIMD reverse complement on encoded data
        group.bench_with_input(
            BenchmarkId::new("simd_encoded", size),
            &(&encoded, size),
            |b, (enc, len)| {
                b.iter(|| reverse_complement_encoded(black_box(enc), black_box(*len)));
            },
        );

        // Benchmark scalar reverse complement for comparison
        group.bench_with_input(BenchmarkId::new("scalar", size), &sequence, |b, seq| {
            b.iter(|| reverse_complement_scalar(black_box(seq)));
        });
    }

    group.finish();
}

// ============================================================================
// Zero-Allocation _into Variants Benchmarks
// ============================================================================

fn bench_encode_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_into");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let sequence = generate_dna_sequence(size);
        let needed = required_encoded_len(size);

        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark allocating version (baseline)
        group.bench_with_input(BenchmarkId::new("allocating", size), &sequence, |b, seq| {
            b.iter(|| encode_dna_prefer_simd(black_box(seq)));
        });

        // Benchmark _into version with pre-allocated buffer
        group.bench_with_input(
            BenchmarkId::new("into_preallocated", size),
            &sequence,
            |b, seq| {
                let mut output = vec![0u8; needed];
                b.iter(|| encode_dna_into(black_box(seq), black_box(&mut output)).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_decode_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_into");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let sequence = generate_dna_sequence(size);
        let encoded = encode_dna_prefer_simd(&sequence);
        let needed = required_decoded_len(size);

        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark allocating version (baseline)
        group.bench_with_input(
            BenchmarkId::new("allocating", size),
            &(&encoded, size),
            |b, (enc, len)| {
                b.iter(|| decode_dna_prefer_simd(black_box(enc), black_box(*len)));
            },
        );

        // Benchmark _into version with pre-allocated buffer
        group.bench_with_input(
            BenchmarkId::new("into_preallocated", size),
            &(&encoded, size),
            |b, (enc, len)| {
                let mut output = vec![0u8; needed];
                b.iter(|| {
                    decode_dna_into(black_box(enc), black_box(*len), black_box(&mut output))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_reverse_complement_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_complement_into");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let sequence = generate_dna_sequence(size);

        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark allocating version (baseline)
        group.bench_with_input(BenchmarkId::new("allocating", size), &sequence, |b, seq| {
            b.iter(|| reverse_complement(black_box(seq)));
        });

        // Benchmark _into version with pre-allocated buffer
        group.bench_with_input(
            BenchmarkId::new("into_preallocated", size),
            &sequence,
            |b, seq| {
                let mut output = vec![0u8; size];
                b.iter(|| reverse_complement_into(black_box(seq), black_box(&mut output)).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_reverse_complement_encoded_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_complement_encoded_into");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let sequence = generate_dna_sequence(size);
        let encoded = encode_dna_prefer_simd(&sequence);

        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark allocating version (baseline)
        group.bench_with_input(
            BenchmarkId::new("allocating", size),
            &(&encoded, size),
            |b, (enc, len)| {
                b.iter(|| reverse_complement_encoded(black_box(enc), black_box(*len)));
            },
        );

        // Benchmark _into version with pre-allocated buffer
        group.bench_with_input(
            BenchmarkId::new("into_preallocated", size),
            &(&encoded, size),
            |b, (enc, len)| {
                let mut output = vec![0u8; enc.len()];
                b.iter(|| {
                    reverse_complement_encoded_into(
                        black_box(enc),
                        black_box(*len),
                        black_box(&mut output),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Quality Score Encoding Benchmarks
// ============================================================================

/// Generate a realistic Illumina-like quality string.
/// Illumina reads typically have high quality in the middle, lower at ends.
fn generate_quality_string(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            // Simulate Illumina quality profile: ramp up, plateau, ramp down
            let pos_fraction = i as f64 / len as f64;
            let q = if pos_fraction < 0.1 {
                // First 10%: ramping up from Q20 to Q35
                20.0 + (pos_fraction / 0.1) * 15.0
            } else if pos_fraction < 0.8 {
                // Middle 70%: high quality Q35-Q40
                35.0 + ((pos_fraction - 0.1) / 0.7) * 5.0
            } else {
                // Last 20%: declining from Q40 to Q20
                40.0 - ((pos_fraction - 0.8) / 0.2) * 20.0
            };
            q as u8 + 33 // Convert to Phred+33 ASCII
        })
        .collect()
}

/// Generate uniform high-quality string (best-case for RLE compression)
fn generate_uniform_quality_string(len: usize) -> Vec<u8> {
    vec![b'I'; len] // Q40, all high quality
}

/// Scalar quality binning implementation (for comparison)
fn bin_quality_scalar(quality: &[u8], binned: &mut [u8], offset: u8) {
    for (i, &q) in quality.iter().enumerate() {
        let phred = q.saturating_sub(offset);
        binned[i] = match phred {
            0..=9 => 0,
            10..=19 => 1,
            20..=29 => 2,
            _ => 3,
        };
    }
}

/// Scalar RLE encoding (for comparison)
fn encode_rle_scalar(binned: &[u8]) -> Vec<u8> {
    if binned.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(binned.len() / 4);
    let mut current_bin = binned[0];
    let mut run_length = 1usize;

    for &bin in &binned[1..] {
        if bin == current_bin {
            run_length += 1;
        } else {
            // Emit run
            emit_run_scalar(&mut output, current_bin, run_length);
            current_bin = bin;
            run_length = 1;
        }
    }
    // Emit final run
    emit_run_scalar(&mut output, current_bin, run_length);

    output
}

#[inline]
fn emit_run_scalar(output: &mut Vec<u8>, bin: u8, length: usize) {
    if length <= 63 {
        output.push((bin << 6) | (length as u8));
    } else {
        output.push(0xFF);
        output.push(bin);
        output.push((length >> 8) as u8);
        output.push(length as u8);
    }
}

/// Full scalar encode pipeline for comparison
fn encode_quality_scores_scalar(quality: &[u8]) -> Vec<u8> {
    let mut binned = vec![0u8; quality.len()];
    bin_quality_scalar(quality, &mut binned, 33);
    encode_rle_scalar(&binned)
}

fn bench_quality_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_encode");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let quality = generate_quality_string(size);

        group.throughput(Throughput::Bytes(size as u64));

        // SIMD-accelerated encoding
        group.bench_with_input(BenchmarkId::new("simd", size), &quality, |b, q| {
            b.iter(|| encode_quality_scores(black_box(q), PhredEncoding::Phred33));
        });

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", size), &quality, |b, q| {
            b.iter(|| encode_quality_scores_scalar(black_box(q)));
        });
    }

    group.finish();
}

fn bench_quality_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_decode");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let quality = generate_quality_string(size);
        let encoded = encode_quality_scores(&quality, PhredEncoding::Phred33);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), &encoded, |b, enc| {
            b.iter(|| decode_quality_scores(black_box(enc), PhredEncoding::Phred33));
        });
    }

    group.finish();
}

fn bench_quality_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_roundtrip");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let quality = generate_quality_string(size);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), &quality, |b, q| {
            b.iter(|| {
                let encoded = encode_quality_scores(black_box(q), PhredEncoding::Phred33);
                decode_quality_scores(black_box(&encoded), PhredEncoding::Phred33)
            });
        });
    }

    group.finish();
}

fn bench_quality_binning(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_binning");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let quality = generate_quality_string(size);
        let mut binned_simd = vec![0u8; size];
        let mut binned_scalar = vec![0u8; size];

        group.throughput(Throughput::Bytes(size as u64));

        // SIMD binning
        group.bench_with_input(BenchmarkId::new("simd", size), &quality, |b, q| {
            b.iter(|| {
                bin_quality(
                    black_box(q),
                    black_box(&mut binned_simd),
                    PhredEncoding::Phred33,
                )
            });
        });

        // Scalar binning
        group.bench_with_input(BenchmarkId::new("scalar", size), &quality, |b, q| {
            b.iter(|| bin_quality_scalar(black_box(q), black_box(&mut binned_scalar), 33));
        });
    }

    group.finish();
}

fn bench_quality_encode_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_encode_into");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let quality = generate_quality_string(size);
        // Worst case: 4 bytes per score (if every score is a long run)
        // Realistic: ~size/4 for good compression
        let max_output_size = size * 4;

        group.throughput(Throughput::Bytes(size as u64));

        // Allocating version (baseline)
        group.bench_with_input(BenchmarkId::new("allocating", size), &quality, |b, q| {
            b.iter(|| encode_quality_scores(black_box(q), PhredEncoding::Phred33));
        });

        // _into version with pre-allocated buffer
        group.bench_with_input(
            BenchmarkId::new("into_preallocated", size),
            &quality,
            |b, q| {
                let mut output = vec![0u8; max_output_size];
                b.iter(|| {
                    encode_quality_scores_into(
                        black_box(q),
                        PhredEncoding::Phred33,
                        black_box(&mut output),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_quality_decode_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_decode_into");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let quality = generate_quality_string(size);
        let encoded = encode_quality_scores(&quality, PhredEncoding::Phred33);

        group.throughput(Throughput::Bytes(size as u64));

        // Allocating version (baseline)
        group.bench_with_input(BenchmarkId::new("allocating", size), &encoded, |b, enc| {
            b.iter(|| decode_quality_scores(black_box(enc), PhredEncoding::Phred33));
        });

        // _into version with pre-allocated buffer
        group.bench_with_input(
            BenchmarkId::new("into_preallocated", size),
            &encoded,
            |b, enc| {
                let mut output = vec![0u8; size];
                b.iter(|| {
                    decode_quality_scores_into(
                        black_box(enc),
                        PhredEncoding::Phred33,
                        black_box(&mut output),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_quality_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_stats");

    for size in [
        15, 16, 17, 32, 33, 63, 64, 127, 128, 255, 256, 512, 1023, 1024, 2048, 4095, 4096, 8192,
        9999, 10000,
    ] {
        let quality = generate_quality_string(size);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("compute", size), &quality, |b, q| {
            b.iter(|| quality_stats(black_box(q), PhredEncoding::Phred33));
        });
    }

    group.finish();
}

fn bench_quality_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_compression");

    // Test different quality profiles
    for size in [150, 250, 500, 1000, 10000] {
        let realistic_quality = generate_quality_string(size);
        let uniform_quality = generate_uniform_quality_string(size);

        group.throughput(Throughput::Bytes(size as u64));

        // Realistic Illumina profile
        group.bench_with_input(
            BenchmarkId::new("realistic", size),
            &realistic_quality,
            |b, q| {
                b.iter(|| encode_quality_scores(black_box(q), PhredEncoding::Phred33));
            },
        );

        // Uniform high quality (best case)
        group.bench_with_input(
            BenchmarkId::new("uniform_high", size),
            &uniform_quality,
            |b, q| {
                b.iter(|| encode_quality_scores(black_box(q), PhredEncoding::Phred33));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_roundtrip,
    bench_reverse_complement,
    bench_encode_into,
    bench_decode_into,
    bench_reverse_complement_into,
    bench_reverse_complement_encoded_into,
    bench_quality_encode,
    bench_quality_decode,
    bench_quality_roundtrip,
    bench_quality_binning,
    bench_quality_encode_into,
    bench_quality_decode_into,
    bench_quality_stats,
    bench_quality_compression_ratio
);
criterion_main!(benches);
