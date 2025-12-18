// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

// Import SIMD-accelerated functions from the main crate
use simdna::dna_simd_encoder::{
    decode_dna_prefer_simd, encode_dna_prefer_simd, reverse_complement, reverse_complement_encoded,
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

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_roundtrip,
    bench_reverse_complement
);
criterion_main!(benches);
