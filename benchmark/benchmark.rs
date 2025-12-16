// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

// Import SIMD-accelerated functions from the main crate
use simdna::dna_simd_encoder::{decode_dna_prefer_simd, encode_dna_prefer_simd};

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

/// Encode DNA sequence to 4-bit representation (scalar)
/// Each nucleotide/IUPAC code uses 4 bits:
/// A=0, C=1, G=2, T=3, R=4, Y=5, S=6, W=7, K=8, M=9, B=A, D=B, H=C, V=D, N=E, -=F
/// 2 nucleotides packed per byte
pub fn encode_dna_4bit_scalar(sequence: &[u8]) -> Vec<u8> {
    let padded_len = (sequence.len() + 1) & !1; // Pad to multiple of 2
    let mut output = vec![0u8; padded_len / 2];

    for (i, chunk) in sequence.chunks(2).enumerate() {
        let high = char_to_4bit(chunk[0]);
        let low = if chunk.len() > 1 {
            char_to_4bit(chunk[1])
        } else {
            0 // Pad with A
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

#[inline]
fn char_to_4bit(c: u8) -> u8 {
    match c {
        b'A' | b'a' => 0x0,
        b'C' | b'c' => 0x1,
        b'G' | b'g' => 0x2,
        b'T' | b't' | b'U' | b'u' => 0x3,
        b'R' | b'r' => 0x4,
        b'Y' | b'y' => 0x5,
        b'S' | b's' => 0x6,
        b'W' | b'w' => 0x7,
        b'K' | b'k' => 0x8,
        b'M' | b'm' => 0x9,
        b'B' | b'b' => 0xA,
        b'D' | b'd' => 0xB,
        b'H' | b'h' => 0xC,
        b'V' | b'v' => 0xD,
        b'N' | b'n' => 0xE,
        b'-' | b'.' => 0xF,
        _ => 0xF, // Default to gap for invalid characters
    }
}

#[inline]
fn four_bit_to_char(bits: u8) -> u8 {
    const DECODE_TABLE: [u8; 16] = [
        b'A', b'C', b'G', b'T', b'R', b'Y', b'S', b'W', b'K', b'M', b'B', b'D', b'H', b'V', b'N',
        b'-',
    ];
    DECODE_TABLE[(bits & 0x0F) as usize]
}

// ============================================================================
// Scalar 2-bit Encoding Implementation (for comparison)
// ============================================================================

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

criterion_group!(benches, bench_encode, bench_decode, bench_roundtrip);
criterion_main!(benches);
