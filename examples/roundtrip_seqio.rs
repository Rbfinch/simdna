// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! FASTQ roundtrip example using seq_io crate for parsing.
//!
//! This example demonstrates parsing a FASTQ file with seq_io and performing
//! roundtrip encoding/decoding tests on each record using simdna's encoding functions.
//!
//! Run with: `cargo run --example roundtrip_seqio`

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::{Duration, Instant};

use seq_io::fastq::{Reader, Record};

use simdna::dna_simd_encoder::{
    decode_dna_into, decode_dna_prefer_simd, encode_dna_into, encode_dna_prefer_simd,
    required_encoded_len, reverse_complement, reverse_complement_encoded,
    reverse_complement_encoded_into, reverse_complement_into,
};

use simdna::quality_encoder::{
    PhredEncoding, decode_quality_scores, decode_quality_scores_into, encode_quality_scores,
    encode_quality_scores_into,
};

/// Statistics collected during the roundtrip tests.
#[derive(Default)]
struct RoundtripStats {
    records_processed: usize,
    total_sequence_bytes: usize,
    total_quality_bytes: usize,
    total_encoded_dna_bytes: usize,
    total_encoded_quality_bytes: usize,

    // Timing for allocating variants
    dna_encode_allocating: Duration,
    dna_decode_allocating: Duration,
    revcomp_allocating: Duration,
    revcomp_encoded_allocating: Duration,
    quality_encode_allocating: Duration,
    quality_decode_allocating: Duration,

    // Timing for zero-allocation (_into) variants
    dna_encode_into: Duration,
    dna_decode_into: Duration,
    revcomp_into: Duration,
    revcomp_encoded_into: Duration,
    quality_encode_into: Duration,
    quality_decode_into: Duration,
}

impl RoundtripStats {
    fn print_summary(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║              Roundtrip Performance Summary                     ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        println!("Records processed:       {:>10}", self.records_processed);
        println!(
            "Total sequence bytes:    {:>10} ({:.2} KB)",
            self.total_sequence_bytes,
            self.total_sequence_bytes as f64 / 1024.0
        );
        println!(
            "Total quality bytes:     {:>10} ({:.2} KB)",
            self.total_quality_bytes,
            self.total_quality_bytes as f64 / 1024.0
        );
        println!(
            "Encoded DNA bytes:       {:>10} ({:.1}x compression)",
            self.total_encoded_dna_bytes,
            self.total_sequence_bytes as f64 / self.total_encoded_dna_bytes.max(1) as f64
        );
        println!(
            "Encoded quality bytes:   {:>10} ({:.1}x compression)",
            self.total_encoded_quality_bytes,
            self.total_quality_bytes as f64 / self.total_encoded_quality_bytes.max(1) as f64
        );

        println!("\n─── Allocating Variants ───");
        self.print_timing(
            "DNA encode",
            self.dna_encode_allocating,
            self.total_sequence_bytes,
        );
        self.print_timing(
            "DNA decode",
            self.dna_decode_allocating,
            self.total_sequence_bytes,
        );
        self.print_timing(
            "Reverse complement",
            self.revcomp_allocating,
            self.total_sequence_bytes,
        );
        self.print_timing(
            "Revcomp encoded",
            self.revcomp_encoded_allocating,
            self.total_encoded_dna_bytes,
        );
        self.print_timing(
            "Quality encode",
            self.quality_encode_allocating,
            self.total_quality_bytes,
        );
        self.print_timing(
            "Quality decode",
            self.quality_decode_allocating,
            self.total_quality_bytes,
        );

        println!("\n─── Zero-Allocation Variants (_into) ───");
        self.print_timing(
            "DNA encode_into",
            self.dna_encode_into,
            self.total_sequence_bytes,
        );
        self.print_timing(
            "DNA decode_into",
            self.dna_decode_into,
            self.total_sequence_bytes,
        );
        self.print_timing("Revcomp into", self.revcomp_into, self.total_sequence_bytes);
        self.print_timing(
            "Revcomp encoded_into",
            self.revcomp_encoded_into,
            self.total_encoded_dna_bytes,
        );
        self.print_timing(
            "Quality encode_into",
            self.quality_encode_into,
            self.total_quality_bytes,
        );
        self.print_timing(
            "Quality decode_into",
            self.quality_decode_into,
            self.total_quality_bytes,
        );

        println!("\n✓ All roundtrip assertions passed!");
    }

    fn print_timing(&self, name: &str, duration: Duration, bytes: usize) {
        let secs = duration.as_secs_f64();
        let throughput = if secs > 0.0 {
            (bytes as f64 / secs) / (1024.0 * 1024.0 * 1024.0)
        } else {
            0.0
        };
        println!(
            "  {:<22} {:>10.3} ms  ({:>6.2} GiB/s)",
            name,
            duration.as_secs_f64() * 1000.0,
            throughput
        );
    }
}

/// Test DNA encoding/decoding roundtrip for a single sequence.
fn test_dna_roundtrip(
    seq: &[u8],
    stats: &mut RoundtripStats,
    encode_buffer: &mut Vec<u8>,
    decode_buffer: &mut Vec<u8>,
) {
    let seq_len = seq.len();

    // --- Allocating variants ---

    // Encode (allocating)
    let start = Instant::now();
    let encoded = encode_dna_prefer_simd(seq);
    stats.dna_encode_allocating += start.elapsed();

    stats.total_encoded_dna_bytes += encoded.len();

    // Decode (allocating)
    let start = Instant::now();
    let decoded = decode_dna_prefer_simd(&encoded, seq_len);
    stats.dna_decode_allocating += start.elapsed();

    // Assert roundtrip - note: lowercase converts to uppercase
    for (i, (&orig, &dec)) in seq.iter().zip(decoded.iter()).enumerate() {
        let orig_upper = orig.to_ascii_uppercase();
        assert_eq!(
            orig_upper, dec,
            "DNA roundtrip mismatch at position {}: expected '{}', got '{}'",
            i, orig_upper as char, dec as char
        );
    }

    // --- Zero-allocation variants ---

    let encoded_len = required_encoded_len(seq_len);
    encode_buffer.resize(encoded_len, 0);
    decode_buffer.resize(seq_len, 0); // Decode buffer needs original sequence length

    // Encode into preallocated buffer
    let start = Instant::now();
    let bytes_written = encode_dna_into(seq, encode_buffer).expect("encode buffer too small");
    stats.dna_encode_into += start.elapsed();
    assert_eq!(bytes_written, encoded_len);

    // Decode into preallocated buffer
    let start = Instant::now();
    let decoded_len = decode_dna_into(&encode_buffer[..bytes_written], seq_len, decode_buffer)
        .expect("decode buffer too small");
    stats.dna_decode_into += start.elapsed();
    assert_eq!(decoded_len, seq_len);

    // Assert zero-allocation roundtrip
    for (i, (&orig, &dec)) in seq.iter().zip(decode_buffer[..seq_len].iter()).enumerate() {
        let orig_upper = orig.to_ascii_uppercase();
        assert_eq!(
            orig_upper, dec,
            "DNA into roundtrip mismatch at position {}: expected '{}', got '{}'",
            i, orig_upper as char, dec as char
        );
    }
}

/// Test reverse complement roundtrip for a single sequence.
fn test_revcomp_roundtrip(
    seq: &[u8],
    stats: &mut RoundtripStats,
    revcomp_buffer: &mut Vec<u8>,
    encode_buffer: &mut Vec<u8>,
    encoded_revcomp_buffer: &mut Vec<u8>,
) {
    let seq_len = seq.len();

    // --- Allocating variant (on raw sequence) ---
    let start = Instant::now();
    let revcomp = reverse_complement(seq);
    stats.revcomp_allocating += start.elapsed();

    // Double reverse complement should give back original
    let revcomp_revcomp = reverse_complement(&revcomp);
    for (i, (&orig, &rr)) in seq.iter().zip(revcomp_revcomp.iter()).enumerate() {
        let orig_upper = orig.to_ascii_uppercase();
        assert_eq!(
            orig_upper, rr,
            "Revcomp double roundtrip mismatch at position {}: expected '{}', got '{}'",
            i, orig_upper as char, rr as char
        );
    }

    // --- Zero-allocation variant ---
    revcomp_buffer.resize(seq_len, 0);
    let start = Instant::now();
    reverse_complement_into(seq, revcomp_buffer).expect("revcomp buffer too small");
    stats.revcomp_into += start.elapsed();

    // Verify matches allocating version
    assert_eq!(&revcomp, revcomp_buffer, "revcomp_into mismatch");

    // --- Encoded reverse complement (allocating) ---
    let encoded = encode_dna_prefer_simd(seq);
    let encoded_len = encoded.len();

    let start = Instant::now();
    let encoded_rc = reverse_complement_encoded(&encoded, seq_len);
    stats.revcomp_encoded_allocating += start.elapsed();

    // Decode and verify it matches the raw reverse complement
    let decoded_rc = decode_dna_prefer_simd(&encoded_rc, seq_len);
    assert_eq!(
        revcomp, decoded_rc,
        "Encoded revcomp doesn't match raw revcomp"
    );

    // --- Encoded reverse complement (zero-allocation) ---
    encode_buffer.resize(encoded_len, 0);
    encode_buffer.copy_from_slice(&encoded);
    encoded_revcomp_buffer.resize(encoded_len, 0);

    let start = Instant::now();
    reverse_complement_encoded_into(encode_buffer, seq_len, encoded_revcomp_buffer)
        .expect("encoded revcomp buffer too small");
    stats.revcomp_encoded_into += start.elapsed();

    assert_eq!(
        &encoded_rc,
        &encoded_revcomp_buffer[..encoded_len],
        "revcomp_encoded_into mismatch"
    );
}

/// Test quality score encoding/decoding roundtrip for a single quality string.
///
/// Note: Quality encoding uses binning, so decoded values are representative
/// bin values, not exact original values. We verify length matches and that
/// the roundtrip is consistent.
fn test_quality_roundtrip(
    qual: &[u8],
    stats: &mut RoundtripStats,
    encode_buffer: &mut Vec<u8>,
    decode_buffer: &mut Vec<u8>,
) {
    let qual_len = qual.len();
    let encoding = PhredEncoding::Phred33;

    // --- Allocating variants ---

    // Encode (allocating)
    let start = Instant::now();
    let encoded = encode_quality_scores(qual, encoding);
    stats.quality_encode_allocating += start.elapsed();

    stats.total_encoded_quality_bytes += encoded.len();

    // Decode (allocating)
    let start = Instant::now();
    let decoded = decode_quality_scores(&encoded, encoding);
    stats.quality_decode_allocating += start.elapsed();

    // Assert length matches (values will differ due to binning)
    assert_eq!(
        decoded.len(),
        qual_len,
        "Quality roundtrip length mismatch: expected {}, got {}",
        qual_len,
        decoded.len()
    );

    // Verify decoded values are valid Phred+33 ASCII
    for (i, &dec) in decoded.iter().enumerate() {
        assert!(
            (b'!'..=b'~').contains(&dec),
            "Invalid decoded quality at position {}: {} (ASCII {})",
            i,
            dec as char,
            dec
        );
    }

    // --- Zero-allocation variants ---

    // For encode_into, we need to estimate buffer size
    // Worst case is 4 bytes per run for long runs, but typically much smaller
    encode_buffer.resize(qual_len * 2, 0); // Conservative estimate

    let start = Instant::now();
    let bytes_written =
        encode_quality_scores_into(qual, encoding, encode_buffer).expect("encode buffer too small");
    stats.quality_encode_into += start.elapsed();

    // Decode into preallocated buffer
    decode_buffer.resize(qual_len, 0);
    let start = Instant::now();
    let decoded_len =
        decode_quality_scores_into(&encode_buffer[..bytes_written], encoding, decode_buffer)
            .expect("decode buffer too small");
    stats.quality_decode_into += start.elapsed();

    assert_eq!(
        decoded_len, qual_len,
        "Quality into roundtrip length mismatch"
    );

    // Verify consistency with allocating version
    assert_eq!(
        &decoded[..],
        &decode_buffer[..qual_len],
        "Quality into roundtrip doesn't match allocating version"
    );
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     simdna Roundtrip Example with seq_io FASTQ Parsing         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Platform info
    println!("Platform: {}", std::env::consts::ARCH);
    #[cfg(target_arch = "x86_64")]
    println!(
        "SSSE3 support: {}",
        std::arch::is_x86_feature_detected!("ssse3")
    );
    #[cfg(target_arch = "aarch64")]
    println!("NEON support: always available on aarch64");
    println!();

    // Open the FASTQ file
    let fastq_path = Path::new("examples/small.fastq");
    if !fastq_path.exists() {
        eprintln!("Error: FASTQ file not found at {:?}", fastq_path);
        eprintln!("Please run this example from the project root directory.");
        std::process::exit(1);
    }

    let file = File::open(fastq_path).expect("Failed to open FASTQ file");
    let reader = BufReader::new(file);
    let mut fastq_reader = Reader::new(reader);

    let mut stats = RoundtripStats::default();

    // Preallocate reusable buffers for zero-allocation variants
    let mut encode_buffer = Vec::with_capacity(1024);
    let mut decode_buffer = Vec::with_capacity(2048);
    let mut revcomp_buffer = Vec::with_capacity(2048);
    let mut encoded_revcomp_buffer = Vec::with_capacity(1024);
    let mut quality_encode_buffer = Vec::with_capacity(2048);
    let mut quality_decode_buffer = Vec::with_capacity(2048);

    println!("Processing FASTQ records...\n");

    let overall_start = Instant::now();

    // Process each record
    while let Some(result) = fastq_reader.next() {
        let record = result.expect("Error reading FASTQ record");

        let seq = record.seq();
        let qual = record.qual();

        stats.records_processed += 1;
        stats.total_sequence_bytes += seq.len();
        stats.total_quality_bytes += qual.len();

        // Test DNA encoding/decoding roundtrip
        test_dna_roundtrip(seq, &mut stats, &mut encode_buffer, &mut decode_buffer);

        // Test reverse complement roundtrip
        test_revcomp_roundtrip(
            seq,
            &mut stats,
            &mut revcomp_buffer,
            &mut encode_buffer,
            &mut encoded_revcomp_buffer,
        );

        // Test quality score roundtrip
        test_quality_roundtrip(
            qual,
            &mut stats,
            &mut quality_encode_buffer,
            &mut quality_decode_buffer,
        );

        // Progress indicator for large files
        if stats.records_processed % 500 == 0 {
            print!("\rProcessed {} records...", stats.records_processed);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }

    let overall_duration = overall_start.elapsed();

    // Clear progress line if used
    if stats.records_processed >= 500 {
        println!();
    }

    println!(
        "Total processing time: {:.3} ms",
        overall_duration.as_secs_f64() * 1000.0
    );

    stats.print_summary();
}
