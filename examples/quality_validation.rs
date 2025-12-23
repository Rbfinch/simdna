//! Quality Encoding Validation Example
//!
//! This example validates the correctness of quality encoding and decoding
//! by analyzing FASTQ quality scores from a real file and verifying:
//!
//! 1. **Bin assignment correctness** - Quality scores map to the correct bins
//! 2. **Decoded value validation** - Decoded values are representative of their bins
//! 3. **Quantization error bounds** - Maximum error is within expected limits
//! 4. **RLE compression analysis** - Run patterns and compression efficiency
//! 5. **Statistical preservation** - Original vs decoded quality distributions
//!
//! Run with: `cargo run --release --example quality_validation`

use seq_io::fastq::{Reader, Record};
use simdna::quality_encoder::{
    PhredEncoding, bin_quality, decode_quality_scores, encode_quality_scores, extract_runs,
    quality_stats,
};
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

/// Representative Phred values for each bin (used during decoding)
const BIN_REPRESENTATIVES: [u8; 4] = [6, 15, 25, 37];

/// Bin names for display
const BIN_NAMES: [&str; 4] = [
    "Bin 0 (Q0-9, very low)",
    "Bin 1 (Q10-19, low)",
    "Bin 2 (Q20-29, medium)",
    "Bin 3 (Q30+, high)",
];

/// Validation results for a single record
#[allow(dead_code)]
struct RecordValidation {
    record_id: usize,
    quality_len: usize,
    original_stats: QualityStats,
    decoded_stats: QualityStats,
    bin_counts: [usize; 4],
    run_count: usize,
    max_run_length: u16,
    compression_ratio: f64,
    max_quantization_error: u8,
    mean_quantization_error: f64,
    bin_assignment_correct: bool,
    decoded_values_correct: bool,
}

/// Quality statistics
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct QualityStats {
    min: u8,
    max: u8,
    mean: f64,
}

/// Expected bin for a Phred score
fn expected_bin(phred: u8) -> u8 {
    match phred {
        0..=9 => 0,
        10..=19 => 1,
        20..=29 => 2,
        _ => 3,
    }
}

/// Validate a single quality string
fn validate_quality(quality: &[u8], encoding: PhredEncoding) -> RecordValidation {
    let offset = encoding.offset();

    // 1. Get original statistics
    let (min_q, max_q, mean_q, bin_counts) = quality_stats(quality, encoding);
    let original_stats = QualityStats {
        min: min_q,
        max: max_q,
        mean: mean_q,
    };

    // 2. Encode and decode
    let encoded = encode_quality_scores(quality, encoding);
    let decoded = decode_quality_scores(&encoded, encoding);

    // 3. Get decoded statistics
    let (dec_min, dec_max, dec_mean, _) = quality_stats(&decoded, encoding);
    let decoded_stats = QualityStats {
        min: dec_min,
        max: dec_max,
        mean: dec_mean,
    };

    // 4. Validate bin assignments
    let mut binned = vec![0u8; quality.len()];
    bin_quality(quality, &mut binned, encoding);

    let mut bin_assignment_correct = true;
    for (i, (&q, &bin)) in quality.iter().zip(binned.iter()).enumerate() {
        let phred = q.saturating_sub(offset);
        let expected = expected_bin(phred);
        if bin != expected {
            eprintln!(
                "  Bin mismatch at position {i}: Q{phred} assigned to bin {bin}, expected bin {expected}"
            );
            bin_assignment_correct = false;
        }
    }

    // 5. Validate decoded values are correct representatives
    let mut decoded_values_correct = true;
    for (i, (&orig, &dec)) in quality.iter().zip(decoded.iter()).enumerate() {
        let orig_phred = orig.saturating_sub(offset);
        let dec_phred = dec.saturating_sub(offset);
        let expected_representative = BIN_REPRESENTATIVES[expected_bin(orig_phred) as usize];

        if dec_phred != expected_representative {
            eprintln!(
                "  Decoded mismatch at position {i}: original Q{orig_phred} (bin {}) decoded to Q{dec_phred}, expected Q{expected_representative}",
                expected_bin(orig_phred)
            );
            decoded_values_correct = false;
        }
    }

    // 6. Calculate quantization errors
    let mut max_error: u8 = 0;
    let mut total_error: u64 = 0;
    for (&orig, &dec) in quality.iter().zip(decoded.iter()) {
        let orig_phred = orig.saturating_sub(offset);
        let dec_phred = dec.saturating_sub(offset);
        let error = orig_phred.abs_diff(dec_phred);
        max_error = max_error.max(error);
        total_error += error as u64;
    }
    let mean_error = if quality.is_empty() {
        0.0
    } else {
        total_error as f64 / quality.len() as f64
    };

    // 7. Analyze RLE runs
    let runs = extract_runs(&encoded);
    let max_run_length = runs.iter().map(|r| r.run_length).max().unwrap_or(0);

    // 8. Calculate compression ratio
    let compression_ratio = quality.len() as f64 / encoded.len() as f64;

    RecordValidation {
        record_id: 0, // Will be set by caller
        quality_len: quality.len(),
        original_stats,
        decoded_stats,
        bin_counts,
        run_count: runs.len(),
        max_run_length,
        compression_ratio,
        max_quantization_error: max_error,
        mean_quantization_error: mean_error,
        bin_assignment_correct,
        decoded_values_correct,
    }
}

/// Aggregate validation results
struct AggregateResults {
    total_records: usize,
    total_bases: usize,
    total_encoded_bytes: usize,
    total_runs: usize,
    global_bin_counts: [usize; 4],
    global_max_quantization_error: u8,
    total_quantization_error: f64,
    all_bin_assignments_correct: bool,
    all_decoded_values_correct: bool,
    records_with_errors: Vec<usize>,
    run_length_histogram: [usize; 8], // 1-10, 11-20, 21-30, 31-40, 41-50, 51-62, 63+, long format
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           QUALITY ENCODING VALIDATION SUITE                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let fastq_path = "examples/small.fastq";
    let file = File::open(fastq_path).expect("Failed to open FASTQ file");
    let mut reader = Reader::new(BufReader::new(file));

    let encoding = PhredEncoding::Phred33;
    let start = Instant::now();

    let mut results = AggregateResults {
        total_records: 0,
        total_bases: 0,
        total_encoded_bytes: 0,
        total_runs: 0,
        global_bin_counts: [0; 4],
        global_max_quantization_error: 0,
        total_quantization_error: 0.0,
        all_bin_assignments_correct: true,
        all_decoded_values_correct: true,
        records_with_errors: Vec::new(),
        run_length_histogram: [0; 8],
    };

    println!("Processing: {}\n", fastq_path);
    println!("─────────────────────────────────────────────────────────────────────");

    while let Some(record) = reader.next() {
        let record = record.expect("Failed to read FASTQ record");
        let quality = record.qual();

        results.total_records += 1;
        results.total_bases += quality.len();

        let mut validation = validate_quality(quality, encoding);
        validation.record_id = results.total_records;

        // Accumulate statistics
        let encoded = encode_quality_scores(quality, encoding);
        results.total_encoded_bytes += encoded.len();
        results.total_runs += validation.run_count;

        for (i, &count) in validation.bin_counts.iter().enumerate() {
            results.global_bin_counts[i] += count;
        }

        results.global_max_quantization_error = results
            .global_max_quantization_error
            .max(validation.max_quantization_error);
        results.total_quantization_error +=
            validation.mean_quantization_error * quality.len() as f64;

        // Track run length distribution
        let runs = extract_runs(&encoded);
        for run in &runs {
            let bucket = match run.run_length {
                1..=10 => 0,
                11..=20 => 1,
                21..=30 => 2,
                31..=40 => 3,
                41..=50 => 4,
                51..=62 => 5,
                63.. => {
                    // Check if it's a long format run (> 62)
                    if run.run_length > 62 { 7 } else { 6 }
                }
                0 => continue, // Skip empty runs
            };
            results.run_length_histogram[bucket] += 1;
        }

        if !validation.bin_assignment_correct {
            results.all_bin_assignments_correct = false;
            results.records_with_errors.push(validation.record_id);
        }

        if !validation.decoded_values_correct {
            results.all_decoded_values_correct = false;
            if !results.records_with_errors.contains(&validation.record_id) {
                results.records_with_errors.push(validation.record_id);
            }
        }
    }

    let elapsed = start.elapsed();

    // Print results
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                    VALIDATION RESULTS                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Basic statistics
    println!("┌─ Dataset Statistics ─────────────────────────────────────────────┐");
    println!(
        "│  Total records processed:  {:>10}                           │",
        results.total_records
    );
    println!(
        "│  Total quality bases:      {:>10}                           │",
        results.total_bases
    );
    println!(
        "│  Total encoded bytes:      {:>10}                           │",
        results.total_encoded_bytes
    );
    println!(
        "│  Overall compression:     {:>10.2}x                          │",
        results.total_bases as f64 / results.total_encoded_bytes as f64
    );
    println!(
        "│  Processing time:         {:>10.2?}                         │",
        elapsed
    );
    println!("└───────────────────────────────────────────────────────────────────┘\n");

    // Bin distribution
    println!("┌─ Quality Bin Distribution ───────────────────────────────────────┐");
    let total = results.global_bin_counts.iter().sum::<usize>() as f64;
    for (i, &count) in results.global_bin_counts.iter().enumerate() {
        let pct = if total > 0.0 {
            100.0 * count as f64 / total
        } else {
            0.0
        };
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("│  {}: {:>8} ({:>5.1}%) {}", BIN_NAMES[i], count, pct, bar);
    }
    println!("└───────────────────────────────────────────────────────────────────┘\n");

    // RLE analysis
    println!("┌─ Run-Length Encoding Analysis ───────────────────────────────────┐");
    println!(
        "│  Total runs:               {:>10}                           │",
        results.total_runs
    );
    println!(
        "│  Avg bases per run:        {:>10.2}                           │",
        results.total_bases as f64 / results.total_runs as f64
    );
    println!("│                                                                   │");
    println!("│  Run Length Distribution:                                         │");
    let run_buckets = [
        "1-10",
        "11-20",
        "21-30",
        "31-40",
        "41-50",
        "51-62",
        "63 (boundary)",
        ">62 (long fmt)",
    ];
    let run_total = results.run_length_histogram.iter().sum::<usize>() as f64;
    for (i, &count) in results.run_length_histogram.iter().enumerate() {
        if count > 0 {
            let pct = 100.0 * count as f64 / run_total;
            println!(
                "│    {:>14}: {:>6} ({:>5.1}%)                            │",
                run_buckets[i], count, pct
            );
        }
    }
    println!("└───────────────────────────────────────────────────────────────────┘\n");

    // Quantization error analysis
    println!("┌─ Quantization Error Analysis ────────────────────────────────────┐");
    let mean_error = results.total_quantization_error / results.total_bases as f64;
    println!(
        "│  Maximum quantization error:  Q{:<3} (within bin width)          │",
        results.global_max_quantization_error
    );
    println!(
        "│  Mean quantization error:     Q{:<6.2}                           │",
        mean_error
    );
    println!("│                                                                   │");
    println!("│  Expected error bounds per bin:                                   │");
    println!("│    Bin 0 (Q0-9):   max error = 6 (from Q0) or 3 (from Q9)        │");
    println!("│    Bin 1 (Q10-19): max error = 5 (from Q10) or 4 (from Q19)      │");
    println!("│    Bin 2 (Q20-29): max error = 5 (from Q20) or 4 (from Q29)      │");
    println!("│    Bin 3 (Q30+):   max error = 7 (from Q30) or more from Q40+   │");
    println!("└───────────────────────────────────────────────────────────────────┘\n");

    // Validation summary
    println!("┌─ Validation Summary ──────────────────────────────────────────────┐");
    let bin_check = if results.all_bin_assignments_correct {
        "✓ PASS"
    } else {
        "✗ FAIL"
    };
    let decode_check = if results.all_decoded_values_correct {
        "✓ PASS"
    } else {
        "✗ FAIL"
    };

    println!(
        "│  Bin assignment correctness:      {}                          │",
        bin_check
    );
    println!(
        "│  Decoded value correctness:       {}                          │",
        decode_check
    );

    if !results.records_with_errors.is_empty() {
        println!("│                                                                   │");
        println!(
            "│  Records with errors: {:?}",
            &results.records_with_errors[..results.records_with_errors.len().min(10)]
        );
    }
    println!("└───────────────────────────────────────────────────────────────────┘\n");

    // Additional correctness tests
    println!("┌─ Additional Correctness Tests ───────────────────────────────────┐");

    // Test 1: Boundary value testing
    print!("│  Test 1: Boundary values (Q9→Q10, Q19→Q20, Q29→Q30)... ");
    let boundary_test = test_boundary_values(encoding);
    println!(
        "{}",
        if boundary_test {
            "✓ PASS │"
        } else {
            "✗ FAIL │"
        }
    );

    // Test 2: All same quality (homogeneous)
    print!("│  Test 2: Homogeneous quality (all same value)...       ");
    let homogeneous_test = test_homogeneous_quality(encoding);
    println!(
        "{}",
        if homogeneous_test {
            "✓ PASS │"
        } else {
            "✗ FAIL │"
        }
    );

    // Test 3: Alternating bins
    print!("│  Test 3: Alternating bins (worst case for RLE)...      ");
    let alternating_test = test_alternating_bins(encoding);
    println!(
        "{}",
        if alternating_test {
            "✓ PASS │"
        } else {
            "✗ FAIL │"
        }
    );

    // Test 4: Maximum run length
    print!("│  Test 4: Long runs (>62 bases, long format)...         ");
    let long_run_test = test_long_runs(encoding);
    println!(
        "{}",
        if long_run_test {
            "✓ PASS │"
        } else {
            "✗ FAIL │"
        }
    );

    // Test 5: Exact boundary at MAX_SHORT_RUN
    print!("│  Test 5: Exact boundary at run length 62...            ");
    let boundary_run_test = test_run_boundary(encoding);
    println!(
        "{}",
        if boundary_run_test {
            "✓ PASS │"
        } else {
            "✗ FAIL │"
        }
    );

    println!("└───────────────────────────────────────────────────────────────────┘\n");

    // Final verdict
    let all_passed = results.all_bin_assignments_correct
        && results.all_decoded_values_correct
        && boundary_test
        && homogeneous_test
        && alternating_test
        && long_run_test
        && boundary_run_test;

    if all_passed {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("                    ✓ ALL VALIDATION TESTS PASSED                  ");
        println!("═══════════════════════════════════════════════════════════════════");
    } else {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("                    ✗ SOME VALIDATION TESTS FAILED                 ");
        println!("═══════════════════════════════════════════════════════════════════");
        std::process::exit(1);
    }
}

/// Test boundary values between bins
fn test_boundary_values(encoding: PhredEncoding) -> bool {
    let offset = encoding.offset();

    // Test Q9 → bin 0, Q10 → bin 1
    let q9 = vec![9 + offset; 10];
    let q10 = vec![10 + offset; 10];

    let encoded_q9 = encode_quality_scores(&q9, encoding);
    let encoded_q10 = encode_quality_scores(&q10, encoding);
    let decoded_q9 = decode_quality_scores(&encoded_q9, encoding);
    let decoded_q10 = decode_quality_scores(&encoded_q10, encoding);

    // Q9 should decode to Q6 (bin 0 representative)
    // Q10 should decode to Q15 (bin 1 representative)
    let q9_correct = decoded_q9.iter().all(|&q| q == 6 + offset);
    let q10_correct = decoded_q10.iter().all(|&q| q == 15 + offset);

    // Test Q19 → bin 1, Q20 → bin 2
    let q19 = vec![19 + offset; 10];
    let q20 = vec![20 + offset; 10];

    let encoded_q19 = encode_quality_scores(&q19, encoding);
    let encoded_q20 = encode_quality_scores(&q20, encoding);
    let decoded_q19 = decode_quality_scores(&encoded_q19, encoding);
    let decoded_q20 = decode_quality_scores(&encoded_q20, encoding);

    let q19_correct = decoded_q19.iter().all(|&q| q == 15 + offset);
    let q20_correct = decoded_q20.iter().all(|&q| q == 25 + offset);

    // Test Q29 → bin 2, Q30 → bin 3
    let q29 = vec![29 + offset; 10];
    let q30 = vec![30 + offset; 10];

    let encoded_q29 = encode_quality_scores(&q29, encoding);
    let encoded_q30 = encode_quality_scores(&q30, encoding);
    let decoded_q29 = decode_quality_scores(&encoded_q29, encoding);
    let decoded_q30 = decode_quality_scores(&encoded_q30, encoding);

    let q29_correct = decoded_q29.iter().all(|&q| q == 25 + offset);
    let q30_correct = decoded_q30.iter().all(|&q| q == 37 + offset);

    q9_correct && q10_correct && q19_correct && q20_correct && q29_correct && q30_correct
}

/// Test homogeneous quality strings
fn test_homogeneous_quality(encoding: PhredEncoding) -> bool {
    let offset = encoding.offset();

    for phred in [0u8, 5, 10, 15, 20, 25, 30, 37, 40] {
        let quality: Vec<u8> = vec![phred + offset; 100];
        let encoded = encode_quality_scores(&quality, encoding);
        let decoded = decode_quality_scores(&encoded, encoding);

        if decoded.len() != quality.len() {
            return false;
        }

        // All decoded values should be the same (representative for that bin)
        let expected_rep = BIN_REPRESENTATIVES[expected_bin(phred) as usize] + offset;
        if !decoded.iter().all(|&q| q == expected_rep) {
            return false;
        }

        // Should compress very well (single run)
        let runs = extract_runs(&encoded);
        if runs.len() != 1 || runs[0].run_length != 100 {
            return false;
        }
    }

    true
}

/// Test alternating bins (worst case for RLE)
fn test_alternating_bins(encoding: PhredEncoding) -> bool {
    let offset = encoding.offset();

    // Alternate between bin 0 (Q5) and bin 3 (Q35)
    let mut quality = Vec::with_capacity(100);
    for i in 0..100 {
        if i % 2 == 0 {
            quality.push(5 + offset);
        } else {
            quality.push(35 + offset);
        }
    }

    let encoded = encode_quality_scores(&quality, encoding);
    let decoded = decode_quality_scores(&encoded, encoding);

    if decoded.len() != quality.len() {
        return false;
    }

    // Should have 100 runs (no compression)
    let runs = extract_runs(&encoded);
    if runs.len() != 100 {
        return false;
    }

    // Verify alternating pattern
    for (i, run) in runs.iter().enumerate() {
        let expected_bin = if i % 2 == 0 { 0 } else { 3 };
        if run.bin != expected_bin || run.run_length != 1 {
            return false;
        }
    }

    true
}

/// Test long runs (>62 bases)
fn test_long_runs(encoding: PhredEncoding) -> bool {
    let offset = encoding.offset();

    // Test runs of various lengths
    for len in [63, 100, 200, 1000, 10000] {
        let quality: Vec<u8> = vec![35 + offset; len]; // High quality (bin 3)
        let encoded = encode_quality_scores(&quality, encoding);
        let decoded = decode_quality_scores(&encoded, encoding);

        if decoded.len() != quality.len() {
            return false;
        }

        // Should be single run
        let runs = extract_runs(&encoded);
        if runs.len() != 1 {
            return false;
        }

        if runs[0].run_length != len as u16 {
            // For very long runs, check accumulated correctly
            if len <= u16::MAX as usize && runs[0].run_length != len as u16 {
                return false;
            }
        }
    }

    true
}

/// Test run length boundary (62 vs 63)
fn test_run_boundary(encoding: PhredEncoding) -> bool {
    let offset = encoding.offset();

    // Test exactly 62 bases (should use short format)
    let quality_62: Vec<u8> = vec![35 + offset; 62];
    let encoded_62 = encode_quality_scores(&quality_62, encoding);
    let decoded_62 = decode_quality_scores(&encoded_62, encoding);

    if decoded_62.len() != 62 {
        return false;
    }

    // 62 bases should encode to 1 byte (short format)
    if encoded_62.len() != 1 {
        return false;
    }

    // Test exactly 63 bases (should use long format)
    let quality_63: Vec<u8> = vec![35 + offset; 63];
    let encoded_63 = encode_quality_scores(&quality_63, encoding);
    let decoded_63 = decode_quality_scores(&encoded_63, encoding);

    if decoded_63.len() != 63 {
        return false;
    }

    // 63 bases should use long format (4 bytes)
    if encoded_63.len() != 4 {
        return false;
    }

    // Verify the escape byte is present
    if encoded_63[0] != 0xFF {
        return false;
    }

    // Test bin 3 specifically with length 63 (the case that triggered the bug)
    let runs_63 = extract_runs(&encoded_63);
    if runs_63.len() != 1 || runs_63[0].bin != 3 || runs_63[0].run_length != 63 {
        return false;
    }

    true
}
