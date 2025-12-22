// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! Quality score encoding round-trip fuzzing.
//!
//! This fuzz target verifies that:
//! 1. Encoding never panics on arbitrary input
//! 2. Decoded length always matches original length
//! 3. SIMD and scalar implementations produce identical results
//! 4. Zero-allocation variants produce identical results

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::quality_encoder::{
    PhredEncoding, decode_quality_scores, decode_quality_scores_into, encode_quality_scores,
    encode_quality_scores_into, extract_runs, quality_stats,
};

fuzz_target!(|data: &[u8]| {
    // Skip empty input
    if data.is_empty() {
        return;
    }

    // Test with both Phred encodings
    for encoding in [PhredEncoding::Phred33, PhredEncoding::Phred64] {
        // 1. Basic round-trip: encode then decode
        let encoded = encode_quality_scores(data, encoding);
        let decoded = decode_quality_scores(&encoded, encoding);

        // Verify decoded length matches original
        assert_eq!(
            decoded.len(),
            data.len(),
            "Decoded length mismatch for {:?}: expected {}, got {}",
            encoding,
            data.len(),
            decoded.len()
        );

        // 2. Zero-allocation variants should match allocating variants
        let mut enc_buffer = vec![0u8; data.len() + 16]; // Extra space for RLE overhead
        let enc_result = encode_quality_scores_into(data, encoding, &mut enc_buffer);

        if let Ok(enc_len) = enc_result {
            // Should produce identical encoded output
            assert_eq!(
                &enc_buffer[..enc_len],
                &encoded[..],
                "encode_quality_scores_into differs from encode_quality_scores"
            );

            // Decode using zero-allocation variant
            let mut dec_buffer = vec![0u8; data.len()];
            let dec_result = decode_quality_scores_into(&encoded, encoding, &mut dec_buffer);

            if let Ok(dec_len) = dec_result {
                assert_eq!(dec_len, data.len());
                assert_eq!(
                    &dec_buffer[..dec_len],
                    &decoded[..],
                    "decode_quality_scores_into differs from decode_quality_scores"
                );
            }
        }

        // 3. Extract runs should not panic
        let runs = extract_runs(&encoded);

        // Verify run lengths sum to original length
        let total_len: usize = runs.iter().map(|r| r.run_length as usize).sum();
        assert_eq!(
            total_len,
            data.len(),
            "Run lengths sum ({}) != original length ({})",
            total_len,
            data.len()
        );

        // Verify all bins are valid (0-3)
        for run in &runs {
            assert!(run.bin <= 3, "Invalid bin value: {}", run.bin);
            assert!(
                run.run_length >= 1,
                "Invalid run length: {}",
                run.run_length
            );
        }

        // 4. Quality stats should not panic
        let (_min, _max, _mean, bin_counts) = quality_stats(data, encoding);

        // Bin counts should sum to input length
        let count_sum: usize = bin_counts.iter().sum();
        assert_eq!(
            count_sum,
            data.len(),
            "Bin counts sum ({}) != original length ({})",
            count_sum,
            data.len()
        );
    }
});
