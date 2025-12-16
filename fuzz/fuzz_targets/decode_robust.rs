//! Decode robustness fuzzing.
//!
//! Tests that the decoder handles arbitrary (potentially malformed) input
//! gracefully without panicking or causing undefined behavior.

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::decode_dna_prefer_simd;

fuzz_target!(|data: &[u8]| {
    // Test various decode lengths to stress boundary handling
    // The fuzzer provides arbitrary "encoded" data

    // Test with exact capacity (2 nucleotides per byte)
    let max_len = data.len() * 2;
    let _ = decode_dna_prefer_simd(data, max_len);

    // Test with length less than capacity
    if max_len > 0 {
        let _ = decode_dna_prefer_simd(data, max_len / 2);
        let _ = decode_dna_prefer_simd(data, 1);
    }

    // Test with odd length (tests proper handling of partial bytes)
    if max_len > 1 {
        let _ = decode_dna_prefer_simd(data, max_len - 1);
    }

    // Test with zero length
    let result = decode_dna_prefer_simd(data, 0);
    assert!(result.is_empty(), "Zero length should produce empty output");

    // Test lengths at SIMD boundaries
    for boundary in [15, 16, 17, 31, 32, 33, 63, 64, 65] {
        if max_len >= boundary {
            let _ = decode_dna_prefer_simd(data, boundary);
        }
    }
});
