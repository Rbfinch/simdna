// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! SIMD vs Scalar equivalence fuzzing.
//!
//! Ensures that the SIMD and scalar implementations produce identical results
//! for both encoding and decoding operations.

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::{decode_scalar, encode_scalar};

fuzz_target!(|data: &[u8]| {
    // Skip empty input
    if data.is_empty() {
        return;
    }

    // Normalize input (as the main encode function does)
    let normalized: Vec<u8> = data.iter().map(|&c| c.to_ascii_uppercase()).collect();

    // Pad to multiple of 2
    let padded_len = (normalized.len() + 1) & !1;
    let mut padded = normalized.clone();
    padded.resize(padded_len, b'-');

    // === Test Encoding ===

    // Scalar encoding
    let mut scalar_encoded = vec![0u8; padded_len / 2];
    encode_scalar(&padded, &mut scalar_encoded);

    // SIMD encoding (via the prefer_simd function)
    let simd_encoded = simdna::dna_simd_encoder::encode_dna_prefer_simd(data);

    assert_eq!(
        scalar_encoded,
        simd_encoded,
        "Encoding mismatch between SIMD and scalar for input length {}",
        data.len()
    );

    // === Test Decoding ===

    // Use the encoded data to test decoding
    let original_len = data.len();

    // Scalar decoding
    let mut scalar_decoded = vec![0u8; original_len];
    decode_scalar(&scalar_encoded, &mut scalar_decoded, original_len);

    // SIMD decoding (via the prefer_simd function)
    let simd_decoded =
        simdna::dna_simd_encoder::decode_dna_prefer_simd(&simd_encoded, original_len);

    assert_eq!(
        scalar_decoded,
        simd_decoded,
        "Decoding mismatch between SIMD and scalar for input length {}",
        data.len()
    );
});
