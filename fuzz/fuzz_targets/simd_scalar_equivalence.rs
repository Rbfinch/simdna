// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! SIMD vs Scalar equivalence fuzzing.
//!
//! Ensures that the SIMD and scalar implementations produce identical results
//! for encoding, decoding, and reverse complement operations.

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::{
    decode_scalar, encode_dna_prefer_simd, encode_scalar, reverse_complement,
    reverse_complement_encoded,
};

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

    // === Test Reverse Complement ===
    // The high-level API uses SIMD for sequences >= 32 bytes,
    // the low-level encode→reverse→decode path should produce the same result

    // High-level reverse complement (may use SIMD + nibble shift internally)
    let rc_high_level = reverse_complement(data);

    // Low-level path: encode → reverse complement encoded → decode
    let encoded = encode_dna_prefer_simd(data);
    let rc_encoded = reverse_complement_encoded(&encoded, original_len);
    let rc_low_level = simdna::dna_simd_encoder::decode_dna_prefer_simd(&rc_encoded, original_len);

    assert_eq!(
        rc_high_level,
        rc_low_level,
        "Reverse complement mismatch between high-level and low-level API for input length {}",
        data.len()
    );

    // Verify involution: double reverse complement should equal normalized original
    let rc_rc = reverse_complement(&rc_high_level);
    let normalized_input: Vec<u8> = data.iter().map(|&c| normalize_iupac(c)).collect();

    assert_eq!(
        rc_rc,
        normalized_input,
        "Double reverse complement failed for input length {}",
        data.len()
    );
});

/// Normalize a character to its expected decoded form.
#[inline]
fn normalize_iupac(c: u8) -> u8 {
    match c {
        b'A' | b'a' => b'A',
        b'C' | b'c' => b'C',
        b'G' | b'g' => b'G',
        b'T' | b't' | b'U' | b'u' => b'T',
        b'R' | b'r' => b'R',
        b'Y' | b'y' => b'Y',
        b'S' | b's' => b'S',
        b'W' | b'w' => b'W',
        b'K' | b'k' => b'K',
        b'M' | b'm' => b'M',
        b'B' | b'b' => b'B',
        b'D' | b'd' => b'D',
        b'H' | b'h' => b'H',
        b'V' | b'v' => b'V',
        b'N' | b'n' => b'N',
        b'-' | b'.' => b'-',
        _ => b'-',
    }
}
