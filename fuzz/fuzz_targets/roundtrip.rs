// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! Round-trip fuzzing: encode -> decode should produce the original sequence.
//!
//! This is the most important fuzz target as it verifies the fundamental
//! correctness property of the encoding/decoding.

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::{decode_dna_prefer_simd, encode_dna_prefer_simd};

fuzz_target!(|data: &[u8]| {
    // Skip empty input
    if data.is_empty() {
        return;
    }

    // Encode the input
    let encoded = encode_dna_prefer_simd(data);

    // Decode back to original length
    let decoded = decode_dna_prefer_simd(&encoded, data.len());

    // Verify the round-trip property:
    // Each byte should decode to its expected normalized form
    assert_eq!(
        decoded.len(),
        data.len(),
        "Decoded length mismatch: expected {}, got {}",
        data.len(),
        decoded.len()
    );

    // The decoded sequence should match the normalized input
    // (uppercase, with invalid chars mapped to '-')
    for (i, (&original, &decoded_byte)) in data.iter().zip(decoded.iter()).enumerate() {
        let expected = normalize_char(original);
        assert_eq!(
            decoded_byte,
            expected,
            "Mismatch at position {}: original={:02X} ({:?}), expected={:02X} ({:?}), got={:02X} ({:?})",
            i,
            original,
            original as char,
            expected,
            expected as char,
            decoded_byte,
            decoded_byte as char
        );
    }
});

/// Normalize a character to its expected decoded form.
/// Maps valid IUPAC codes to uppercase, invalid chars to '-'.
#[inline]
fn normalize_char(c: u8) -> u8 {
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
        _ => b'-', // Unknown characters become gaps
    }
}
