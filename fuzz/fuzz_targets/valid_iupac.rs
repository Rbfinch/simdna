// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! Valid IUPAC input fuzzing.
//!
//! Generates sequences containing only valid IUPAC nucleotide codes and
//! verifies that encoding always succeeds and round-trips correctly.

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::{decode_dna_prefer_simd, encode_dna_prefer_simd};

/// Valid IUPAC nucleotide codes (uppercase) including U for RNA
/// Standard codes: A, C, G, T, U (RNA), R, Y, S, W, K, M, B, D, H, V, N, -
const VALID_BASES_UPPER: &[u8] = b"ACGTURYSWKMBDHVN-";
/// Valid IUPAC nucleotide codes (lowercase) including u for RNA
const VALID_BASES_LOWER: &[u8] = b"acgturyswkmbdhvn.";

fuzz_target!(|data: &[u8]| {
    // Skip empty input
    if data.is_empty() {
        return;
    }

    // Generate a valid IUPAC sequence from the input bytes
    let valid_sequence: Vec<u8> = data
        .iter()
        .map(|&b| {
            // Use the byte to select from valid bases
            // Alternate between upper and lower case based on high bit
            if b & 0x80 != 0 {
                VALID_BASES_LOWER[(b & 0x0F) as usize % VALID_BASES_LOWER.len()]
            } else {
                VALID_BASES_UPPER[(b & 0x0F) as usize % VALID_BASES_UPPER.len()]
            }
        })
        .collect();

    // Encode the valid sequence
    let encoded = encode_dna_prefer_simd(&valid_sequence);

    // Verify encoded length is correct (ceil(len/2))
    let expected_encoded_len = (valid_sequence.len() + 1) / 2;
    assert_eq!(
        encoded.len(),
        expected_encoded_len,
        "Encoded length mismatch for input length {}",
        valid_sequence.len()
    );

    // Decode back
    let decoded = decode_dna_prefer_simd(&encoded, valid_sequence.len());

    // Verify round-trip
    assert_eq!(
        decoded.len(),
        valid_sequence.len(),
        "Decoded length mismatch"
    );

    // Compare with expected normalized output
    for (i, (&input, &output)) in valid_sequence.iter().zip(decoded.iter()).enumerate() {
        let expected = normalize_iupac(input);
        assert_eq!(
            output, expected,
            "Mismatch at position {}: input={:?}, expected={:?}, got={:?}",
            i, input as char, expected as char, output as char
        );
    }
});

/// Normalize an IUPAC character to its canonical uppercase form.
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
