// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! Reverse complement fuzzing.
//!
//! This fuzz target verifies the correctness of reverse complement operations:
//! 1. Double reverse complement equals original: revcomp(revcomp(x)) == x
//! 2. High-level and low-level APIs produce consistent results
//! 3. Length preservation: output length equals input length
//! 4. IUPAC complement correctness for all valid codes

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::{
    decode_dna_prefer_simd, encode_dna_prefer_simd, reverse_complement,
    reverse_complement_encoded,
};

fuzz_target!(|data: &[u8]| {
    // Skip empty input
    if data.is_empty() {
        return;
    }

    // Property 1: Length preservation
    let rc = reverse_complement(data);
    assert_eq!(
        rc.len(),
        data.len(),
        "Reverse complement length mismatch: input={}, output={}",
        data.len(),
        rc.len()
    );

    // Property 2: Double reverse complement equals normalized original
    // Note: The first revcomp normalizes the input (uppercase, invalid→gap)
    // so we compare against the normalized version
    let rc_rc = reverse_complement(&rc);
    let normalized = normalize_sequence(data);

    assert_eq!(
        rc_rc.len(),
        normalized.len(),
        "Double reverse complement length mismatch"
    );

    for (i, (&expected, &actual)) in normalized.iter().zip(rc_rc.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Double reverse complement mismatch at position {}: expected {:02X} ({:?}), got {:02X} ({:?})",
            i, expected, expected as char, actual, actual as char
        );
    }

    // Property 3: High-level and low-level API consistency
    // reverse_complement(data) should equal:
    // decode(reverse_complement_encoded(encode(data), len), len)
    let encoded = encode_dna_prefer_simd(data);
    let rc_encoded = reverse_complement_encoded(&encoded, data.len());
    let rc_from_encoded = decode_dna_prefer_simd(&rc_encoded, data.len());

    assert_eq!(
        rc.len(),
        rc_from_encoded.len(),
        "API consistency: length mismatch between high-level and low-level APIs"
    );

    for (i, (&hl, &ll)) in rc.iter().zip(rc_from_encoded.iter()).enumerate() {
        assert_eq!(
            hl, ll,
            "API consistency mismatch at position {}: high-level={:02X} ({:?}), low-level={:02X} ({:?})",
            i, hl, hl as char, ll, ll as char
        );
    }

    // Property 4: Verify complement relationships for each position
    // After reverse, position i in output corresponds to position (len-1-i) in input
    // and should be the complement of that base
    for i in 0..data.len() {
        let original_pos = data.len() - 1 - i;
        let original_char = normalize_char(data[original_pos]);
        let rc_char = rc[i];
        let expected_complement = complement_char(original_char);

        assert_eq!(
            rc_char, expected_complement,
            "Complement mismatch at output position {} (input position {}): \
             original={:02X} ({:?}), got={:02X} ({:?}), expected complement={:02X} ({:?})",
            i,
            original_pos,
            original_char,
            original_char as char,
            rc_char,
            rc_char as char,
            expected_complement,
            expected_complement as char
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

/// Normalize a sequence (uppercase, invalid chars to gap)
#[inline]
fn normalize_sequence(data: &[u8]) -> Vec<u8> {
    data.iter().map(|&c| normalize_char(c)).collect()
}

/// Get the complement of a normalized character
#[inline]
fn complement_char(c: u8) -> u8 {
    match c {
        b'A' => b'T',
        b'T' => b'A',
        b'C' => b'G',
        b'G' => b'C',
        b'R' => b'Y',
        b'Y' => b'R',
        b'S' => b'S', // Self-complementary
        b'W' => b'W', // Self-complementary
        b'K' => b'M',
        b'M' => b'K',
        b'B' => b'V',
        b'V' => b'B',
        b'D' => b'H',
        b'H' => b'D',
        b'N' => b'N', // Self-complementary
        b'-' => b'-', // Self-complementary
        _ => b'-',
    }
}
