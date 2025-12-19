// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! Reverse complement boundary condition fuzzing.
//!
//! Tests edge cases around SIMD width boundaries (16, 32, 64 bytes) and
//! odd vs even length sequences, where the reverse complement algorithm
//! has different code paths:
//! - < 32 bytes: Uses scalar implementation
//! - >= 32 bytes even length: SIMD only
//! - >= 32 bytes odd length: SIMD + nibble shift post-processing
//!
//! This is critical for ensuring the SIMD optimization doesn't introduce
//! edge case bugs at transition points.

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::{
    decode_dna_prefer_simd, encode_dna_prefer_simd, reverse_complement, reverse_complement_encoded,
};

/// Reverse complement-relevant boundary lengths to test.
/// Includes the SIMD threshold (32 bytes) and both odd/even variants.
const BOUNDARY_LENGTHS: &[usize] = &[
    0, 1, 2, 3, // Very small edge cases
    15, 16, 17, // First SIMD boundary (16 bytes)
    31, 32, 33, // SIMD threshold boundary (scalar vs SIMD transition)
    47, 48, 49, // Triple SIMD width
    63, 64, 65, // Quad SIMD width (common odd/even transition)
    127, 128, 129, // 8x SIMD width
    255, 256, 257, // 16x SIMD width
    511, 512, 513, // Larger sequences
    1023, 1024, 1025, // Large odd/even boundary
];

fuzz_target!(|data: &[u8]| {
    // Use input bytes to generate valid DNA sequences at various boundary lengths
    for &target_len in BOUNDARY_LENGTHS {
        if target_len == 0 {
            // Test empty sequence explicitly
            let empty_rc = reverse_complement(&[]);
            assert!(
                empty_rc.is_empty(),
                "Empty sequence should produce empty result"
            );
            continue;
        }

        // Generate a sequence of the target length using input as seed
        let sequence = generate_dna_sequence(data, target_len);

        // Test high-level API
        let rc = reverse_complement(&sequence);
        assert_eq!(
            rc.len(),
            sequence.len(),
            "High-level API: length mismatch at boundary {} (odd={})",
            target_len,
            target_len % 2 == 1
        );

        // Test low-level API
        let encoded = encode_dna_prefer_simd(&sequence);
        let rc_encoded = reverse_complement_encoded(&encoded, sequence.len());
        let rc_decoded = decode_dna_prefer_simd(&rc_encoded, sequence.len());

        // Verify both APIs produce same result
        assert_eq!(
            rc,
            rc_decoded,
            "API mismatch at boundary {} (odd={}): high-level vs low-level differ",
            target_len,
            target_len % 2 == 1
        );

        // Verify double reverse complement equals original (normalized)
        let rc_rc = reverse_complement(&rc);
        let normalized = normalize_sequence(&sequence);
        assert_eq!(
            rc_rc,
            normalized,
            "Double reverse complement failed at boundary {} (odd={})",
            target_len,
            target_len % 2 == 1
        );

        // Verify complement correctness at each position
        for i in 0..sequence.len() {
            let original_pos = sequence.len() - 1 - i;
            let original_char = normalize_char(sequence[original_pos]);
            let expected_complement = complement_char(original_char);
            assert_eq!(
                rc[i],
                expected_complement,
                "Complement incorrect at boundary {}, position {} (odd={})",
                target_len,
                i,
                target_len % 2 == 1
            );
        }
    }

    // Also test the exact input length from fuzzer (catches unlisted boundary lengths)
    if !data.is_empty() {
        let rc = reverse_complement(data);
        let rc_rc = reverse_complement(&rc);
        let normalized = normalize_sequence(data);

        assert_eq!(
            rc_rc,
            normalized,
            "Double reverse complement failed for fuzzer length {}",
            data.len()
        );
    }
});

/// Generate a DNA sequence of a specific length using input bytes as entropy.
fn generate_dna_sequence(seed: &[u8], length: usize) -> Vec<u8> {
    const BASES: &[u8] = b"ACGTRYSWKMBDHVN-";

    if length == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(length);
    for i in 0..length {
        // Use seed bytes cyclically to determine each base
        let seed_byte = if seed.is_empty() {
            i as u8
        } else {
            seed[i % seed.len()]
        };
        result.push(BASES[(seed_byte as usize) % BASES.len()]);
    }
    result
}

/// Normalize a character to its expected decoded form.
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
        _ => b'-',
    }
}

/// Normalize a sequence
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
        b'S' => b'S',
        b'W' => b'W',
        b'K' => b'M',
        b'M' => b'K',
        b'B' => b'V',
        b'V' => b'B',
        b'D' => b'H',
        b'H' => b'D',
        b'N' => b'N',
        b'-' => b'-',
        _ => b'-',
    }
}
