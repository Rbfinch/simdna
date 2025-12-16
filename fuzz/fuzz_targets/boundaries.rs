// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! SIMD boundary condition fuzzing.
//!
//! Tests edge cases around SIMD width boundaries (16, 32, 64 bytes) where
//! transitions between SIMD and scalar processing occur.

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::{decode_dna_prefer_simd, encode_dna_prefer_simd};

/// SIMD-relevant boundary lengths to test
const BOUNDARY_LENGTHS: &[usize] = &[
    0, 1, 2, // Edge cases
    15, 16, 17, // First SIMD boundary (16 bytes)
    31, 32, 33, // Double SIMD width
    47, 48, 49, // Triple SIMD width
    63, 64, 65, // Quad SIMD width
    127, 128, 129, // 8x SIMD width
    255, 256, 257, // 16x SIMD width
];

fuzz_target!(|data: &[u8]| {
    // Use input bytes to generate valid DNA sequences at various boundary lengths
    for &target_len in BOUNDARY_LENGTHS {
        // Generate a sequence of the target length using input as seed
        let sequence = generate_dna_sequence(data, target_len);

        // Encode
        let encoded = encode_dna_prefer_simd(&sequence);

        // Verify encoded length
        let expected_encoded_len = (sequence.len() + 1) / 2;
        assert_eq!(
            encoded.len(),
            expected_encoded_len,
            "Encoded length mismatch at boundary {}",
            target_len
        );

        // Decode
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());

        // Verify decoded length
        assert_eq!(
            decoded.len(),
            sequence.len(),
            "Decoded length mismatch at boundary {}",
            target_len
        );

        // Verify round-trip (normalized)
        for (i, (&input, &output)) in sequence.iter().zip(decoded.iter()).enumerate() {
            let expected = normalize_char(input);
            assert_eq!(
                output, expected,
                "Round-trip failed at boundary {}, position {}",
                target_len, i
            );
        }
    }

    // Also test the exact input length from fuzzer
    if !data.is_empty() {
        let encoded = encode_dna_prefer_simd(data);
        let decoded = decode_dna_prefer_simd(&encoded, data.len());

        for (i, (&input, &output)) in data.iter().zip(decoded.iter()).enumerate() {
            let expected = normalize_char(input);
            assert_eq!(
                output,
                expected,
                "Round-trip failed at fuzzer length {}, position {}",
                data.len(),
                i
            );
        }
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
