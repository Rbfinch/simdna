// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! Bit rotation complement fuzzing.
//!
//! This fuzz target verifies the mathematical properties of the bit rotation
//! complement operation:
//! 1. Involution: complement(complement(x)) == x for all valid 4-bit values
//! 2. Consistency: packed byte complement equals individual nibble complements
//! 3. Known pairs: A↔T, C↔G, R↔Y, K↔M, D↔H, B↔V (and self-complementary codes)

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::{complement_4bit, complement_packed_byte, encoding};

fuzz_target!(|data: &[u8]| {
    for &byte in data {
        // Test each nibble separately
        let high_nibble = (byte >> 4) & 0x0F;
        let low_nibble = byte & 0x0F;

        // Property 1: Involution - complement of complement equals original
        let comp_high = complement_4bit(high_nibble);
        let comp_low = complement_4bit(low_nibble);

        assert_eq!(
            complement_4bit(comp_high),
            high_nibble,
            "Involution failed for high nibble: complement(complement(0x{:X})) = 0x{:X}, expected 0x{:X}",
            high_nibble,
            complement_4bit(comp_high),
            high_nibble
        );

        assert_eq!(
            complement_4bit(comp_low),
            low_nibble,
            "Involution failed for low nibble: complement(complement(0x{:X})) = 0x{:X}, expected 0x{:X}",
            low_nibble,
            complement_4bit(comp_low),
            low_nibble
        );

        // Property 2: Packed byte complement consistency
        // complement_packed_byte should equal applying complement to each nibble
        let packed_comp = complement_packed_byte(byte);
        let expected_packed = (comp_high << 4) | comp_low;

        assert_eq!(
            packed_comp,
            expected_packed,
            "Packed byte complement inconsistent: complement_packed_byte(0x{:02X}) = 0x{:02X}, expected 0x{:02X}",
            byte,
            packed_comp,
            expected_packed
        );

        // Property 3: Bit rotation formula verification
        // complement = ((bits << 2) | (bits >> 2)) & 0xF
        let rotation_high = ((high_nibble << 2) | (high_nibble >> 2)) & 0x0F;
        let rotation_low = ((low_nibble << 2) | (low_nibble >> 2)) & 0x0F;

        assert_eq!(
            comp_high,
            rotation_high,
            "Bit rotation formula failed for 0x{:X}: function returned 0x{:X}, formula gives 0x{:X}",
            high_nibble,
            comp_high,
            rotation_high
        );

        assert_eq!(
            comp_low,
            rotation_low,
            "Bit rotation formula failed for 0x{:X}: function returned 0x{:X}, formula gives 0x{:X}",
            low_nibble,
            comp_low,
            rotation_low
        );
    }

    // Verify known complement pairs using encoding constants
    verify_complement_pair(encoding::A, encoding::T, "A", "T");
    verify_complement_pair(encoding::C, encoding::G, "C", "G");
    verify_complement_pair(encoding::T, encoding::A, "T", "A");
    verify_complement_pair(encoding::G, encoding::C, "G", "C");
    verify_complement_pair(encoding::R, encoding::Y, "R", "Y");
    verify_complement_pair(encoding::Y, encoding::R, "Y", "R");
    verify_complement_pair(encoding::K, encoding::M, "K", "M");
    verify_complement_pair(encoding::M, encoding::K, "M", "K");
    verify_complement_pair(encoding::D, encoding::H, "D", "H");
    verify_complement_pair(encoding::H, encoding::D, "H", "D");
    verify_complement_pair(encoding::B, encoding::V, "B", "V");
    verify_complement_pair(encoding::V, encoding::B, "V", "B");

    // Self-complementary codes
    verify_self_complement(encoding::GAP, "GAP");
    verify_self_complement(encoding::W, "W");
    verify_self_complement(encoding::S, "S");
    verify_self_complement(encoding::N, "N");
});

/// Verify that a complement pair is correct
#[inline]
fn verify_complement_pair(a: u8, b: u8, name_a: &str, name_b: &str) {
    let comp_a = complement_4bit(a);
    assert_eq!(
        comp_a, b,
        "Complement pair failed: complement({}) = 0x{:X}, expected {} (0x{:X})",
        name_a, comp_a, name_b, b
    );
}

/// Verify that a code is self-complementary
#[inline]
fn verify_self_complement(code: u8, name: &str) {
    let comp = complement_4bit(code);
    assert_eq!(
        comp, code,
        "Self-complement failed: complement({}) = 0x{:X}, expected 0x{:X}",
        name, comp, code
    );
}
