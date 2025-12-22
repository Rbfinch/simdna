// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! Fuzz testing for zero-allocation `_into` variants.
//!
//! Verifies that:
//! 1. `_into` variants produce identical results to allocating variants
//! 2. Buffer size validation works correctly
//! 3. All SIMD paths work correctly with caller-provided buffers

#![no_main]

use libfuzzer_sys::fuzz_target;
use simdna::dna_simd_encoder::{
    BufferError, decode_dna_into, decode_dna_prefer_simd, encode_dna_into, encode_dna_prefer_simd,
    required_decoded_len, required_encoded_len, reverse_complement, reverse_complement_encoded,
    reverse_complement_encoded_into, reverse_complement_into,
};

fuzz_target!(|data: &[u8]| {
    // Skip empty input
    if data.is_empty() {
        return;
    }

    let len = data.len();

    // Test 1: encode_dna_into matches encode_dna_prefer_simd
    {
        let expected = encode_dna_prefer_simd(data);
        let needed = required_encoded_len(len);
        assert_eq!(expected.len(), needed, "required_encoded_len mismatch");

        // Test with exact-size buffer
        let mut output = vec![0u8; needed];
        let result = encode_dna_into(data, &mut output);
        assert!(result.is_ok(), "encode_dna_into failed with exact buffer");
        assert_eq!(result.unwrap(), needed);
        assert_eq!(output, expected, "encode_dna_into output mismatch");

        // Test with too-small buffer (if needed > 0)
        if needed > 0 {
            let mut small_output = vec![0u8; needed - 1];
            let result = encode_dna_into(data, &mut small_output);
            assert!(
                result.is_err(),
                "encode_dna_into should fail with small buffer"
            );
            match result {
                Err(BufferError::BufferTooSmall {
                    needed: n,
                    actual: a,
                }) => {
                    assert_eq!(n, needed);
                    assert_eq!(a, needed - 1);
                }
                _ => unreachable!("Expected BufferTooSmall error"),
            }
        }
    }

    // Test 2: decode_dna_into matches decode_dna_prefer_simd
    {
        let encoded = encode_dna_prefer_simd(data);
        let expected = decode_dna_prefer_simd(&encoded, len);
        let needed = required_decoded_len(len);
        assert_eq!(expected.len(), needed, "required_decoded_len mismatch");

        // Test with exact-size buffer
        let mut output = vec![0u8; needed];
        let result = decode_dna_into(&encoded, len, &mut output);
        assert!(result.is_ok(), "decode_dna_into failed with exact buffer");
        assert_eq!(result.unwrap(), needed);
        assert_eq!(output, expected, "decode_dna_into output mismatch");

        // Test with too-small buffer (if needed > 0)
        if needed > 0 {
            let mut small_output = vec![0u8; needed - 1];
            let result = decode_dna_into(&encoded, len, &mut small_output);
            assert!(
                result.is_err(),
                "decode_dna_into should fail with small buffer"
            );
            match result {
                Err(BufferError::BufferTooSmall {
                    needed: n,
                    actual: a,
                }) => {
                    assert_eq!(n, needed);
                    assert_eq!(a, needed - 1);
                }
                _ => unreachable!("Expected BufferTooSmall error"),
            }
        }
    }

    // Test 3: reverse_complement_encoded_into matches reverse_complement_encoded
    {
        let encoded = encode_dna_prefer_simd(data);
        let expected = reverse_complement_encoded(&encoded, len);

        // Test with exact-size buffer
        let mut output = vec![0u8; encoded.len()];
        let result = reverse_complement_encoded_into(&encoded, len, &mut output);
        assert!(
            result.is_ok(),
            "reverse_complement_encoded_into failed with exact buffer"
        );
        assert_eq!(result.unwrap(), encoded.len());
        assert_eq!(
            output, expected,
            "reverse_complement_encoded_into output mismatch"
        );

        // Test with too-small buffer (if encoded.len() > 0)
        if !encoded.is_empty() {
            let mut small_output = vec![0u8; encoded.len() - 1];
            let result = reverse_complement_encoded_into(&encoded, len, &mut small_output);
            assert!(
                result.is_err(),
                "reverse_complement_encoded_into should fail with small buffer"
            );
            match result {
                Err(BufferError::BufferTooSmall {
                    needed: n,
                    actual: a,
                }) => {
                    assert_eq!(n, encoded.len());
                    assert_eq!(a, encoded.len() - 1);
                }
                _ => unreachable!("Expected BufferTooSmall error"),
            }
        }
    }

    // Test 4: reverse_complement_into matches reverse_complement
    {
        let expected = reverse_complement(data);

        // Test with exact-size buffer
        let mut output = vec![0u8; len];
        let result = reverse_complement_into(data, &mut output);
        assert!(
            result.is_ok(),
            "reverse_complement_into failed with exact buffer"
        );
        assert_eq!(result.unwrap(), len);
        assert_eq!(output, expected, "reverse_complement_into output mismatch");

        // Test with too-small buffer (if len > 0)
        if len > 0 {
            let mut small_output = vec![0u8; len - 1];
            let result = reverse_complement_into(data, &mut small_output);
            assert!(
                result.is_err(),
                "reverse_complement_into should fail with small buffer"
            );
            match result {
                Err(BufferError::BufferTooSmall {
                    needed: n,
                    actual: a,
                }) => {
                    assert_eq!(n, len);
                    assert_eq!(a, len - 1);
                }
                _ => unreachable!("Expected BufferTooSmall error"),
            }
        }
    }

    // Test 5: Verify double reverse complement is identity (via _into variants)
    {
        let mut rc1 = vec![0u8; len];
        reverse_complement_into(data, &mut rc1).unwrap();

        let mut rc2 = vec![0u8; len];
        reverse_complement_into(&rc1, &mut rc2).unwrap();

        // The double RC should match the normalized input
        let normalized = decode_dna_prefer_simd(&encode_dna_prefer_simd(data), len);
        assert_eq!(
            rc2, normalized,
            "Double reverse complement should be identity"
        );
    }
});
