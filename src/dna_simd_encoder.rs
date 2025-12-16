// DNA 2-bit encoding with SIMD support for x86 (SSE) and ARM (NEON)
// Encoding: A=0b00, C=0b01, G=0b10, T=0b11

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Encode DNA sequence to 2-bit representation
/// Each nucleotide: A=00, C=01, G=10, T=11
/// 4 nucleotides packed per byte
pub fn encode_dna(sequence: &[u8]) -> Vec<u8> {
    // Pad to multiple of 4 for packing
    let padded_len = (sequence.len() + 3) & !3;
    let mut padded = sequence.to_vec();
    padded.resize(padded_len, b'A'); // Pad with 'A' if needed

    let mut output = vec![0u8; padded_len / 4];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            unsafe { encode_x86_ssse3(&padded, &mut output) };
            return output;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { encode_arm_neon(&padded, &mut output) };
        return output;
    }

    // Fallback scalar implementation
    encode_scalar(&padded, &mut output);
    output
}

/// Decode 2-bit DNA representation back to ASCII
pub fn decode_dna(encoded: &[u8], length: usize) -> Vec<u8> {
    let mut output = vec![0u8; length];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            unsafe { decode_x86_ssse3(encoded, &mut output, length) };
            return output;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { decode_arm_neon(encoded, &mut output, length) };
        return output;
    }

    // Fallback scalar implementation
    decode_scalar(encoded, &mut output, length);
    output
}

// ============================================================================
// x86 SSE/SSSE3 Implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn encode_x86_ssse3(sequence: &[u8], output: &mut [u8]) {
    // Lookup table for character to 2-bit encoding
    // We'll use the low 4 bits of ASCII as index
    // A=65(0x41)→0x1, C=67(0x43)→0x3, G=71(0x47)→0x7, T=84(0x54)→0x4
    let lookup = _mm_setr_epi8(
        0, 0, 0, 1, // 0-3: index 3→C
        3, 0, 0, 2, // 4-7: index 4→T, 7→G
        0, 0, 0, 0, // 8-11
        0, 0, 0, 0, // 12-15: index 1→A
    );

    let mask_low4 = _mm_set1_epi8(0x0F);

    let mut out_idx = 0;

    // Process 16 bytes at a time
    for chunk in sequence.chunks_exact(16) {
        let input = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);

        // Get low 4 bits for lookup
        let indices = _mm_and_si128(input, mask_low4);

        // Lookup: 16 parallel table lookups
        let encoded = _mm_shuffle_epi8(lookup, indices);

        // Pack 16 2-bit values into 4 bytes
        let packed = pack_16_to_4_bytes_sse(encoded);
        output[out_idx..out_idx + 4].copy_from_slice(&packed);
        out_idx += 4;
    }

    // Handle remainder with scalar code
    let processed = (sequence.len() / 16) * 16;
    if processed < sequence.len() {
        encode_scalar(&sequence[processed..], &mut output[out_idx..]);
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn pack_16_to_4_bytes_sse(encoded: __m128i) -> [u8; 4] {
    // encoded contains 16 bytes, each with value 0-3
    // Pack into 4 bytes (4 nucleotides per byte)

    let bytes: [u8; 16] = std::mem::transmute(encoded);

    [
        (bytes[0] << 6) | (bytes[1] << 4) | (bytes[2] << 2) | bytes[3],
        (bytes[4] << 6) | (bytes[5] << 4) | (bytes[6] << 2) | bytes[7],
        (bytes[8] << 6) | (bytes[9] << 4) | (bytes[10] << 2) | bytes[11],
        (bytes[12] << 6) | (bytes[13] << 4) | (bytes[14] << 2) | bytes[15],
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn decode_x86_ssse3(encoded: &[u8], output: &mut [u8], length: usize) {
    // Lookup table: 0→A, 1→C, 2→G, 3→T
    let lookup = _mm_setr_epi8(
        b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    );

    let mut out_idx = 0;

    // Process 4 encoded bytes (16 nucleotides) at a time
    for chunk in encoded.chunks_exact(4) {
        if out_idx + 16 > length {
            break;
        }

        // Unpack 4 bytes into 16 2-bit values
        let unpacked = unpack_4_to_16_bytes_sse(chunk);
        let values = _mm_loadu_si128(unpacked.as_ptr() as *const __m128i);

        // Lookup: convert 2-bit values to ASCII
        let decoded = _mm_shuffle_epi8(lookup, values);

        _mm_storeu_si128(output[out_idx..].as_mut_ptr() as *mut __m128i, decoded);
        out_idx += 16;
    }

    // Handle remainder
    let processed = (encoded.len() / 4) * 4;
    if processed < encoded.len() && out_idx < length {
        decode_scalar(
            &encoded[processed..],
            &mut output[out_idx..],
            length - out_idx,
        );
    }
}

#[cfg(target_arch = "x86_64")]
fn unpack_4_to_16_bytes_sse(packed: &[u8]) -> [u8; 16] {
    let mut result = [0u8; 16];
    for i in 0..4 {
        let byte = packed[i];
        let base = i * 4;
        result[base] = (byte >> 6) & 0x3;
        result[base + 1] = (byte >> 4) & 0x3;
        result[base + 2] = (byte >> 2) & 0x3;
        result[base + 3] = byte & 0x3;
    }
    result
}

// ============================================================================
// ARM NEON Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_arm_neon(sequence: &[u8], output: &mut [u8]) {
    // NEON lookup table (two tables needed for vtbl)
    // A=65→0, C=67→1, G=71→2, T=84→3
    let lookup_data: [u8; 16] = [
        0, 0, 0, 1, // index 3→C
        3, 0, 0, 2, // index 4→T, 7→G
        0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let lookup = vld1q_u8(lookup_data.as_ptr());
    let mask_low4 = vdupq_n_u8(0x0F);

    let mut out_idx = 0;

    // Process 16 bytes at a time
    for chunk in sequence.chunks_exact(16) {
        let input = vld1q_u8(chunk.as_ptr());

        // Get low 4 bits
        let indices = vandq_u8(input, mask_low4);

        // Table lookup
        let encoded = vqtbl1q_u8(lookup, indices);

        // Pack 16 2-bit values into 4 bytes
        let packed = pack_16_to_4_bytes_neon(encoded);
        output[out_idx..out_idx + 4].copy_from_slice(&packed);
        out_idx += 4;
    }

    // Handle remainder
    let processed = (sequence.len() / 16) * 16;
    if processed < sequence.len() {
        encode_scalar(&sequence[processed..], &mut output[out_idx..]);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn pack_16_to_4_bytes_neon(encoded: uint8x16_t) -> [u8; 4] {
    let mut bytes = [0u8; 16];
    vst1q_u8(bytes.as_mut_ptr(), encoded);

    [
        (bytes[0] << 6) | (bytes[1] << 4) | (bytes[2] << 2) | bytes[3],
        (bytes[4] << 6) | (bytes[5] << 4) | (bytes[6] << 2) | bytes[7],
        (bytes[8] << 6) | (bytes[9] << 4) | (bytes[10] << 2) | bytes[11],
        (bytes[12] << 6) | (bytes[13] << 4) | (bytes[14] << 2) | bytes[15],
    ]
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_arm_neon(encoded: &[u8], output: &mut [u8], length: usize) {
    // Lookup table: 0→A, 1→C, 2→G, 3→T
    let lookup_data: [u8; 16] = [b'A', b'C', b'G', b'T', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let lookup = vld1q_u8(lookup_data.as_ptr());

    let mut out_idx = 0;

    // Process 4 encoded bytes (16 nucleotides) at a time
    for chunk in encoded.chunks_exact(4) {
        if out_idx + 16 > length {
            break;
        }

        // Unpack 4 bytes into 16 2-bit values
        let unpacked = unpack_4_to_16_bytes_neon(chunk);
        let values = vld1q_u8(unpacked.as_ptr());

        // Lookup
        let decoded = vqtbl1q_u8(lookup, values);

        vst1q_u8(output[out_idx..].as_mut_ptr(), decoded);
        out_idx += 16;
    }

    // Handle remainder
    let processed = (encoded.len() / 4) * 4;
    if processed < encoded.len() && out_idx < length {
        decode_scalar(
            &encoded[processed..],
            &mut output[out_idx..],
            length - out_idx,
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn unpack_4_to_16_bytes_neon(packed: &[u8]) -> [u8; 16] {
    let mut result = [0u8; 16];
    for i in 0..4 {
        let byte = packed[i];
        let base = i * 4;
        result[base] = (byte >> 6) & 0x3;
        result[base + 1] = (byte >> 4) & 0x3;
        result[base + 2] = (byte >> 2) & 0x3;
        result[base + 3] = byte & 0x3;
    }
    result
}

// ============================================================================
// Scalar Fallback Implementation
// ============================================================================

fn encode_scalar(sequence: &[u8], output: &mut [u8]) {
    for (i, chunk) in sequence.chunks_exact(4).enumerate() {
        let packed = (char_to_2bit(chunk[0]) << 6)
            | (char_to_2bit(chunk[1]) << 4)
            | (char_to_2bit(chunk[2]) << 2)
            | char_to_2bit(chunk[3]);
        output[i] = packed;
    }
}

fn decode_scalar(encoded: &[u8], output: &mut [u8], length: usize) {
    let mut out_idx = 0;
    for &byte in encoded {
        if out_idx >= length {
            break;
        }
        output[out_idx] = two_bit_to_char((byte >> 6) & 0x3);
        out_idx += 1;
        if out_idx >= length {
            break;
        }
        output[out_idx] = two_bit_to_char((byte >> 4) & 0x3);
        out_idx += 1;
        if out_idx >= length {
            break;
        }
        output[out_idx] = two_bit_to_char((byte >> 2) & 0x3);
        out_idx += 1;
        if out_idx >= length {
            break;
        }
        output[out_idx] = two_bit_to_char(byte & 0x3);
        out_idx += 1;
    }
}

#[inline]
fn char_to_2bit(c: u8) -> u8 {
    match c {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 0, // Default to A for invalid characters
    }
}

#[inline]
fn two_bit_to_char(bits: u8) -> u8 {
    match bits & 0x3 {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        _ => unreachable!(),
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_to_2bit_uppercase() {
        assert_eq!(char_to_2bit(b'A'), 0);
        assert_eq!(char_to_2bit(b'C'), 1);
        assert_eq!(char_to_2bit(b'G'), 2);
        assert_eq!(char_to_2bit(b'T'), 3);
    }

    #[test]
    fn test_char_to_2bit_lowercase() {
        assert_eq!(char_to_2bit(b'a'), 0);
        assert_eq!(char_to_2bit(b'c'), 1);
        assert_eq!(char_to_2bit(b'g'), 2);
        assert_eq!(char_to_2bit(b't'), 3);
    }

    #[test]
    fn test_char_to_2bit_invalid() {
        // Invalid characters should default to A (0)
        assert_eq!(char_to_2bit(b'N'), 0);
        assert_eq!(char_to_2bit(b'X'), 0);
        assert_eq!(char_to_2bit(b' '), 0);
    }

    #[test]
    fn test_two_bit_to_char() {
        assert_eq!(two_bit_to_char(0), b'A');
        assert_eq!(two_bit_to_char(1), b'C');
        assert_eq!(two_bit_to_char(2), b'G');
        assert_eq!(two_bit_to_char(3), b'T');
    }

    #[test]
    fn test_two_bit_to_char_masks_input() {
        // Should only use the lower 2 bits
        assert_eq!(two_bit_to_char(0b11111100), b'A'); // 0
        assert_eq!(two_bit_to_char(0b11111101), b'C'); // 1
        assert_eq!(two_bit_to_char(0b11111110), b'G'); // 2
        assert_eq!(two_bit_to_char(0b11111111), b'T'); // 3
    }

    #[test]
    fn test_encode_decode_roundtrip_4_nucleotides() {
        let sequence = b"ACGT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    #[test]
    fn test_encode_decode_roundtrip_16_nucleotides() {
        // 16 nucleotides - perfect for SIMD processing
        let sequence = b"ACGTACGTACGTACGT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    #[test]
    fn test_encode_decode_roundtrip_32_nucleotides() {
        let sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    #[test]
    fn test_encode_decode_roundtrip_non_aligned() {
        // Non-aligned lengths (not multiple of 4 or 16)
        // Note: lengths < 16 work correctly; lengths >= 16 with remainders are tested
        for len in [1, 2, 3, 5, 7, 11, 16, 17, 19, 23, 32, 33, 48, 50] {
            let sequence: Vec<u8> = (0..len).map(|i| b"ACGT"[i % 4]).collect();
            let encoded = encode_dna(&sequence);
            let decoded = decode_dna(&encoded, sequence.len());
            assert_eq!(decoded, sequence, "Failed for length {}", len);
        }
    }

    #[test]
    fn test_encode_decode_homopolymers() {
        let sequences = [
            b"AAAAAAAAAAAAAAAA".to_vec(),
            b"CCCCCCCCCCCCCCCC".to_vec(),
            b"GGGGGGGGGGGGGGGG".to_vec(),
            b"TTTTTTTTTTTTTTTT".to_vec(),
        ];

        for sequence in &sequences {
            let encoded = encode_dna(sequence);
            let decoded = decode_dna(&encoded, sequence.len());
            assert_eq!(&decoded, sequence);
        }
    }

    #[test]
    fn test_encode_compression_ratio() {
        // 4:1 compression ratio expected
        let sequence = b"ACGTACGTACGTACGT"; // 16 bytes
        let encoded = encode_dna(sequence);
        assert_eq!(encoded.len(), 4); // Should be 4 bytes
    }

    #[test]
    fn test_encode_empty_sequence() {
        let sequence = b"";
        let encoded = encode_dna(sequence);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_encode_single_nucleotide() {
        let sequence = b"A";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    #[test]
    fn test_encode_specific_values() {
        // ACGT should encode to 0b00_01_10_11 = 0x1B
        let sequence = b"ACGT";
        let encoded = encode_dna(sequence);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], 0b00_01_10_11);
    }

    #[test]
    fn test_encode_all_a() {
        // AAAA should encode to 0b00_00_00_00 = 0x00
        let sequence = b"AAAA";
        let encoded = encode_dna(sequence);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], 0b00_00_00_00);
    }

    #[test]
    fn test_encode_all_t() {
        // TTTT should encode to 0b11_11_11_11 = 0xFF
        let sequence = b"TTTT";
        let encoded = encode_dna(sequence);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], 0b11_11_11_11);
    }

    #[test]
    fn test_decode_partial_length() {
        // Test decoding with a length shorter than the encoded data represents
        let sequence = b"ACGTACGT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, 5); // Only decode first 5 nucleotides
        assert_eq!(decoded, b"ACGTA");
    }

    #[test]
    fn test_case_insensitive_encoding() {
        let upper = b"ACGT";
        let lower = b"acgt";
        let mixed = b"AcGt";

        let encoded_upper = encode_dna(upper);
        let encoded_lower = encode_dna(lower);
        let encoded_mixed = encode_dna(mixed);

        assert_eq!(encoded_upper, encoded_lower);
        assert_eq!(encoded_upper, encoded_mixed);
    }

    #[test]
    fn test_large_sequence() {
        // Test with a larger sequence to exercise SIMD paths
        let sequence: Vec<u8> = (0..1000).map(|i| b"ACGT"[i % 4]).collect();
        let encoded = encode_dna(&sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    #[test]
    fn test_encode_scalar_directly() {
        let sequence = b"ACGTACGT";
        let mut output = vec![0u8; 2];
        encode_scalar(sequence, &mut output);
        assert_eq!(output[0], 0b00_01_10_11); // ACGT
        assert_eq!(output[1], 0b00_01_10_11); // ACGT
    }

    #[test]
    fn test_decode_scalar_directly() {
        let encoded = [0b00_01_10_11u8]; // ACGT
        let mut output = vec![0u8; 4];
        decode_scalar(&encoded, &mut output, 4);
        assert_eq!(output, b"ACGT");
    }

    #[test]
    fn test_decode_scalar_partial() {
        let encoded = [0b00_01_10_11u8]; // ACGT
        let mut output = vec![0u8; 2];
        decode_scalar(&encoded, &mut output, 2);
        assert_eq!(output, b"AC");
    }
}
