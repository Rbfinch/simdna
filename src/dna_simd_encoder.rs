//! # DNA 4-bit Encoding Library
//!
//! High-performance DNA sequence encoding and decoding using SIMD instructions
//! with automatic fallback to scalar implementations.
//!
//! ## Encoding Scheme
//!
//! DNA nucleotides and IUPAC ambiguity codes are encoded using 4 bits each:
//!
//! | Code | Meaning              | Binary |
//! |------|----------------------|--------|
//! | A    | Adenine              | 0x0    |
//! | C    | Cytosine             | 0x1    |
//! | G    | Guanine              | 0x2    |
//! | T    | Thymine              | 0x3    |
//! | R    | A or G (purine)      | 0x4    |
//! | Y    | C or T (pyrimidine)  | 0x5    |
//! | S    | G or C (strong)      | 0x6    |
//! | W    | A or T (weak)        | 0x7    |
//! | K    | G or T (keto)        | 0x8    |
//! | M    | A or C (amino)       | 0x9    |
//! | B    | C, G, or T (not A)   | 0xA    |
//! | D    | A, G, or T (not C)   | 0xB    |
//! | H    | A, C, or T (not G)   | 0xC    |
//! | V    | A, C, or G (not T)   | 0xD    |
//! | N    | Any base             | 0xE    |
//! | -    | Gap / Unknown        | 0xF    |
//!
//! This allows 2 nucleotides to be packed into a single byte, achieving a 2:1
//! compression ratio compared to ASCII representation while faithfully preserving
//! all IUPAC ambiguity codes.
//!
//! ## Platform Support
//!
//! The library automatically detects and uses the best available SIMD instructions:
//! - **x86_64**: Uses SSE/SSSE3 instructions when available
//! - **aarch64 (ARM64)**: Uses NEON instructions (always available on ARM64)
//! - **Other platforms**: Falls back to optimized scalar implementation
//!
//! ## Thread Safety
//!
//! All public functions in this module are thread-safe and can be safely called
//! from multiple threads concurrently. The functions:
//! - Have no global or static mutable state
//! - Are pure functions (output depends only on input)
//! - Use only thread-safe feature detection (atomic caching)
//!
//! This makes them suitable for use with `rayon`, `std::thread`, or any other
//! multi-threading framework.
//!
//! ## Example Usage
//!
//! ```rust
//! use simdna::dna_simd_encoder::{encode_dna, decode_dna};
//!
//! // Encode a DNA sequence with IUPAC codes
//! let sequence = b"ACGTNRYSWKMBDHV-";
//! let encoded = encode_dna(sequence);
//!
//! // The encoded data is 2x smaller (2 nucleotides per byte)
//! assert_eq!(encoded.len(), sequence.len() / 2);
//!
//! // Decode back to the original sequence
//! let decoded = decode_dna(&encoded, sequence.len());
//! assert_eq!(decoded, sequence);
//! ```
//!
//! ## Case Handling
//!
//! Input sequences are automatically normalized to uppercase before encoding.
//! Both `"ACGT"` and `"acgt"` produce identical encoded output, and decoding
//! always produces uppercase nucleotides.
//!
//! ## Invalid Characters
//!
//! Characters not in the IUPAC nucleotide alphabet are encoded as gap/unknown
//! (`0xF`) and decode back to `-`.
//!
//! ## Performance Considerations
//!
//! - SIMD processing handles 16 nucleotides per iteration
//! - Sequences not divisible by 16 use scalar fallback for remainder
//! - Sequences shorter than 16 nucleotides use the scalar path entirely
//! - Input is padded to a multiple of 2 for byte-aligned packing

// Allow unsafe operations in unsafe functions (Rust 2024 compatibility)
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// 4-bit encoding values for IUPAC nucleotide codes.
/// These constants define the encoding scheme used throughout the library.
pub mod encoding {
    /// Adenine
    pub const A: u8 = 0x0;
    /// Cytosine
    pub const C: u8 = 0x1;
    /// Guanine
    pub const G: u8 = 0x2;
    /// Thymine
    pub const T: u8 = 0x3;
    /// A or G (purine)
    pub const R: u8 = 0x4;
    /// C or T (pyrimidine)
    pub const Y: u8 = 0x5;
    /// G or C (strong)
    pub const S: u8 = 0x6;
    /// A or T (weak)
    pub const W: u8 = 0x7;
    /// G or T (keto)
    pub const K: u8 = 0x8;
    /// A or C (amino)
    pub const M: u8 = 0x9;
    /// C, G, or T (not A)
    pub const B: u8 = 0xA;
    /// A, G, or T (not C)
    pub const D: u8 = 0xB;
    /// A, C, or T (not G)
    pub const H: u8 = 0xC;
    /// A, C, or G (not T)
    pub const V: u8 = 0xD;
    /// Any base
    pub const N: u8 = 0xE;
    /// Gap / Unknown
    pub const GAP: u8 = 0xF;
}

/// Encodes a DNA sequence into a compact 4-bit representation.
///
/// Each nucleotide or IUPAC ambiguity code is encoded using 4 bits,
/// with 2 nucleotides packed per byte.
///
/// # Encoding Table
///
/// | Input | Encoding | Input | Encoding |
/// |-------|----------|-------|----------|
/// | A     | 0x0      | K     | 0x8      |
/// | C     | 0x1      | M     | 0x9      |
/// | G     | 0x2      | B     | 0xA      |
/// | T     | 0x3      | D     | 0xB      |
/// | R     | 0x4      | H     | 0xC      |
/// | Y     | 0x5      | V     | 0xD      |
/// | S     | 0x6      | N     | 0xE      |
/// | W     | 0x7      | -/.   | 0xF      |
///
/// # Arguments
///
/// * `sequence` - A byte slice containing ASCII-encoded DNA nucleotides.
///   Both uppercase and lowercase letters are accepted.
///
/// # Returns
///
/// A `Vec<u8>` containing the encoded sequence. The length will be
/// `ceil(sequence.len() / 2)` bytes.
///
/// # Input Handling
///
/// - **Case insensitivity**: Input is normalized to uppercase before encoding.
/// - **Padding**: Sequences with odd length are padded with gap values.
/// - **Invalid characters**: Non-IUPAC characters are encoded as gap (`0xF`).
///
/// # Performance
///
/// This function automatically uses SIMD instructions when available:
/// - x86_64 with SSSE3: Processes 16 nucleotides per iteration
/// - ARM64 (aarch64): Uses NEON to process 16 nucleotides per iteration
/// - Other platforms: Uses optimized scalar implementation
///
/// # Thread Safety
///
/// This function is thread-safe and can be called from multiple threads
/// concurrently without synchronization.
///
/// # Examples
///
/// Basic encoding:
/// ```rust
/// use simdna::dna_simd_encoder::encode_dna;
///
/// let encoded = encode_dna(b"ACGT");
/// // A=0x0, C=0x1, G=0x2, T=0x3
/// // Packed: (A<<4|C), (G<<4|T) = 0x01, 0x23
/// assert_eq!(encoded, vec![0x01, 0x23]);
/// ```
///
/// IUPAC ambiguity codes are preserved:
/// ```rust
/// use simdna::dna_simd_encoder::{encode_dna, decode_dna};
///
/// let sequence = b"ACNGT";
/// let encoded = encode_dna(sequence);
/// let decoded = decode_dna(&encoded, sequence.len());
/// assert_eq!(decoded, b"ACNGT");
/// ```
///
/// Compression ratio:
/// ```rust
/// use simdna::dna_simd_encoder::encode_dna;
///
/// let sequence = b"ACGTACGTACGTACGT"; // 16 bytes
/// let encoded = encode_dna(sequence);
/// assert_eq!(encoded.len(), 8); // 8 bytes (2:1 compression)
/// ```
pub fn encode_dna(sequence: &[u8]) -> Vec<u8> {
    // Normalize to uppercase for consistent encoding
    let normalized: Vec<u8> = sequence.iter().map(|&c| c.to_ascii_uppercase()).collect();

    // Pad to multiple of 2 for packing
    let padded_len = (normalized.len() + 1) & !1;
    let mut padded = normalized;
    padded.resize(padded_len, b'-'); // Pad with gap if needed

    let mut output = vec![0u8; padded_len / 2];

    // Try SIMD implementations first, fall back to scalar
    let used_simd = encode_with_simd_if_available(&padded, &mut output);

    if !used_simd {
        // Fallback scalar implementation for unsupported architectures
        encode_scalar(&padded, &mut output);
    }

    output
}

/// Attempts to encode using SIMD instructions if available.
///
/// This is an internal helper function that dispatches to the appropriate
/// SIMD implementation based on the detected CPU features.
///
/// # Returns
///
/// `true` if SIMD encoding was used, `false` if caller should use scalar fallback.
#[inline]
fn encode_with_simd_if_available(padded: &[u8], output: &mut [u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            unsafe { encode_x86_ssse3(padded, output) };
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { encode_arm_neon(padded, output) };
        return true;
    }

    #[allow(unreachable_code)]
    false
}

/// Decodes a 4-bit encoded DNA sequence back to ASCII nucleotides.
///
/// Converts the compact 4-bit representation back to ASCII bytes,
/// where each 4-bit value maps to its corresponding IUPAC nucleotide code.
///
/// # Decoding Table
///
/// | Encoding | Output | Encoding | Output |
/// |----------|--------|----------|--------|
/// | 0x0      | A      | 0x8      | K      |
/// | 0x1      | C      | 0x9      | M      |
/// | 0x2      | G      | 0xA      | B      |
/// | 0x3      | T      | 0xB      | D      |
/// | 0x4      | R      | 0xC      | H      |
/// | 0x5      | Y      | 0xD      | V      |
/// | 0x6      | S      | 0xE      | N      |
/// | 0x7      | W      | 0xF      | -      |
///
/// # Arguments
///
/// * `encoded` - A byte slice containing the 4-bit encoded DNA data,
///   as produced by [`encode_dna`].
/// * `length` - The number of nucleotides to decode. This should match
///   the original sequence length before encoding.
///
/// # Returns
///
/// A `Vec<u8>` containing uppercase ASCII nucleotides and IUPAC codes.
///
/// # Important Notes
///
/// - The `length` parameter is required because the encoded data may
///   have been padded during encoding. Specifying the original length
///   ensures no padding bytes are included in the output.
/// - If `length` exceeds the capacity of `encoded` (i.e., `encoded.len() * 2`),
///   the output will be truncated to the available data.
/// - Output is always uppercase, regardless of the original input case.
///
/// # Performance
///
/// This function automatically uses SIMD instructions when available:
/// - x86_64 with SSSE3: Processes 16 nucleotides per iteration
/// - ARM64 (aarch64): Uses NEON to process 16 nucleotides per iteration
/// - Other platforms: Uses optimized scalar implementation
///
/// # Thread Safety
///
/// This function is thread-safe and can be called from multiple threads
/// concurrently without synchronization.
///
/// # Examples
///
/// Basic decoding:
/// ```rust
/// use simdna::dna_simd_encoder::{encode_dna, decode_dna};
///
/// let original = b"ACGTRYSWKMBDHVN-";
/// let encoded = encode_dna(original);
/// let decoded = decode_dna(&encoded, original.len());
/// assert_eq!(decoded, original);
/// ```
///
/// Partial decoding:
/// ```rust
/// use simdna::dna_simd_encoder::{encode_dna, decode_dna};
///
/// let original = b"ACGTACGT";
/// let encoded = encode_dna(original);
/// let first_four = decode_dna(&encoded, 4);
/// assert_eq!(first_four, b"ACGT");
/// ```
pub fn decode_dna(encoded: &[u8], length: usize) -> Vec<u8> {
    let mut output = vec![0u8; length];

    // Try SIMD implementations first, fall back to scalar
    let used_simd = decode_with_simd_if_available(encoded, &mut output, length);

    if !used_simd {
        // Fallback scalar implementation for unsupported architectures
        decode_scalar(encoded, &mut output, length);
    }

    output
}

/// Attempts to decode using SIMD instructions if available.
///
/// This is an internal helper function that dispatches to the appropriate
/// SIMD implementation based on the detected CPU features.
///
/// # Returns
///
/// `true` if SIMD decoding was used, `false` if caller should use scalar fallback.
#[inline]
fn decode_with_simd_if_available(encoded: &[u8], output: &mut [u8], length: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            unsafe { decode_x86_ssse3(encoded, output, length) };
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { decode_arm_neon(encoded, output, length) };
        return true;
    }

    #[allow(unreachable_code)]
    false
}

// ============================================================================
// x86 SSE/SSSE3 Implementation
// ============================================================================

/// SSSE3-accelerated DNA encoding for x86_64.
///
/// Processes 16 nucleotides at a time using SIMD instructions:
/// 1. Loads 16 ASCII bytes into an SSE register
/// 2. Converts each byte to 4-bit encoding
/// 3. Packs 16 4-bit values into 8 bytes
///
/// Remainder nucleotides (less than 16) are handled by scalar fallback.
///
/// # Safety
///
/// This function requires SSSE3 support. Caller must verify via
/// `is_x86_feature_detected!("ssse3")` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn encode_x86_ssse3(sequence: &[u8], output: &mut [u8]) {
    let mut out_idx = 0;

    // Process 16 bytes at a time
    for chunk in sequence.chunks_exact(16) {
        // Convert to 4-bit encoding using scalar lookup for each byte
        // (SIMD lookup tables for full ASCII range are complex, so we use
        // a hybrid approach: load via SIMD, encode with scalar, pack with SIMD)
        let mut encoded_bytes = [0u8; 16];
        for (i, &byte) in chunk.iter().enumerate() {
            encoded_bytes[i] = char_to_4bit(byte);
        }

        // Pack 16 4-bit values into 8 bytes
        let encoded = _mm_loadu_si128(encoded_bytes.as_ptr() as *const __m128i);
        let packed = pack_16_to_8_bytes_sse(encoded);
        output[out_idx..out_idx + 8].copy_from_slice(&packed);
        out_idx += 8;
    }

    // Handle remainder with scalar code
    let processed = (sequence.len() / 16) * 16;
    if processed < sequence.len() {
        encode_scalar(&sequence[processed..], &mut output[out_idx..]);
    }
}

/// Packs 16 4-bit encoded values into 8 bytes for x86.
///
/// Takes an SSE register containing 16 bytes (each with value 0-15)
/// and packs them into 8 bytes with 2 nucleotides per byte.
///
/// # Safety
///
/// Requires valid SSE register input.
#[cfg(target_arch = "x86_64")]
unsafe fn pack_16_to_8_bytes_sse(encoded: __m128i) -> [u8; 8] {
    let bytes: [u8; 16] = std::mem::transmute(encoded);

    [
        (bytes[0] << 4) | bytes[1],
        (bytes[2] << 4) | bytes[3],
        (bytes[4] << 4) | bytes[5],
        (bytes[6] << 4) | bytes[7],
        (bytes[8] << 4) | bytes[9],
        (bytes[10] << 4) | bytes[11],
        (bytes[12] << 4) | bytes[13],
        (bytes[14] << 4) | bytes[15],
    ]
}

/// SSSE3-accelerated DNA decoding for x86_64.
///
/// Processes 8 encoded bytes (16 nucleotides) at a time:
/// 1. Unpacks 8 bytes into 16 4-bit values
/// 2. Uses `pshufb` for parallel table lookup to convert to ASCII
/// 3. Stores 16 ASCII nucleotides
///
/// Remainder nucleotides (when length is not a multiple of 16) are
/// handled by scalar fallback.
///
/// # Safety
///
/// This function requires SSSE3 support. Caller must verify via
/// `is_x86_feature_detected!("ssse3")` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn decode_x86_ssse3(encoded: &[u8], output: &mut [u8], length: usize) {
    // Lookup table for 4-bit to ASCII conversion
    // Index: 0=A, 1=C, 2=G, 3=T, 4=R, 5=Y, 6=S, 7=W, 8=K, 9=M, A=B, B=D, C=H, D=V, E=N, F=-
    let lookup = _mm_setr_epi8(
        b'A' as i8, b'C' as i8, b'G' as i8, b'T' as i8, b'R' as i8, b'Y' as i8, b'S' as i8,
        b'W' as i8, b'K' as i8, b'M' as i8, b'B' as i8, b'D' as i8, b'H' as i8, b'V' as i8,
        b'N' as i8, b'-' as i8,
    );

    let mut out_idx = 0;
    let mut enc_idx = 0;

    // Process 8 encoded bytes (16 nucleotides) at a time
    for chunk in encoded.chunks_exact(8) {
        if out_idx + 16 > length {
            break;
        }

        // Unpack 8 bytes into 16 4-bit values
        let unpacked = unpack_8_to_16_bytes(chunk);
        let values = _mm_loadu_si128(unpacked.as_ptr() as *const __m128i);

        // Lookup: convert 4-bit values to ASCII using SIMD shuffle
        let decoded = _mm_shuffle_epi8(lookup, values);

        _mm_storeu_si128(output[out_idx..].as_mut_ptr() as *mut __m128i, decoded);
        out_idx += 16;
        enc_idx += 8;
    }

    // Handle remainder
    if out_idx < length {
        decode_scalar(
            &encoded[enc_idx..],
            &mut output[out_idx..],
            length - out_idx,
        );
    }
}

/// Unpacks 8 encoded bytes into 16 4-bit values.
///
/// Each input byte contains 2 nucleotides (8 bits / 4 bits each).
/// This function extracts each 4-bit value into its own byte for
/// subsequent SIMD lookup.
fn unpack_8_to_16_bytes(packed: &[u8]) -> [u8; 16] {
    let mut result = [0u8; 16];
    for (i, &byte) in packed.iter().enumerate().take(8) {
        let base = i * 2;
        result[base] = (byte >> 4) & 0xF;
        result[base + 1] = byte & 0xF;
    }
    result
}

// ============================================================================
// ARM NEON Implementation
// ============================================================================

/// NEON-accelerated DNA encoding for ARM64.
///
/// Processes 16 nucleotides at a time using NEON vector instructions:
/// 1. Loads 16 ASCII bytes into a NEON register
/// 2. Converts each byte to 4-bit encoding
/// 3. Packs 16 4-bit values into 8 bytes
///
/// Remainder nucleotides (less than 16) are handled by scalar fallback.
///
/// # Safety
///
/// This function requires NEON support (always available on ARM64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_arm_neon(sequence: &[u8], output: &mut [u8]) {
    let mut out_idx = 0;

    // Process 16 bytes at a time using SIMD
    for chunk in sequence.chunks_exact(16) {
        // Convert to 4-bit encoding using scalar lookup
        let mut encoded_bytes = [0u8; 16];
        for (i, &byte) in chunk.iter().enumerate() {
            encoded_bytes[i] = char_to_4bit(byte);
        }

        // Pack 16 4-bit values into 8 bytes
        let encoded = vld1q_u8(encoded_bytes.as_ptr());
        let packed = pack_16_to_8_bytes_neon(encoded);
        output[out_idx..out_idx + 8].copy_from_slice(&packed);
        out_idx += 8;
    }

    // Handle remainder with scalar code
    let processed = (sequence.len() / 16) * 16;
    if processed < sequence.len() {
        encode_scalar(&sequence[processed..], &mut output[out_idx..]);
    }
}

/// Packs 16 4-bit encoded values into 8 bytes for ARM64.
///
/// Takes a NEON register containing 16 bytes (each with value 0-15)
/// and packs them into 8 bytes with 2 nucleotides per byte.
///
/// # Safety
///
/// Requires valid NEON register input.
#[cfg(target_arch = "aarch64")]
unsafe fn pack_16_to_8_bytes_neon(encoded: uint8x16_t) -> [u8; 8] {
    let mut bytes = [0u8; 16];
    vst1q_u8(bytes.as_mut_ptr(), encoded);

    [
        (bytes[0] << 4) | bytes[1],
        (bytes[2] << 4) | bytes[3],
        (bytes[4] << 4) | bytes[5],
        (bytes[6] << 4) | bytes[7],
        (bytes[8] << 4) | bytes[9],
        (bytes[10] << 4) | bytes[11],
        (bytes[12] << 4) | bytes[13],
        (bytes[14] << 4) | bytes[15],
    ]
}

/// NEON-accelerated DNA decoding for ARM64.
///
/// Processes 8 encoded bytes (16 nucleotides) at a time:
/// 1. Unpacks 8 bytes into 16 4-bit values
/// 2. Uses `vqtbl1q_u8` for parallel table lookup to convert to ASCII
/// 3. Stores 16 ASCII nucleotides
///
/// Remainder nucleotides (when length is not a multiple of 16) are
/// handled by scalar fallback.
///
/// # Safety
///
/// This function requires NEON support (always available on ARM64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_arm_neon(encoded: &[u8], output: &mut [u8], length: usize) {
    // Lookup table for 4-bit to ASCII conversion
    let lookup_data: [u8; 16] = [
        b'A', b'C', b'G', b'T', b'R', b'Y', b'S', b'W', b'K', b'M', b'B', b'D', b'H', b'V', b'N',
        b'-',
    ];
    let lookup = vld1q_u8(lookup_data.as_ptr());

    let mut out_idx = 0;
    let mut enc_idx = 0;

    // Process 8 encoded bytes (16 nucleotides) at a time
    for chunk in encoded.chunks_exact(8) {
        if out_idx + 16 > length {
            break;
        }

        // Unpack 8 bytes into 16 4-bit values
        let unpacked = unpack_8_to_16_bytes(chunk);
        let values = vld1q_u8(unpacked.as_ptr());

        // Lookup: convert 4-bit values to ASCII
        let decoded = vqtbl1q_u8(lookup, values);

        vst1q_u8(output[out_idx..].as_mut_ptr(), decoded);
        out_idx += 16;
        enc_idx += 8;
    }

    // Handle remainder with scalar code
    if out_idx < length {
        decode_scalar(
            &encoded[enc_idx..],
            &mut output[out_idx..],
            length - out_idx,
        );
    }
}

// ============================================================================
// Scalar Fallback Implementation
// ============================================================================

/// Scalar (non-SIMD) DNA encoding fallback.
///
/// Processes 2 nucleotides at a time, packing them into a single byte.
/// This function is used:
/// - On platforms without SIMD support
/// - For remainder nucleotides when sequence length is not a multiple of 16
///
/// # Arguments
///
/// * `sequence` - Slice of uppercase ASCII nucleotide bytes
/// * `output` - Pre-allocated output buffer for encoded bytes
///
/// # Panics
///
/// Panics if `output` is too small to hold the encoded result.
fn encode_scalar(sequence: &[u8], output: &mut [u8]) {
    for (i, chunk) in sequence.chunks_exact(2).enumerate() {
        let packed = (char_to_4bit(chunk[0]) << 4) | char_to_4bit(chunk[1]);
        output[i] = packed;
    }
}

/// Scalar (non-SIMD) DNA decoding fallback.
///
/// Unpacks encoded bytes back into ASCII nucleotide characters.
/// Processes each encoded byte to extract 2 nucleotides (4 bits each).
/// This function is used:
/// - On platforms without SIMD support
/// - For remainder nucleotides when decoded length is not a multiple of 16
///
/// # Arguments
///
/// * `encoded` - Slice of encoded bytes (2 nucleotides per byte)
/// * `output` - Pre-allocated output buffer for decoded ASCII bytes
/// * `length` - Number of nucleotides to decode (may be less than encoded capacity)
///
/// # Behavior
///
/// Stops decoding when `length` nucleotides have been written, handling
/// cases where the original sequence length was not a multiple of 2.
fn decode_scalar(encoded: &[u8], output: &mut [u8], length: usize) {
    let mut out_idx = 0;
    for &byte in encoded {
        if out_idx >= length {
            break;
        }
        output[out_idx] = fourbit_to_char((byte >> 4) & 0xF);
        out_idx += 1;
        if out_idx >= length {
            break;
        }
        output[out_idx] = fourbit_to_char(byte & 0xF);
        out_idx += 1;
    }
}

/// Converts an ASCII nucleotide byte to its 4-bit encoding.
///
/// Handles all IUPAC nucleotide codes. Invalid characters are
/// encoded as gap (0xF).
///
/// # Encoding Table
///
/// | Input | Output | Input | Output |
/// |-------|--------|-------|--------|
/// | A, a  | 0x0    | K, k  | 0x8    |
/// | C, c  | 0x1    | M, m  | 0x9    |
/// | G, g  | 0x2    | B, b  | 0xA    |
/// | T, t  | 0x3    | D, d  | 0xB    |
/// | U, u  | 0x3    | H, h  | 0xC    |
/// | R, r  | 0x4    | V, v  | 0xD    |
/// | Y, y  | 0x5    | N, n  | 0xE    |
/// | S, s  | 0x6    | -, .  | 0xF    |
/// | W, w  | 0x7    | other | 0xF    |
///
/// # Arguments
///
/// * `c` - ASCII byte representing a nucleotide
///
/// # Returns
///
/// 4-bit encoding (0-15)
#[inline]
fn char_to_4bit(c: u8) -> u8 {
    match c {
        b'A' | b'a' => encoding::A,
        b'C' | b'c' => encoding::C,
        b'G' | b'g' => encoding::G,
        b'T' | b't' | b'U' | b'u' => encoding::T, // U (RNA) maps to T
        b'R' | b'r' => encoding::R,
        b'Y' | b'y' => encoding::Y,
        b'S' | b's' => encoding::S,
        b'W' | b'w' => encoding::W,
        b'K' | b'k' => encoding::K,
        b'M' | b'm' => encoding::M,
        b'B' | b'b' => encoding::B,
        b'D' | b'd' => encoding::D,
        b'H' | b'h' => encoding::H,
        b'V' | b'v' => encoding::V,
        b'N' | b'n' => encoding::N,
        b'-' | b'.' => encoding::GAP,
        _ => encoding::GAP, // Unknown characters become gaps
    }
}

/// Converts a 4-bit encoding back to its ASCII nucleotide character.
///
/// # Decoding Table
///
/// | Input | Output | Input | Output |
/// |-------|--------|-------|--------|
/// | 0x0   | A      | 0x8   | K      |
/// | 0x1   | C      | 0x9   | M      |
/// | 0x2   | G      | 0xA   | B      |
/// | 0x3   | T      | 0xB   | D      |
/// | 0x4   | R      | 0xC   | H      |
/// | 0x5   | Y      | 0xD   | V      |
/// | 0x6   | S      | 0xE   | N      |
/// | 0x7   | W      | 0xF   | -      |
///
/// # Arguments
///
/// * `bits` - 4-bit encoded value (only lower 4 bits are used)
///
/// # Returns
///
/// ASCII byte for the nucleotide character
#[inline]
fn fourbit_to_char(bits: u8) -> u8 {
    const DECODE_TABLE: [u8; 16] = [
        b'A', b'C', b'G', b'T', b'R', b'Y', b'S', b'W', b'K', b'M', b'B', b'D', b'H', b'V', b'N',
        b'-',
    ];
    DECODE_TABLE[(bits & 0xF) as usize]
}

// ============================================================================
// Unit Tests
// ============================================================================

/// Unit tests for the DNA SIMD encoder/decoder module.
///
/// The test suite is organized into the following categories:
///
/// ## Core Conversion Tests
/// - `test_char_to_4bit_*` - Tests for the character-to-4bit conversion function
/// - `test_fourbit_to_char*` - Tests for the 4bit-to-character conversion function
///
/// ## Encode/Decode Roundtrip Tests
/// - `test_encode_decode_roundtrip_*` - Verify that encoding followed by decoding
///   produces the original sequence for various lengths
///
/// ## IUPAC Ambiguity Code Tests
/// - `test_iupac_*` - Tests for all IUPAC ambiguous nucleotide codes
///
/// ## Decode-Specific Tests
/// - `test_decode_*` - Tests for decoding 4-bit encoded data back to ASCII
/// - Tests verify correct SIMD and scalar paths, boundary conditions, and IUPAC codes
///
/// ## Case Handling Tests
/// - `test_lowercase_*` - Verify that lowercase input is handled correctly
/// - `test_mixed_case_*` - Verify mixed case sequences work correctly
///
/// ## Thread Safety Tests
/// - `test_*_thread_*` - Concurrent encoding/decoding tests verifying thread safety
#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Core Conversion Function Tests
    // ========================================================================

    /// Tests standard nucleotide to 4-bit conversion.
    #[test]
    fn test_char_to_4bit_standard_bases() {
        assert_eq!(char_to_4bit(b'A'), 0x0);
        assert_eq!(char_to_4bit(b'C'), 0x1);
        assert_eq!(char_to_4bit(b'G'), 0x2);
        assert_eq!(char_to_4bit(b'T'), 0x3);
    }

    /// Tests lowercase nucleotide to 4-bit conversion.
    #[test]
    fn test_char_to_4bit_lowercase() {
        assert_eq!(char_to_4bit(b'a'), 0x0);
        assert_eq!(char_to_4bit(b'c'), 0x1);
        assert_eq!(char_to_4bit(b'g'), 0x2);
        assert_eq!(char_to_4bit(b't'), 0x3);
    }

    /// Tests IUPAC ambiguity codes to 4-bit conversion.
    #[test]
    fn test_char_to_4bit_iupac() {
        assert_eq!(char_to_4bit(b'R'), 0x4);
        assert_eq!(char_to_4bit(b'Y'), 0x5);
        assert_eq!(char_to_4bit(b'S'), 0x6);
        assert_eq!(char_to_4bit(b'W'), 0x7);
        assert_eq!(char_to_4bit(b'K'), 0x8);
        assert_eq!(char_to_4bit(b'M'), 0x9);
        assert_eq!(char_to_4bit(b'B'), 0xA);
        assert_eq!(char_to_4bit(b'D'), 0xB);
        assert_eq!(char_to_4bit(b'H'), 0xC);
        assert_eq!(char_to_4bit(b'V'), 0xD);
        assert_eq!(char_to_4bit(b'N'), 0xE);
    }

    /// Tests gap and unknown characters.
    #[test]
    fn test_char_to_4bit_gaps() {
        assert_eq!(char_to_4bit(b'-'), 0xF);
        assert_eq!(char_to_4bit(b'.'), 0xF);
        assert_eq!(char_to_4bit(b'X'), 0xF); // Invalid -> gap
        assert_eq!(char_to_4bit(b' '), 0xF); // Invalid -> gap
    }

    /// Tests RNA uracil to 4-bit conversion (maps to T).
    #[test]
    fn test_char_to_4bit_uracil() {
        assert_eq!(char_to_4bit(b'U'), 0x3);
        assert_eq!(char_to_4bit(b'u'), 0x3);
    }

    /// Tests 4-bit to ASCII character conversion.
    #[test]
    fn test_fourbit_to_char() {
        assert_eq!(fourbit_to_char(0x0), b'A');
        assert_eq!(fourbit_to_char(0x1), b'C');
        assert_eq!(fourbit_to_char(0x2), b'G');
        assert_eq!(fourbit_to_char(0x3), b'T');
        assert_eq!(fourbit_to_char(0x4), b'R');
        assert_eq!(fourbit_to_char(0x5), b'Y');
        assert_eq!(fourbit_to_char(0x6), b'S');
        assert_eq!(fourbit_to_char(0x7), b'W');
        assert_eq!(fourbit_to_char(0x8), b'K');
        assert_eq!(fourbit_to_char(0x9), b'M');
        assert_eq!(fourbit_to_char(0xA), b'B');
        assert_eq!(fourbit_to_char(0xB), b'D');
        assert_eq!(fourbit_to_char(0xC), b'H');
        assert_eq!(fourbit_to_char(0xD), b'V');
        assert_eq!(fourbit_to_char(0xE), b'N');
        assert_eq!(fourbit_to_char(0xF), b'-');
    }

    /// Tests that only the lower 4 bits are used in conversion.
    #[test]
    fn test_fourbit_to_char_masks_input() {
        // Values > 15 should use only lower 4 bits
        assert_eq!(fourbit_to_char(0x10), b'A'); // 0x10 & 0xF = 0
        assert_eq!(fourbit_to_char(0xFF), b'-'); // 0xFF & 0xF = 15
    }

    // ========================================================================
    // Encode/Decode Roundtrip Tests
    // ========================================================================

    /// Tests roundtrip for standard bases.
    #[test]
    fn test_encode_decode_roundtrip_standard_bases() {
        let sequence = b"ACGT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests roundtrip for all IUPAC codes.
    #[test]
    fn test_encode_decode_roundtrip_all_iupac() {
        let sequence = b"ACGTRYSWKMBDHVN-";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests roundtrip for 16 nucleotides (SIMD boundary).
    #[test]
    fn test_encode_decode_roundtrip_16_nucleotides() {
        let sequence = b"ACGTACGTACGTACGT";
        let encoded = encode_dna(sequence);
        assert_eq!(encoded.len(), 8); // 16 / 2 = 8
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests roundtrip for 32 nucleotides (2 SIMD blocks).
    #[test]
    fn test_encode_decode_roundtrip_32_nucleotides() {
        let sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
        let encoded = encode_dna(sequence);
        assert_eq!(encoded.len(), 16);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests roundtrip for various non-aligned lengths.
    #[test]
    fn test_encode_decode_roundtrip_non_aligned() {
        for len in [1, 2, 3, 5, 7, 11, 15, 16, 17, 19, 23, 31, 32, 33, 48, 50] {
            let sequence: Vec<u8> = (0..len).map(|i| b"ACGT"[i % 4]).collect();
            let encoded = encode_dna(&sequence);
            let decoded = decode_dna(&encoded, sequence.len());
            assert_eq!(decoded, sequence, "Failed for length {}", len);
        }
    }

    /// Tests compression ratio (2:1).
    #[test]
    fn test_encode_compression_ratio() {
        let sequence = b"ACGTACGTACGTACGT"; // 16 bytes
        let encoded = encode_dna(sequence);
        assert_eq!(encoded.len(), 8); // 2:1 compression
    }

    /// Tests empty sequence handling.
    #[test]
    fn test_encode_empty_sequence() {
        let sequence = b"";
        let encoded = encode_dna(sequence);
        assert!(encoded.is_empty());
    }

    /// Tests single nucleotide encoding/decoding.
    #[test]
    fn test_encode_single_nucleotide() {
        for &base in b"ACGTRYSWKMBDHVN-".iter() {
            let sequence = [base];
            let encoded = encode_dna(&sequence);
            let decoded = decode_dna(&encoded, 1);
            assert_eq!(decoded[0], base, "Failed for base {}", base as char);
        }
    }

    /// Tests specific encoding values.
    #[test]
    fn test_encode_specific_values() {
        // AC should encode to 0x01 (A=0x0 in high nibble, C=0x1 in low nibble)
        let sequence = b"AC";
        let encoded = encode_dna(sequence);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], 0x01);

        // GT should encode to 0x23
        let sequence = b"GT";
        let encoded = encode_dna(sequence);
        assert_eq!(encoded[0], 0x23);
    }

    // ========================================================================
    // IUPAC Ambiguity Code Tests
    // ========================================================================

    /// Tests that all IUPAC codes are faithfully preserved.
    #[test]
    fn test_iupac_codes_preserved() {
        let codes = b"ACGTRYSWKMBDHVN-";
        let encoded = encode_dna(codes);
        let decoded = decode_dna(&encoded, codes.len());
        assert_eq!(decoded, codes);
    }

    /// Tests IUPAC codes in longer sequences.
    #[test]
    fn test_iupac_in_long_sequence() {
        // Mix of standard and IUPAC codes
        let sequence = b"ACGTNNNNACGTRYSWACGTKMBDHVNACGT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests N codes at SIMD boundary.
    #[test]
    fn test_iupac_n_at_simd_boundary() {
        let sequence = b"NNNNNNNNNNNNNNNN"; // 16 N's
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests all IUPAC codes in lowercase.
    #[test]
    fn test_iupac_lowercase() {
        let sequence = b"acgtryswkmbdhvn";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        // Decoded should be uppercase
        assert_eq!(decoded, b"ACGTRYSWKMBDHVN");
    }

    /// Tests specific encoding values for each IUPAC ambiguity code.
    #[test]
    fn test_iupac_specific_encoding_values() {
        // Each pair should encode to a specific byte value
        // Format: (high_nibble << 4) | low_nibble

        // RY should encode to 0x45 (R=0x4, Y=0x5)
        let encoded = encode_dna(b"RY");
        assert_eq!(encoded[0], 0x45, "RY should encode to 0x45");

        // SW should encode to 0x67 (S=0x6, W=0x7)
        let encoded = encode_dna(b"SW");
        assert_eq!(encoded[0], 0x67, "SW should encode to 0x67");

        // KM should encode to 0x89 (K=0x8, M=0x9)
        let encoded = encode_dna(b"KM");
        assert_eq!(encoded[0], 0x89, "KM should encode to 0x89");

        // BD should encode to 0xAB (B=0xA, D=0xB)
        let encoded = encode_dna(b"BD");
        assert_eq!(encoded[0], 0xAB, "BD should encode to 0xAB");

        // HV should encode to 0xCD (H=0xC, V=0xD)
        let encoded = encode_dna(b"HV");
        assert_eq!(encoded[0], 0xCD, "HV should encode to 0xCD");

        // N- should encode to 0xEF (N=0xE, -=0xF)
        let encoded = encode_dna(b"N-");
        assert_eq!(encoded[0], 0xEF, "N- should encode to 0xEF");
    }

    /// Tests each IUPAC code individually at odd positions (second nibble).
    #[test]
    fn test_iupac_at_odd_positions() {
        let codes: [(u8, u8); 12] = [
            (b'R', 0x4),
            (b'Y', 0x5),
            (b'S', 0x6),
            (b'W', 0x7),
            (b'K', 0x8),
            (b'M', 0x9),
            (b'B', 0xA),
            (b'D', 0xB),
            (b'H', 0xC),
            (b'V', 0xD),
            (b'N', 0xE),
            (b'-', 0xF),
        ];

        for (code, expected_nibble) in codes {
            // Put code in second position (low nibble)
            // A encodes to 0x0, so expected_byte is just the low nibble value
            let sequence = [b'A', code];
            let encoded = encode_dna(&sequence);
            let expected_byte = expected_nibble; // A (0x0) in high nibble
            assert_eq!(
                encoded[0], expected_byte,
                "A{} should encode to 0x{:02X}",
                code as char, expected_byte
            );
        }
    }

    /// Tests each IUPAC code individually at even positions (first nibble).
    #[test]
    fn test_iupac_at_even_positions() {
        let codes: [(u8, u8); 12] = [
            (b'R', 0x4),
            (b'Y', 0x5),
            (b'S', 0x6),
            (b'W', 0x7),
            (b'K', 0x8),
            (b'M', 0x9),
            (b'B', 0xA),
            (b'D', 0xB),
            (b'H', 0xC),
            (b'V', 0xD),
            (b'N', 0xE),
            (b'-', 0xF),
        ];

        for (code, expected_nibble) in codes {
            // Put code in first position (high nibble)
            // A encodes to 0x0, so expected_byte is just the shifted nibble
            let sequence = [code, b'A'];
            let encoded = encode_dna(&sequence);
            let expected_byte = expected_nibble << 4; // A (0x0) in low nibble
            assert_eq!(
                encoded[0], expected_byte,
                "{}A should encode to 0x{:02X}",
                code as char, expected_byte
            );
        }
    }

    /// Tests lowercase IUPAC ambiguity codes conversion.
    #[test]
    fn test_iupac_lowercase_conversion() {
        let codes: [(u8, u8); 11] = [
            (b'r', 0x4),
            (b'y', 0x5),
            (b's', 0x6),
            (b'w', 0x7),
            (b'k', 0x8),
            (b'm', 0x9),
            (b'b', 0xA),
            (b'd', 0xB),
            (b'h', 0xC),
            (b'v', 0xD),
            (b'n', 0xE),
        ];

        for (code, expected_value) in codes {
            assert_eq!(
                char_to_4bit(code),
                expected_value,
                "Lowercase '{}' should encode to 0x{:X}",
                code as char,
                expected_value
            );
        }
    }

    /// Tests IUPAC codes crossing SIMD boundaries (position 15-16).
    #[test]
    fn test_iupac_crossing_simd_boundary() {
        // Create a 32-byte sequence with IUPAC codes at positions 14-17
        // (crossing the 16-byte SIMD boundary)
        let mut sequence = vec![b'A'; 32];
        sequence[14] = b'N';
        sequence[15] = b'R';
        sequence[16] = b'Y';
        sequence[17] = b'S';

        let encoded = encode_dna(&sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests all IUPAC codes repeated to fill multiple SIMD blocks.
    #[test]
    fn test_iupac_multiple_simd_blocks() {
        // 16 IUPAC codes × 4 = 64 bytes (4 SIMD blocks)
        let pattern = b"ACGTRYSWKMBDHVN-";
        let sequence: Vec<u8> = pattern.iter().cycle().take(64).copied().collect();

        let encoded = encode_dna(&sequence);
        assert_eq!(encoded.len(), 32); // 64 / 2 = 32
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests each IUPAC code in isolation with roundtrip.
    #[test]
    fn test_iupac_individual_roundtrip() {
        let iupac_codes = b"RYSWKMBDHVN-";

        for &code in iupac_codes {
            // Single character
            let single = [code];
            let encoded = encode_dna(&single);
            let decoded = decode_dna(&encoded, 1);
            assert_eq!(decoded[0], code, "Single {} failed roundtrip", code as char);

            // Paired with itself
            let paired = [code, code];
            let encoded = encode_dna(&paired);
            let decoded = decode_dna(&encoded, 2);
            assert_eq!(decoded, paired, "Paired {} failed roundtrip", code as char);

            // 16 of the same code (SIMD path)
            let sixteen: Vec<u8> = vec![code; 16];
            let encoded = encode_dna(&sixteen);
            let decoded = decode_dna(&encoded, 16);
            assert_eq!(decoded, sixteen, "16x {} failed roundtrip", code as char);
        }
    }

    /// Tests IUPAC codes interspersed with standard bases.
    #[test]
    fn test_iupac_interspersed_with_bases() {
        let sequence = b"ANCTGNRTYCSWAGKT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests dot character as gap (alternative gap notation).
    #[test]
    fn test_iupac_dot_as_gap() {
        let sequence = b"AC.GT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        // Dot should decode as dash (gap)
        assert_eq!(decoded, b"AC-GT");
    }

    // ========================================================================
    // Case Handling Tests
    // ========================================================================

    /// Tests all lowercase encoding.
    #[test]
    fn test_all_lowercase_encoding() {
        let sequence = b"acgtacgtacgtacgt";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, b"ACGTACGTACGTACGT");
    }

    /// Tests mixed case encoding.
    #[test]
    fn test_mixed_case_encoding() {
        let sequence = b"AcGtAcGtAcGtAcGt";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, b"ACGTACGTACGTACGT");
    }

    /// Tests that uppercase and lowercase produce same encoding.
    #[test]
    fn test_case_insensitive_encoding() {
        let upper = b"ACGTRYSWKMBDHVN";
        let lower = b"acgtryswkmbdhvn";
        let mixed = b"AcGtRySw";

        let enc_upper = encode_dna(upper);
        let enc_lower = encode_dna(lower);

        assert_eq!(enc_upper, enc_lower);

        let enc_mixed = encode_dna(mixed);
        let enc_upper_part = encode_dna(b"ACGTRYSW");
        assert_eq!(enc_mixed, enc_upper_part);
    }

    // ========================================================================
    // Invalid Character Tests
    // ========================================================================

    /// Tests that invalid characters encode as gaps.
    #[test]
    fn test_invalid_characters_become_gaps() {
        let sequence = b"ACXZGT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        // X and Z are invalid, should become '-'
        assert_eq!(decoded, b"AC--GT");
    }

    /// Tests spaces and whitespace handling.
    #[test]
    fn test_whitespace_becomes_gaps() {
        let sequence = b"AC GT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, b"AC-GT");
    }

    /// Tests numeric characters handling.
    #[test]
    fn test_numbers_become_gaps() {
        let sequence = b"A1C2G3T4";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, b"A-C-G-T-");
    }

    // ========================================================================
    // Edge Cases and Stress Tests
    // ========================================================================

    /// Tests large sequence with various codes.
    #[test]
    fn test_large_sequence() {
        let sequence: Vec<u8> = (0..1000).map(|i| b"ACGTRYSWKMBDHVN-"[i % 16]).collect();
        let encoded = encode_dna(&sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests very large sequence with remainder.
    #[test]
    fn test_very_large_sequence_with_remainder() {
        let len = 10_007; // Prime number
        let sequence: Vec<u8> = (0..len).map(|i| b"ACGTN"[i % 5]).collect();
        let encoded = encode_dna(&sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests partial decoding.
    #[test]
    fn test_decode_partial_length() {
        let sequence = b"ACGTACGT";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, 5);
        assert_eq!(decoded, b"ACGTA");
    }

    /// Tests all lengths from 1 to 100.
    #[test]
    fn test_lengths_1_to_100() {
        for len in 1..=100 {
            let sequence: Vec<u8> = (0..len).map(|i| b"ACGTN"[i % 5]).collect();
            let encoded = encode_dna(&sequence);
            let decoded = decode_dna(&encoded, sequence.len());
            assert_eq!(decoded, sequence, "Failed for length {}", len);
        }
    }

    /// Tests scalar encoding directly.
    #[test]
    fn test_encode_scalar_directly() {
        let sequence = b"ACGTACGT";
        let mut output = vec![0u8; 4];
        encode_scalar(sequence, &mut output);
        assert_eq!(output[0], 0x01); // AC
        assert_eq!(output[1], 0x23); // GT
        assert_eq!(output[2], 0x01); // AC
        assert_eq!(output[3], 0x23); // GT
    }

    /// Tests scalar decoding directly.
    #[test]
    fn test_decode_scalar_directly() {
        let encoded = [0x01u8, 0x23]; // AC, GT
        let mut output = vec![0u8; 4];
        decode_scalar(&encoded, &mut output, 4);
        assert_eq!(output, b"ACGT");
    }

    // ========================================================================
    // Decode-Specific Tests
    // ========================================================================

    /// Tests decoding specific 4-bit values to IUPAC codes.
    #[test]
    fn test_decode_specific_4bit_values() {
        // Manually construct encoded bytes and verify decoding
        // Format: high nibble = first nucleotide, low nibble = second

        // 0x01 = AC, 0x23 = GT
        let decoded = decode_dna(&[0x01, 0x23], 4);
        assert_eq!(decoded, b"ACGT");

        // 0x45 = RY, 0x67 = SW
        let decoded = decode_dna(&[0x45, 0x67], 4);
        assert_eq!(decoded, b"RYSW");

        // 0x89 = KM, 0xAB = BD
        let decoded = decode_dna(&[0x89, 0xAB], 4);
        assert_eq!(decoded, b"KMBD");

        // 0xCD = HV, 0xEF = N-
        let decoded = decode_dna(&[0xCD, 0xEF], 4);
        assert_eq!(decoded, b"HVN-");
    }

    /// Tests decoding each 4-bit value individually (0x0 to 0xF).
    #[test]
    fn test_decode_all_4bit_values() {
        let expected: [u8; 16] = [
            b'A', b'C', b'G', b'T', b'R', b'Y', b'S', b'W', b'K', b'M', b'B', b'D', b'H', b'V',
            b'N', b'-',
        ];

        for (value, &expected_char) in expected.iter().enumerate() {
            // Test in high nibble position
            let encoded_high = [(value as u8) << 4];
            let decoded = decode_dna(&encoded_high, 1);
            assert_eq!(
                decoded[0], expected_char,
                "Value 0x{:X} in high nibble should decode to '{}'",
                value, expected_char as char
            );

            // Test in low nibble position
            let encoded_low = [value as u8];
            let decoded = decode_dna(&encoded_low, 2);
            assert_eq!(
                decoded[1], expected_char,
                "Value 0x{:X} in low nibble should decode to '{}'",
                value, expected_char as char
            );
        }
    }

    /// Tests decoding with length less than encoded capacity.
    #[test]
    fn test_decode_truncated_length() {
        let sequence = b"ACGTRYSWKMBDHVN-";
        let encoded = encode_dna(sequence);

        // Decode only first 8 characters
        let decoded = decode_dna(&encoded, 8);
        assert_eq!(decoded, b"ACGTRYSW");

        // Decode only first 1 character
        let decoded = decode_dna(&encoded, 1);
        assert_eq!(decoded, b"A");

        // Decode only first 3 characters (odd)
        let decoded = decode_dna(&encoded, 3);
        assert_eq!(decoded, b"ACG");
    }

    /// Tests decoding empty input.
    #[test]
    fn test_decode_empty() {
        let decoded = decode_dna(&[], 0);
        assert!(decoded.is_empty());
    }

    /// Tests decoding with zero length.
    #[test]
    fn test_decode_zero_length() {
        let encoded = [0x01, 0x23];
        let decoded = decode_dna(&encoded, 0);
        assert!(decoded.is_empty());
    }

    /// Tests decoding 16 bytes (SIMD path - 32 nucleotides).
    #[test]
    fn test_decode_simd_full_block() {
        // Create 16 encoded bytes representing 32 nucleotides
        let encoded: Vec<u8> = (0..16)
            .map(|i| {
                let high = (i * 2) % 16;
                let low = (i * 2 + 1) % 16;
                ((high << 4) | low) as u8
            })
            .collect();

        let decoded = decode_dna(&encoded, 32);
        assert_eq!(decoded.len(), 32);

        // Verify each decoded character
        for (i, &c) in decoded.iter().enumerate() {
            let expected_value = i % 16;
            let expected_char = fourbit_to_char(expected_value as u8);
            assert_eq!(
                c, expected_char,
                "Position {} should be '{}'",
                i, expected_char as char
            );
        }
    }

    /// Tests decoding at SIMD boundary (exactly 16 nucleotides from 8 bytes).
    #[test]
    fn test_decode_exactly_16_nucleotides() {
        let encoded = [0x01u8, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let decoded = decode_dna(&encoded, 16);
        assert_eq!(decoded, b"ACGTRYSWKMBDHVN-");
    }

    /// Tests decoding with remainder (not multiple of 16).
    #[test]
    fn test_decode_with_remainder() {
        // 20 nucleotides = 10 encoded bytes
        // SIMD handles first 16, scalar handles remaining 4
        let sequence = b"ACGTRYSWKMBDHVN-ACGT";
        let encoded = encode_dna(sequence);
        assert_eq!(encoded.len(), 10);

        let decoded = decode_dna(&encoded, 20);
        assert_eq!(decoded, sequence);
    }

    /// Tests decoding multiple SIMD blocks.
    #[test]
    fn test_decode_multiple_simd_blocks() {
        // 64 nucleotides = 32 encoded bytes = 4 SIMD blocks
        let pattern = b"ACGTRYSWKMBDHVN-";
        let sequence: Vec<u8> = pattern.iter().cycle().take(64).copied().collect();
        let encoded = encode_dna(&sequence);
        assert_eq!(encoded.len(), 32);

        let decoded = decode_dna(&encoded, 64);
        assert_eq!(decoded, sequence);
    }

    /// Tests decoding IUPAC codes specifically.
    #[test]
    fn test_decode_iupac_codes() {
        // R=4, Y=5, S=6, W=7, K=8, M=9, B=A, D=B, H=C, V=D, N=E
        let iupac_only = [
            0x45u8, // RY
            0x67,   // SW
            0x89,   // KM
            0xAB,   // BD
            0xCD,   // HV
            0xEE,   // NN
        ];

        let decoded = decode_dna(&iupac_only, 12);
        assert_eq!(decoded, b"RYSWKMBDHVNN");
    }

    /// Tests decoding gaps.
    #[test]
    fn test_decode_gaps() {
        // All gaps (0xF)
        let all_gaps = [0xFFu8, 0xFF, 0xFF, 0xFF];
        let decoded = decode_dna(&all_gaps, 8);
        assert_eq!(decoded, b"--------");
    }

    /// Tests decode scalar fallback for small inputs.
    #[test]
    fn test_decode_scalar_small_inputs() {
        // Test sizes that use scalar path (< 16 nucleotides)
        for size in [2, 4, 6, 8, 10, 12, 14] {
            let sequence: Vec<u8> = (0..size).map(|i| b"ACGTN"[i % 5]).collect();
            let encoded = encode_dna(&sequence);
            let decoded = decode_dna(&encoded, size);
            assert_eq!(decoded, sequence, "Failed for size {}", size);
        }
    }

    /// Tests decode produces uppercase output for all cases.
    #[test]
    fn test_decode_always_uppercase() {
        // Decode should always produce uppercase regardless of original encoding
        let sequence = b"acgtryswkmbdhvn";
        let encoded = encode_dna(sequence);
        let decoded = decode_dna(&encoded, sequence.len());

        for &c in &decoded {
            assert!(
                c.is_ascii_uppercase() || c == b'-',
                "Decoded character '{}' should be uppercase",
                c as char
            );
        }
        assert_eq!(decoded, b"ACGTRYSWKMBDHVN");
    }

    // ========================================================================
    // SIMD Boundary Tests
    // ========================================================================

    /// Tests 15 nucleotides (just under SIMD boundary).
    #[test]
    fn test_simd_boundary_15_nucleotides() {
        let sequence: Vec<u8> = (0..15).map(|i| b"ACGTN"[i % 5]).collect();
        let encoded = encode_dna(&sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests 17 nucleotides (just over SIMD boundary).
    #[test]
    fn test_simd_boundary_17_nucleotides() {
        let sequence: Vec<u8> = (0..17).map(|i| b"ACGTN"[i % 5]).collect();
        let encoded = encode_dna(&sequence);
        let decoded = decode_dna(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests exact multiples of 16.
    #[test]
    fn test_simd_boundary_exact_multiples() {
        for len in [16, 32, 48, 64, 128, 256] {
            let sequence: Vec<u8> = (0..len).map(|i| b"ACGTN"[i % 5]).collect();
            let encoded = encode_dna(&sequence);
            let decoded = decode_dna(&encoded, sequence.len());
            assert_eq!(decoded, sequence, "Failed for length {}", len);
        }
    }

    // ========================================================================
    // Thread Safety Tests
    // ========================================================================

    /// Tests concurrent encoding from multiple threads.
    #[test]
    fn test_concurrent_encoding() {
        use std::thread;

        let sequences: Vec<Vec<u8>> = (0..100)
            .map(|i| {
                let len = 100 + i * 10;
                (0..len).map(|j| b"ACGTN"[j % 5]).collect()
            })
            .collect();

        let handles: Vec<_> = sequences
            .into_iter()
            .map(|seq| {
                thread::spawn(move || {
                    let encoded = encode_dna(&seq);
                    let decoded = decode_dna(&encoded, seq.len());
                    assert_eq!(decoded, seq);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    /// Tests concurrent decoding from multiple threads.
    #[test]
    fn test_concurrent_decoding() {
        use std::thread;

        let test_data: Vec<(Vec<u8>, usize)> = (0..100)
            .map(|i| {
                let len = 100 + i * 10;
                let seq: Vec<u8> = (0..len).map(|j| b"ACGTN"[j % 5]).collect();
                let encoded = encode_dna(&seq);
                (encoded, len)
            })
            .collect();

        let handles: Vec<_> = test_data
            .into_iter()
            .map(|(encoded, len)| {
                thread::spawn(move || {
                    let decoded = decode_dna(&encoded, len);
                    for &nucleotide in &decoded {
                        assert!(
                            b"ACGTN".contains(&nucleotide),
                            "Invalid nucleotide: {}",
                            nucleotide as char
                        );
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    /// Tests concurrent encode/decode with shared data.
    #[test]
    fn test_concurrent_encode_decode_same_data() {
        use std::sync::Arc;
        use std::thread;

        let sequence: Arc<Vec<u8>> =
            Arc::new((0..10000).map(|i| b"ACGTRYSWKMBDHVN-"[i % 16]).collect());

        let handles: Vec<_> = (0..50)
            .map(|_| {
                let seq = Arc::clone(&sequence);
                thread::spawn(move || {
                    let encoded = encode_dna(&seq);
                    let decoded = decode_dna(&encoded, seq.len());
                    assert_eq!(decoded.as_slice(), seq.as_slice());
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    /// Tests thread safety across various lengths (scalar vs SIMD paths).
    #[test]
    fn test_thread_safety_with_varying_lengths() {
        use std::thread;

        let lengths: Vec<usize> = vec![1, 2, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 256, 1000];

        let handles: Vec<_> = lengths
            .into_iter()
            .flat_map(|len| {
                (0..10).map(move |_| {
                    thread::spawn(move || {
                        let seq: Vec<u8> = (0..len).map(|j| b"ACGTN"[j % 5]).collect();
                        let encoded = encode_dna(&seq);
                        let decoded = decode_dna(&encoded, seq.len());
                        assert_eq!(decoded, seq, "Failed for length {}", len);
                    })
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }
}
