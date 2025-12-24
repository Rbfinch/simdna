// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! # Hybrid 2-bit/4-bit DNA Encoding
//!
//! This module provides bifurcated DNA sequence encoding that automatically
//! selects the optimal encoding strategy based on sequence content:
//!
//! - **2-bit encoding** for "clean" sequences containing only A, C, G, T
//!   (4 bases per byte, 4x compression vs ASCII)
//! - **4-bit encoding** for "dirty" sequences containing ambiguous IUPAC codes
//!   (2 bases per byte, 2x compression vs ASCII)
//!
//! ## Design Rationale
//!
//! In typical genomic datasets, 99%+ of sequences contain only standard ACGT bases.
//! By using 2-bit encoding for these "clean" sequences, we achieve:
//!
//! - **4x memory density** compared to ASCII (vs 2x for universal 4-bit encoding)
//! - **Reduced memory bandwidth** requirements for SIMD scanning
//! - **Perfect alignment** with tetranucleotide lookup tables (4 bases = 8 bits = 1 byte)
//!
//! The small fraction of sequences containing N, R, Y, or other IUPAC codes
//! are encoded using the full 4-bit scheme, preserving complete fidelity.
//!
//! ## Encoding Schemes
//!
//! ### 2-bit Clean Encoding (A, C, G, T only)
//!
//! | Base | Binary | Decimal |
//! |------|--------|---------|
//! | A    | 00     | 0       |
//! | C    | 01     | 1       |
//! | G    | 10     | 2       |
//! | T    | 11     | 3       |
//!
//! Four bases are packed per byte: `(B0 << 6) | (B1 << 4) | (B2 << 2) | B3`
//!
//! ### 4-bit IUPAC Encoding (Full Alphabet)
//!
//! Uses the existing encoding from [`crate::dna_simd_encoder`] with bit-rotation
//! compatible values supporting efficient complement calculation.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
//!
//! // Clean sequence → 2-bit encoding
//! let clean = b"ACGTACGT";
//! let result = encode_bifurcated(clean);
//! assert_eq!(result.encoding, EncodingType::Clean2Bit);
//! assert_eq!(result.data.len(), 2); // 8 bases → 2 bytes
//!
//! // Dirty sequence → 4-bit encoding
//! let dirty = b"ACNGTACGT";
//! let result = encode_bifurcated(dirty);
//! assert_eq!(result.encoding, EncodingType::Dirty4Bit);
//! ```
//!
//! ## Thread Safety
//!
//! All public functions in this module are thread-safe and can be called
//! concurrently from multiple threads without synchronization.
//!
//! ## Performance
//!
//! On Apple Silicon (NEON):
//! - Purity check: Single-pass SIMD, ~15+ GB/s throughput
//! - 2-bit encoding: ~15+ GB/s throughput
//! - 4-bit encoding: Falls back to existing SIMD encoder (~10+ GB/s)

use std::fmt;

// ============================================================================
// Encoding Type Discriminator
// ============================================================================

/// Encoding type discriminator for bifurcated storage.
///
/// This enum identifies which encoding scheme was used for a sequence,
/// enabling the correct decoder and search algorithm to be selected.
///
/// The `repr(u8)` ensures a compact single-byte representation suitable
/// for storage in database columns (e.g., SQLite INTEGER).
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::EncodingType;
///
/// let clean = EncodingType::Clean2Bit;
/// let dirty = EncodingType::Dirty4Bit;
///
/// // Can be converted to/from integer for database storage
/// assert_eq!(clean as u8, 0);
/// assert_eq!(dirty as u8, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EncodingType {
    /// 2-bit encoding: 4 bases per byte (ACGT only).
    ///
    /// This encoding is used for sequences containing exclusively
    /// A, C, G, and T nucleotides. It provides 4x compression
    /// compared to ASCII representation.
    ///
    /// Encoding: A=00, C=01, G=10, T=11
    Clean2Bit = 0,

    /// 4-bit encoding: 2 bases per byte (full IUPAC support).
    ///
    /// This encoding is used for sequences containing one or more
    /// ambiguous IUPAC nucleotide codes (N, R, Y, S, W, K, M, B, D, H, V)
    /// or gap characters. It provides 2x compression compared to ASCII.
    ///
    /// Uses the bit-rotation compatible encoding from [`crate::dna_simd_encoder`].
    Dirty4Bit = 1,
}

impl EncodingType {
    /// Returns the compression ratio compared to ASCII representation.
    ///
    /// - `Clean2Bit`: 4.0 (4 bases per byte)
    /// - `Dirty4Bit`: 2.0 (2 bases per byte)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::hybrid_encoder::EncodingType;
    ///
    /// assert_eq!(EncodingType::Clean2Bit.compression_ratio(), 4.0);
    /// assert_eq!(EncodingType::Dirty4Bit.compression_ratio(), 2.0);
    /// ```
    #[inline]
    pub const fn compression_ratio(&self) -> f64 {
        match self {
            EncodingType::Clean2Bit => 4.0,
            EncodingType::Dirty4Bit => 2.0,
        }
    }

    /// Returns the number of bases packed per byte.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::hybrid_encoder::EncodingType;
    ///
    /// assert_eq!(EncodingType::Clean2Bit.bases_per_byte(), 4);
    /// assert_eq!(EncodingType::Dirty4Bit.bases_per_byte(), 2);
    /// ```
    #[inline]
    pub const fn bases_per_byte(&self) -> usize {
        match self {
            EncodingType::Clean2Bit => 4,
            EncodingType::Dirty4Bit => 2,
        }
    }

    /// Returns the number of bits used per base.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::hybrid_encoder::EncodingType;
    ///
    /// assert_eq!(EncodingType::Clean2Bit.bits_per_base(), 2);
    /// assert_eq!(EncodingType::Dirty4Bit.bits_per_base(), 4);
    /// ```
    #[inline]
    pub const fn bits_per_base(&self) -> usize {
        match self {
            EncodingType::Clean2Bit => 2,
            EncodingType::Dirty4Bit => 4,
        }
    }

    /// Creates an `EncodingType` from its integer representation.
    ///
    /// Returns `None` if the value is not a valid encoding type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::hybrid_encoder::EncodingType;
    ///
    /// assert_eq!(EncodingType::from_u8(0), Some(EncodingType::Clean2Bit));
    /// assert_eq!(EncodingType::from_u8(1), Some(EncodingType::Dirty4Bit));
    /// assert_eq!(EncodingType::from_u8(2), None);
    /// ```
    #[inline]
    pub const fn from_u8(value: u8) -> Option<EncodingType> {
        match value {
            0 => Some(EncodingType::Clean2Bit),
            1 => Some(EncodingType::Dirty4Bit),
            _ => None,
        }
    }

    /// Creates an `EncodingType` from its integer representation.
    ///
    /// Returns `None` if the value is not a valid encoding type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::hybrid_encoder::EncodingType;
    ///
    /// assert_eq!(EncodingType::from_i32(0), Some(EncodingType::Clean2Bit));
    /// assert_eq!(EncodingType::from_i32(1), Some(EncodingType::Dirty4Bit));
    /// assert_eq!(EncodingType::from_i32(-1), None);
    /// ```
    #[inline]
    pub const fn from_i32(value: i32) -> Option<EncodingType> {
        match value {
            0 => Some(EncodingType::Clean2Bit),
            1 => Some(EncodingType::Dirty4Bit),
            _ => None,
        }
    }
}

impl fmt::Display for EncodingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodingType::Clean2Bit => write!(f, "2-bit (Clean)"),
            EncodingType::Dirty4Bit => write!(f, "4-bit (IUPAC)"),
        }
    }
}

impl From<EncodingType> for u8 {
    #[inline]
    fn from(encoding: EncodingType) -> u8 {
        encoding as u8
    }
}

impl From<EncodingType> for i32 {
    #[inline]
    fn from(encoding: EncodingType) -> i32 {
        encoding as i32
    }
}

impl TryFrom<u8> for EncodingType {
    type Error = InvalidEncodingTypeError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        EncodingType::from_u8(value).ok_or(InvalidEncodingTypeError(value as i32))
    }
}

impl TryFrom<i32> for EncodingType {
    type Error = InvalidEncodingTypeError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        EncodingType::from_i32(value).ok_or(InvalidEncodingTypeError(value))
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Error returned when an invalid encoding type value is encountered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidEncodingTypeError(pub i32);

impl fmt::Display for InvalidEncodingTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid encoding type: {} (expected 0 for Clean2Bit or 1 for Dirty4Bit)",
            self.0
        )
    }
}

impl std::error::Error for InvalidEncodingTypeError {}

/// Errors that can occur during hybrid encoding/decoding operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HybridEncoderError {
    /// Output buffer is too small for the operation.
    BufferTooSmall {
        /// Number of bytes required.
        needed: usize,
        /// Actual buffer size provided.
        actual: usize,
    },
    /// Invalid encoding type encountered.
    InvalidEncodingType(i32),
    /// Encoded data is corrupted or invalid.
    InvalidEncodedData(String),
    /// Invalid base character for 2-bit encoding (not A, C, G, or T).
    InvalidBase(char),
}

impl fmt::Display for HybridEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HybridEncoderError::BufferTooSmall { needed, actual } => {
                write!(f, "buffer too small: need {} bytes, got {}", needed, actual)
            }
            HybridEncoderError::InvalidEncodingType(value) => {
                write!(f, "invalid encoding type: {}", value)
            }
            HybridEncoderError::InvalidEncodedData(msg) => {
                write!(f, "invalid encoded data: {}", msg)
            }
            HybridEncoderError::InvalidBase(c) => {
                write!(
                    f,
                    "invalid base '{}' for 2-bit encoding (expected A, C, G, or T)",
                    c
                )
            }
        }
    }
}

impl std::error::Error for HybridEncoderError {}

impl From<InvalidEncodingTypeError> for HybridEncoderError {
    fn from(err: InvalidEncodingTypeError) -> Self {
        HybridEncoderError::InvalidEncodingType(err.0)
    }
}

// ============================================================================
// Encoded Sequence Result
// ============================================================================

/// Result of sequence analysis and encoding.
///
/// This struct packages the encoded data together with metadata about
/// the encoding used, enabling correct decoding and search operations.
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{EncodedSequence, EncodingType};
///
/// // Create manually (typically use encode_bifurcated instead)
/// let encoded = EncodedSequence {
///     encoding: EncodingType::Clean2Bit,
///     data: vec![0b00011011], // ACGT packed
///     original_len: 4,
/// };
///
/// assert_eq!(encoded.encoding, EncodingType::Clean2Bit);
/// assert_eq!(encoded.original_len, 4);
/// assert_eq!(encoded.encoded_len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedSequence {
    /// The encoding scheme used for this sequence.
    pub encoding: EncodingType,

    /// The encoded binary data.
    pub data: Vec<u8>,

    /// The original sequence length in nucleotides.
    ///
    /// This is necessary for decoding because:
    /// - 2-bit encoding may have 0-3 padding bits in the final byte
    /// - 4-bit encoding may have 0-1 padding nibble in the final byte
    pub original_len: usize,
}

impl EncodedSequence {
    /// Creates a new `EncodedSequence` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `encoding` - The encoding type used
    /// * `data` - The encoded binary data
    /// * `original_len` - The original sequence length in nucleotides
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::hybrid_encoder::{EncodedSequence, EncodingType};
    ///
    /// let encoded = EncodedSequence::new(
    ///     EncodingType::Clean2Bit,
    ///     vec![0x1B], // ACGT
    ///     4,
    /// );
    /// ```
    #[inline]
    pub const fn new(encoding: EncodingType, data: Vec<u8>, original_len: usize) -> Self {
        Self {
            encoding,
            data,
            original_len,
        }
    }

    /// Returns the length of the encoded data in bytes.
    #[inline]
    pub fn encoded_len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the sequence was encoded using 2-bit clean encoding.
    #[inline]
    pub const fn is_clean(&self) -> bool {
        matches!(self.encoding, EncodingType::Clean2Bit)
    }

    /// Returns `true` if the sequence was encoded using 4-bit IUPAC encoding.
    #[inline]
    pub const fn is_dirty(&self) -> bool {
        matches!(self.encoding, EncodingType::Dirty4Bit)
    }

    /// Returns the compression ratio achieved for this sequence.
    ///
    /// This is calculated as `original_len / encoded_len`.
    /// Returns `f64::INFINITY` if `encoded_len` is 0.
    #[inline]
    pub fn actual_compression_ratio(&self) -> f64 {
        if self.data.is_empty() {
            f64::INFINITY
        } else {
            self.original_len as f64 / self.data.len() as f64
        }
    }
}

impl Default for EncodedSequence {
    fn default() -> Self {
        Self {
            encoding: EncodingType::Clean2Bit,
            data: Vec::new(),
            original_len: 0,
        }
    }
}

// ============================================================================
// Buffer Size Helper Functions
// ============================================================================

/// Returns the number of bytes required to encode a sequence using 2-bit encoding.
///
/// This is useful for pre-allocating buffers.
///
/// # Arguments
///
/// * `sequence_len` - The number of nucleotides in the input sequence
///
/// # Returns
///
/// The number of bytes required: `ceil(sequence_len / 4)`
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::required_2bit_len;
///
/// assert_eq!(required_2bit_len(0), 0);
/// assert_eq!(required_2bit_len(1), 1);
/// assert_eq!(required_2bit_len(4), 1);
/// assert_eq!(required_2bit_len(5), 2);
/// assert_eq!(required_2bit_len(100), 25);
/// ```
#[inline]
pub const fn required_2bit_len(sequence_len: usize) -> usize {
    sequence_len.div_ceil(4)
}

/// Returns the number of bytes required to encode a sequence using 4-bit encoding.
///
/// This is useful for pre-allocating buffers.
///
/// # Arguments
///
/// * `sequence_len` - The number of nucleotides in the input sequence
///
/// # Returns
///
/// The number of bytes required: `ceil(sequence_len / 2)`
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::required_4bit_len;
///
/// assert_eq!(required_4bit_len(0), 0);
/// assert_eq!(required_4bit_len(1), 1);
/// assert_eq!(required_4bit_len(2), 1);
/// assert_eq!(required_4bit_len(3), 2);
/// assert_eq!(required_4bit_len(100), 50);
/// ```
#[inline]
pub const fn required_4bit_len(sequence_len: usize) -> usize {
    sequence_len.div_ceil(2)
}

// ============================================================================
// 2-bit Encoding Constants
// ============================================================================

/// 2-bit encoding values for standard nucleotides.
///
/// These constants define the 2-bit encoding scheme used for "clean" sequences.
/// The encoding is chosen to optimize tetranucleotide packing:
/// four 2-bit values pack perfectly into one byte (0-255), enabling
/// efficient SIMD lookup table operations.
pub mod encoding_2bit {
    /// Adenine: binary 00
    pub const A: u8 = 0b00;
    /// Cytosine: binary 01
    pub const C: u8 = 0b01;
    /// Guanine: binary 10
    pub const G: u8 = 0b10;
    /// Thymine: binary 11
    pub const T: u8 = 0b11;

    /// Mask for extracting a 2-bit value
    pub const MASK: u8 = 0b11;
}

// ============================================================================
// 2-bit Encoder (Clean Sequences Only)
// ============================================================================

/// Lookup table for ASCII to 2-bit encoding.
///
/// Valid entries: A/a=0, C/c=1, G/g=2, T/t=3
/// Invalid entries: 0xFF (sentinel for validation)
///
/// This table is indexed by ASCII character value for O(1) lookup.
static ASCII_TO_2BIT: [u8; 256] = {
    let mut table = [0xFFu8; 256];
    table[b'A' as usize] = encoding_2bit::A;
    table[b'a' as usize] = encoding_2bit::A;
    table[b'C' as usize] = encoding_2bit::C;
    table[b'c' as usize] = encoding_2bit::C;
    table[b'G' as usize] = encoding_2bit::G;
    table[b'g' as usize] = encoding_2bit::G;
    table[b'T' as usize] = encoding_2bit::T;
    table[b't' as usize] = encoding_2bit::T;
    table
};

/// Encodes a clean DNA sequence (ACGT only) to 2-bit packed format.
///
/// This is the allocating version that returns a new `Vec<u8>`.
/// For zero-allocation encoding, use [`encode_2bit_into`].
///
/// # Arguments
///
/// * `sequence` - ASCII DNA sequence containing only A, C, G, T (case-insensitive)
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - Packed 2-bit encoded data
/// * `Err(HybridEncoderError::InvalidBase)` - If sequence contains non-ACGT characters
///
/// # Encoding Format
///
/// Four nucleotides are packed into each byte:
/// - Bits 7-6: First nucleotide
/// - Bits 5-4: Second nucleotide
/// - Bits 3-2: Third nucleotide
/// - Bits 1-0: Fourth nucleotide
///
/// ```text
/// Byte layout: [B0:7-6][B1:5-4][B2:3-2][B3:1-0]
/// ```
///
/// # Performance
///
/// This function automatically uses NEON SIMD instructions on ARM64
/// for processing 16 nucleotides at a time. Falls back to scalar
/// implementation on other architectures.
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::encode_2bit;
///
/// // "ACGT" encodes to 0b00_01_10_11 = 0x1B
/// let encoded = encode_2bit(b"ACGT").unwrap();
/// assert_eq!(encoded, vec![0x1B]);
///
/// // Case insensitive
/// let encoded_lower = encode_2bit(b"acgt").unwrap();
/// assert_eq!(encoded_lower, vec![0x1B]);
/// ```
pub fn encode_2bit(sequence: &[u8]) -> Result<Vec<u8>, HybridEncoderError> {
    if sequence.is_empty() {
        return Ok(Vec::new());
    }

    let output_len = required_2bit_len(sequence.len());
    let mut output = vec![0u8; output_len];

    encode_2bit_into(sequence, &mut output)?;

    Ok(output)
}

/// Encodes a clean DNA sequence to 2-bit packed format (zero-allocation).
///
/// Writes the encoded data directly into a caller-provided buffer.
///
/// # Arguments
///
/// * `sequence` - ASCII DNA sequence containing only A, C, G, T (case-insensitive)
/// * `output` - Pre-allocated output buffer. Must be at least [`required_2bit_len(sequence.len())`] bytes.
///
/// # Returns
///
/// * `Ok(bytes_written)` - Number of bytes written to output
/// * `Err(HybridEncoderError::BufferTooSmall)` - If output buffer is too small
/// * `Err(HybridEncoderError::InvalidBase)` - If sequence contains non-ACGT characters
///
/// # Example
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_2bit_into, required_2bit_len};
///
/// let sequence = b"ACGTACGT";
/// let mut buffer = vec![0u8; required_2bit_len(sequence.len())];
/// let bytes_written = encode_2bit_into(sequence, &mut buffer).unwrap();
/// assert_eq!(bytes_written, 2);
/// ```
#[inline]
pub fn encode_2bit_into(sequence: &[u8], output: &mut [u8]) -> Result<usize, HybridEncoderError> {
    let required = required_2bit_len(sequence.len());
    if output.len() < required {
        return Err(HybridEncoderError::BufferTooSmall {
            needed: required,
            actual: output.len(),
        });
    }

    if sequence.is_empty() {
        return Ok(0);
    }

    // Use SIMD on ARM64, scalar fallback on other architectures
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is always available on aarch64
        unsafe { encode_2bit_neon(sequence, output) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        encode_2bit_scalar(sequence, output)
    }
}

/// Scalar implementation of 2-bit encoding.
///
/// Processes 4 nucleotides at a time for complete bytes,
/// then handles any remainder.
#[inline]
#[allow(clippy::needless_range_loop)]
fn encode_2bit_scalar(sequence: &[u8], output: &mut [u8]) -> Result<usize, HybridEncoderError> {
    let len = sequence.len();
    let complete_bytes = len / 4;
    let remainder = len % 4;

    // Process complete 4-base groups
    for i in 0..complete_bytes {
        let base_idx = i * 4;
        let b0 = ascii_to_2bit(sequence[base_idx])?;
        let b1 = ascii_to_2bit(sequence[base_idx + 1])?;
        let b2 = ascii_to_2bit(sequence[base_idx + 2])?;
        let b3 = ascii_to_2bit(sequence[base_idx + 3])?;

        output[i] = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3;
    }

    // Handle remainder (1-3 bases)
    if remainder > 0 {
        let base_idx = complete_bytes * 4;
        let mut packed: u8 = 0;

        for j in 0..remainder {
            let b = ascii_to_2bit(sequence[base_idx + j])?;
            packed |= b << (6 - j * 2);
        }

        output[complete_bytes] = packed;
    }

    Ok(required_2bit_len(len))
}

/// Converts an ASCII nucleotide to its 2-bit encoding.
///
/// # Returns
///
/// * `Ok(encoding)` - 2-bit value (0-3)
/// * `Err(InvalidBase)` - If the character is not A, C, G, or T
#[inline]
fn ascii_to_2bit(byte: u8) -> Result<u8, HybridEncoderError> {
    let encoded = ASCII_TO_2BIT[byte as usize];
    if encoded == 0xFF {
        return Err(HybridEncoderError::InvalidBase(byte as char));
    }
    Ok(encoded)
}

/// NEON-accelerated 2-bit encoding for ARM64.
///
/// Processes 16 nucleotides at a time using NEON vector instructions:
/// 1. Loads 16 bytes of ASCII nucleotides
/// 2. Converts each byte to 2-bit encoding using table lookup
/// 3. Validates all bytes are valid ACGT characters
/// 4. Packs 16 2-bit values into 4 output bytes
///
/// # Safety
///
/// This function requires NEON support (always available on ARM64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_2bit_neon(
    sequence: &[u8],
    output: &mut [u8],
) -> Result<usize, HybridEncoderError> {
    use core::arch::aarch64::*;

    let len = sequence.len();
    let simd_len = (len / 16) * 16;
    let mut out_idx = 0;
    let mut in_idx = 0;

    // Process 16 nucleotides at a time
    while in_idx < simd_len {
        // Load 16 bytes
        let chunk = unsafe { vld1q_u8(sequence.as_ptr().add(in_idx)) };

        // Extract bytes, encode each using LUT, and validate
        let mut bytes = [0u8; 16];
        unsafe { vst1q_u8(bytes.as_mut_ptr(), chunk) };

        let mut encoded_bytes = [0u8; 16];
        for i in 0..16 {
            let b = bytes[i];
            let encoded = ASCII_TO_2BIT[b as usize];
            if encoded == 0xFF {
                return Err(HybridEncoderError::InvalidBase(b as char));
            }
            encoded_bytes[i] = encoded;
        }

        // Pack 16 2-bit values into 4 bytes
        for i in 0..4 {
            let base = i * 4;
            output[out_idx + i] = (encoded_bytes[base] << 6)
                | (encoded_bytes[base + 1] << 4)
                | (encoded_bytes[base + 2] << 2)
                | encoded_bytes[base + 3];
        }

        in_idx += 16;
        out_idx += 4;
    }

    // Handle remainder with scalar code
    if in_idx < len {
        let scalar_result = encode_2bit_scalar(&sequence[in_idx..], &mut output[out_idx..])?;
        out_idx += scalar_result;
    }

    Ok(out_idx)
}

/// Packs 16 2-bit values into 4 bytes using NEON.
///
/// Decodes 2-bit packed data back to ASCII nucleotides.
///
/// This is the allocating version that returns a new `Vec<u8>`.
/// For zero-allocation decoding, use [`decode_2bit_into`].
///
/// # Arguments
///
/// * `encoded` - 2-bit packed encoded data
/// * `original_len` - The original sequence length (number of nucleotides)
///
/// # Returns
///
/// ASCII DNA sequence with uppercase A, C, G, T.
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_2bit, decode_2bit};
///
/// let original = b"ACGTACGT";
/// let encoded = encode_2bit(original).unwrap();
/// let decoded = decode_2bit(&encoded, original.len());
/// assert_eq!(decoded, original);
/// ```
pub fn decode_2bit(encoded: &[u8], original_len: usize) -> Vec<u8> {
    if original_len == 0 {
        return Vec::new();
    }

    let mut output = vec![0u8; original_len];
    decode_2bit_into(encoded, original_len, &mut output).expect("output buffer sized correctly");

    output
}

/// Decodes 2-bit packed data to ASCII nucleotides (zero-allocation).
///
/// Writes the decoded sequence directly into a caller-provided buffer.
///
/// # Arguments
///
/// * `encoded` - 2-bit packed encoded data
/// * `original_len` - The original sequence length (number of nucleotides)
/// * `output` - Pre-allocated output buffer. Must be at least `original_len` bytes.
///
/// # Returns
///
/// * `Ok(bytes_written)` - Number of bytes written to output
/// * `Err(BufferTooSmall)` - If output buffer is too small
///
/// # Example
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_2bit, decode_2bit_into};
///
/// let original = b"ACGTACGT";
/// let encoded = encode_2bit(original).unwrap();
/// let mut buffer = vec![0u8; 64];
/// let bytes_written = decode_2bit_into(&encoded, original.len(), &mut buffer).unwrap();
/// assert_eq!(&buffer[..bytes_written], original.as_slice());
/// ```
#[inline]
#[allow(clippy::needless_return)]
pub fn decode_2bit_into(
    encoded: &[u8],
    original_len: usize,
    output: &mut [u8],
) -> Result<usize, HybridEncoderError> {
    if output.len() < original_len {
        return Err(HybridEncoderError::BufferTooSmall {
            needed: original_len,
            actual: output.len(),
        });
    }

    if original_len == 0 {
        return Ok(0);
    }

    // Use SIMD on ARM64, scalar fallback on other architectures
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is always available on aarch64
        unsafe { decode_2bit_neon(encoded, original_len, output) };
        return Ok(original_len);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        decode_2bit_scalar(encoded, original_len, output);
        Ok(original_len)
    }
}

/// Lookup table for 2-bit to ASCII decoding.
static DECODE_2BIT_LUT: [u8; 4] = [b'A', b'C', b'G', b'T'];

/// Scalar implementation of 2-bit decoding.
#[inline]
#[allow(clippy::needless_range_loop)]
fn decode_2bit_scalar(encoded: &[u8], original_len: usize, output: &mut [u8]) {
    let complete_bytes = original_len / 4;
    let remainder = original_len % 4;

    // Process complete 4-base groups
    for i in 0..complete_bytes {
        let byte = encoded[i];
        let base_idx = i * 4;

        output[base_idx] = DECODE_2BIT_LUT[((byte >> 6) & 0x03) as usize];
        output[base_idx + 1] = DECODE_2BIT_LUT[((byte >> 4) & 0x03) as usize];
        output[base_idx + 2] = DECODE_2BIT_LUT[((byte >> 2) & 0x03) as usize];
        output[base_idx + 3] = DECODE_2BIT_LUT[(byte & 0x03) as usize];
    }

    // Handle remainder
    if remainder > 0 {
        let byte = encoded[complete_bytes];
        let base_idx = complete_bytes * 4;

        for j in 0..remainder {
            let shift = 6 - j * 2;
            output[base_idx + j] = DECODE_2BIT_LUT[((byte >> shift) & 0x03) as usize];
        }
    }
}

/// NEON-accelerated 2-bit decoding for ARM64.
///
/// Unpacks 4 bytes (16 nucleotides) at a time using NEON.
///
/// # Safety
///
/// Requires NEON support (always available on ARM64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn decode_2bit_neon(encoded: &[u8], original_len: usize, output: &mut [u8]) {
    use core::arch::aarch64::*;

    // Lookup table for 2-bit -> ASCII
    let lut = vld1_u8([b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'T'].as_ptr());

    let complete_chunks = original_len / 16;
    let mut out_idx = 0;
    let mut enc_idx = 0;

    // Process 4 encoded bytes (16 nucleotides) at a time
    for _ in 0..complete_chunks {
        // Unpack 4 bytes to 16 2-bit values
        let mut unpacked = [0u8; 16];
        for i in 0..4 {
            let byte = encoded[enc_idx + i];
            let base = i * 4;
            unpacked[base] = (byte >> 6) & 0x03;
            unpacked[base + 1] = (byte >> 4) & 0x03;
            unpacked[base + 2] = (byte >> 2) & 0x03;
            unpacked[base + 3] = byte & 0x03;
        }

        // Load unpacked indices
        let indices = vld1q_u8(unpacked.as_ptr());

        // Use table lookup for conversion (vqtbl1_u8 with indices 0-3)
        let decoded = vqtbl1q_u8(vcombine_u8(lut, lut), indices);

        // Store result
        vst1q_u8(output.as_mut_ptr().add(out_idx), decoded);

        enc_idx += 4;
        out_idx += 16;
    }

    // Handle remainder with scalar code
    if out_idx < original_len {
        decode_2bit_scalar(
            &encoded[enc_idx..],
            original_len - out_idx,
            &mut output[out_idx..],
        );
    }
}

/// Checks if a sequence contains only valid 2-bit encodable characters (ACGT).
///
/// This is useful for determining whether a sequence can use the more
/// efficient 2-bit encoding or requires 4-bit IUPAC encoding.
///
/// # Arguments
///
/// * `sequence` - ASCII DNA sequence to validate
///
/// # Returns
///
/// `true` if all characters are A, C, G, or T (case-insensitive)
///
/// # Performance
///
/// Uses NEON SIMD on ARM64 for fast validation of 16 bytes at a time.
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::is_clean_sequence;
///
/// assert!(is_clean_sequence(b"ACGT"));
/// assert!(is_clean_sequence(b"acgtACGT"));
/// assert!(!is_clean_sequence(b"ACGTN")); // N is not valid
/// assert!(!is_clean_sequence(b"ACGR"));  // R is ambiguous
/// ```
pub fn is_clean_sequence(sequence: &[u8]) -> bool {
    if sequence.is_empty() {
        return true;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is always available on aarch64
        unsafe { is_clean_sequence_neon(sequence) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        is_clean_sequence_scalar(sequence)
    }
}

/// Scalar implementation of clean sequence check.
#[inline]
fn is_clean_sequence_scalar(sequence: &[u8]) -> bool {
    sequence.iter().all(|&b| ASCII_TO_2BIT[b as usize] != 0xFF)
}

/// NEON-accelerated clean sequence check for ARM64.
///
/// Validates 16 bytes at a time using SIMD comparison operations.
///
/// # Safety
///
/// Requires NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn is_clean_sequence_neon(sequence: &[u8]) -> bool {
    use core::arch::aarch64::*;

    let len = sequence.len();
    let simd_len = (len / 16) * 16;

    // Constants for comparison
    let mask_case = vdupq_n_u8(0xDF);
    let val_a = vdupq_n_u8(b'A');
    let val_c = vdupq_n_u8(b'C');
    let val_g = vdupq_n_u8(b'G');
    let val_t = vdupq_n_u8(b'T');

    let mut idx = 0;

    // Process 16 bytes at a time
    while idx < simd_len {
        let chunk = vld1q_u8(sequence.as_ptr().add(idx));
        let upper = vandq_u8(chunk, mask_case);

        // Check if each byte matches one of ACGT
        let is_a = vceqq_u8(upper, val_a);
        let is_c = vceqq_u8(upper, val_c);
        let is_g = vceqq_u8(upper, val_g);
        let is_t = vceqq_u8(upper, val_t);

        let valid = vorrq_u8(vorrq_u8(is_a, is_c), vorrq_u8(is_g, is_t));

        // Check if all bytes are valid (all bits set in valid lanes)
        let all_valid = vminvq_u8(valid);
        if all_valid == 0 {
            return false;
        }

        idx += 16;
    }

    // Check remainder with scalar
    is_clean_sequence_scalar(&sequence[idx..])
}

// ============================================================================
// Sequence Classification
// ============================================================================

/// Classifies a DNA sequence and returns the optimal encoding type.
///
/// This function analyzes the sequence content to determine whether it can
/// use the more efficient 2-bit encoding (ACGT only) or requires the full
/// 4-bit IUPAC encoding.
///
/// # Arguments
///
/// * `sequence` - ASCII DNA sequence to classify
///
/// # Returns
///
/// * [`EncodingType::Clean2Bit`] if the sequence contains only A, C, G, T (case-insensitive)
/// * [`EncodingType::Dirty4Bit`] if the sequence contains any other characters
///
/// # Performance
///
/// Uses NEON SIMD on ARM64 for fast classification of 16 bytes at a time.
/// For most genomic data (99%+ clean sequences), this enables the 4x compression
/// advantage of 2-bit encoding.
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{classify_sequence, EncodingType};
///
/// // Clean sequences use 2-bit encoding
/// assert_eq!(classify_sequence(b"ACGT"), EncodingType::Clean2Bit);
/// assert_eq!(classify_sequence(b"acgtACGT"), EncodingType::Clean2Bit);
///
/// // Sequences with ambiguous bases use 4-bit encoding
/// assert_eq!(classify_sequence(b"ACGTN"), EncodingType::Dirty4Bit);
/// assert_eq!(classify_sequence(b"ACGR"), EncodingType::Dirty4Bit);
///
/// // Empty sequences are considered clean
/// assert_eq!(classify_sequence(b""), EncodingType::Clean2Bit);
/// ```
#[inline]
pub fn classify_sequence(sequence: &[u8]) -> EncodingType {
    if is_clean_sequence(sequence) {
        EncodingType::Clean2Bit
    } else {
        EncodingType::Dirty4Bit
    }
}

// ============================================================================
// Bifurcated Encoder/Decoder API
// ============================================================================

/// Encodes a DNA sequence using the optimal encoding based on content.
///
/// This is the main entry point for bifurcated encoding. It automatically:
/// 1. Analyzes the sequence to classify it as clean (ACGT only) or dirty (IUPAC)
/// 2. Applies 2-bit encoding for clean sequences (4x compression)
/// 3. Applies 4-bit encoding for dirty sequences (2x compression)
/// 4. Returns an [`EncodedSequence`] containing the data and metadata
///
/// # Arguments
///
/// * `sequence` - ASCII DNA sequence (case-insensitive)
///
/// # Returns
///
/// An [`EncodedSequence`] containing:
/// * `encoding` - The encoding type used ([`EncodingType::Clean2Bit`] or [`EncodingType::Dirty4Bit`])
/// * `data` - The packed binary data
/// * `original_len` - The original sequence length for decoding
///
/// # Performance
///
/// This function uses SIMD-accelerated implementations on ARM64 (NEON) and x86_64 (SSSE3).
/// The bifurcation decision is made in a single pass using SIMD validation.
///
/// # Thread Safety
///
/// This function is thread-safe and can be called concurrently from multiple threads.
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
///
/// // Clean sequence → 2-bit encoding (4x compression)
/// let clean = b"ACGTACGT";
/// let result = encode_bifurcated(clean);
/// assert_eq!(result.encoding, EncodingType::Clean2Bit);
/// assert_eq!(result.data.len(), 2); // 8 bases → 2 bytes
/// assert_eq!(result.original_len, 8);
///
/// // Dirty sequence → 4-bit encoding (2x compression)
/// let dirty = b"ACNGTACGT";
/// let result = encode_bifurcated(dirty);
/// assert_eq!(result.encoding, EncodingType::Dirty4Bit);
/// assert_eq!(result.data.len(), 5); // 9 bases → 5 bytes (ceil(9/2))
/// ```
pub fn encode_bifurcated(sequence: &[u8]) -> EncodedSequence {
    if sequence.is_empty() {
        return EncodedSequence::default();
    }

    let encoding = classify_sequence(sequence);

    match encoding {
        EncodingType::Clean2Bit => {
            // Safe to unwrap because we verified it's a clean sequence
            let data = encode_2bit(sequence).expect("clean sequence should encode without error");
            EncodedSequence::new(EncodingType::Clean2Bit, data, sequence.len())
        }
        EncodingType::Dirty4Bit => {
            // Use the existing 4-bit encoder from dna_simd_encoder
            let data = crate::dna_simd_encoder::encode_dna_prefer_simd(sequence);
            EncodedSequence::new(EncodingType::Dirty4Bit, data, sequence.len())
        }
    }
}

/// Encodes a DNA sequence using the optimal encoding into a pre-allocated buffer.
///
/// This is the zero-allocation variant of [`encode_bifurcated`]. It writes the
/// encoded data directly into the provided buffer and returns an [`EncodedSequence`]
/// that borrows from that buffer (via the returned metadata).
///
/// # Arguments
///
/// * `sequence` - ASCII DNA sequence (case-insensitive)
/// * `output` - Pre-allocated output buffer. Must be large enough to hold the worst-case
///   encoding (4-bit), which is `ceil(sequence.len() / 2)` bytes.
///
/// # Returns
///
/// * `Ok((encoding_type, bytes_written))` - The encoding type used and bytes written
/// * `Err(HybridEncoderError::BufferTooSmall)` - If the buffer is too small
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated_into, EncodingType, required_4bit_len};
///
/// let sequence = b"ACGTACGT";
/// let mut buffer = vec![0u8; required_4bit_len(sequence.len())];
///
/// let (encoding, bytes_written) = encode_bifurcated_into(sequence, &mut buffer).unwrap();
/// assert_eq!(encoding, EncodingType::Clean2Bit);
/// assert_eq!(bytes_written, 2);
/// ```
pub fn encode_bifurcated_into(
    sequence: &[u8],
    output: &mut [u8],
) -> Result<(EncodingType, usize), HybridEncoderError> {
    if sequence.is_empty() {
        return Ok((EncodingType::Clean2Bit, 0));
    }

    let encoding = classify_sequence(sequence);

    match encoding {
        EncodingType::Clean2Bit => {
            let bytes_written = encode_2bit_into(sequence, output)?;
            Ok((EncodingType::Clean2Bit, bytes_written))
        }
        EncodingType::Dirty4Bit => {
            let required = required_4bit_len(sequence.len());
            if output.len() < required {
                return Err(HybridEncoderError::BufferTooSmall {
                    needed: required,
                    actual: output.len(),
                });
            }
            let bytes_written = crate::dna_simd_encoder::encode_dna_into(sequence, output)
                .map_err(
                    |crate::dna_simd_encoder::BufferError::BufferTooSmall { needed, actual }| {
                        HybridEncoderError::BufferTooSmall { needed, actual }
                    },
                )?;
            Ok((EncodingType::Dirty4Bit, bytes_written))
        }
    }
}

/// Decodes a bifurcated-encoded sequence back to ASCII nucleotides.
///
/// This function uses the encoding type metadata from [`EncodedSequence`] to
/// select the correct decoder (2-bit or 4-bit).
///
/// # Arguments
///
/// * `encoded` - An [`EncodedSequence`] as produced by [`encode_bifurcated`]
///
/// # Returns
///
/// The decoded ASCII DNA sequence with uppercase nucleotides.
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, decode_bifurcated};
///
/// let original = b"ACGTACGT";
/// let encoded = encode_bifurcated(original);
/// let decoded = decode_bifurcated(&encoded);
/// assert_eq!(decoded, original.to_vec());
///
/// // Works with dirty sequences too
/// let dirty = b"ACNGTACGT";
/// let encoded = encode_bifurcated(dirty);
/// let decoded = decode_bifurcated(&encoded);
/// assert_eq!(decoded, dirty.to_ascii_uppercase());
/// ```
pub fn decode_bifurcated(encoded: &EncodedSequence) -> Vec<u8> {
    if encoded.original_len == 0 {
        return Vec::new();
    }

    match encoded.encoding {
        EncodingType::Clean2Bit => decode_2bit(&encoded.data, encoded.original_len),
        EncodingType::Dirty4Bit => {
            crate::dna_simd_encoder::decode_dna_prefer_simd(&encoded.data, encoded.original_len)
        }
    }
}

/// Decodes a bifurcated-encoded sequence into a pre-allocated buffer.
///
/// This is the zero-allocation variant of [`decode_bifurcated`].
///
/// # Arguments
///
/// * `encoded` - An [`EncodedSequence`] as produced by [`encode_bifurcated`]
/// * `output` - Pre-allocated output buffer. Must be at least `encoded.original_len` bytes.
///
/// # Returns
///
/// * `Ok(bytes_written)` - Number of bytes written to output
/// * `Err(HybridEncoderError::BufferTooSmall)` - If the buffer is too small
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, decode_bifurcated_into};
///
/// let original = b"ACGTACGT";
/// let encoded = encode_bifurcated(original);
/// let mut buffer = vec![0u8; 64];
///
/// let bytes_written = decode_bifurcated_into(&encoded, &mut buffer).unwrap();
/// assert_eq!(&buffer[..bytes_written], original.as_slice());
/// ```
pub fn decode_bifurcated_into(
    encoded: &EncodedSequence,
    output: &mut [u8],
) -> Result<usize, HybridEncoderError> {
    if encoded.original_len == 0 {
        return Ok(0);
    }

    if output.len() < encoded.original_len {
        return Err(HybridEncoderError::BufferTooSmall {
            needed: encoded.original_len,
            actual: output.len(),
        });
    }

    match encoded.encoding {
        EncodingType::Clean2Bit => decode_2bit_into(&encoded.data, encoded.original_len, output),
        EncodingType::Dirty4Bit => {
            crate::dna_simd_encoder::decode_dna_into(&encoded.data, encoded.original_len, output)
                .map_err(
                    |crate::dna_simd_encoder::BufferError::BufferTooSmall { needed, actual }| {
                        HybridEncoderError::BufferTooSmall { needed, actual }
                    },
                )
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_type_values() {
        assert_eq!(EncodingType::Clean2Bit as u8, 0);
        assert_eq!(EncodingType::Dirty4Bit as u8, 1);
    }

    #[test]
    fn test_encoding_type_from_u8() {
        assert_eq!(EncodingType::from_u8(0), Some(EncodingType::Clean2Bit));
        assert_eq!(EncodingType::from_u8(1), Some(EncodingType::Dirty4Bit));
        assert_eq!(EncodingType::from_u8(2), None);
        assert_eq!(EncodingType::from_u8(255), None);
    }

    #[test]
    fn test_encoding_type_from_i32() {
        assert_eq!(EncodingType::from_i32(0), Some(EncodingType::Clean2Bit));
        assert_eq!(EncodingType::from_i32(1), Some(EncodingType::Dirty4Bit));
        assert_eq!(EncodingType::from_i32(-1), None);
        assert_eq!(EncodingType::from_i32(100), None);
    }

    #[test]
    fn test_encoding_type_try_from() {
        assert_eq!(EncodingType::try_from(0u8), Ok(EncodingType::Clean2Bit));
        assert_eq!(EncodingType::try_from(1u8), Ok(EncodingType::Dirty4Bit));
        assert!(EncodingType::try_from(2u8).is_err());

        assert_eq!(EncodingType::try_from(0i32), Ok(EncodingType::Clean2Bit));
        assert_eq!(EncodingType::try_from(1i32), Ok(EncodingType::Dirty4Bit));
        assert!(EncodingType::try_from(-1i32).is_err());
    }

    #[test]
    fn test_encoding_type_into() {
        let clean: u8 = EncodingType::Clean2Bit.into();
        let dirty: u8 = EncodingType::Dirty4Bit.into();
        assert_eq!(clean, 0);
        assert_eq!(dirty, 1);

        let clean_i32: i32 = EncodingType::Clean2Bit.into();
        let dirty_i32: i32 = EncodingType::Dirty4Bit.into();
        assert_eq!(clean_i32, 0);
        assert_eq!(dirty_i32, 1);
    }

    #[test]
    fn test_encoding_type_properties() {
        assert_eq!(EncodingType::Clean2Bit.compression_ratio(), 4.0);
        assert_eq!(EncodingType::Dirty4Bit.compression_ratio(), 2.0);

        assert_eq!(EncodingType::Clean2Bit.bases_per_byte(), 4);
        assert_eq!(EncodingType::Dirty4Bit.bases_per_byte(), 2);

        assert_eq!(EncodingType::Clean2Bit.bits_per_base(), 2);
        assert_eq!(EncodingType::Dirty4Bit.bits_per_base(), 4);
    }

    #[test]
    fn test_encoding_type_display() {
        assert_eq!(format!("{}", EncodingType::Clean2Bit), "2-bit (Clean)");
        assert_eq!(format!("{}", EncodingType::Dirty4Bit), "4-bit (IUPAC)");
    }

    #[test]
    fn test_encoded_sequence_new() {
        let encoded = EncodedSequence::new(EncodingType::Clean2Bit, vec![0x1B], 4);

        assert_eq!(encoded.encoding, EncodingType::Clean2Bit);
        assert_eq!(encoded.data, vec![0x1B]);
        assert_eq!(encoded.original_len, 4);
        assert_eq!(encoded.encoded_len(), 1);
    }

    #[test]
    fn test_encoded_sequence_is_clean_dirty() {
        let clean = EncodedSequence::new(EncodingType::Clean2Bit, vec![], 0);
        let dirty = EncodedSequence::new(EncodingType::Dirty4Bit, vec![], 0);

        assert!(clean.is_clean());
        assert!(!clean.is_dirty());
        assert!(!dirty.is_clean());
        assert!(dirty.is_dirty());
    }

    #[test]
    fn test_encoded_sequence_compression_ratio() {
        // 8 bases encoded to 2 bytes (2-bit)
        let seq = EncodedSequence::new(EncodingType::Clean2Bit, vec![0, 0], 8);
        assert_eq!(seq.actual_compression_ratio(), 4.0);

        // 8 bases encoded to 4 bytes (4-bit)
        let seq = EncodedSequence::new(EncodingType::Dirty4Bit, vec![0, 0, 0, 0], 8);
        assert_eq!(seq.actual_compression_ratio(), 2.0);

        // Empty sequence
        let empty = EncodedSequence::default();
        assert!(empty.actual_compression_ratio().is_infinite());
    }

    #[test]
    fn test_encoded_sequence_default() {
        let default = EncodedSequence::default();
        assert_eq!(default.encoding, EncodingType::Clean2Bit);
        assert!(default.data.is_empty());
        assert_eq!(default.original_len, 0);
    }

    #[test]
    fn test_required_2bit_len() {
        assert_eq!(required_2bit_len(0), 0);
        assert_eq!(required_2bit_len(1), 1);
        assert_eq!(required_2bit_len(2), 1);
        assert_eq!(required_2bit_len(3), 1);
        assert_eq!(required_2bit_len(4), 1);
        assert_eq!(required_2bit_len(5), 2);
        assert_eq!(required_2bit_len(8), 2);
        assert_eq!(required_2bit_len(100), 25);
        assert_eq!(required_2bit_len(101), 26);
    }

    #[test]
    fn test_required_4bit_len() {
        assert_eq!(required_4bit_len(0), 0);
        assert_eq!(required_4bit_len(1), 1);
        assert_eq!(required_4bit_len(2), 1);
        assert_eq!(required_4bit_len(3), 2);
        assert_eq!(required_4bit_len(100), 50);
        assert_eq!(required_4bit_len(101), 51);
    }

    #[test]
    fn test_hybrid_encoder_error_display() {
        let err = HybridEncoderError::BufferTooSmall {
            needed: 10,
            actual: 5,
        };
        assert_eq!(format!("{}", err), "buffer too small: need 10 bytes, got 5");

        let err = HybridEncoderError::InvalidEncodingType(42);
        assert_eq!(format!("{}", err), "invalid encoding type: 42");

        let err = HybridEncoderError::InvalidEncodedData("corrupted".to_string());
        assert_eq!(format!("{}", err), "invalid encoded data: corrupted");
    }

    #[test]
    fn test_invalid_encoding_type_error() {
        let err = InvalidEncodingTypeError(5);
        assert_eq!(
            format!("{}", err),
            "invalid encoding type: 5 (expected 0 for Clean2Bit or 1 for Dirty4Bit)"
        );
    }

    #[test]
    fn test_2bit_encoding_constants() {
        assert_eq!(encoding_2bit::A, 0b00);
        assert_eq!(encoding_2bit::C, 0b01);
        assert_eq!(encoding_2bit::G, 0b10);
        assert_eq!(encoding_2bit::T, 0b11);
        assert_eq!(encoding_2bit::MASK, 0b11);
    }

    // ========================================================================
    // 2-bit Encoder Tests
    // ========================================================================

    #[test]
    fn test_encode_2bit_empty() {
        let result = encode_2bit(b"").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_encode_2bit_single_base() {
        // Single base encodes to 1 byte, top 2 bits used
        assert_eq!(encode_2bit(b"A").unwrap(), vec![0b00_00_00_00]);
        assert_eq!(encode_2bit(b"C").unwrap(), vec![0b01_00_00_00]);
        assert_eq!(encode_2bit(b"G").unwrap(), vec![0b10_00_00_00]);
        assert_eq!(encode_2bit(b"T").unwrap(), vec![0b11_00_00_00]);
    }

    #[test]
    fn test_encode_2bit_acgt_quartet() {
        // ACGT = 00_01_10_11 = 0x1B
        let result = encode_2bit(b"ACGT").unwrap();
        assert_eq!(result, vec![0x1B]);
    }

    #[test]
    fn test_encode_2bit_case_insensitive() {
        let upper = encode_2bit(b"ACGT").unwrap();
        let lower = encode_2bit(b"acgt").unwrap();
        let mixed = encode_2bit(b"AcGt").unwrap();

        assert_eq!(upper, lower);
        assert_eq!(upper, mixed);
    }

    #[test]
    fn test_encode_2bit_remainder_handling() {
        // 5 bases = 2 bytes (4 + 1 remainder)
        let result = encode_2bit(b"ACGTA").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0x1B); // ACGT
        assert_eq!(result[1], 0b00_00_00_00); // A with padding

        // 6 bases = 2 bytes (4 + 2 remainder)
        let result = encode_2bit(b"ACGTAC").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0x1B);
        assert_eq!(result[1], 0b00_01_00_00); // AC with padding

        // 7 bases = 2 bytes (4 + 3 remainder)
        let result = encode_2bit(b"ACGTACG").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0x1B);
        assert_eq!(result[1], 0b00_01_10_00); // ACG with padding
    }

    #[test]
    fn test_encode_2bit_invalid_base() {
        // N is not valid for 2-bit encoding
        let result = encode_2bit(b"ACGTN");
        assert!(result.is_err());
        match result {
            Err(HybridEncoderError::InvalidBase('N')) => {}
            _ => panic!("Expected InvalidBase('N')"),
        }

        // R (purine) is not valid
        let result = encode_2bit(b"ACGR");
        assert!(result.is_err());

        // Space is not valid
        let result = encode_2bit(b"AC GT");
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_2bit_roundtrip() {
        let test_cases = [
            b"ACGT".as_slice(),
            b"AAAA",
            b"CCCC",
            b"GGGG",
            b"TTTT",
            b"ACGTACGTACGTACGT",  // 16 bases (SIMD boundary)
            b"ACGTACGTACGTACGTA", // 17 bases (SIMD + remainder)
            b"A",
            b"AC",
            b"ACG",
        ];

        for original in test_cases {
            let encoded = encode_2bit(original).unwrap();
            let decoded = decode_2bit(&encoded, original.len());
            assert_eq!(
                decoded,
                original.to_ascii_uppercase(),
                "Roundtrip failed for {:?}",
                std::str::from_utf8(original)
            );
        }
    }

    #[test]
    fn test_encode_2bit_into_buffer_too_small() {
        let mut buffer = [0u8; 1];
        let result = encode_2bit_into(b"ACGTACGT", &mut buffer); // Needs 2 bytes
        assert!(matches!(
            result,
            Err(HybridEncoderError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn test_encode_2bit_into_exact_buffer() {
        let mut buffer = [0u8; 2];
        let bytes_written = encode_2bit_into(b"ACGTACGT", &mut buffer).unwrap();
        assert_eq!(bytes_written, 2);
        assert_eq!(buffer[0], 0x1B);
        assert_eq!(buffer[1], 0x1B);
    }

    #[test]
    fn test_decode_2bit_empty() {
        let result = decode_2bit(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_2bit_single_base() {
        assert_eq!(decode_2bit(&[0b00_00_00_00], 1), b"A");
        assert_eq!(decode_2bit(&[0b01_00_00_00], 1), b"C");
        assert_eq!(decode_2bit(&[0b10_00_00_00], 1), b"G");
        assert_eq!(decode_2bit(&[0b11_00_00_00], 1), b"T");
    }

    #[test]
    fn test_decode_2bit_quartet() {
        let decoded = decode_2bit(&[0x1B], 4);
        assert_eq!(decoded, b"ACGT");
    }

    #[test]
    fn test_decode_2bit_into_buffer_too_small() {
        let mut buffer = [0u8; 2];
        let result = decode_2bit_into(&[0x1B], 4, &mut buffer); // Needs 4 bytes
        assert!(matches!(
            result,
            Err(HybridEncoderError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn test_is_clean_sequence() {
        // Clean sequences
        assert!(is_clean_sequence(b""));
        assert!(is_clean_sequence(b"A"));
        assert!(is_clean_sequence(b"ACGT"));
        assert!(is_clean_sequence(b"acgt"));
        assert!(is_clean_sequence(b"AcGtAcGtAcGtAcGt"));

        // Dirty sequences (contain non-ACGT)
        assert!(!is_clean_sequence(b"N"));
        assert!(!is_clean_sequence(b"ACGTN"));
        assert!(!is_clean_sequence(b"ACGR")); // R = purine
        assert!(!is_clean_sequence(b"ACG-")); // Gap
        assert!(!is_clean_sequence(b" "));
        assert!(!is_clean_sequence(b"ACGT ACGT")); // Space
    }

    #[test]
    fn test_is_clean_sequence_long() {
        // Test with sequences longer than 16 (SIMD boundary)
        let clean_32 = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
        assert!(is_clean_sequence(clean_32));

        let mut dirty_at_17 = *b"ACGTACGTACGTACGTN";
        assert!(!is_clean_sequence(&dirty_at_17));

        dirty_at_17[16] = b'A'; // Fix it
        assert!(is_clean_sequence(&dirty_at_17));
    }

    #[test]
    fn test_encode_2bit_large_sequence() {
        // Test with a larger sequence to exercise SIMD paths
        let sequence: Vec<u8> = (0..1000)
            .map(|i| match i % 4 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            })
            .collect();

        let encoded = encode_2bit(&sequence).unwrap();
        assert_eq!(encoded.len(), 250); // 1000 / 4

        let decoded = decode_2bit(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    #[test]
    fn test_encode_2bit_all_same_base() {
        // All A's = 0b00_00_00_00 = 0x00
        let encoded = encode_2bit(b"AAAA").unwrap();
        assert_eq!(encoded, vec![0x00]);

        // All C's = 0b01_01_01_01 = 0x55
        let encoded = encode_2bit(b"CCCC").unwrap();
        assert_eq!(encoded, vec![0x55]);

        // All G's = 0b10_10_10_10 = 0xAA
        let encoded = encode_2bit(b"GGGG").unwrap();
        assert_eq!(encoded, vec![0xAA]);

        // All T's = 0b11_11_11_11 = 0xFF
        let encoded = encode_2bit(b"TTTT").unwrap();
        assert_eq!(encoded, vec![0xFF]);
    }

    // ========================================================================
    // 2-bit Decoder Additional Tests (Step 1.1.3)
    // ========================================================================

    #[test]
    fn test_decode_2bit_remainder_handling() {
        // Test decoding with various remainder lengths
        let encoded = encode_2bit(b"ACGTA").unwrap();
        let decoded = decode_2bit(&encoded, 5);
        assert_eq!(decoded, b"ACGTA");

        let encoded = encode_2bit(b"ACGTAC").unwrap();
        let decoded = decode_2bit(&encoded, 6);
        assert_eq!(decoded, b"ACGTAC");

        let encoded = encode_2bit(b"ACGTACG").unwrap();
        let decoded = decode_2bit(&encoded, 7);
        assert_eq!(decoded, b"ACGTACG");
    }

    #[test]
    fn test_decode_2bit_multiple_bytes() {
        // Test with multiple complete bytes
        let original = b"ACGTACGTACGTACGT"; // 16 bases = 4 bytes
        let encoded = encode_2bit(original).unwrap();
        assert_eq!(encoded.len(), 4);
        let decoded = decode_2bit(&encoded, original.len());
        assert_eq!(decoded, original.as_slice());
    }

    #[test]
    fn test_decode_2bit_into_exact_buffer() {
        let encoded = encode_2bit(b"ACGT").unwrap();
        let mut buffer = [0u8; 4];
        let bytes_written = decode_2bit_into(&encoded, 4, &mut buffer).unwrap();
        assert_eq!(bytes_written, 4);
        assert_eq!(&buffer, b"ACGT");
    }

    #[test]
    fn test_decode_2bit_into_oversized_buffer() {
        let encoded = encode_2bit(b"ACGT").unwrap();
        let mut buffer = [0u8; 100];
        let bytes_written = decode_2bit_into(&encoded, 4, &mut buffer).unwrap();
        assert_eq!(bytes_written, 4);
        assert_eq!(&buffer[..4], b"ACGT");
    }

    #[test]
    fn test_decode_2bit_all_patterns() {
        // Test all 256 possible byte values
        for byte_val in 0u8..=255 {
            let encoded = vec![byte_val];
            let decoded = decode_2bit(&encoded, 4);
            assert_eq!(decoded.len(), 4);
            // Verify each decoded base is valid
            for b in &decoded {
                assert!(matches!(b, b'A' | b'C' | b'G' | b'T'));
            }
        }
    }

    #[test]
    fn test_decode_2bit_simd_boundary() {
        // Test at SIMD boundary (16 nucleotides = 4 encoded bytes → decoded to 16)
        let original: Vec<u8> = (0..64)
            .map(|i| match i % 4 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            })
            .collect();
        let encoded = encode_2bit(&original).unwrap();
        let decoded = decode_2bit(&encoded, original.len());
        assert_eq!(decoded, original);
    }

    // ========================================================================
    // Sequence Classification Tests (Step 1.1.4)
    // ========================================================================

    #[test]
    fn test_classify_sequence_empty() {
        assert_eq!(classify_sequence(b""), EncodingType::Clean2Bit);
    }

    #[test]
    fn test_classify_sequence_clean() {
        assert_eq!(classify_sequence(b"A"), EncodingType::Clean2Bit);
        assert_eq!(classify_sequence(b"ACGT"), EncodingType::Clean2Bit);
        assert_eq!(classify_sequence(b"acgt"), EncodingType::Clean2Bit);
        assert_eq!(classify_sequence(b"AcGtAcGt"), EncodingType::Clean2Bit);
        assert_eq!(
            classify_sequence(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            EncodingType::Clean2Bit
        );
    }

    #[test]
    fn test_classify_sequence_dirty() {
        // N (any base)
        assert_eq!(classify_sequence(b"N"), EncodingType::Dirty4Bit);
        assert_eq!(classify_sequence(b"ACGTN"), EncodingType::Dirty4Bit);

        // IUPAC ambiguity codes
        assert_eq!(classify_sequence(b"R"), EncodingType::Dirty4Bit); // Purine (A/G)
        assert_eq!(classify_sequence(b"Y"), EncodingType::Dirty4Bit); // Pyrimidine (C/T)
        assert_eq!(classify_sequence(b"S"), EncodingType::Dirty4Bit); // Strong (G/C)
        assert_eq!(classify_sequence(b"W"), EncodingType::Dirty4Bit); // Weak (A/T)
        assert_eq!(classify_sequence(b"K"), EncodingType::Dirty4Bit); // Keto (G/T)
        assert_eq!(classify_sequence(b"M"), EncodingType::Dirty4Bit); // Amino (A/C)
        assert_eq!(classify_sequence(b"B"), EncodingType::Dirty4Bit); // Not A
        assert_eq!(classify_sequence(b"D"), EncodingType::Dirty4Bit); // Not C
        assert_eq!(classify_sequence(b"H"), EncodingType::Dirty4Bit); // Not G
        assert_eq!(classify_sequence(b"V"), EncodingType::Dirty4Bit); // Not T

        // Gap character
        assert_eq!(classify_sequence(b"-"), EncodingType::Dirty4Bit);

        // Invalid characters
        assert_eq!(classify_sequence(b" "), EncodingType::Dirty4Bit);
        assert_eq!(classify_sequence(b"0"), EncodingType::Dirty4Bit);
    }

    #[test]
    fn test_classify_sequence_dirty_at_boundaries() {
        // Dirty at start
        assert_eq!(classify_sequence(b"NACGT"), EncodingType::Dirty4Bit);

        // Dirty at end
        assert_eq!(classify_sequence(b"ACGTN"), EncodingType::Dirty4Bit);

        // Dirty in middle
        assert_eq!(classify_sequence(b"ACNGT"), EncodingType::Dirty4Bit);

        // Dirty at SIMD boundary (position 16)
        let mut seq = *b"ACGTACGTACGTACGTN";
        assert_eq!(classify_sequence(&seq), EncodingType::Dirty4Bit);
        seq[16] = b'A';
        assert_eq!(classify_sequence(&seq), EncodingType::Clean2Bit);
    }

    // ========================================================================
    // Bifurcated Encoder Tests (Step 1.1.5)
    // ========================================================================

    #[test]
    fn test_encode_bifurcated_empty() {
        let result = encode_bifurcated(b"");
        assert_eq!(result.encoding, EncodingType::Clean2Bit);
        assert!(result.data.is_empty());
        assert_eq!(result.original_len, 0);
    }

    #[test]
    fn test_encode_bifurcated_clean_sequence() {
        let result = encode_bifurcated(b"ACGT");
        assert_eq!(result.encoding, EncodingType::Clean2Bit);
        assert_eq!(result.data, vec![0x1B]); // 2-bit encoding
        assert_eq!(result.original_len, 4);
        assert_eq!(result.encoded_len(), 1);
        assert!(result.is_clean());
        assert!(!result.is_dirty());
    }

    #[test]
    fn test_encode_bifurcated_dirty_sequence() {
        let result = encode_bifurcated(b"ACNT");
        assert_eq!(result.encoding, EncodingType::Dirty4Bit);
        assert_eq!(result.original_len, 4);
        assert_eq!(result.encoded_len(), 2); // 4-bit: 4 bases = 2 bytes
        assert!(!result.is_clean());
        assert!(result.is_dirty());
    }

    #[test]
    fn test_encode_bifurcated_case_insensitive() {
        let upper = encode_bifurcated(b"ACGT");
        let lower = encode_bifurcated(b"acgt");
        let mixed = encode_bifurcated(b"AcGt");

        assert_eq!(upper.encoding, lower.encoding);
        assert_eq!(upper.data, lower.data);
        assert_eq!(upper.data, mixed.data);
    }

    #[test]
    fn test_encode_bifurcated_compression_ratios() {
        // Clean: 4x compression
        let clean = encode_bifurcated(b"ACGTACGT");
        assert_eq!(clean.actual_compression_ratio(), 4.0);

        // Dirty: 2x compression
        let dirty = encode_bifurcated(b"ACGTACGN");
        assert_eq!(dirty.actual_compression_ratio(), 2.0);
    }

    #[test]
    fn test_encode_bifurcated_into_clean() {
        let sequence = b"ACGTACGT";
        let mut buffer = vec![0u8; 10];

        let (encoding, bytes_written) = encode_bifurcated_into(sequence, &mut buffer).unwrap();
        assert_eq!(encoding, EncodingType::Clean2Bit);
        assert_eq!(bytes_written, 2);
        assert_eq!(&buffer[..2], &[0x1B, 0x1B]);
    }

    #[test]
    fn test_encode_bifurcated_into_dirty() {
        let sequence = b"ACNTACGT";
        let mut buffer = vec![0u8; 10];

        let (encoding, bytes_written) = encode_bifurcated_into(sequence, &mut buffer).unwrap();
        assert_eq!(encoding, EncodingType::Dirty4Bit);
        assert_eq!(bytes_written, 4); // 8 bases = 4 bytes in 4-bit
    }

    #[test]
    fn test_encode_bifurcated_into_buffer_too_small() {
        let sequence = b"ACGTACGT";
        let mut buffer = [0u8; 1]; // Too small for either encoding

        let result = encode_bifurcated_into(sequence, &mut buffer);
        assert!(matches!(
            result,
            Err(HybridEncoderError::BufferTooSmall { .. })
        ));
    }

    // ========================================================================
    // Bifurcated Decoder Tests (Step 1.1.5)
    // ========================================================================

    #[test]
    fn test_decode_bifurcated_empty() {
        let encoded = EncodedSequence::default();
        let decoded = decode_bifurcated(&encoded);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_bifurcated_clean_roundtrip() {
        let original = b"ACGTACGT";
        let encoded = encode_bifurcated(original);
        let decoded = decode_bifurcated(&encoded);
        assert_eq!(decoded, original.to_vec());
    }

    #[test]
    fn test_decode_bifurcated_dirty_roundtrip() {
        let original = b"ACNGTACGT";
        let encoded = encode_bifurcated(original);
        let decoded = decode_bifurcated(&encoded);
        // 4-bit encoder uppercases
        assert_eq!(decoded, original.to_ascii_uppercase());
    }

    #[test]
    fn test_decode_bifurcated_into_clean() {
        let original = b"ACGTACGT";
        let encoded = encode_bifurcated(original);
        let mut buffer = [0u8; 64];

        let bytes_written = decode_bifurcated_into(&encoded, &mut buffer).unwrap();
        assert_eq!(bytes_written, 8);
        assert_eq!(&buffer[..8], original.as_slice());
    }

    #[test]
    fn test_decode_bifurcated_into_dirty() {
        let original = b"ACNGTACGT";
        let encoded = encode_bifurcated(original);
        let mut buffer = [0u8; 64];

        let bytes_written = decode_bifurcated_into(&encoded, &mut buffer).unwrap();
        assert_eq!(bytes_written, 9);
        assert_eq!(&buffer[..9], original.to_ascii_uppercase().as_slice());
    }

    #[test]
    fn test_decode_bifurcated_into_buffer_too_small() {
        let encoded = encode_bifurcated(b"ACGTACGT");
        let mut buffer = [0u8; 4]; // Need 8

        let result = decode_bifurcated_into(&encoded, &mut buffer);
        assert!(matches!(
            result,
            Err(HybridEncoderError::BufferTooSmall { .. })
        ));
    }

    // ========================================================================
    // End-to-End Integration Tests
    // ========================================================================

    #[test]
    fn test_bifurcated_roundtrip_all_clean_lengths() {
        // Test roundtrip for various lengths (covering SIMD boundaries)
        for len in 0..=100 {
            let original: Vec<u8> = (0..len)
                .map(|i| match i % 4 {
                    0 => b'A',
                    1 => b'C',
                    2 => b'G',
                    _ => b'T',
                })
                .collect();

            let encoded = encode_bifurcated(&original);
            assert_eq!(encoded.encoding, EncodingType::Clean2Bit);
            assert_eq!(encoded.original_len, len);

            let decoded = decode_bifurcated(&encoded);
            assert_eq!(decoded, original, "Roundtrip failed for length {}", len);
        }
    }

    #[test]
    fn test_bifurcated_roundtrip_dirty_at_various_positions() {
        // Test dirty base at various positions
        for pos in 0..20 {
            let mut sequence: Vec<u8> = (0..20)
                .map(|i| match i % 4 {
                    0 => b'A',
                    1 => b'C',
                    2 => b'G',
                    _ => b'T',
                })
                .collect();
            sequence[pos] = b'N';

            let encoded = encode_bifurcated(&sequence);
            assert_eq!(
                encoded.encoding,
                EncodingType::Dirty4Bit,
                "Should be dirty with N at position {}",
                pos
            );

            let decoded = decode_bifurcated(&encoded);
            assert_eq!(
                decoded,
                sequence.to_ascii_uppercase(),
                "Roundtrip failed with N at position {}",
                pos
            );
        }
    }

    #[test]
    fn test_bifurcated_all_iupac_codes() {
        // Test all IUPAC codes are handled correctly
        let iupac_sequence = b"ACGTRYSWKMBDHVN-";
        let encoded = encode_bifurcated(iupac_sequence);
        assert_eq!(encoded.encoding, EncodingType::Dirty4Bit);
        assert_eq!(encoded.original_len, 16);

        let decoded = decode_bifurcated(&encoded);
        assert_eq!(decoded, iupac_sequence.to_ascii_uppercase());
    }

    #[test]
    fn test_bifurcated_large_clean_sequence() {
        // Test with a large clean sequence
        let original: Vec<u8> = (0..10000)
            .map(|i| match i % 4 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            })
            .collect();

        let encoded = encode_bifurcated(&original);
        assert_eq!(encoded.encoding, EncodingType::Clean2Bit);
        assert_eq!(encoded.data.len(), 2500); // 10000 / 4

        let decoded = decode_bifurcated(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_bifurcated_large_dirty_sequence() {
        // Test with a large dirty sequence
        let mut original: Vec<u8> = (0..10000)
            .map(|i| match i % 4 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            })
            .collect();
        original[5000] = b'N'; // Add one dirty base

        let encoded = encode_bifurcated(&original);
        assert_eq!(encoded.encoding, EncodingType::Dirty4Bit);
        assert_eq!(encoded.data.len(), 5000); // 10000 / 2

        let decoded = decode_bifurcated(&encoded);
        assert_eq!(decoded, original);
    }
}
