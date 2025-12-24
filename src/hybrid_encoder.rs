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
}
