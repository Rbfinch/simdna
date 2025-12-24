// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! # Encoding Serialization Module
//!
//! This module provides efficient binary serialization and deserialization for
//! encoded DNA sequences, optimized for database BLOB storage in SQLite.
//!
//! ## Binary Format
//!
//! The serialization format is designed for:
//! - **Minimal overhead**: Only 10 bytes of header per sequence
//! - **Self-describing**: Contains all metadata needed for decoding
//! - **Version-aware**: Format version field for future compatibility
//! - **Checksum verification**: Optional CRC32 for data integrity
//!
//! ### Wire Format (Version 1)
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │ Byte 0: Magic byte (0xDN = 0x44, 0x4E for "DN")                │
//! │ Byte 1: Format version (currently 0x01)                        │
//! │ Byte 2: Encoding type (0x00 = Clean2Bit, 0x01 = Dirty4Bit)    │
//! │ Byte 3: Flags (bit 0 = has_checksum, bits 1-7 reserved)       │
//! │ Bytes 4-7: Original sequence length (little-endian u32)       │
//! │ Bytes 8-11: Encoded data length (little-endian u32)           │
//! │ Bytes 12+: Encoded sequence data                               │
//! │ [Optional] Last 4 bytes: CRC32 checksum (if flags bit 0 set)  │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## BLOB Compatibility
//!
//! The [`BlobWriter`] and [`BlobReader`] types provide high-level interfaces
//! for SQLite BLOB storage, with optional compression and integrity checking.
//!
//! ## Usage
//!
//! ```rust
//! use simdna::hybrid_encoder::{encode_bifurcated, EncodedSequence, EncodingType};
//! use simdna::serialization::{to_bytes, from_bytes, to_blob, from_blob};
//!
//! // Encode a DNA sequence
//! let sequence = b"ACGTACGTACGTACGT";
//! let encoded = encode_bifurcated(sequence);
//!
//! // Serialize to bytes
//! let bytes = to_bytes(&encoded);
//! assert!(bytes.len() > 0);
//!
//! // Deserialize back
//! let decoded = from_bytes(&bytes).unwrap();
//! assert_eq!(decoded.encoding, encoded.encoding);
//! assert_eq!(decoded.original_len, encoded.original_len);
//! assert_eq!(decoded.data, encoded.data);
//!
//! // For database storage with checksum
//! let blob = to_blob(&encoded, true);
//! let from_db = from_blob(&blob).unwrap();
//! assert_eq!(from_db.data, encoded.data);
//! ```
//!
//! ## Thread Safety
//!
//! All functions in this module are thread-safe and can be called concurrently.

use std::fmt;

use crate::hybrid_encoder::{EncodedSequence, EncodingType};

// ============================================================================
// Constants
// ============================================================================

/// Magic bytes identifying a simdna serialized blob ("DN" in ASCII).
const MAGIC_BYTES: [u8; 2] = [0x44, 0x4E]; // 'D', 'N'

/// Current format version.
const FORMAT_VERSION: u8 = 1;

/// Minimum header size in bytes (without checksum).
/// Layout: 2 magic + 1 version + 1 encoding + 1 flags + 1 reserved + 4 original_len + 4 data_len = 14
const HEADER_SIZE: usize = 14;

/// Flag bit: checksum present.
const FLAG_HAS_CHECKSUM: u8 = 0x01;

/// CRC32 checksum size in bytes.
const CHECKSUM_SIZE: usize = 4;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during serialization/deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializationError {
    /// Invalid magic bytes in header.
    InvalidMagic { expected: [u8; 2], found: [u8; 2] },
    /// Unsupported format version.
    UnsupportedVersion { version: u8, max_supported: u8 },
    /// Invalid encoding type value.
    InvalidEncodingType(u8),
    /// Buffer is too small to contain a valid serialized sequence.
    BufferTooSmall { needed: usize, actual: usize },
    /// Checksum verification failed.
    ChecksumMismatch { expected: u32, computed: u32 },
    /// Encoded data length doesn't match header.
    DataLengthMismatch { header: usize, actual: usize },
    /// Original length is inconsistent with encoded data.
    InconsistentLength {
        original_len: usize,
        encoded_len: usize,
        encoding: EncodingType,
    },
}

impl fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerializationError::InvalidMagic { expected, found } => {
                write!(
                    f,
                    "invalid magic bytes: expected {:02X}{:02X}, found {:02X}{:02X}",
                    expected[0], expected[1], found[0], found[1]
                )
            }
            SerializationError::UnsupportedVersion {
                version,
                max_supported,
            } => {
                write!(
                    f,
                    "unsupported format version {} (max supported: {})",
                    version, max_supported
                )
            }
            SerializationError::InvalidEncodingType(v) => {
                write!(f, "invalid encoding type: {} (expected 0 or 1)", v)
            }
            SerializationError::BufferTooSmall { needed, actual } => {
                write!(
                    f,
                    "buffer too small: need {} bytes, got {} bytes",
                    needed, actual
                )
            }
            SerializationError::ChecksumMismatch { expected, computed } => {
                write!(
                    f,
                    "checksum mismatch: expected 0x{:08X}, computed 0x{:08X}",
                    expected, computed
                )
            }
            SerializationError::DataLengthMismatch { header, actual } => {
                write!(
                    f,
                    "data length mismatch: header says {} bytes, but {} bytes present",
                    header, actual
                )
            }
            SerializationError::InconsistentLength {
                original_len,
                encoded_len,
                encoding,
            } => {
                write!(
                    f,
                    "inconsistent length: {} bases encoded as {} bytes using {} encoding",
                    original_len, encoded_len, encoding
                )
            }
        }
    }
}

impl std::error::Error for SerializationError {}

// ============================================================================
// CRC32 Implementation (IEEE polynomial, no external dependencies)
// ============================================================================

/// IEEE 802.3 CRC32 lookup table (generated at compile time).
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let polynomial: u32 = 0xEDB88320; // IEEE polynomial (reversed)
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ polynomial;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Computes CRC32 checksum of a byte slice using IEEE polynomial.
///
/// This is a fast table-driven implementation suitable for data integrity
/// verification in database storage.
///
/// # Examples
///
/// ```rust
/// use simdna::serialization::crc32;
///
/// let data = b"ACGTACGT";
/// let checksum = crc32(data);
/// assert_eq!(crc32(data), checksum); // Deterministic
/// ```
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        let index = ((crc ^ (byte as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    !crc
}

/// Incrementally updates a CRC32 checksum with additional data.
///
/// # Arguments
///
/// * `current` - The current CRC state (use `0xFFFFFFFF` to start)
/// * `data` - Additional data to include
///
/// # Returns
///
/// The updated CRC state. Call with empty slice and invert to finalize.
#[inline]
pub fn crc32_update(current: u32, data: &[u8]) -> u32 {
    let mut crc = current;
    for &byte in data {
        let index = ((crc ^ (byte as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    crc
}

/// Finalizes a CRC32 computation started with `crc32_update`.
#[inline]
pub fn crc32_finalize(crc: u32) -> u32 {
    !crc
}

// ============================================================================
// Header Structure
// ============================================================================

/// Parsed header from a serialized blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SerializedHeader {
    /// Format version number.
    pub version: u8,
    /// Encoding type used.
    pub encoding: EncodingType,
    /// Whether a checksum is present at the end.
    pub has_checksum: bool,
    /// Original sequence length in nucleotides.
    pub original_len: u32,
    /// Encoded data length in bytes.
    pub data_len: u32,
}

impl SerializedHeader {
    /// Returns the total expected blob size including header and optional checksum.
    #[inline]
    pub const fn total_size(&self) -> usize {
        HEADER_SIZE + self.data_len as usize + if self.has_checksum { CHECKSUM_SIZE } else { 0 }
    }
}

// ============================================================================
// Serialization Functions (Step 1.3.1: Binary Serialization Format)
// ============================================================================

/// Serializes an `EncodedSequence` to a byte vector.
///
/// This is the primary serialization function for storing encoded sequences
/// in memory or sending over a network. For database storage with integrity
/// checking, use [`to_blob`] instead.
///
/// # Arguments
///
/// * `encoded` - The encoded sequence to serialize
///
/// # Returns
///
/// A byte vector containing the serialized data (without checksum).
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
/// use simdna::serialization::to_bytes;
///
/// let sequence = b"ACGTACGT";
/// let encoded = encode_bifurcated(sequence);
/// let bytes = to_bytes(&encoded);
///
/// // Verify header structure
/// assert_eq!(bytes[0], 0x44); // 'D'
/// assert_eq!(bytes[1], 0x4E); // 'N'
/// assert_eq!(bytes[2], 0x01); // Version 1
/// assert_eq!(bytes[3], 0x00); // Clean2Bit encoding
/// ```
pub fn to_bytes(encoded: &EncodedSequence) -> Vec<u8> {
    let data_len = encoded.data.len();
    let total_size = HEADER_SIZE + data_len;

    let mut buffer = Vec::with_capacity(total_size);

    // Magic bytes
    buffer.extend_from_slice(&MAGIC_BYTES);

    // Version
    buffer.push(FORMAT_VERSION);

    // Encoding type
    buffer.push(encoded.encoding as u8);

    // Flags (no checksum for to_bytes)
    buffer.push(0x00);

    // Reserved byte for alignment
    buffer.push(0x00);

    // Original length (little-endian u32)
    buffer.extend_from_slice(&(encoded.original_len as u32).to_le_bytes());

    // Data length (little-endian u32)
    buffer.extend_from_slice(&(data_len as u32).to_le_bytes());

    // Encoded data
    buffer.extend_from_slice(&encoded.data);

    buffer
}

/// Serializes an `EncodedSequence` into a pre-allocated buffer.
///
/// This is a zero-allocation variant of [`to_bytes`] for high-throughput
/// pipelines where buffer reuse is important.
///
/// # Arguments
///
/// * `encoded` - The encoded sequence to serialize
/// * `buffer` - Output buffer (must be large enough)
///
/// # Returns
///
/// * `Ok(usize)` - Number of bytes written
/// * `Err(SerializationError)` - If buffer is too small
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
/// use simdna::serialization::{to_bytes_into, required_serialized_len};
///
/// let sequence = b"ACGTACGT";
/// let encoded = encode_bifurcated(sequence);
///
/// let mut buffer = vec![0u8; required_serialized_len(&encoded, false)];
/// let written = to_bytes_into(&encoded, &mut buffer).unwrap();
/// assert_eq!(written, buffer.len());
/// ```
pub fn to_bytes_into(
    encoded: &EncodedSequence,
    buffer: &mut [u8],
) -> Result<usize, SerializationError> {
    let data_len = encoded.data.len();
    let total_size = HEADER_SIZE + data_len;

    if buffer.len() < total_size {
        return Err(SerializationError::BufferTooSmall {
            needed: total_size,
            actual: buffer.len(),
        });
    }

    // Magic bytes
    buffer[0] = MAGIC_BYTES[0];
    buffer[1] = MAGIC_BYTES[1];

    // Version
    buffer[2] = FORMAT_VERSION;

    // Encoding type
    buffer[3] = encoded.encoding as u8;

    // Flags (no checksum)
    buffer[4] = 0x00;

    // Reserved
    buffer[5] = 0x00;

    // Original length
    buffer[6..10].copy_from_slice(&(encoded.original_len as u32).to_le_bytes());

    // Data length
    buffer[10..14].copy_from_slice(&(data_len as u32).to_le_bytes());

    // Encoded data
    buffer[14..14 + data_len].copy_from_slice(&encoded.data);

    Ok(total_size)
}

/// Deserializes an `EncodedSequence` from a byte slice.
///
/// # Arguments
///
/// * `data` - Serialized data (from [`to_bytes`] or [`to_blob`])
///
/// # Returns
///
/// * `Ok(EncodedSequence)` - The deserialized sequence
/// * `Err(SerializationError)` - If data is invalid or corrupted
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
/// use simdna::serialization::{to_bytes, from_bytes};
///
/// let sequence = b"ACGTACGT";
/// let encoded = encode_bifurcated(sequence);
/// let bytes = to_bytes(&encoded);
///
/// let decoded = from_bytes(&bytes).unwrap();
/// assert_eq!(decoded.encoding, encoded.encoding);
/// assert_eq!(decoded.original_len, encoded.original_len);
/// assert_eq!(decoded.data, encoded.data);
/// ```
pub fn from_bytes(data: &[u8]) -> Result<EncodedSequence, SerializationError> {
    // Parse header first
    let header = parse_header(data)?;

    // Validate total size
    let expected_size = header.total_size();
    if data.len() < expected_size {
        return Err(SerializationError::BufferTooSmall {
            needed: expected_size,
            actual: data.len(),
        });
    }

    // Extract encoded data
    let data_start = HEADER_SIZE;
    let data_end = data_start + header.data_len as usize;
    let encoded_data = data[data_start..data_end].to_vec();

    // Verify checksum if present
    if header.has_checksum {
        let checksum_start = data_end;
        let checksum_end = checksum_start + CHECKSUM_SIZE;

        if data.len() < checksum_end {
            return Err(SerializationError::BufferTooSmall {
                needed: checksum_end,
                actual: data.len(),
            });
        }

        let stored_checksum = u32::from_le_bytes([
            data[checksum_start],
            data[checksum_start + 1],
            data[checksum_start + 2],
            data[checksum_start + 3],
        ]);

        // Compute checksum over header + data (excluding the checksum itself)
        let computed_checksum = crc32(&data[..data_end]);

        if stored_checksum != computed_checksum {
            return Err(SerializationError::ChecksumMismatch {
                expected: stored_checksum,
                computed: computed_checksum,
            });
        }
    }

    // Validate consistency
    validate_length_consistency(
        header.original_len as usize,
        encoded_data.len(),
        header.encoding,
    )?;

    Ok(EncodedSequence {
        encoding: header.encoding,
        data: encoded_data,
        original_len: header.original_len as usize,
    })
}

/// Parses the header from a serialized blob.
///
/// This is useful for inspecting metadata without fully deserializing.
///
/// # Arguments
///
/// * `data` - At least [`HEADER_SIZE`] bytes of serialized data
///
/// # Returns
///
/// * `Ok(SerializedHeader)` - The parsed header
/// * `Err(SerializationError)` - If header is invalid
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
/// use simdna::serialization::{to_bytes, parse_header};
///
/// let sequence = b"ACGTACGT";
/// let encoded = encode_bifurcated(sequence);
/// let bytes = to_bytes(&encoded);
///
/// let header = parse_header(&bytes).unwrap();
/// assert_eq!(header.encoding, EncodingType::Clean2Bit);
/// assert_eq!(header.original_len, 8);
/// assert!(!header.has_checksum);
/// ```
pub fn parse_header(data: &[u8]) -> Result<SerializedHeader, SerializationError> {
    if data.len() < HEADER_SIZE {
        return Err(SerializationError::BufferTooSmall {
            needed: HEADER_SIZE,
            actual: data.len(),
        });
    }

    // Validate magic bytes
    let magic = [data[0], data[1]];
    if magic != MAGIC_BYTES {
        return Err(SerializationError::InvalidMagic {
            expected: MAGIC_BYTES,
            found: magic,
        });
    }

    // Validate version
    let version = data[2];
    if version > FORMAT_VERSION {
        return Err(SerializationError::UnsupportedVersion {
            version,
            max_supported: FORMAT_VERSION,
        });
    }

    // Parse encoding type
    let encoding_byte = data[3];
    let encoding = EncodingType::from_u8(encoding_byte)
        .ok_or(SerializationError::InvalidEncodingType(encoding_byte))?;

    // Parse flags
    let flags = data[4];
    let has_checksum = (flags & FLAG_HAS_CHECKSUM) != 0;

    // Parse lengths (bytes 6-9 for original_len, 10-13 for data_len)
    let original_len = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);
    let data_len = u32::from_le_bytes([data[10], data[11], data[12], data[13]]);

    Ok(SerializedHeader {
        version,
        encoding,
        has_checksum,
        original_len,
        data_len,
    })
}

/// Returns the required buffer size for serialization.
///
/// # Arguments
///
/// * `encoded` - The encoded sequence to serialize
/// * `with_checksum` - Whether to include CRC32 checksum
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
/// use simdna::serialization::required_serialized_len;
///
/// let sequence = b"ACGTACGT";
/// let encoded = encode_bifurcated(sequence);
///
/// // Without checksum: 14 byte header + 2 bytes data = 16 bytes
/// assert_eq!(required_serialized_len(&encoded, false), 16);
///
/// // With checksum: adds 4 bytes
/// assert_eq!(required_serialized_len(&encoded, true), 20);
/// ```
#[inline]
pub fn required_serialized_len(encoded: &EncodedSequence, with_checksum: bool) -> usize {
    HEADER_SIZE + encoded.data.len() + if with_checksum { CHECKSUM_SIZE } else { 0 }
}

// ============================================================================
// BLOB Compatibility Layer (Step 1.3.2)
// ============================================================================

/// Serializes an `EncodedSequence` to a BLOB-ready byte vector with optional checksum.
///
/// This is the recommended function for database storage as it includes
/// CRC32 integrity checking.
///
/// # Arguments
///
/// * `encoded` - The encoded sequence to serialize
/// * `with_checksum` - Whether to include CRC32 checksum for integrity verification
///
/// # Returns
///
/// A byte vector ready for SQLite BLOB storage.
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
/// use simdna::serialization::{to_blob, from_blob};
///
/// let sequence = b"ACGTACGT";
/// let encoded = encode_bifurcated(sequence);
///
/// // Serialize with checksum for database storage
/// let blob = to_blob(&encoded, true);
///
/// // Deserialize and verify integrity
/// let decoded = from_blob(&blob).unwrap();
/// assert_eq!(decoded.data, encoded.data);
/// ```
pub fn to_blob(encoded: &EncodedSequence, with_checksum: bool) -> Vec<u8> {
    let data_len = encoded.data.len();
    let total_size = HEADER_SIZE + data_len + if with_checksum { CHECKSUM_SIZE } else { 0 };

    let mut buffer = Vec::with_capacity(total_size);

    // Magic bytes
    buffer.extend_from_slice(&MAGIC_BYTES);

    // Version
    buffer.push(FORMAT_VERSION);

    // Encoding type
    buffer.push(encoded.encoding as u8);

    // Flags
    let flags = if with_checksum { FLAG_HAS_CHECKSUM } else { 0 };
    buffer.push(flags);

    // Reserved byte
    buffer.push(0x00);

    // Original length (little-endian u32)
    buffer.extend_from_slice(&(encoded.original_len as u32).to_le_bytes());

    // Data length (little-endian u32)
    buffer.extend_from_slice(&(data_len as u32).to_le_bytes());

    // Encoded data
    buffer.extend_from_slice(&encoded.data);

    // Checksum if requested
    if with_checksum {
        let checksum = crc32(&buffer);
        buffer.extend_from_slice(&checksum.to_le_bytes());
    }

    buffer
}

/// Serializes an `EncodedSequence` into a pre-allocated buffer with optional checksum.
///
/// Zero-allocation variant of [`to_blob`].
///
/// # Arguments
///
/// * `encoded` - The encoded sequence to serialize
/// * `buffer` - Output buffer (must be large enough)
/// * `with_checksum` - Whether to include CRC32 checksum
///
/// # Returns
///
/// * `Ok(usize)` - Number of bytes written
/// * `Err(SerializationError)` - If buffer is too small
pub fn to_blob_into(
    encoded: &EncodedSequence,
    buffer: &mut [u8],
    with_checksum: bool,
) -> Result<usize, SerializationError> {
    let data_len = encoded.data.len();
    let total_size = HEADER_SIZE + data_len + if with_checksum { CHECKSUM_SIZE } else { 0 };

    if buffer.len() < total_size {
        return Err(SerializationError::BufferTooSmall {
            needed: total_size,
            actual: buffer.len(),
        });
    }

    // Magic bytes
    buffer[0] = MAGIC_BYTES[0];
    buffer[1] = MAGIC_BYTES[1];

    // Version
    buffer[2] = FORMAT_VERSION;

    // Encoding type
    buffer[3] = encoded.encoding as u8;

    // Flags
    buffer[4] = if with_checksum { FLAG_HAS_CHECKSUM } else { 0 };

    // Reserved
    buffer[5] = 0x00;

    // Original length
    buffer[6..10].copy_from_slice(&(encoded.original_len as u32).to_le_bytes());

    // Data length
    buffer[10..14].copy_from_slice(&(data_len as u32).to_le_bytes());

    // Encoded data
    buffer[14..14 + data_len].copy_from_slice(&encoded.data);

    // Checksum if requested
    if with_checksum {
        let checksum = crc32(&buffer[..14 + data_len]);
        buffer[14 + data_len..total_size].copy_from_slice(&checksum.to_le_bytes());
    }

    Ok(total_size)
}

/// Deserializes an `EncodedSequence` from a BLOB.
///
/// Automatically detects and verifies checksum if present.
///
/// # Arguments
///
/// * `blob` - BLOB data from database
///
/// # Returns
///
/// * `Ok(EncodedSequence)` - The deserialized sequence
/// * `Err(SerializationError)` - If data is invalid or checksum fails
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
/// use simdna::serialization::{to_blob, from_blob};
///
/// let sequence = b"ACGTACGT";
/// let encoded = encode_bifurcated(sequence);
/// let blob = to_blob(&encoded, true);
///
/// let decoded = from_blob(&blob).unwrap();
/// assert_eq!(decoded.original_len, 8);
/// ```
pub fn from_blob(blob: &[u8]) -> Result<EncodedSequence, SerializationError> {
    // Use the same deserialization logic as from_bytes
    from_bytes(blob)
}

/// Validates a BLOB without fully deserializing.
///
/// This is useful for checking data integrity before full deserialization.
///
/// # Arguments
///
/// * `blob` - BLOB data to validate
///
/// # Returns
///
/// * `Ok(SerializedHeader)` - The header if valid
/// * `Err(SerializationError)` - If validation fails
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
/// use simdna::serialization::{to_blob, validate_blob};
///
/// let sequence = b"ACGTACGT";
/// let encoded = encode_bifurcated(sequence);
/// let blob = to_blob(&encoded, true);
///
/// let header = validate_blob(&blob).unwrap();
/// assert!(header.has_checksum);
/// ```
pub fn validate_blob(blob: &[u8]) -> Result<SerializedHeader, SerializationError> {
    let header = parse_header(blob)?;

    // Validate total size
    let expected_size = header.total_size();
    if blob.len() < expected_size {
        return Err(SerializationError::BufferTooSmall {
            needed: expected_size,
            actual: blob.len(),
        });
    }

    // Verify checksum if present
    if header.has_checksum {
        let data_end = HEADER_SIZE + header.data_len as usize;
        let checksum_start = data_end;

        let stored_checksum = u32::from_le_bytes([
            blob[checksum_start],
            blob[checksum_start + 1],
            blob[checksum_start + 2],
            blob[checksum_start + 3],
        ]);

        let computed_checksum = crc32(&blob[..data_end]);

        if stored_checksum != computed_checksum {
            return Err(SerializationError::ChecksumMismatch {
                expected: stored_checksum,
                computed: computed_checksum,
            });
        }
    }

    // Validate length consistency
    validate_length_consistency(
        header.original_len as usize,
        header.data_len as usize,
        header.encoding,
    )?;

    Ok(header)
}

// ============================================================================
// Batch Serialization (for high-throughput pipelines)
// ============================================================================

/// Estimates the total buffer size needed for serializing multiple sequences.
///
/// # Arguments
///
/// * `sequences` - Iterator of sequences to serialize
/// * `with_checksum` - Whether checksums will be included
///
/// # Returns
///
/// Total bytes needed for all serialized sequences.
pub fn estimate_batch_size<'a, I>(sequences: I, with_checksum: bool) -> usize
where
    I: Iterator<Item = &'a EncodedSequence>,
{
    sequences
        .map(|s| required_serialized_len(s, with_checksum))
        .sum()
}

/// Serializes multiple sequences into a single buffer.
///
/// Returns offsets and lengths for each serialized sequence.
///
/// # Arguments
///
/// * `sequences` - Sequences to serialize
/// * `with_checksum` - Whether to include checksums
///
/// # Returns
///
/// Tuple of (buffer, offsets) where offsets[i] is (start, length) for sequence i.
///
/// # Examples
///
/// ```rust
/// use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
/// use simdna::serialization::serialize_batch;
///
/// let seq1 = encode_bifurcated(b"ACGT");
/// let seq2 = encode_bifurcated(b"GCTA");
/// let sequences = vec![seq1, seq2];
///
/// let (buffer, offsets) = serialize_batch(&sequences, false);
/// assert_eq!(offsets.len(), 2);
/// ```
pub fn serialize_batch(
    sequences: &[EncodedSequence],
    with_checksum: bool,
) -> (Vec<u8>, Vec<(usize, usize)>) {
    let total_size = estimate_batch_size(sequences.iter(), with_checksum);
    let mut buffer = Vec::with_capacity(total_size);
    let mut offsets = Vec::with_capacity(sequences.len());

    for seq in sequences {
        let start = buffer.len();
        let blob = if with_checksum {
            to_blob(seq, true)
        } else {
            to_bytes(seq)
        };
        let len = blob.len();
        buffer.extend_from_slice(&blob);
        offsets.push((start, len));
    }

    (buffer, offsets)
}

/// Deserializes multiple sequences from a batch buffer.
///
/// # Arguments
///
/// * `buffer` - Buffer containing serialized sequences
/// * `offsets` - Start positions and lengths from `serialize_batch`
///
/// # Returns
///
/// Vector of deserialized sequences.
pub fn deserialize_batch(
    buffer: &[u8],
    offsets: &[(usize, usize)],
) -> Result<Vec<EncodedSequence>, SerializationError> {
    offsets
        .iter()
        .map(|&(start, len)| {
            let end = start + len;
            if end > buffer.len() {
                return Err(SerializationError::BufferTooSmall {
                    needed: end,
                    actual: buffer.len(),
                });
            }
            from_bytes(&buffer[start..end])
        })
        .collect()
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validates that the original length is consistent with the encoded length.
fn validate_length_consistency(
    original_len: usize,
    encoded_len: usize,
    encoding: EncodingType,
) -> Result<(), SerializationError> {
    let expected_encoded_len = match encoding {
        EncodingType::Clean2Bit => original_len.div_ceil(4),
        EncodingType::Dirty4Bit => original_len.div_ceil(2),
    };

    // Allow exact match or empty for empty sequences
    if original_len == 0 && encoded_len == 0 {
        return Ok(());
    }

    if encoded_len != expected_encoded_len {
        return Err(SerializationError::InconsistentLength {
            original_len,
            encoded_len,
            encoding,
        });
    }

    Ok(())
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_encoder::encode_bifurcated;

    // ------------------------------------------------------------------------
    // CRC32 Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(&[]), 0x00000000);
    }

    #[test]
    fn test_crc32_known_values() {
        // Known test vectors for IEEE CRC32
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_crc32_deterministic() {
        let data = b"ACGTACGTACGTACGT";
        let crc1 = crc32(data);
        let crc2 = crc32(data);
        assert_eq!(crc1, crc2);
    }

    #[test]
    fn test_crc32_incremental() {
        let data = b"Hello, World!";
        let full_crc = crc32(data);

        // Compute incrementally
        let state = crc32_update(0xFFFFFFFF, &data[..5]);
        let state = crc32_update(state, &data[5..]);
        let incremental_crc = crc32_finalize(state);

        assert_eq!(full_crc, incremental_crc);
    }

    // ------------------------------------------------------------------------
    // Header Parsing Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_parse_header_valid() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);
        let bytes = to_bytes(&encoded);

        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.version, FORMAT_VERSION);
        assert_eq!(header.encoding, EncodingType::Clean2Bit);
        assert!(!header.has_checksum);
        assert_eq!(header.original_len, 8);
        assert_eq!(header.data_len, 2);
    }

    #[test]
    fn test_parse_header_invalid_magic() {
        let data = vec![
            0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let result = parse_header(&data);
        assert!(matches!(
            result,
            Err(SerializationError::InvalidMagic { .. })
        ));
    }

    #[test]
    fn test_parse_header_unsupported_version() {
        let mut data = vec![
            0x44, 0x4E, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        data[2] = 0xFF; // Invalid version
        let result = parse_header(&data);
        assert!(matches!(
            result,
            Err(SerializationError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn test_parse_header_invalid_encoding() {
        let mut data = vec![
            0x44, 0x4E, 0x01, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        data[3] = 0xFF; // Invalid encoding
        let result = parse_header(&data);
        assert!(matches!(
            result,
            Err(SerializationError::InvalidEncodingType(_))
        ));
    }

    #[test]
    fn test_parse_header_buffer_too_small() {
        let data = vec![0x44, 0x4E, 0x01]; // Only 3 bytes
        let result = parse_header(&data);
        assert!(matches!(
            result,
            Err(SerializationError::BufferTooSmall { .. })
        ));
    }

    // ------------------------------------------------------------------------
    // Basic Serialization Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_to_bytes_clean_sequence() {
        let sequence = b"ACGT";
        let encoded = encode_bifurcated(sequence);
        let bytes = to_bytes(&encoded);

        // 14-byte header + 1 byte data
        assert_eq!(bytes.len(), 15);

        // Verify magic
        assert_eq!(&bytes[0..2], &MAGIC_BYTES);

        // Verify version
        assert_eq!(bytes[2], FORMAT_VERSION);

        // Verify encoding type
        assert_eq!(bytes[3], EncodingType::Clean2Bit as u8);
    }

    #[test]
    fn test_roundtrip_clean_sequence() {
        let sequence = b"ACGTACGTACGTACGT";
        let encoded = encode_bifurcated(sequence);
        let bytes = to_bytes(&encoded);
        let decoded = from_bytes(&bytes).unwrap();

        assert_eq!(decoded.encoding, encoded.encoding);
        assert_eq!(decoded.original_len, encoded.original_len);
        assert_eq!(decoded.data, encoded.data);
    }

    #[test]
    fn test_roundtrip_dirty_sequence() {
        let sequence = b"ACNGTACGT"; // Contains N
        let encoded = encode_bifurcated(sequence);
        let bytes = to_bytes(&encoded);
        let decoded = from_bytes(&bytes).unwrap();

        assert_eq!(decoded.encoding, EncodingType::Dirty4Bit);
        assert_eq!(decoded.original_len, encoded.original_len);
        assert_eq!(decoded.data, encoded.data);
    }

    #[test]
    fn test_roundtrip_empty_sequence() {
        let sequence = b"";
        let encoded = encode_bifurcated(sequence);
        let bytes = to_bytes(&encoded);
        let decoded = from_bytes(&bytes).unwrap();

        assert_eq!(decoded.original_len, 0);
        assert!(decoded.data.is_empty());
    }

    #[test]
    fn test_roundtrip_single_base() {
        for base in b"ACGT" {
            let sequence = [*base];
            let encoded = encode_bifurcated(&sequence);
            let bytes = to_bytes(&encoded);
            let decoded = from_bytes(&bytes).unwrap();

            assert_eq!(decoded.original_len, 1);
            assert_eq!(decoded.data.len(), 1);
        }
    }

    #[test]
    fn test_to_bytes_into_success() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);

        let mut buffer = vec![0u8; 100];
        let written = to_bytes_into(&encoded, &mut buffer).unwrap();

        assert_eq!(written, HEADER_SIZE + encoded.data.len());
    }

    #[test]
    fn test_to_bytes_into_buffer_too_small() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);

        let mut buffer = vec![0u8; 5]; // Too small
        let result = to_bytes_into(&encoded, &mut buffer);

        assert!(matches!(
            result,
            Err(SerializationError::BufferTooSmall { .. })
        ));
    }

    // ------------------------------------------------------------------------
    // BLOB Tests with Checksum
    // ------------------------------------------------------------------------

    #[test]
    fn test_to_blob_with_checksum() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);
        let blob = to_blob(&encoded, true);

        // 14-byte header + 2 bytes data + 4 bytes checksum
        assert_eq!(blob.len(), HEADER_SIZE + encoded.data.len() + CHECKSUM_SIZE);

        // Verify checksum flag is set
        assert_eq!(blob[4] & FLAG_HAS_CHECKSUM, FLAG_HAS_CHECKSUM);
    }

    #[test]
    fn test_to_blob_without_checksum() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);
        let blob = to_blob(&encoded, false);

        // 14-byte header + 2 bytes data (no checksum)
        assert_eq!(blob.len(), HEADER_SIZE + encoded.data.len());

        // Verify checksum flag is not set
        assert_eq!(blob[4] & FLAG_HAS_CHECKSUM, 0);
    }

    #[test]
    fn test_blob_roundtrip_with_checksum() {
        let sequence = b"ACGTACGTACGTACGT";
        let encoded = encode_bifurcated(sequence);
        let blob = to_blob(&encoded, true);
        let decoded = from_blob(&blob).unwrap();

        assert_eq!(decoded.encoding, encoded.encoding);
        assert_eq!(decoded.original_len, encoded.original_len);
        assert_eq!(decoded.data, encoded.data);
    }

    #[test]
    fn test_blob_corrupted_data() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);
        let mut blob = to_blob(&encoded, true);

        // Corrupt a data byte
        let data_start = HEADER_SIZE;
        blob[data_start] ^= 0xFF;

        // Deserialization should fail with checksum mismatch
        let result = from_blob(&blob);
        assert!(matches!(
            result,
            Err(SerializationError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn test_blob_corrupted_checksum() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);
        let mut blob = to_blob(&encoded, true);

        // Corrupt the checksum bytes
        let checksum_start = blob.len() - CHECKSUM_SIZE;
        blob[checksum_start] ^= 0xFF;

        let result = from_blob(&blob);
        assert!(matches!(
            result,
            Err(SerializationError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_blob_success() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);
        let blob = to_blob(&encoded, true);

        let header = validate_blob(&blob).unwrap();
        assert!(header.has_checksum);
        assert_eq!(header.encoding, EncodingType::Clean2Bit);
    }

    #[test]
    fn test_validate_blob_corrupted() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);
        let mut blob = to_blob(&encoded, true);

        blob[HEADER_SIZE] ^= 0xFF; // Corrupt data

        let result = validate_blob(&blob);
        assert!(matches!(
            result,
            Err(SerializationError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn test_to_blob_into_success() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);

        let mut buffer = vec![0u8; 100];
        let written = to_blob_into(&encoded, &mut buffer, true).unwrap();

        assert_eq!(written, HEADER_SIZE + encoded.data.len() + CHECKSUM_SIZE);
    }

    // ------------------------------------------------------------------------
    // Required Length Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_required_serialized_len() {
        let sequence = b"ACGTACGT";
        let encoded = encode_bifurcated(sequence);

        let without_checksum = required_serialized_len(&encoded, false);
        let with_checksum = required_serialized_len(&encoded, true);

        assert_eq!(with_checksum - without_checksum, CHECKSUM_SIZE);
        assert_eq!(without_checksum, HEADER_SIZE + encoded.data.len());
    }

    // ------------------------------------------------------------------------
    // Batch Serialization Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_serialize_batch() {
        let seq1 = encode_bifurcated(b"ACGT");
        let seq2 = encode_bifurcated(b"GCTA");
        let seq3 = encode_bifurcated(b"TTTT");
        let sequences = vec![seq1.clone(), seq2.clone(), seq3.clone()];

        let (buffer, offsets) = serialize_batch(&sequences, false);

        assert_eq!(offsets.len(), 3);
        assert!(!buffer.is_empty());

        // Verify each offset points to valid data
        for (i, &(start, len)) in offsets.iter().enumerate() {
            let end = start + len;
            assert!(end <= buffer.len());

            let decoded = from_bytes(&buffer[start..end]).unwrap();
            assert_eq!(decoded.data, sequences[i].data);
        }
    }

    #[test]
    fn test_deserialize_batch() {
        let seq1 = encode_bifurcated(b"ACGT");
        let seq2 = encode_bifurcated(b"GCTA");
        let sequences = vec![seq1.clone(), seq2.clone()];

        let (buffer, offsets) = serialize_batch(&sequences, true);
        let deserialized = deserialize_batch(&buffer, &offsets).unwrap();

        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized[0].data, seq1.data);
        assert_eq!(deserialized[1].data, seq2.data);
    }

    #[test]
    fn test_estimate_batch_size() {
        let seq1 = encode_bifurcated(b"ACGT");
        let seq2 = encode_bifurcated(b"GCTA");
        let sequences = vec![seq1, seq2];

        let estimated = estimate_batch_size(sequences.iter(), false);
        let (buffer, _) = serialize_batch(&sequences, false);

        assert_eq!(estimated, buffer.len());
    }

    // ------------------------------------------------------------------------
    // Edge Cases and Large Sequences
    // ------------------------------------------------------------------------

    #[test]
    fn test_large_sequence() {
        // Create a 10KB sequence
        let sequence: Vec<u8> = (0..10000).map(|i| b"ACGT"[i % 4]).collect();
        let encoded = encode_bifurcated(&sequence);
        let blob = to_blob(&encoded, true);
        let decoded = from_blob(&blob).unwrap();

        assert_eq!(decoded.original_len, 10000);
        assert_eq!(decoded.data, encoded.data);
    }

    #[test]
    fn test_various_lengths() {
        // Test sequences of various lengths to check remainder handling
        for len in 0..=20 {
            let sequence: Vec<u8> = (0..len).map(|i| b"ACGT"[i % 4]).collect();
            let encoded = encode_bifurcated(&sequence);
            let bytes = to_bytes(&encoded);
            let decoded = from_bytes(&bytes).unwrap();

            assert_eq!(decoded.original_len, len);
            assert_eq!(decoded.data.len(), encoded.data.len());
        }
    }

    #[test]
    fn test_all_iupac_codes() {
        // Sequence with all IUPAC codes
        let sequence = b"ACGTRYSWKMBDHVN-";
        let encoded = encode_bifurcated(sequence);
        let blob = to_blob(&encoded, true);
        let decoded = from_blob(&blob).unwrap();

        assert_eq!(decoded.encoding, EncodingType::Dirty4Bit);
        assert_eq!(decoded.original_len, 16);
        assert_eq!(decoded.data, encoded.data);
    }

    // ------------------------------------------------------------------------
    // Length Consistency Validation
    // ------------------------------------------------------------------------

    #[test]
    fn test_inconsistent_length_detection() {
        // Manually create invalid data with wrong encoded length
        let blob = vec![
            0x44, 0x4E, // Magic
            0x01, // Version
            0x00, // Clean2Bit
            0x00, // No checksum
            0x00, // Reserved
            0x10, 0x00, 0x00, 0x00, // Original len = 16
            0x01, 0x00, 0x00, 0x00, // Data len = 1 (should be 4 for 16 bases)
            0xAB, // 1 byte of data
        ];

        let result = from_bytes(&blob);
        assert!(matches!(
            result,
            Err(SerializationError::InconsistentLength { .. })
        ));
    }

    // ------------------------------------------------------------------------
    // Display Trait Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let err = SerializationError::InvalidMagic {
            expected: MAGIC_BYTES,
            found: [0x00, 0x00],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("magic"));

        let err = SerializationError::ChecksumMismatch {
            expected: 0x12345678,
            computed: 0xABCDEF00,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("checksum"));
    }
}
