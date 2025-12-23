// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! # FASTQ Quality Score Encoding Library
//!
//! High-performance FASTQ quality score encoding and decoding using SIMD instructions
//! with automatic fallback to scalar implementations.
//!
//! ## Encoding Pipeline
//!
//! The encoding process consists of three stages:
//! 1. **Phred Conversion**: Convert ASCII quality scores to Phred values (supports Phred+33 and Phred+64)
//! 2. **Binning**: Reduce Phred scores to 4 quality levels (2 bits each)
//! 3. **Run-Length Encoding**: Compress consecutive identical bins
//!
//! ## Binning Scheme
//!
//! Quality scores are binned into 4 levels for efficient compression:
//!
//! | Phred Range | Bin | Representative Q | Meaning         |
//! |-------------|-----|------------------|-----------------|
//! | Q0-9        | 0   | Q6               | Very low quality|
//! | Q10-19      | 1   | Q15              | Low quality     |
//! | Q20-29      | 2   | Q25              | Medium quality  |
//! | Q30+        | 3   | Q37              | High quality    |
//!
//! ## Phred Encoding Support
//!
//! | Format      | ASCII Offset | Phred Range | Common Usage            |
//! |-------------|--------------|-------------|-------------------------|
//! | Phred+33    | 33           | 0-41        | Illumina 1.8+, Sanger   |
//! | Phred+64    | 64           | 0-40        | Illumina 1.3-1.7        |
//!
//! ## RLE Packing Format
//!
//! Runs are packed efficiently:
//! - **Short runs (≤63)**: `[bin:2][length:6]` = 1 byte
//! - **Long runs (>63)**: `0xFF` escape + `[bin:8]` + `[length:16]` = 4 bytes
//!
//! ## Platform Support
//!
//! The library automatically detects and uses the best available SIMD instructions:
//! - **x86_64**: Uses SSE2/SSE4.1 instructions when available
//! - **aarch64 (ARM64)**: Uses NEON instructions (always available on ARM64)
//! - **Other platforms**: Falls back to optimized scalar implementation
//!
//! ## Thread Safety
//!
//! All public functions in this module are thread-safe and can be safely called
//! from multiple threads concurrently.
//!
//! ## Example Usage
//!
//! ```rust
//! use simdna::quality_encoder::{
//!     encode_quality_scores, decode_quality_scores, PhredEncoding
//! };
//!
//! // Encode quality scores (Phred+33 is default)
//! let quality = b"IIIIIIIIIIFFFFFFFF@@@@@@@@@55555";
//! let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
//!
//! // Decode back (with representative bin values)
//! let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred33);
//! assert_eq!(decoded.len(), quality.len());
//! ```
//!
//! ## Compression Ratios
//!
//! Typical compression ratios for Illumina quality strings:
//! - **High-quality reads**: 85-95% compression (mostly Q30+ = bin 3)
//! - **Mixed quality**: 70-85% compression
//! - **Low-quality reads**: 60-75% compression
//!
//! For a typical 150bp Illumina read with good quality, expect 10-20 bytes
//! of compressed output vs 150 bytes of ASCII quality scores.

// Allow unsafe operations in unsafe functions (Rust 2024 compatibility)
#![allow(unsafe_op_in_unsafe_fn)]

use std::fmt;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during quality score operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityError {
    /// Output buffer is too small to hold the result.
    BufferTooSmall {
        /// Number of bytes needed.
        needed: usize,
        /// Number of bytes provided.
        actual: usize,
    },
    /// Invalid encoded data format.
    InvalidEncodedData,
    /// Quality score out of valid range.
    InvalidQualityScore {
        /// The invalid score value.
        score: u8,
        /// The encoding being used.
        encoding: PhredEncoding,
    },
}

impl fmt::Display for QualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall { needed, actual } => {
                write!(f, "buffer too small: needed {needed} bytes, got {actual}")
            }
            Self::InvalidEncodedData => write!(f, "invalid encoded quality data"),
            Self::InvalidQualityScore { score, encoding } => {
                write!(
                    f,
                    "invalid quality score {} for {:?} encoding",
                    score, encoding
                )
            }
        }
    }
}

impl std::error::Error for QualityError {}

// ============================================================================
// Phred Encoding Types
// ============================================================================

/// Phred quality score encoding format.
///
/// Different sequencing platforms use different ASCII offsets for quality scores:
/// - **Phred+33**: Modern Illumina (1.8+), Sanger, most current platforms
/// - **Phred+64**: Older Illumina (1.3-1.7)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhredEncoding {
    /// Phred+33 encoding (ASCII 33-74 → Phred 0-41)
    /// Used by: Illumina 1.8+, Sanger, PacBio, ONT
    #[default]
    Phred33,
    /// Phred+64 encoding (ASCII 64-104 → Phred 0-40)
    /// Used by: Illumina 1.3-1.7
    Phred64,
}

impl PhredEncoding {
    /// Returns the ASCII offset for this encoding.
    #[inline]
    pub const fn offset(self) -> u8 {
        match self {
            Self::Phred33 => 33,
            Self::Phred64 => 64,
        }
    }

    /// Returns the minimum valid ASCII character for this encoding.
    #[inline]
    pub const fn min_char(self) -> u8 {
        self.offset()
    }

    /// Returns the maximum valid ASCII character for this encoding.
    #[inline]
    pub const fn max_char(self) -> u8 {
        match self {
            Self::Phred33 => 126, // Phred 93 max (theoretical)
            Self::Phred64 => 126, // Phred 62 max (theoretical)
        }
    }

    /// Attempts to detect the encoding from a quality string.
    ///
    /// Returns `None` if the encoding cannot be determined (e.g., all scores
    /// are in the overlapping range).
    ///
    /// # Detection Logic
    /// - If any character < 59 (ASCII ';'): Must be Phred+33
    /// - If any character > 74 (ASCII 'J'): Must be Phred+64
    /// - Otherwise: Ambiguous, returns `None`
    pub fn detect(quality: &[u8]) -> Option<Self> {
        let mut has_low = false;
        let mut has_high = false;

        for &q in quality {
            if q < 59 {
                has_low = true;
            }
            if q > 74 {
                has_high = true;
            }
        }

        if has_low && !has_high {
            Some(Self::Phred33)
        } else if has_high && !has_low {
            Some(Self::Phred64)
        } else if has_low && has_high {
            // Contains both low and high - invalid data or mixed
            None
        } else {
            // All in overlapping range - cannot determine
            None
        }
    }
}

// ============================================================================
// Binning Constants and Tables
// ============================================================================

/// Quality bin levels (2 bits each, values 0-3).
pub mod bins {
    /// Bin 0: Very low quality (Q0-9, representative Q6)
    pub const BIN_VERY_LOW: u8 = 0;
    /// Bin 1: Low quality (Q10-19, representative Q15)
    pub const BIN_LOW: u8 = 1;
    /// Bin 2: Medium quality (Q20-29, representative Q25)
    pub const BIN_MEDIUM: u8 = 2;
    /// Bin 3: High quality (Q30+, representative Q37)
    pub const BIN_HIGH: u8 = 3;
}

/// Representative Phred values for each bin (used during decoding).
const BIN_REPRESENTATIVE_Q: [u8; 4] = [6, 15, 25, 37];

/// Pre-computed lookup table for Phred score → bin mapping.
/// Index by Phred value (0-41+), returns bin (0-3).
/// Values beyond Q41 are clamped to bin 3 (high quality).
#[repr(align(64))]
struct AlignedBinLUT([u8; 64]);

static BIN_LUT: AlignedBinLUT = {
    let mut lut = [bins::BIN_HIGH; 64]; // Default to high quality for out-of-range
    let mut i = 0;
    while i < 64 {
        lut[i] = if i < 10 {
            bins::BIN_VERY_LOW
        } else if i < 20 {
            bins::BIN_LOW
        } else if i < 30 {
            bins::BIN_MEDIUM
        } else {
            bins::BIN_HIGH
        };
        i += 1;
    }
    AlignedBinLUT(lut)
};

/// Fast Phred score to bin conversion using lookup table.
#[inline(always)]
fn phred_to_bin(phred: u8) -> u8 {
    // Clamp to LUT range and lookup
    BIN_LUT.0[phred.min(63) as usize]
}

/// Convert bin back to representative Phred value.
#[inline(always)]
fn bin_to_phred(bin: u8) -> u8 {
    BIN_REPRESENTATIVE_Q[(bin & 0x03) as usize]
}

// ============================================================================
// RLE Structures
// ============================================================================

/// A single run-length encoded segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinnedRun {
    /// Quality bin (0-3).
    pub bin: u8,
    /// Run length (1-65535).
    pub run_length: u16,
}

/// Maximum run length that fits in the short format (6 bits).
const MAX_SHORT_RUN: u16 = 62; // Must be < 63 to avoid collision with LONG_RUN_ESCAPE (0xFF)

/// Escape byte indicating a long run follows.
const LONG_RUN_ESCAPE: u8 = 0xFF;

// ============================================================================
// SIMD Configuration
// ============================================================================

/// Minimum input size to use SIMD processing.
/// Below this threshold, scalar is faster due to setup overhead.
const SIMD_THRESHOLD: usize = 32;

// ============================================================================
// Buffer Size Helpers
// ============================================================================

/// Returns the maximum number of bytes needed for RLE encoding.
///
/// In the worst case (alternating bins), each quality score becomes one RLE entry.
/// Short runs use 1 byte, so worst case is `length` bytes.
/// However, we add extra space for potential long run escapes.
#[inline]
pub const fn max_encoded_len(quality_len: usize) -> usize {
    // Worst case: every score is different, each needs 1 byte
    // Plus some overhead for the format
    quality_len + 4
}

/// Returns the exact number of bytes needed for the decoded quality string.
#[inline]
pub const fn required_decoded_len(original_len: usize) -> usize {
    original_len
}

// ============================================================================
// High-Level API
// ============================================================================

/// Encodes FASTQ quality scores into a compact representation.
///
/// The encoding process:
/// 1. Converts ASCII quality scores to Phred values
/// 2. Bins Phred values into 4 quality levels
/// 3. Applies run-length encoding to binned values
///
/// # Arguments
///
/// * `quality` - ASCII-encoded quality string from FASTQ file
/// * `encoding` - Phred encoding format (Phred+33 or Phred+64)
///
/// # Returns
///
/// Compressed quality data as a byte vector.
///
/// # Example
///
/// ```rust
/// use simdna::quality_encoder::{encode_quality_scores, PhredEncoding};
///
/// let quality = b"IIIIIIIIIIFFFFFFFF@@@@@@@@@55555";
/// let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
///
/// // Encoded is much smaller due to binning + RLE
/// assert!(encoded.len() < quality.len());
/// ```
pub fn encode_quality_scores(quality: &[u8], encoding: PhredEncoding) -> Vec<u8> {
    if quality.is_empty() {
        return Vec::new();
    }

    // Stage 1: Bin quality scores (SIMD-accelerated)
    let mut binned = vec![0u8; quality.len()];
    bin_quality_scores(quality, &mut binned, encoding);

    // Stage 2: Run-length encode the binned scores
    let mut output = Vec::with_capacity(max_encoded_len(quality.len()));
    encode_rle(&binned, &mut output);

    output
}

/// Encodes FASTQ quality scores into a caller-provided buffer (zero-allocation).
///
/// # Arguments
///
/// * `quality` - ASCII-encoded quality string from FASTQ file
/// * `encoding` - Phred encoding format
/// * `output` - Pre-allocated output buffer
///
/// # Returns
///
/// On success, returns `Ok(bytes_written)`.
///
/// # Errors
///
/// Returns [`QualityError::BufferTooSmall`] if the output buffer is insufficient.
///
/// # Example
///
/// ```rust
/// use simdna::quality_encoder::{encode_quality_scores_into, PhredEncoding};
///
/// let quality = b"IIIIIIIIII";
/// let mut buffer = [0u8; 64];
///
/// let bytes = encode_quality_scores_into(quality, PhredEncoding::Phred33, &mut buffer).unwrap();
/// assert!(bytes > 0);
/// ```
pub fn encode_quality_scores_into(
    quality: &[u8],
    encoding: PhredEncoding,
    output: &mut [u8],
) -> Result<usize, QualityError> {
    if quality.is_empty() {
        return Ok(0);
    }

    // We need a temporary buffer for binned scores
    // For zero-allocation in hot paths, consider a dedicated streaming API
    let mut binned = vec![0u8; quality.len()];
    bin_quality_scores(quality, &mut binned, encoding);

    encode_rle_into(&binned, output)
}

/// Decodes compressed quality scores back to ASCII format.
///
/// The decoded quality scores use representative values for each bin:
/// - Bin 0 → Q6 (ASCII 39 for Phred+33)
/// - Bin 1 → Q15 (ASCII 48 for Phred+33)
/// - Bin 2 → Q25 (ASCII 58 for Phred+33)
/// - Bin 3 → Q37 (ASCII 70 for Phred+33)
///
/// # Arguments
///
/// * `encoded` - Compressed quality data from [`encode_quality_scores`]
/// * `encoding` - Phred encoding format for output
///
/// # Returns
///
/// Decoded ASCII quality string.
///
/// # Example
///
/// ```rust
/// use simdna::quality_encoder::{encode_quality_scores, decode_quality_scores, PhredEncoding};
///
/// let quality = b"IIIIIIIIIIFFFFFFFF";
/// let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
/// let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred33);
///
/// // Decoded length matches original
/// assert_eq!(decoded.len(), quality.len());
/// ```
pub fn decode_quality_scores(encoded: &[u8], encoding: PhredEncoding) -> Vec<u8> {
    if encoded.is_empty() {
        return Vec::new();
    }

    // First pass: calculate total length
    let total_len = decode_rle_length(encoded);

    // Second pass: decode into buffer
    let mut output = vec![0u8; total_len];
    decode_rle(encoded, &mut output, encoding);

    output
}

/// Decodes compressed quality scores into a caller-provided buffer.
///
/// # Arguments
///
/// * `encoded` - Compressed quality data
/// * `encoding` - Phred encoding format for output
/// * `output` - Pre-allocated output buffer
///
/// # Returns
///
/// On success, returns `Ok(bytes_written)`.
///
/// # Errors
///
/// Returns [`QualityError::BufferTooSmall`] if the output buffer is insufficient.
pub fn decode_quality_scores_into(
    encoded: &[u8],
    encoding: PhredEncoding,
    output: &mut [u8],
) -> Result<usize, QualityError> {
    if encoded.is_empty() {
        return Ok(0);
    }

    let total_len = decode_rle_length(encoded);

    if output.len() < total_len {
        return Err(QualityError::BufferTooSmall {
            needed: total_len,
            actual: output.len(),
        });
    }

    decode_rle(encoded, &mut output[..total_len], encoding);
    Ok(total_len)
}

// ============================================================================
// Binning Implementation
// ============================================================================

/// Bins quality scores using SIMD acceleration when available.
///
/// This is the main binning entry point that dispatches to SIMD or scalar.
fn bin_quality_scores(quality: &[u8], binned: &mut [u8], encoding: PhredEncoding) {
    let used_simd = bin_with_simd_if_available(quality, binned, encoding);

    if !used_simd {
        bin_scalar(quality, binned, encoding);
    }
}

/// Attempts to use SIMD binning if available.
/// Falls back to scalar for small inputs where SIMD overhead dominates.
#[inline]
fn bin_with_simd_if_available(quality: &[u8], binned: &mut [u8], encoding: PhredEncoding) -> bool {
    // For small inputs, scalar is faster due to SIMD setup overhead
    if quality.len() < SIMD_THRESHOLD {
        return false;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe { bin_x86_sse2(quality, binned, encoding) };
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { bin_arm_neon(quality, binned, encoding) };
        return true;
    }

    #[allow(unreachable_code)]
    false
}

/// Scalar binning implementation.
///
/// Processes 4 scores at a time for better performance.
fn bin_scalar(quality: &[u8], binned: &mut [u8], encoding: PhredEncoding) {
    let offset = encoding.offset();

    // Process 4 at a time
    let chunks = quality.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        for j in 0..4 {
            let phred = quality[base + j].saturating_sub(offset);
            binned[base + j] = phred_to_bin(phred);
        }
    }

    // Handle remainder
    for i in (chunks * 4)..quality.len() {
        let phred = quality[i].saturating_sub(offset);
        binned[i] = phred_to_bin(phred);
    }
}

// ============================================================================
// x86 SSE2 Implementation
// ============================================================================

/// SSE2-accelerated quality score binning for x86_64.
///
/// Processes 32 quality scores at a time (2 vectors) for better ILP.
///
/// # Safety
///
/// Requires SSE2 support (available on all x86_64 CPUs).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn bin_x86_sse2(quality: &[u8], binned: &mut [u8], encoding: PhredEncoding) {
    let len = quality.len();
    let offset = encoding.offset();

    // SIMD constants
    let offset_vec = _mm_set1_epi8(offset as i8);
    let q10 = _mm_set1_epi8(10);
    let q20 = _mm_set1_epi8(20);
    let q30 = _mm_set1_epi8(30);
    let one = _mm_set1_epi8(1);

    // Process 32 bytes at a time for better ILP
    let simd_len_32 = (len / 32) * 32;
    let mut i = 0;

    while i < simd_len_32 {
        // Load 32 quality scores (2 vectors)
        let q0 = _mm_loadu_si128(quality.as_ptr().add(i) as *const __m128i);
        let q1 = _mm_loadu_si128(quality.as_ptr().add(i + 16) as *const __m128i);

        // Convert to Phred (subtract offset with saturation)
        let phred0 = _mm_subs_epu8(q0, offset_vec);
        let phred1 = _mm_subs_epu8(q1, offset_vec);

        // Create masks for each threshold
        // For unsigned comparison: a >= b iff max(a, b) == a
        let ge_10_0 = _mm_cmpeq_epi8(_mm_max_epu8(phred0, q10), phred0);
        let ge_10_1 = _mm_cmpeq_epi8(_mm_max_epu8(phred1, q10), phred1);
        let ge_20_0 = _mm_cmpeq_epi8(_mm_max_epu8(phred0, q20), phred0);
        let ge_20_1 = _mm_cmpeq_epi8(_mm_max_epu8(phred1, q20), phred1);
        let ge_30_0 = _mm_cmpeq_epi8(_mm_max_epu8(phred0, q30), phred0);
        let ge_30_1 = _mm_cmpeq_epi8(_mm_max_epu8(phred1, q30), phred1);

        // Convert masks (0xFF → 0x01) and sum
        let result0 = _mm_add_epi8(
            _mm_add_epi8(_mm_and_si128(ge_10_0, one), _mm_and_si128(ge_20_0, one)),
            _mm_and_si128(ge_30_0, one),
        );
        let result1 = _mm_add_epi8(
            _mm_add_epi8(_mm_and_si128(ge_10_1, one), _mm_and_si128(ge_20_1, one)),
            _mm_and_si128(ge_30_1, one),
        );

        // Store results
        _mm_storeu_si128(binned.as_mut_ptr().add(i) as *mut __m128i, result0);
        _mm_storeu_si128(binned.as_mut_ptr().add(i + 16) as *mut __m128i, result1);

        i += 32;
    }

    // Handle remaining bytes after 32-byte main loop.
    // Remaining bytes = len - i, where i is a multiple of 32.
    //
    // Cases:
    // - 0 bytes remain: done
    // - 1-15 bytes remain: overlapped 16-byte block from end (recomputes some)
    // - 16 bytes remain: single 16-byte block
    // - 17-31 bytes remain: 16-byte block + overlapped 16-byte block
    let remaining = len - i;

    if remaining == 0 {
        // Nothing to do
    } else if remaining <= 16 {
        // 1-16 bytes remain: single overlapped (or exact) 16-byte block from end
        // For remaining == 16, this is exact; for 1-15, it recomputes some bytes
        let final_offset = len - 16;
        let q = _mm_loadu_si128(quality.as_ptr().add(final_offset) as *const __m128i);
        let phred = _mm_subs_epu8(q, offset_vec);
        let ge_10 = _mm_cmpeq_epi8(_mm_max_epu8(phred, q10), phred);
        let ge_20 = _mm_cmpeq_epi8(_mm_max_epu8(phred, q20), phred);
        let ge_30 = _mm_cmpeq_epi8(_mm_max_epu8(phred, q30), phred);
        let result = _mm_add_epi8(
            _mm_add_epi8(_mm_and_si128(ge_10, one), _mm_and_si128(ge_20, one)),
            _mm_and_si128(ge_30, one),
        );
        _mm_storeu_si128(
            binned.as_mut_ptr().add(final_offset) as *mut __m128i,
            result,
        );
    } else {
        // 17-31 bytes remain: need 16-byte block + overlapped 16-byte block
        // First, process aligned 16-byte block at position i
        let q = _mm_loadu_si128(quality.as_ptr().add(i) as *const __m128i);
        let phred = _mm_subs_epu8(q, offset_vec);
        let ge_10 = _mm_cmpeq_epi8(_mm_max_epu8(phred, q10), phred);
        let ge_20 = _mm_cmpeq_epi8(_mm_max_epu8(phred, q20), phred);
        let ge_30 = _mm_cmpeq_epi8(_mm_max_epu8(phred, q30), phred);
        let result = _mm_add_epi8(
            _mm_add_epi8(_mm_and_si128(ge_10, one), _mm_and_si128(ge_20, one)),
            _mm_and_si128(ge_30, one),
        );
        _mm_storeu_si128(binned.as_mut_ptr().add(i) as *mut __m128i, result);

        // Then, overlapped 16-byte block from end for remaining 1-15 bytes
        let final_offset = len - 16;
        let q2 = _mm_loadu_si128(quality.as_ptr().add(final_offset) as *const __m128i);
        let phred2 = _mm_subs_epu8(q2, offset_vec);
        let ge_10_2 = _mm_cmpeq_epi8(_mm_max_epu8(phred2, q10), phred2);
        let ge_20_2 = _mm_cmpeq_epi8(_mm_max_epu8(phred2, q20), phred2);
        let ge_30_2 = _mm_cmpeq_epi8(_mm_max_epu8(phred2, q30), phred2);
        let result2 = _mm_add_epi8(
            _mm_add_epi8(_mm_and_si128(ge_10_2, one), _mm_and_si128(ge_20_2, one)),
            _mm_and_si128(ge_30_2, one),
        );
        _mm_storeu_si128(
            binned.as_mut_ptr().add(final_offset) as *mut __m128i,
            result2,
        );
    }
}

// ============================================================================
// ARM NEON Implementation
// ============================================================================

/// NEON-accelerated quality score binning for ARM64.
///
/// Uses an optimized arithmetic approach that computes bins with fewer operations:
/// bin = saturating((phred + 236) / 10) - 23, clamped to [0, 3]
///
/// But we use the threshold-count approach with better ILP by processing
/// 32 bytes per iteration using two vector registers.
///
/// # Safety
///
/// Requires NEON support (always available on ARM64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn bin_arm_neon(quality: &[u8], binned: &mut [u8], encoding: PhredEncoding) {
    let len = quality.len();
    let offset = encoding.offset();

    // SIMD constants
    let offset_vec = vdupq_n_u8(offset);
    let q10 = vdupq_n_u8(10);
    let q20 = vdupq_n_u8(20);
    let q30 = vdupq_n_u8(30);
    let one = vdupq_n_u8(1);

    // Process 32 bytes at a time for better ILP
    let simd_len_32 = (len / 32) * 32;
    let mut i = 0;

    while i < simd_len_32 {
        // Load 32 quality scores (2 vectors)
        let q0 = vld1q_u8(quality.as_ptr().add(i));
        let q1 = vld1q_u8(quality.as_ptr().add(i + 16));

        // Convert to Phred (subtract offset with saturation)
        let phred0 = vqsubq_u8(q0, offset_vec);
        let phred1 = vqsubq_u8(q1, offset_vec);

        // Create masks for each threshold using unsigned comparison
        // vcgeq_u8 returns 0xFF when >= threshold, 0x00 otherwise
        // Process both vectors in parallel for better ILP
        let ge_10_0 = vcgeq_u8(phred0, q10);
        let ge_10_1 = vcgeq_u8(phred1, q10);
        let ge_20_0 = vcgeq_u8(phred0, q20);
        let ge_20_1 = vcgeq_u8(phred1, q20);
        let ge_30_0 = vcgeq_u8(phred0, q30);
        let ge_30_1 = vcgeq_u8(phred1, q30);

        // Convert masks (0xFF → 0x01) and sum
        // bin = (phred >= 10) + (phred >= 20) + (phred >= 30)
        let result0 = vaddq_u8(
            vaddq_u8(vandq_u8(ge_10_0, one), vandq_u8(ge_20_0, one)),
            vandq_u8(ge_30_0, one),
        );
        let result1 = vaddq_u8(
            vaddq_u8(vandq_u8(ge_10_1, one), vandq_u8(ge_20_1, one)),
            vandq_u8(ge_30_1, one),
        );

        // Store results
        vst1q_u8(binned.as_mut_ptr().add(i), result0);
        vst1q_u8(binned.as_mut_ptr().add(i + 16), result1);

        i += 32;
    }

    // Handle remaining bytes after 32-byte main loop.
    // Remaining bytes = len - i, where i is a multiple of 32.
    //
    // Cases:
    // - 0 bytes remain: done
    // - 1-15 bytes remain: overlapped 16-byte block from end (recomputes some)
    // - 16 bytes remain: single 16-byte block
    // - 17-31 bytes remain: 16-byte block + overlapped 16-byte block
    let remaining = len - i;

    if remaining == 0 {
        // Nothing to do
    } else if remaining <= 16 {
        // 1-16 bytes remain: single overlapped (or exact) 16-byte block from end
        // For remaining == 16, this is exact; for 1-15, it recomputes some bytes
        let final_offset = len - 16;
        let q = vld1q_u8(quality.as_ptr().add(final_offset));
        let phred = vqsubq_u8(q, offset_vec);
        let ge_10 = vcgeq_u8(phred, q10);
        let ge_20 = vcgeq_u8(phred, q20);
        let ge_30 = vcgeq_u8(phred, q30);
        let result = vaddq_u8(
            vaddq_u8(vandq_u8(ge_10, one), vandq_u8(ge_20, one)),
            vandq_u8(ge_30, one),
        );
        vst1q_u8(binned.as_mut_ptr().add(final_offset), result);
    } else {
        // 17-31 bytes remain: need 16-byte block + overlapped 16-byte block
        // First, process aligned 16-byte block at position i
        let q = vld1q_u8(quality.as_ptr().add(i));
        let phred = vqsubq_u8(q, offset_vec);
        let ge_10 = vcgeq_u8(phred, q10);
        let ge_20 = vcgeq_u8(phred, q20);
        let ge_30 = vcgeq_u8(phred, q30);
        let result = vaddq_u8(
            vaddq_u8(vandq_u8(ge_10, one), vandq_u8(ge_20, one)),
            vandq_u8(ge_30, one),
        );
        vst1q_u8(binned.as_mut_ptr().add(i), result);

        // Then, overlapped 16-byte block from end for remaining 1-15 bytes
        let final_offset = len - 16;
        let q2 = vld1q_u8(quality.as_ptr().add(final_offset));
        let phred2 = vqsubq_u8(q2, offset_vec);
        let ge_10_2 = vcgeq_u8(phred2, q10);
        let ge_20_2 = vcgeq_u8(phred2, q20);
        let ge_30_2 = vcgeq_u8(phred2, q30);
        let result2 = vaddq_u8(
            vaddq_u8(vandq_u8(ge_10_2, one), vandq_u8(ge_20_2, one)),
            vandq_u8(ge_30_2, one),
        );
        vst1q_u8(binned.as_mut_ptr().add(final_offset), result2);
    }
}

// ============================================================================
// RLE Implementation
// ============================================================================

/// Counts how many bytes starting at `ptr` match `target`.
/// Uses SIMD when available for fast run length detection.
///
/// # Safety
/// Caller must ensure ptr..ptr+max_len is valid to read.
#[inline]
fn count_matching_bytes(data: &[u8], start: usize, target: u8) -> usize {
    let remaining = data.len() - start;
    if remaining == 0 {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Use SIMD for run counting on ARM
        unsafe { count_matching_neon(&data[start..], target) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe { count_matching_sse2(&data[start..], target) }
        } else {
            count_matching_scalar(&data[start..], target)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        count_matching_scalar(&data[start..], target)
    }
}

/// Scalar fallback for counting matching bytes.
#[inline]
fn count_matching_scalar(data: &[u8], target: u8) -> usize {
    let mut count = 0;
    for &b in data {
        if b != target {
            break;
        }
        count += 1;
    }
    count
}

/// NEON-accelerated matching byte counter.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn count_matching_neon(data: &[u8], target: u8) -> usize {
    let len = data.len();
    if len < 16 {
        return count_matching_scalar(data, target);
    }

    let target_vec = vdupq_n_u8(target);
    let mut count = 0;

    // Process 16 bytes at a time
    while count + 16 <= len {
        let chunk = vld1q_u8(data.as_ptr().add(count));
        let cmp = vceqq_u8(chunk, target_vec);

        // Get the comparison result as a bitmask
        // Each lane that matches has 0xFF, mismatches have 0x00
        // We need to find the first non-match

        // Use horizontal operations to check if all match
        let min_val = vminvq_u8(cmp);
        if min_val == 0xFF {
            // All 16 bytes match
            count += 16;
        } else {
            // Find first non-matching byte
            // Store comparison result and scan
            let mut cmp_bytes = [0u8; 16];
            vst1q_u8(cmp_bytes.as_mut_ptr(), cmp);
            for (i, &b) in cmp_bytes.iter().enumerate() {
                if b == 0 {
                    return count + i;
                }
            }
            return count + 16;
        }
    }

    // Handle remainder
    while count < len && data[count] == target {
        count += 1;
    }

    count
}

/// SSE2-accelerated matching byte counter.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn count_matching_sse2(data: &[u8], target: u8) -> usize {
    let len = data.len();
    if len < 16 {
        return count_matching_scalar(data, target);
    }

    let target_vec = _mm_set1_epi8(target as i8);
    let mut count = 0;

    // Process 16 bytes at a time
    while count + 16 <= len {
        let chunk = _mm_loadu_si128(data.as_ptr().add(count) as *const __m128i);
        let cmp = _mm_cmpeq_epi8(chunk, target_vec);
        let mask = _mm_movemask_epi8(cmp) as u32;

        if mask == 0xFFFF {
            // All 16 bytes match
            count += 16;
        } else {
            // Find first non-matching byte (first 0 bit)
            // Invert mask so matching bytes become 0, then find first 1
            let first_mismatch = (!mask).trailing_zeros() as usize;
            return count + first_mismatch;
        }
    }

    // Handle remainder
    while count < len && data[count] == target {
        count += 1;
    }

    count
}

/// Run-length encodes binned quality scores.
///
/// # Packing Format
/// - Short runs (≤63): `[bin:2][length:6]` = 1 byte
/// - Long runs (>63): `0xFF` + `[bin:8]` + `[length_hi:8]` + `[length_lo:8]` = 4 bytes
fn encode_rle(binned: &[u8], output: &mut Vec<u8>) {
    if binned.is_empty() {
        return;
    }

    let mut pos = 0;
    while pos < binned.len() {
        let bin = binned[pos];

        // Use SIMD-accelerated run counting
        let run_len = count_matching_bytes(binned, pos, bin).min(u16::MAX as usize) as u16;
        let run_len = run_len.max(1); // At least 1

        // Pack the run
        if run_len <= MAX_SHORT_RUN {
            // Short format: [bin:2][length:6]
            output.push((bin << 6) | (run_len as u8));
        } else {
            // Long format: escape + bin + 16-bit length
            output.push(LONG_RUN_ESCAPE);
            output.push(bin);
            output.push((run_len >> 8) as u8);
            output.push((run_len & 0xFF) as u8);
        }

        pos += run_len as usize;
    }
}

/// Run-length encodes into a caller-provided buffer.
fn encode_rle_into(binned: &[u8], output: &mut [u8]) -> Result<usize, QualityError> {
    if binned.is_empty() {
        return Ok(0);
    }

    let mut out_pos = 0;
    let mut pos = 0;

    while pos < binned.len() {
        let bin = binned[pos];

        // Use SIMD-accelerated run counting
        let run_len = count_matching_bytes(binned, pos, bin).min(u16::MAX as usize) as u16;
        let run_len = run_len.max(1); // At least 1

        // Pack the run
        if run_len <= MAX_SHORT_RUN {
            if out_pos >= output.len() {
                return Err(QualityError::BufferTooSmall {
                    needed: out_pos + 1,
                    actual: output.len(),
                });
            }
            output[out_pos] = (bin << 6) | (run_len as u8);
            out_pos += 1;
        } else {
            if out_pos + 4 > output.len() {
                return Err(QualityError::BufferTooSmall {
                    needed: out_pos + 4,
                    actual: output.len(),
                });
            }
            output[out_pos] = LONG_RUN_ESCAPE;
            output[out_pos + 1] = bin;
            output[out_pos + 2] = (run_len >> 8) as u8;
            output[out_pos + 3] = (run_len & 0xFF) as u8;
            out_pos += 4;
        }

        pos += run_len as usize;
    }

    Ok(out_pos)
}

/// Calculates the total decoded length from RLE data.
fn decode_rle_length(encoded: &[u8]) -> usize {
    let mut total = 0usize;
    let mut i = 0;

    while i < encoded.len() {
        if encoded[i] == LONG_RUN_ESCAPE {
            // Long format: escape + bin + 16-bit length
            if i + 3 < encoded.len() {
                let run_len = ((encoded[i + 2] as u16) << 8) | (encoded[i + 3] as u16);
                total += run_len as usize;
                i += 4;
            } else {
                break; // Invalid data
            }
        } else {
            // Short format: [bin:2][length:6]
            let run_len = (encoded[i] & 0x3F) as usize;
            total += run_len;
            i += 1;
        }
    }

    total
}

/// Decodes RLE data into quality scores.
fn decode_rle(encoded: &[u8], output: &mut [u8], encoding: PhredEncoding) {
    let offset = encoding.offset();
    let mut out_pos = 0;
    let mut i = 0;

    while i < encoded.len() && out_pos < output.len() {
        let (bin, run_len): (u8, u16);

        if encoded[i] == LONG_RUN_ESCAPE {
            // Long format
            if i + 3 >= encoded.len() {
                break;
            }
            bin = encoded[i + 1];
            run_len = ((encoded[i + 2] as u16) << 8) | (encoded[i + 3] as u16);
            i += 4;
        } else {
            // Short format
            bin = (encoded[i] >> 6) & 0x03;
            run_len = (encoded[i] & 0x3F) as u16;
            i += 1;
        }

        // Convert bin to ASCII quality
        let phred = bin_to_phred(bin);
        let ascii_q = phred + offset;

        // Fill output with the quality character
        let end = (out_pos + run_len as usize).min(output.len());
        output[out_pos..end].fill(ascii_q);
        out_pos = end;
    }
}

// ============================================================================
// Low-Level API
// ============================================================================

/// Bins quality scores in-place or into a provided buffer.
///
/// This is a lower-level function for users who want to perform binning
/// without RLE, or who want to inspect the binned values.
///
/// # Example
///
/// ```rust
/// use simdna::quality_encoder::{bin_quality, PhredEncoding};
///
/// let quality = b"IIIIII"; // All high quality (Phred ~40)
/// let mut binned = vec![0u8; quality.len()];
/// bin_quality(quality, &mut binned, PhredEncoding::Phred33);
///
/// // All should be bin 3 (high quality)
/// assert!(binned.iter().all(|&b| b == 3));
/// ```
pub fn bin_quality(quality: &[u8], binned: &mut [u8], encoding: PhredEncoding) {
    assert!(binned.len() >= quality.len());
    bin_quality_scores(quality, &mut binned[..quality.len()], encoding);
}

/// Extracts runs from RLE-encoded data without fully decoding.
///
/// Useful for analysis or when you need to process runs individually.
///
/// # Example
///
/// ```rust
/// use simdna::quality_encoder::{encode_quality_scores, extract_runs, PhredEncoding};
///
/// let quality = b"IIIIIIIIII"; // 10 high-quality scores
/// let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
/// let runs = extract_runs(&encoded);
///
/// // Should be a single run
/// assert_eq!(runs.len(), 1);
/// assert_eq!(runs[0].bin, 3); // High quality
/// assert_eq!(runs[0].run_length, 10);
/// ```
pub fn extract_runs(encoded: &[u8]) -> Vec<BinnedRun> {
    let mut runs = Vec::new();
    let mut i = 0;

    while i < encoded.len() {
        if encoded[i] == LONG_RUN_ESCAPE {
            if i + 3 >= encoded.len() {
                break;
            }
            let bin = encoded[i + 1];
            let run_length = ((encoded[i + 2] as u16) << 8) | (encoded[i + 3] as u16);
            runs.push(BinnedRun { bin, run_length });
            i += 4;
        } else {
            let bin = (encoded[i] >> 6) & 0x03;
            let run_length = (encoded[i] & 0x3F) as u16;
            runs.push(BinnedRun { bin, run_length });
            i += 1;
        }
    }

    runs
}

/// Returns statistics about a quality string.
///
/// # Returns
///
/// A tuple of (min_phred, max_phred, mean_phred, bin_counts)
/// where bin_counts is [count_bin0, count_bin1, count_bin2, count_bin3].
pub fn quality_stats(quality: &[u8], encoding: PhredEncoding) -> (u8, u8, f64, [usize; 4]) {
    if quality.is_empty() {
        return (0, 0, 0.0, [0; 4]);
    }

    let offset = encoding.offset();
    let mut min_q = u8::MAX;
    let mut max_q = 0u8;
    let mut sum = 0u64;
    let mut bin_counts = [0usize; 4];

    for &q in quality {
        let phred = q.saturating_sub(offset);
        min_q = min_q.min(phred);
        max_q = max_q.max(phred);
        sum += phred as u64;
        let bin = phred_to_bin(phred) as usize;
        bin_counts[bin] += 1;
    }

    let mean = sum as f64 / quality.len() as f64;
    (min_q, max_q, mean, bin_counts)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Phred Encoding Tests
    // ========================================================================

    #[test]
    fn test_phred_encoding_offset() {
        assert_eq!(PhredEncoding::Phred33.offset(), 33);
        assert_eq!(PhredEncoding::Phred64.offset(), 64);
    }

    #[test]
    fn test_phred_encoding_detection() {
        // Clear Phred+33 (contains low ASCII like '!')
        let phred33 = b"!\"#$%&'()*+,-./0123456789";
        assert_eq!(PhredEncoding::detect(phred33), Some(PhredEncoding::Phred33));

        // Clear Phred+64 (contains high ASCII like 'h')
        let phred64 = b"efghijklmnopqrstuvwxyz";
        assert_eq!(PhredEncoding::detect(phred64), Some(PhredEncoding::Phred64));

        // Ambiguous (overlapping range)
        let ambiguous = b"ABCDEFGHIJ";
        assert_eq!(PhredEncoding::detect(ambiguous), None);
    }

    #[test]
    fn test_phred_encoding_default() {
        assert_eq!(PhredEncoding::default(), PhredEncoding::Phred33);
    }

    // ========================================================================
    // Binning Tests
    // ========================================================================

    #[test]
    fn test_phred_to_bin() {
        // Bin 0: Q0-9
        for q in 0..10 {
            assert_eq!(phred_to_bin(q), 0, "Q{} should be bin 0", q);
        }
        // Bin 1: Q10-19
        for q in 10..20 {
            assert_eq!(phred_to_bin(q), 1, "Q{} should be bin 1", q);
        }
        // Bin 2: Q20-29
        for q in 20..30 {
            assert_eq!(phred_to_bin(q), 2, "Q{} should be bin 2", q);
        }
        // Bin 3: Q30+
        for q in 30..50 {
            assert_eq!(phred_to_bin(q), 3, "Q{} should be bin 3", q);
        }
    }

    #[test]
    fn test_bin_to_phred() {
        assert_eq!(bin_to_phred(0), 6);
        assert_eq!(bin_to_phred(1), 15);
        assert_eq!(bin_to_phred(2), 25);
        assert_eq!(bin_to_phred(3), 37);
    }

    #[test]
    fn test_bin_quality_scalar() {
        // Q40 in Phred+33 is ASCII 'I' (73)
        let quality = b"IIIIII";
        let mut binned = vec![0u8; quality.len()];
        bin_scalar(quality, &mut binned, PhredEncoding::Phred33);
        assert!(binned.iter().all(|&b| b == 3)); // All high quality

        // Q10 in Phred+33 is ASCII '+' (43)
        let quality_low = b"++++++";
        let mut binned_low = vec![0u8; quality_low.len()];
        bin_scalar(quality_low, &mut binned_low, PhredEncoding::Phred33);
        assert!(binned_low.iter().all(|&b| b == 1)); // All low quality
    }

    #[test]
    fn test_bin_quality_simd_matches_scalar() {
        // Create a quality string with varied scores
        let quality: Vec<u8> = (33..=73).cycle().take(100).collect();
        let mut binned_simd = vec![0u8; quality.len()];
        let mut binned_scalar = vec![0u8; quality.len()];

        bin_quality_scores(&quality, &mut binned_simd, PhredEncoding::Phred33);
        bin_scalar(&quality, &mut binned_scalar, PhredEncoding::Phred33);

        assert_eq!(binned_simd, binned_scalar);
    }

    #[test]
    fn test_bin_quality_various_lengths() {
        for len in 0..=50 {
            let quality: Vec<u8> = vec![b'I'; len]; // All high quality
            let mut binned = vec![0u8; len];
            bin_quality_scores(&quality, &mut binned, PhredEncoding::Phred33);
            assert!(binned.iter().all(|&b| b == 3 || len == 0));
        }
    }

    // ========================================================================
    // RLE Tests
    // ========================================================================

    #[test]
    fn test_rle_single_run() {
        let binned = vec![3u8; 10]; // 10 high-quality bins
        let mut output = Vec::new();
        encode_rle(&binned, &mut output);

        // Should be single byte: [bin:2][length:6] = [11][001010] = 0xCA
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], (3 << 6) | 10);

        // Verify length calculation
        assert_eq!(decode_rle_length(&output), 10);
    }

    #[test]
    fn test_rle_multiple_runs() {
        // Alternating bins: 3, 3, 3, 2, 2, 1, 1, 1, 1
        let binned = vec![3, 3, 3, 2, 2, 1, 1, 1, 1];
        let mut output = Vec::new();
        encode_rle(&binned, &mut output);

        // Should be 3 bytes: [3,3], [2,2], [1,4]
        assert_eq!(output.len(), 3);

        let runs = extract_runs(&output);
        assert_eq!(runs.len(), 3);
        assert_eq!(
            runs[0],
            BinnedRun {
                bin: 3,
                run_length: 3
            }
        );
        assert_eq!(
            runs[1],
            BinnedRun {
                bin: 2,
                run_length: 2
            }
        );
        assert_eq!(
            runs[2],
            BinnedRun {
                bin: 1,
                run_length: 4
            }
        );
    }

    #[test]
    fn test_rle_long_run() {
        let binned = vec![2u8; 100]; // 100 medium-quality bins (> 63)
        let mut output = Vec::new();
        encode_rle(&binned, &mut output);

        // Should use long format: 0xFF + bin + 2-byte length
        assert_eq!(output.len(), 4);
        assert_eq!(output[0], LONG_RUN_ESCAPE);
        assert_eq!(output[1], 2);
        assert_eq!(((output[2] as u16) << 8) | (output[3] as u16), 100);

        assert_eq!(decode_rle_length(&output), 100);
    }

    #[test]
    fn test_rle_max_short_run() {
        let binned = vec![1u8; 62]; // Exactly max short run (62 to avoid 0xFF collision)
        let mut output = Vec::new();
        encode_rle(&binned, &mut output);

        assert_eq!(output.len(), 1);
        assert_eq!(output[0], (1 << 6) | 62);
    }

    #[test]
    fn test_rle_just_over_short_run() {
        let binned = vec![1u8; 63]; // Just over max short run
        let mut output = Vec::new();
        encode_rle(&binned, &mut output);

        // Should use long format
        assert_eq!(output[0], LONG_RUN_ESCAPE);
        assert_eq!(decode_rle_length(&output), 63);
    }

    #[test]
    fn test_rle_bin3_length63_no_collision() {
        // This test verifies that bin=3 with length=63 doesn't create a 0xFF collision
        // Previously, (3 << 6) | 63 = 255 = 0xFF = LONG_RUN_ESCAPE which caused decoding errors
        let binned = vec![3u8; 63];
        let mut output = Vec::new();
        encode_rle(&binned, &mut output);

        // With MAX_SHORT_RUN=62, this should use long format (not produce 0xFF short format)
        assert_eq!(output[0], LONG_RUN_ESCAPE);
        assert_eq!(decode_rle_length(&output), 63);
    }

    // ========================================================================
    // Roundtrip Tests
    // ========================================================================

    #[test]
    fn test_roundtrip_basic() {
        let quality = b"IIIIIIIIII"; // 10 high-quality scores
        let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
        let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred33);

        // Length should match
        assert_eq!(decoded.len(), quality.len());

        // All decoded should be representative of bin 3
        let expected_q = BIN_REPRESENTATIVE_Q[3] + 33; // Q37 + 33 = 70 = 'F'
        assert!(decoded.iter().all(|&q| q == expected_q));
    }

    #[test]
    fn test_roundtrip_mixed_quality() {
        // Mix of quality levels with distinct bins
        // '!' = Q0 → bin 0, '+' = Q10 → bin 1, '5' = Q20 → bin 2, 'I' = Q40 → bin 3
        let quality = b"!!!!!+++++55555IIIII";
        let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
        let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred33);

        assert_eq!(decoded.len(), quality.len());

        // Check bins are preserved (not exact values)
        let runs = extract_runs(&encoded);
        assert_eq!(runs.len(), 4);
        assert_eq!(runs[0].bin, 0); // Q0-9
        assert_eq!(runs[1].bin, 1); // Q10-19
        assert_eq!(runs[2].bin, 2); // Q20-29
        assert_eq!(runs[3].bin, 3); // Q30+
    }

    #[test]
    fn test_roundtrip_empty() {
        let quality = b"";
        let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
        let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred33);

        assert!(encoded.is_empty());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_roundtrip_single_score() {
        let quality = b"I";
        let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
        let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred33);

        assert_eq!(decoded.len(), 1);
    }

    #[test]
    fn test_roundtrip_long_sequence() {
        // 1000 quality scores
        let quality: Vec<u8> = (0..1000).map(|i| b'!' + (i % 41) as u8).collect();
        let encoded = encode_quality_scores(&quality, PhredEncoding::Phred33);
        let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred33);

        assert_eq!(decoded.len(), quality.len());
    }

    #[test]
    fn test_roundtrip_phred64() {
        // Phred+64: '@' = Q0, 'h' = Q40
        let quality = b"@@@@@@@@@@hhhhhhhhhh";
        let encoded = encode_quality_scores(quality, PhredEncoding::Phred64);
        let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred64);

        assert_eq!(decoded.len(), quality.len());
    }

    // ========================================================================
    // Zero-Allocation API Tests
    // ========================================================================

    #[test]
    fn test_encode_into_success() {
        let quality = b"IIIIIIIIII";
        let mut buffer = [0u8; 64];

        let bytes =
            encode_quality_scores_into(quality, PhredEncoding::Phred33, &mut buffer).unwrap();

        assert!(bytes > 0);
        assert!(bytes < quality.len()); // Should compress
    }

    #[test]
    fn test_decode_into_success() {
        let quality = b"IIIIIIIIII";
        let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
        let mut buffer = [0u8; 64];

        let bytes =
            decode_quality_scores_into(&encoded, PhredEncoding::Phred33, &mut buffer).unwrap();

        assert_eq!(bytes, quality.len());
    }

    #[test]
    fn test_decode_into_buffer_too_small() {
        let quality = b"IIIIIIIIII";
        let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
        let mut buffer = [0u8; 2]; // Too small

        let result = decode_quality_scores_into(&encoded, PhredEncoding::Phred33, &mut buffer);

        assert!(matches!(result, Err(QualityError::BufferTooSmall { .. })));
    }

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_quality_stats() {
        // Q10, Q20, Q30, Q40 in Phred+33
        let quality = b"+5?I";
        let (min_q, max_q, mean, bin_counts) = quality_stats(quality, PhredEncoding::Phred33);

        assert_eq!(min_q, 10);
        assert_eq!(max_q, 40);
        assert!((mean - 25.0).abs() < 0.1);
        assert_eq!(bin_counts[1], 1); // Q10 in bin 1
        assert_eq!(bin_counts[2], 1); // Q20 in bin 2
        assert_eq!(bin_counts[3], 2); // Q30, Q40 in bin 3
    }

    #[test]
    fn test_quality_stats_empty() {
        let (min_q, max_q, mean, bin_counts) = quality_stats(b"", PhredEncoding::Phred33);
        assert_eq!(min_q, 0);
        assert_eq!(max_q, 0);
        assert_eq!(mean, 0.0);
        assert_eq!(bin_counts, [0, 0, 0, 0]);
    }

    // ========================================================================
    // Compression Ratio Tests
    // ========================================================================

    #[test]
    fn test_compression_uniform_quality() {
        // Uniform high quality should compress very well
        let quality: Vec<u8> = vec![b'I'; 150]; // Typical Illumina read length
        let encoded = encode_quality_scores(&quality, PhredEncoding::Phred33);

        // Should compress to just a few bytes (one RLE entry)
        assert!(encoded.len() <= 4);
        let ratio = 1.0 - (encoded.len() as f64 / quality.len() as f64);
        assert!(
            ratio > 0.95,
            "Expected >95% compression, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_compression_realistic_illumina() {
        // Simulate realistic Illumina quality: starts high, degrades toward end
        let mut quality = Vec::with_capacity(150);
        for i in 0..150 {
            let q = if i < 30 {
                b'I' // Q40
            } else if i < 100 {
                b'F' // Q37
            } else if i < 130 {
                b'?' // Q30
            } else {
                b'5' // Q20
            };
            quality.push(q);
        }

        let encoded = encode_quality_scores(&quality, PhredEncoding::Phred33);

        // Should compress significantly due to long runs
        let ratio = 1.0 - (encoded.len() as f64 / quality.len() as f64);
        assert!(
            ratio > 0.9,
            "Expected >90% compression, got {:.1}%",
            ratio * 100.0
        );
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_all_bins_present() {
        // Create quality string with all 4 bins
        // Bin 0: Q0-9, Bin 1: Q10-19, Bin 2: Q20-29, Bin 3: Q30+
        let quality = b"!!!!!+++++55555IIIII"; // Q0, Q10, Q20, Q40
        let mut binned = vec![0u8; quality.len()];
        bin_quality_scores(quality, &mut binned, PhredEncoding::Phred33);

        // Verify all bins present
        assert!(binned[..5].iter().all(|&b| b == 0));
        assert!(binned[5..10].iter().all(|&b| b == 1));
        assert!(binned[10..15].iter().all(|&b| b == 2));
        assert!(binned[15..20].iter().all(|&b| b == 3));
    }

    #[test]
    fn test_boundary_scores() {
        let encoding = PhredEncoding::Phred33;
        let offset = encoding.offset();

        // Test exact boundaries
        let boundaries = [(9, 0), (10, 1), (19, 1), (20, 2), (29, 2), (30, 3)];

        for (phred, expected_bin) in boundaries {
            let quality = [phred + offset];
            let mut binned = [255u8];
            bin_quality_scores(&quality, &mut binned, encoding);
            assert_eq!(
                binned[0], expected_bin,
                "Phred {} should be bin {}",
                phred, expected_bin
            );
        }
    }

    #[test]
    fn test_alternating_bins() {
        // Worst case for RLE: alternating bins
        let quality: Vec<u8> = (0..100)
            .map(|i| if i % 2 == 0 { b'I' } else { b'!' })
            .collect();

        let encoded = encode_quality_scores(&quality, PhredEncoding::Phred33);
        let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred33);

        assert_eq!(decoded.len(), quality.len());

        // RLE should still work, just not compress well
        let runs = extract_runs(&encoded);
        assert_eq!(runs.len(), 100); // Each score is its own run
    }
}
