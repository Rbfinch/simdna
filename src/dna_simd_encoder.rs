// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! # DNA 4-bit Encoding Library
//!
//! High-performance DNA sequence encoding and decoding using SIMD instructions
//! with automatic fallback to scalar implementations.
//!
//! ## Encoding Scheme
//!
//! DNA/RNA nucleotides and IUPAC ambiguity codes are encoded using 4 bits each.
//! The encoding is designed to support efficient complement calculation via
//! 2-bit rotation: rotating any 4-bit encoding 2 positions yields its complement.
//!
//! ### Standard Nucleotides
//!
//! | Code | Meaning              | Binary | Complement |
//! |------|----------------------|--------|------------|
//! | A    | Adenine              | 0x1    | T (0x4)    |
//! | C    | Cytosine             | 0x2    | G (0x8)    |
//! | T    | Thymine              | 0x4    | A (0x1)    |
//! | G    | Guanine              | 0x8    | C (0x2)    |
//! | U    | Uracil (RNA → T)     | 0x4    | A (0x1)    |
//!
//! ### Two-Base Ambiguity Codes
//!
//! | Code | Meaning              | Binary | Complement |
//! |------|----------------------|--------|------------|
//! | R    | A or G (purine)      | 0x9    | Y (0x6)    |
//! | Y    | C or T (pyrimidine)  | 0x6    | R (0x9)    |
//! | S    | G or C (strong)      | 0xA    | S (0xA)    |
//! | W    | A or T (weak)        | 0x5    | W (0x5)    |
//! | K    | G or T (keto)        | 0xC    | M (0x3)    |
//! | M    | A or C (amino)       | 0x3    | K (0xC)    |
//!
//! ### Three-Base Ambiguity Codes
//!
//! | Code | Meaning              | Binary | Complement |
//! |------|----------------------|--------|------------|
//! | B    | C, G, or T (not A)   | 0xE    | V (0xB)    |
//! | D    | A, G, or T (not C)   | 0xD    | H (0x7)    |
//! | H    | A, C, or T (not G)   | 0x7    | D (0xD)    |
//! | V    | A, C, or G (not T)   | 0xB    | B (0xE)    |
//!
//! ### Wildcards and Gaps
//!
//! | Code | Meaning              | Binary | Complement |
//! |------|----------------------|--------|------------|
//! | N    | Any base             | 0xF    | N (0xF)    |
//! | -    | Gap / deletion       | 0x0    | - (0x0)    |
//! | .    | Gap (alternative)    | 0x0    | - (0x0)    |
//!
//! ### Bit Rotation Property
//!
//! The complement of any nucleotide can be computed efficiently using:
//! ```text
//! complement = ((bits << 2) | (bits >> 2)) & 0xF
//! ```
//! This enables SIMD-accelerated reverse complement operations.
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
//! use simdna::dna_simd_encoder::{encode_dna_prefer_simd, decode_dna_prefer_simd};
//!
//! // Encode a DNA sequence with IUPAC codes
//! let sequence = b"ACGTNRYSWKMBDHV-";
//! let encoded = encode_dna_prefer_simd(sequence);
//!
//! // The encoded data is 2x smaller (2 nucleotides per byte)
//! assert_eq!(encoded.len(), sequence.len() / 2);
//!
//! // Decode back to the original sequence
//! let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
//! assert_eq!(decoded, sequence);
//! ```
//!
//! ## Case Handling
//!
//! Input sequences are handled case-insensitively via a lookup table.
//! Both `"ACGT"` and `"acgt"` produce identical encoded output, and decoding
//! always produces uppercase nucleotides. No intermediate uppercase conversion
//! is performed, making the encoding zero-allocation for the common case.
//!
//! ## Invalid Characters
//!
//! Characters not in the IUPAC nucleotide alphabet (including X, digits,
//! whitespace, and other symbols) are encoded as gap (`0x0`) and
//! decode back to `-`.
//!
//! ## RNA Support
//!
//! RNA sequences using U (Uracil) are fully supported. Uracil is encoded
//! identically to Thymine (0x4) and decodes back to T, enabling seamless
//! processing of both DNA and RNA sequences.
//!
//! ## Performance Considerations
//!
//! The library employs several optimization strategies:
//!
//! - **Static Lookup Tables**: Pre-computed 256-byte encode and 16-byte decode
//!   tables eliminate branch mispredictions (~15-20% faster than match statements)
//! - **SIMD Processing**: Handles 32 nucleotides per iteration (two 16-byte chunks)
//!   for improved instruction-level parallelism
//! - **Prefetch Hints**: Prefetches data 128 bytes (2 cache lines) ahead to reduce
//!   cache misses on large sequences
//! - **Direct Case Handling**: LUT handles case-insensitivity directly, avoiding
//!   an extra O(n) uppercase conversion pass
//! - **4-at-a-time Scalar**: Remainder processing uses an optimized scalar path
//!   that processes 4 nucleotides per iteration
//! - **Aligned Memory**: LUTs use 64-byte alignment for optimal cache access
//!
//! Sequences shorter than 32 nucleotides use the optimized scalar path entirely.
//! Input is padded to a multiple of 2 for byte-aligned packing.

// Allow unsafe operations in unsafe functions (Rust 2024 compatibility)
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// 4-bit encoding values for IUPAC nucleotide codes.
///
/// These constants define the encoding scheme used throughout the library.
/// The encoding is designed so that the complement of any nucleotide can be
/// computed via 2-bit rotation: `complement = ((bits << 2) | (bits >> 2)) & 0xF`
///
/// # Complement Relationships
///
/// - A (0x1) ↔ T (0x4)
/// - C (0x2) ↔ G (0x8)
/// - R (0x9) ↔ Y (0x6)
/// - K (0xC) ↔ M (0x3)
/// - D (0xD) ↔ H (0x7)
/// - B (0xE) ↔ V (0xB)
/// - S (0xA), W (0x5), N (0xF), Gap (0x0) are self-complementary
#[allow(dead_code)]
pub mod encoding {
    /// Gap / Unknown / Deletion (self-complementary)
    pub const GAP: u8 = 0x0;
    /// Adenine (complements to T)
    pub const A: u8 = 0x1;
    /// Cytosine (complements to G)
    pub const C: u8 = 0x2;
    /// A or C (amino) - complements to K
    pub const M: u8 = 0x3;
    /// Thymine (complements to A)
    pub const T: u8 = 0x4;
    /// A or T (weak) - self-complementary
    pub const W: u8 = 0x5;
    /// C or T (pyrimidine) - complements to R
    pub const Y: u8 = 0x6;
    /// A, C, or T (not G) - complements to D
    pub const H: u8 = 0x7;
    /// Guanine (complements to C)
    pub const G: u8 = 0x8;
    /// A or G (purine) - complements to Y
    pub const R: u8 = 0x9;
    /// G or C (strong) - self-complementary
    pub const S: u8 = 0xA;
    /// A, C, or G (not T) - complements to B
    pub const V: u8 = 0xB;
    /// G or T (keto) - complements to M
    pub const K: u8 = 0xC;
    /// A, G, or T (not C) - complements to H
    pub const D: u8 = 0xD;
    /// C, G, or T (not A) - complements to V
    pub const B: u8 = 0xE;
    /// Any base (self-complementary)
    pub const N: u8 = 0xF;
}

// ============================================================================
// Optimized Static Lookup Tables
// ============================================================================

/// Pre-computed 256-byte lookup table for ASCII → 4-bit encoding.
/// This is faster than a match statement because it avoids branch mispredictions.
/// Index by ASCII value, returns 4-bit encoding (0x0 for invalid/unknown characters).
///
/// This table is aligned to 64 bytes for optimal cache line access.
#[repr(align(64))]
struct AlignedEncodeLUT([u8; 256]);

static ENCODE_LUT: AlignedEncodeLUT = {
    let mut table = [0u8; 256];
    // Standard nucleotides (bit-rotation encoding)
    table[b'A' as usize] = 0x1;
    table[b'a' as usize] = 0x1;
    table[b'C' as usize] = 0x2;
    table[b'c' as usize] = 0x2;
    table[b'T' as usize] = 0x4;
    table[b't' as usize] = 0x4;
    table[b'U' as usize] = 0x4;
    table[b'u' as usize] = 0x4; // RNA → T
    table[b'G' as usize] = 0x8;
    table[b'g' as usize] = 0x8;
    // Two-base ambiguity codes
    table[b'M' as usize] = 0x3;
    table[b'm' as usize] = 0x3; // A or C
    table[b'W' as usize] = 0x5;
    table[b'w' as usize] = 0x5; // A or T
    table[b'Y' as usize] = 0x6;
    table[b'y' as usize] = 0x6; // C or T
    table[b'R' as usize] = 0x9;
    table[b'r' as usize] = 0x9; // A or G
    table[b'S' as usize] = 0xA;
    table[b's' as usize] = 0xA; // G or C
    table[b'K' as usize] = 0xC;
    table[b'k' as usize] = 0xC; // G or T
    // Three-base ambiguity codes
    table[b'H' as usize] = 0x7;
    table[b'h' as usize] = 0x7; // A, C, or T
    table[b'V' as usize] = 0xB;
    table[b'v' as usize] = 0xB; // A, C, or G
    table[b'D' as usize] = 0xD;
    table[b'd' as usize] = 0xD; // A, G, or T
    table[b'B' as usize] = 0xE;
    table[b'b' as usize] = 0xE; // C, G, or T
    // Any base / wildcards
    table[b'N' as usize] = 0xF;
    table[b'n' as usize] = 0xF;
    // Gaps (explicitly 0, but set for clarity)
    table[b'-' as usize] = 0x0;
    table[b'.' as usize] = 0x0;
    AlignedEncodeLUT(table)
};

/// Pre-computed 16-byte lookup table for 4-bit → ASCII decoding.
/// Aligned for optimal SIMD access.
#[repr(align(16))]
struct AlignedDecodeLUT([u8; 16]);

static DECODE_LUT: AlignedDecodeLUT = AlignedDecodeLUT([
    b'-', b'A', b'C', b'M', b'T', b'W', b'Y', b'H', b'G', b'R', b'S', b'V', b'K', b'D', b'B', b'N',
]);

/// Fast 4-bit encoding using static lookup table.
/// This is ~15-20% faster than the match-based implementation.
#[inline(always)]
fn char_to_4bit_fast(c: u8) -> u8 {
    // SAFETY: u8 always fits in 0..256, which is the table size
    ENCODE_LUT.0[c as usize]
}

/// Fast 4-bit decoding using static lookup table.
#[inline(always)]
fn fourbit_to_char_fast(bits: u8) -> u8 {
    DECODE_LUT.0[(bits & 0xF) as usize]
}

// ============================================================================
// Prefetch Hints for Large Sequences
// ============================================================================

/// Prefetch data into L1 cache for upcoming reads.
/// This hint tells the CPU to load data before we need it.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_read(ptr: *const u8) {
    #[cfg(target_feature = "sse")]
    {
        use std::arch::x86_64::_mm_prefetch;
        _mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_read(ptr: *const u8) {
    // ARM prefetch using inline assembly
    std::arch::asm!(
        "prfm pldl1keep, [{ptr}]",
        ptr = in(reg) ptr,
        options(nostack, preserves_flags)
    );
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
fn prefetch_read(_ptr: *const u8) {
    // No-op on unsupported platforms
}

/// Prefetch distance in bytes (2 cache lines ahead)
const PREFETCH_DISTANCE: usize = 128;

/// Encodes a DNA sequence into a compact 4-bit representation.
///
/// Each nucleotide or IUPAC ambiguity code is encoded using 4 bits,
/// with 2 nucleotides packed per byte. The encoding supports efficient
/// complement calculation via 2-bit rotation.
///
/// # Encoding Table
///
/// | Input | Encoding | Input | Encoding |
/// |-------|----------|-------|----------|
/// | -, .  | 0x0      | G     | 0x8      |
/// | A     | 0x1      | R     | 0x9      |
/// | C     | 0x2      | S     | 0xA      |
/// | M     | 0x3      | V     | 0xB      |
/// | T, U  | 0x4      | K     | 0xC      |
/// | W     | 0x5      | D     | 0xD      |
/// | Y     | 0x6      | B     | 0xE      |
/// | H     | 0x7      | N     | 0xF      |
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
/// - **Case insensitivity**: Handled directly via lookup table (no conversion overhead).
/// - **Padding**: Sequences with odd length are padded with gap values.
/// - **Invalid characters**: Non-IUPAC characters are encoded as gap (`0x0`).
///
/// # Performance
///
/// This function automatically uses SIMD instructions when available:
/// - x86_64 with SSSE3: Processes 32 nucleotides per iteration with prefetching
/// - ARM64 (aarch64): Uses NEON to process 32 nucleotides per iteration with prefetching
/// - Other platforms: Uses optimized 4-at-a-time scalar implementation
///
/// Additional optimizations include static lookup tables, cache-aligned memory,
/// and prefetch hints for large sequences (>32 nucleotides).
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
/// use simdna::dna_simd_encoder::encode_dna_prefer_simd;
///
/// let encoded = encode_dna_prefer_simd(b"ACGT");
/// // A=0x1, C=0x2, G=0x8, T=0x4
/// // Packed: (A<<4|C), (G<<4|T) = 0x12, 0x84
/// assert_eq!(encoded, vec![0x12, 0x84]);
/// ```
///
/// IUPAC ambiguity codes are preserved:
/// ```rust
/// use simdna::dna_simd_encoder::{encode_dna_prefer_simd, decode_dna_prefer_simd};
///
/// let sequence = b"ACNGT";
/// let encoded = encode_dna_prefer_simd(sequence);
/// let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
/// assert_eq!(decoded, b"ACNGT");
/// ```
///
/// Compression ratio:
/// ```rust
/// use simdna::dna_simd_encoder::encode_dna_prefer_simd;
///
/// let sequence = b"ACGTACGTACGTACGT"; // 16 bytes
/// let encoded = encode_dna_prefer_simd(sequence);
/// assert_eq!(encoded.len(), 8); // 8 bytes (2:1 compression)
/// ```
pub fn encode_dna_prefer_simd(sequence: &[u8]) -> Vec<u8> {
    // The LUT handles case-insensitivity directly, so no uppercase conversion needed.
    // This saves one allocation and O(n) pass over the data.
    let len = sequence.len();

    // Pad to multiple of 2 for packing
    let padded_len = (len + 1) & !1;
    let mut output = vec![0u8; padded_len / 2];

    // Handle the input directly, padding handled by scalar/SIMD fallback
    // Try SIMD implementations first, fall back to scalar
    if len >= 32 {
        // For large sequences, use SIMD path with padding handled internally
        let used_simd = encode_with_simd_if_available_direct(sequence, &mut output);
        if used_simd {
            return output;
        }
    }

    // Fallback: use optimized scalar implementation
    encode_scalar_optimized(sequence, &mut output);

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
#[allow(dead_code)]
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

/// Direct SIMD encoding without pre-padding.
/// Handles padding internally for odd-length sequences.
#[inline]
fn encode_with_simd_if_available_direct(sequence: &[u8], output: &mut [u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            unsafe { encode_x86_ssse3_direct(sequence, output) };
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { encode_arm_neon_direct(sequence, output) };
        return true;
    }

    #[allow(unreachable_code)]
    false
}

/// Optimized scalar encoding that processes 4 nucleotides at a time.
/// Uses the static LUT for fast case-insensitive lookups.
#[inline]
fn encode_scalar_optimized(sequence: &[u8], output: &mut [u8]) {
    let len = sequence.len();
    let mut i = 0;
    let mut out_idx = 0;

    // Process 4 nucleotides (2 output bytes) at a time
    while i + 4 <= len {
        // Use fast LUT lookups - handles case insensitivity
        let b0 = char_to_4bit_fast(sequence[i]);
        let b1 = char_to_4bit_fast(sequence[i + 1]);
        let b2 = char_to_4bit_fast(sequence[i + 2]);
        let b3 = char_to_4bit_fast(sequence[i + 3]);

        output[out_idx] = (b0 << 4) | b1;
        output[out_idx + 1] = (b2 << 4) | b3;

        i += 4;
        out_idx += 2;
    }

    // Handle remaining 2-3 nucleotides
    if i + 2 <= len {
        let high = char_to_4bit_fast(sequence[i]);
        let low = char_to_4bit_fast(sequence[i + 1]);
        output[out_idx] = (high << 4) | low;
        i += 2;
        out_idx += 1;
    }

    // Handle final odd nucleotide
    if i < len {
        let high = char_to_4bit_fast(sequence[i]);
        output[out_idx] = high << 4; // Low nibble padded with gap (0x0)
    }
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
/// | 0x0      | -      | 0x8      | G      |
/// | 0x1      | A      | 0x9      | R      |
/// | 0x2      | C      | 0xA      | S      |
/// | 0x3      | M      | 0xB      | V      |
/// | 0x4      | T      | 0xC      | K      |
/// | 0x5      | W      | 0xD      | D      |
/// | 0x6      | Y      | 0xE      | B      |
/// | 0x7      | H      | 0xF      | N      |
///
/// # Arguments
///
/// * `encoded` - A byte slice containing the 4-bit encoded DNA data,
///   as produced by [`encode_dna_prefer_simd`].
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
/// - x86_64 with SSSE3: Processes 16 nucleotides per iteration using SIMD shuffle
/// - ARM64 (aarch64): Uses NEON to process 16 nucleotides per iteration
/// - All platforms: Uses static lookup table for fast 4-bit → ASCII conversion
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
/// use simdna::dna_simd_encoder::{encode_dna_prefer_simd, decode_dna_prefer_simd};
///
/// let original = b"ACGTRYSWKMBDHVN-";
/// let encoded = encode_dna_prefer_simd(original);
/// let decoded = decode_dna_prefer_simd(&encoded, original.len());
/// assert_eq!(decoded, original);
/// ```
///
/// Partial decoding:
/// ```rust
/// use simdna::dna_simd_encoder::{encode_dna_prefer_simd, decode_dna_prefer_simd};
///
/// let original = b"ACGTACGT";
/// let encoded = encode_dna_prefer_simd(original);
/// let first_four = decode_dna_prefer_simd(&encoded, 4);
/// assert_eq!(first_four, b"ACGT");
/// ```
pub fn decode_dna_prefer_simd(encoded: &[u8], length: usize) -> Vec<u8> {
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
/// Optimized implementation that processes 32 nucleotides at a time using
/// SIMD instructions:
/// 1. Loads two 16-byte chunks into SSE registers (32 nucleotides)
/// 2. Uses SIMD shuffle for parallel ASCII→4-bit encoding
/// 3. Packs 32 4-bit values into 16 bytes using SIMD operations
///
/// This approach amortizes the packing overhead across more data and
/// utilizes instruction-level parallelism by processing two registers
/// in parallel.
///
/// Remainder nucleotides (less than 32) are handled by a 16-byte path
/// or scalar fallback.
///
/// # Safety
///
/// This function requires SSSE3 support. Caller must verify via
/// `is_x86_feature_detected!("ssse3")` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn encode_x86_ssse3(sequence: &[u8], output: &mut [u8]) {
    let len = sequence.len();
    let simd32_len = (len / 32) * 32;
    let mut out_idx = 0;
    let mut in_idx = 0;

    // Lookup tables for SIMD-based ASCII to 4-bit encoding
    // We use a two-table approach based on nibble decomposition:
    // - For letters A-Z (0x41-0x5A), we mask to get the low nibble (0x1-0x1A)
    // - The lookup table maps these nibbles to 4-bit codes
    //
    // Mapping for uppercase letters (low nibble in range 0x1-0x16):
    // A(1)=0, B(2)=A, C(3)=1, D(4)=B, G(7)=2, H(8)=C, K(B)=8, M(D)=9,
    // N(E)=E, R(2)=4, S(3)=6, T(4)=3, U(5)=3, V(6)=D, W(7)=7, Y(9)=5
    //
    // Note: Some letters share low nibbles (e.g., R and B both have low nibble 2)
    // We handle this with a high-nibble check

    // Process 32 bytes (two 16-byte chunks) at a time for better ILP
    while in_idx < simd32_len {
        // Load 32 bytes into two registers
        let chunk0 = _mm_loadu_si128(sequence.as_ptr().add(in_idx) as *const __m128i);
        let chunk1 = _mm_loadu_si128(sequence.as_ptr().add(in_idx + 16) as *const __m128i);

        // Encode both chunks in parallel (utilizing superscalar execution)
        let encoded0 = encode_16_bytes_simd_x86(chunk0);
        let encoded1 = encode_16_bytes_simd_x86(chunk1);

        // Pack both 16-byte encoded results into 8 bytes each using SIMD
        let packed0 = pack_16_to_8_bytes_simd_x86(encoded0);
        let packed1 = pack_16_to_8_bytes_simd_x86(encoded1);

        // Store 16 bytes of packed output
        _mm_storel_epi64(output.as_mut_ptr().add(out_idx) as *mut __m128i, packed0);
        _mm_storel_epi64(
            output.as_mut_ptr().add(out_idx + 8) as *mut __m128i,
            packed1,
        );

        in_idx += 32;
        out_idx += 16;
    }

    // Handle 16-byte remainder if present
    if in_idx + 16 <= len {
        let chunk = _mm_loadu_si128(sequence.as_ptr().add(in_idx) as *const __m128i);
        let encoded = encode_16_bytes_simd_x86(chunk);
        let packed = pack_16_to_8_bytes_simd_x86(encoded);
        _mm_storel_epi64(output.as_mut_ptr().add(out_idx) as *mut __m128i, packed);
        in_idx += 16;
        out_idx += 8;
    }

    // Handle final remainder with scalar code
    if in_idx < len {
        encode_scalar(&sequence[in_idx..], &mut output[out_idx..]);
    }
}

/// SSSE3-accelerated DNA encoding with prefetching for x86_64.
/// Direct version that handles case-insensitivity via LUT and doesn't require pre-padding.
///
/// # Safety
///
/// This function requires SSSE3 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn encode_x86_ssse3_direct(sequence: &[u8], output: &mut [u8]) {
    let len = sequence.len();
    let simd32_len = (len / 32) * 32;
    let mut out_idx = 0;
    let mut in_idx = 0;

    // Process 32 bytes at a time with prefetching
    while in_idx < simd32_len {
        // Prefetch ahead for large sequences
        if in_idx + PREFETCH_DISTANCE < len {
            prefetch_read(sequence.as_ptr().add(in_idx + PREFETCH_DISTANCE));
        }

        // Load 32 bytes into two registers
        let chunk0 = _mm_loadu_si128(sequence.as_ptr().add(in_idx) as *const __m128i);
        let chunk1 = _mm_loadu_si128(sequence.as_ptr().add(in_idx + 16) as *const __m128i);

        // Encode both chunks using fast LUT
        let encoded0 = encode_16_bytes_simd_x86_fast(chunk0);
        let encoded1 = encode_16_bytes_simd_x86_fast(chunk1);

        // Pack both 16-byte encoded results into 8 bytes each
        let packed0 = pack_16_to_8_bytes_simd_x86(encoded0);
        let packed1 = pack_16_to_8_bytes_simd_x86(encoded1);

        // Store 16 bytes of packed output
        _mm_storel_epi64(output.as_mut_ptr().add(out_idx) as *mut __m128i, packed0);
        _mm_storel_epi64(
            output.as_mut_ptr().add(out_idx + 8) as *mut __m128i,
            packed1,
        );

        in_idx += 32;
        out_idx += 16;
    }

    // Handle 16-byte remainder if present
    if in_idx + 16 <= len {
        let chunk = _mm_loadu_si128(sequence.as_ptr().add(in_idx) as *const __m128i);
        let encoded = encode_16_bytes_simd_x86_fast(chunk);
        let packed = pack_16_to_8_bytes_simd_x86(encoded);
        _mm_storel_epi64(output.as_mut_ptr().add(out_idx) as *mut __m128i, packed);
        in_idx += 16;
        out_idx += 8;
    }

    // Handle final remainder with optimized scalar code
    if in_idx < len {
        encode_scalar_optimized(&sequence[in_idx..], &mut output[out_idx..]);
    }
}

/// Encodes 16 ASCII nucleotide bytes using the fast static LUT.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn encode_16_bytes_simd_x86_fast(input: __m128i) -> __m128i {
    let mut bytes: [u8; 16] = std::mem::transmute(input);
    for b in bytes.iter_mut() {
        *b = char_to_4bit_fast(*b);
    }
    _mm_loadu_si128(bytes.as_ptr() as *const __m128i)
}

/// Encodes 16 ASCII nucleotide bytes to 16 4-bit encoded bytes using SIMD.
///
/// Uses scalar lookup per byte since DNA uses a sparse portion of ASCII.
/// The overhead of complex SIMD lookup for sparse ASCII isn't worth it,
/// but we keep the data in registers for efficient packing.
///
/// # Safety
///
/// Requires SSSE3 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn encode_16_bytes_simd_x86(input: __m128i) -> __m128i {
    // Extract bytes, encode each, and reload
    // This hybrid approach is efficient because the DNA alphabet is sparse
    // and a full SIMD lookup would require multiple tables
    let mut bytes: [u8; 16] = std::mem::transmute(input);
    for b in bytes.iter_mut() {
        *b = char_to_4bit(*b);
    }
    _mm_loadu_si128(bytes.as_ptr() as *const __m128i)
}

/// Packs 16 4-bit encoded values into 8 bytes using SIMD operations.
///
/// Takes an SSE register containing 16 bytes (each with value 0-15)
/// and packs adjacent pairs into 8 bytes: (byte[0] << 4) | byte[1], etc.
///
/// Uses SIMD shuffle and bitwise operations for efficient packing.
///
/// # Safety
///
/// Requires SSSE3 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn pack_16_to_8_bytes_simd_x86(encoded: __m128i) -> __m128i {
    // Shuffle to separate even and odd bytes:
    // evens: bytes at positions 0, 2, 4, 6, 8, 10, 12, 14 → positions 0-7
    // odds:  bytes at positions 1, 3, 5, 7, 9, 11, 13, 15 → positions 0-7
    let shuffle_evens = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    let shuffle_odds = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    // Get evens and odds in the low 8 bytes of each register
    let evens = _mm_shuffle_epi8(encoded, shuffle_evens);
    let odds = _mm_shuffle_epi8(encoded, shuffle_odds);

    // Shift evens left by 4 bits: each byte becomes (even << 4)
    let evens_shifted = _mm_slli_epi16(evens, 4);

    // Mask to keep only the lower nibble of odds and upper nibble of shifted evens
    let mask_upper = _mm_set1_epi8(0xF0_u8 as i8);
    let mask_lower = _mm_set1_epi8(0x0F);

    let evens_masked = _mm_and_si128(evens_shifted, mask_upper);
    let odds_masked = _mm_and_si128(odds, mask_lower);

    // Combine: (even << 4) | odd
    _mm_or_si128(evens_masked, odds_masked)
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
    // Index: 0=Gap, 1=A, 2=C, 3=M, 4=T, 5=W, 6=Y, 7=H, 8=G, 9=R, A=S, B=V, C=K, D=D, E=B, F=N
    let lookup = _mm_setr_epi8(
        b'-' as i8, b'A' as i8, b'C' as i8, b'M' as i8, b'T' as i8, b'W' as i8, b'Y' as i8,
        b'H' as i8, b'G' as i8, b'R' as i8, b'S' as i8, b'V' as i8, b'K' as i8, b'D' as i8,
        b'B' as i8, b'N' as i8,
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
/// Optimized implementation that processes 32 nucleotides at a time using
/// NEON vector instructions:
/// 1. Loads two 16-byte chunks into NEON registers (32 nucleotides)
/// 2. Converts each byte to 4-bit encoding
/// 3. Packs 32 4-bit values into 16 bytes using SIMD operations
///
/// This approach amortizes the packing overhead across more data and
/// utilizes instruction-level parallelism by processing two registers
/// in parallel.
///
/// Remainder nucleotides (less than 32) are handled by a 16-byte path
/// or scalar fallback.
///
/// # Safety
///
/// This function requires NEON support (always available on ARM64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
unsafe fn encode_arm_neon(sequence: &[u8], output: &mut [u8]) {
    let len = sequence.len();
    let simd32_len = (len / 32) * 32;
    let mut out_idx = 0;
    let mut in_idx = 0;

    // Process 32 bytes (two 16-byte chunks) at a time for better ILP
    while in_idx < simd32_len {
        // Load 32 bytes into two registers
        let chunk0 = vld1q_u8(sequence.as_ptr().add(in_idx));
        let chunk1 = vld1q_u8(sequence.as_ptr().add(in_idx + 16));

        // Encode both chunks in parallel (utilizing superscalar execution)
        let encoded0 = encode_16_bytes_simd_neon(chunk0);
        let encoded1 = encode_16_bytes_simd_neon(chunk1);

        // Pack both 16-byte encoded results into 8 bytes each using SIMD
        let packed0 = pack_16_to_8_bytes_simd_neon(encoded0);
        let packed1 = pack_16_to_8_bytes_simd_neon(encoded1);

        // Store 16 bytes of packed output
        vst1_u8(output.as_mut_ptr().add(out_idx), packed0);
        vst1_u8(output.as_mut_ptr().add(out_idx + 8), packed1);

        in_idx += 32;
        out_idx += 16;
    }

    // Handle 16-byte remainder if present
    if in_idx + 16 <= len {
        let chunk = vld1q_u8(sequence.as_ptr().add(in_idx));
        let encoded = encode_16_bytes_simd_neon(chunk);
        let packed = pack_16_to_8_bytes_simd_neon(encoded);
        vst1_u8(output.as_mut_ptr().add(out_idx), packed);
        in_idx += 16;
        out_idx += 8;
    }

    // Handle final remainder with scalar code
    if in_idx < len {
        encode_scalar(&sequence[in_idx..], &mut output[out_idx..]);
    }
}

/// NEON-accelerated DNA encoding with prefetching (direct, no prior allocation).
///
/// Optimized implementation that:
/// 1. Uses prefetch hints for cache efficiency
/// 2. Processes 32 nucleotides at a time using NEON vector instructions
/// 3. Uses fast LUT for encoding
///
/// # Safety
///
/// This function requires NEON support (always available on ARM64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_arm_neon_direct(sequence: &[u8], output: &mut [u8]) {
    let len = sequence.len();
    let simd32_len = (len / 32) * 32;
    let mut out_idx = 0;
    let mut in_idx = 0;

    // Process 32 bytes (two 16-byte chunks) at a time with prefetching
    while in_idx < simd32_len {
        // Prefetch ahead for next iterations
        if in_idx + PREFETCH_DISTANCE < len {
            prefetch_read(sequence.as_ptr().add(in_idx + PREFETCH_DISTANCE));
        }

        // Load 32 bytes into two registers
        let chunk0 = vld1q_u8(sequence.as_ptr().add(in_idx));
        let chunk1 = vld1q_u8(sequence.as_ptr().add(in_idx + 16));

        // Encode both chunks in parallel using fast LUT
        let encoded0 = encode_16_bytes_simd_neon_fast(chunk0);
        let encoded1 = encode_16_bytes_simd_neon_fast(chunk1);

        // Pack both 16-byte encoded results into 8 bytes each using SIMD
        let packed0 = pack_16_to_8_bytes_simd_neon(encoded0);
        let packed1 = pack_16_to_8_bytes_simd_neon(encoded1);

        // Store 16 bytes of packed output
        vst1_u8(output.as_mut_ptr().add(out_idx), packed0);
        vst1_u8(output.as_mut_ptr().add(out_idx + 8), packed1);

        in_idx += 32;
        out_idx += 16;
    }

    // Handle 16-byte remainder if present
    if in_idx + 16 <= len {
        let chunk = vld1q_u8(sequence.as_ptr().add(in_idx));
        let encoded = encode_16_bytes_simd_neon_fast(chunk);
        let packed = pack_16_to_8_bytes_simd_neon(encoded);
        vst1_u8(output.as_mut_ptr().add(out_idx), packed);
        in_idx += 16;
        out_idx += 8;
    }

    // Handle final remainder with optimized scalar code
    if in_idx < len {
        encode_scalar_optimized(&sequence[in_idx..], &mut output[out_idx..]);
    }
}

/// Encodes 16 ASCII nucleotide bytes to 16 4-bit encoded bytes using NEON (fast LUT version).
///
/// Uses the static ENCODE_LUT for fast case-insensitive encoding.
///
/// # Safety
///
/// Requires NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn encode_16_bytes_simd_neon_fast(input: uint8x16_t) -> uint8x16_t {
    // Extract bytes, encode each using fast LUT, and reload
    let mut bytes = [0u8; 16];
    vst1q_u8(bytes.as_mut_ptr(), input);
    for b in bytes.iter_mut() {
        *b = char_to_4bit_fast(*b);
    }
    vld1q_u8(bytes.as_ptr())
}

/// Encodes 16 ASCII nucleotide bytes to 16 4-bit encoded bytes using NEON.
///
/// Uses scalar lookup per byte since DNA uses a sparse portion of ASCII.
/// The overhead of complex SIMD lookup for sparse ASCII isn't worth it,
/// but we keep the data in registers for efficient packing.
///
/// # Safety
///
/// Requires NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(dead_code)]
unsafe fn encode_16_bytes_simd_neon(input: uint8x16_t) -> uint8x16_t {
    // Extract bytes, encode each, and reload
    let mut bytes = [0u8; 16];
    vst1q_u8(bytes.as_mut_ptr(), input);
    for b in bytes.iter_mut() {
        *b = char_to_4bit(*b);
    }
    vld1q_u8(bytes.as_ptr())
}

/// Packs 16 4-bit encoded values into 8 bytes using NEON SIMD operations.
///
/// Takes a NEON register containing 16 bytes (each with value 0-15)
/// and packs adjacent pairs into 8 bytes: (byte[0] << 4) | byte[1], etc.
///
/// Uses NEON table lookup and bitwise operations for efficient packing.
///
/// # Safety
///
/// Requires NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn pack_16_to_8_bytes_simd_neon(encoded: uint8x16_t) -> uint8x8_t {
    // Create shuffle indices to separate evens and odds
    // evens: 0, 2, 4, 6, 8, 10, 12, 14
    // odds:  1, 3, 5, 7, 9, 11, 13, 15
    let shuffle_evens: [u8; 8] = [0, 2, 4, 6, 8, 10, 12, 14];
    let shuffle_odds: [u8; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

    let evens_idx = vld1_u8(shuffle_evens.as_ptr());
    let odds_idx = vld1_u8(shuffle_odds.as_ptr());

    // Use table lookup to gather evens and odds
    let evens = vqtbl1_u8(encoded, evens_idx);
    let odds = vqtbl1_u8(encoded, odds_idx);

    // Shift evens left by 4 bits
    let evens_shifted = vshl_n_u8::<4>(evens);

    // Combine: (even << 4) | odd
    vorr_u8(evens_shifted, odds)
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
    // Index: 0=Gap, 1=A, 2=C, 3=M, 4=T, 5=W, 6=Y, 7=H, 8=G, 9=R, A=S, B=V, C=K, D=D, E=B, F=N
    let lookup_data: [u8; 16] = [
        b'-', b'A', b'C', b'M', b'T', b'W', b'Y', b'H', b'G', b'R', b'S', b'V', b'K', b'D', b'B',
        b'N',
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
/// This legacy function is preserved for compatibility. Prefer using
/// `encode_scalar_optimized` which processes 4 nucleotides per iteration.
///
/// # Arguments
///
/// * `sequence` - Slice of ASCII nucleotide bytes (case-insensitive)
/// * `output` - Pre-allocated output buffer for encoded bytes
///
/// # Panics
///
/// Panics if `output` is too small to hold the encoded result.
#[allow(dead_code)]
pub fn encode_scalar(sequence: &[u8], output: &mut [u8]) {
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
pub fn decode_scalar(encoded: &[u8], output: &mut [u8], length: usize) {
    let mut out_idx = 0;
    for &byte in encoded {
        if out_idx >= length {
            break;
        }
        output[out_idx] = fourbit_to_char_fast((byte >> 4) & 0xF);
        out_idx += 1;
        if out_idx >= length {
            break;
        }
        output[out_idx] = fourbit_to_char_fast(byte & 0xF);
        out_idx += 1;
    }
}

/// Optimized scalar decoding that processes 2 encoded bytes (4 nucleotides) at a time.
/// Uses the static LUT for fast lookups.
#[inline]
#[allow(dead_code)]
pub fn decode_scalar_optimized(encoded: &[u8], output: &mut [u8], length: usize) {
    let mut out_idx = 0;
    let mut enc_idx = 0;
    let enc_len = encoded.len();

    // Process 2 encoded bytes (4 nucleotides) at a time
    while enc_idx + 2 <= enc_len && out_idx + 4 <= length {
        let byte0 = encoded[enc_idx];
        let byte1 = encoded[enc_idx + 1];

        output[out_idx] = fourbit_to_char_fast((byte0 >> 4) & 0xF);
        output[out_idx + 1] = fourbit_to_char_fast(byte0 & 0xF);
        output[out_idx + 2] = fourbit_to_char_fast((byte1 >> 4) & 0xF);
        output[out_idx + 3] = fourbit_to_char_fast(byte1 & 0xF);

        enc_idx += 2;
        out_idx += 4;
    }

    // Handle remaining bytes
    while enc_idx < enc_len && out_idx < length {
        let byte = encoded[enc_idx];
        output[out_idx] = fourbit_to_char_fast((byte >> 4) & 0xF);
        out_idx += 1;
        if out_idx < length {
            output[out_idx] = fourbit_to_char_fast(byte & 0xF);
            out_idx += 1;
        }
        enc_idx += 1;
    }
}

/// Converts an ASCII nucleotide byte to its 4-bit encoding.
///
/// Handles all IUPAC nucleotide codes. Invalid characters are
/// encoded as gap (0x0).
///
/// # Encoding Table
///
/// | Input | Output | Input | Output |
/// |-------|--------|-------|--------|
/// | A, a  | 0x1    | K, k  | 0xC    |
/// | C, c  | 0x2    | M, m  | 0x3    |
/// | G, g  | 0x8    | B, b  | 0xE    |
/// | T, t  | 0x4    | D, d  | 0xD    |
/// | U, u  | 0x4    | H, h  | 0x7    |
/// | R, r  | 0x9    | V, v  | 0xB    |
/// | Y, y  | 0x6    | N, n  | 0xF    |
/// | S, s  | 0xA    | -, .  | 0x0    |
/// | W, w  | 0x5    | other | 0x0    |
///
/// # Arguments
///
/// * `c` - ASCII byte representing a nucleotide
///
/// # Returns
///
/// 4-bit encoding (0-15)
#[inline]
#[allow(dead_code)]
pub fn char_to_4bit(c: u8) -> u8 {
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
/// | 0x0   | -      | 0x8   | G      |
/// | 0x1   | A      | 0x9   | R      |
/// | 0x2   | C      | 0xA   | S      |
/// | 0x3   | M      | 0xB   | V      |
/// | 0x4   | T      | 0xC   | K      |
/// | 0x5   | W      | 0xD   | D      |
/// | 0x6   | Y      | 0xE   | B      |
/// | 0x7   | H      | 0xF   | N      |
///
/// # Arguments
///
/// * `bits` - 4-bit encoded value (only lower 4 bits are used)
///
/// # Returns
///
/// ASCII byte for the nucleotide character
#[inline]
#[allow(dead_code)]
pub fn fourbit_to_char(bits: u8) -> u8 {
    // Indexed by 4-bit encoding value
    // 0=Gap, 1=A, 2=C, 3=M, 4=T, 5=W, 6=Y, 7=H, 8=G, 9=R, A=S, B=V, C=K, D=D, E=B, F=N
    const DECODE_TABLE: [u8; 16] = [
        b'-', b'A', b'C', b'M', b'T', b'W', b'Y', b'H', b'G', b'R', b'S', b'V', b'K', b'D', b'B',
        b'N',
    ];
    DECODE_TABLE[(bits & 0xF) as usize]
}

// ============================================================================
// Complement and Reverse Complement
// ============================================================================

/// Computes the complement of a 4-bit encoded nucleotide using bit rotation.
///
/// The encoding scheme is designed so that rotating any 4-bit value by 2 bits
/// produces its Watson-Crick complement. This is ~2x faster than lookup tables
/// for complement calculation.
///
/// # Complement Pairs
///
/// - A (0x1) ↔ T (0x4)
/// - C (0x2) ↔ G (0x8)
/// - R (0x9) ↔ Y (0x6)
/// - K (0xC) ↔ M (0x3)
/// - D (0xD) ↔ H (0x7)
/// - B (0xE) ↔ V (0xB)
/// - S (0xA), W (0x5), N (0xF), Gap (0x0) are self-complementary
///
/// # Arguments
///
/// * `bits` - 4-bit encoded nucleotide value
///
/// # Returns
///
/// The 4-bit encoded complement
///
/// # Example
///
/// ```rust
/// use simdna::dna_simd_encoder::{complement_4bit, encoding};
///
/// assert_eq!(complement_4bit(encoding::A), encoding::T);
/// assert_eq!(complement_4bit(encoding::C), encoding::G);
/// assert_eq!(complement_4bit(encoding::N), encoding::N); // Self-complementary
/// ```
#[inline]
pub fn complement_4bit(bits: u8) -> u8 {
    // 2-bit rotation within a 4-bit value: rotate left by 2
    ((bits << 2) | (bits >> 2)) & 0xF
}

/// Computes the complement of a packed byte containing two 4-bit encoded nucleotides.
///
/// Both nibbles are complemented in parallel using bit manipulation.
///
/// # Arguments
///
/// * `packed` - A byte containing two 4-bit encoded nucleotides (high nibble, low nibble)
///
/// # Returns
///
/// A byte with both nibbles complemented
#[inline]
pub fn complement_packed_byte(packed: u8) -> u8 {
    // Extract high and low nibbles
    let high = packed >> 4;
    let low = packed & 0xF;
    // Complement each nibble and repack
    (complement_4bit(high) << 4) | complement_4bit(low)
}

/// Minimum byte count below which scalar is faster than SIMD due to setup overhead.
/// Based on benchmarking, SIMD overhead is worthwhile only for larger sequences.
const SIMD_REVCOMP_THRESHOLD: usize = 32;

/// Computes the reverse complement of encoded DNA data.
///
/// This function operates directly on 4-bit packed encoded data, avoiding
/// the overhead of decoding to ASCII and re-encoding. It uses the bit rotation
/// property of the encoding scheme for efficient complement calculation.
///
/// # Algorithm
///
/// 1. Reverse the byte order
/// 2. Swap nibbles within each byte (to reverse nucleotide order)
/// 3. Complement each nibble using 2-bit rotation
///
/// # Arguments
///
/// * `encoded` - Slice of 4-bit packed encoded DNA data
/// * `length` - Original sequence length (number of nucleotides)
///
/// # Returns
///
/// A `Vec<u8>` containing the reverse complement in the same packed format
///
/// # Thread Safety
///
/// This function is thread-safe and can be called from multiple threads
/// concurrently without synchronization.
///
/// # Performance
///
/// This function uses SIMD instructions when available:
/// - x86_64 with SSSE3: Processes 16 bytes per iteration
/// - ARM64 (aarch64): Uses NEON to process 16 bytes per iteration
/// - Other platforms: Uses optimized scalar implementation
///
/// # Example
///
/// ```rust
/// use simdna::dna_simd_encoder::{encode_dna_prefer_simd, decode_dna_prefer_simd, reverse_complement_encoded};
///
/// let sequence = b"ACGT";
/// let encoded = encode_dna_prefer_simd(sequence);
/// let rc_encoded = reverse_complement_encoded(&encoded, sequence.len());
/// let rc_decoded = decode_dna_prefer_simd(&rc_encoded, sequence.len());
/// assert_eq!(rc_decoded, b"ACGT"); // ACGT is its own reverse complement
///
/// let sequence2 = b"AACG";
/// let encoded2 = encode_dna_prefer_simd(sequence2);
/// let rc_encoded2 = reverse_complement_encoded(&encoded2, sequence2.len());
/// let rc_decoded2 = decode_dna_prefer_simd(&rc_encoded2, sequence2.len());
/// assert_eq!(rc_decoded2, b"CGTT"); // AACG -> reverse: GCAA -> complement: CGTT
/// ```
pub fn reverse_complement_encoded(encoded: &[u8], length: usize) -> Vec<u8> {
    if encoded.is_empty() {
        return Vec::new();
    }

    let num_bytes = encoded.len();
    let is_odd_length = length % 2 == 1;
    let mut output = vec![0u8; num_bytes];

    // For very small sequences, scalar is faster due to SIMD setup overhead
    if num_bytes < SIMD_REVCOMP_THRESHOLD {
        reverse_complement_scalar(encoded, &mut output, length);
        return output;
    }

    // Try SIMD implementations - works for both even and odd lengths
    // For odd lengths, SIMD does byte-level reverse+complement, then we fix nibble alignment
    let used_simd = reverse_complement_with_simd_if_available(encoded, &mut output, length);

    if !used_simd {
        reverse_complement_scalar(encoded, &mut output, length);
        return output;
    }

    // For odd-length sequences, SIMD processed as if even-length.
    // The padding nibble (which was in low nibble of last input byte) is now
    // in the HIGH nibble of the first output byte after SIMD processing.
    // We need to shift all nibbles left by one position to move padding back
    // to the low nibble of the last byte.
    if is_odd_length {
        shift_nibbles_left(&mut output);
    }

    output
}

/// Shifts all nibbles in the buffer left by one position.
/// The high nibble of byte[0] is discarded (it was padding).
/// The low nibble of the last byte becomes 0 (new padding).
///
/// This efficiently handles odd-length sequences after SIMD processing.
#[inline]
fn shift_nibbles_left(buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    let len = buffer.len();

    // Process bytes from start to end-1
    // Each byte takes its high nibble from its own low nibble,
    // and its low nibble from the next byte's high nibble.
    for i in 0..len - 1 {
        let current_low = buffer[i] & 0x0F;
        let next_high = buffer[i + 1] >> 4;
        buffer[i] = (current_low << 4) | next_high;
    }

    // Last byte: high nibble from its own low nibble, low nibble becomes 0 (padding)
    buffer[len - 1] = (buffer[len - 1] & 0x0F) << 4;
}

/// Attempts to use SIMD for reverse complement if available.
/// Only called for even-length sequences >= SIMD_REVCOMP_THRESHOLD bytes.
#[inline]
fn reverse_complement_with_simd_if_available(
    encoded: &[u8],
    output: &mut [u8],
    length: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            unsafe { reverse_complement_x86_ssse3(encoded, output, length) };
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { reverse_complement_arm_neon(encoded, output, length) };
        return true;
    }

    #[allow(unreachable_code)]
    false
}

/// Scalar implementation of reverse complement.
///
/// Processes the encoded data to produce the reverse complement:
/// 1. Iterates through bytes in reverse order
/// 2. Swaps nibbles within each byte
/// 3. Complements each nibble
///
/// For odd-length sequences, handles the nibble shift without allocation.
fn reverse_complement_scalar(encoded: &[u8], output: &mut [u8], length: usize) {
    let num_bytes = encoded.len();
    let is_odd_length = length % 2 == 1;

    if !is_odd_length {
        // Even length: simple byte-by-byte reversal with nibble swap and complement
        for (i, &byte) in encoded.iter().enumerate() {
            let reverse_idx = num_bytes - 1 - i;

            // Extract nibbles
            let high = byte >> 4;
            let low = byte & 0xF;

            // Complement each nibble and swap positions
            let high_comp = complement_4bit(high);
            let low_comp = complement_4bit(low);

            // Swap and pack: the low nibble becomes high, high becomes low
            output[reverse_idx] = (low_comp << 4) | high_comp;
        }
    } else {
        // Odd length: need to handle nibble shifting without allocation
        // The input has `length` nucleotides packed into `num_bytes` bytes,
        // with the last byte having a padding nibble in the low position.
        //
        // For reverse complement, we need to:
        // 1. Read nucleotides in reverse order (ignoring padding)
        // 2. Complement each
        // 3. Pack into output with padding at the end
        //
        // Process nucleotide by nucleotide to handle the shift correctly.
        for out_nuc_idx in 0..length {
            // Input nucleotide index (reversed)
            let in_nuc_idx = length - 1 - out_nuc_idx;

            // Find the byte and nibble position in input
            let in_byte_idx = in_nuc_idx / 2;
            let in_is_high = in_nuc_idx.is_multiple_of(2);

            // Extract the nibble
            let nibble = if in_is_high {
                encoded[in_byte_idx] >> 4
            } else {
                encoded[in_byte_idx] & 0xF
            };

            // Complement
            let comp_nibble = complement_4bit(nibble);

            // Find the byte and nibble position in output
            let out_byte_idx = out_nuc_idx / 2;
            let out_is_high = out_nuc_idx.is_multiple_of(2);

            // Pack into output
            if out_is_high {
                output[out_byte_idx] = comp_nibble << 4;
            } else {
                output[out_byte_idx] |= comp_nibble;
            }
        }
    }
}

/// SSSE3-accelerated reverse complement for x86_64.
/// Handles both even and odd-length sequences. For odd-length, caller must
/// call shift_nibbles_left() on the output to fix nibble alignment.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn reverse_complement_x86_ssse3(encoded: &[u8], output: &mut [u8], _length: usize) {
    let num_bytes = encoded.len();

    // Caller should ensure sufficient size
    if num_bytes < 16 {
        // Fallback to scalar for small inputs (caller handles odd-length shift)
        for (i, &byte) in encoded.iter().enumerate() {
            let reverse_idx = num_bytes - 1 - i;
            let high = byte >> 4;
            let low = byte & 0xF;
            let high_comp = complement_4bit(high);
            let low_comp = complement_4bit(low);
            output[reverse_idx] = (low_comp << 4) | high_comp;
        }
        return;
    }

    // Byte reverse shuffle mask
    let reverse_mask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let mask_0f = _mm_set1_epi8(0x0F);

    // Calculate how many full 16-byte chunks we can process
    let full_chunks = num_bytes / 16;
    let remainder = num_bytes % 16;

    // Process full 16-byte chunks with SIMD
    for chunk_idx in 0..full_chunks {
        let in_offset = chunk_idx * 16;
        let out_offset = num_bytes - (chunk_idx + 1) * 16;

        // Load 16 bytes from input
        let data = _mm_loadu_si128(encoded[in_offset..].as_ptr() as *const __m128i);

        // Reverse byte order
        let reversed = _mm_shuffle_epi8(data, reverse_mask);

        // Swap nibbles within each byte: ((x & 0x0F) << 4) | ((x >> 4) & 0x0F)
        let low_nibbles = _mm_and_si128(reversed, mask_0f);
        let high_nibbles = _mm_and_si128(_mm_srli_epi16(reversed, 4), mask_0f);
        let swapped = _mm_or_si128(_mm_slli_epi16(low_nibbles, 4), high_nibbles);

        // Complement each nibble using bit rotation
        // For each nibble: ((n << 2) | (n >> 2)) & 0xF
        let low = _mm_and_si128(swapped, mask_0f);
        let high = _mm_and_si128(_mm_srli_epi16(swapped, 4), mask_0f);

        // Rotate nibbles: ((n << 2) | (n >> 2)) & 0xF
        let low_rot = _mm_and_si128(
            _mm_or_si128(_mm_slli_epi16(low, 2), _mm_srli_epi16(low, 2)),
            mask_0f,
        );
        let high_rot = _mm_and_si128(
            _mm_or_si128(_mm_slli_epi16(high, 2), _mm_srli_epi16(high, 2)),
            mask_0f,
        );

        // Repack nibbles
        let result = _mm_or_si128(_mm_slli_epi16(high_rot, 4), low_rot);

        _mm_storeu_si128(output[out_offset..].as_mut_ptr() as *mut __m128i, result);
    }

    // Handle remainder bytes with efficient scalar loop (no function call overhead)
    if remainder > 0 {
        let in_start = full_chunks * 16;
        let out_end = remainder;

        for i in 0..remainder {
            let byte = encoded[in_start + i];
            let reverse_idx = out_end - 1 - i;

            // Extract, complement, and swap nibbles
            let high = byte >> 4;
            let low = byte & 0xF;
            let high_comp = complement_4bit(high);
            let low_comp = complement_4bit(low);

            output[reverse_idx] = (low_comp << 4) | high_comp;
        }
    }
}

/// NEON-accelerated reverse complement for ARM64.
/// Handles both even and odd-length sequences. For odd-length, caller must
/// call shift_nibbles_left() on the output to fix nibble alignment.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn reverse_complement_arm_neon(encoded: &[u8], output: &mut [u8], _length: usize) {
    let num_bytes = encoded.len();

    // Caller should ensure sufficient size
    if num_bytes < 16 {
        // Fallback to scalar for small inputs (caller handles odd-length shift)
        for (i, &byte) in encoded.iter().enumerate() {
            let reverse_idx = num_bytes - 1 - i;
            let high = byte >> 4;
            let low = byte & 0xF;
            let high_comp = complement_4bit(high);
            let low_comp = complement_4bit(low);
            output[reverse_idx] = (low_comp << 4) | high_comp;
        }
        return;
    }

    // Byte reverse indices for vqtbl1q
    let reverse_indices: [u8; 16] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    let reverse_tbl = vld1q_u8(reverse_indices.as_ptr());

    let mask_0f = vdupq_n_u8(0x0F);

    // Calculate how many full 16-byte chunks we can process
    let full_chunks = num_bytes / 16;
    let remainder = num_bytes % 16;

    // Process full 16-byte chunks with SIMD
    for chunk_idx in 0..full_chunks {
        let in_offset = chunk_idx * 16;
        let out_offset = num_bytes - (chunk_idx + 1) * 16;

        // Load 16 bytes from input
        let data = vld1q_u8(encoded[in_offset..].as_ptr());

        // Reverse byte order
        let reversed = vqtbl1q_u8(data, reverse_tbl);

        // Swap nibbles within each byte
        let low_nibbles = vandq_u8(reversed, mask_0f);
        let high_nibbles = vandq_u8(vshrq_n_u8(reversed, 4), mask_0f);
        let swapped = vorrq_u8(vshlq_n_u8(low_nibbles, 4), high_nibbles);

        // Complement each nibble using bit rotation
        let low = vandq_u8(swapped, mask_0f);
        let high = vandq_u8(vshrq_n_u8(swapped, 4), mask_0f);

        // Rotate nibbles: ((n << 2) | (n >> 2)) & 0xF
        let low_rot = vandq_u8(vorrq_u8(vshlq_n_u8(low, 2), vshrq_n_u8(low, 2)), mask_0f);
        let high_rot = vandq_u8(vorrq_u8(vshlq_n_u8(high, 2), vshrq_n_u8(high, 2)), mask_0f);

        // Repack nibbles
        let result = vorrq_u8(vshlq_n_u8(high_rot, 4), low_rot);

        vst1q_u8(output[out_offset..].as_mut_ptr(), result);
    }

    // Handle remainder bytes with efficient scalar loop (no function call overhead)
    if remainder > 0 {
        let in_start = full_chunks * 16;
        let out_end = remainder;

        for i in 0..remainder {
            let byte = encoded[in_start + i];
            let reverse_idx = out_end - 1 - i;

            // Extract, complement, and swap nibbles
            let high = byte >> 4;
            let low = byte & 0xF;
            let high_comp = complement_4bit(high);
            let low_comp = complement_4bit(low);

            output[reverse_idx] = (low_comp << 4) | high_comp;
        }
    }
}

/// Computes the reverse complement of a DNA sequence.
///
/// This is a convenience function that encodes the input, computes the
/// reverse complement on the encoded data (using efficient bit rotation),
/// and decodes the result.
///
/// # Arguments
///
/// * `sequence` - A byte slice containing ASCII-encoded DNA nucleotides
///
/// # Returns
///
/// A `Vec<u8>` containing the reverse complement as ASCII nucleotides
///
/// # Example
///
/// ```rust
/// use simdna::dna_simd_encoder::reverse_complement;
///
/// let rc = reverse_complement(b"AACG");
/// assert_eq!(rc, b"CGTT"); // AACG -> reverse: GCAA -> complement: CGTT
///
/// let rc2 = reverse_complement(b"ACGT");
/// assert_eq!(rc2, b"ACGT"); // Palindromic sequence
/// ```
pub fn reverse_complement(sequence: &[u8]) -> Vec<u8> {
    if sequence.is_empty() {
        return Vec::new();
    }

    let encoded = encode_dna_prefer_simd(sequence);
    let rc_encoded = reverse_complement_encoded(&encoded, sequence.len());
    decode_dna_prefer_simd(&rc_encoded, sequence.len())
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
        assert_eq!(char_to_4bit(b'A'), 0x1);
        assert_eq!(char_to_4bit(b'C'), 0x2);
        assert_eq!(char_to_4bit(b'G'), 0x8);
        assert_eq!(char_to_4bit(b'T'), 0x4);
    }

    /// Tests lowercase nucleotide to 4-bit conversion.
    #[test]
    fn test_char_to_4bit_lowercase() {
        assert_eq!(char_to_4bit(b'a'), 0x1);
        assert_eq!(char_to_4bit(b'c'), 0x2);
        assert_eq!(char_to_4bit(b'g'), 0x8);
        assert_eq!(char_to_4bit(b't'), 0x4);
    }

    /// Tests IUPAC ambiguity codes to 4-bit conversion.
    #[test]
    fn test_char_to_4bit_iupac() {
        assert_eq!(char_to_4bit(b'R'), 0x9);
        assert_eq!(char_to_4bit(b'Y'), 0x6);
        assert_eq!(char_to_4bit(b'S'), 0xA);
        assert_eq!(char_to_4bit(b'W'), 0x5);
        assert_eq!(char_to_4bit(b'K'), 0xC);
        assert_eq!(char_to_4bit(b'M'), 0x3);
        assert_eq!(char_to_4bit(b'B'), 0xE);
        assert_eq!(char_to_4bit(b'D'), 0xD);
        assert_eq!(char_to_4bit(b'H'), 0x7);
        assert_eq!(char_to_4bit(b'V'), 0xB);
        assert_eq!(char_to_4bit(b'N'), 0xF);
    }

    /// Tests gap and unknown characters.
    #[test]
    fn test_char_to_4bit_gaps() {
        assert_eq!(char_to_4bit(b'-'), 0x0);
        assert_eq!(char_to_4bit(b'.'), 0x0);
        assert_eq!(char_to_4bit(b'X'), 0x0); // Invalid -> gap
        assert_eq!(char_to_4bit(b' '), 0x0); // Invalid -> gap
    }

    /// Tests RNA uracil to 4-bit conversion (maps to T).
    #[test]
    fn test_char_to_4bit_uracil() {
        assert_eq!(char_to_4bit(b'U'), 0x4);
        assert_eq!(char_to_4bit(b'u'), 0x4);
    }

    /// Tests 4-bit to ASCII character conversion.
    #[test]
    fn test_fourbit_to_char() {
        // New encoding: 0=Gap, 1=A, 2=C, 3=M, 4=T, 5=W, 6=Y, 7=H, 8=G, 9=R, A=S, B=V, C=K, D=D, E=B, F=N
        assert_eq!(fourbit_to_char(0x0), b'-');
        assert_eq!(fourbit_to_char(0x1), b'A');
        assert_eq!(fourbit_to_char(0x2), b'C');
        assert_eq!(fourbit_to_char(0x3), b'M');
        assert_eq!(fourbit_to_char(0x4), b'T');
        assert_eq!(fourbit_to_char(0x5), b'W');
        assert_eq!(fourbit_to_char(0x6), b'Y');
        assert_eq!(fourbit_to_char(0x7), b'H');
        assert_eq!(fourbit_to_char(0x8), b'G');
        assert_eq!(fourbit_to_char(0x9), b'R');
        assert_eq!(fourbit_to_char(0xA), b'S');
        assert_eq!(fourbit_to_char(0xB), b'V');
        assert_eq!(fourbit_to_char(0xC), b'K');
        assert_eq!(fourbit_to_char(0xD), b'D');
        assert_eq!(fourbit_to_char(0xE), b'B');
        assert_eq!(fourbit_to_char(0xF), b'N');
    }

    /// Tests that only the lower 4 bits are used in conversion.
    #[test]
    fn test_fourbit_to_char_masks_input() {
        // Values > 15 should use only lower 4 bits
        assert_eq!(fourbit_to_char(0x10), b'-'); // 0x10 & 0xF = 0 = Gap
        assert_eq!(fourbit_to_char(0xFF), b'N'); // 0xFF & 0xF = 15 = N
    }

    // ========================================================================
    // Encode/Decode Roundtrip Tests
    // ========================================================================

    /// Tests roundtrip for standard bases.
    #[test]
    fn test_encode_decode_roundtrip_standard_bases() {
        let sequence = b"ACGT";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests roundtrip for all IUPAC codes.
    #[test]
    fn test_encode_decode_roundtrip_all_iupac() {
        let sequence = b"ACGTRYSWKMBDHVN-";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests roundtrip for 16 nucleotides (SIMD boundary).
    #[test]
    fn test_encode_decode_roundtrip_16_nucleotides() {
        let sequence = b"ACGTACGTACGTACGT";
        let encoded = encode_dna_prefer_simd(sequence);
        assert_eq!(encoded.len(), 8); // 16 / 2 = 8
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests roundtrip for 32 nucleotides (2 SIMD blocks).
    #[test]
    fn test_encode_decode_roundtrip_32_nucleotides() {
        let sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
        let encoded = encode_dna_prefer_simd(sequence);
        assert_eq!(encoded.len(), 16);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests roundtrip for various non-aligned lengths.
    #[test]
    fn test_encode_decode_roundtrip_non_aligned() {
        for len in [1, 2, 3, 5, 7, 11, 15, 16, 17, 19, 23, 31, 32, 33, 48, 50] {
            let sequence: Vec<u8> = (0..len).map(|i| b"ACGT"[i % 4]).collect();
            let encoded = encode_dna_prefer_simd(&sequence);
            let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
            assert_eq!(decoded, sequence, "Failed for length {}", len);
        }
    }

    /// Tests compression ratio (2:1).
    #[test]
    fn test_encode_compression_ratio() {
        let sequence = b"ACGTACGTACGTACGT"; // 16 bytes
        let encoded = encode_dna_prefer_simd(sequence);
        assert_eq!(encoded.len(), 8); // 2:1 compression
    }

    /// Tests empty sequence handling.
    #[test]
    fn test_encode_empty_sequence() {
        let sequence = b"";
        let encoded = encode_dna_prefer_simd(sequence);
        assert!(encoded.is_empty());
    }

    /// Tests single nucleotide encoding/decoding.
    #[test]
    fn test_encode_single_nucleotide() {
        for &base in b"ACGTRYSWKMBDHVN-".iter() {
            let sequence = [base];
            let encoded = encode_dna_prefer_simd(&sequence);
            let decoded = decode_dna_prefer_simd(&encoded, 1);
            assert_eq!(decoded[0], base, "Failed for base {}", base as char);
        }
    }

    /// Tests specific encoding values.
    #[test]
    fn test_encode_specific_values() {
        // AC should encode to 0x12 (A=0x1 in high nibble, C=0x2 in low nibble)
        let sequence = b"AC";
        let encoded = encode_dna_prefer_simd(sequence);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], 0x12);

        // GT should encode to 0x84 (G=0x8 in high nibble, T=0x4 in low nibble)
        let sequence = b"GT";
        let encoded = encode_dna_prefer_simd(sequence);
        assert_eq!(encoded[0], 0x84);
    }

    // ========================================================================
    // IUPAC Ambiguity Code Tests
    // ========================================================================

    /// Tests that all IUPAC codes are faithfully preserved.
    #[test]
    fn test_iupac_codes_preserved() {
        let codes = b"ACGTRYSWKMBDHVN-";
        let encoded = encode_dna_prefer_simd(codes);
        let decoded = decode_dna_prefer_simd(&encoded, codes.len());
        assert_eq!(decoded, codes);
    }

    /// Tests IUPAC codes in longer sequences.
    #[test]
    fn test_iupac_in_long_sequence() {
        // Mix of standard and IUPAC codes
        let sequence = b"ACGTNNNNACGTRYSWACGTKMBDHVNACGT";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests N codes at SIMD boundary.
    #[test]
    fn test_iupac_n_at_simd_boundary() {
        let sequence = b"NNNNNNNNNNNNNNNN"; // 16 N's
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests all IUPAC codes in lowercase.
    #[test]
    fn test_iupac_lowercase() {
        let sequence = b"acgtryswkmbdhvn";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        // Decoded should be uppercase
        assert_eq!(decoded, b"ACGTRYSWKMBDHVN");
    }

    /// Tests specific encoding values for each IUPAC ambiguity code.
    #[test]
    fn test_iupac_specific_encoding_values() {
        // Each pair should encode to a specific byte value
        // Format: (high_nibble << 4) | low_nibble
        // New encoding: R=0x9, Y=0x6, S=0xA, W=0x5, K=0xC, M=0x3, B=0xE, D=0xD, H=0x7, V=0xB, N=0xF, Gap=0x0

        // RY should encode to 0x96 (R=0x9, Y=0x6)
        let encoded = encode_dna_prefer_simd(b"RY");
        assert_eq!(encoded[0], 0x96, "RY should encode to 0x96");

        // SW should encode to 0xA5 (S=0xA, W=0x5)
        let encoded = encode_dna_prefer_simd(b"SW");
        assert_eq!(encoded[0], 0xA5, "SW should encode to 0xA5");

        // KM should encode to 0xC3 (K=0xC, M=0x3)
        let encoded = encode_dna_prefer_simd(b"KM");
        assert_eq!(encoded[0], 0xC3, "KM should encode to 0xC3");

        // BD should encode to 0xED (B=0xE, D=0xD)
        let encoded = encode_dna_prefer_simd(b"BD");
        assert_eq!(encoded[0], 0xED, "BD should encode to 0xED");

        // HV should encode to 0x7B (H=0x7, V=0xB)
        let encoded = encode_dna_prefer_simd(b"HV");
        assert_eq!(encoded[0], 0x7B, "HV should encode to 0x7B");

        // N- should encode to 0xF0 (N=0xF, Gap=0x0)
        let encoded = encode_dna_prefer_simd(b"N-");
        assert_eq!(encoded[0], 0xF0, "N- should encode to 0xF0");
    }

    /// Tests each IUPAC code individually at odd positions (second nibble).
    #[test]
    fn test_iupac_at_odd_positions() {
        // New encoding: R=0x9, Y=0x6, S=0xA, W=0x5, K=0xC, M=0x3, B=0xE, D=0xD, H=0x7, V=0xB, N=0xF, Gap=0x0
        let codes: [(u8, u8); 12] = [
            (b'R', 0x9),
            (b'Y', 0x6),
            (b'S', 0xA),
            (b'W', 0x5),
            (b'K', 0xC),
            (b'M', 0x3),
            (b'B', 0xE),
            (b'D', 0xD),
            (b'H', 0x7),
            (b'V', 0xB),
            (b'N', 0xF),
            (b'-', 0x0),
        ];

        for (code, expected_nibble) in codes {
            // Put code in second position (low nibble)
            // A encodes to 0x1, so expected_byte is (0x1 << 4) | expected_nibble
            let sequence = [b'A', code];
            let encoded = encode_dna_prefer_simd(&sequence);
            let expected_byte = (0x1 << 4) | expected_nibble; // A (0x1) in high nibble
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
        // New encoding: R=0x9, Y=0x6, S=0xA, W=0x5, K=0xC, M=0x3, B=0xE, D=0xD, H=0x7, V=0xB, N=0xF, Gap=0x0
        let codes: [(u8, u8); 12] = [
            (b'R', 0x9),
            (b'Y', 0x6),
            (b'S', 0xA),
            (b'W', 0x5),
            (b'K', 0xC),
            (b'M', 0x3),
            (b'B', 0xE),
            (b'D', 0xD),
            (b'H', 0x7),
            (b'V', 0xB),
            (b'N', 0xF),
            (b'-', 0x0),
        ];

        for (code, expected_nibble) in codes {
            // Put code in first position (high nibble)
            // A encodes to 0x1, so expected_byte is (expected_nibble << 4) | 0x1
            let sequence = [code, b'A'];
            let encoded = encode_dna_prefer_simd(&sequence);
            let expected_byte = (expected_nibble << 4) | 0x1; // A (0x1) in low nibble
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
        // New encoding: R=0x9, Y=0x6, S=0xA, W=0x5, K=0xC, M=0x3, B=0xE, D=0xD, H=0x7, V=0xB, N=0xF
        let codes: [(u8, u8); 11] = [
            (b'r', 0x9),
            (b'y', 0x6),
            (b's', 0xA),
            (b'w', 0x5),
            (b'k', 0xC),
            (b'm', 0x3),
            (b'b', 0xE),
            (b'd', 0xD),
            (b'h', 0x7),
            (b'v', 0xB),
            (b'n', 0xF),
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

        let encoded = encode_dna_prefer_simd(&sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests all IUPAC codes repeated to fill multiple SIMD blocks.
    #[test]
    fn test_iupac_multiple_simd_blocks() {
        // 16 IUPAC codes × 4 = 64 bytes (4 SIMD blocks)
        let pattern = b"ACGTRYSWKMBDHVN-";
        let sequence: Vec<u8> = pattern.iter().cycle().take(64).copied().collect();

        let encoded = encode_dna_prefer_simd(&sequence);
        assert_eq!(encoded.len(), 32); // 64 / 2 = 32
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests each IUPAC code in isolation with roundtrip.
    #[test]
    fn test_iupac_individual_roundtrip() {
        let iupac_codes = b"RYSWKMBDHVN-";

        for &code in iupac_codes {
            // Single character
            let single = [code];
            let encoded = encode_dna_prefer_simd(&single);
            let decoded = decode_dna_prefer_simd(&encoded, 1);
            assert_eq!(decoded[0], code, "Single {} failed roundtrip", code as char);

            // Paired with itself
            let paired = [code, code];
            let encoded = encode_dna_prefer_simd(&paired);
            let decoded = decode_dna_prefer_simd(&encoded, 2);
            assert_eq!(decoded, paired, "Paired {} failed roundtrip", code as char);

            // 16 of the same code (SIMD path)
            let sixteen: Vec<u8> = vec![code; 16];
            let encoded = encode_dna_prefer_simd(&sixteen);
            let decoded = decode_dna_prefer_simd(&encoded, 16);
            assert_eq!(decoded, sixteen, "16x {} failed roundtrip", code as char);
        }
    }

    /// Tests IUPAC codes interspersed with standard bases.
    #[test]
    fn test_iupac_interspersed_with_bases() {
        let sequence = b"ANCTGNRTYCSWAGKT";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests dot character as gap (alternative gap notation).
    #[test]
    fn test_iupac_dot_as_gap() {
        let sequence = b"AC.GT";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
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
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, b"ACGTACGTACGTACGT");
    }

    /// Tests mixed case encoding.
    #[test]
    fn test_mixed_case_encoding() {
        let sequence = b"AcGtAcGtAcGtAcGt";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, b"ACGTACGTACGTACGT");
    }

    /// Tests that uppercase and lowercase produce same encoding.
    #[test]
    fn test_case_insensitive_encoding() {
        let upper = b"ACGTRYSWKMBDHVN";
        let lower = b"acgtryswkmbdhvn";
        let mixed = b"AcGtRySw";

        let enc_upper = encode_dna_prefer_simd(upper);
        let enc_lower = encode_dna_prefer_simd(lower);

        assert_eq!(enc_upper, enc_lower);

        let enc_mixed = encode_dna_prefer_simd(mixed);
        let enc_upper_part = encode_dna_prefer_simd(b"ACGTRYSW");
        assert_eq!(enc_mixed, enc_upper_part);
    }

    // ========================================================================
    // Invalid Character Tests
    // ========================================================================

    /// Tests that invalid characters encode as gaps.
    #[test]
    fn test_invalid_characters_become_gaps() {
        let sequence = b"ACXZGT";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        // X and Z are invalid, should become '-'
        assert_eq!(decoded, b"AC--GT");
    }

    /// Tests spaces and whitespace handling.
    #[test]
    fn test_whitespace_becomes_gaps() {
        let sequence = b"AC GT";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, b"AC-GT");
    }

    /// Tests numeric characters handling.
    #[test]
    fn test_numbers_become_gaps() {
        let sequence = b"A1C2G3T4";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, b"A-C-G-T-");
    }

    // ========================================================================
    // Edge Cases and Stress Tests
    // ========================================================================

    /// Tests large sequence with various codes.
    #[test]
    fn test_large_sequence() {
        let sequence: Vec<u8> = (0..1000).map(|i| b"ACGTRYSWKMBDHVN-"[i % 16]).collect();
        let encoded = encode_dna_prefer_simd(&sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests very large sequence with remainder.
    #[test]
    fn test_very_large_sequence_with_remainder() {
        let len = 10_007; // Prime number
        let sequence: Vec<u8> = (0..len).map(|i| b"ACGTN"[i % 5]).collect();
        let encoded = encode_dna_prefer_simd(&sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests partial decoding.
    #[test]
    fn test_decode_partial_length() {
        let sequence = b"ACGTACGT";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, 5);
        assert_eq!(decoded, b"ACGTA");
    }

    /// Tests all lengths from 1 to 100.
    #[test]
    fn test_lengths_1_to_100() {
        for len in 1..=100 {
            let sequence: Vec<u8> = (0..len).map(|i| b"ACGTN"[i % 5]).collect();
            let encoded = encode_dna_prefer_simd(&sequence);
            let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
            assert_eq!(decoded, sequence, "Failed for length {}", len);
        }
    }

    /// Tests scalar encoding directly.
    #[test]
    fn test_encode_scalar_directly() {
        let sequence = b"ACGTACGT";
        let mut output = vec![0u8; 4];
        encode_scalar(sequence, &mut output);
        assert_eq!(output[0], 0x12); // AC: (A=0x1<<4) | C=0x2
        assert_eq!(output[1], 0x84); // GT: (G=0x8<<4) | T=0x4
        assert_eq!(output[2], 0x12); // AC
        assert_eq!(output[3], 0x84); // GT
    }

    /// Tests scalar decoding directly.
    #[test]
    fn test_decode_scalar_directly() {
        let encoded = [0x12u8, 0x84]; // AC, GT
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
        // New encoding: 0=Gap, 1=A, 2=C, 3=M, 4=T, 5=W, 6=Y, 7=H, 8=G, 9=R, A=S, B=V, C=K, D=D, E=B, F=N

        // 0x12 = AC (A=0x1, C=0x2)
        // 0x84 = GT (G=0x8, T=0x4)
        let decoded = decode_dna_prefer_simd(&[0x12, 0x84], 4);
        assert_eq!(decoded, b"ACGT");

        // 0x96 = RY (R=0x9, Y=0x6)
        // 0xA5 = SW (S=0xA, W=0x5)
        let decoded = decode_dna_prefer_simd(&[0x96, 0xA5], 4);
        assert_eq!(decoded, b"RYSW");

        // 0xC3 = KM (K=0xC, M=0x3)
        // 0xED = BD (B=0xE, D=0xD)
        let decoded = decode_dna_prefer_simd(&[0xC3, 0xED], 4);
        assert_eq!(decoded, b"KMBD");

        // 0x7B = HV (H=0x7, V=0xB)
        // 0xF0 = N- (N=0xF, Gap=0x0)
        let decoded = decode_dna_prefer_simd(&[0x7B, 0xF0], 4);
        assert_eq!(decoded, b"HVN-");
    }

    /// Tests decoding each 4-bit value individually (0x0 to 0xF).
    #[test]
    fn test_decode_all_4bit_values() {
        // New encoding: 0=Gap, 1=A, 2=C, 3=M, 4=T, 5=W, 6=Y, 7=H, 8=G, 9=R, A=S, B=V, C=K, D=D, E=B, F=N
        let expected: [u8; 16] = [
            b'-', b'A', b'C', b'M', b'T', b'W', b'Y', b'H', b'G', b'R', b'S', b'V', b'K', b'D',
            b'B', b'N',
        ];

        for (value, &expected_char) in expected.iter().enumerate() {
            // Test in high nibble position
            let encoded_high = [(value as u8) << 4];
            let decoded = decode_dna_prefer_simd(&encoded_high, 1);
            assert_eq!(
                decoded[0], expected_char,
                "Value 0x{:X} in high nibble should decode to '{}'",
                value, expected_char as char
            );

            // Test in low nibble position (need length 2 to read both nibbles)
            let encoded_low = [value as u8];
            let decoded = decode_dna_prefer_simd(&encoded_low, 2);
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
        let encoded = encode_dna_prefer_simd(sequence);

        // Decode only first 8 characters
        let decoded = decode_dna_prefer_simd(&encoded, 8);
        assert_eq!(decoded, b"ACGTRYSW");

        // Decode only first 1 character
        let decoded = decode_dna_prefer_simd(&encoded, 1);
        assert_eq!(decoded, b"A");

        // Decode only first 3 characters (odd)
        let decoded = decode_dna_prefer_simd(&encoded, 3);
        assert_eq!(decoded, b"ACG");
    }

    /// Tests decoding empty input.
    #[test]
    fn test_decode_empty() {
        let decoded = decode_dna_prefer_simd(&[], 0);
        assert!(decoded.is_empty());
    }

    /// Tests decoding with zero length.
    #[test]
    fn test_decode_zero_length() {
        let encoded = [0x01, 0x23];
        let decoded = decode_dna_prefer_simd(&encoded, 0);
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

        let decoded = decode_dna_prefer_simd(&encoded, 32);
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
        // Use encoding values that correspond to the IUPAC sequence
        // New encoding: 0=Gap, 1=A, 2=C, 3=M, 4=T, 5=W, 6=Y, 7=H, 8=G, 9=R, A=S, B=V, C=K, D=D, E=B, F=N
        // Encode the sequence via roundtrip instead of hardcoding
        let sequence = b"ACGTRYSWKMBDHVN-";
        let encoded = encode_dna_prefer_simd(sequence);
        assert_eq!(encoded.len(), 8);
        let decoded = decode_dna_prefer_simd(&encoded, 16);
        assert_eq!(decoded, b"ACGTRYSWKMBDHVN-");
    }

    /// Tests decoding with remainder (not multiple of 16).
    #[test]
    fn test_decode_with_remainder() {
        // 20 nucleotides = 10 encoded bytes
        // SIMD handles first 16, scalar handles remaining 4
        let sequence = b"ACGTRYSWKMBDHVN-ACGT";
        let encoded = encode_dna_prefer_simd(sequence);
        assert_eq!(encoded.len(), 10);

        let decoded = decode_dna_prefer_simd(&encoded, 20);
        assert_eq!(decoded, sequence);
    }

    /// Tests decoding multiple SIMD blocks.
    #[test]
    fn test_decode_multiple_simd_blocks() {
        // 64 nucleotides = 32 encoded bytes = 4 SIMD blocks
        let pattern = b"ACGTRYSWKMBDHVN-";
        let sequence: Vec<u8> = pattern.iter().cycle().take(64).copied().collect();
        let encoded = encode_dna_prefer_simd(&sequence);
        assert_eq!(encoded.len(), 32);

        let decoded = decode_dna_prefer_simd(&encoded, 64);
        assert_eq!(decoded, sequence);
    }

    /// Tests decoding IUPAC codes specifically.
    #[test]
    fn test_decode_iupac_codes() {
        // New encoding: R=0x9, Y=0x6, S=0xA, W=0x5, K=0xC, M=0x3, B=0xE, D=0xD, H=0x7, V=0xB, N=0xF
        let iupac_only = [
            0x96u8, // RY (R=0x9, Y=0x6)
            0xA5,   // SW (S=0xA, W=0x5)
            0xC3,   // KM (K=0xC, M=0x3)
            0xED,   // BD (B=0xE, D=0xD)
            0x7B,   // HV (H=0x7, V=0xB)
            0xFF,   // NN (N=0xF, N=0xF)
        ];

        let decoded = decode_dna_prefer_simd(&iupac_only, 12);
        assert_eq!(decoded, b"RYSWKMBDHVNN");
    }

    /// Tests decoding gaps.
    #[test]
    fn test_decode_gaps() {
        // All gaps (Gap=0x0 in new encoding)
        let all_gaps = [0x00u8, 0x00, 0x00, 0x00];
        let decoded = decode_dna_prefer_simd(&all_gaps, 8);
        assert_eq!(decoded, b"--------");
    }

    /// Tests decode scalar fallback for small inputs.
    #[test]
    fn test_decode_scalar_small_inputs() {
        // Test sizes that use scalar path (< 16 nucleotides)
        for size in [2, 4, 6, 8, 10, 12, 14] {
            let sequence: Vec<u8> = (0..size).map(|i| b"ACGTN"[i % 5]).collect();
            let encoded = encode_dna_prefer_simd(&sequence);
            let decoded = decode_dna_prefer_simd(&encoded, size);
            assert_eq!(decoded, sequence, "Failed for size {}", size);
        }
    }

    /// Tests decode produces uppercase output for all cases.
    #[test]
    fn test_decode_always_uppercase() {
        // Decode should always produce uppercase regardless of original encoding
        let sequence = b"acgtryswkmbdhvn";
        let encoded = encode_dna_prefer_simd(sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());

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
        let encoded = encode_dna_prefer_simd(&sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests 17 nucleotides (just over SIMD boundary).
    #[test]
    fn test_simd_boundary_17_nucleotides() {
        let sequence: Vec<u8> = (0..17).map(|i| b"ACGTN"[i % 5]).collect();
        let encoded = encode_dna_prefer_simd(&sequence);
        let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
        assert_eq!(decoded, sequence);
    }

    /// Tests exact multiples of 16.
    #[test]
    fn test_simd_boundary_exact_multiples() {
        for len in [16, 32, 48, 64, 128, 256] {
            let sequence: Vec<u8> = (0..len).map(|i| b"ACGTN"[i % 5]).collect();
            let encoded = encode_dna_prefer_simd(&sequence);
            let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
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
                    let encoded = encode_dna_prefer_simd(&seq);
                    let decoded = decode_dna_prefer_simd(&encoded, seq.len());
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
                let encoded = encode_dna_prefer_simd(&seq);
                (encoded, len)
            })
            .collect();

        let handles: Vec<_> = test_data
            .into_iter()
            .map(|(encoded, len)| {
                thread::spawn(move || {
                    let decoded = decode_dna_prefer_simd(&encoded, len);
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
                    let encoded = encode_dna_prefer_simd(&seq);
                    let decoded = decode_dna_prefer_simd(&encoded, seq.len());
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
                        let encoded = encode_dna_prefer_simd(&seq);
                        let decoded = decode_dna_prefer_simd(&encoded, seq.len());
                        assert_eq!(decoded, seq, "Failed for length {}", len);
                    })
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    // ========================================================================
    // Complement and Reverse Complement Tests
    // ========================================================================

    /// Tests bit rotation complement for standard bases.
    #[test]
    fn test_complement_4bit_standard_bases() {
        // A (0x1) <-> T (0x4)
        assert_eq!(complement_4bit(encoding::A), encoding::T);
        assert_eq!(complement_4bit(encoding::T), encoding::A);

        // C (0x2) <-> G (0x8)
        assert_eq!(complement_4bit(encoding::C), encoding::G);
        assert_eq!(complement_4bit(encoding::G), encoding::C);
    }

    /// Tests bit rotation complement for ambiguity codes.
    #[test]
    fn test_complement_4bit_ambiguity_codes() {
        // R (0x9) <-> Y (0x6)
        assert_eq!(complement_4bit(encoding::R), encoding::Y);
        assert_eq!(complement_4bit(encoding::Y), encoding::R);

        // K (0xC) <-> M (0x3)
        assert_eq!(complement_4bit(encoding::K), encoding::M);
        assert_eq!(complement_4bit(encoding::M), encoding::K);

        // D (0xD) <-> H (0x7)
        assert_eq!(complement_4bit(encoding::D), encoding::H);
        assert_eq!(complement_4bit(encoding::H), encoding::D);

        // B (0xE) <-> V (0xB)
        assert_eq!(complement_4bit(encoding::B), encoding::V);
        assert_eq!(complement_4bit(encoding::V), encoding::B);
    }

    /// Tests self-complementary codes.
    #[test]
    fn test_complement_4bit_self_complementary() {
        // S (0xA), W (0x5), N (0xF), Gap (0x0) are self-complementary
        assert_eq!(complement_4bit(encoding::S), encoding::S);
        assert_eq!(complement_4bit(encoding::W), encoding::W);
        assert_eq!(complement_4bit(encoding::N), encoding::N);
        assert_eq!(complement_4bit(encoding::GAP), encoding::GAP);
    }

    /// Tests that complement is its own inverse for all values.
    #[test]
    fn test_complement_4bit_involution() {
        for i in 0..16u8 {
            let comp = complement_4bit(i);
            let double_comp = complement_4bit(comp);
            assert_eq!(
                double_comp, i,
                "Double complement of 0x{:X} should be itself",
                i
            );
        }
    }

    /// Tests reverse complement of a palindromic sequence.
    #[test]
    fn test_reverse_complement_palindrome() {
        // ACGT is its own reverse complement
        let rc = reverse_complement(b"ACGT");
        assert_eq!(rc, b"ACGT");

        // ATAT is its own reverse complement
        let rc = reverse_complement(b"ATAT");
        assert_eq!(rc, b"ATAT");
    }

    /// Tests reverse complement of standard sequences.
    #[test]
    fn test_reverse_complement_standard() {
        // AACG -> reverse: GCAA -> complement: CGTT
        let rc = reverse_complement(b"AACG");
        assert_eq!(rc, b"CGTT");

        // A -> T
        let rc = reverse_complement(b"A");
        assert_eq!(rc, b"T");

        // AC -> reverse: CA -> complement: GT
        let rc = reverse_complement(b"AC");
        assert_eq!(rc, b"GT");

        // AAA -> reverse: AAA -> complement: TTT
        let rc = reverse_complement(b"AAA");
        assert_eq!(rc, b"TTT");

        // GATTACA -> reverse: ACATTAG -> complement: TGTAATC
        let rc = reverse_complement(b"GATTACA");
        assert_eq!(rc, b"TGTAATC");
    }

    /// Tests reverse complement with IUPAC ambiguity codes.
    #[test]
    fn test_reverse_complement_iupac() {
        // R (A|G) <-> Y (C|T)
        let rc = reverse_complement(b"R");
        assert_eq!(rc, b"Y");

        // S is self-complementary
        let rc = reverse_complement(b"S");
        assert_eq!(rc, b"S");

        // W is self-complementary
        let rc = reverse_complement(b"W");
        assert_eq!(rc, b"W");

        // N is self-complementary
        let rc = reverse_complement(b"N");
        assert_eq!(rc, b"N");

        // Mixed: ACRYGT -> reverse: TGYRAC -> complement: ACRMGT... let's compute properly
        // Actually: ACRYGT, reversed = TGYRCA, complement each:
        // T->A, G->C, Y->R, R->Y, C->G, A->T => ACRYGT
        // Wait, that's the same? Let me recalculate:
        // Original: A C R Y G T
        // Reversed: T G Y R C A
        // Complement: A C R Y G T
        // So ACRYGT is palindromic!
        let rc = reverse_complement(b"ACRYGT");
        assert_eq!(rc, b"ACRYGT");
    }

    /// Tests reverse complement of empty sequence.
    #[test]
    fn test_reverse_complement_empty() {
        let rc = reverse_complement(b"");
        assert!(rc.is_empty());
    }

    /// Tests reverse complement of long sequences (SIMD path).
    #[test]
    fn test_reverse_complement_long_sequence() {
        // 64 nucleotides - should use SIMD
        let sequence: Vec<u8> = b"ACGT".iter().cycle().take(64).copied().collect();
        let rc = reverse_complement(&sequence);
        assert_eq!(rc.len(), 64);

        // ACGT repeated is palindromic
        assert_eq!(rc, sequence);

        // Non-palindromic long sequence
        let sequence: Vec<u8> = b"AACG".iter().cycle().take(64).copied().collect();
        let rc = reverse_complement(&sequence);
        let rc_rc = reverse_complement(&rc);
        // Reverse complement of reverse complement should be original
        assert_eq!(rc_rc, sequence);
    }

    /// Tests that reverse_complement_encoded works correctly.
    #[test]
    fn test_reverse_complement_encoded() {
        let sequence = b"GATTACA";
        let encoded = encode_dna_prefer_simd(sequence);
        let rc_encoded = reverse_complement_encoded(&encoded, sequence.len());
        let rc_decoded = decode_dna_prefer_simd(&rc_encoded, sequence.len());
        assert_eq!(rc_decoded, b"TGTAATC");
    }

    /// Tests complement of packed byte.
    #[test]
    fn test_complement_packed_byte() {
        // AC (0x12) -> complement each nibble -> TG (0x48)
        let packed = 0x12u8; // A=0x1 high, C=0x2 low
        let comp = complement_packed_byte(packed);
        assert_eq!(comp, 0x48); // T=0x4 high, G=0x8 low
    }

    /// Tests thread safety of reverse complement.
    #[test]
    fn test_reverse_complement_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let sequence = Arc::new(b"GATTACA".to_vec());

        let handles: Vec<_> = (0..50)
            .map(|_| {
                let seq = Arc::clone(&sequence);
                thread::spawn(move || {
                    let rc = reverse_complement(&seq);
                    assert_eq!(rc, b"TGTAATC");
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }
}
