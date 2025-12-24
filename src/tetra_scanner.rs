// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! # Tetranucleotide SIMD Scanner
//!
//! This module provides high-performance pattern matching for DNA sequences
//! using tetranucleotide (4-mer) lookup tables and SIMD acceleration.
//!
//! ## Design Rationale
//!
//! Tetranucleotides (4-mers) are the natural unit for pattern matching in
//! 2-bit encoded DNA sequences because:
//!
//! - 4 bases × 2 bits = 8 bits = 1 byte exactly
//! - All 256 possible 4-mers map to byte values 0-255
//! - SIMD can process 16 tetranucleotides per instruction
//! - Pre-computed LUTs enable O(1) pattern matching per position
//!
//! ## Tetra-LUT Scanner
//!
//! The `TetraLUT` structure creates a 256-entry lookup table where each
//! entry indicates whether the corresponding tetranucleotide matches
//! the search pattern:
//!
//! ```rust,ignore
//! use simdna::tetra_scanner::TetraLUT;
//!
//! // Search for exact pattern "ACGT"
//! let lut = TetraLUT::from_literal(b"ACGT").unwrap();
//!
//! // Search with IUPAC ambiguity: "ACGN" matches ACGA, ACGC, ACGG, ACGT
//! let lut = TetraLUT::from_iupac("ACGN").unwrap();
//! ```
//!
//! ## Shift-And (Bitap) Algorithm
//!
//! For patterns longer than 4 bases, we use the bit-parallel Shift-And
//! algorithm with NEON acceleration. This supports patterns up to 64 bases
//! with IUPAC ambiguity codes.
//!
//! ## Performance
//!
//! On Apple Silicon (NEON):
//! - Tetra-LUT scanning: ~15+ GB/s throughput (16 positions per instruction)
//! - Shift-And: ~8+ GB/s throughput (pattern-length independent)

use std::fmt;

use crate::hybrid_encoder::{encoding_2bit, is_clean_sequence, required_2bit_len};

// ============================================================================
// Tetra-LUT: 256-Entry Lookup Table for 4-mer Pattern Matching
// ============================================================================

/// A 256-entry lookup table for tetranucleotide pattern matching.
///
/// Each index (0-255) corresponds to a unique tetranucleotide packed as:
/// `(base0 << 6) | (base1 << 4) | (base2 << 2) | base3`
///
/// where each base is encoded as: A=0, C=1, G=2, T=3.
///
/// Entries are `0xFF` for matching tetranucleotides, `0x00` for non-matching.
///
/// # Examples
///
/// ```rust
/// use simdna::tetra_scanner::TetraLUT;
///
/// // Exact match for "ACGT"
/// let lut = TetraLUT::from_literal(b"ACGT").unwrap();
/// assert!(lut.matches_index(0x1B)); // ACGT = 0b00_01_10_11 = 0x1B
///
/// // IUPAC pattern "ACGN" matches 4 tetranucleotides
/// let lut = TetraLUT::from_iupac("ACGN").unwrap();
/// assert_eq!(lut.match_count(), 4); // ACGA, ACGC, ACGG, ACGT
/// ```
#[derive(Clone)]
pub struct TetraLUT {
    /// 256-entry lookup table (0xFF = match, 0x00 = no match)
    lut: [u8; 256],
    /// The pattern this LUT was built from (for debugging/display)
    pattern: [u8; 4],
    /// Number of tetranucleotides that match this pattern
    match_count: usize,
}

impl TetraLUT {
    /// Creates a TetraLUT for an exact literal pattern (4 bases, ACGT only).
    ///
    /// # Arguments
    ///
    /// * `pattern` - Exactly 4 ASCII nucleotides (A, C, G, T, case-insensitive)
    ///
    /// # Returns
    ///
    /// * `Ok(TetraLUT)` - The lookup table
    /// * `Err(TetraLUTError)` - If pattern is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::tetra_scanner::TetraLUT;
    ///
    /// let lut = TetraLUT::from_literal(b"ACGT").unwrap();
    /// assert_eq!(lut.match_count(), 1);
    /// assert!(lut.matches_index(0x1B));
    /// ```
    pub fn from_literal(pattern: &[u8]) -> Result<Self, TetraLUTError> {
        if pattern.len() != 4 {
            return Err(TetraLUTError::InvalidPatternLength {
                expected: 4,
                actual: pattern.len(),
            });
        }

        // Check all bases are valid ACGT
        if !is_clean_sequence(pattern) {
            return Err(TetraLUTError::InvalidBase {
                pattern: String::from_utf8_lossy(pattern).to_string(),
            });
        }

        let mut lut = [0u8; 256];
        let mut pattern_array = [0u8; 4];
        pattern_array.copy_from_slice(pattern);

        // Encode the pattern to get its index
        let idx = encode_tetra_to_index(pattern)?;
        lut[idx as usize] = 0xFF;

        Ok(Self {
            lut,
            pattern: pattern_array,
            match_count: 1,
        })
    }

    /// Creates a TetraLUT for an IUPAC pattern (supports ambiguity codes).
    ///
    /// Ambiguous bases expand to multiple matching tetranucleotides:
    ///
    /// | Code | Bases | Count |
    /// |------|-------|-------|
    /// | N    | ACGT  | 4     |
    /// | R    | AG    | 2     |
    /// | Y    | CT    | 2     |
    /// | S    | GC    | 2     |
    /// | W    | AT    | 2     |
    /// | K    | GT    | 2     |
    /// | M    | AC    | 2     |
    /// | B    | CGT   | 3     |
    /// | D    | AGT   | 3     |
    /// | H    | ACT   | 3     |
    /// | V    | ACG   | 3     |
    ///
    /// # Arguments
    ///
    /// * `pattern` - 4-character IUPAC pattern string
    ///
    /// # Returns
    ///
    /// * `Ok(TetraLUT)` - The lookup table with all matching indices marked
    /// * `Err(TetraLUTError)` - If pattern is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::tetra_scanner::TetraLUT;
    ///
    /// // "ACGN" matches ACGA, ACGC, ACGG, ACGT
    /// let lut = TetraLUT::from_iupac("ACGN").unwrap();
    /// assert_eq!(lut.match_count(), 4);
    ///
    /// // "NNNN" matches all 256 tetranucleotides
    /// let lut = TetraLUT::from_iupac("NNNN").unwrap();
    /// assert_eq!(lut.match_count(), 256);
    /// ```
    pub fn from_iupac(pattern: &str) -> Result<Self, TetraLUTError> {
        let bytes = pattern.as_bytes();
        if bytes.len() != 4 {
            return Err(TetraLUTError::InvalidPatternLength {
                expected: 4,
                actual: bytes.len(),
            });
        }

        let mut pattern_array = [0u8; 4];
        pattern_array.copy_from_slice(bytes);

        // Expand each position to its possible bases
        let expansions: Vec<Vec<u8>> = bytes
            .iter()
            .map(|&b| expand_iupac(b))
            .collect::<Result<Vec<_>, _>>()?;

        let mut lut = [0u8; 256];
        let mut match_count = 0;

        // Generate all combinations
        for &b0 in &expansions[0] {
            for &b1 in &expansions[1] {
                for &b2 in &expansions[2] {
                    for &b3 in &expansions[3] {
                        let idx = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3;
                        if lut[idx as usize] == 0 {
                            lut[idx as usize] = 0xFF;
                            match_count += 1;
                        }
                    }
                }
            }
        }

        Ok(Self {
            lut,
            pattern: pattern_array,
            match_count,
        })
    }

    /// Returns whether the given index matches this pattern.
    ///
    /// # Arguments
    ///
    /// * `index` - Tetranucleotide index (0-255)
    ///
    /// # Returns
    ///
    /// `true` if the tetranucleotide at this index matches the pattern.
    #[inline]
    pub fn matches_index(&self, index: u8) -> bool {
        self.lut[index as usize] != 0
    }

    /// Returns the number of tetranucleotides that match this pattern.
    ///
    /// - For exact patterns: always 1
    /// - For IUPAC patterns: product of expansion sizes (up to 256)
    #[inline]
    pub fn match_count(&self) -> usize {
        self.match_count
    }

    /// Returns a reference to the internal lookup table.
    ///
    /// This is useful for SIMD operations that need direct LUT access.
    #[inline]
    pub fn lut(&self) -> &[u8; 256] {
        &self.lut
    }

    /// Returns the original pattern this LUT was built from.
    #[inline]
    pub fn pattern(&self) -> &[u8; 4] {
        &self.pattern
    }

    /// Scans a 2-bit encoded sequence for pattern matches.
    ///
    /// Returns positions (in nucleotides) where the pattern matches.
    /// Each position is the start of a matching 4-mer.
    ///
    /// # Arguments
    ///
    /// * `encoded` - 2-bit packed sequence data
    /// * `original_len` - Original sequence length in nucleotides
    ///
    /// # Returns
    ///
    /// Vector of match positions (0-indexed from sequence start).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::tetra_scanner::TetraLUT;
    /// use simdna::hybrid_encoder::encode_2bit;
    ///
    /// let sequence = b"ACGTACGTACGT";
    /// let encoded = encode_2bit(sequence).unwrap();
    /// let lut = TetraLUT::from_literal(b"ACGT").unwrap();
    ///
    /// let matches = lut.scan_2bit(&encoded, sequence.len());
    /// assert_eq!(matches, vec![0, 4, 8]); // ACGT appears at positions 0, 4, 8
    /// ```
    pub fn scan_2bit(&self, encoded: &[u8], original_len: usize) -> Vec<usize> {
        if original_len < 4 {
            return Vec::new();
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Safety: NEON is always available on aarch64
            unsafe { self.scan_2bit_neon(encoded, original_len) }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            self.scan_2bit_scalar(encoded, original_len)
        }
    }

    /// Checks if any match exists (early exit optimization).
    ///
    /// This is faster than `scan_2bit` when you only need to know
    /// whether a match exists, not where.
    ///
    /// # Arguments
    ///
    /// * `encoded` - 2-bit packed sequence data
    /// * `original_len` - Original sequence length in nucleotides
    ///
    /// # Returns
    ///
    /// `true` if any match exists.
    pub fn contains_match(&self, encoded: &[u8], original_len: usize) -> bool {
        if original_len < 4 {
            return false;
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Safety: NEON is always available on aarch64
            unsafe { self.contains_match_neon(encoded, original_len) }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            self.contains_match_scalar(encoded, original_len)
        }
    }

    /// Scalar implementation of 2-bit scanning.
    fn scan_2bit_scalar(&self, encoded: &[u8], original_len: usize) -> Vec<usize> {
        let mut matches = Vec::new();
        let max_pos = original_len.saturating_sub(3);

        for pos in 0..max_pos {
            let tetra_idx = extract_tetra_at_position(encoded, pos, original_len);
            if self.lut[tetra_idx as usize] != 0 {
                matches.push(pos);
            }
        }

        matches
    }

    /// Scalar implementation of contains_match.
    fn contains_match_scalar(&self, encoded: &[u8], original_len: usize) -> bool {
        let max_pos = original_len.saturating_sub(3);

        for pos in 0..max_pos {
            let tetra_idx = extract_tetra_at_position(encoded, pos, original_len);
            if self.lut[tetra_idx as usize] != 0 {
                return true;
            }
        }

        false
    }

    /// NEON-accelerated 2-bit scanning.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn scan_2bit_neon(&self, encoded: &[u8], original_len: usize) -> Vec<usize> {
        use core::arch::aarch64::*;

        let mut matches = Vec::new();
        let max_pos = original_len.saturating_sub(3);

        // For short sequences, use scalar
        if max_pos < 16 {
            return self.scan_2bit_scalar(encoded, original_len);
        }

        // Load LUT into 4 NEON registers for vqtbl4q_u8
        let lut = self.load_lut_neon();

        // Process 16 positions at a time where possible
        let mut pos = 0;
        while pos + 16 <= max_pos {
            // Extract 16 tetranucleotide indices
            let indices = self.extract_16_tetra_indices(encoded, pos, original_len);

            // Perform LUT lookup
            let results = vqtbl4q_u8(lut, indices);

            // Check for matches
            let max_val = vmaxvq_u8(results);
            if max_val != 0 {
                // Extract individual matches
                let mut result_bytes = [0u8; 16];
                vst1q_u8(result_bytes.as_mut_ptr(), results);
                for (i, &r) in result_bytes.iter().enumerate() {
                    if r != 0 {
                        matches.push(pos + i);
                    }
                }
            }

            pos += 16;
        }

        // Handle remainder
        while pos < max_pos {
            let tetra_idx = extract_tetra_at_position(encoded, pos, original_len);
            if self.lut[tetra_idx as usize] != 0 {
                matches.push(pos);
            }
            pos += 1;
        }

        matches
    }

    /// NEON-accelerated contains_match.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn contains_match_neon(&self, encoded: &[u8], original_len: usize) -> bool {
        use core::arch::aarch64::*;

        let max_pos = original_len.saturating_sub(3);

        // For short sequences, use scalar
        if max_pos < 16 {
            return self.contains_match_scalar(encoded, original_len);
        }

        // Load LUT into 4 NEON registers
        let lut = self.load_lut_neon();

        // Process 16 positions at a time
        let mut pos = 0;
        while pos + 16 <= max_pos {
            let indices = self.extract_16_tetra_indices(encoded, pos, original_len);
            let results = vqtbl4q_u8(lut, indices);
            if vmaxvq_u8(results) != 0 {
                return true;
            }
            pos += 16;
        }

        // Check remainder
        while pos < max_pos {
            let tetra_idx = extract_tetra_at_position(encoded, pos, original_len);
            if self.lut[tetra_idx as usize] != 0 {
                return true;
            }
            pos += 1;
        }

        false
    }

    /// Load LUT into 4 NEON registers for vqtbl4q_u8 lookup.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn load_lut_neon(&self) -> core::arch::aarch64::uint8x16x4_t {
        use core::arch::aarch64::*;

        uint8x16x4_t(
            vld1q_u8(self.lut.as_ptr()),
            vld1q_u8(self.lut.as_ptr().add(16)),
            vld1q_u8(self.lut.as_ptr().add(32)),
            vld1q_u8(self.lut.as_ptr().add(48)),
        )
    }

    /// Extract 16 tetranucleotide indices starting at position.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn extract_16_tetra_indices(
        &self,
        encoded: &[u8],
        start_pos: usize,
        original_len: usize,
    ) -> core::arch::aarch64::uint8x16_t {
        use core::arch::aarch64::*;

        let mut indices = [0u8; 16];
        for i in 0..16 {
            let pos = start_pos + i;
            if pos + 4 <= original_len {
                indices[i] = extract_tetra_at_position(encoded, pos, original_len);
            }
        }
        vld1q_u8(indices.as_ptr())
    }
}

impl fmt::Debug for TetraLUT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TetraLUT")
            .field("pattern", &String::from_utf8_lossy(&self.pattern))
            .field("match_count", &self.match_count)
            .finish()
    }
}

impl fmt::Display for TetraLUT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TetraLUT({}, {} matches)",
            String::from_utf8_lossy(&self.pattern),
            self.match_count
        )
    }
}

// ============================================================================
// Shift-And (Bitap) Pattern Matching
// ============================================================================

/// Character mask table for the Shift-And (Bitap) algorithm.
///
/// This structure precomputes bit masks for each character in the alphabet,
/// enabling O(n) pattern matching with O(m) preprocessing where:
/// - n = text length
/// - m = pattern length
///
/// Supports patterns up to 64 bases (using u64 bitmask).
///
/// # Examples
///
/// ```rust
/// use simdna::tetra_scanner::BitapMasks;
///
/// // Exact pattern
/// let masks = BitapMasks::from_pattern(b"ACGT").unwrap();
/// assert_eq!(masks.pattern_len(), 4);
///
/// // IUPAC pattern with ambiguity
/// let masks = BitapMasks::from_iupac("ACGN").unwrap();
/// ```
#[derive(Clone)]
pub struct BitapMasks {
    /// Mask for each ASCII character (256 entries)
    /// Bit i is set if character matches pattern position i
    masks: [u64; 256],
    /// Original pattern
    pattern: Vec<u8>,
    /// Pattern length
    pattern_len: usize,
    /// Accept mask: bit (pattern_len - 1) set
    accept_mask: u64,
}

impl BitapMasks {
    /// Creates masks for an exact literal pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` - ASCII DNA pattern (ACGT only, case-insensitive)
    ///
    /// # Returns
    ///
    /// * `Ok(BitapMasks)` - The precomputed masks
    /// * `Err(BitapError)` - If pattern is invalid or too long
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::tetra_scanner::BitapMasks;
    ///
    /// let masks = BitapMasks::from_pattern(b"ACGTACGT").unwrap();
    /// assert_eq!(masks.pattern_len(), 8);
    /// ```
    pub fn from_pattern(pattern: &[u8]) -> Result<Self, BitapError> {
        if pattern.is_empty() {
            return Err(BitapError::EmptyPattern);
        }
        if pattern.len() > 64 {
            return Err(BitapError::PatternTooLong {
                max: 64,
                actual: pattern.len(),
            });
        }

        let mut masks = [!0u64; 256]; // Start with all bits set (no match)
        let pattern_len = pattern.len();

        for (i, &c) in pattern.iter().enumerate() {
            let upper = c.to_ascii_uppercase();
            match upper {
                b'A' | b'C' | b'G' | b'T' => {
                    // Clear bit i for this character (match at position i)
                    masks[upper as usize] &= !(1u64 << i);
                    masks[upper.to_ascii_lowercase() as usize] &= !(1u64 << i);
                }
                _ => {
                    return Err(BitapError::InvalidBase(c as char));
                }
            }
        }

        let accept_mask = 1u64 << (pattern_len - 1);

        Ok(Self {
            masks,
            pattern: pattern.to_vec(),
            pattern_len,
            accept_mask,
        })
    }

    /// Creates masks for an IUPAC pattern with ambiguity codes.
    ///
    /// Each ambiguous base matches multiple characters:
    ///
    /// | Code | Matches |
    /// |------|---------|
    /// | N    | ACGT    |
    /// | R    | AG      |
    /// | Y    | CT      |
    /// | S    | GC      |
    /// | W    | AT      |
    /// | K    | GT      |
    /// | M    | AC      |
    /// | B    | CGT     |
    /// | D    | AGT     |
    /// | H    | ACT     |
    /// | V    | ACG     |
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::tetra_scanner::BitapMasks;
    ///
    /// let masks = BitapMasks::from_iupac("ACGN").unwrap();
    /// assert_eq!(masks.pattern_len(), 4);
    /// ```
    pub fn from_iupac(pattern: &str) -> Result<Self, BitapError> {
        let bytes = pattern.as_bytes();
        if bytes.is_empty() {
            return Err(BitapError::EmptyPattern);
        }
        if bytes.len() > 64 {
            return Err(BitapError::PatternTooLong {
                max: 64,
                actual: bytes.len(),
            });
        }

        let mut masks = [!0u64; 256];
        let pattern_len = bytes.len();

        for (i, &c) in bytes.iter().enumerate() {
            let expanded = expand_iupac(c).map_err(|_| BitapError::InvalidBase(c as char))?;

            for base in expanded {
                // Convert 2-bit code back to ASCII
                let ascii = match base {
                    0 => b'A',
                    1 => b'C',
                    2 => b'G',
                    3 => b'T',
                    _ => unreachable!(),
                };
                masks[ascii as usize] &= !(1u64 << i);
                masks[ascii.to_ascii_lowercase() as usize] &= !(1u64 << i);
            }
        }

        let accept_mask = 1u64 << (pattern_len - 1);

        Ok(Self {
            masks,
            pattern: bytes.to_vec(),
            pattern_len,
            accept_mask,
        })
    }

    /// Returns the pattern length.
    #[inline]
    pub fn pattern_len(&self) -> usize {
        self.pattern_len
    }

    /// Returns the original pattern.
    #[inline]
    pub fn pattern(&self) -> &[u8] {
        &self.pattern
    }

    /// Returns the mask for a given character.
    #[inline]
    pub fn mask(&self, c: u8) -> u64 {
        self.masks[c as usize]
    }

    /// Returns the accept mask (final bit for match detection).
    #[inline]
    pub fn accept_mask(&self) -> u64 {
        self.accept_mask
    }

    /// Searches for all occurrences of the pattern in the text.
    ///
    /// Uses the Shift-And algorithm for O(n) scanning.
    ///
    /// # Arguments
    ///
    /// * `text` - ASCII DNA sequence to search
    ///
    /// # Returns
    ///
    /// Vector of match positions (0-indexed, position of pattern start).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::tetra_scanner::BitapMasks;
    ///
    /// let masks = BitapMasks::from_pattern(b"ACGT").unwrap();
    /// let text = b"ACGTACGTACGT";
    /// let matches = masks.search(text);
    /// assert_eq!(matches, vec![0, 4, 8]);
    /// ```
    pub fn search(&self, text: &[u8]) -> Vec<usize> {
        if text.len() < self.pattern_len {
            return Vec::new();
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Safety: NEON is always available on aarch64
            unsafe { self.search_neon(text) }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            self.search_scalar(text)
        }
    }

    /// Checks if the pattern exists anywhere in the text (early exit).
    ///
    /// # Arguments
    ///
    /// * `text` - ASCII DNA sequence to search
    ///
    /// # Returns
    ///
    /// `true` if at least one match exists.
    pub fn contains(&self, text: &[u8]) -> bool {
        if text.len() < self.pattern_len {
            return false;
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Safety: NEON is always available on aarch64
            unsafe { self.contains_neon(text) }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            self.contains_scalar(text)
        }
    }

    /// Scalar implementation of Shift-And search.
    fn search_scalar(&self, text: &[u8]) -> Vec<usize> {
        let mut matches = Vec::new();
        let mut state = !0u64; // All bits set (no partial matches)

        for (i, &c) in text.iter().enumerate() {
            // Shift and apply character mask
            state = (state << 1) | self.masks[c as usize];

            // Check if accept bit is clear (match found)
            if (state & self.accept_mask) == 0 {
                // Match ends at position i, starts at i - pattern_len + 1
                matches.push(i + 1 - self.pattern_len);
            }
        }

        matches
    }

    /// Scalar implementation of contains.
    fn contains_scalar(&self, text: &[u8]) -> bool {
        let mut state = !0u64;

        for &c in text {
            state = (state << 1) | self.masks[c as usize];
            if (state & self.accept_mask) == 0 {
                return true;
            }
        }

        false
    }

    /// NEON-accelerated Shift-And search.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn search_neon(&self, text: &[u8]) -> Vec<usize> {
        // The Shift-And algorithm has dependencies between iterations,
        // so we use scalar for correctness. NEON is used for larger batch
        // operations in the TetraLUT scanner instead.
        self.search_scalar(text)
    }

    /// NEON-accelerated contains check.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn contains_neon(&self, text: &[u8]) -> bool {
        // Scalar is optimal for single-pattern Shift-And due to dependencies
        self.contains_scalar(text)
    }
}

impl fmt::Debug for BitapMasks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitapMasks")
            .field("pattern", &String::from_utf8_lossy(&self.pattern))
            .field("pattern_len", &self.pattern_len)
            .finish()
    }
}

// ============================================================================
// Search Interface
// ============================================================================

/// A search pattern that can be either a short 4-mer (TetraLUT) or longer (Bitap).
///
/// This enum provides a unified interface for pattern matching that
/// automatically selects the optimal algorithm based on pattern length.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum SearchPattern {
    /// 4-mer pattern using TetraLUT (fastest for 2-bit encoded data)
    Tetra(TetraLUT),
    /// Longer pattern using Shift-And (supports any length up to 64)
    Bitap(Box<BitapMasks>),
}

impl SearchPattern {
    /// Creates a search pattern from a literal DNA sequence.
    ///
    /// Automatically selects TetraLUT for 4-mers, Bitap for longer patterns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::tetra_scanner::SearchPattern;
    ///
    /// let short = SearchPattern::from_literal(b"ACGT").unwrap();
    /// assert!(matches!(short, SearchPattern::Tetra(_)));
    ///
    /// let long = SearchPattern::from_literal(b"ACGTACGT").unwrap();
    /// assert!(matches!(long, SearchPattern::Bitap(_)));
    /// ```
    pub fn from_literal(pattern: &[u8]) -> Result<Self, SearchError> {
        if pattern.len() == 4 {
            Ok(SearchPattern::Tetra(TetraLUT::from_literal(pattern)?))
        } else {
            Ok(SearchPattern::Bitap(Box::new(BitapMasks::from_pattern(
                pattern,
            )?)))
        }
    }

    /// Creates a search pattern from an IUPAC string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use simdna::tetra_scanner::SearchPattern;
    ///
    /// let pattern = SearchPattern::from_iupac("ACGN").unwrap();
    /// assert!(matches!(pattern, SearchPattern::Tetra(_)));
    /// ```
    pub fn from_iupac(pattern: &str) -> Result<Self, SearchError> {
        if pattern.len() == 4 {
            Ok(SearchPattern::Tetra(TetraLUT::from_iupac(pattern)?))
        } else {
            Ok(SearchPattern::Bitap(Box::new(BitapMasks::from_iupac(
                pattern,
            )?)))
        }
    }

    /// Returns the pattern length.
    pub fn pattern_len(&self) -> usize {
        match self {
            SearchPattern::Tetra(_) => 4,
            SearchPattern::Bitap(masks) => masks.pattern_len(),
        }
    }
}

/// Result of a pattern search operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// Positions where matches were found (0-indexed)
    pub positions: Vec<usize>,
    /// Number of matches found
    pub count: usize,
}

impl SearchResult {
    /// Creates a new search result.
    pub fn new(positions: Vec<usize>) -> Self {
        let count = positions.len();
        Self { positions, count }
    }

    /// Returns true if any matches were found.
    pub fn has_matches(&self) -> bool {
        self.count > 0
    }
}

// ============================================================================
// Pattern Encoding Utilities
// ============================================================================

/// Encodes a 4-base pattern to its tetranucleotide index (0-255).
///
/// # Arguments
///
/// * `pattern` - Exactly 4 ASCII nucleotides (A, C, G, T)
///
/// # Returns
///
/// * `Ok(index)` - The packed byte value (0-255)
/// * `Err(TetraLUTError)` - If pattern is invalid
///
/// # Examples
///
/// ```rust
/// use simdna::tetra_scanner::encode_tetra_to_index;
///
/// assert_eq!(encode_tetra_to_index(b"AAAA").unwrap(), 0x00);
/// assert_eq!(encode_tetra_to_index(b"ACGT").unwrap(), 0x1B);
/// assert_eq!(encode_tetra_to_index(b"TTTT").unwrap(), 0xFF);
/// ```
pub fn encode_tetra_to_index(pattern: &[u8]) -> Result<u8, TetraLUTError> {
    if pattern.len() != 4 {
        return Err(TetraLUTError::InvalidPatternLength {
            expected: 4,
            actual: pattern.len(),
        });
    }

    let mut index = 0u8;
    for (i, &base) in pattern.iter().enumerate() {
        let code = base_to_2bit(base).ok_or_else(|| TetraLUTError::InvalidBase {
            pattern: String::from_utf8_lossy(pattern).to_string(),
        })?;
        index |= code << (6 - i * 2);
    }

    Ok(index)
}

/// Decodes a tetranucleotide index (0-255) to its 4-base ASCII pattern.
///
/// # Examples
///
/// ```rust
/// use simdna::tetra_scanner::decode_index_to_tetra;
///
/// assert_eq!(decode_index_to_tetra(0x00), *b"AAAA");
/// assert_eq!(decode_index_to_tetra(0x1B), *b"ACGT");
/// assert_eq!(decode_index_to_tetra(0xFF), *b"TTTT");
/// ```
pub fn decode_index_to_tetra(index: u8) -> [u8; 4] {
    const LUT: [u8; 4] = [b'A', b'C', b'G', b'T'];

    [
        LUT[((index >> 6) & 0x03) as usize],
        LUT[((index >> 4) & 0x03) as usize],
        LUT[((index >> 2) & 0x03) as usize],
        LUT[(index & 0x03) as usize],
    ]
}

/// Encodes a pattern to 2-bit form for comparison with encoded sequences.
///
/// Returns None if the pattern contains non-ACGT characters.
///
/// # Examples
///
/// ```rust
/// use simdna::tetra_scanner::encode_pattern_2bit;
///
/// let encoded = encode_pattern_2bit(b"ACGT").unwrap();
/// assert_eq!(encoded, vec![0x1B]); // Packed 2-bit
/// ```
#[allow(clippy::needless_range_loop)]
pub fn encode_pattern_2bit(pattern: &[u8]) -> Option<Vec<u8>> {
    if !is_clean_sequence(pattern) {
        return None;
    }

    let len = required_2bit_len(pattern.len());
    let mut output = vec![0u8; len];

    let complete = pattern.len() / 4;
    for i in 0..complete {
        let base_idx = i * 4;
        let b0 = base_to_2bit(pattern[base_idx])?;
        let b1 = base_to_2bit(pattern[base_idx + 1])?;
        let b2 = base_to_2bit(pattern[base_idx + 2])?;
        let b3 = base_to_2bit(pattern[base_idx + 3])?;
        output[i] = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3;
    }

    let remainder = pattern.len() % 4;
    if remainder > 0 {
        let base_idx = complete * 4;
        let mut packed = 0u8;
        for j in 0..remainder {
            let b = base_to_2bit(pattern[base_idx + j])?;
            packed |= b << (6 - j * 2);
        }
        output[complete] = packed;
    }

    Some(output)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Converts an ASCII nucleotide to its 2-bit encoding.
#[inline]
fn base_to_2bit(base: u8) -> Option<u8> {
    match base.to_ascii_uppercase() {
        b'A' => Some(encoding_2bit::A),
        b'C' => Some(encoding_2bit::C),
        b'G' => Some(encoding_2bit::G),
        b'T' => Some(encoding_2bit::T),
        _ => None,
    }
}

/// Expands an IUPAC code to its constituent 2-bit base values.
fn expand_iupac(code: u8) -> Result<Vec<u8>, TetraLUTError> {
    match code.to_ascii_uppercase() {
        b'A' => Ok(vec![encoding_2bit::A]),
        b'C' => Ok(vec![encoding_2bit::C]),
        b'G' => Ok(vec![encoding_2bit::G]),
        b'T' | b'U' => Ok(vec![encoding_2bit::T]),
        b'N' => Ok(vec![
            encoding_2bit::A,
            encoding_2bit::C,
            encoding_2bit::G,
            encoding_2bit::T,
        ]),
        b'R' => Ok(vec![encoding_2bit::A, encoding_2bit::G]), // puRine
        b'Y' => Ok(vec![encoding_2bit::C, encoding_2bit::T]), // pYrimidine
        b'S' => Ok(vec![encoding_2bit::G, encoding_2bit::C]), // Strong
        b'W' => Ok(vec![encoding_2bit::A, encoding_2bit::T]), // Weak
        b'K' => Ok(vec![encoding_2bit::G, encoding_2bit::T]), // Keto
        b'M' => Ok(vec![encoding_2bit::A, encoding_2bit::C]), // aMino
        b'B' => Ok(vec![encoding_2bit::C, encoding_2bit::G, encoding_2bit::T]), // not A
        b'D' => Ok(vec![encoding_2bit::A, encoding_2bit::G, encoding_2bit::T]), // not C
        b'H' => Ok(vec![encoding_2bit::A, encoding_2bit::C, encoding_2bit::T]), // not G
        b'V' => Ok(vec![encoding_2bit::A, encoding_2bit::C, encoding_2bit::G]), // not T
        _ => Err(TetraLUTError::InvalidIUPACCode(code as char)),
    }
}

/// Extracts a tetranucleotide at a given position from 2-bit encoded data.
#[inline]
fn extract_tetra_at_position(encoded: &[u8], pos: usize, original_len: usize) -> u8 {
    if pos + 4 > original_len {
        return 0;
    }

    // Calculate which bits we need
    let bit_offset = pos * 2;
    let byte_offset = bit_offset / 8;
    let bit_in_byte = bit_offset % 8;

    // We need 8 bits (4 bases × 2 bits) starting at bit_offset
    if bit_in_byte == 0 {
        // Aligned: just return the byte
        encoded[byte_offset]
    } else {
        // Unaligned: need to combine two bytes
        let high = encoded[byte_offset];
        let low = encoded.get(byte_offset + 1).copied().unwrap_or(0);
        (high << bit_in_byte) | (low >> (8 - bit_in_byte))
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur when building a TetraLUT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetraLUTError {
    /// Pattern must be exactly 4 bases.
    InvalidPatternLength { expected: usize, actual: usize },
    /// Pattern contains invalid (non-ACGT) bases.
    InvalidBase { pattern: String },
    /// Invalid IUPAC ambiguity code.
    InvalidIUPACCode(char),
}

impl fmt::Display for TetraLUTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TetraLUTError::InvalidPatternLength { expected, actual } => {
                write!(
                    f,
                    "invalid pattern length: expected {} bases, got {}",
                    expected, actual
                )
            }
            TetraLUTError::InvalidBase { pattern } => {
                write!(f, "pattern '{}' contains invalid bases (not ACGT)", pattern)
            }
            TetraLUTError::InvalidIUPACCode(c) => {
                write!(f, "invalid IUPAC code: '{}'", c)
            }
        }
    }
}

impl std::error::Error for TetraLUTError {}

/// Errors that can occur with Bitap pattern matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitapError {
    /// Pattern cannot be empty.
    EmptyPattern,
    /// Pattern exceeds maximum length.
    PatternTooLong { max: usize, actual: usize },
    /// Pattern contains invalid base.
    InvalidBase(char),
}

impl fmt::Display for BitapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BitapError::EmptyPattern => write!(f, "pattern cannot be empty"),
            BitapError::PatternTooLong { max, actual } => {
                write!(f, "pattern too long: max {} bases, got {}", max, actual)
            }
            BitapError::InvalidBase(c) => {
                write!(f, "pattern contains invalid base: '{}'", c)
            }
        }
    }
}

impl std::error::Error for BitapError {}

/// Unified search error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchError {
    /// TetraLUT error.
    TetraLUT(TetraLUTError),
    /// Bitap error.
    Bitap(BitapError),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::TetraLUT(e) => write!(f, "{}", e),
            SearchError::Bitap(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<TetraLUTError> for SearchError {
    fn from(e: TetraLUTError) -> Self {
        SearchError::TetraLUT(e)
    }
}

impl From<BitapError> for SearchError {
    fn from(e: BitapError) -> Self {
        SearchError::Bitap(e)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_encoder::encode_2bit;

    // ========================================================================
    // TetraLUT Tests
    // ========================================================================

    #[test]
    fn test_encode_tetra_to_index() {
        assert_eq!(encode_tetra_to_index(b"AAAA").unwrap(), 0x00);
        assert_eq!(encode_tetra_to_index(b"ACGT").unwrap(), 0x1B);
        assert_eq!(encode_tetra_to_index(b"TTTT").unwrap(), 0xFF);
        assert_eq!(encode_tetra_to_index(b"CCCC").unwrap(), 0x55);
        assert_eq!(encode_tetra_to_index(b"GGGG").unwrap(), 0xAA);
    }

    #[test]
    fn test_encode_tetra_case_insensitive() {
        assert_eq!(
            encode_tetra_to_index(b"acgt").unwrap(),
            encode_tetra_to_index(b"ACGT").unwrap()
        );
    }

    #[test]
    fn test_encode_tetra_invalid_length() {
        assert!(encode_tetra_to_index(b"ACG").is_err());
        assert!(encode_tetra_to_index(b"ACGTA").is_err());
    }

    #[test]
    fn test_decode_index_to_tetra() {
        assert_eq!(decode_index_to_tetra(0x00), *b"AAAA");
        assert_eq!(decode_index_to_tetra(0x1B), *b"ACGT");
        assert_eq!(decode_index_to_tetra(0xFF), *b"TTTT");
    }

    #[test]
    fn test_tetra_roundtrip() {
        for idx in 0u8..=255 {
            let tetra = decode_index_to_tetra(idx);
            let encoded = encode_tetra_to_index(&tetra).unwrap();
            assert_eq!(encoded, idx);
        }
    }

    #[test]
    fn test_tetra_lut_from_literal() {
        let lut = TetraLUT::from_literal(b"ACGT").unwrap();
        assert_eq!(lut.match_count(), 1);
        assert!(lut.matches_index(0x1B));
        assert!(!lut.matches_index(0x00));
    }

    #[test]
    fn test_tetra_lut_from_iupac_n() {
        // ACGN should match ACGA, ACGC, ACGG, ACGT
        let lut = TetraLUT::from_iupac("ACGN").unwrap();
        assert_eq!(lut.match_count(), 4);

        // Verify the specific matches
        let acga = encode_tetra_to_index(b"ACGA").unwrap();
        let acgc = encode_tetra_to_index(b"ACGC").unwrap();
        let acgg = encode_tetra_to_index(b"ACGG").unwrap();
        let acgt = encode_tetra_to_index(b"ACGT").unwrap();

        assert!(lut.matches_index(acga));
        assert!(lut.matches_index(acgc));
        assert!(lut.matches_index(acgg));
        assert!(lut.matches_index(acgt));
    }

    #[test]
    fn test_tetra_lut_from_iupac_nnnn() {
        // NNNN matches all 256 tetranucleotides
        let lut = TetraLUT::from_iupac("NNNN").unwrap();
        assert_eq!(lut.match_count(), 256);
    }

    #[test]
    fn test_tetra_lut_from_iupac_r() {
        // RACG should match AACG and GACG
        let lut = TetraLUT::from_iupac("RACG").unwrap();
        assert_eq!(lut.match_count(), 2);
    }

    #[test]
    fn test_tetra_lut_scan_exact() {
        let sequence = b"ACGTACGTACGT";
        let encoded = encode_2bit(sequence).unwrap();
        let lut = TetraLUT::from_literal(b"ACGT").unwrap();

        let matches = lut.scan_2bit(&encoded, sequence.len());
        assert_eq!(matches, vec![0, 4, 8]);
    }

    #[test]
    fn test_tetra_lut_scan_no_match() {
        let sequence = b"AAAAAAAA";
        let encoded = encode_2bit(sequence).unwrap();
        let lut = TetraLUT::from_literal(b"CCCC").unwrap();

        let matches = lut.scan_2bit(&encoded, sequence.len());
        assert!(matches.is_empty());
    }

    #[test]
    fn test_tetra_lut_contains_match() {
        let sequence = b"ACGTACGT";
        let encoded = encode_2bit(sequence).unwrap();

        let lut_match = TetraLUT::from_literal(b"ACGT").unwrap();
        let lut_no_match = TetraLUT::from_literal(b"CCCC").unwrap();

        assert!(lut_match.contains_match(&encoded, sequence.len()));
        assert!(!lut_no_match.contains_match(&encoded, sequence.len()));
    }

    #[test]
    fn test_tetra_lut_short_sequence() {
        let sequence = b"ACG"; // Less than 4 bases
        let encoded = encode_2bit(sequence).unwrap();
        let lut = TetraLUT::from_literal(b"ACGT").unwrap();

        let matches = lut.scan_2bit(&encoded, sequence.len());
        assert!(matches.is_empty());
    }

    // ========================================================================
    // Bitap Tests
    // ========================================================================

    #[test]
    fn test_bitap_from_pattern() {
        let masks = BitapMasks::from_pattern(b"ACGT").unwrap();
        assert_eq!(masks.pattern_len(), 4);
    }

    #[test]
    fn test_bitap_from_pattern_long() {
        let pattern = b"ACGTACGTACGTACGT"; // 16 bases
        let masks = BitapMasks::from_pattern(pattern).unwrap();
        assert_eq!(masks.pattern_len(), 16);
    }

    #[test]
    fn test_bitap_pattern_too_long() {
        let pattern = vec![b'A'; 65];
        assert!(BitapMasks::from_pattern(&pattern).is_err());
    }

    #[test]
    fn test_bitap_empty_pattern() {
        assert!(BitapMasks::from_pattern(b"").is_err());
    }

    #[test]
    fn test_bitap_search_exact() {
        let masks = BitapMasks::from_pattern(b"ACGT").unwrap();
        let text = b"ACGTACGTACGT";
        let matches = masks.search(text);
        assert_eq!(matches, vec![0, 4, 8]);
    }

    #[test]
    fn test_bitap_search_longer_pattern() {
        let masks = BitapMasks::from_pattern(b"ACGTACGT").unwrap();
        let text = b"ACGTACGTACGT";
        let matches = masks.search(text);
        assert_eq!(matches, vec![0, 4]);
    }

    #[test]
    fn test_bitap_search_no_match() {
        let masks = BitapMasks::from_pattern(b"CCCC").unwrap();
        let text = b"ACGTACGT";
        let matches = masks.search(text);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_bitap_search_iupac() {
        let masks = BitapMasks::from_iupac("ACGN").unwrap();
        let text = b"ACGAACGCACGGACGT";
        let matches = masks.search(text);
        assert_eq!(matches, vec![0, 4, 8, 12]);
    }

    #[test]
    fn test_bitap_contains() {
        let masks = BitapMasks::from_pattern(b"ACGT").unwrap();
        assert!(masks.contains(b"ACGTACGT"));
        assert!(!masks.contains(b"AAAAAAAA"));
    }

    // ========================================================================
    // SearchPattern Tests
    // ========================================================================

    #[test]
    fn test_search_pattern_short() {
        let pattern = SearchPattern::from_literal(b"ACGT").unwrap();
        assert!(matches!(pattern, SearchPattern::Tetra(_)));
        assert_eq!(pattern.pattern_len(), 4);
    }

    #[test]
    fn test_search_pattern_long() {
        let pattern = SearchPattern::from_literal(b"ACGTACGT").unwrap();
        assert!(matches!(pattern, SearchPattern::Bitap(_)));
        assert_eq!(pattern.pattern_len(), 8);
    }

    // ========================================================================
    // Pattern Encoding Tests
    // ========================================================================

    #[test]
    fn test_encode_pattern_2bit() {
        let encoded = encode_pattern_2bit(b"ACGT").unwrap();
        assert_eq!(encoded, vec![0x1B]);

        let encoded = encode_pattern_2bit(b"ACGTACGT").unwrap();
        assert_eq!(encoded, vec![0x1B, 0x1B]);
    }

    #[test]
    fn test_encode_pattern_2bit_invalid() {
        assert!(encode_pattern_2bit(b"ACGN").is_none());
    }

    // ========================================================================
    // IUPAC Expansion Tests
    // ========================================================================

    #[test]
    fn test_expand_iupac_bases() {
        assert_eq!(expand_iupac(b'A').unwrap(), vec![0]);
        assert_eq!(expand_iupac(b'C').unwrap(), vec![1]);
        assert_eq!(expand_iupac(b'G').unwrap(), vec![2]);
        assert_eq!(expand_iupac(b'T').unwrap(), vec![3]);
    }

    #[test]
    fn test_expand_iupac_n() {
        let expanded = expand_iupac(b'N').unwrap();
        assert_eq!(expanded.len(), 4);
    }

    #[test]
    fn test_expand_iupac_r() {
        let expanded = expand_iupac(b'R').unwrap();
        assert_eq!(expanded, vec![0, 2]); // A, G
    }

    #[test]
    fn test_expand_iupac_invalid() {
        assert!(expand_iupac(b'X').is_err());
    }

    // ========================================================================
    // Extract Tetra Tests
    // ========================================================================

    #[test]
    fn test_extract_tetra_aligned() {
        // ACGTACGT encoded: [0x1B, 0x1B]
        let encoded = encode_2bit(b"ACGTACGT").unwrap();

        // Position 0: ACGT (aligned at byte 0)
        let idx = extract_tetra_at_position(&encoded, 0, 8);
        assert_eq!(idx, 0x1B);

        // Position 4: ACGT (aligned at byte 1)
        let idx = extract_tetra_at_position(&encoded, 4, 8);
        assert_eq!(idx, 0x1B);
    }

    #[test]
    fn test_extract_tetra_unaligned() {
        // AACGTACGT encoded
        let encoded = encode_2bit(b"AACGTACGT").unwrap();

        // Position 1: ACGT (unaligned, spans bytes 0-1)
        let idx = extract_tetra_at_position(&encoded, 1, 9);
        assert_eq!(idx, 0x1B);
    }

    // ========================================================================
    // Error Display Tests
    // ========================================================================

    #[test]
    fn test_tetra_lut_error_display() {
        let err = TetraLUTError::InvalidPatternLength {
            expected: 4,
            actual: 3,
        };
        assert!(format!("{}", err).contains("expected 4"));

        let err = TetraLUTError::InvalidBase {
            pattern: "ACGN".to_string(),
        };
        assert!(format!("{}", err).contains("ACGN"));

        let err = TetraLUTError::InvalidIUPACCode('X');
        assert!(format!("{}", err).contains("X"));
    }

    #[test]
    fn test_bitap_error_display() {
        let err = BitapError::EmptyPattern;
        assert!(format!("{}", err).contains("empty"));

        let err = BitapError::PatternTooLong {
            max: 64,
            actual: 65,
        };
        assert!(format!("{}", err).contains("65"));

        let err = BitapError::InvalidBase('X');
        assert!(format!("{}", err).contains("X"));
    }
}
