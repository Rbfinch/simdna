// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! # simdna
//!
//! High-performance DNA/RNA sequence encoding and decoding using SIMD instructions
//! with automatic fallback to scalar implementations.
//!
//! ## Features
//!
//! - **Hybrid 2-bit/4-bit encoding** with automatic bifurcation based on sequence content
//! - **2-bit encoding** for clean ACGT sequences (4x compression, 4 bases per byte)
//! - **4-bit encoding** supporting all IUPAC nucleotide codes (2x compression)
//! - **Bit-rotation-compatible encoding** enabling efficient complement calculation
//! - **SIMD acceleration** on x86_64 (SSSE3) and ARM64 (NEON)
//! - **Zero-allocation API** via `_into` variants for high-throughput pipelines
//! - **FASTQ quality score encoding** with binning and run-length encoding
//!
//! ## Hybrid Encoding (Bifurcated Storage)
//!
//! For optimal storage efficiency, use the hybrid encoder that automatically
//! selects 2-bit or 4-bit encoding based on sequence content:
//!
//! ```rust,ignore
//! use simdna::hybrid_encoder::{encode_bifurcated, EncodingType};
//!
//! // Clean sequences use 2-bit encoding (4x compression)
//! let clean = b"ACGTACGT";
//! let result = encode_bifurcated(clean);
//! assert_eq!(result.encoding, EncodingType::Clean2Bit);
//!
//! // Sequences with ambiguous bases use 4-bit encoding (2x compression)
//! let dirty = b"ACNGTACGT";
//! let result = encode_bifurcated(dirty);
//! assert_eq!(result.encoding, EncodingType::Dirty4Bit);
//! ```
//!
//! ## Quick Start
//!
//! ```rust
//! use simdna::dna_simd_encoder::{encode_dna_prefer_simd, decode_dna_prefer_simd};
//!
//! let sequence = b"ACGTRYSWKMBDHVN-";
//! let encoded = encode_dna_prefer_simd(sequence);
//! let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
//! assert_eq!(decoded, sequence);
//! ```
//!
//! ## Zero-Allocation API
//!
//! For high-throughput applications, use the `_into` variants:
//!
//! ```rust
//! use simdna::dna_simd_encoder::{encode_dna_into, decode_dna_into, required_encoded_len};
//!
//! let sequence = b"ACGTACGT";
//! let mut buffer = [0u8; 64];
//!
//! let bytes = encode_dna_into(sequence, &mut buffer).unwrap();
//! assert_eq!(bytes, 4);  // 8 nucleotides → 4 bytes
//! ```
//!
//! ## Reverse Complement
//!
//! ```rust
//! use simdna::dna_simd_encoder::reverse_complement;
//!
//! let sequence = b"ATGCAACG";
//! let rc = reverse_complement(sequence);
//! assert_eq!(rc, b"CGTTGCAT");
//! ```
//!
//! ## Quality Score Encoding
//!
//! Compress FASTQ quality scores with binning and run-length encoding:
//!
//! ```rust
//! use simdna::quality_encoder::{encode_quality_scores, decode_quality_scores, PhredEncoding};
//!
//! let quality = b"IIIIIIIIII"; // High-quality Illumina scores
//! let encoded = encode_quality_scores(quality, PhredEncoding::Phred33);
//! let decoded = decode_quality_scores(&encoded, PhredEncoding::Phred33);
//! assert_eq!(decoded.len(), quality.len());
//! ```
//!
//! ## Serialization for Database Storage
//!
//! Efficient binary serialization for SQLite BLOB storage:
//!
//! ```rust
//! use simdna::hybrid_encoder::encode_bifurcated;
//! use simdna::serialization::{to_blob, from_blob};
//!
//! let sequence = b"ACGTACGT";
//! let encoded = encode_bifurcated(sequence);
//!
//! // Serialize with checksum for database storage
//! let blob = to_blob(&encoded, true);
//! let decoded = from_blob(&blob).unwrap();
//! assert_eq!(decoded.original_len, 8);
//! ```
//!
//! ## Examples
//!
//! Run the comprehensive examples demonstrating all library functions:
//!
//! ```bash
//! cargo run --example examples
//! ```
//!
//! See the [`dna_simd_encoder`], [`hybrid_encoder`], [`tetra_scanner`], [`serialization`], and [`quality_encoder`] modules for the complete API.

pub mod dna_simd_encoder;
pub mod hybrid_encoder;
pub mod quality_encoder;
pub mod serialization;
pub mod tetra_scanner;
