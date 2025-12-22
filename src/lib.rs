// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! # simdna
//!
//! High-performance DNA/RNA sequence encoding and decoding using SIMD instructions
//! with automatic fallback to scalar implementations.
//!
//! ## Features
//!
//! - **4-bit encoding** supporting all IUPAC nucleotide codes (16 standard + U for RNA)
//! - **Bit-rotation-compatible encoding** enabling efficient complement calculation
//! - **SIMD acceleration** on x86_64 (SSSE3) and ARM64 (NEON)
//! - **Zero-allocation API** via `_into` variants for high-throughput pipelines
//! - **2:1 compression** ratio compared to ASCII representation
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
//! ## Examples
//!
//! Run the comprehensive examples demonstrating all library functions:
//!
//! ```bash
//! cargo run --example examples
//! ```
//!
//! See the [`dna_simd_encoder`] module for the complete API.

pub mod dna_simd_encoder;
