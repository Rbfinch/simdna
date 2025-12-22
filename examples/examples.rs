// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

//! Comprehensive examples demonstrating all simdna library functions.
//!
//! Run with: `cargo run --example examples`

use simdna::dna_simd_encoder::{
    // Error type for _into variants
    BufferError,
    // Low-level utilities
    char_to_4bit,
    complement_4bit,
    complement_packed_byte,
    // Zero-allocation encoding/decoding functions
    decode_dna_into,
    // Core encoding/decoding functions (allocating)
    decode_dna_prefer_simd,
    encode_dna_into,
    encode_dna_prefer_simd,
    fourbit_to_char,
    // Helper functions for buffer sizing
    required_decoded_len,
    required_encoded_len,
    // Reverse complement functions (allocating)
    reverse_complement,
    reverse_complement_encoded,
    // Zero-allocation reverse complement functions
    reverse_complement_encoded_into,
    reverse_complement_into,
};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              simdna Library Examples                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Architecture detection
    println!("Platform: {}", std::env::consts::ARCH);
    #[cfg(target_arch = "x86_64")]
    println!(
        "SSSE3 support: {}",
        std::arch::is_x86_feature_detected!("ssse3")
    );
    #[cfg(target_arch = "aarch64")]
    println!("NEON support: always available on aarch64");
    println!();

    basic_encoding_decoding();
    iupac_ambiguity_codes();
    rna_sequences();
    reverse_complement_examples();
    zero_allocation_variants();
    low_level_utilities();
    practical_bioinformatics_patterns();
}

/// Basic encoding and decoding with SIMD acceleration
fn basic_encoding_decoding() {
    println!("═══ Basic Encoding and Decoding ═══\n");

    // Simple DNA sequence
    let sequence = b"ACGTACGT";
    println!("Original sequence: {}", String::from_utf8_lossy(sequence));

    // Encode the sequence (2 nucleotides per byte = 2:1 compression)
    let encoded = encode_dna_prefer_simd(sequence);
    println!(
        "Encoded length: {} bytes (from {} nucleotides)",
        encoded.len(),
        sequence.len()
    );
    println!("Encoded bytes: {:02X?}", encoded);

    // Decode back to original
    let decoded = decode_dna_prefer_simd(&encoded, sequence.len());
    println!("Decoded sequence: {}", String::from_utf8_lossy(&decoded));
    assert_eq!(decoded, sequence);

    // Empty sequence handling
    let empty: &[u8] = b"";
    let encoded_empty = encode_dna_prefer_simd(empty);
    assert!(encoded_empty.is_empty());
    println!("Empty sequence encodes to empty result ✓");

    // Single nucleotide (odd length)
    let single = b"A";
    let encoded_single = encode_dna_prefer_simd(single);
    assert_eq!(encoded_single.len(), 1); // Padded to 1 byte
    let decoded_single = decode_dna_prefer_simd(&encoded_single, 1);
    assert_eq!(decoded_single, single);
    println!("Single nucleotide roundtrip works ✓");

    println!();
}

/// Working with IUPAC ambiguity codes
fn iupac_ambiguity_codes() {
    println!("═══ IUPAC Ambiguity Codes ═══\n");

    // All IUPAC codes
    let all_iupac = b"ACGTRYSWKMBDHVN-";
    println!("All IUPAC codes: {}", String::from_utf8_lossy(all_iupac));

    let encoded = encode_dna_prefer_simd(all_iupac);
    let decoded = decode_dna_prefer_simd(&encoded, all_iupac.len());
    println!("Roundtrip result: {}", String::from_utf8_lossy(&decoded));
    assert_eq!(decoded, all_iupac);

    // Case insensitivity
    let lowercase = b"acgtryswkmbdhvn-";
    let encoded_lower = encode_dna_prefer_simd(lowercase);
    let decoded_lower = decode_dna_prefer_simd(&encoded_lower, lowercase.len());
    // Decoding always produces uppercase
    assert_eq!(decoded_lower, all_iupac);
    println!("Case insensitive encoding ✓ (lowercase → uppercase)");

    // Invalid characters become gaps
    let with_invalid = b"ACxGT123";
    let encoded_invalid = encode_dna_prefer_simd(with_invalid);
    let decoded_invalid = decode_dna_prefer_simd(&encoded_invalid, with_invalid.len());
    println!(
        "Invalid chars '{}' → '{}'",
        String::from_utf8_lossy(with_invalid),
        String::from_utf8_lossy(&decoded_invalid)
    );
    // x, 1, 2, 3 become gaps '-'
    assert_eq!(decoded_invalid, b"AC-GT---");

    println!();
}

/// RNA sequence handling
fn rna_sequences() {
    println!("═══ RNA Sequences ═══\n");

    // RNA uses U (Uracil) instead of T (Thymine)
    let rna = b"ACGU";
    println!("RNA sequence: {}", String::from_utf8_lossy(rna));

    let encoded = encode_dna_prefer_simd(rna);
    let decoded = decode_dna_prefer_simd(&encoded, rna.len());

    // U maps to T internally, decodes as T
    println!("Decoded as DNA: {}", String::from_utf8_lossy(&decoded));
    assert_eq!(decoded, b"ACGT");
    println!("RNA U maps to DNA T ✓");

    // Works with mixed case
    let rna_mixed = b"AcGu";
    let decoded_mixed = decode_dna_prefer_simd(&encode_dna_prefer_simd(rna_mixed), rna_mixed.len());
    assert_eq!(decoded_mixed, b"ACGT");

    println!();
}

/// Reverse complement operations
fn reverse_complement_examples() {
    println!("═══ Reverse Complement ═══\n");

    // High-level API: ASCII in, ASCII out
    let sequence = b"ATGCAACG";
    let rc = reverse_complement(sequence);
    println!(
        "Sequence:           5'-{}-3'",
        String::from_utf8_lossy(sequence)
    );
    println!("Reverse complement: 3'-{}-5'", String::from_utf8_lossy(&rc));
    assert_eq!(rc, b"CGTTGCAT");

    // Palindromic sequence (its own reverse complement)
    let palindrome = b"ACGT";
    let rc_palindrome = reverse_complement(palindrome);
    println!(
        "\nPalindrome: {} → {}",
        String::from_utf8_lossy(palindrome),
        String::from_utf8_lossy(&rc_palindrome)
    );
    assert_eq!(rc_palindrome, palindrome);

    // Double reverse complement = original
    let original = b"ATGCNNRYSWKM";
    let rc1 = reverse_complement(original);
    let rc2 = reverse_complement(&rc1);
    assert_eq!(rc2.as_slice(), original);
    println!("Double reverse complement = original ✓");

    // IUPAC ambiguity code complements
    println!("\nIUPAC complement pairs:");
    let pairs = [
        (b"R", "Y", "purine ↔ pyrimidine"),
        (b"K", "M", "keto ↔ amino"),
        (b"B", "V", "not-A ↔ not-T"),
        (b"D", "H", "not-C ↔ not-G"),
        (b"S", "S", "strong (self-complementary)"),
        (b"W", "W", "weak (self-complementary)"),
        (b"N", "N", "any (self-complementary)"),
    ];
    for (code, expected, desc) in pairs {
        let rc = reverse_complement(code);
        println!("  {} → {} ({})", code[0] as char, rc[0] as char, desc);
        assert_eq!(rc, expected.as_bytes());
    }

    // Low-level API: operates on encoded data for maximum performance
    println!("\nLow-level encoded reverse complement:");
    let seq = b"ACGTACGT";
    let encoded = encode_dna_prefer_simd(seq);
    let rc_encoded = reverse_complement_encoded(&encoded, seq.len());
    let rc_decoded = decode_dna_prefer_simd(&rc_encoded, seq.len());
    println!(
        "  {} → {} (via encoded representation)",
        String::from_utf8_lossy(seq),
        String::from_utf8_lossy(&rc_decoded)
    );
    // Verify consistency with high-level API
    assert_eq!(rc_decoded, reverse_complement(seq));

    println!();
}

/// Zero-allocation _into variants for performance-critical code
fn zero_allocation_variants() {
    println!("═══ Zero-Allocation _into Variants ═══\n");

    // These functions write to caller-provided buffers, avoiding heap allocations
    // Ideal for high-throughput processing loops

    let sequence = b"ACGTRYSWKMBDHVN-";
    let len = sequence.len();

    // 1. Check required buffer sizes
    let enc_len = required_encoded_len(len);
    let dec_len = required_decoded_len(len);
    println!("Sequence length: {} nucleotides", len);
    println!("Required encoded buffer: {} bytes", enc_len);
    println!("Required decoded buffer: {} bytes", dec_len);

    // 2. encode_dna_into: encode without allocation
    let mut encoded_buf = vec![0u8; enc_len];
    let bytes_written = encode_dna_into(sequence, &mut encoded_buf).unwrap();
    println!("\nencode_dna_into wrote {} bytes", bytes_written);

    // Verify it matches the allocating version
    let encoded_alloc = encode_dna_prefer_simd(sequence);
    assert_eq!(encoded_buf, encoded_alloc);
    println!("  Matches encode_dna_prefer_simd ✓");

    // 3. decode_dna_into: decode without allocation
    let mut decoded_buf = vec![0u8; dec_len];
    let bytes_written = decode_dna_into(&encoded_buf, len, &mut decoded_buf).unwrap();
    println!("\ndecode_dna_into wrote {} bytes", bytes_written);
    assert_eq!(decoded_buf.as_slice(), sequence);
    println!("  Roundtrip successful ✓");

    // 4. reverse_complement_into: reverse complement without allocation
    let mut rc_buf = vec![0u8; len];
    let bytes_written = reverse_complement_into(sequence, &mut rc_buf).unwrap();
    println!("\nreverse_complement_into wrote {} bytes", bytes_written);

    let rc_alloc = reverse_complement(sequence);
    assert_eq!(rc_buf, rc_alloc);
    println!("  Matches reverse_complement ✓");

    // 5. reverse_complement_encoded_into: on encoded data
    let mut rc_enc_buf = vec![0u8; enc_len];
    let bytes_written =
        reverse_complement_encoded_into(&encoded_buf, len, &mut rc_enc_buf).unwrap();
    println!(
        "\nreverse_complement_encoded_into wrote {} bytes",
        bytes_written
    );

    let rc_enc_alloc = reverse_complement_encoded(&encoded_buf, len);
    assert_eq!(rc_enc_buf, rc_enc_alloc);
    println!("  Matches reverse_complement_encoded ✓");

    // 6. Error handling: buffer too small
    println!("\nBuffer error handling:");
    let mut small_buf = vec![0u8; 1]; // Too small for 16 nucleotides
    match encode_dna_into(sequence, &mut small_buf) {
        Err(BufferError::BufferTooSmall { needed, actual }) => {
            println!("  BufferTooSmall: needed {} bytes, got {}", needed, actual);
        }
        Ok(_) => unreachable!("Buffer should be too small"),
    }

    // 7. Reusing buffers in a loop (the main use case)
    println!("\nReusing buffers in a processing loop:");
    let sequences = [b"ACGT".as_slice(), b"NNNNNN", b"ATGCATGC"];
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap();
    let mut enc_buffer = vec![0u8; required_encoded_len(max_len)];
    let mut dec_buffer = vec![0u8; required_decoded_len(max_len)];

    for seq in &sequences {
        let enc_needed = required_encoded_len(seq.len());
        encode_dna_into(seq, &mut enc_buffer[..enc_needed]).unwrap();
        decode_dna_into(
            &enc_buffer[..enc_needed],
            seq.len(),
            &mut dec_buffer[..seq.len()],
        )
        .unwrap();
        println!(
            "  {} → encoded → {}",
            String::from_utf8_lossy(seq),
            String::from_utf8_lossy(&dec_buffer[..seq.len()])
        );
    }
    println!("  Zero allocations in the loop ✓");

    println!();
}

/// Low-level utility functions
fn low_level_utilities() {
    println!("═══ Low-Level Utilities ═══\n");

    // char_to_4bit: convert a single nucleotide to its 4-bit encoding
    println!("char_to_4bit conversions:");
    for &c in b"ACGT" {
        let bits = char_to_4bit(c);
        println!("  '{}' → 0x{:X}", c as char, bits);
    }

    // fourbit_to_char: convert 4-bit encoding back to character
    println!("\nfourbit_to_char conversions:");
    for bits in [0x1, 0x2, 0x4, 0x8] {
        let c = fourbit_to_char(bits);
        println!("  0x{:X} → '{}'", bits, c as char);
    }

    // complement_4bit: compute complement using bit rotation
    println!("\ncomplement_4bit (bit rotation):");
    for &c in b"ACGT" {
        let bits = char_to_4bit(c);
        let comp_bits = complement_4bit(bits);
        let comp_char = fourbit_to_char(comp_bits);
        println!(
            "  '{}' (0x{:X}) → '{}' (0x{:X})",
            c as char, bits, comp_char as char, comp_bits
        );
    }

    // complement_packed_byte: complement two nucleotides packed in one byte
    println!("\ncomplement_packed_byte (two nucleotides):");
    let packed = (char_to_4bit(b'A') << 4) | char_to_4bit(b'C'); // AC
    let comp_packed = complement_packed_byte(packed);
    let high = fourbit_to_char((comp_packed >> 4) & 0xF);
    let low = fourbit_to_char(comp_packed & 0xF);
    println!(
        "  'AC' (0x{:02X}) → '{}{}' (0x{:02X})",
        packed, high as char, low as char, comp_packed
    );

    // Bit rotation property demonstration
    println!("\nBit rotation property: complement = ((bits << 2) | (bits >> 2)) & 0xF");
    let bits = char_to_4bit(b'A'); // 0x1
    let rotated = ((bits << 2) | (bits >> 2)) & 0xF;
    println!("  A: 0x{:X} → rotate → 0x{:X} (T)", bits, rotated);
    assert_eq!(rotated, char_to_4bit(b'T'));

    println!();
}

/// Practical bioinformatics patterns
fn practical_bioinformatics_patterns() {
    println!("═══ Practical Bioinformatics Patterns ═══\n");

    // Pattern 1: Batch processing with buffer reuse
    println!("Pattern 1: Batch processing with buffer reuse");
    println!("  (See zero_allocation_variants example above)\n");

    // Pattern 2: Finding reverse complement for primer design
    println!("Pattern 2: Primer design - finding reverse complement");
    let forward_primer = b"ATGCGTACGATCGATCG";
    let reverse_primer = reverse_complement(forward_primer);
    println!(
        "  Forward: 5'-{}-3'",
        String::from_utf8_lossy(forward_primer)
    );
    println!(
        "  Reverse: 5'-{}-3'",
        String::from_utf8_lossy(&reverse_primer)
    );

    // Pattern 3: Compressed storage
    println!("\nPattern 3: Compressed sequence storage");
    let genome_fragment = b"ATGCATGCATGCATGCATGCATGCATGCATGC"; // 32 nucleotides
    let compressed = encode_dna_prefer_simd(genome_fragment);
    println!(
        "  Original: {} bytes, Compressed: {} bytes ({}% reduction)",
        genome_fragment.len(),
        compressed.len(),
        100 - (compressed.len() * 100 / genome_fragment.len())
    );

    // Pattern 4: Working with ambiguous sequences
    println!("\nPattern 4: Handling ambiguous sequences");
    let ambiguous = b"ATGNNNCGATCGATCG";
    println!("  Input:  {}", String::from_utf8_lossy(ambiguous));
    let encoded = encode_dna_prefer_simd(ambiguous);
    let decoded = decode_dna_prefer_simd(&encoded, ambiguous.len());
    println!("  Output: {}", String::from_utf8_lossy(&decoded));
    println!("  N (any base) preserved ✓");

    // Pattern 5: Computing both strands
    println!("\nPattern 5: Computing both strands of a sequence");
    let sequence = b"ATGCAACGTGCA";
    let forward_encoded = encode_dna_prefer_simd(sequence);
    let reverse_encoded = reverse_complement_encoded(&forward_encoded, sequence.len());

    // Both strands now in compressed form
    println!("  Forward: {}", String::from_utf8_lossy(sequence));
    println!(
        "  Reverse: {}",
        String::from_utf8_lossy(&decode_dna_prefer_simd(&reverse_encoded, sequence.len()))
    );
    println!(
        "  Both stored in {} + {} = {} bytes (vs {} ASCII)",
        forward_encoded.len(),
        reverse_encoded.len(),
        forward_encoded.len() + reverse_encoded.len(),
        sequence.len() * 2
    );

    // Pattern 6: Zero-allocation reverse complement for streaming
    println!("\nPattern 6: Streaming reverse complement (zero-allocation)");
    let mut rc_buffer = [0u8; 64]; // Stack buffer for small sequences
    let reads = [b"ACGT".as_slice(), b"ATGCATGC", b"NNNNNNNN"];
    for read in &reads {
        if let Ok(len) = reverse_complement_into(read, &mut rc_buffer[..read.len()]) {
            println!(
                "  {} → {}",
                String::from_utf8_lossy(read),
                String::from_utf8_lossy(&rc_buffer[..len])
            );
        }
    }

    println!("\n═══ Examples Complete ═══\n");
}
