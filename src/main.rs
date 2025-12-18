// Copyright (c) 2025-present Nicholas D. Crosbie
// SPDX-License-Identifier: MIT

mod dna_simd_encoder;

use dna_simd_encoder::{
    complement_4bit, complement_packed_byte, decode_dna_prefer_simd as decode_dna,
    encode_dna_prefer_simd as encode_dna, reverse_complement, reverse_complement_encoded,
};

fn main() {
    // Test sequences including IUPAC ambiguity codes
    let test_sequences: [Vec<u8>; 6] = [
        b"ACGTACGTACGTACGT".to_vec(), // 16 nucleotides (perfect for SIMD)
        b"GATCGATCGATCGATC".to_vec(),
        b"ACGTRYSWKMBDHVN-".to_vec(), // All 16 IUPAC codes
        b"NNNNNNNNNNNNNNNN".to_vec(), // All N's (common in sequencing)
        b"ACGTACGTACGTACGTACGTACGTACGTACGT".to_vec(), // 32 nucleotides
        b"ATGCNNNNRYSWACGT".to_vec(), // Mixed with ambiguity codes
    ];

    println!("DNA 4-bit Encoding Test (SIMD Implementation)\n");
    println!("Architecture: {}", std::env::consts::ARCH);

    #[cfg(target_arch = "x86_64")]
    println!("SSSE3 support: {}", is_x86_feature_detected!("ssse3"));

    println!("\n{}\n", "=".repeat(80));

    for (idx, sequence) in test_sequences.iter().enumerate() {
        println!("Test #{}", idx + 1);
        println!(
            "Original ({} bp): {}",
            sequence.len(),
            String::from_utf8_lossy(sequence)
        );

        // Encode
        let encoded = encode_dna(sequence);
        println!("Encoded ({} bytes): {:02X?}", encoded.len(), encoded);

        // Show binary representation
        print!("Binary: ");
        for byte in &encoded {
            print!("{:08b} ", byte);
        }
        println!();

        // Decode
        let decoded = decode_dna(&encoded, sequence.len());
        println!(
            "Decoded ({} bp): {}",
            decoded.len(),
            String::from_utf8_lossy(&decoded)
        );

        // Verify
        let original_trimmed = &sequence[..decoded.len()];
        let matches = original_trimmed == &decoded[..];
        println!(
            "Verification: {}",
            if matches { "✓ PASS" } else { "✗ FAIL" }
        );

        // Compression ratio
        let ratio = sequence.len() as f64 / encoded.len() as f64;
        println!(
            "Compression: {:.2}x ({} bytes → {} bytes)",
            ratio,
            sequence.len(),
            encoded.len()
        );

        println!("{}\n", "-".repeat(80));
    }

    // ========================================================================
    // Reverse Complement Examples
    // ========================================================================
    println!("\n{}", "=".repeat(80));
    println!("REVERSE COMPLEMENT EXAMPLES");
    println!("{}\n", "=".repeat(80));

    // Example 1: High-level reverse complement API
    println!("1. High-Level Reverse Complement API");
    println!("   (Works directly on ASCII sequences)\n");

    let dna_sequence = b"ACGTACGT";
    let revcomp = reverse_complement(dna_sequence);
    println!(
        "   Original:   5'-{}-3'",
        String::from_utf8_lossy(dna_sequence)
    );
    println!("   Rev. Comp:  3'-{}-5'", String::from_utf8_lossy(&revcomp));
    println!("   Expected:   ACGTACGT → ACGTACGT (palindrome!)");

    println!();

    let asymmetric = b"AAACCCGGG";
    let revcomp_asym = reverse_complement(asymmetric);
    println!(
        "   Original:   5'-{}-3'",
        String::from_utf8_lossy(asymmetric)
    );
    println!(
        "   Rev. Comp:  3'-{}-5'",
        String::from_utf8_lossy(&revcomp_asym)
    );

    println!("\n{}\n", "-".repeat(80));

    // Example 2: IUPAC ambiguity codes in reverse complement
    println!("2. IUPAC Ambiguity Codes in Reverse Complement\n");

    let iupac_seq = b"ACRYSWKM";
    let iupac_revcomp = reverse_complement(iupac_seq);
    println!(
        "   Original:   5'-{}-3'",
        String::from_utf8_lossy(iupac_seq)
    );
    println!(
        "   Rev. Comp:  3'-{}-5'",
        String::from_utf8_lossy(&iupac_revcomp)
    );
    println!();
    println!("   Complement mappings:");
    println!("   A↔T, C↔G, R(A/G)↔Y(C/T), S(G/C)↔S, W(A/T)↔W, K(G/T)↔M(A/C)");

    println!("\n{}\n", "-".repeat(80));

    // Example 3: Low-level bit rotation complement
    println!("3. Bit Rotation Complement (Low-Level API)\n");
    println!("   The encoding uses a clever bit layout where complement = rotate by 2 bits:");
    println!("   complement = ((bits << 2) | (bits >> 2)) & 0xF\n");

    let bases = [
        (0x1u8, 'A', 'T'),
        (0x2, 'C', 'G'),
        (0x4, 'T', 'A'),
        (0x8, 'G', 'C'),
    ];

    for (bits, base, comp_base) in bases {
        let comp = complement_4bit(bits);
        println!(
            "   {} (0x{:X} = {:04b}) → {} (0x{:X} = {:04b})",
            base, bits, bits, comp_base, comp, comp
        );
    }

    println!("\n{}\n", "-".repeat(80));

    // Example 4: Packed byte complement
    println!("4. Packed Byte Complement\n");
    println!("   Each byte contains 2 nucleotides (4 bits each).");
    println!("   complement_packed_byte complements both nibbles:\n");

    // AC encoded: A=0x1, C=0x2, packed as 0x12
    let packed_ac = 0x12u8;
    let comp_packed = complement_packed_byte(packed_ac);
    println!("   AC (0x{:02X}) → TG (0x{:02X})", packed_ac, comp_packed);

    // GT encoded: G=0x8, T=0x4, packed as 0x84
    let packed_gt = 0x84u8;
    let comp_gt = complement_packed_byte(packed_gt);
    println!("   GT (0x{:02X}) → CA (0x{:02X})", packed_gt, comp_gt);

    println!("\n{}\n", "-".repeat(80));

    // Example 5: Reverse complement on encoded data (fastest)
    println!("5. Reverse Complement on Encoded Data (SIMD-Accelerated)\n");
    println!("   For maximum performance, work directly with encoded data:\n");

    let long_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGT"; // 32 bp
    let encoded_long = encode_dna(long_seq);
    let revcomp_encoded = reverse_complement_encoded(&encoded_long, long_seq.len());
    let revcomp_decoded = decode_dna(&revcomp_encoded, long_seq.len());

    println!(
        "   Original ({} bp):   {}",
        long_seq.len(),
        String::from_utf8_lossy(long_seq)
    );
    println!(
        "   Encoded ({} bytes):  {:02X?}",
        encoded_long.len(),
        encoded_long
    );
    println!("   RevComp encoded:    {:02X?}", revcomp_encoded);
    println!(
        "   RevComp decoded:    {}",
        String::from_utf8_lossy(&revcomp_decoded)
    );

    println!("\n{}\n", "=".repeat(80));

    // Performance notes
    println!("Performance Notes:");
    println!("- SIMD processes 16 nucleotides per iteration");
    println!("- 2:1 compression ratio (4 bits per nucleotide vs 8 bits ASCII)");
    println!(
        "- Supports all 16 IUPAC nucleotide codes (A, C, G, T, R, Y, S, W, K, M, B, D, H, V, N, -)"
    );
    println!("- Bit rotation complement: ~2x faster than lookup tables");
    println!("- reverse_complement_encoded: Up to 21 GiB/s on encoded data");
    println!("- Expected speedup: 4-8x over scalar code on modern CPUs");
}
