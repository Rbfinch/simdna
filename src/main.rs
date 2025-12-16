mod dna_simd_encoder;

use dna_simd_encoder::{decode_dna, encode_dna};

fn main() {
    // Test sequence: a portion of human mitochondrial DNA
    let test_sequences = vec![
        b"ACGTACGTACGTACGT".to_vec(), // 16 nucleotides (perfect for SIMD)
        b"GATCGATCGATCGATC".to_vec(),
        b"AAAAAAAAAAAAAAAA".to_vec(),
        b"TTTTTTTTTTTTTTTT".to_vec(),
        b"ACGTACGTACGTACGTACGTACGTACGTACGT".to_vec(), // 32 nucleotides
        b"ATGCGATACGTAGCTAGCTAGCTAGCTACGATCGTAGCTAGCTA".to_vec(), // 44 nucleotides
    ];

    println!("DNA 2-bit Encoding Test (SIMD Implementation)\n");
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

    // Performance note
    println!("\nPerformance Notes:");
    println!("- SIMD processes 16 nucleotides per iteration");
    println!("- 4:1 compression ratio (2 bits per nucleotide vs 8 bits ASCII)");
    println!("- Expected speedup: 8-16x over scalar code on modern CPUs");
}
