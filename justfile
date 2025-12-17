# simdna - justfile for development and publishing

# Extract version from Cargo.toml
get_version:
    @grep '^version' Cargo.toml | head -1 | sed 's/.*"\([0-9]*\.[0-9]*\.[0-9]*\)".*/\1/'

# Check if version is in CHANGELOG.md
check_changelog:
    #!/bin/bash
    set -e
    VERSION=$(just get_version)
    if ! grep -q "\[$VERSION\]" CHANGELOG.md; then
        echo "Error: Version $VERSION not found in CHANGELOG.md"
        exit 1
    fi
    echo "✓ Version $VERSION found in CHANGELOG.md"

# Run all tests
test:
    cargo test

# Run tests with verbose output
test-verbose:
    cargo test -- --nocapture

# Run benchmarks
bench:
    cargo bench

# Build release binary
build:
    cargo build --release

# Run clippy lints
lint:
    cargo clippy -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting without modifying
fmt-check:
    cargo fmt -- --check

# Run the example binary
run:
    cargo run --release

# Generate documentation
doc:
    cargo doc --no-deps --open

# Clean build artifacts
clean:
    cargo clean

# Run fuzz tests (requires nightly)
fuzz target="roundtrip" duration="60":
    cd fuzz && cargo +nightly fuzz run {{ target }} -- -max_total_time={{ duration }}

# Publish package
do_publish:
    cargo run --release
    git add .; git commit -m "Release v$(just get_version)"; git push
    cargo publish

# Full pre-publish check
check: fmt-check lint test
    @echo "✓ All checks passed"

# Main publish command that runs all steps
publish: check_changelog check do_publish
    @echo "✓ Successfully published simdna version $(just get_version)"
