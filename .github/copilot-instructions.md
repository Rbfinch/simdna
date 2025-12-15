# Rust Best Practices for GitHub Copilot

This document outlines the coding standards and best practices to follow when generating Rust code for the SpikeQ project.

## Performance Principles

- **Zero-cost abstractions**: Prefer abstractions that don't add runtime overhead.
- **Minimize allocations**: Avoid unnecessary heap allocations and prefer stack allocation when appropriate.
- **Use iterators**: Leverage Rust's iterator methods instead of explicit loops when applicable for better performance and readability.
- **Consider inlining**: Use `#[inline]` for small, frequently called functions.
- **Batch operations**: Process data in batches to reduce overhead when working with large datasets.
- **Proper benchmarking**: Use `criterion` for benchmarking performance-critical code.

## Idiomatic Rust

- **Follow Rust naming conventions**:
  - Use `snake_case` for variables, functions, and modules
  - Use `CamelCase` for types and traits
  - Use `SCREAMING_SNAKE_CASE` for constants
- **Leverage the type system**: Use Rust's type system for safety and to express intent.
- **Use `Option` and `Result`**: Handle nullable values with `Option` and errors with `Result`, never use `unwrap()` or `expect()` in production code.
- **Pattern matching**: Prefer pattern matching over conditional logic where appropriate.
- **Use Rust 2021 edition features**: Take advantage of the latest language features.
- **Prefer methods to functions**: Use methods when operating on specific types.
- **Builder pattern**: For complex object construction, implement the builder pattern.

## Memory Safety and Error Handling

- **Ownership and borrowing**: Properly use Rust's ownership system, prefer borrowing (`&T`, `&mut T`) over taking ownership when possible.
- **No unsafe code**: Avoid `unsafe` blocks unless absolutely necessary and thoroughly document them when used.
- **Proper error handling**: Use the `?` operator for error propagation and create custom error types for complex error scenarios.
- **Context for errors**: Add context to errors using crates like `anyhow` or `thiserror`.
- **Avoid panics**: Design APIs to return `Result` instead of panicking.

## Code Structure and Organization

- **Modular design**: Break code into logical modules using Rust's module system.
- **Private by default**: Keep implementation details private, only expose what's necessary.
- **Clean interfaces**: Design clear, consistent APIs with well-defined behavior.
- **Single responsibility**: Functions and types should have a single responsibility.
- **DRY principle**: Don't repeat yourself, extract common functionality into reusable components.

## Documentation and Comments

- **Document public items**: All public functions, types, and modules should have documentation comments.
- **Include examples**: Add examples in documentation to demonstrate usage.
- **Document panics and errors**: Clearly document when functions can panic or return errors.
- **Comment complex logic**: Add inline comments for complex algorithms or non-obvious code.
- **Follow rustdoc conventions**: Use `///` for documentation comments and `//` for regular comments.
- **Use markdown in doc comments**: Format documentation with markdown for readability.

## Testing

- **Unit tests**: Write comprehensive unit tests for all public functionality.
- **Integration tests**: Test interactions between components with integration tests.
- **Property-based testing**: Use property-based testing (via `proptest` or similar) for complex behavior.
- **Test edge cases**: Explicitly test boundary conditions and error paths.
- **Doc tests**: Include runnable examples in documentation that serve as tests.

## Performance Optimization

- **Profile before optimizing**: Use profiling tools to identify bottlenecks before optimizing.
- **Parallelism**: Use `rayon` for data parallelism where appropriate.
- **Async/Await**: Use async/await for I/O-bound operations.
- **Efficient data structures**: Choose appropriate data structures for the task.
- **Minimize string formatting**: Avoid unnecessary string formatting in performance-critical paths.

## Dependency Management

- **Minimize dependencies**: Only add dependencies when necessary.
- **Vet dependencies**: Check for maintenance status, security issues, and performance.
- **Specify version constraints**: Use appropriate version constraints in `Cargo.toml`.
- **Feature flags**: Use feature flags to enable only needed functionality from dependencies.

## Specific Project Patterns

- **Follow existing patterns**: Maintain consistency with the current codebase patterns.
- **FASTQ file processing**: Use efficient parsing strategies for FASTQ files.
- **Regular expression usage**: Be cautious with regex performance, compile regexes once and reuse them.
- **JSON handling**: Use serde efficiently for JSON processing.

## Coding Style

- **Run rustfmt**: Ensure code follows the project's rustfmt configuration.
- **Run clippy**: Address all relevant clippy lints.
- **Consistent formatting**: Maintain consistent style throughout the codebase.
- **Line length**: Keep lines under 100 characters when possible.
- **Logical grouping**: Group related functionality together.

By following these guidelines, we can ensure that the code generated for SpikeQ is performant, idiomatic, maintainable, and well-documented.
