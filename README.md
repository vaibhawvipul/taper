# taper
A lightweight neural network library in Rust with automatic differentiation.

## Features
- Dynamic computational graph with tape-based autograd
- SIMD-optimized tensor operations (AVX/SSE/NEON)
- Neural network layers: Linear, ReLU, Sigmoid
- Optimizers: SGD, Adam, AdamW with learning rate scheduling
- Loss functions: MSE, Cross-entropy, BCE
- MNIST dataset support with data loading utilities

## Performance
- Optional BLAS acceleration for matrix operations
- Cross-platform SIMD optimizations
- Memory-efficient gradient computation

## Usage
```sh
# Basic training
cargo run --example train_mnist

# With BLAS acceleration
cargo run --release --features blas-accelerate --example train_mnist
```
