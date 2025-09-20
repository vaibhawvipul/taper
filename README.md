# taper
A lightweight neural network library in Rust with automatic differentiation.

## Goal - >99% with CNN in 50 epochs

## Features
- Dynamic computational graph with tape-based autograd
- SIMD-optimized tensor operations (AVX/SSE/NEON)
- Neural network layers: Linear, ReLU, Sigmoid, Conv2D, MaxPool2D, AvgPool2D, Flatten
- Optimizers: SGD, Adam, AdamW with learning rate scheduling
- Loss functions: MSE, Cross-entropy, BCE
- MNIST dataset support with data loading utilities

## Performance
- Optional BLAS acceleration for matrix operations
- Cross-platform SIMD optimizations
- Memory-efficient gradient computation

## Technical Implementation

- Operation Fusion: Combines multiple ops (Conv+ReLU) into single kernels to reduce memory traffic
- GEMM: Cache-blocked matrix multiplication with AVX vectorization for optimal CPU utilization
- Convolution: Direct 3x3 kernels bypass im2col; 1x1 kernels use pure matrix multiplication

## Usage
```sh
# Basic training
cargo run --example train_mnist

# With BLAS acceleration (98% accuracy in 10 epochs)
cargo run --release --features blas-accelerate --example train_mnist

# Or MNIST CNN training (around 95% accuracy in 50 epochs)
cargo run --release --features blas-accelerate --example train_mnist_cnn
```
