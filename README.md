# taper
A simple neural network library in Rust.

This library is an autograd engine with dynamic computational graph and neural network library implemented from scratch in Rust. It supports basic tensor operations, automatic differentiation, and simple neural network layers.

We support -
- Basic tensor operations (addition, multiplication, etc.)
- Automatic differentiation
- Simple neural network layers (Linear, ReLU, Sigmoid)
- Sequential model container
- Mean Squared Error loss
- Stochastic Gradient Descent optimizer
- Training a simple model (XOR problem)

This is currently a work-in-progress and is not optimized for performance. SIMD optimizations are being worked on. The code is partially zero-copy, it needs more work.

## Example

```sh
cargo run
```
