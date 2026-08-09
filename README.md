# taper
A lightweight neural network library in Rust with automatic differentiation.

## Features
- Dynamic computational graph with tape-based autograd
- SIMD-optimized tensor operations (AVX/SSE/NEON, with a scalar fallback)
- Neural network layers: Linear, ReLU, Sigmoid, Conv2D, MaxPool2D, AvgPool2D, Dropout, Flatten
- Optimizers: SGD (with momentum), Adam, AdamW, and learning rate scheduling
- Loss functions: MSE, Cross-entropy, BCE
- Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
- MNIST dataset support with data loading utilities

## Performance
- Optional BLAS acceleration for matrix operations
- Cross-platform SIMD optimizations
- Memory-efficient gradient computation

Measured on a MacBook Pro M4 Pro (12 cores), with `--features blas-accelerate`:

| Model | Result |
| --- | --- |
| MLP, 10 epochs | 99.1% train / 97.9% test, ~0.26 s/epoch |
| CNN, 2 epochs | 96.9% test |

## Technical Implementation

- GEMM: cache-blocked matrix multiplication, or CBLAS when the `blas` feature is on
- Convolution: im2col + GEMM, with a fused Conv+ReLU layer to cut memory traffic

## Usage
```sh
# Basic training
cargo run --example train_mnist

# With BLAS acceleration
cargo run --release --features blas-accelerate --example train_mnist

# Or MNIST CNN training
cargo run --release --features blas-accelerate --example train_mnist_cnn
```

### Inference

Forward passes record backward closures on a thread-local tape. Wrap inference
in a `no_grad` guard so evaluation does not accumulate a graph it will never
use, and switch stochastic layers such as `Dropout` into evaluation mode:

```rust
use taper::{nn::Module, tape};

model.set_training(false);
let _guard = tape::no_grad();
let predictions = model.forward(&inputs);
```

`Tape::reset()` clears the recorded graph between training steps.

## Quantization

Storage quantization for model compression. Weights are stored quantized and
dequantized to f32 for computation, so this reduces model size rather than
inference latency.

| Type | Size vs f32 | Notes |
| --- | --- | --- |
| `Int8` | 4x smaller | affine, per-tensor scale and zero point |
| `Int4` | 8x smaller | affine, two values packed per byte |
| `Float16` | 2x smaller | IEEE 754 half precision |
| `BFloat16` | 2x smaller | truncated f32, round-to-nearest-even |
| `NF4` | 8x smaller | normal-float 4-bit, absmax-scaled |

On the MNIST CNN, Int8 and Float16 both land within ~0.15% of the f32 model.

```sh
# Train and quantize a model
cargo run --release --features blas-accelerate --example ptq_quantize

# Quantization-aware training
cargo run --release --features blas-accelerate --example qat_example
```

QAT layers observe their inputs and weights during the forward pass to derive
scales, and are gated on the global switch plus training mode:

```rust
use taper::quantization::qat_manager::global;

global::enable_qat();
global::set_training_mode(true);
```

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all
```

Note that `.cargo/config.toml` sets `-C target-cpu=native`. That is right for
local benchmarking but produces binaries that may not run on other machines;
override `RUSTFLAGS` when building artifacts for distribution.
