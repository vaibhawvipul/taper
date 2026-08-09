# taper
A lightweight neural network library in Rust with automatic differentiation.

## Features
- Dynamic computational graph with tape-based autograd
- SIMD-optimized tensor operations (AVX/SSE/NEON, with a scalar fallback)
- Neural network layers: Linear, Conv2D, MaxPool2D, AvgPool2D, LayerNorm, BatchNorm2d, Dropout, Flatten, ReLU, Sigmoid
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

### Training on your own data

Training works against the [`Dataset`] trait, not any particular dataset. For
data already in memory, `TensorDataset` takes a stacked input tensor and a
stacked target tensor — anything after the leading sample dimension is your
feature shape, so `[N, features]` and `[N, C, H, W]` both work:

```rust
use taper::data::{DataLoader, TensorDataset};

let dataset = TensorDataset::new(
    Tensor::new(features, &[n_samples, n_features]),
    Tensor::new(labels, &[n_samples]),
);
let mut loader = DataLoader::new(dataset, 32, true);

trainer.fit(&mut loader, &mut val_loader, 10, true);
```

For data that does not fit in memory, or that needs decoding per batch,
implement the trait directly:

```rust
impl Dataset for MyDataset {
    fn len(&self) -> usize { self.rows.len() }

    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        // gather these samples into (inputs, targets)
    }
}
```

Batching is a gather over indices rather than one sample at a time, so a whole
batch can be assembled in a single pass.

[`Dataset`]: https://docs.rs/taper/latest/taper/data/trait.Dataset.html

### Normalization

`LayerNorm` normalizes the trailing dimensions of each sample, so it is
independent of batch composition. `BatchNorm2d` normalizes each channel across
the batch, so it tracks running statistics during training and uses those at
evaluation:

```rust
use taper::norm::{BatchNorm2d, LayerNorm};

let model = Sequential::new(vec![
    Box::new(Conv2dReLU::conv3x3(1, 32, 1, 1)),
    Box::new(BatchNorm2d::new(32)),
    // ...
]);

model.set_training(false); // BatchNorm switches to its running statistics
```

Running statistics are **buffers**, not parameters: they are saved and loaded
with the model but never handed to an optimizer, which would apply momentum and
weight decay to what are observations rather than weights. Normalizing with
batch statistics at inference would make a prediction depend on whatever else
shared its batch, and would divide by ~`eps` for a batch of one.

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

## Saving and loading

Models serialize to [safetensors](https://github.com/huggingface/safetensors),
so weights interoperate with the wider ecosystem:

```rust
use taper::{Tensor, safetensors};

safetensors::save(&[("weight", &w), ("bias", &b)], "model.safetensors")?;

let loaded = safetensors::load("model.safetensors")?;
assert_eq!(loaded["weight"].shape(), &[128, 784]);
```

Whole modules work too, with parameters named by position:

```rust
safetensors::save_module(&model, "model.safetensors")?;
safetensors::load_module(&model, "model.safetensors")?;
```

`F32`, `BF16`, `F16`, `I32` and `U8` round-trip exactly. Wider dtypes found in
other tools' checkpoints are narrowed on read — `F64` to `f32`, and
`I64`/`U64`/`U32` to `i32`. Malformed files (overlapping tensors, gaps in the
data buffer, a span that disagrees with its shape) are rejected rather than
silently producing wrong weights.

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
