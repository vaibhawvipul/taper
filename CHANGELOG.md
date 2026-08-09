# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

The tensor representation changed substantially in this release; code written
against 0.1 that only used `Tensor`, `nn` and `optim` should still compile.

### Fixed — correctness

- **Convolution weights never received a gradient.** `im2col` and `transpose_4d`
  rebuilt their outputs without a backward edge, severing the graph on both
  sides of the matmul. A CNN's accuracy came entirely from its Linear head
  learning on frozen random features. Measured on MNIST, the first conv layer's
  weights moved by exactly zero before the fix.
- **`Tensor::cat` declared only one of its inputs as a tape dependency.** Once
  `backward` began traversing the dependency graph, every other input's subgraph
  became unreachable, so a grouped convolution updated only its first group.
- **Gradients were silently dropped for single-operation graphs.** `0` was used
  both as a tape node id and as the "not an operation output" sentinel.
- **`max(dim)` returned wrong values** for any non-final dimension of a rank-3+
  tensor, with a clamp hiding the out-of-range indices.
- **Asymmetric int8 quantization discarded half its range.** The zero point was
  derived as `-min/scale`, correct only for an unsigned grid.
- **conv2d paired weights with the wrong patches.** The weight was reshaped to
  `[K, C_out]`, which reinterprets the buffer rather than transposing it.
- **The 1×1 im2col path was a plain memcpy**, correct only when the spatial size
  was 1; the general path underflowed `usize` on padded windows.
- **SIMD kernels produced silent zeros** on targets without AVX/SSE2/NEON, and
  indexed their second and third operands using the first's length.
- Both cross-entropy backward passes recorded new operations *during* backward.
- `SGD` accepted a `momentum` argument and discarded it.
- Evaluation grew the tape without bound; `Dropout` inside a `Sequential` could
  never be switched off.
- Quantization-aware training was inert: the scale stayed at its placeholder
  `1.0`, and module ids collided between layers of the same shape.

### Added

- `taper::gradcheck` — public finite-difference verification, applicable to your
  own `Module`. Every differentiable operation in the crate is checked with it.
- `taper::error::ShapeError` and `try_*` forms of the shape-critical operations
  (`try_new`, `try_reshape`, `try_matmul`, `try_add`/`sub`/`mul`/`div`). The
  panicking forms delegate to them, so the two cannot drift.
- `taper::safetensors` — read and write the safetensors format, including
  `F64`/`I64` narrowing on read and rejection of malformed files.
- Strided tensors: `(storage, shape, stride, offset)`, making `transpose`,
  `reshape` and `narrow` zero-copy views.
- NumPy-rule broadcasting via `expand`, implemented as stride-0 views.
- `DType` — narrow storage (`bf16`/`f16`/`i32`/`u8`) with `f32` computation.
- `LayerNorm` and `BatchNorm2d`, the latter with running statistics.
- `Module::buffers` — non-learnable state that is checkpointed but never
  optimized.
- `Dataset` trait, generic `DataLoader`, and `TensorDataset`, so training is no
  longer restricted to MNIST.
- `tape::no_grad` for inference; `Module::set_training` for train/eval.
- `Trainer::load_checkpoint`, `save_safetensors`, `load_safetensors`.
- Int4, BFloat16 and NF4 quantization, replacing `unimplemented!()`.
- CI over five configurations: three SIMD paths and two BLAS vendors.

### Changed

- `backward` traverses only the operations its output depends on. Two
  independent graphs on one tape no longer run each other's backward passes.
- A poisoned lock no longer disables a tensor permanently.
- `data()`/`data_mut()` assert dense `f32` storage; use `elements()` or
  `to_vec()` for any layout or dtype.
- `ReduceLROnPlateau` takes a `PlateauMode` enum rather than a `String`, and no
  longer prints to stdout.
- `Adam` and `AdamW` implement the `Optimizer` trait.
- `Linear` uses He-uniform initialization (`sqrt(6/fan_in)`); the previous bound
  gave a third of the intended variance.

### Removed

- `conv2d_direct_3x3` (unused, no autograd), `HistogramObserver`,
  `ObserverManager`, `BasicBlock`, `QuantizationSchema`, `fma_f32_simd`, and the
  `Int8Tensor::min_val` placeholder.

### Known limitations

- CPU only; the autograd tape is thread-local, so no data-parallel training.
- No `Embedding`, attention, `GELU`/`SiLU` or recurrent layers — a transformer
  cannot yet be expressed.
- `Trainer` is classification-only (cross-entropy, accuracy).
- Performance has not been benchmarked against other frameworks.
