use crate::{ops, quantization::QuantizationConfig, tape::Tape};
use smallvec::SmallVec;
use std::sync::{RwLockReadGuard, RwLockWriteGuard, atomic::Ordering};

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::{Arc, RwLock};

// SIMD utility module for cross-platform support
pub mod simd {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    // SIMD width detection
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    pub const SIMD_WIDTH: usize = 8;

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx")))]
    pub const SIMD_WIDTH: usize = 4;

    #[cfg(target_arch = "aarch64")]
    pub const SIMD_WIDTH: usize = 4;

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub const SIMD_WIDTH: usize = 1;

    // Cross-platform SIMD operations
    #[inline(always)]
    pub unsafe fn add_f32_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            if is_x86_feature_detected!("avx") {
                add_f32_avx(a, b, out);
                return;
            }
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        {
            if is_x86_feature_detected!("sse2") {
                unsafe {
                    add_f32_sse(a, b, out);
                }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { add_f32_neon(a, b, out) };
            return;
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    #[target_feature(enable = "avx")]
    unsafe fn add_f32_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let idx = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
            let result = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(idx), result);
        }
        // Handle remainder
        for i in (chunks * 8)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    #[target_feature(enable = "sse")]
    unsafe fn add_f32_sse(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = _mm_loadu_ps(unsafe { a.as_ptr().add(idx) });
            let vb = _mm_loadu_ps(unsafe { b.as_ptr().add(idx) });
            let result = _mm_add_ps(va, vb);
            _mm_storeu_ps(unsafe { out.as_mut_ptr().add(idx) }, result);
        }
        // Handle remainder
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn add_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
            let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
            let result = unsafe { vaddq_f32(va, vb) };
            unsafe { vst1q_f32(out.as_mut_ptr().add(idx), result) };
        }
        // Handle remainder
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    #[allow(dead_code)]
    fn add_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    // Multiplication operations
    #[inline(always)]
    pub unsafe fn mul_f32_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            if is_x86_feature_detected!("avx") {
                mul_f32_avx(a, b, out);
                return;
            }
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        {
            if is_x86_feature_detected!("sse") {
                unsafe {
                    mul_f32_sse(a, b, out);
                }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { mul_f32_neon(a, b, out) };
            return;
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    #[target_feature(enable = "avx")]
    unsafe fn mul_f32_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let idx = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
            let result = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(idx), result);
        }
        for i in (chunks * 8)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    #[target_feature(enable = "sse")]
    unsafe fn mul_f32_sse(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = _mm_loadu_ps(unsafe { a.as_ptr().add(idx) });
            let vb = _mm_loadu_ps(unsafe { b.as_ptr().add(idx) });
            let result = _mm_mul_ps(va, vb);
            _mm_storeu_ps(unsafe { out.as_mut_ptr().add(idx) }, result);
        }
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = unsafe { vld1q_f32(a.as_ptr().add(idx)) };
            let vb = unsafe { vld1q_f32(b.as_ptr().add(idx)) };
            let result = unsafe { vmulq_f32(va, vb) };
            unsafe { vst1q_f32(out.as_mut_ptr().add(idx), result) };
        }
        for i in (chunks * 4)..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    #[allow(dead_code)]
    fn mul_f32_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    // FMA operations for matmul
    #[inline(always)]
    pub unsafe fn fma_f32_simd(a: f32, b: &[f32], c: &mut [f32]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        {
            if is_x86_feature_detected!("fma") {
                fma_f32_avx(a, b, c);
                return;
            }
        }

        // Fallback to mul-add
        for i in 0..b.len() {
            c[i] += a * b[i];
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    #[target_feature(enable = "avx,fma")]
    unsafe fn fma_f32_avx(a: f32, b: &[f32], c: &mut [f32]) {
        let va = _mm256_set1_ps(a);
        let chunks = b.len() / 8;
        for i in 0..chunks {
            let idx = i * 8;
            let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
            let vc = _mm256_loadu_ps(c.as_ptr().add(idx));
            let result = _mm256_fmadd_ps(va, vb, vc);
            _mm256_storeu_ps(c.as_mut_ptr().add(idx), result);
        }
        for i in (chunks * 8)..b.len() {
            c[i] += a * b[i];
        }
    }
}

#[derive(Clone)]
pub struct Tensor {
    // Use RwLock for read-heavy workloads (most operations read data)
    data: Arc<RwLock<Vec<f32>>>,
    pub(crate) shape: SmallVec<[usize; 4]>,
    pub grad: Arc<RwLock<Option<Vec<f32>>>>,
    pub requires_grad: bool,
    pub tape_node: Arc<std::sync::atomic::AtomicUsize>,
}

/// Quantized tensor that can hold different precision types
#[derive(Clone, Debug)]
pub enum QuantizedTensor {
    /// Int8 quantized tensor
    Int8(Int8Tensor),
    /// Int4 quantized tensor (packed representation)
    Int4(Int4Tensor),
    /// Float16 quantized tensor
    Float16(Float16Tensor),
    /// BFloat16 quantized tensor
    BFloat16(BFloat16Tensor),
    /// NF4 quantized tensor
    NF4(NF4Tensor),
}

/// Int8 quantized tensor
#[derive(Clone, Debug)]
pub struct Int8Tensor {
    data: Arc<RwLock<Vec<i8>>>,
    shape: SmallVec<[usize; 4]>,
    scale: f32,
    zero_point: i32,
    min_val: f32, // Minimum value in original data
}

/// Int4 quantized tensor (packed representation - 2 values per byte)
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Int4Tensor {
    data: Arc<RwLock<Vec<u8>>>, // Packed: 2 int4 values per u8
    shape: SmallVec<[usize; 4]>,
    scale: f32,
    zero_point: i32,
}

/// Float16 quantized tensor
#[derive(Clone, Debug)]
pub struct Float16Tensor {
    data: Arc<RwLock<Vec<u16>>>, // Float16 stored as u16
    shape: SmallVec<[usize; 4]>,
}

/// BFloat16 quantized tensor
#[derive(Clone, Debug)]
pub struct BFloat16Tensor {
    data: Arc<RwLock<Vec<u16>>>, // BFloat16 stored as u16
    shape: SmallVec<[usize; 4]>,
}

/// NF4 quantized tensor
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct NF4Tensor {
    data: Arc<RwLock<Vec<u8>>>, // Packed NF4 values
    shape: SmallVec<[usize; 4]>,
    scale: f32,
    zero_point: i32,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data();
        let has_grad = self.grad.read().unwrap().is_some();
        f.debug_struct("Tensor")
            .field("data", &data.as_slice())
            .field("shape", &self.shape)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &has_grad)
            .finish()
    }
}

impl QuantizedTensor {
    /// Dequantize back to f32 tensor
    pub fn dequantize(&self) -> Tensor {
        match self {
            QuantizedTensor::Int8(tensor) => tensor.dequantize(),
            QuantizedTensor::Int4(tensor) => tensor.dequantize(),
            QuantizedTensor::Float16(tensor) => tensor.dequantize(),
            QuantizedTensor::BFloat16(tensor) => tensor.dequantize(),
            QuantizedTensor::NF4(tensor) => tensor.dequantize(),
        }
    }

    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        match self {
            QuantizedTensor::Int8(tensor) => &tensor.shape,
            QuantizedTensor::Int4(tensor) => &tensor.shape,
            QuantizedTensor::Float16(tensor) => &tensor.shape,
            QuantizedTensor::BFloat16(tensor) => &tensor.shape,
            QuantizedTensor::NF4(tensor) => &tensor.shape,
        }
    }
}

impl Int8Tensor {
    pub fn new(data: Vec<i8>, shape: SmallVec<[usize; 4]>, scale: f32, zero_point: i32) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
            scale,
            zero_point,
            min_val: 0.0, // Placeholder, should be set during quantization
        }
    }

    pub fn dequantize(&self) -> Tensor {
        let data = self.data();
        let f32_data: Vec<f32> = data
            .iter()
            .map(|&q| (q as i32 - self.zero_point) as f32 * self.scale + self.min_val)
            .collect();

        Tensor::new(f32_data, &self.shape)
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<i8>> {
        self.data.read().expect("Int8Tensor data lock poisoned")
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn zero_point(&self) -> i32 {
        self.zero_point
    }
}

impl Int4Tensor {
    pub fn new(data: Vec<u8>, shape: SmallVec<[usize; 4]>, scale: f32, zero_point: i32) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
            scale,
            zero_point,
        }
    }

    pub fn dequantize(&self) -> Tensor {
        unimplemented!(
            "Int4 dequantization not yet implemented: unpacking and scale/zero_point conversion needed"
        )
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<u8>> {
        self.data.read().expect("Int4Tensor data lock poisoned")
    }
}

impl Float16Tensor {
    pub fn new(data: Vec<u16>, shape: SmallVec<[usize; 4]>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
        }
    }

    /// Create a float16 tensor from an f32 tensor
    pub fn from_f32_tensor(tensor: &Tensor) -> Self {
        let data = tensor.data();
        let f16_data: Vec<u16> = data.iter().map(|&x| Tensor::f32_to_f16(x)).collect();
        Self::new(f16_data, tensor.shape.clone())
    }

    pub fn dequantize(&self) -> Tensor {
        let data = self.data();
        let f32_data: Vec<f32> = data.iter().map(|&x| Tensor::f16_to_f32(x)).collect();

        Tensor::new(f32_data, &self.shape)
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<u16>> {
        self.data.read().expect("Float16Tensor data lock poisoned")
    }
}

impl BFloat16Tensor {
    pub fn new(data: Vec<u16>, shape: SmallVec<[usize; 4]>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
        }
    }

    pub fn dequantize(&self) -> Tensor {
        unimplemented!("BFloat16 dequantization not yet implemented: bf16-to-f32 conversion needed")
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<u16>> {
        self.data.read().expect("BFloat16Tensor data lock poisoned")
    }
}

impl NF4Tensor {
    pub fn new(data: Vec<u8>, shape: SmallVec<[usize; 4]>, scale: f32, zero_point: i32) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
            scale,
            zero_point,
        }
    }

    pub fn dequantize(&self) -> Tensor {
        unimplemented!(
            "NF4 dequantization not yet implemented: NF4 unpacking and lookup table conversion needed"
        )
    }

    pub fn data(&self) -> std::sync::RwLockReadGuard<'_, Vec<u8>> {
        self.data.read().expect("NF4Tensor data lock poisoned")
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Self {
        Tensor {
            data: Arc::new(RwLock::new(data)),
            shape: shape.iter().cloned().collect(),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: false,
            tape_node: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn scalar(value: f32) -> Self {
        Tensor::new(vec![value], &[1])
    }

    pub fn requires_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn data(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.data.read().expect("data RwLock poisoned")
    }

    #[inline]
    pub fn data_mut(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.data.write().expect("data RwLock poisoned")
    }

    /// Read-only view of grad vector if it exists
    #[inline]
    pub fn grad_ref(&self) -> Option<std::sync::Arc<Vec<f32>>> {
        let g = self.grad.read().expect("grad RwLock poisoned");
        g.as_ref().map(|v| std::sync::Arc::new(v.clone()))
    }

    /// Convenience: clone grads into a new non-requiring-grad Tensor
    pub fn grad(&self) -> Option<Arc<Tensor>> {
        let g = self.grad.read().expect("grad RwLock poisoned");
        g.as_ref().map(|v| {
            let mut t = Tensor::new(v.clone(), &self.shape);
            t.requires_grad = false;
            Arc::new(t)
        })
    }

    pub fn backward(&self) {
        let ones = vec![1.0; self.data().len()];
        *self.grad.write().unwrap() = Some(ones);

        // Use AtomicUsize::load; define a sentinel like 0 = “no node”
        let node_id = self.tape_node.load(Ordering::SeqCst);
        if node_id != 0 {
            crate::tape::backward(node_id);
        }
    }

    pub fn zero_grad(&self) {
        *self.grad.write().unwrap() = None;
    }

    pub fn from_data(&self, data: Vec<f32>, shape: &[usize]) -> Tensor {
        let mut tensor = Tensor::new(data, shape);
        if self.requires_grad {
            tensor.requires_grad = true;
        }
        tensor
    }

    /// Cache-friendly blocked transpose
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Can only transpose 2D tensors");

        let rows = self.shape[0];
        let cols = self.shape[1];
        let data = self.data().clone(); // Clone once, no more locks
        let mut result = vec![0.0; data.len()];
        let block_size = 16; // Optimal for most cache sizes

        for i0 in (0..rows).step_by(block_size) {
            for j0 in (0..cols).step_by(block_size) {
                let i_max = (i0 + block_size).min(rows);
                let j_max = (j0 + block_size).min(cols);

                for i in i0..i_max {
                    for j in j0..j_max {
                        result[j * rows + i] = data[i * cols + j];
                    }
                }
            }
        }

        let mut output = Tensor::new(result, &[cols, rows]);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let (rows, cols) = (rows, cols);

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; rows * cols]);
                    }
                    let gin = slot.as_mut().unwrap();
                    for i in 0..rows {
                        for j in 0..cols {
                            gin[i * cols + j] += gout[j * rows + i];
                        }
                    }
                }
            });
        }

        output
    }

    /// SIMD-optimized sigmoid using fast approximation
    pub fn sigmoid(&self) -> Tensor {
        let data = self.data().clone();
        let mut result = vec![0.0; data.len()];

        // Fast sigmoid approximation: σ(x) ≈ 0.5 + 0.5 * tanh(0.5 * x)
        // Or use exact: 1 / (1 + exp(-x))
        for (i, &x) in data.iter().enumerate() {
            result[i] = if x > 0.0 {
                let exp_neg_x = (-x).exp();
                1.0 / (1.0 + exp_neg_x)
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            };
        }

        let mut output = Tensor::new(result, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let out_data = output.data().clone(); // Clone output data BEFORE closure

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // Use out_data instead of out.data()
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; out_data.len()]);
                    }
                    let gin = slot.as_mut().unwrap();
                    for ((gi, &g), &s) in gin.iter_mut().zip(gout.iter()).zip(out_data.iter()) {
                        *gi += g * s * (1.0 - s);
                    }
                }
            });
        }

        output
    }

    pub fn add_broadcast(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            return self + other;
        }

        if self.shape.len() == 2 && other.shape.len() == 1 {
            assert_eq!(
                self.shape[1], other.shape[0],
                "Last dimension must match for broadcasting"
            );

            let batch_size = self.shape[0];
            let features = self.shape[1];
            let (self_data, other_data) = {
                let a_guard = self.data();
                let b_guard = other.data();
                (a_guard.clone(), b_guard.clone()) // Clone data once
            };

            let mut result = vec![0.0; self_data.len()];

            // Optimized broadcasting with better memory access pattern
            for b in 0..batch_size {
                let offset = b * features;
                for f in 0..features {
                    result[offset + f] = self_data[offset + f] + other_data[f];
                }
            }

            let mut output = Tensor::new(result, &self.shape);

            if self.requires_grad || other.requires_grad {
                output.requires_grad = true;
                let a = self.clone();
                let b = other.clone();
                let out = output.clone();
                let (batch_size, features) = (batch_size, features);

                Tape::push_binary_op(self, other, &output, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        if a.requires_grad {
                            ops::accumulate_grad(&a, gout);
                        }

                        if b.requires_grad {
                            let mut slot = b.grad.write().unwrap();
                            if slot.is_none() {
                                *slot = Some(vec![0.0; features]);
                            }
                            let gb = slot.as_mut().unwrap();
                            for batch in 0..batch_size {
                                let base = batch * features;
                                for f in 0..features {
                                    gb[f] += gout[base + f];
                                }
                            }
                        }
                    }
                });
            }

            output
        } else {
            panic!(
                "Unsupported broadcasting shapes: {:?} and {:?}",
                self.shape, other.shape
            );
        }
    }

    /// Row-wise broadcasted subtraction: [B,C] - [B,1] -> [B,C]
    pub fn sub_broadcast_rows(&self, other: &Tensor) -> Tensor {
        // Fast path: exact same shape
        if self.shape == other.shape {
            return self - other;
        }

        // Expect [B,C] - [B,1]
        assert!(
            self.shape.len() == 2
                && other.shape.len() == 2
                && self.shape[0] == other.shape[0]
                && other.shape[1] == 1,
            "Unsupported broadcasting shapes for sub_broadcast_rows: {:?} - {:?}",
            self.shape,
            other.shape
        );

        let (b, c) = (self.shape[0], self.shape[1]);
        let a_data = self.data();
        let r_data = other.data();

        // Forward
        let mut out_data = vec![0.0; a_data.len()];
        for row in 0..b {
            let base = row * c;
            let r = r_data[row];
            for col in 0..c {
                out_data[base + col] = a_data[base + col] - r;
            }
        }
        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a = self.clone();
            let r = other.clone();
            let o = out.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout) = o.grad.read().unwrap().as_ref() {
                    // dL/dA = gout
                    if a.requires_grad {
                        ops::accumulate_grad(&a, gout);
                    }
                    // dL/dR[row] = -sum_c gout[row, c]
                    if r.requires_grad {
                        let (b, c) = (a.shape[0], a.shape[1]);
                        let mut grad_r = vec![0.0; b];
                        for row in 0..b {
                            let base = row * c;
                            let mut s = 0.0;
                            for col in 0..c {
                                s += gout[base + col];
                            }
                            grad_r[row] -= s;
                        }
                        ops::accumulate_grad(&r, &grad_r);
                    }
                }
            });
        }

        out
    }

    /// Row-wise broadcasted division: [B,C] / [B,1] -> [B,C]
    pub fn div_broadcast_rows(&self, other: &Tensor) -> Tensor {
        // Fast path: exact same shape
        if self.shape == other.shape {
            return self / other;
        }

        // Expect [B,C] / [B,1]
        assert!(
            self.shape.len() == 2
                && other.shape.len() == 2
                && self.shape[0] == other.shape[0]
                && other.shape[1] == 1,
            "Unsupported broadcasting shapes for div_broadcast_rows: {:?} / {:?}",
            self.shape,
            other.shape
        );

        let (b, c) = (self.shape[0], self.shape[1]);
        let a_data = self.data();
        let r_data = other.data();

        // Forward
        let mut out_data = vec![0.0; a_data.len()];
        for row in 0..b {
            let base = row * c;
            let r = r_data[row];
            for col in 0..c {
                out_data[base + col] = a_data[base + col] / r;
            }
        }
        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a = self.clone();
            let r = other.clone();
            let o = out.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout) = o.grad.read().unwrap().as_ref() {
                    // dL/dA[row,col] = gout[row,col] / R[row]
                    if a.requires_grad {
                        let (b, c) = (a.shape[0], a.shape[1]);
                        let r_data = r.data();
                        let mut grad_a = vec![0.0; b * c];
                        for row in 0..b {
                            let base = row * c;
                            let rv = r_data[row];
                            for col in 0..c {
                                grad_a[base + col] = gout[base + col] / rv;
                            }
                        }
                        ops::accumulate_grad(&a, &grad_a);
                    }
                    // dL/dR[row] = -sum_c( gout[row,c] * A[row,c] ) / R[row]^2
                    if r.requires_grad {
                        let (b, c) = (a.shape[0], a.shape[1]);
                        let a_data = a.data();
                        let r_data = r.data();
                        let mut grad_r = vec![0.0; b];
                        for row in 0..b {
                            let base = row * c;
                            let rv = r_data[row];
                            let mut s = 0.0;
                            for col in 0..c {
                                s += gout[base + col] * a_data[base + col];
                            }
                            grad_r[row] = -s / (rv * rv);
                        }
                        ops::accumulate_grad(&r, &grad_r);
                    }
                }
            });
        }

        out
    }

    pub fn mean(&self) -> Tensor {
        let data = self.data();
        let sum: f32 = data.iter().sum();
        let mean_val = sum / data.len() as f32;

        let mut output = Tensor::scalar(mean_val);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let n = data.len() as f32;

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    let g_each = gout[0] / n;
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; input.data().len()]);
                    }
                    for gi in slot.as_mut().unwrap().iter_mut() {
                        *gi += g_each;
                    }
                }
            });
        }

        output
    }

    /// Reshape tensor to new shape (must have same total elements)
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let total_elements: usize = shape.iter().product();
        assert_eq!(
            self.data().len(),
            total_elements,
            "Cannot reshape tensor of size {} to shape {:?}",
            self.data().len(),
            shape
        );

        // Reshaping doesn't change data, just the view
        let data = self.data().clone();
        let mut output = Tensor::new(data, shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            // let orig_shape = self.shape.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // Gradient just needs to be reshaped back
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; gout.len()]);
                    }
                    let gin = slot.as_mut().unwrap();
                    // Simply copy gradients (reshape doesn't change values)
                    for (g_in, g_out) in gin.iter_mut().zip(gout.iter()) {
                        *g_in += g_out;
                    }
                }
            });
        }

        output
    }

    /// Flatten tensor starting from start_dim
    pub fn flatten(&self, start_dim: usize) -> Tensor {
        assert!(start_dim < self.shape.len(), "start_dim out of bounds");

        let mut new_shape = Vec::new();

        // Keep dimensions before start_dim
        for i in 0..start_dim {
            new_shape.push(self.shape[i]);
        }

        // Flatten remaining dimensions
        let flattened_size: usize = self.shape[start_dim..].iter().product();
        new_shape.push(flattened_size);

        self.reshape(&new_shape)
    }

    /// Remove dimensions of size 1
    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        let new_shape: Vec<usize> = if let Some(d) = dim {
            assert!(d < self.shape.len(), "Dimension out of bounds");
            assert_eq!(self.shape[d], 1, "Can only squeeze dimensions of size 1");
            self.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != d)
                .map(|(_, &s)| s)
                .collect()
        } else {
            // Squeeze all dimensions of size 1
            self.shape.iter().filter(|&&s| s != 1).copied().collect()
        };

        self.reshape(&new_shape)
    }

    /// Add a dimension of size 1 at the specified position
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        assert!(dim <= self.shape.len(), "Dimension out of bounds");

        let mut new_shape = self.shape.to_vec();
        new_shape.insert(dim, 1);

        self.reshape(&new_shape)
    }

    /// Sum over dimensions
    pub fn sum(&self, dim: Option<usize>, keepdim: bool) -> Tensor {
        let data = self.data();

        if let Some(d) = dim {
            assert!(d < self.shape.len(), "Dimension {} out of bounds", d);

            // Calculate output shape
            let mut out_shape = self.shape.to_vec();
            if keepdim {
                out_shape[d] = 1;
            } else {
                out_shape.remove(d);
            }

            // Calculate strides for summation
            let mut strides = vec![1; self.shape.len()];
            for i in (0..self.shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * self.shape[i + 1];
            }

            let out_size: usize = out_shape.iter().product();
            let mut result = vec![0.0; out_size];

            // Perform summation over dimension d
            // let dim_size = self.shape[d];
            // let dim_stride = strides[d];

            for i in 0..data.len() {
                // Calculate which output element this contributes to
                let mut idx = i;
                let mut out_idx = 0;
                let mut multiplier = 1;

                for j in (0..self.shape.len()).rev() {
                    let coord = idx % self.shape[j];
                    idx /= self.shape[j];

                    if j != d {
                        let out_j = if j > d && !keepdim { j - 1 } else { j };
                        if out_j < out_shape.len() {
                            out_idx += coord * multiplier;
                            multiplier *= out_shape[out_j];
                        }
                    }
                }

                result[out_idx] += data[i];
            }

            let mut output = Tensor::new(result, &out_shape);

            if self.requires_grad {
                output.requires_grad = true;
                let input = self.clone();
                let out = output.clone();
                let in_shape = self.shape.clone();
                let out_shape_captured = out_shape.clone();
                let keepdim = keepdim;
                let d = d;

                Tape::push_unary_op(self, &output, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        let mut slot = input.grad.write().unwrap();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; input.data().len()]);
                        }
                        let gin = slot.as_mut().unwrap();

                        // Gradient is broadcasted back
                        // Each element that was summed gets the same gradient
                        for i in 0..gin.len() {
                            // Calculate which output element this came from
                            let mut idx = i;
                            let mut out_idx = 0;
                            let mut multiplier = 1;

                            for j in (0..in_shape.len()).rev() {
                                let coord = idx % in_shape[j];
                                idx /= in_shape[j];

                                if j != d {
                                    let out_j = if j > d && !keepdim { j - 1 } else { j };
                                    if out_j < out_shape_captured.len() {
                                        out_idx += coord * multiplier;
                                        multiplier *= out_shape_captured[out_j];
                                    }
                                }
                            }

                            debug_assert!(
                                out_idx < gout.len(),
                                "sum backward: out_idx {} out of bounds for gout len {}",
                                out_idx,
                                gout.len()
                            );
                            gin[i] += gout[out_idx];
                        }
                    }
                });
            }

            output
        } else {
            // Sum all elements
            let sum_val: f32 = data.iter().sum();
            let mut output = Tensor::scalar(sum_val);

            if self.requires_grad {
                output.requires_grad = true;
                let input = self.clone();
                let out = output.clone();
                let size = data.len();

                Tape::push_unary_op(self, &output, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        // Each element gets the same gradient
                        let grad_val = gout[0];
                        let grad_vec = vec![grad_val; size];
                        ops::accumulate_grad(&input, &grad_vec);
                    }
                });
            }

            output
        }
    }

    /// Max over dimensions, returns (values, indices)
    pub fn max(&self, dim: Option<usize>) -> (Tensor, Tensor) {
        let data = self.data();

        if let Some(d) = dim {
            assert!(d < self.shape.len(), "Dimension {} out of bounds", d);

            // Calculate output shape
            let mut out_shape = self.shape.to_vec();
            out_shape[d] = 1;

            let out_size: usize = out_shape.iter().product();
            let mut max_values = vec![f32::NEG_INFINITY; out_size];
            let mut max_indices = vec![0.0; out_size];

            // Calculate strides
            let mut strides = vec![1; self.shape.len()];
            for i in (0..self.shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * self.shape[i + 1];
            }

            // Find max values and indices
            for i in 0..data.len() {
                let mut idx = i;
                let mut out_idx = 0;
                let mut dim_idx = 0;
                let mut multiplier = 1;

                for j in (0..self.shape.len()).rev() {
                    let coord = idx % self.shape[j];
                    idx /= self.shape[j];

                    if j == d {
                        dim_idx = coord;
                    } else {
                        out_idx += coord * multiplier;
                        multiplier *= if j < d { self.shape[j] } else { 1 };
                    }
                }

                out_idx = out_idx.min(out_size - 1);

                if data[i] > max_values[out_idx] {
                    max_values[out_idx] = data[i];
                    max_indices[out_idx] = dim_idx as f32;
                }
            }

            let mut values = Tensor::new(max_values, &out_shape);
            let indices = Tensor::new(max_indices.clone(), &out_shape);

            if self.requires_grad {
                values.requires_grad = true;
                let input = self.clone();
                let out = values.clone();
                let in_shape = self.shape.clone();
                let out_shape = out_shape.clone();
                let d = d;

                Tape::push_unary_op(self, &values, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        let mut slot = input.grad.write().unwrap();
                        let in_size: usize = in_shape.iter().product();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; in_size]);
                        }
                        let gin = slot.as_mut().unwrap();

                        // For each output element, scatter gradient to the argmax position
                        let out_size = gout.len();
                        for oi in 0..out_size {
                            let dim_idx = max_indices[oi] as usize;

                            // Convert flat out_idx to coordinates in out_shape
                            // Compute input strides
                            let mut in_strides = vec![1usize; in_shape.len()];
                            for j in (0..in_shape.len() - 1).rev() {
                                in_strides[j] = in_strides[j + 1] * in_shape[j + 1];
                            }

                            // Compute output strides
                            let mut out_strides = vec![1usize; out_shape.len()];
                            for j in (0..out_shape.len() - 1).rev() {
                                out_strides[j] = out_strides[j + 1] * out_shape[j + 1];
                            }

                            // Decompose out_idx into coords, replace dim d with argmax
                            let mut in_flat = 0;
                            let mut remaining = oi;
                            for j in 0..out_shape.len() {
                                let coord = remaining / out_strides[j];
                                remaining %= out_strides[j];

                                let in_coord = if j == d { dim_idx } else { coord };
                                in_flat += in_coord * in_strides[j];
                            }

                            gin[in_flat] += gout[oi];
                        }
                    }
                });
            }

            (values, indices)
        } else {
            // Global max
            let (max_val, max_idx) = data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, &v)| (v, i))
                .unwrap_or((0.0, 0));

            let mut values = Tensor::scalar(max_val);
            let indices = Tensor::scalar(max_idx as f32);

            if self.requires_grad {
                values.requires_grad = true;
                let input = self.clone();
                let out = values.clone();
                let max_idx = max_idx;

                Tape::push_unary_op(self, &values, move || {
                    if let Some(gout) = out.grad.read().unwrap().as_ref() {
                        let mut slot = input.grad.write().unwrap();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; input.data().len()]);
                        }
                        let gin = slot.as_mut().unwrap();
                        gin[max_idx] += gout[0];
                    }
                });
            }

            (values, indices)
        }
    }

    /// Argmax - returns indices of maximum values
    pub fn argmax(&self, dim: Option<usize>) -> Tensor {
        self.max(dim).1
    }

    /// Exponential function (SIMD optimized)
    pub fn exp(&self) -> Tensor {
        let data = self.data().clone();
        let mut result = vec![0.0; data.len()];

        // SIMD exp using approximation for better performance
        // For accurate exp, we use scalar for now (can be optimized with SIMD exp approximation)
        for (i, &x) in data.iter().enumerate() {
            result[i] = x.exp();
        }

        let mut output = Tensor::new(result.clone(), &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let exp_values = result;

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // d/dx e^x = e^x
                    let exp_x = &exp_values;
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; exp_x.len()]);
                    }
                    let gin = slot.as_mut().unwrap();

                    // Use SIMD for gradient computation
                    unsafe {
                        let mut temp = vec![0.0; gin.len()];
                        crate::tensor::simd::mul_f32_simd(gout, &exp_x, &mut temp);
                        let gin_buf = gin.clone();
                        let mut result_buf = vec![0.0; gin_buf.len()];
                        crate::tensor::simd::add_f32_simd(&gin_buf, &temp, &mut result_buf);
                        gin.copy_from_slice(&result_buf);
                    }
                }
            });
        }

        output
    }

    /// Natural logarithm
    pub fn log(&self) -> Tensor {
        let data = self.data();
        let mut result = vec![0.0; data.len()];

        for (i, &x) in data.iter().enumerate() {
            result[i] = x.ln();
        }

        let mut output = Tensor::new(result, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // d/dx ln(x) = 1/x
                    let x = input.data();
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; x.len()]);
                    }
                    let gin = slot.as_mut().unwrap();

                    for ((gi, &g), &x_val) in gin.iter_mut().zip(gout.iter()).zip(x.iter()) {
                        *gi += g / x_val;
                    }
                }
            });
        }

        output
    }

    /// Power function
    pub fn pow(&self, exp: f32) -> Tensor {
        let data = self.data();
        let mut result = vec![0.0; data.len()];

        for (i, &x) in data.iter().enumerate() {
            result[i] = x.powf(exp);
        }

        let mut output = Tensor::new(result, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let exp = exp;

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // d/dx x^n = n * x^(n-1)
                    let x = input.data();
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; x.len()]);
                    }
                    let gin = slot.as_mut().unwrap();

                    for ((gi, &g), &x_val) in gin.iter_mut().zip(gout.iter()).zip(x.iter()) {
                        *gi += g * exp * x_val.powf(exp - 1.0);
                    }
                }
            });
        }

        output
    }

    /// Square root
    pub fn sqrt(&self) -> Tensor {
        self.pow(0.5)
    }

    /// View tensor with new shape (alias for reshape)
    pub fn view(&self, shape: &[usize]) -> Tensor {
        self.reshape(shape)
    }

    /// Efficient 2D convolution using im2col + GEMM approach
    /// Input: [N, C_in, H, W], Weight: [C_out, C_in, K_h, K_w]
    /// Output: [N, C_out, H_out, W_out]
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D: [N, C_in, H, W]");
        assert_eq!(
            weight.shape.len(),
            4,
            "Weight must be 4D: [C_out, C_in, K_h, K_w]"
        );

        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, c_in_w, k_h, k_w) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );

        assert_eq!(
            c_in, c_in_w,
            "Input and weight channel dimensions must match"
        );

        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let (dil_h, dil_w) = dilation;

        // Calculate output dimensions
        let h_out = (h_in + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1;

        // im2col transformation
        let k = c_in * k_h * k_w;
        let col_matrix = self.im2col_optimized(k_h, k_w, stride, padding, dilation); // [NW, K]

        // reshape weights as [K, C_out] (no transpose needed)
        let weight_reshaped = weight.reshape(&[k, c_out]);

        // now: [NW, K] @ [K, C_out] -> [NW, C_out], zero extra copy
        let output_2d = col_matrix.matmul(&weight_reshaped);
        // col_matrix shape: [N * H_out * W_out, C_in * K_h * K_w]

        // Reshape weight for GEMM: [C_out, C_in * K_h * K_w]
        // let weight_reshaped = weight.reshape(&[c_out, c_in * k_h * k_w]);

        // // GEMM: [N * H_out * W_out, C_out] = [N * H_out * W_out, C_in * K_h * K_w] @ [C_in * K_h * K_w, C_out]
        // let output_2d = col_matrix.matmul(&weight_reshaped.transpose());

        // Reshape back to 4D: [N, H_out, W_out, C_out] -> [N, C_out, H_out, W_out]
        let mut output = output_2d.reshape(&[n, h_out, w_out, c_out]);
        output = output.transpose_4d(&[0, 3, 1, 2]); // NHWC -> NCHW

        // Add bias if provided
        if let Some(b) = bias {
            assert_eq!(b.shape(), &[c_out], "Bias must be 1D with C_out elements");
            output = output.add_bias_4d(b);
        }

        // Register with Tape system for autograd
        if self.requires_grad || weight.requires_grad || bias.map_or(false, |b| b.requires_grad) {
            output.requires_grad = true;
            let input = self.clone();
            let weight_t = weight.clone();
            let bias_t = bias.cloned();
            let out = output.clone();
            let (_stride, _padding, _dilation) = (stride, padding, dilation);

            Tape::push_binary_op(self, weight, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    // CRITICAL FIX: Proper convolution gradient computation
                    
                    // Gradient w.r.t. input: Use convolution transpose (deconvolution)
                    if input.requires_grad {
                        let input_grad = Self::conv_transpose2d_for_grad(
                            gout, &weight_t, n, c_in, h_in, w_in, c_out, h_out, w_out,
                            k_h, k_w, stride_h, stride_w, pad_h, pad_w
                        );
                        ops::accumulate_grad(&input, &input_grad);
                    }

                    // Gradient w.r.t. weight: Use proper convolution with input
                    if weight_t.requires_grad {
                        let weight_grad = Self::conv2d_weight_grad(
                            &input, gout, n, c_in, h_in, w_in, c_out, h_out, w_out,
                            k_h, k_w, stride_h, stride_w, pad_h, pad_w
                        );
                        ops::accumulate_grad(&weight_t, &weight_grad);
                    }

                    // Gradient w.r.t. bias: sum over spatial dimensions (this was correct)
                    if let Some(bias_tensor) = bias_t.as_ref() {
                        if bias_tensor.requires_grad {
                            let mut bias_grad = vec![0.0; c_out];
                            for n_idx in 0..n {
                                for c in 0..c_out {
                                    for h in 0..h_out {
                                        for w in 0..w_out {
                                            let idx = n_idx * c_out * h_out * w_out + c * h_out * w_out + h * w_out + w;
                                            if idx < gout.len() {
                                                bias_grad[c] += gout[idx];
                                            }
                                        }
                                    }
                                }
                            }
                            
                            ops::accumulate_grad(bias_tensor, &bias_grad);
                        }
                    }
                }
            });
        }

        output
    }


    pub fn conv2d_direct_3x3(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor {
        use rayon::prelude::*;

        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, _, _, _) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );

        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        let h_out = (h_in + 2 * pad_h - 2) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - 2) / stride_w + 1;

        let input_data = self.data().clone();
        let weight_data = weight.data().clone();
        let mut output = vec![0.0; n * c_out * h_out * w_out];

        // Parallel over batch and output channels
        output
            .par_chunks_mut(h_out * w_out)
            .enumerate()
            .for_each(|(idx, out_spatial)| {
                let batch = idx / c_out;
                let out_ch = idx % c_out;

                // For each output position
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;

                        // Convolution kernel - unrolled for 3x3
                        for in_ch in 0..c_in {
                            let weight_base = (out_ch * c_in + in_ch) * 9;
                            let input_base = batch * c_in * h_in * w_in + in_ch * h_in * w_in;

                            // Unrolled 3x3 loop with bounds checking hoisted
                            for kh in 0..3 {
                                let ih = oh * stride_h + kh;
                                if ih < pad_h || ih >= h_in + pad_h {
                                    continue;
                                }
                                let ih_idx = ih - pad_h;

                                for kw in 0..3 {
                                    let iw = ow * stride_w + kw;
                                    if iw < pad_w || iw >= w_in + pad_w {
                                        continue;
                                    }
                                    let iw_idx = iw - pad_w;

                                    let in_idx = input_base + ih_idx * w_in + iw_idx;
                                    let w_idx = weight_base + kh * 3 + kw;

                                    sum += input_data[in_idx] * weight_data[w_idx];
                                }
                            }
                        }

                        out_spatial[oh * w_out + ow] = sum;
                    }
                }
            });

        // Add bias if provided
        if let Some(b) = bias {
            let bias_data = b.data().clone();
            output
                .par_chunks_mut(h_out * w_out)
                .enumerate()
                .for_each(|(idx, out_spatial)| {
                    let out_ch = idx % c_out;
                    let bias_val = bias_data[out_ch];
                    for v in out_spatial.iter_mut() {
                        *v += bias_val;
                    }
                });
        }

        Tensor::new(output, &[n, c_out, h_out, w_out])
    }

    /// Fused convolution + ReLU operation for better performance
    pub fn conv2d_relu(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor {
        let conv_out = self.conv2d(weight, bias, stride, padding, dilation);
        conv_out.relu()
    }

    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Tensor {
        use rayon::prelude::*;

        assert_eq!(self.shape.len(), 4, "Input must be 4D: [N, C, H, W]");

        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride.unwrap_or(kernel_size);
        let (pad_h, pad_w) = padding;

        let h_out = (h_in + 2 * pad_h - k_h) / s_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / s_w + 1;
        let out_spatial = h_out * w_out;

        // read-only input slice (hold the guard just to obtain a slice, then drop it)
        let data_guard = self.data();
        let x: &[f32] = &data_guard;

        // forward: outputs + argmax (absolute input indices)
        let mut output_data = vec![f32::NEG_INFINITY; n * c * out_spatial];
        let mut argmax = vec![0usize; n * c * out_spatial];

        // We parallelize across (b,c) chunks, which are disjoint in out/argmax.
        output_data
            .par_chunks_mut(out_spatial)
            .enumerate()
            .zip(argmax.par_chunks_mut(out_spatial))
            .for_each(|((bc, out_chunk), arg_chunk)| {
                let b = bc / c;
                let ch = bc % c;

                let in_base = b * c * h_in * w_in + ch * h_in * w_in;

                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut best = f32::NEG_INFINITY;
                        let mut best_idx = in_base; // any valid default

                        // window scan
                        for kh in 0..k_h {
                            let ih_pad = oh * s_h + kh;
                            if ih_pad < pad_h || ih_pad >= h_in + pad_h {
                                continue;
                            }
                            let ih = ih_pad - pad_h;

                            for kw in 0..k_w {
                                let iw_pad = ow * s_w + kw;
                                if iw_pad < pad_w || iw_pad >= w_in + pad_w {
                                    continue;
                                }
                                let iw = iw_pad - pad_w;

                                let idx = in_base + ih * w_in + iw;
                                let v = unsafe { *x.get_unchecked(idx) }; // bounds are checked by our math above
                                if v > best {
                                    best = v;
                                    best_idx = idx;
                                }
                            }
                        }

                        let oidx = oh * w_out + ow;
                        // these are per-(b,c) chunk slices; no conflicts across threads
                        out_chunk[oidx] = best;
                        arg_chunk[oidx] = best_idx;
                    }
                }
            });

        drop(data_guard); // explicitly drop guard; x stays valid as an immutable slice reference

        let mut output = Tensor::new(output_data, &[n, c, h_out, w_out]);

        if self.requires_grad {
            output.requires_grad = true;

            // capture what we need; avoid touching input again during backprop
            let out_clone = output.clone();
            let input = self.clone();
            let argmax_capture = argmax; // move into closure
            let (n, c, h_in, w_in, h_out, w_out) = (n, c, h_in, w_in, h_out, w_out);

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out_clone.grad.read().unwrap().as_ref() {
                    // allocate gin once
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; n * c * h_in * w_in]);
                    }
                    // safe mutable view
                    let gin = slot.as_mut().unwrap();

                    // shard gin/gout/argmax by (b,c) — disjoint slices, so no atomics needed
                    let out_spatial = h_out * w_out;
                    gin.par_chunks_mut(h_in * w_in)
                        .enumerate()
                        .zip(gout.par_chunks(out_spatial))
                        .zip(argmax_capture.par_chunks(out_spatial))
                        .for_each(|(((bc, gin_chunk), gout_chunk), arg_chunk)| {
                            // zero this (b,c) slice before accumulation (since we reuse memory across backprops)
                            // note: if your engine ensures fresh zeroed grads, you can skip this memset.
                            for v in gin_chunk.iter_mut() {
                                *v = 0.0;
                            }

                            // scatter-add: each output position contributes to its max position
                            // all writes stay within this gin_chunk; no inter-thread races.
                            for o in 0..out_spatial {
                                let in_abs = unsafe { *arg_chunk.get_unchecked(o) };
                                // convert absolute index to (b,c)-local index
                                // bc = b*c + c_idx; compute its base to turn absolute->local
                                let b = bc / c;
                                let ch = bc % c;
                                let base = b * c * h_in * w_in + ch * h_in * w_in;
                                let local = in_abs - base;
                                debug_assert!(local < gin_chunk.len());
                                gin_chunk[local] += unsafe { *gout_chunk.get_unchecked(o) };
                            }
                        });
                }
            });
        }

        output
    }

    /// 2D Average Pooling
    pub fn avg_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Tensor {
        use rayon::prelude::*;

        assert_eq!(self.shape.len(), 4, "Input must be 4D: [N, C, H, W]");

        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride.unwrap_or(kernel_size);
        let (pad_h, pad_w) = padding;

        let h_out = (h_in + 2 * pad_h - k_h) / s_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / s_w + 1;
        let out_spatial = h_out * w_out;

        // read-only input slice (drop the lock immediately)
        let data_guard = self.data();
        let x: &[f32] = &data_guard;

        let mut output_data = vec![0.0; n * c * out_spatial];
        let pool_size = (k_h * k_w) as f32;

        // Parallelize across (b,c) chunks: disjoint writes, no atomics needed.
        output_data
            .par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(|(bc, out_chunk)| {
                let b = bc / c;
                let ch = bc % c;

                let in_base = b * c * h_in * w_in + ch * h_in * w_in;

                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;

                        // window sum with bounds checks hoisted; use unchecked inside
                        for kh in 0..k_h {
                            let ih_pad = oh * s_h + kh;
                            if ih_pad < pad_h || ih_pad >= h_in + pad_h {
                                continue;
                            }
                            let ih = ih_pad - pad_h;

                            for kw in 0..k_w {
                                let iw_pad = ow * s_w + kw;
                                if iw_pad < pad_w || iw_pad >= w_in + pad_w {
                                    continue;
                                }
                                let iw = iw_pad - pad_w;

                                let idx = in_base + ih * w_in + iw;
                                // safety: idx computed within bounds by the checks above
                                sum += unsafe { *x.get_unchecked(idx) };
                            }
                        }

                        out_chunk[oh * w_out + ow] = sum / pool_size;
                    }
                }
            });

        drop(data_guard); // release read guard

        let mut output = Tensor::new(output_data, &[n, c, h_out, w_out]);

        if self.requires_grad {
            output.requires_grad = true;

            let out_clone = output.clone();
            let input = self.clone();
            let (n, c, h_in, w_in, h_out, w_out) = (n, c, h_in, w_in, h_out, w_out);
            let (k_h, k_w, s_h, s_w, pad_h, pad_w) = (k_h, k_w, s_h, s_w, pad_h, pad_w);
            let out_spatial = out_spatial;
            let pool_size = pool_size;

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out_clone.grad.read().unwrap().as_ref() {
                    // allocate or reuse gin
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; n * c * h_in * w_in]);
                    }
                    let gin = slot.as_mut().unwrap();

                    // shard (b,c) disjoint regions; we only add to our slice, so no races.
                    gin.par_chunks_mut(h_in * w_in)
                        .enumerate()
                        .zip(gout.par_chunks(out_spatial))
                        .for_each(|((bc, gin_chunk), gout_chunk)| {
                            let _b = bc / c;
                            let _ch = bc % c;

                            // we accumulate into this chunk only; do NOT zero if engine expects accumulation
                            // (your code already created zeros on first touch above; subsequent ops add onto it)

                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let grad_val =
                                        unsafe { *gout_chunk.get_unchecked(oh * w_out + ow) }
                                            / pool_size;

                                    for kh in 0..k_h {
                                        let ih_pad = oh * s_h + kh;
                                        if ih_pad < pad_h || ih_pad >= h_in + pad_h {
                                            continue;
                                        }
                                        let ih = ih_pad - pad_h;

                                        for kw in 0..k_w {
                                            let iw_pad = ow * s_w + kw;
                                            if iw_pad < pad_w || iw_pad >= w_in + pad_w {
                                                continue;
                                            }
                                            let iw = iw_pad - pad_w;

                                            // local index within this (b,c) gin chunk
                                            let local = ih * w_in + iw;
                                            // safety: ih<i_h, iw<w_in
                                            unsafe {
                                                *gin_chunk.get_unchecked_mut(local) += grad_val;
                                            }
                                        }
                                    }
                                }
                            }
                        });
                }
            });
        }

        output
    }

    /// SIMD-optimized im2col transformation
    fn im2col_optimized(
        &self,
        k_h: usize,
        k_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Tensor {
        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let (dil_h, dil_w) = dilation;

        let h_out = (h_in + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1;

        let col_size = c * k_h * k_w;
        let num_windows = n * h_out * w_out;
        let mut col_data = vec![0.0; num_windows * col_size];

        let data = self.data().clone(); // Clone once to avoid lock contention

        // Special cases remain the same
        if k_h == 3 && k_w == 3 && stride_h == 1 && stride_w == 1 && dil_h == 1 && dil_w == 1 {
            // Parallelize the 3x3 case
            self.im2col_3x3_stride1_parallel(
                &data,
                &mut col_data,
                n,
                c,
                h_in,
                w_in,
                h_out,
                w_out,
                pad_h,
                pad_w,
            );
        } else if k_h == 1 && k_w == 1 && stride_h == 1 && stride_w == 1 {
            // 1x1 with stride=1 is already optimal (no spatial downsampling)
            self.im2col_1x1(&data, &mut col_data, n, c, h_in, w_in, h_out, w_out);
        } else {
            // Parallelize general case
            self.im2col_general_simd(
                &data,
                &mut col_data,
                n,
                c,
                h_in,
                w_in,
                h_out,
                w_out,
                k_h,
                k_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dil_h,
                dil_w,
            );
        }

        Tensor::new(col_data, &[num_windows, col_size])
    }

    fn im2col_3x3_stride1_parallel(
        &self,
        input: &[f32],
        output: &mut [f32],
        _n: usize,
        c: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
        pad_h: usize,
        pad_w: usize,
    ) {
        let col_size = c * 9;

        // Parallel over batches and output positions
        output
            .par_chunks_mut(col_size)
            .enumerate()
            .for_each(|(window_idx, out_chunk)| {
                let batch = window_idx / (h_out * w_out);
                let pos = window_idx % (h_out * w_out);
                let out_h = pos / w_out;
                let out_w = pos % w_out;

                let batch_offset = batch * c * h_in * w_in;

                for ch in 0..c {
                    let ch_offset = ch * 9;

                    // Unrolled 3x3 kernel with SIMD-friendly access pattern
                    for k_row in 0..3 {
                        let in_h = out_h + k_row;
                        let in_h_valid = in_h >= pad_h && in_h < h_in + pad_h;
                        let in_h_idx = if in_h_valid { in_h - pad_h } else { 0 };

                        for k_col in 0..3 {
                            let in_w = out_w + k_col;
                            let idx = ch_offset + k_row * 3 + k_col;

                            if in_h_valid && in_w >= pad_w && in_w < w_in + pad_w {
                                let in_w_idx = in_w - pad_w;
                                let in_idx =
                                    batch_offset + ch * h_in * w_in + in_h_idx * w_in + in_w_idx;
                                out_chunk[idx] = input[in_idx];
                            } else {
                                out_chunk[idx] = 0.0;
                            }
                        }
                    }
                }
            });
    }

    /// Ultra-fast 1x1 convolution (just reshape)
    #[inline]
    fn im2col_1x1(
        &self,
        input: &[f32],
        output: &mut [f32],
        _n: usize,
        _c: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
    ) {
        // 1x1 conv is just a reshape - use SIMD memcpy
        assert_eq!(h_in, h_out);
        assert_eq!(w_in, w_out);

        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), output.as_mut_ptr(), input.len());
        }
    }

    /// General case with SIMD optimizations
    #[inline]
    fn im2col_general_simd(
        &self,
        input: &[f32],
        output: &mut [f32],
        n: usize,
        c: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
        k_h: usize,
        k_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dil_h: usize,
        dil_w: usize,
    ) {
        let col_size = c * k_h * k_w;

        // Vectorized when copying contiguous regions
        for batch in 0..n {
            for out_h in 0..h_out {
                for out_w in 0..w_out {
                    let window_idx = batch * h_out * w_out + out_h * w_out + out_w;
                    let col_base = window_idx * col_size;

                    for ch in 0..c {
                        for k_row in 0..k_h {
                            let in_h = out_h * stride_h + k_row * dil_h;

                            if in_h >= pad_h && in_h < h_in + pad_h {
                                let in_h_idx = in_h - pad_h;

                                // Try to copy entire rows when possible (SIMD opportunity)
                                let mut consecutive_count = 0;
                                let mut start_k_col = 0;

                                for k_col in 0..k_w {
                                    let in_w = out_w * stride_w + k_col * dil_w;

                                    if in_w >= pad_w
                                        && in_w < w_in + pad_w
                                        && consecutive_count == k_col - start_k_col
                                    {
                                        consecutive_count += 1;
                                    } else {
                                        // Copy accumulated consecutive elements
                                        if consecutive_count > 0 {
                                            self.copy_consecutive_elements(
                                                input,
                                                output,
                                                batch,
                                                ch,
                                                c,
                                                h_in,
                                                w_in,
                                                in_h_idx,
                                                out_w * stride_w + start_k_col * dil_w - pad_w,
                                                col_base
                                                    + ch * k_h * k_w
                                                    + k_row * k_w
                                                    + start_k_col,
                                                consecutive_count,
                                            );
                                        }

                                        // Handle non-consecutive element
                                        let col_idx =
                                            col_base + ch * k_h * k_w + k_row * k_w + k_col;

                                        if in_w >= pad_w && in_w < w_in + pad_w {
                                            let in_w_idx = in_w - pad_w;
                                            let in_idx = batch * c * h_in * w_in
                                                + ch * h_in * w_in
                                                + in_h_idx * w_in
                                                + in_w_idx;
                                            output[col_idx] = input[in_idx];
                                        }

                                        start_k_col = k_col + 1;
                                        consecutive_count = 0;
                                    }
                                }

                                // Handle remaining consecutive elements
                                if consecutive_count > 0 {
                                    self.copy_consecutive_elements(
                                        input,
                                        output,
                                        batch,
                                        ch,
                                        c,
                                        h_in,
                                        w_in,
                                        in_h_idx,
                                        out_w * stride_w + start_k_col * dil_w - pad_w,
                                        col_base + ch * k_h * k_w + k_row * k_w + start_k_col,
                                        consecutive_count,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// SIMD-optimized copy of consecutive elements
    #[inline]
    fn copy_consecutive_elements(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
        ch: usize,
        c_total: usize,
        h_in: usize,
        w_in: usize,
        in_h: usize,
        in_w_start: usize,
        out_start: usize,
        count: usize,
    ) {
        if count >= 8 && in_w_start + count <= w_in {
            // Use SIMD for larger copies
            let in_base =
                batch * c_total * h_in * w_in + ch * h_in * w_in + in_h * w_in + in_w_start;

            unsafe {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx") && count >= 8 {
                        use std::arch::x86_64::*;
                        let chunks = count / 8;
                        for i in 0..chunks {
                            let v = _mm256_loadu_ps(input.as_ptr().add(in_base + i * 8));
                            _mm256_storeu_ps(output.as_mut_ptr().add(out_start + i * 8), v);
                        }

                        // Handle remainder
                        for i in (chunks * 8)..count {
                            output[out_start + i] = input[in_base + i];
                        }
                        return;
                    }
                }

                // Fallback: vectorizable memcpy
                std::ptr::copy_nonoverlapping(
                    input.as_ptr().add(in_base),
                    output.as_mut_ptr().add(out_start),
                    count,
                );
            }
        } else {
            // Scalar copy for small counts
            for i in 0..count {
                let in_w = in_w_start + i;
                if in_w < w_in {
                    let in_idx =
                        batch * c_total * h_in * w_in + ch * h_in * w_in + in_h * w_in + in_w;
                    output[out_start + i] = input[in_idx];
                }
            }
        }
    }

    /// Helper: Add bias to 4D tensor (broadcast along channel dimension)
    fn add_bias_4d(&self, bias: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 4);
        assert_eq!(bias.shape.len(), 1);
        assert_eq!(self.shape[1], bias.shape[0]); // C_out

        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let self_data = self.data();
        let bias_data = bias.data();

        let mut result_data = vec![0.0; self_data.len()];

        for batch in 0..n {
            for channel in 0..c {
                let bias_val = bias_data[channel];
                let base_idx = batch * c * h * w + channel * h * w;

                for spatial in 0..(h * w) {
                    result_data[base_idx + spatial] = self_data[base_idx + spatial] + bias_val;
                }
            }
        }

        let mut output = Tensor::new(result_data, &self.shape);

        if self.requires_grad || bias.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let bias_t = bias.clone();
            let out = output.clone();
            let (n, c, h, w) = (n, c, h, w);

            Tape::push_binary_op(self, bias, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    if input.requires_grad {
                        ops::accumulate_grad(&input, gout);
                    }

                    if bias_t.requires_grad {
                        let mut slot = bias_t.grad.write().unwrap();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; c]);
                        }
                        let gb = slot.as_mut().unwrap();

                        // Sum gradients across N, H, W dimensions for each channel
                        for batch in 0..n {
                            for channel in 0..c {
                                let base_idx = batch * c * h * w + channel * h * w;
                                for spatial in 0..(h * w) {
                                    gb[channel] += gout[base_idx + spatial];
                                }
                            }
                        }
                    }
                }
            });
        }

        output
    }

    /// Helper: 4D tensor transpose for dimension reordering
    fn transpose_4d(&self, axes: &[usize; 4]) -> Tensor {
        assert_eq!(self.shape.len(), 4);

        let old_shape = &self.shape;
        let new_shape = [
            old_shape[axes[0]],
            old_shape[axes[1]],
            old_shape[axes[2]],
            old_shape[axes[3]],
        ];

        let data = self.data();
        let mut result_data = vec![0.0; data.len()];

        let (d0, d1, d2, d3) = (old_shape[0], old_shape[1], old_shape[2], old_shape[3]);

        for i0 in 0..d0 {
            for i1 in 0..d1 {
                for i2 in 0..d2 {
                    for i3 in 0..d3 {
                        let old_idx = i0 * d1 * d2 * d3 + i1 * d2 * d3 + i2 * d3 + i3;

                        let new_indices = [i0, i1, i2, i3];
                        let (n0, n1, n2, n3) = (
                            new_indices[axes[0]],
                            new_indices[axes[1]],
                            new_indices[axes[2]],
                            new_indices[axes[3]],
                        );

                        let new_idx = n0 * new_shape[1] * new_shape[2] * new_shape[3]
                            + n1 * new_shape[2] * new_shape[3]
                            + n2 * new_shape[3]
                            + n3;

                        result_data[new_idx] = data[old_idx];
                    }
                }
            }
        }

        Tensor::new(result_data, &new_shape)
    }

    /// Quantize tensor based on configuration
    pub fn quantize(&self, config: &QuantizationConfig) -> QuantizedTensor {
        if !config.enabled {
            // If quantization is disabled, return a "fake" quantized tensor
            return QuantizedTensor::Float16(Float16Tensor::from_f32_tensor(self));
        }

        match config.quant_type {
            crate::quantization::QuantizationType::Int8 => {
                QuantizedTensor::Int8(self.quantize_to_int8(config))
            }
            crate::quantization::QuantizationType::Int4 => {
                QuantizedTensor::Int4(self.quantize_to_int4(config))
            }
            crate::quantization::QuantizationType::Float16 => {
                QuantizedTensor::Float16(self.quantize_to_float16())
            }
            crate::quantization::QuantizationType::BFloat16 => {
                QuantizedTensor::BFloat16(self.quantize_to_bfloat16())
            }
            crate::quantization::QuantizationType::NF4 => {
                QuantizedTensor::NF4(self.quantize_to_nf4(config))
            }
        }
    }

    /// Quantize to int8
    fn quantize_to_int8(&self, config: &QuantizationConfig) -> Int8Tensor {
        let data = self.data();
        let (qmin, qmax) = config.compute_range().unwrap();

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &val in data.iter() {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        if min_val == max_val {
            min_val -= 0.1;
            max_val += 0.1;
        }

        let qrange = (qmax - qmin) as f32;
        let scale = (max_val - min_val) / qrange;
        let zero_point = qmin;

        let quantized_data: Vec<i8> = data
            .iter()
            .map(|&x| {
                let q = ((x - min_val) / scale).round() as i32 + qmin;
                q.clamp(qmin, qmax) as i8
            })
            .collect();

        Int8Tensor {
            data: Arc::new(RwLock::new(quantized_data)),
            shape: self.shape.clone(),
            scale,
            zero_point,
            min_val, // Store it!
        }
    }

    /// Quantize to int4 (packed representation)
    fn quantize_to_int4(&self, _config: &QuantizationConfig) -> Int4Tensor {
        unimplemented!(
            "Int4 quantization not yet implemented: packing two 4-bit values per byte needed"
        )
    }

    /// Convert to float16
    fn quantize_to_float16(&self) -> Float16Tensor {
        let data = self.data();
        let f16_data: Vec<u16> = data.iter().map(|&x| Self::f32_to_f16(x)).collect();

        Float16Tensor::new(f16_data, self.shape.clone())
    }

    /// Convert to bfloat16
    fn quantize_to_bfloat16(&self) -> BFloat16Tensor {
        unimplemented!("BFloat16 quantization not yet implemented: f32-to-bf16 conversion needed")
    }

    /// Quantize to NF4
    fn quantize_to_nf4(&self, _config: &QuantizationConfig) -> NF4Tensor {
        unimplemented!("NF4 quantization not yet implemented: NF4 lookup table and packing needed")
    }

    /// Convert f32 to f16 (IEEE 754 half precision)
    pub fn f32_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();

        // Extract sign, exponent, mantissa from f32
        let sign = (bits >> 31) & 0x1;
        let exponent = (bits >> 23) & 0xFF;
        let mantissa = bits & 0x7FFFFF;

        // Handle special cases
        if exponent == 0xFF {
            // Infinity or NaN
            let f16_mantissa = if mantissa != 0 { 0x200 } else { 0 }; // NaN gets non-zero mantissa
            return ((sign << 15) | (0x1F << 10) | f16_mantissa) as u16;
        }

        if exponent == 0 && mantissa == 0 {
            // Zero (positive or negative)
            return (sign << 15) as u16;
        }

        // Convert exponent from f32 bias (127) to f16 bias (15)
        let f16_exponent = (exponent as i32) - 127 + 15;

        // Handle overflow/underflow
        if f16_exponent >= 0x1F {
            // Overflow to infinity
            return ((sign << 15) | (0x1F << 10)) as u16;
        }

        if f16_exponent <= 0 {
            // Underflow - convert to denormalized or zero
            if f16_exponent < -10 {
                // Too small, round to zero
                return (sign << 15) as u16;
            }

            // Denormalized number
            let shift = 1 - f16_exponent;
            let f16_mantissa = (mantissa | 0x800000) >> (shift + 13);
            return ((sign << 15) | f16_mantissa) as u16;
        }

        // Normal number
        // Convert mantissa from 23 bits to 10 bits (with rounding)
        let f16_mantissa = (mantissa + 0x1000) >> 13; // Round to nearest

        ((sign << 15) | ((f16_exponent as u32) << 10) | (f16_mantissa as u32)) as u16
    }

    /// Helper function for convolution transpose (deconvolution) for input gradients
    fn conv_transpose2d_for_grad(
        grad_output: &[f32],
        weight: &Tensor,
        n: usize, c_in: usize, h_in: usize, w_in: usize,
        c_out: usize, h_out: usize, w_out: usize,
        k_h: usize, k_w: usize, stride_h: usize, stride_w: usize,
        _pad_h: usize, _pad_w: usize
    ) -> Vec<f32> {
        let mut grad_input = vec![0.0; n * c_in * h_in * w_in];
        let weight_data = weight.data();
        
        // For each output position, distribute gradient back to input positions
        for n_idx in 0..n {
            for c_out_idx in 0..c_out {
                for h_out_idx in 0..h_out {
                    for w_out_idx in 0..w_out {
                        let out_idx = n_idx * c_out * h_out * w_out + 
                                     c_out_idx * h_out * w_out + 
                                     h_out_idx * w_out + w_out_idx;
                        let grad_val = grad_output[out_idx];
                        
                        // Apply weight to distribute gradient back to input
                        for k_h_idx in 0..k_h {
                            for k_w_idx in 0..k_w {
                                let h_in_idx = h_out_idx * stride_h + k_h_idx;
                                let w_in_idx = w_out_idx * stride_w + k_w_idx;
                                
                                // Check bounds
                                if h_in_idx < h_in && w_in_idx < w_in {
                                    for c_in_idx in 0..c_in {
                                        let weight_idx = c_out_idx * c_in * k_h * k_w + 
                                                       c_in_idx * k_h * k_w + 
                                                       k_h_idx * k_w + k_w_idx;
                                        let input_idx = n_idx * c_in * h_in * w_in + 
                                                      c_in_idx * h_in * w_in + 
                                                      h_in_idx * w_in + w_in_idx;
                                        
                                        grad_input[input_idx] += grad_val * weight_data[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        grad_input
    }
    
    /// Helper function for weight gradients using convolution
    fn conv2d_weight_grad(
        input: &Tensor,
        grad_output: &[f32],
        n: usize, c_in: usize, h_in: usize, w_in: usize,
        c_out: usize, h_out: usize, w_out: usize,
        k_h: usize, k_w: usize, stride_h: usize, stride_w: usize,
        _pad_h: usize, _pad_w: usize
    ) -> Vec<f32> {
        let mut grad_weight = vec![0.0; c_out * c_in * k_h * k_w];
        let input_data = input.data();
        
        // For each weight position, accumulate gradients
        for c_out_idx in 0..c_out {
            for c_in_idx in 0..c_in {
                for k_h_idx in 0..k_h {
                    for k_w_idx in 0..k_w {
                        let weight_idx = c_out_idx * c_in * k_h * k_w + 
                                       c_in_idx * k_h * k_w + 
                                       k_h_idx * k_w + k_w_idx;
                        
                        // Accumulate gradients from all input positions
                        for n_idx in 0..n {
                            for h_out_idx in 0..h_out {
                                for w_out_idx in 0..w_out {
                                    let h_in_idx = h_out_idx * stride_h + k_h_idx;
                                    let w_in_idx = w_out_idx * stride_w + k_w_idx;
                                    
                                    // Check bounds
                                    if h_in_idx < h_in && w_in_idx < w_in {
                                        let input_idx = n_idx * c_in * h_in * w_in + 
                                                      c_in_idx * h_in * w_in + 
                                                      h_in_idx * w_in + w_in_idx;
                                        let out_idx = n_idx * c_out * h_out * w_out + 
                                                    c_out_idx * h_out * w_out + 
                                                    h_out_idx * w_out + w_out_idx;
                                        
                                        grad_weight[weight_idx] += input_data[input_idx] * grad_output[out_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        grad_weight
    }

    /// Convert f16 to f32 (IEEE 754 half precision)
    pub fn f16_to_f32(value: u16) -> f32 {
        let bits = value as u32;

        // Extract sign, exponent, mantissa from f16
        let sign = (bits >> 15) & 0x1;
        let exponent = (bits >> 10) & 0x1F;
        let mantissa = bits & 0x3FF;

        // Handle special cases
        if exponent == 0x1F {
            // Infinity or NaN
            let f32_exponent = 0xFF;
            let f32_mantissa = if mantissa != 0 { mantissa << 13 } else { 0 };
            let result_bits = (sign << 31) | (f32_exponent << 23) | f32_mantissa;
            return f32::from_bits(result_bits);
        }

        if exponent == 0 {
            if mantissa == 0 {
                // Zero (positive or negative)
                return f32::from_bits((sign << 31) as u32);
            }

            // Denormalized number - convert to normalized f32
            let mut exp = -14i32;
            let mut mant = mantissa;

            // Normalize mantissa
            while (mant & 0x400) == 0 {
                mant <<= 1;
                exp -= 1;
            }

            mant &= 0x3FF; // Remove leading 1
            let f32_exponent = ((exp + 127) as u32) & 0xFF;
            let f32_mantissa = mant << 13;
            let result_bits = (sign << 31) | (f32_exponent << 23) | f32_mantissa;
            return f32::from_bits(result_bits);
        }

        // Normal number
        // Convert exponent from f16 bias (15) to f32 bias (127)
        let f32_exponent = (exponent + 127 - 15) & 0xFF;
        let f32_mantissa = mantissa << 13;

        let result_bits = (sign << 31) | (f32_exponent << 23) | f32_mantissa;
        f32::from_bits(result_bits)
    }
}
