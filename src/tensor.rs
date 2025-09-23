use crate::{ops, tape::Tape, quantization::QuantizationConfig};
use smallvec::SmallVec;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
                add_f32_sse(a, b, out);
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
            let va = _mm_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm_loadu_ps(b.as_ptr().add(idx));
            let result = _mm_add_ps(va, vb);
            _mm_storeu_ps(out.as_mut_ptr().add(idx), result);
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
                mul_f32_sse(a, b, out);
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
            let va = _mm_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm_loadu_ps(b.as_ptr().add(idx));
            let result = _mm_mul_ps(va, vb);
            _mm_storeu_ps(out.as_mut_ptr().add(idx), result);
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
    data: Rc<RefCell<Vec<f32>>>,
    pub(crate) shape: SmallVec<[usize; 4]>,
    pub grad: Rc<RefCell<Option<Vec<f32>>>>,
    pub requires_grad: bool,
    pub tape_node: Cell<Option<usize>>,
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
    data: Rc<RefCell<Vec<i8>>>,
    shape: SmallVec<[usize; 4]>,
    scale: f32,
    zero_point: i32,
}

/// Int4 quantized tensor (packed representation - 2 values per byte)
#[derive(Clone, Debug)]
pub struct Int4Tensor {
    data: Rc<RefCell<Vec<u8>>>, // Packed: 2 int4 values per u8
    shape: SmallVec<[usize; 4]>,
    scale: f32,
    zero_point: i32,
}

/// Float16 quantized tensor
#[derive(Clone, Debug)]
pub struct Float16Tensor {
    data: Rc<RefCell<Vec<u16>>>, // Float16 stored as u16
    shape: SmallVec<[usize; 4]>,
}

/// BFloat16 quantized tensor
#[derive(Clone, Debug)]
pub struct BFloat16Tensor {
    data: Rc<RefCell<Vec<u16>>>, // BFloat16 stored as u16
    shape: SmallVec<[usize; 4]>,
}

/// NF4 quantized tensor
#[derive(Clone, Debug)]
pub struct NF4Tensor {
    data: Rc<RefCell<Vec<u8>>>, // Packed NF4 values
    shape: SmallVec<[usize; 4]>,
    scale: f32,
    zero_point: i32,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.data.borrow().as_slice())
            .field("shape", &self.shape)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.borrow().is_some())
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
            data: Rc::new(RefCell::new(data)),
            shape,
            scale,
            zero_point,
        }
    }
    
    pub fn dequantize(&self) -> Tensor {
        let data = self.data();
        let f32_data: Vec<f32> = data
            .iter()
            .map(|&q| (q as f32 - self.zero_point as f32) * self.scale)
            .collect();
        
        Tensor::new(f32_data, &self.shape)
    }
    
    pub fn data(&self) -> std::cell::Ref<'_, Vec<i8>> {
        self.data.borrow()
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
            data: Rc::new(RefCell::new(data)),
            shape,
            scale,
            zero_point,
        }
    }
    
    pub fn dequantize(&self) -> Tensor {
        // TODO: Implement int4 unpacking and dequantization
        // For now, return a dummy tensor
        let size: usize = self.shape.iter().product();
        Tensor::new(vec![0.0; size], &self.shape)
    }
    
    pub fn data(&self) -> std::cell::Ref<'_, Vec<u8>> {
        self.data.borrow()
    }
}

impl Float16Tensor {
    pub fn new(data: Vec<u16>, shape: SmallVec<[usize; 4]>) -> Self {
        Self {
            data: Rc::new(RefCell::new(data)),
            shape,
        }
    }
    
    /// Create a float16 tensor from an f32 tensor (for non-quantized case)
    pub fn from_f32_tensor(tensor: &Tensor) -> Self {
        // For non-quantized case, just store as dummy float16
        let size: usize = tensor.shape.iter().product();
        let dummy_data = vec![0u16; size];
        
        Self::new(dummy_data, tensor.shape.clone())
    }
    
    pub fn dequantize(&self) -> Tensor {
        // TODO: Implement float16 to f32 conversion
        // For now, return a dummy tensor
        let size: usize = self.shape.iter().product();
        Tensor::new(vec![0.0; size], &self.shape)
    }
    
    pub fn data(&self) -> std::cell::Ref<'_, Vec<u16>> {
        self.data.borrow()
    }
}

impl BFloat16Tensor {
    pub fn new(data: Vec<u16>, shape: SmallVec<[usize; 4]>) -> Self {
        Self {
            data: Rc::new(RefCell::new(data)),
            shape,
        }
    }
    
    pub fn dequantize(&self) -> Tensor {
        // TODO: Implement bfloat16 to f32 conversion
        // For now, return a dummy tensor
        let size: usize = self.shape.iter().product();
        Tensor::new(vec![0.0; size], &self.shape)
    }
    
    pub fn data(&self) -> std::cell::Ref<'_, Vec<u16>> {
        self.data.borrow()
    }
}

impl NF4Tensor {
    pub fn new(data: Vec<u8>, shape: SmallVec<[usize; 4]>, scale: f32, zero_point: i32) -> Self {
        Self {
            data: Rc::new(RefCell::new(data)),
            shape,
            scale,
            zero_point,
        }
    }
    
    pub fn dequantize(&self) -> Tensor {
        // TODO: Implement NF4 unpacking and dequantization
        // For now, return a dummy tensor
        let size: usize = self.shape.iter().product();
        Tensor::new(vec![0.0; size], &self.shape)
    }
    
    pub fn data(&self) -> std::cell::Ref<'_, Vec<u8>> {
        self.data.borrow()
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Self {
        Tensor {
            data: Rc::new(RefCell::new(data)),
            shape: shape.iter().cloned().collect(),
            grad: Rc::new(RefCell::new(None)),
            requires_grad: false,
            tape_node: Cell::new(None),
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

    pub fn data(&self) -> std::cell::Ref<'_, Vec<f32>> {
        self.data.borrow()
    }

    pub fn grad_ref(&self) -> Option<std::cell::Ref<'_, Vec<f32>>> {
        let r = self.grad.borrow();
        if r.is_some() {
            Some(std::cell::Ref::map(r, |opt| opt.as_ref().unwrap()))
        } else {
            None
        }
    }

    pub fn grad(&self) -> Option<Rc<Tensor>> {
        let r = self.grad.borrow();
        r.as_ref().map(|g| {
            let mut t = Tensor::new(g.clone(), &self.shape);
            t.requires_grad = false;
            Rc::new(t)
        })
    }

    pub fn backward(&self) {
        let ones = vec![1.0; self.data().len()];
        *self.grad.borrow_mut() = Some(ones);

        if let Some(node_id) = self.tape_node.get() {
            crate::tape::backward(node_id);
        }
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    pub fn from_data(&self, data: Vec<f32>, shape: &[usize]) -> Tensor {
        let mut tensor = Tensor::new(data, shape);
        if self.requires_grad {
            tensor.requires_grad = true;
        }
        tensor
    }

    pub fn data_mut(&self) -> std::cell::RefMut<'_, Vec<f32>> {
        self.data.borrow_mut()
    }

    /// Cache-friendly blocked transpose
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Can only transpose 2D tensors");

        let rows = self.shape[0];
        let cols = self.shape[1];
        let data = self.data();

        // Use cache-friendly blocked transpose
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
                if let Some(gout) = out.grad.borrow().as_ref() {
                    let mut slot = input.grad.borrow_mut();
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
        let data = self.data();
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

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.borrow().as_ref() {
                    let y = out.data();
                    let mut slot = input.grad.borrow_mut();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; y.len()]);
                    }
                    let gin = slot.as_mut().unwrap();
                    for ((gi, &g), &s) in gin.iter_mut().zip(gout.iter()).zip(y.iter()) {
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
            let self_data = self.data();
            let other_data = other.data();

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
                    if let Some(gout) = out.grad.borrow().as_ref() {
                        if a.requires_grad {
                            ops::accumulate_grad(&a, gout);
                        }

                        if b.requires_grad {
                            let mut slot = b.grad.borrow_mut();
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
                if let Some(gout) = o.grad.borrow().as_ref() {
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
                if let Some(gout) = out.grad.borrow().as_ref() {
                    let g_each = gout[0] / n;
                    let mut slot = input.grad.borrow_mut();
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
                if let Some(gout) = out.grad.borrow().as_ref() {
                    // Gradient just needs to be reshaped back
                    let mut slot = input.grad.borrow_mut();
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
                let keepdim = keepdim;
                let d = d;

                Tape::push_unary_op(self, &output, move || {
                    if let Some(gout) = out.grad.borrow().as_ref() {
                        let mut slot = input.grad.borrow_mut();
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
                                    if out_j < gout.len() {
                                        out_idx += coord * multiplier;
                                        multiplier *= if keepdim {
                                            if j == d { 1 } else { in_shape[j] }
                                        } else {
                                            if j < d {
                                                in_shape[j]
                                            } else if j > d {
                                                in_shape[j]
                                            } else {
                                                1
                                            }
                                        };
                                    }
                                }
                            }

                            gin[i] += gout[out_idx.min(gout.len() - 1)];
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
                    if let Some(gout) = out.grad.borrow().as_ref() {
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

            let values = Tensor::new(max_values, &out_shape);
            let indices = Tensor::new(max_indices, &out_shape);

            (values, indices)
        } else {
            // Global max
            let (max_val, max_idx) = data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, &v)| (v, i))
                .unwrap_or((0.0, 0));

            (Tensor::scalar(max_val), Tensor::scalar(max_idx as f32))
        }
    }

    /// Argmax - returns indices of maximum values
    pub fn argmax(&self, dim: Option<usize>) -> Tensor {
        self.max(dim).1
    }

    /// Exponential function (SIMD optimized)
    pub fn exp(&self) -> Tensor {
        let data = self.data();
        let mut result = vec![0.0; data.len()];

        // SIMD exp using approximation for better performance
        // For accurate exp, we use scalar for now (can be optimized with SIMD exp approximation)
        for (i, &x) in data.iter().enumerate() {
            result[i] = x.exp();
        }

        let mut output = Tensor::new(result, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.borrow().as_ref() {
                    // d/dx e^x = e^x
                    let exp_x = out.data();
                    let mut slot = input.grad.borrow_mut();
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
                if let Some(gout) = out.grad.borrow().as_ref() {
                    // d/dx ln(x) = 1/x
                    let x = input.data();
                    let mut slot = input.grad.borrow_mut();
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
                if let Some(gout) = out.grad.borrow().as_ref() {
                    // d/dx x^n = n * x^(n-1)
                    let x = input.data();
                    let mut slot = input.grad.borrow_mut();
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
        assert_eq!(weight.shape.len(), 4, "Weight must be 4D: [C_out, C_in, K_h, K_w]");

        let (n, c_in, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, c_in_w, k_h, k_w) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);

        assert_eq!(c_in, c_in_w, "Input and weight channel dimensions must match");

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

        output
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
        conv_out.relu_inplace()
    }

    /// 2D Max Pooling
    /// Input: [N, C, H, W], Output: [N, C, H_out, W_out]
    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Tensor {
        assert_eq!(self.shape.len(), 4, "Input must be 4D: [N, C, H, W]");

        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride.unwrap_or(kernel_size);
        let (pad_h, pad_w) = padding;

        let h_out = (h_in + 2 * pad_h - k_h) / s_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / s_w + 1;

        let data = self.data();
        let mut output_data = vec![f32::NEG_INFINITY; n * c * h_out * w_out];

        // Efficient pooling with proper indexing
        for batch in 0..n {
            for channel in 0..c {
                for out_h in 0..h_out {
                    for out_w in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;

                        for k_row in 0..k_h {
                            for k_col in 0..k_w {
                                let in_h = out_h * s_h + k_row;
                                let in_w = out_w * s_w + k_col;

                                if in_h >= pad_h && in_w >= pad_w {
                                    let in_h = in_h - pad_h;
                                    let in_w = in_w - pad_w;

                                    if in_h < h_in && in_w < w_in {
                                        let in_idx = batch * c * h_in * w_in +
                                                   channel * h_in * w_in +
                                                   in_h * w_in + in_w;
                                        max_val = max_val.max(data[in_idx]);
                                    }
                                }
                            }
                        }

                        let out_idx = batch * c * h_out * w_out +
                                     channel * h_out * w_out +
                                     out_h * w_out + out_w;
                        output_data[out_idx] = max_val;
                    }
                }
            }
        }

        let mut output = Tensor::new(output_data, &[n, c, h_out, w_out]);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let (k_h, k_w, s_h, s_w, pad_h, pad_w) = (k_h, k_w, s_h, s_w, pad_h, pad_w);
            let (n, c, h_in, w_in, h_out, w_out) = (n, c, h_in, w_in, h_out, w_out);

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.borrow().as_ref() {
                    let mut slot = input.grad.borrow_mut();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; n * c * h_in * w_in]);
                    }
                    let gin = slot.as_mut().unwrap();
                    let in_data = input.data();

                    // Backpropagate gradients through max pooling
                    for batch in 0..n {
                        for channel in 0..c {
                            for out_h in 0..h_out {
                                for out_w in 0..w_out {
                                    let mut max_val = f32::NEG_INFINITY;
                                    let mut max_pos = (0, 0);

                                    // Find the position of maximum value
                                    for k_row in 0..k_h {
                                        for k_col in 0..k_w {
                                            let in_h = out_h * s_h + k_row;
                                            let in_w = out_w * s_w + k_col;

                                            if in_h >= pad_h && in_w >= pad_w {
                                                let in_h = in_h - pad_h;
                                                let in_w = in_w - pad_w;

                                                if in_h < h_in && in_w < w_in {
                                                    let in_idx = batch * c * h_in * w_in +
                                                               channel * h_in * w_in +
                                                               in_h * w_in + in_w;
                                                    if in_data[in_idx] > max_val {
                                                        max_val = in_data[in_idx];
                                                        max_pos = (in_h, in_w);
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Add gradient to the maximum position
                                    let out_idx = batch * c * h_out * w_out +
                                                 channel * h_out * w_out +
                                                 out_h * w_out + out_w;
                                    let in_idx = batch * c * h_in * w_in +
                                               channel * h_in * w_in +
                                               max_pos.0 * w_in + max_pos.1;
                                    gin[in_idx] += gout[out_idx];
                                }
                            }
                        }
                    }
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
        assert_eq!(self.shape.len(), 4, "Input must be 4D: [N, C, H, W]");

        let (n, c, h_in, w_in) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (k_h, k_w) = kernel_size;
        let (s_h, s_w) = stride.unwrap_or(kernel_size);
        let (pad_h, pad_w) = padding;

        let h_out = (h_in + 2 * pad_h - k_h) / s_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / s_w + 1;

        let data = self.data();
        let mut output_data = vec![0.0; n * c * h_out * w_out];
        let pool_size = (k_h * k_w) as f32;

        for batch in 0..n {
            for channel in 0..c {
                for out_h in 0..h_out {
                    for out_w in 0..w_out {
                        let mut sum = 0.0;

                        for k_row in 0..k_h {
                            for k_col in 0..k_w {
                                let in_h = out_h * s_h + k_row;
                                let in_w = out_w * s_w + k_col;

                                if in_h >= pad_h && in_w >= pad_w {
                                    let in_h = in_h - pad_h;
                                    let in_w = in_w - pad_w;

                                    if in_h < h_in && in_w < w_in {
                                        let in_idx = batch * c * h_in * w_in +
                                                   channel * h_in * w_in +
                                                   in_h * w_in + in_w;
                                        sum += data[in_idx];
                                    }
                                }
                            }
                        }

                        let out_idx = batch * c * h_out * w_out +
                                     channel * h_out * w_out +
                                     out_h * w_out + out_w;
                        output_data[out_idx] = sum / pool_size;
                    }
                }
            }
        }

        let mut output = Tensor::new(output_data, &[n, c, h_out, w_out]);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();
            let (k_h, k_w, s_h, s_w, pad_h, pad_w) = (k_h, k_w, s_h, s_w, pad_h, pad_w);
            let (n, c, h_in, w_in, h_out, w_out) = (n, c, h_in, w_in, h_out, w_out);

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.borrow().as_ref() {
                    let mut slot = input.grad.borrow_mut();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; n * c * h_in * w_in]);
                    }
                    let gin = slot.as_mut().unwrap();
                    let pool_size = (k_h * k_w) as f32;

                    // Distribute gradient equally among pooled elements
                    for batch in 0..n {
                        for channel in 0..c {
                            for out_h in 0..h_out {
                                for out_w in 0..w_out {
                                    let out_idx = batch * c * h_out * w_out +
                                                 channel * h_out * w_out +
                                                 out_h * w_out + out_w;
                                    let grad_val = gout[out_idx] / pool_size;

                                    for k_row in 0..k_h {
                                        for k_col in 0..k_w {
                                            let in_h = out_h * s_h + k_row;
                                            let in_w = out_w * s_w + k_col;

                                            if in_h >= pad_h && in_w >= pad_w {
                                                let in_h = in_h - pad_h;
                                                let in_w = in_w - pad_w;

                                                if in_h < h_in && in_w < w_in {
                                                    let in_idx = batch * c * h_in * w_in +
                                                               channel * h_in * w_in +
                                                               in_h * w_in + in_w;
                                                    gin[in_idx] += grad_val;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
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

        let data = self.data();

        // Special case optimizations for common kernel sizes
        if k_h == 3 && k_w == 3 && stride_h == 1 && stride_w == 1 && dil_h == 1 && dil_w == 1 {
            self.im2col_3x3_stride1(&data, &mut col_data, n, c, h_in, w_in, h_out, w_out, pad_h, pad_w);
        } else if k_h == 1 && k_w == 1 {
            self.im2col_1x1(&data, &mut col_data, n, c, h_in, w_in, h_out, w_out);
        } else {
            // General case with SIMD optimization
            self.im2col_general_simd(&data, &mut col_data, n, c, h_in, w_in, h_out, w_out,
                                      k_h, k_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w);
        }

        Tensor::new(col_data, &[num_windows, col_size])
    }

    /// Highly optimized 3x3 convolution im2col (most common case)
    #[inline]
    fn im2col_3x3_stride1(
        &self,
        input: &[f32],
        output: &mut [f32],
        n: usize, c: usize, h_in: usize, w_in: usize,
        h_out: usize, w_out: usize,
        pad_h: usize, pad_w: usize,
    ) {
        let col_size = c * 9; // 3x3 = 9

        // Process in blocks for better cache locality
        const BLOCK_SIZE: usize = 8;

        for batch in 0..n {
            let batch_offset = batch * c * h_in * w_in;

            for out_h_block in (0..h_out).step_by(BLOCK_SIZE) {
                let h_end = (out_h_block + BLOCK_SIZE).min(h_out);

                for out_w_block in (0..w_out).step_by(BLOCK_SIZE) {
                    let w_end = (out_w_block + BLOCK_SIZE).min(w_out);

                    // Process block
                    for out_h in out_h_block..h_end {
                        for out_w in out_w_block..w_end {
                            let window_idx = batch * h_out * w_out + out_h * w_out + out_w;
                            let col_base = window_idx * col_size;

                            for ch in 0..c {
                                let ch_offset = ch * 9;

                                // Unrolled 3x3 kernel
                                for k_row in 0..3 {
                                    let in_h = out_h + k_row;
                                    let in_h_valid = in_h >= pad_h && in_h < h_in + pad_h;
                                    let in_h_idx = if in_h_valid { in_h - pad_h } else { 0 };

                                    for k_col in 0..3 {
                                        let in_w = out_w + k_col;
                                        let col_idx = col_base + ch_offset + k_row * 3 + k_col;

                                        if in_h_valid && in_w >= pad_w && in_w < w_in + pad_w {
                                            let in_w_idx = in_w - pad_w;
                                            let in_idx = batch_offset + ch * h_in * w_in +
                                                    in_h_idx * w_in + in_w_idx;
                                            output[col_idx] = input[in_idx];
                                        }
                                        // else: stays 0.0 (padding)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Ultra-fast 1x1 convolution (just reshape)
    #[inline]
    fn im2col_1x1(
        &self,
        input: &[f32],
        output: &mut [f32],
        _n: usize, _c: usize, h_in: usize, w_in: usize,
        h_out: usize, w_out: usize,
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
        n: usize, c: usize, h_in: usize, w_in: usize,
        h_out: usize, w_out: usize,
        k_h: usize, k_w: usize,
        stride_h: usize, stride_w: usize,
        pad_h: usize, pad_w: usize,
        dil_h: usize, dil_w: usize,
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

                                    if in_w >= pad_w && in_w < w_in + pad_w && consecutive_count == k_col - start_k_col {
                                        consecutive_count += 1;
                                    } else {
                                        // Copy accumulated consecutive elements
                                        if consecutive_count > 0 {
                                            self.copy_consecutive_elements(
                                                input, output, batch, ch, h_in, w_in,
                                                in_h_idx, out_w * stride_w + start_k_col * dil_w - pad_w,
                                                col_base + ch * k_h * k_w + k_row * k_w + start_k_col,
                                                consecutive_count
                                            );
                                        }

                                        // Handle non-consecutive element
                                        let col_idx = col_base + ch * k_h * k_w + k_row * k_w + k_col;

                                        if in_w >= pad_w && in_w < w_in + pad_w {
                                            let in_w_idx = in_w - pad_w;
                                            let in_idx = batch * c * h_in * w_in +
                                                    ch * h_in * w_in +
                                                    in_h_idx * w_in + in_w_idx;
                                            output[col_idx] = input[in_idx];
                                        }

                                        start_k_col = k_col + 1;
                                        consecutive_count = 0;
                                    }
                                }

                                // Handle remaining consecutive elements
                                if consecutive_count > 0 {
                                    self.copy_consecutive_elements(
                                        input, output, batch, ch, h_in, w_in,
                                        in_h_idx, out_w * stride_w + start_k_col * dil_w - pad_w,
                                        col_base + ch * k_h * k_w + k_row * k_w + start_k_col,
                                        consecutive_count
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
        input: &[f32], output: &mut [f32],
        batch: usize, ch: usize, h_in: usize, w_in: usize,
        in_h: usize, in_w_start: usize,
        out_start: usize, count: usize
    ) {
        if count >= 8 && in_w_start + count <= w_in {
            // Use SIMD for larger copies
            let in_base = batch * ch * h_in * w_in + ch * h_in * w_in + in_h * w_in + in_w_start;

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
                    count
                );
            }
        } else {
            // Scalar copy for small counts
            for i in 0..count {
                let in_w = in_w_start + i;
                if in_w < w_in {
                    let in_idx = batch * ch * h_in * w_in + ch * h_in * w_in + in_h * w_in + in_w;
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
                if let Some(gout) = out.grad.borrow().as_ref() {
                    if input.requires_grad {
                        ops::accumulate_grad(&input, gout);
                    }

                    if bias_t.requires_grad {
                        let mut slot = bias_t.grad.borrow_mut();
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
        let new_shape = [old_shape[axes[0]], old_shape[axes[1]], old_shape[axes[2]], old_shape[axes[3]]];

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
                            new_indices[axes[3]]
                        );

                        let new_idx = n0 * new_shape[1] * new_shape[2] * new_shape[3] +
                                     n1 * new_shape[2] * new_shape[3] +
                                     n2 * new_shape[3] + n3;

                        result_data[new_idx] = data[old_idx];
                    }
                }
            }
        }

        Tensor::new(result_data, &new_shape)
    }

    /// In-place ReLU for fusion operations
    fn relu_inplace(self) -> Tensor {
        self.relu()
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
        
        // Calculate min/max and scale/zero_point
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let scale = config.compute_scale(min_val, max_val).unwrap();
        let zero_point = config.compute_zero_point(min_val, scale).unwrap();
        
        // Quantize
        let quantized_data: Vec<i8> = data
            .iter()
            .map(|&x| {
                let q = ((x / scale) + zero_point as f32).round() as i32;
                q.clamp(qmin, qmax) as i8
            })
            .collect();
        
        Int8Tensor::new(quantized_data, self.shape.clone(), scale, zero_point)
    }
    
    /// Quantize to int4 (packed representation)
    fn quantize_to_int4(&self, config: &QuantizationConfig) -> Int4Tensor {
        // For now, create a dummy int4 tensor
        // TODO: Implement proper int4 quantization with packing
        let size: usize = self.shape.iter().product();
        let packed_size = (size + 1) / 2; // 2 int4 values per byte
        let dummy_data = vec![0u8; packed_size];
        
        Int4Tensor::new(dummy_data, self.shape.clone(), 1.0, 0)
    }
    
    /// Convert to float16
    fn quantize_to_float16(&self) -> Float16Tensor {
        // For now, create a dummy float16 tensor
        // TODO: Implement proper float16 conversion
        let size: usize = self.shape.iter().product();
        let dummy_data = vec![0u16; size];
        
        Float16Tensor::new(dummy_data, self.shape.clone())
    }
    
    /// Convert to bfloat16
    fn quantize_to_bfloat16(&self) -> BFloat16Tensor {
        // For now, create a dummy bfloat16 tensor
        // TODO: Implement proper bfloat16 conversion
        let size: usize = self.shape.iter().product();
        let dummy_data = vec![0u16; size];
        
        BFloat16Tensor::new(dummy_data, self.shape.clone())
    }
    
    /// Quantize to NF4
    fn quantize_to_nf4(&self, config: &QuantizationConfig) -> NF4Tensor {
        // For now, create a dummy NF4 tensor
        // TODO: Implement proper NF4 quantization
        let size: usize = self.shape.iter().product();
        let packed_size = (size + 1) / 2; // 2 NF4 values per byte
        let dummy_data = vec![0u8; packed_size];
        
        NF4Tensor::new(dummy_data, self.shape.clone(), 1.0, 0)
    }
}
