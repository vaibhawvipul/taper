// tensor.rs - SIMD-optimized tensor implementation using stable Rust
use crate::{ops, tape::Tape};
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
                add_f32_sse2(a, b, out);
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
}
