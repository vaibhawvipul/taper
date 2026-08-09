use crate::gemm::{TransFlag, n as no_trans, sgemm_rowmajor, t as trans};
use crate::tensor::GemmLayout;
use crate::{Tensor, tape::Tape};
use std::ops::{Add, Div, Mul, Sub};

/// The GEMM trans flag that makes `operand` present as `op(X) = Xᵀ` when
/// `want_transpose`, or as `X` otherwise.
///
/// A tensor stored transposed already *is* its own transpose in memory, so the
/// two conditions cancel: the flag is `T` exactly when they disagree. This one
/// rule covers the forward product and both backward products.
#[inline]
fn trans_flag(operand: &Tensor, want_transpose: bool) -> TransFlag {
    let stored_transposed = matches!(operand.gemm_layout_2d(), Some(GemmLayout::Transposed));
    if stored_transposed != want_transpose {
        trans()
    } else {
        no_trans()
    }
}

// Import the SIMD utilities from tensor module
use crate::tensor::simd;

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        assert_same_shape(self, other, "add");

        let a_data = self.elements();
        let b_data = other.elements();
        let mut out_data = vec![0.0; a_data.len()];

        // Use SIMD operations
        unsafe {
            simd::add_f32_simd(&a_data, &b_data, &mut out_data);
        }

        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a = self.clone();
            let b = other.clone();
            let o = out.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout) = o.grad.read().unwrap().as_ref() {
                    if a.requires_grad {
                        accumulate_grad(&a, gout);
                    }
                    if b.requires_grad {
                        accumulate_grad(&b, gout);
                    }
                }
            });
        }
        out
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    // The `+=` below accumulates gradients in the backward closure; it is not
    // the multiplication itself, which clippy's heuristic cannot distinguish.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, other: &Tensor) -> Tensor {
        assert_same_shape(self, other, "mul");

        let self_data = self.elements();
        let other_data = other.elements();
        let mut out_data = vec![0.0; self_data.len()];

        // Use SIMD operations
        unsafe {
            simd::mul_f32_simd(&self_data, &other_data, &mut out_data);
        }

        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a = self.clone();
            let b = other.clone();
            let o = out.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout) = o.grad.read().expect("grad RwLock poisoned").as_ref() {
                    // d/da (a*b) = b, d/db (a*b) = a. Each branch used to size the
                    // gradient from the *other* operand's data and round-trip
                    // through two scratch buffers; fold it into one pass.
                    if a.requires_grad {
                        let bdat = b.elements();
                        let mut slot = a.grad.write().expect("grad RwLock poisoned");
                        let ga = slot.get_or_insert_with(|| vec![0.0; a.numel()]);
                        for ((g, &go), &bv) in ga.iter_mut().zip(gout.iter()).zip(bdat.iter()) {
                            *g += go * bv;
                        }
                    }
                    if b.requires_grad {
                        let adat = a.elements();
                        let mut slot = b.grad.write().expect("grad RwLock poisoned");
                        let gb = slot.get_or_insert_with(|| vec![0.0; b.numel()]);
                        for ((g, &go), &av) in gb.iter_mut().zip(gout.iter()).zip(adat.iter()) {
                            *g += go * av;
                        }
                    }
                }
            });
        }
        out
    }
}

// Helper function to accumulate gradients with SIMD
#[inline]
pub fn accumulate_grad(t: &Tensor, src: &[f32]) {
    let mut slot = t.grad.write().expect("grad RwLock poisoned");
    let g = slot.get_or_insert_with(|| vec![0.0; t.numel()]);
    debug_assert_eq!(
        g.len(),
        src.len(),
        "accumulate_grad: gradient length does not match the tensor"
    );
    // Accumulate in place; the previous version allocated a temporary the size
    // of the gradient on every call and then moved it back over `g`.
    unsafe {
        simd::add_assign_f32_simd(g, src);
    }
}

#[inline]
pub fn accumulate_grad_scaled(t: &Tensor, src: &[f32], scale: f32) {
    let mut slot = t.grad.write().expect("grad RwLock poisoned");
    let g = slot.get_or_insert_with(|| vec![0.0; t.numel()]);
    debug_assert_eq!(
        g.len(),
        src.len(),
        "accumulate_grad_scaled: gradient length does not match the tensor"
    );

    // Scale and accumulate
    for (gi, &s) in g.iter_mut().zip(src) {
        *gi += scale * s;
    }
}

/// Shared precondition for the element-wise operators: identical shapes.
///
/// Checking only the flat length let `[2,3] + [3,2]` through and silently
/// produced a result labelled with the left operand's shape.
#[inline]
fn assert_same_shape(a: &Tensor, b: &Tensor, op: &str) {
    assert_eq!(
        a.shape(),
        b.shape(),
        "{op}: tensor shapes must match ({:?} vs {:?})",
        a.shape(),
        b.shape()
    );
}

// Implement other trait combinations
impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        (&self).add(other)
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        self.add(&other)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        (&self).add(&other)
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        (&self).mul(other)
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        self.mul(&other)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        (&self).mul(&other)
    }
}

impl Tensor {
    /// SIMD-optimized matrix multiplication
    /// Fast matrix multiplication: [m,k] @ [k,n] -> [m,n]
    /// Fast matrix multiplication: [m,k] @ [k,n_out] -> [m,n_out]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "First tensor must be 2D");
        assert_eq!(other.shape.len(), 2, "Second tensor must be 2D");

        let m = self.shape[0] as i32;
        let k = self.shape[1] as i32;
        let k2 = other.shape[0] as i32;
        let n_out = other.shape[1] as i32;
        assert_eq!(k, k2, "Inner dimensions must match: {} vs {}", k, k2);

        // A transposed view is already the layout BLAS wants under its trans
        // flag, so it feeds straight in — no materialization. Anything with a
        // layout GEMM can't address (a strided sub-window) is packed first.
        let a = self.packed_operand();
        let b = other.packed_operand();

        let a_store = a.storage();
        let b_store = b.storage();
        let mut c = vec![0.0f32; (m * n_out) as usize];

        // Forward: C = A * B
        sgemm_rowmajor(
            trans_flag(&a, false),
            trans_flag(&b, false),
            m,
            n_out,
            k,
            1.0,
            &a_store[a.offset..],
            &b_store[b.offset..],
            0.0,
            &mut c,
        );
        drop(a_store);
        drop(b_store);

        let mut out = Tensor::new(c, &[m as usize, n_out as usize]);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a_t = self.clone();
            let b_t = other.clone();
            let out_t = out.clone();
            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout_vec) = out_t.grad.read().unwrap().as_ref() {
                    let gout = &gout_vec[..];

                    // dA += dC * B^T   (m×k) = (m×n_out) * (n_out×k)
                    if a_t.requires_grad {
                        let m = a_shape[0] as i32;
                        let k = a_shape[1] as i32;
                        let n_out = b_shape[1] as i32;

                        // Gradients are always dense row-major, so only the
                        // operand read out of the graph needs a layout flag.
                        let b_op = b_t.packed_operand();
                        let b_store = b_op.storage();
                        let mut slot = a_t.grad.write().expect("grad RwLock poisoned");
                        let ga = slot.get_or_insert_with(|| vec![0.0; (m * k) as usize]);
                        sgemm_rowmajor(
                            no_trans(),
                            trans_flag(&b_op, true),
                            m,
                            k,
                            n_out,
                            1.0,
                            gout,
                            &b_store[b_op.offset..],
                            1.0,
                            ga,
                        );
                    }

                    // dB += A^T * dC   (k×n_out) = (k×m) * (m×n_out)
                    if b_t.requires_grad {
                        let kdim = b_shape[0] as i32;
                        let n_out = b_shape[1] as i32;
                        let m = a_shape[0] as i32;

                        let a_op = a_t.packed_operand();
                        let a_store = a_op.storage();
                        let mut slot = b_t.grad.write().expect("grad RwLock poisoned");
                        let gb = slot.get_or_insert_with(|| vec![0.0; (kdim * n_out) as usize]);
                        sgemm_rowmajor(
                            trans_flag(&a_op, true),
                            no_trans(),
                            kdim,
                            n_out,
                            m,
                            1.0,
                            &a_store[a_op.offset..],
                            gout,
                            1.0,
                            gb,
                        );
                    }
                }
            });
        }

        out
    }

    /// Create a random tensor with values from normal distribution
    pub fn randn(shape: &[usize]) -> Tensor {
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand::thread_rng();

        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| StandardNormal.sample(&mut rng)).collect();

        Tensor::new(data, shape)
    }

    /// SIMD-optimized ReLU activation
    pub fn relu(&self) -> Tensor {
        let data = self.elements();
        let mut result = vec![0.0; data.len()];

        // SIMD ReLU using max with zero
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse") {
                unsafe {
                    use std::arch::x86_64::*;
                    let zero = _mm_setzero_ps();
                    let chunks = data.len() / 4;

                    for i in 0..chunks {
                        let idx = i * 4;
                        let v = _mm_loadu_ps(data.as_ptr().add(idx));
                        let r = _mm_max_ps(v, zero);
                        _mm_storeu_ps(result.as_mut_ptr().add(idx), r);
                    }

                    // Handle remainder
                    for i in (chunks * 4)..data.len() {
                        result[i] = data[i].max(0.0);
                    }
                }
            } else {
                for (i, &x) in data.iter().enumerate() {
                    result[i] = x.max(0.0);
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            for (i, &x) in data.iter().enumerate() {
                result[i] = x.max(0.0);
            }
        }

        let mut output = Tensor::new(result, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;
            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.read().unwrap().as_ref() {
                    let x = input.data();
                    let mut slot = input.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; x.len()]);
                    }
                    let gin = slot.as_mut().unwrap();
                    for ((gi, &g), &v) in gin.iter_mut().zip(gout.iter()).zip(x.iter()) {
                        *gi += if v > 0.0 { g } else { 0.0 };
                    }
                }
            });
        }

        output
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor {
        assert_same_shape(self, other, "sub");

        let self_data = self.elements();
        let other_data = other.elements();
        let mut out_data = vec![0.0; self_data.len()];

        // Subtract using SIMD - we can reuse add with negation
        for i in 0..out_data.len() {
            out_data[i] = self_data[i] - other_data[i];
        }

        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a = self.clone();
            let b = other.clone();
            let o = out.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout) = o.grad.read().unwrap().as_ref() {
                    if a.requires_grad {
                        accumulate_grad(&a, gout);
                    }
                    if b.requires_grad {
                        accumulate_grad_scaled(&b, gout, -1.0);
                    }
                }
            });
        }
        out
    }
}

// Implement other combinations
impl Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor {
        (&self).sub(other)
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Tensor {
        self.sub(&other)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Tensor {
        (&self).sub(&other)
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, other: &Tensor) -> Tensor {
        assert_same_shape(self, other, "div");

        let self_data = self.elements();
        let other_data = other.elements();
        let mut out_data = vec![0.0; self_data.len()];

        // Element-wise division
        for i in 0..out_data.len() {
            out_data[i] = self_data[i] / other_data[i];
        }

        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a = self.clone();
            let b = other.clone();
            let o = out.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout) = o.grad.read().expect("grad RwLock poisoned").as_ref() {
                    // d/da (a/b) = 1/b, d/db (a/b) = -a/b²
                    if a.requires_grad {
                        let bdat = b.elements();
                        let mut slot = a.grad.write().expect("grad RwLock poisoned");
                        let ga = slot.get_or_insert_with(|| vec![0.0; a.numel()]);
                        for ((g, &go), &bv) in ga.iter_mut().zip(gout.iter()).zip(bdat.iter()) {
                            *g += go / bv;
                        }
                    }
                    if b.requires_grad {
                        let adat = a.elements();
                        let bdat = b.elements();
                        let mut slot = b.grad.write().expect("grad RwLock poisoned");
                        let gb = slot.get_or_insert_with(|| vec![0.0; b.numel()]);
                        for (((g, &go), &av), &bv) in gb
                            .iter_mut()
                            .zip(gout.iter())
                            .zip(adat.iter())
                            .zip(bdat.iter())
                        {
                            *g -= go * av / (bv * bv);
                        }
                    }
                }
            });
        }
        out
    }
}

// Implement other combinations
impl Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, other: &Tensor) -> Tensor {
        (&self).div(other)
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, other: Tensor) -> Tensor {
        self.div(&other)
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, other: Tensor) -> Tensor {
        (&self).div(&other)
    }
}
