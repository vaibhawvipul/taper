use crate::gemm::{n as no_trans, sgemm_rowmajor, t as trans};
use crate::{Tensor, tape::Tape};
use std::ops::{Add, Div, Mul, Sub};

// Import the SIMD utilities from tensor module
use crate::tensor::simd;

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.data().len(),
            other.data().len(),
            "Tensor dimensions must match"
        );

        let self_data = self.data();
        let other_data = other.data();
        let mut out_data = vec![0.0; self_data.len()];

        // Use SIMD operations
        unsafe {
            simd::add_f32_simd(&self_data, &other_data, &mut out_data);
        }

        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a = self.clone();
            let b = other.clone();
            let o = out.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout) = o.grad.borrow().as_ref() {
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
    fn mul(self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.data().len(),
            other.data().len(),
            "Tensor dimensions must match"
        );

        let self_data = self.data();
        let other_data = other.data();
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
                if let Some(gout) = o.grad.borrow().as_ref() {
                    if a.requires_grad {
                        let bdat = b.data();
                        let mut slot = a.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; bdat.len()]);
                        }
                        let ga = slot.as_mut().unwrap();
                        let mut out = vec![0.0; ga.len()];

                        // SIMD gradient multiplication and accumulation
                        unsafe {
                            let mut temp = vec![0.0; ga.len()];
                            simd::mul_f32_simd(gout, &bdat, &mut temp);
                            simd::add_f32_simd(ga, &temp, &mut out);
                        }
                        *ga = out;
                    }
                    if b.requires_grad {
                        let adat = a.data();
                        let mut slot = b.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; adat.len()]);
                        }
                        let gb = slot.as_mut().unwrap();
                        let mut out = gb.clone();

                        // SIMD gradient multiplication and accumulation
                        unsafe {
                            let mut temp = vec![0.0; gb.len()];
                            simd::mul_f32_simd(gout, &adat, &mut temp);
                            simd::add_f32_simd(gb, &temp, &mut out);
                        }
                        *gb = out;
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
    let mut slot = t.grad.borrow_mut();
    if slot.is_none() {
        *slot = Some(vec![0.0; t.data().len()]);
    }
    let g = slot.as_mut().unwrap();
    // SIMD accumulate
    let mut temp = vec![0.0; g.len()];

    unsafe {
        simd::add_f32_simd(g, src, &mut temp);
    }
    *g = temp;
}

#[inline]
pub fn accumulate_grad_scaled(t: &Tensor, src: &[f32], scale: f32) {
    let mut slot = t.grad.borrow_mut();
    if slot.is_none() {
        *slot = Some(vec![0.0; t.data().len()]);
    }
    let g = slot.as_mut().unwrap();

    // Scale and accumulate
    for (gi, &s) in g.iter_mut().zip(src) {
        *gi += scale * s;
    }
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

        let a = self.data();
        let b = other.data();
        let mut c = vec![0.0f32; (m * n_out) as usize];

        // Forward: C = A * B
        sgemm_rowmajor(
            no_trans(),
            no_trans(),
            m,
            n_out,
            k,
            1.0,
            &a[..],
            &b[..],
            0.0,
            &mut c,
        );

        let mut out = Tensor::new(c, &[m as usize, n_out as usize]);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;
            let a_t = self.clone();
            let b_t = other.clone();
            let out_t = out.clone();
            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();

            Tape::push_binary_op(self, other, &out, move || {
                if let Some(gout_vec) = out_t.grad.borrow().as_ref() {
                    let gout = &gout_vec[..];

                    // dA += dC * B^T   (m×k) = (m×n_out) * (n_out×k)
                    if a_t.requires_grad {
                        let m = a_shape[0] as i32;
                        let k = a_shape[1] as i32;
                        let n_out = b_shape[1] as i32;

                        let bdat = b_t.data();
                        let mut slot = a_t.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; (m * k) as usize]);
                        }
                        let ga = slot.as_mut().unwrap();
                        sgemm_rowmajor(
                            no_trans(),
                            trans(),
                            m,
                            k,
                            n_out,
                            1.0,
                            gout,
                            &bdat[..],
                            1.0,
                            ga,
                        );
                    }

                    // dB += A^T * dC   (k×n_out) = (k×m) * (m×n_out)
                    if b_t.requires_grad {
                        let kdim = b_shape[0] as i32;
                        let n_out = b_shape[1] as i32;
                        let m = a_shape[0] as i32;

                        let adat = a_t.data();
                        let mut slot = b_t.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; (kdim * n_out) as usize]);
                        }
                        let gb = slot.as_mut().unwrap();
                        sgemm_rowmajor(
                            trans(),
                            no_trans(),
                            kdim,
                            n_out,
                            m,
                            1.0,
                            &adat[..],
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
        let data = self.data();
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
                if let Some(gout) = out.grad.borrow().as_ref() {
                    let x = input.data();
                    let mut slot = input.grad.borrow_mut();
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
        assert_eq!(
            self.data().len(),
            other.data().len(),
            "Tensor dimensions must match"
        );

        let self_data = self.data();
        let other_data = other.data();
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
                if let Some(gout) = o.grad.borrow().as_ref() {
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
        assert_eq!(
            self.data().len(),
            other.data().len(),
            "Tensor dimensions must match"
        );

        let self_data = self.data();
        let other_data = other.data();
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
                if let Some(gout) = o.grad.borrow().as_ref() {
                    if a.requires_grad {
                        let bdat = b.data();
                        let mut slot = a.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; bdat.len()]);
                        }
                        let ga = slot.as_mut().unwrap();
                        for i in 0..ga.len() {
                            ga[i] += gout[i] / bdat[i];
                        }
                    }
                    if b.requires_grad {
                        let adat = a.data();
                        let bdat = b.data();
                        let mut slot = b.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; adat.len()]);
                        }
                        let gb = slot.as_mut().unwrap();
                        for i in 0..gb.len() {
                            gb[i] -= gout[i] * adat[i] / (bdat[i] * bdat[i]);
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
