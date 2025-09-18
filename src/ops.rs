use crate::{Tensor, tape::Tape};
use std::ops::{Add, Mul, Sub};

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
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "First tensor must be 2D");
        assert_eq!(other.shape.len(), 2, "Second tensor must be 2D");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        assert_eq!(
            k, other.shape[0],
            "Inner dimensions must match: {}x{} @ {}x{}",
            m, k, other.shape[0], n
        );

        let mut result = vec![0.0f32; m * n];
        let a_data = self.data();
        let b_data = other.data();

        // Optimized matmul with better cache locality
        // Using tiling for cache efficiency
        let tile_size = 64.min(m).min(n).min(k);

        for i0 in (0..m).step_by(tile_size) {
            for j0 in (0..n).step_by(tile_size) {
                for k0 in (0..k).step_by(tile_size) {
                    // Process tile
                    let i_max = (i0 + tile_size).min(m);
                    let j_max = (j0 + tile_size).min(n);
                    let k_max = (k0 + tile_size).min(k);

                    for i in i0..i_max {
                        for j in j0..j_max {
                            let mut sum = result[i * n + j];

                            // Inner loop - potential for SIMD
                            for k_idx in k0..k_max {
                                sum += a_data[i * k + k_idx] * b_data[k_idx * n + j];
                            }

                            result[i * n + j] = sum;
                        }
                    }
                }
            }
        }

        let mut output = Tensor::new(result, &[m, n]);

        // Backward pass setup
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;

            let a = self.clone();
            let b = other.clone();
            let out = output.clone();

            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();

            Tape::push_binary_op(self, other, &output, move || {
                if let Some(gout) = out.grad.borrow().as_ref() {
                    if a.requires_grad {
                        // Gradient w.r.t A: dL/dA = dL/dC @ B^T
                        let (m, k) = (a_shape[0], a_shape[1]);
                        let n = b_shape[1];
                        let bdat = b.data();
                        let mut slot = a.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; m * k]);
                        }
                        let ga = slot.as_mut().unwrap();

                        // Tiled gradient computation
                        for i in 0..m {
                            for j in 0..k {
                                let mut acc = 0.0;
                                for t in 0..n {
                                    acc += gout[i * n + t] * bdat[j * n + t];
                                }
                                ga[i * k + j] += acc;
                            }
                        }
                    }
                    if b.requires_grad {
                        // Gradient w.r.t B: dL/dB = A^T @ dL/dC
                        let (k, n) = (b_shape[0], b_shape[1]);
                        let m = a_shape[0];
                        let adat = a.data();
                        let mut slot = b.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; k * n]);
                        }
                        let gb = slot.as_mut().unwrap();

                        for i in 0..k {
                            for j in 0..n {
                                let mut acc = 0.0;
                                for t in 0..m {
                                    acc += adat[t * a_shape[1] + i] * gout[t * n + j];
                                }
                                gb[i * n + j] += acc;
                            }
                        }
                    }
                }
            });
        }

        output
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
