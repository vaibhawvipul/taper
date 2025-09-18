use crate::{Tensor, tape::Tape};
use std::ops::{Add, Mul, Sub};

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.data().len(),
            other.data().len(),
            "Tensor dimensions must match"
        );

        // forward (still allocates output, as it should)
        let out_data: Vec<f32> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a + b)
            .collect();
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

        let out_data: Vec<f32> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a * b)
            .collect();
        let mut out = Tensor::new(out_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            out.requires_grad = true;

            // capture handles; borrow inside closure (no clones)
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
                        for ((gi, &g), &bv) in ga.iter_mut().zip(gout.iter()).zip(bdat.iter()) {
                            *gi += g * bv;
                        }
                    }
                    if b.requires_grad {
                        let adat = a.data();
                        let mut slot = b.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; adat.len()]);
                        }
                        let gb = slot.as_mut().unwrap();
                        for ((gi, &g), &av) in gb.iter_mut().zip(gout.iter()).zip(adat.iter()) {
                            *gi += g * av;
                        }
                    }
                }
            });
        }
        out
    }
}

// Helper function to accumulate gradients
#[inline]
pub fn accumulate_grad(t: &Tensor, src: &[f32]) {
    let mut slot = t.grad.borrow_mut();
    if slot.is_none() {
        *slot = Some(vec![0.0; t.data().len()]);
    }
    let g = slot.as_mut().unwrap();
    for (gi, &s) in g.iter_mut().zip(src) {
        *gi += s;
    }
}

#[inline]
pub fn accumulate_grad_scaled(t: &Tensor, src: &[f32], scale: f32) {
    let mut slot = t.grad.borrow_mut();
    if slot.is_none() {
        *slot = Some(vec![0.0; t.data().len()]);
    }
    let g = slot.as_mut().unwrap();
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
    /// Matrix multiplication: [m, k] @ [k, n] -> [m, n]
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

        // Forward pass - naive implementation (optimize with SIMD later)
        let mut result = vec![0.0f32; m * n];
        let a_data = self.data();
        let b_data = other.data();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k_idx in 0..k {
                    sum += a_data[i * k + k_idx] * b_data[k_idx * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        let mut output = Tensor::new(result, &[m, n]);

        // Backward pass setup
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;

            let a = self.clone();
            let b = other.clone();
            let out = output.clone();

            // Capture shapes and data for backward
            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();
            let a_data = self.data().clone();
            let b_data = other.data().clone();

            Tape::push_binary_op(self, other, &output, move || {
                if let Some(gout) = out.grad.borrow().as_ref() {
                    let g = &*gout;
                    // A: [m,k], B: [k,n], G: [m,n]
                    if a.requires_grad {
                        let (m, k) = (a_shape[0], a_shape[1]);
                        let n = b_shape[1];
                        let bdat = b_data.clone(); // could also borrow each time via b.data()
                        let mut slot = a.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; m * k]);
                        }
                        let ga = slot.as_mut().unwrap();
                        for i in 0..m {
                            for j in 0..k {
                                let mut acc = 0.0;
                                for t in 0..n {
                                    acc += g[i * n + t] * bdat[j * n + t];
                                }
                                ga[i * k + j] += acc;
                            }
                        }
                    }
                    if b.requires_grad {
                        let (k, n) = (b_shape[0], b_shape[1]);
                        let m = a_shape[0];
                        let adat = a_data.clone();
                        let mut slot = b.grad.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(vec![0.0; k * n]);
                        }
                        let gb = slot.as_mut().unwrap();
                        for i in 0..k {
                            for j in 0..n {
                                let mut acc = 0.0;
                                for t in 0..m {
                                    acc += adat[t * a_shape[1] + i] * g[t * n + j];
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
        use rand_distr::Distribution;
        use rand_distr::StandardNormal;
        let mut rng = rand::thread_rng();

        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| StandardNormal.sample(&mut rng)).collect();

        Tensor::new(data, shape)
    }

    /// ReLU activation function
    pub fn relu(&self) -> Tensor {
        let result_data: Vec<f32> = self.data().iter().map(|&x| x.max(0.0)).collect();

        let mut output = Tensor::new(result_data, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;

            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.borrow().as_ref() {
                    let x = input.data(); // mask
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
        let out_data: Vec<f32> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a - b)
            .collect();
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
