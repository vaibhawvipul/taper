use crate::{Tensor, tape::Tape};
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        assert_eq!(self.data().len(), other.data().len(), "Tensor dimensions must match");

        let result_data: Vec<f32> = self.data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a + b)
            .collect();

        let mut output = Tensor::new(result_data, &self.shape);

        // Setup backward pass if needed
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;

            let a = self.clone();
            let b = other.clone();
            let out = output.clone();

            Tape::push_binary_op(self, other, &output, move || {
                // Get output gradient
                if let Some(grad_output) = out.grad.borrow().as_ref() {
                    // d(a+b)/da = 1
                    if a.requires_grad {
                        accumulate_grad(&a, grad_output);
                    }
                    // d(a+b)/db = 1
                    if b.requires_grad {
                        accumulate_grad(&b, grad_output);
                    }
                }
            });
        }

        output
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        assert_eq!(self.data().len(), other.data().len(), "Tensor dimensions must match");

        let result_data: Vec<f32> = self.data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a * b)
            .collect();

        let mut output = Tensor::new(result_data, &self.shape);

        // Setup backward pass if needed
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;

            // Clone tensors for the backward pass
            let a = self.clone();
            let b = other.clone();
            let out = output.clone();

            // Store the data we need for backward
            let a_data = self.data().clone();
            let b_data = other.data().clone();
            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();

            Tape::push_binary_op(self, other, &output, move || {
                // Get output gradient
                if let Some(grad_output) = out.grad.borrow().as_ref() {
                    let grad_output_data = grad_output.data().clone();

                    // d(a*b)/da = b
                    if a.requires_grad {
                        let grad_data: Vec<f32> = grad_output_data
                            .iter()
                            .zip(b_data.iter())
                            .map(|(g, b)| g * b)
                            .collect();

                        let mut grad_tensor = Tensor::new(grad_data, &a_shape);
                        grad_tensor.requires_grad = false;
                        accumulate_grad(&a, &grad_tensor);
                    }

                    // d(a*b)/db = a
                    if b.requires_grad {
                        let grad_data: Vec<f32> = grad_output_data
                            .iter()
                            .zip(a_data.iter())
                            .map(|(g, a)| g * a)
                            .collect();

                        let mut grad_tensor = Tensor::new(grad_data, &b_shape);
                        grad_tensor.requires_grad = false;
                        accumulate_grad(&b, &grad_tensor);
                    }
                }
            });
        }

        output
    }
}

// Helper function to accumulate gradients
pub fn accumulate_grad(tensor: &Tensor, grad: &Tensor) {
    let mut grad_ref = tensor.grad.borrow_mut();

    let new_grad = match grad_ref.as_ref() {
        Some(existing) => {
            // Accumulate with existing gradient
            let accumulated_data: Vec<f32> = existing.data()
                .iter()
                .zip(grad.data().iter())
                .map(|(e, g)| e + g)
                .collect();

            let mut result = Tensor::new(accumulated_data, &tensor.shape);
            result.requires_grad = false;
            result
        }
        None => {
            // First gradient
            let mut result = Tensor::new(grad.data().clone(), &tensor.shape);
            result.requires_grad = false;
            result
        }
    };

    *grad_ref = Some(Rc::new(new_grad));
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

        assert_eq!(k, other.shape[0], "Inner dimensions must match: {}x{} @ {}x{}",
                   m, k, other.shape[0], n);

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
                if let Some(grad_output) = out.grad.borrow().as_ref() {
                    let grad_out_data = grad_output.data().clone();

                    // Gradient for A: grad_out @ B^T
                    if a.requires_grad {
                        let mut grad_a = vec![0.0f32; a_shape[0] * a_shape[1]];

                        for i in 0..a_shape[0] {
                            for j in 0..a_shape[1] {
                                let mut sum = 0.0f32;
                                for k_idx in 0..b_shape[1] {
                                    // grad_out[i, k] * B[j, k]
                                    sum += grad_out_data[i * b_shape[1] + k_idx] *
                                           b_data[j * b_shape[1] + k_idx];
                                }
                                grad_a[i * a_shape[1] + j] = sum;
                            }
                        }

                        let mut grad_tensor = Tensor::new(grad_a, &a_shape);
                        grad_tensor.requires_grad = false;
                        accumulate_grad(&a, &grad_tensor);
                    }

                    // Gradient for B: A^T @ grad_out
                    if b.requires_grad {
                        let mut grad_b = vec![0.0f32; b_shape[0] * b_shape[1]];

                        for i in 0..b_shape[0] {
                            for j in 0..b_shape[1] {
                                let mut sum = 0.0f32;
                                for k_idx in 0..a_shape[0] {
                                    // A[k, i] * grad_out[k, j]
                                    sum += a_data[k_idx * a_shape[1] + i] *
                                           grad_out_data[k_idx * b_shape[1] + j];
                                }
                                grad_b[i * b_shape[1] + j] = sum;
                            }
                        }

                        let mut grad_tensor = Tensor::new(grad_b, &b_shape);
                        grad_tensor.requires_grad = false;
                        accumulate_grad(&b, &grad_tensor);
                    }
                }
            });
        }

        output
    }

    /// Create a random tensor with values from normal distribution
    pub fn randn(shape: &[usize]) -> Tensor {
        use rand_distr::StandardNormal;
        use rand_distr::Distribution;
        let mut rng = rand::thread_rng();

        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size)
                    .map(|_| StandardNormal.sample(&mut rng))
                    .collect();

        Tensor::new(data, shape)
    }

    /// ReLU activation function
    pub fn relu(&self) -> Tensor {
        let result_data: Vec<f32> = self.data()
            .iter()
            .map(|&x| x.max(0.0))
            .collect();

        let mut output = Tensor::new(result_data, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;

            let input = self.clone();
            let out = output.clone();
            let input_data = self.data().clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(grad_output) = out.grad.borrow().as_ref() {
                    let grad_data: Vec<f32> = grad_output.data()
                        .iter()
                        .zip(input_data.iter())
                        .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                        .collect();

                    let mut grad_tensor = Tensor::new(grad_data, &input.shape);
                    grad_tensor.requires_grad = false;
                    accumulate_grad(&input, &grad_tensor);
                }
            });
        }

        output
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        assert_eq!(self.data().len(), other.data().len(), "Tensor dimensions must match");

        let result_data: Vec<f32> = self.data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a - b)
            .collect();

        let mut output = Tensor::new(result_data, &self.shape);

        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;

            let a = self.clone();
            let b = other.clone();
            let out = output.clone();

            Tape::push_binary_op(self, other, &output, move || {
                if let Some(grad_output) = out.grad.borrow().as_ref() {
                    // d(a-b)/da = 1
                    if a.requires_grad {
                        accumulate_grad(&a, grad_output);
                    }
                    // d(a-b)/db = -1
                    if b.requires_grad {
                        let neg_grad: Vec<f32> = grad_output.data()
                            .iter()
                            .map(|g| -g)
                            .collect();
                        let mut grad_tensor = Tensor::new(neg_grad, &b.shape);
                        grad_tensor.requires_grad = false;
                        accumulate_grad(&b, &grad_tensor);
                    }
                }
            });
        }

        output
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
