use std::cell::{RefCell, Cell};
use std::rc::Rc;
use smallvec::SmallVec;
use crate::{tape::Tape, ops};

#[derive(Clone)]
pub struct Tensor {
    data: Rc<RefCell<Vec<f32>>>,
    pub(crate) shape: SmallVec<[usize; 4]>,
    pub grad: Rc<RefCell<Option<Rc<Tensor>>>>,  // Changed to Rc<RefCell<...>>
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

    pub fn grad(&self) -> Option<Rc<Tensor>> {
        self.grad.borrow().clone()
    }

    pub fn backward(&self) {
        // Initialize gradient to 1.0 for the output
        let ones = vec![1.0; self.data().len()];
        let mut init_grad = Tensor::new(ones, &self.shape);
        init_grad.requires_grad = false;  // IMPORTANT: Gradients never require grad
        *self.grad.borrow_mut() = Some(Rc::new(init_grad));

        // Perform backward pass
        if let Some(node_id) = self.tape_node.get() {
            crate::tape::backward(node_id);
        }
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// Create tensor from data with same properties as self
    pub fn from_data(&self, data: Vec<f32>, shape: &[usize]) -> Tensor {
        let mut tensor = Tensor::new(data, shape);
        if self.requires_grad {
            tensor.requires_grad = true;
        }
        tensor
    }

    /// Get mutable access to data
    pub fn data_mut(&self) -> std::cell::RefMut<'_, Vec<f32>> {
        self.data.borrow_mut()
    }

    /// Transpose a 2D tensor
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Can only transpose 2D tensors");

        let rows = self.shape[0];
        let cols = self.shape[1];
        let data = self.data();

        let mut result = vec![0.0; data.len()];
        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = data[i * cols + j];
            }
        }

        let mut output = Tensor::new(result, &[cols, rows]);

        if self.requires_grad {
            output.requires_grad = true;

            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(grad_output) = out.grad.borrow().as_ref() {
                    // Gradient of transpose is just transpose
                    let grad_t = grad_output.transpose();
                    ops::accumulate_grad(&input, &grad_t);
                }
            });
        }

        output
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Tensor {
        let result_data: Vec<f32> = self.data()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        let sigmoid_data = result_data.clone();
        let mut output = Tensor::new(result_data, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;

            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(grad_output) = out.grad.borrow().as_ref() {
                    // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
                    let grad_data: Vec<f32> = grad_output.data()
                        .iter()
                        .zip(sigmoid_data.iter())
                        .map(|(g, s)| g * s * (1.0 - s))
                        .collect();

                    let mut grad_tensor = Tensor::new(grad_data, &input.shape);
                    grad_tensor.requires_grad = false;
                    ops::accumulate_grad(&input, &grad_tensor);
                }
            });
        }

        output
    }

    /// Supports adding [batch, features] + [features] -> [batch, features]
    pub fn add_broadcast(&self, other: &Tensor) -> Tensor {
        // Check if broadcasting is needed
        if self.shape == other.shape {
            // Same shape, use regular addition
            return self + other;
        }

        // Handle bias addition: [batch, features] + [features]
        if self.shape.len() == 2 && other.shape.len() == 1 {
            assert_eq!(self.shape[1], other.shape[0],
                      "Last dimension must match for broadcasting");

            let batch_size = self.shape[0];
            let features = self.shape[1];
            let self_data = self.data();
            let other_data = other.data();

            let mut result = vec![0.0; self_data.len()];

            // Add bias to each batch
            for b in 0..batch_size {
                for f in 0..features {
                    let idx = b * features + f;
                    result[idx] = self_data[idx] + other_data[f];
                }
            }

            let mut output = Tensor::new(result, &self.shape);

            if self.requires_grad || other.requires_grad {
                output.requires_grad = true;

                let a = self.clone();
                let b = other.clone();
                let out = output.clone();
                let batch_size = batch_size;

                Tape::push_unary_op(other, &output, move || {
                    if let Some(grad_output) = out.grad.borrow().as_ref() {
                        // Gradient for input: same as output
                        if a.requires_grad {
                            ops::accumulate_grad(&a, grad_output);
                        }

                        // Gradient for bias: sum over batch dimension
                        if b.requires_grad {
                            let grad_out_data = grad_output.data();
                            let features = b.shape[0];
                            let mut bias_grad = vec![0.0; features];

                            for batch in 0..batch_size {
                                for f in 0..features {
                                    bias_grad[f] += grad_out_data[batch * features + f];
                                }
                            }

                            let mut grad_tensor = Tensor::new(bias_grad, &b.shape);
                            grad_tensor.requires_grad = false;
                            ops::accumulate_grad(&b, &grad_tensor);
                        }
                    }
                });
            }

            output
        } else {
            panic!("Unsupported broadcasting shapes: {:?} and {:?}",
                   self.shape, other.shape);
        }
    }

    /// Mean of all elements
    pub fn mean(&self) -> Tensor {
        let data = self.data();
        let mean_val = data.iter().sum::<f32>() / data.len() as f32;

        let mut output = Tensor::scalar(mean_val);

        if self.requires_grad {
            output.requires_grad = true;

            let input = self.clone();
            let out = output.clone();
            let n = data.len() as f32;

            Tape::push_unary_op(self, &output, move || {
                if let Some(grad_output) = out.grad.borrow().as_ref() {
                    // Gradient of mean is grad_out / n for all elements
                    let grad_val = grad_output.data()[0] / n;
                    let grad_data = vec![grad_val; input.data().len()];

                    let mut grad_tensor = Tensor::new(grad_data, &input.shape);
                    grad_tensor.requires_grad = false;
                    ops::accumulate_grad(&input, &grad_tensor);
                }
            });
        }

        output
    }
}
