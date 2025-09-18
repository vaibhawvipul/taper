use crate::{ops, tape::Tape};
use smallvec::SmallVec;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Clone)]
pub struct Tensor {
    data: Rc<RefCell<Vec<f32>>>,
    pub(crate) shape: SmallVec<[usize; 4]>,
    // In-place gradient accumulation buffer (allocated on demand)
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

    /// Zero-copy view of gradient buffer, if present.
    pub fn grad_ref(&self) -> Option<std::cell::Ref<'_, Vec<f32>>> {
        let r = self.grad.borrow();
        if r.is_some() {
            Some(std::cell::Ref::map(r, |opt| opt.as_ref().unwrap()))
        } else {
            None
        }
    }

    /// Compatibility accessor: materializes a Tensor from the grad buffer (allocates).
    /// Keep for old debug/prints; prefer `grad_ref()` for zero-copy.
    pub fn grad(&self) -> Option<Rc<Tensor>> {
        let r = self.grad.borrow();
        r.as_ref().map(|g| {
            let mut t = Tensor::new(g.clone(), &self.shape);
            t.requires_grad = false;
            Rc::new(t)
        })
    }

    pub fn backward(&self) {
        // Seed ∂L/∂self = 1
        let ones = vec![1.0; self.data().len()];
        *self.grad.borrow_mut() = Some(ones);

        // Walk the tape from the node that produced this tensor.
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

        // Forward
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
            let (rows, cols) = (rows, cols); // capture for closure

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.borrow().as_ref() {
                    // grad_input = transpose(grad_output)
                    let mut slot = input.grad.borrow_mut();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; rows * cols]);
                    }
                    let gin = slot.as_mut().unwrap();
                    // gout shape: [cols, rows], gin shape: [rows, cols]
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

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Tensor {
        // Forward: y = σ(x)
        let result_data: Vec<f32> = self
            .data()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        let mut output = Tensor::new(result_data, &self.shape);

        if self.requires_grad {
            output.requires_grad = true;

            let input = self.clone();
            let out = output.clone();

            Tape::push_unary_op(self, &output, move || {
                if let Some(gout) = out.grad.borrow().as_ref() {
                    let y = out.data(); // σ(x) from forward
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

    /// Supports adding [batch, features] + [features] -> [batch, features]
    pub fn add_broadcast(&self, other: &Tensor) -> Tensor {
        // Fast path: identical shapes
        if self.shape == other.shape {
            return self + other;
        }

        // Bias addition: [batch, features] + [features]
        if self.shape.len() == 2 && other.shape.len() == 1 {
            assert_eq!(
                self.shape[1], other.shape[0],
                "Last dimension must match for broadcasting"
            );

            let batch_size = self.shape[0];
            let features = self.shape[1];
            let self_data = self.data();
            let other_data = other.data();

            // Forward
            let mut result = vec![0.0; self_data.len()];
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
                let (batch_size, features) = (batch_size, features);

                // Use binary op so we record when *either* input requires grad.
                Tape::push_binary_op(self, other, &output, move || {
                    if let Some(gout) = out.grad.borrow().as_ref() {
                        // dL/dA = dL/dY
                        if a.requires_grad {
                            ops::accumulate_grad(&a, gout);
                        }

                        // dL/dB[f] = sum_b dL/dY[b,f]
                        if b.requires_grad {
                            let mut bias_grad = vec![0.0; features];
                            for batch in 0..batch_size {
                                for f in 0..features {
                                    bias_grad[f] += gout[batch * features + f];
                                }
                            }
                            ops::accumulate_grad(&b, &bias_grad);
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
                if let Some(gout) = out.grad.borrow().as_ref() {
                    // Each element gets gout / N
                    let g_each = gout[0] / n;
                    let grad_vec = vec![g_each; input.data().len()];
                    ops::accumulate_grad(&input, &grad_vec);
                }
            });
        }

        output
    }
}
