use crate::Tensor;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32, _momentum: Option<f32>) -> Self {
        // TODO: Implement momentum
        SGD { params, lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for param in &self.params {
            if let Some(grad) = param.grad() {
                let mut param_data = param.data_mut();
                let grad_data = grad.data();

                // Vanilla SGD update: param = param - lr * grad
                for i in 0..param_data.len() {
                    param_data[i] -= self.lr * grad_data[i];
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
