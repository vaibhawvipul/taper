use crate::{Tensor, nn::Module};

/// ReLU activation as a module
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![] // No parameters
    }
}

/// Sigmoid activation
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
