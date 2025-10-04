use crate::{Tensor, nn::{Module, QuantizedModule}, QuantizationConfig};

/// ReLU activation as a module
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn quantize(&self, _qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedReLU)
    }
}

/// Quantized ReLU for inference
pub struct QuantizedReLU;

impl QuantizedModule for QuantizedReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
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

    fn quantize(&self, _qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedSigmoid)
    }
}

/// Quantized Sigmoid for inference
pub struct QuantizedSigmoid;

impl QuantizedModule for QuantizedSigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
