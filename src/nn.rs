use crate::Tensor;
use rand::distributions::{Distribution, Uniform};

/// Trait for any differentiable network component.
pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
}

/// Linear (fully connected) layer: `y = xWᵀ + b`
#[derive(Debug)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, with_bias: bool) -> Self {
        // Xavier/He-style initialization
        let scale = (2.0 / in_features as f32).sqrt();
        let dist = Uniform::new_inclusive(-scale, scale);

        let mut rng = rand::thread_rng();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| dist.sample(&mut rng))
            .collect();

        let weight = Tensor::new(weight_data, &[out_features, in_features]).requires_grad();

        let bias = with_bias
            .then(|| Tensor::new(vec![0.0; out_features], &[out_features]).requires_grad());

        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.matmul(&self.weight.transpose());
        if let Some(b) = &self.bias {
            out = out.add_broadcast(b);
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
}

/// A stack of layers applied in sequence.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers.iter().fold(input.clone(), |x, l| l.forward(&x))
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
