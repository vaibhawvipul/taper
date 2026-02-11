use crate::{Tensor, nn::Module, tape::Tape, ops};

/// Batch Normalization for 2D convolutions (simplified version)
/// 
/// Normalizes inputs across the batch dimension and spatial dimensions,
/// independently for each channel.
/// 
/// Input shape: [N, C, H, W]
/// - N: batch size
/// - C: number of channels (num_features)
/// - H, W: spatial dimensions
/// 
/// This is a simplified version that:
/// - Only supports training mode (computes batch statistics)
/// - Has proper backward pass for gamma, beta, and input
/// - Does not maintain running statistics (no eval mode yet)
pub struct BatchNorm2d {
    num_features: usize,
    
    /// Learnable scale parameter (gamma), shape: [num_features]
    pub gamma: Tensor,
    
    /// Learnable shift parameter (beta), shape: [num_features]
    pub beta: Tensor,
    
    /// Small constant for numerical stability
    epsilon: f32,
}

impl BatchNorm2d {
    /// Create a new BatchNorm2d layer
    /// 
    /// # Arguments
    /// * `num_features` - Number of channels (C dimension)
    /// 
    /// # Example
    /// ```
    /// let bn = BatchNorm2d::new(64);  // For 64 channels
    /// ```
    pub fn new(num_features: usize) -> Self {
        Self::with_epsilon(num_features, 1e-5)
    }
    
    /// Create a BatchNorm2d layer with custom epsilon
    pub fn with_epsilon(num_features: usize, epsilon: f32) -> Self {
        BatchNorm2d {
            num_features,
            gamma: Tensor::new(vec![1.0; num_features], &[num_features])
                .requires_grad(),
            beta: Tensor::new(vec![0.0; num_features], &[num_features])
                .requires_grad(),
            epsilon,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape().len(), 
            4, 
            "BatchNorm2d expects 4D input [N, C, H, W], got shape {:?}", 
            input.shape()
        );
        
        let shape = input.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        
        assert_eq!(
            c, self.num_features,
            "Input channels {} must match num_features {}",
            c, self.num_features
        );
        
        let input_data = input.data();
        let mut output_data = vec![0.0; n * c * h * w];
        
        let gamma_data = self.gamma.data();
        let beta_data = self.beta.data();
        
        // Store batch statistics for backward pass
        let mut means = vec![0.0; c];
        let mut stds = vec![0.0; c];
        let mut normalized = vec![0.0; n * c * h * w];
        
        // Process each channel independently
        for ch in 0..c {
            // Step 1: Compute mean for this channel across (N, H, W)
            let mut sum = 0.0;
            let count = (n * h * w) as f32;
            
            for b in 0..n {
                for i in 0..(h * w) {
                    let idx = b * c * h * w + ch * h * w + i;
                    sum += input_data[idx];
                }
            }
            let mean = sum / count;
            means[ch] = mean;
            
            // Step 2: Compute variance for this channel
            let mut var_sum = 0.0;
            for b in 0..n {
                for i in 0..(h * w) {
                    let idx = b * c * h * w + ch * h * w + i;
                    let diff = input_data[idx] - mean;
                    var_sum += diff * diff;
                }
            }
            let variance = var_sum / count;
            let std = (variance + self.epsilon).sqrt();
            stds[ch] = std;
            
            // Step 3: Normalize, scale, and shift
            let gamma_val = gamma_data[ch];
            let beta_val = beta_data[ch];
            
            for b in 0..n {
                for i in 0..(h * w) {
                    let idx = b * c * h * w + ch * h * w + i;
                    // Normalize: (x - mean) / std
                    let norm_val = (input_data[idx] - mean) / std;
                    normalized[idx] = norm_val;
                    // Scale and shift: gamma * normalized + beta
                    output_data[idx] = gamma_val * norm_val + beta_val;
                }
            }
        }
        
        let mut output = Tensor::new(output_data, shape);
        
        // Attach gradients
        if input.requires_grad || self.gamma.requires_grad || self.beta.requires_grad {
            output.requires_grad = true;
            
            let inp = input.clone();
            let gamma = self.gamma.clone();
            let beta = self.beta.clone();
            let out = output.clone();
            
            Tape::push_unary_op(&inp.clone(), &output, move || {
                if let Some(grad_out) = out.grad.read().unwrap().as_ref() {
                    let grad_out_data = grad_out;
                    
                    // Backward pass for beta: d_beta = sum(grad_out) per channel
                    if beta.requires_grad {
                        let mut d_beta = vec![0.0; c];
                        for ch in 0..c {
                            for b in 0..n {
                                for i in 0..(h * w) {
                                    let idx = b * c * h * w + ch * h * w + i;
                                    d_beta[ch] += grad_out_data[idx];
                                }
                            }
                        }
                        ops::accumulate_grad(&beta, &d_beta);
                    }
                    
                    // Backward pass for gamma: d_gamma = sum(grad_out * normalized) per channel
                    if gamma.requires_grad {
                        let mut d_gamma = vec![0.0; c];
                        for ch in 0..c {
                            for b in 0..n {
                                for i in 0..(h * w) {
                                    let idx = b * c * h * w + ch * h * w + i;
                                    d_gamma[ch] += grad_out_data[idx] * normalized[idx];
                                }
                            }
                        }
                        ops::accumulate_grad(&gamma, &d_gamma);
                    }
                    
                    // Backward pass for input (most complex)
                    if inp.requires_grad {
                        let mut d_input = vec![0.0; n * c * h * w];
                        let count = (n * h * w) as f32;
                        let gamma_data = gamma.data();
                        
                        for ch in 0..c {
                            let _mean = means[ch];
                            let std = stds[ch];
                            let gamma_val = gamma_data[ch];
                            
                            // Compute mean of grad_out for this channel
                            let mut grad_sum = 0.0;
                            for b in 0..n {
                                for i in 0..(h * w) {
                                    let idx = b * c * h * w + ch * h * w + i;
                                    grad_sum += grad_out_data[idx];
                                }
                            }
                            let grad_mean = grad_sum / count;
                            
                            // Compute mean of (grad_out * normalized) for this channel
                            let mut grad_norm_sum = 0.0;
                            for b in 0..n {
                                for i in 0..(h * w) {
                                    let idx = b * c * h * w + ch * h * w + i;
                                    grad_norm_sum += grad_out_data[idx] * normalized[idx];
                                }
                            }
                            let grad_norm_mean = grad_norm_sum / count;
                            
                            // Compute gradient for input
                            // Formula: dx = (gamma/std) * [grad_out - grad_mean - normalized * grad_norm_mean]
                            let scale = gamma_val / std;
                            for b in 0..n {
                                for i in 0..(h * w) {
                                    let idx = b * c * h * w + ch * h * w + i;
                                    d_input[idx] = scale * (
                                        grad_out_data[idx] 
                                        - grad_mean 
                                        - normalized[idx] * grad_norm_mean
                                    );
                                }
                            }
                        }
                        ops::accumulate_grad(&inp, &d_input);
                    }
                }
            });
        }
        
        output
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

impl std::fmt::Debug for BatchNorm2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchNorm2d")
            .field("num_features", &self.num_features)
            .field("epsilon", &self.epsilon)
            .finish()
    }
}


