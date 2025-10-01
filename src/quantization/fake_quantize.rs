//! Fake quantization implementation for Quantization-Aware Training
//! 
//! This module provides fake quantization operations that simulate quantization
//! effects during training while maintaining differentiability through the
//! Straight-Through Estimator (STE).

use crate::Tensor;
use super::config::{QuantizationConfig, QuantizationType};

/// Fake quantization module that simulates quantization during training
#[derive(Debug, Clone)]
pub struct FakeQuantize {
    /// Quantization configuration
    quant_config: QuantizationConfig,
    /// Scale factor for quantization
    scale: f32,
    /// Zero point for quantization
    zero_point: i32,
    /// Whether we're in training mode
    training: bool,
    /// Cached quantization range for efficiency
    qmin: i32,
    qmax: i32,
    /// Whether to use symmetric quantization
    symmetric: bool,
}

impl FakeQuantize {
    /// Create a new fake quantization module
    pub fn new(quant_config: QuantizationConfig, symmetric: bool) -> Self {
        let (qmin, qmax) = quant_config.compute_range().unwrap_or((-128, 127));
        
        Self {
            quant_config,
            scale: 1.0,
            zero_point: 0,
            training: true,
            qmin,
            qmax,
            symmetric,
        }
    }

    /// Create fake quantization for Int8
    pub fn int8(symmetric: bool) -> Self {
        Self::new(QuantizationConfig::int8(true), symmetric)
    }

    /// Create fake quantization for Int4
    pub fn int4(symmetric: bool) -> Self {
        Self::new(QuantizationConfig::int4(true), symmetric)
    }

    /// Create fake quantization for Float16
    pub fn float16() -> Self {
        Self::new(QuantizationConfig::float16(true), false)
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Update quantization parameters based on observed data
    pub fn update_params(&mut self, data: &[f32]) {
        if !self.quant_config.is_enabled() {
            return;
        }

        let (min_val, max_val) = self.compute_min_max(data);
        
        if self.symmetric {
            let max_abs = min_val.abs().max(max_val.abs());
            self.scale = max_abs / (self.qmax as f32);
            self.zero_point = 0;
        } else {
            self.scale = self.quant_config.compute_scale(min_val, max_val).unwrap_or(1.0);
            self.zero_point = self.quant_config.compute_zero_point(min_val, self.scale).unwrap_or(0);
        }
    }

    /// Compute min and max values from data
    fn compute_min_max(&self, data: &[f32]) -> (f32, f32) {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &val in data {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Handle edge case where all values are the same
        if min_val == max_val {
            if min_val == 0.0 {
                (0.0, 1.0)
            } else {
                (min_val * 0.9, min_val * 1.1)
            }
        } else {
            (min_val, max_val)
        }
    }

    /// Apply fake quantization to a tensor
    pub fn forward(&self, input: &Tensor) -> Tensor {
        if !self.quant_config.is_enabled() || !self.training {
            return input.clone();
        }

        let data = input.data();
        let mut result = vec![0.0; data.len()];

        match self.quant_config.quant_type {
            QuantizationType::Int8 | QuantizationType::Int4 | QuantizationType::NF4 => {
                self.quantize_integer(&data, &mut result);
            }
            QuantizationType::Float16 | QuantizationType::BFloat16 => {
                self.quantize_float(&data, &mut result);
            }
        }

        let mut output = Tensor::new(result, input.shape());
        
        // Set up gradient computation with straight-through estimator
        if input.requires_grad {
            output.requires_grad = true;
            let input_clone = input.clone();
            let output_clone = output.clone();
            
            // Use straight-through estimator: forward quantizes, backward passes through
            crate::tape::Tape::push_unary_op(input, &output, move || {
                if let Some(grad_output) = output_clone.grad_ref() {
                    // STE: gradient passes through unchanged
                    let mut slot = input_clone.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; grad_output.len()]);
                    }
                    let grad_input = slot.as_mut().unwrap();
                    
                    for (gi, &go) in grad_input.iter_mut().zip(grad_output.iter()) {
                        *gi += go;
                    }
                }
            });
        }

        output
    }

    /// Quantize integer types (Int8, Int4, NF4)
    fn quantize_integer(&self, input: &[f32], output: &mut [f32]) {
        for (i, &val) in input.iter().enumerate() {
            // Quantize: q = round((x - zero_point) / scale)
            let q = ((val / self.scale) + self.zero_point as f32).round() as i32;
            let q_clamped = q.clamp(self.qmin, self.qmax);
            
            // Dequantize: x' = (q - zero_point) * scale
            output[i] = (q_clamped - self.zero_point) as f32 * self.scale;
        }
    }

    /// Quantize float types (Float16, BFloat16)
    fn quantize_float(&self, input: &[f32], output: &mut [f32]) {
        // For float types, we simulate precision loss by truncating to target precision
        // This is a simplified implementation - real float16 conversion would be more complex
        for (i, &val) in input.iter().enumerate() {
            match self.quant_config.quant_type {
                QuantizationType::Float16 => {
                    // Simulate Float16 precision (simplified)
                    output[i] = self.simulate_float16(val);
                }
                QuantizationType::BFloat16 => {
                    // Simulate BFloat16 precision (simplified)
                    output[i] = self.simulate_bfloat16(val);
                }
                _ => output[i] = val,
            }
        }
    }

    /// Simulate Float16 precision loss
    fn simulate_float16(&self, val: f32) -> f32 {
        // Simplified Float16 simulation - in practice, this would involve
        // proper bit manipulation and rounding
        if val == 0.0 || !val.is_finite() {
            return val;
        }
        
        // Simulate reduced precision by rounding to fewer significant digits
        let magnitude = val.abs();
        if magnitude < 1e-8 {
            return 0.0;
        }
        
        let _exp = magnitude.log2().floor();
        let mantissa_bits = 10; // Float16 has 10 mantissa bits
        let scale = 2_f32.powi(mantissa_bits as i32);
        
        (val * scale).round() / scale
    }

    /// Simulate BFloat16 precision loss
    fn simulate_bfloat16(&self, val: f32) -> f32 {
        // Simplified BFloat16 simulation
        if val == 0.0 || !val.is_finite() {
            return val;
        }
        
        // BFloat16 has 7 mantissa bits (vs 10 for Float16)
        let magnitude = val.abs();
        if magnitude < 1e-8 {
            return 0.0;
        }
        
        let _exp = magnitude.log2().floor();
        let mantissa_bits = 7; // BFloat16 has 7 mantissa bits
        let scale = 2_f32.powi(mantissa_bits as i32);
        
        (val * scale).round() / scale
    }

    /// Get current scale
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get current zero point
    pub fn zero_point(&self) -> i32 {
        self.zero_point
    }

    /// Get quantization configuration
    pub fn quant_config(&self) -> &QuantizationConfig {
        &self.quant_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fake_quantize_creation() {
        let fq = FakeQuantize::int8(true);
        assert!(fq.is_training());
        assert_eq!(fq.quant_config.quant_type, QuantizationType::Int8);
    }

    #[test]
    fn test_fake_quantize_forward() {
        let fq = FakeQuantize::int8(true);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let output = fq.forward(&input);
        
        assert_eq!(output.shape(), input.shape());
        // Output should be quantized (different from input)
        assert_ne!(output.data()[0], input.data()[0]);
    }

    #[test]
    fn test_fake_quantize_training_mode() {
        let mut fq = FakeQuantize::int8(true);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        
        // In training mode, should quantize
        let output_train = fq.forward(&input);
        assert_ne!(output_train.data()[0], input.data()[0]);
        
        // In eval mode, should pass through
        fq.set_training(false);
        let output_eval = fq.forward(&input);
        assert_eq!(output_eval.data()[0], input.data()[0]);
    }

    #[test]
    fn test_update_params() {
        let mut fq = FakeQuantize::int8(true);
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        
        fq.update_params(&data);
        
        // Scale and zero point should be updated
        assert!(fq.scale() > 0.0);
        assert!(fq.zero_point() >= fq.qmin);
        assert!(fq.zero_point() <= fq.qmax);
    }
}
