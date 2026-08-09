//! Fake quantization implementation for Quantization-Aware Training
//!
//! This module provides fake quantization operations that simulate quantization
//! effects during training while maintaining differentiability through the
//! Straight-Through Estimator (STE).

use std::sync::RwLock;

use super::config::{QuantizationConfig, QuantizationType};
use super::observers::MinMaxObserver;
use crate::Tensor;

/// Quantization parameters, refined as the observer sees more data.
#[derive(Debug, Clone, Copy)]
struct QParams {
    scale: f32,
    zero_point: i32,
    /// False until the first observation; an uncalibrated quantizer would
    /// otherwise use `scale = 1.0`, which is a no-op for typical weight
    /// magnitudes and made QAT silently train an unquantized model.
    calibrated: bool,
}

/// Fake quantization module that simulates quantization during training
#[derive(Debug)]
pub struct FakeQuantize {
    /// Quantization configuration
    quant_config: QuantizationConfig,
    /// Live quantization parameters. Interior mutability lets `forward` — which
    /// only has `&self`, as `Module::forward` does — keep the observer current.
    params: RwLock<QParams>,
    /// Running min/max statistics across every tensor observed so far.
    observer: RwLock<MinMaxObserver>,
    /// Whether we're in training mode
    training: bool,
    /// Cached quantization range for efficiency
    qmin: i32,
    qmax: i32,
    /// Whether to use symmetric quantization
    symmetric: bool,
}

impl Clone for FakeQuantize {
    /// Clones own their calibration state; sharing it across layers would let
    /// one layer's activation range redefine another's.
    fn clone(&self) -> Self {
        Self {
            quant_config: self.quant_config.clone(),
            params: RwLock::new(*self.params.read().expect("qparams lock poisoned")),
            observer: RwLock::new(
                self.observer
                    .read()
                    .expect("observer lock poisoned")
                    .clone(),
            ),
            training: self.training,
            qmin: self.qmin,
            qmax: self.qmax,
            symmetric: self.symmetric,
        }
    }
}

impl FakeQuantize {
    /// Create a new fake quantization module
    pub fn new(quant_config: QuantizationConfig, symmetric: bool) -> Self {
        let (qmin, qmax) = quant_config.compute_range().unwrap_or((-128, 127));

        Self {
            quant_config,
            params: RwLock::new(QParams {
                scale: 1.0,
                zero_point: 0,
                calibrated: false,
            }),
            observer: RwLock::new(MinMaxObserver::new()),
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

    /// Whether the observer has seen enough data to derive a scale.
    pub fn is_calibrated(&self) -> bool {
        self.params
            .read()
            .expect("qparams lock poisoned")
            .calibrated
    }

    /// Update quantization parameters based on observed data
    pub fn update_params(&mut self, data: &[f32]) {
        self.observe(data);
    }

    /// Fold `data` into the running statistics and recompute scale/zero-point.
    fn observe(&self, data: &[f32]) {
        if !self.quant_config.is_enabled() {
            return;
        }

        let (min_val, max_val) = {
            let mut observer = self.observer.write().expect("observer lock poisoned");
            observer.observe_slice(data);
            observer.range()
        };
        let (min_val, max_val) = Self::widen_degenerate(min_val, max_val);

        let mut params = self.params.write().expect("qparams lock poisoned");
        if self.symmetric {
            let max_abs = min_val.abs().max(max_val.abs());
            // A tensor of all zeros gives max_abs == 0; dividing by the
            // resulting scale would produce NaN for every element.
            params.scale = if max_abs > 0.0 {
                max_abs / (self.qmax as f32)
            } else {
                1.0
            };
            params.zero_point = 0;
        } else {
            params.scale = self
                .quant_config
                .compute_scale(min_val, max_val)
                .unwrap_or(1.0);
            params.zero_point = self
                .quant_config
                .compute_zero_point(min_val, params.scale)
                .unwrap_or(0);
        }
        params.calibrated = true;
    }

    /// Nudge an empty or single-valued range into something with width.
    fn widen_degenerate(min_val: f32, max_val: f32) -> (f32, f32) {
        if !min_val.is_finite() || !max_val.is_finite() {
            // Empty input, or all NaN/inf: the observer sentinels are still
            // ±infinity, and any scale derived from them would be non-finite.
            return (0.0, 1.0);
        }
        if min_val == max_val {
            if min_val == 0.0 {
                (0.0, 1.0)
            } else {
                (
                    min_val.min(0.0) - min_val.abs() * 0.1,
                    max_val + max_val.abs() * 0.1,
                )
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

        // Observe before quantizing, exactly as a PyTorch observer would. This
        // is what makes the first forward pass produce a real scale instead of
        // reusing the placeholder 1.0.
        self.observe(&data);
        let params = *self.params.read().expect("qparams lock poisoned");

        let mut result = vec![0.0; data.len()];
        match self.quant_config.quant_type {
            QuantizationType::Int8 | QuantizationType::Int4 | QuantizationType::NF4 => {
                self.quantize_integer(&data, &mut result, params);
            }
            QuantizationType::Float16 | QuantizationType::BFloat16 => {
                self.quantize_float(&data, &mut result);
            }
        }
        drop(data);

        let mut output = Tensor::new(result, input.shape());

        // Set up gradient computation with straight-through estimator
        if input.requires_grad {
            output.requires_grad = true;
            let input_clone = input.clone();
            let output_clone = output.clone();

            // Use straight-through estimator: forward quantizes, backward passes through
            crate::tape::Tape::push_unary_op(input, &output, move || {
                if let Some(grad_output) = output_clone
                    .grad
                    .read()
                    .expect("grad RwLock poisoned")
                    .as_ref()
                {
                    // STE: gradient passes through unchanged
                    crate::ops::accumulate_grad(&input_clone, grad_output);
                }
            });
        }

        output
    }

    /// Quantize integer types (Int8, Int4, NF4)
    fn quantize_integer(&self, input: &[f32], output: &mut [f32], params: QParams) {
        for (out, &val) in output.iter_mut().zip(input) {
            // Quantize: q = round(x / scale) + zero_point
            let q = (val / params.scale).round() as i32 + params.zero_point;
            let q_clamped = q.clamp(self.qmin, self.qmax);

            // Dequantize: x' = (q - zero_point) * scale
            *out = (q_clamped - params.zero_point) as f32 * params.scale;
        }
    }

    /// Quantize float types (Float16, BFloat16) by an exact round trip through
    /// the narrower format, rather than approximating the precision loss.
    fn quantize_float(&self, input: &[f32], output: &mut [f32]) {
        match self.quant_config.quant_type {
            QuantizationType::Float16 => {
                for (out, &val) in output.iter_mut().zip(input) {
                    *out = Tensor::f16_to_f32(Tensor::f32_to_f16(val));
                }
            }
            QuantizationType::BFloat16 => {
                for (out, &val) in output.iter_mut().zip(input) {
                    *out = Tensor::bf16_to_f32(Tensor::f32_to_bf16(val));
                }
            }
            _ => output.copy_from_slice(input),
        }
    }

    /// Get current scale
    pub fn scale(&self) -> f32 {
        self.params.read().expect("qparams lock poisoned").scale
    }

    /// Get current zero point
    pub fn zero_point(&self) -> i32 {
        self.params
            .read()
            .expect("qparams lock poisoned")
            .zero_point
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
        assert!(!fq.is_calibrated());
    }

    #[test]
    fn test_fake_quantize_forward() {
        let fq = FakeQuantize::int8(true);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.7], &[2, 2]);
        let output = fq.forward(&input);

        assert_eq!(output.shape(), input.shape());
        assert!(fq.is_calibrated(), "forward must calibrate the observer");
        // Values land on the int8 grid: close to the input but not equal.
        // Probe an interior value — the extreme defines the scale, so it
        // round-trips exactly.
        let (a, b) = (output.data()[0], input.data()[0]);
        assert_ne!(a, b, "value passed through unquantized");
        assert!(
            (a - b).abs() < fq.scale(),
            "quantization error exceeds one step"
        );
    }

    #[test]
    fn test_fake_quantize_training_mode() {
        let mut fq = FakeQuantize::int8(true);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.7], &[2, 2]);

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

    #[test]
    fn asymmetric_int8_uses_the_whole_grid() {
        // With the zero point anchored at qmin, the extremes of the observed
        // range round-trip to within one quantization step. Deriving it as
        // `-min/scale` clipped everything above the midpoint to `max/2`.
        let fq = FakeQuantize::new(QuantizationConfig::int8(true), false);
        let input = Tensor::new(vec![-4.0, -1.0, 1.0, 4.0], &[4]);
        let out = fq.forward(&input);

        for (o, i) in out.data().iter().zip(input.data().iter()) {
            assert!(
                (o - i).abs() <= fq.scale(),
                "value {i} round-tripped to {o}, off by more than one step ({})",
                fq.scale()
            );
        }
    }

    #[test]
    fn all_zero_input_does_not_produce_nan() {
        for symmetric in [true, false] {
            let fq = FakeQuantize::new(QuantizationConfig::int8(true), symmetric);
            let out = fq.forward(&Tensor::new(vec![0.0; 4], &[4]));
            assert!(
                out.data().iter().all(|v| v.is_finite()),
                "symmetric={symmetric} produced non-finite output"
            );
        }
    }

    #[test]
    fn empty_input_does_not_produce_a_non_finite_scale() {
        let fq = FakeQuantize::int8(true);
        let _ = fq.forward(&Tensor::new(vec![], &[0]));
        assert!(fq.scale().is_finite() && fq.scale() > 0.0);
    }
}
