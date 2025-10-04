//! QAT-aware layer implementations
//!
//! This module provides QAT-aware wrappers for existing neural network layers,
//! integrating fake quantization into the forward pass during training.

use super::qat_manager::global;
use super::{FakeQuantize, QATConfig};
use crate::{Tensor, nn::Module};

/// QAT-aware Linear layer that applies fake quantization during training
#[derive(Debug)]
pub struct QATLinear {
    /// Inner linear layer
    inner: crate::nn::Linear,
    /// Weight fake quantization
    weight_fake_quant: Option<FakeQuantize>,
    /// Activation fake quantization
    activation_fake_quant: Option<FakeQuantize>,
    /// Whether QAT is enabled for this layer
    qat_enabled: bool,
    /// Module identifier for QAT management
    module_id: String,
}

impl QATLinear {
    /// Create a new QAT-aware Linear layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        with_bias: bool,
        qat_config: &QATConfig,
        module_id: Option<String>,
    ) -> Self {
        let inner = crate::nn::Linear::new(in_features, out_features, with_bias);
        let module_id = module_id.unwrap_or_else(|| format!("linear_{}", in_features));

        let weight_fake_quant = if qat_config.is_enabled() {
            Some(FakeQuantize::new(
                qat_config.quant_config.clone(),
                qat_config.symmetric,
            ))
        } else {
            None
        };

        let activation_fake_quant = if qat_config.is_enabled() {
            Some(FakeQuantize::new(
                qat_config.quant_config.clone(),
                qat_config.symmetric,
            ))
        } else {
            None
        };

        Self {
            inner,
            weight_fake_quant,
            activation_fake_quant,
            qat_enabled: qat_config.is_enabled(),
            module_id,
        }
    }

    /// Enable or disable QAT for this layer
    pub fn enable_qat(&mut self, enabled: bool) {
        self.qat_enabled = enabled;
        global::set_module_qat(&self.module_id, enabled);
    }

    /// Check if QAT is enabled for this layer
    pub fn is_qat_enabled(&self) -> bool {
        self.qat_enabled && global::is_module_qat_enabled(&self.module_id)
    }

    /// Update quantization parameters for weights
    pub fn update_weight_params(&mut self) {
        if let Some(ref mut fq) = self.weight_fake_quant {
            let weight_data = self.inner.weight.data();
            fq.update_params(&weight_data);
        }
    }

    /// Update quantization parameters for activations
    pub fn update_activation_params(&mut self, activation_data: &[f32]) {
        if let Some(ref mut fq) = self.activation_fake_quant {
            fq.update_params(activation_data);
        }
    }
}

impl Module for QATLinear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Check if QAT is active
        let qat_active = self.is_qat_enabled() && global::is_training();

        if qat_active {
            // Apply fake quantization to weights
            let quantized_weight = if let Some(ref fq) = self.weight_fake_quant {
                fq.forward(&self.inner.weight)
            } else {
                self.inner.weight.clone()
            };

            // Perform linear transformation
            let mut output = input.matmul(&quantized_weight.transpose());
            if let Some(bias) = &self.inner.bias {
                output = output.add_broadcast(bias);
            }

            // Apply fake quantization to activations
            if let Some(ref fq) = self.activation_fake_quant {
                fq.forward(&output)
            } else {
                output
            }
        } else {
            // Standard forward pass without quantization
            self.inner.forward(input)
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.inner.parameters()
    }

    fn quantize(
        &self,
        qconfig: &crate::quantization::QuantizationConfig,
    ) -> Box<dyn crate::nn::QuantizedModule> {
        // Delegate to inner layer for actual quantization
        self.inner.quantize(qconfig)
    }
}

/// QAT-aware Conv2d layer that applies fake quantization during training
#[derive(Debug)]
pub struct QATConv2d {
    /// Inner conv2d layer
    inner: crate::nn::Conv2d,
    /// Weight fake quantization
    weight_fake_quant: Option<FakeQuantize>,
    /// Activation fake quantization
    activation_fake_quant: Option<FakeQuantize>,
    /// Whether QAT is enabled for this layer
    qat_enabled: bool,
    /// Module identifier for QAT management
    module_id: String,
}

impl QATConv2d {
    /// Create a new QAT-aware Conv2d layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: bool,
        qat_config: &QATConfig,
        module_id: Option<String>,
    ) -> Self {
        let inner = crate::nn::Conv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        );
        let module_id =
            module_id.unwrap_or_else(|| format!("conv2d_{}_{}", in_channels, out_channels));

        let weight_fake_quant = if qat_config.is_enabled() {
            Some(FakeQuantize::new(
                qat_config.quant_config.clone(),
                qat_config.symmetric,
            ))
        } else {
            None
        };

        let activation_fake_quant = if qat_config.is_enabled() {
            Some(FakeQuantize::new(
                qat_config.quant_config.clone(),
                qat_config.symmetric,
            ))
        } else {
            None
        };

        Self {
            inner,
            weight_fake_quant,
            activation_fake_quant,
            qat_enabled: qat_config.is_enabled(),
            module_id,
        }
    }

    /// Enable or disable QAT for this layer
    pub fn enable_qat(&mut self, enabled: bool) {
        self.qat_enabled = enabled;
        global::set_module_qat(&self.module_id, enabled);
    }

    /// Check if QAT is enabled for this layer
    pub fn is_qat_enabled(&self) -> bool {
        self.qat_enabled && global::is_module_qat_enabled(&self.module_id)
    }

    /// Update quantization parameters for weights
    pub fn update_weight_params(&mut self) {
        if let Some(ref mut fq) = self.weight_fake_quant {
            let weight_data = self.inner.weight.data();
            fq.update_params(&weight_data);
        }
    }

    /// Update quantization parameters for activations
    pub fn update_activation_params(&mut self, activation_data: &[f32]) {
        if let Some(ref mut fq) = self.activation_fake_quant {
            fq.update_params(activation_data);
        }
    }
}

impl Module for QATConv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Check if QAT is active
        let qat_active = self.is_qat_enabled() && global::is_training();

        if qat_active {
            // Apply fake quantization to weights
            let quantized_weight = if let Some(ref fq) = self.weight_fake_quant {
                fq.forward(&self.inner.weight)
            } else {
                self.inner.weight.clone()
            };

            // Perform convolution
            let output = input.conv2d(
                &quantized_weight,
                self.inner.bias.as_ref(),
                self.inner.stride,
                self.inner.padding,
                self.inner.dilation,
            );

            // Apply fake quantization to activations
            if let Some(ref fq) = self.activation_fake_quant {
                fq.forward(&output)
            } else {
                output
            }
        } else {
            // Standard forward pass without quantization
            self.inner.forward(input)
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.inner.parameters()
    }

    fn quantize(
        &self,
        qconfig: &crate::quantization::QuantizationConfig,
    ) -> Box<dyn crate::nn::QuantizedModule> {
        // Delegate to inner layer for actual quantization
        self.inner.quantize(qconfig)
    }
}

/// QAT-aware Sequential layer that applies fake quantization to all sub-layers
#[allow(dead_code)]
#[derive(Debug)]
pub struct QATSequential {
    /// Inner sequential layer
    inner: crate::nn::Sequential,
    /// QAT configuration
    qat_config: QATConfig,
    /// Whether QAT is enabled
    qat_enabled: bool,
    /// Module identifier
    module_id: String,
}

impl QATSequential {
    /// Create a new QAT-aware Sequential layer
    pub fn new(
        layers: Vec<Box<dyn Module>>,
        qat_config: QATConfig,
        module_id: Option<String>,
    ) -> Self {
        let inner = crate::nn::Sequential::new(layers);
        let module_id = module_id.unwrap_or_else(|| "sequential".to_string());

        let qat_enabled = qat_config.is_enabled();
        Self {
            inner,
            qat_config,
            qat_enabled,
            module_id,
        }
    }

    /// Enable or disable QAT for this layer
    pub fn enable_qat(&mut self, enabled: bool) {
        self.qat_enabled = enabled;
        global::set_module_qat(&self.module_id, enabled);
    }

    /// Check if QAT is enabled for this layer
    pub fn is_qat_enabled(&self) -> bool {
        self.qat_enabled && global::is_module_qat_enabled(&self.module_id)
    }
}

impl Module for QATSequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        // For sequential layers, we just pass through to the inner layer
        // Individual QAT-aware layers will handle their own quantization
        self.inner.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.inner.parameters()
    }

    fn quantize(
        &self,
        qconfig: &crate::quantization::QuantizationConfig,
    ) -> Box<dyn crate::nn::QuantizedModule> {
        self.inner.quantize(qconfig)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;

    #[test]
    fn test_qat_linear_creation() {
        let qat_config = QATConfig::int8(0.001, 5);
        let qat_linear = QATLinear::new(784, 128, true, &qat_config, None);

        assert!(qat_linear.is_qat_enabled());
        assert_eq!(qat_linear.module_id, "linear_784");
    }

    #[test]
    fn test_qat_linear_forward() {
        Tape::reset();

        let qat_config = QATConfig::int8(0.001, 5);
        let qat_linear = QATLinear::new(784, 128, true, &qat_config, None);
        let input = Tensor::randn(&[32, 784]).requires_grad();

        let output = qat_linear.forward(&input);

        assert_eq!(output.shape(), &[32, 128]);
        assert!(output.requires_grad);
    }

    #[test]
    fn test_qat_conv2d_creation() {
        let qat_config = QATConfig::int8(0.001, 5);
        let qat_conv = QATConv2d::new(
            1,
            32,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            None,
            None,
            true,
            &qat_config,
            None,
        );

        assert!(qat_conv.is_qat_enabled());
        assert_eq!(qat_conv.module_id, "conv2d_1_32");
    }

    #[test]
    fn test_qat_conv2d_forward() {
        Tape::reset();

        let qat_config = QATConfig::int8(0.001, 5);
        let qat_conv = QATConv2d::new(
            1,
            32,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            None,
            None,
            true,
            &qat_config,
            None,
        );
        let input = Tensor::randn(&[1, 1, 28, 28]).requires_grad();

        let output = qat_conv.forward(&input);

        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 32);
        assert!(output.requires_grad);
    }

    #[test]
    fn test_qat_enable_disable() {
        let qat_config = QATConfig::int8(0.001, 5);
        let mut qat_linear = QATLinear::new(784, 128, true, &qat_config, None);

        assert!(qat_linear.is_qat_enabled());

        qat_linear.enable_qat(false);
        assert!(!qat_linear.is_qat_enabled());

        qat_linear.enable_qat(true);
        assert!(qat_linear.is_qat_enabled());
    }
}
