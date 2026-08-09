use std::sync::{Arc, RwLock};

use crate::{QuantizationConfig, QuantizedTensor, Tensor};
use rand::{
    Rng,
    distributions::{Distribution, Uniform},
};

/// Trait for any differentiable network component.
pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;

    /// Switch this module (and any children) between training and inference
    /// behaviour. Layers whose forward pass is mode-independent can ignore it.
    ///
    /// Without this, a `Dropout` inside a `Sequential` could never be turned
    /// off — `forward` takes `&self`, so evaluation kept sampling masks and
    /// reported noisy metrics.
    fn set_training(&mut self, _training: bool) {}

    /// Quantize the model for inference
    fn quantize(&self, _qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        panic!("Quantization not implemented for this module type")
    }
}

/// Trait for quantized modules used during inference
pub trait QuantizedModule {
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
        assert!(in_features > 0, "Linear: in_features must be non-zero");
        // He-uniform: U(-b, b) has variance b²/3, so b = sqrt(6/fan_in) is what
        // gives the target variance 2/fan_in. Using sqrt(2/fan_in) as the bound
        // (as this did) makes the initial variance 3x too small, which slows
        // early training in deep stacks. Conv2d already derives its bound this way.
        let bound = (6.0 / in_features as f32).sqrt();
        let dist = Uniform::new_inclusive(-bound, bound);

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

    fn quantize(&self, qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedLinear {
            weight: self.weight.quantize(qconfig),
            bias: self.bias.as_ref().map(|b| b.quantize(qconfig)),
            cached_weight: Arc::new(RwLock::new(None)),
            cached_bias: Arc::new(RwLock::new(None)), // Initialize cache
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }
}

/// Quantized Linear layer for inference
pub struct QuantizedLinear {
    weight: QuantizedTensor,
    bias: Option<QuantizedTensor>,
    cached_weight: Arc<RwLock<Option<Tensor>>>,
    cached_bias: Arc<RwLock<Option<Tensor>>>,
}

impl QuantizedModule for QuantizedLinear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Cache dequantized weight
        let weight_f32 = {
            let cache = self.cached_weight.read().unwrap();
            if let Some(ref w) = *cache {
                w.clone()
            } else {
                drop(cache);
                let w = self.weight.dequantize();
                *self.cached_weight.write().unwrap() = Some(w.clone());
                w
            }
        };

        let mut out = input.matmul(&weight_f32.transpose());

        // Cache and apply bias if present
        if let Some(ref bias_q) = self.bias {
            let bias_f32 = {
                let cache = self.cached_bias.read().unwrap();
                if let Some(ref b) = *cache {
                    b.clone()
                } else {
                    drop(cache);
                    let b = bias_q.dequantize();
                    *self.cached_bias.write().unwrap() = Some(b.clone());
                    b
                }
            };
            out = out.add_broadcast(&bias_f32);
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// A stack of layers applied in sequence.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("layers", &self.layers.len())
            .finish()
    }
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

    fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    fn quantize(&self, qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedSequential {
            layers: self.layers.iter().map(|l| l.quantize(qconfig)).collect(),
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

/// Quantized Sequential layer for inference
pub struct QuantizedSequential {
    layers: Vec<Box<dyn QuantizedModule>>,
}

impl QuantizedModule for QuantizedSequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers.iter().fold(input.clone(), |x, l| l.forward(&x))
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

#[derive(Debug)]
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)] // mirrors the standard conv2d signature
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: bool,
    ) -> Self {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));
        let groups = groups.unwrap_or(1);

        assert_eq!(
            in_channels % groups,
            0,
            "in_channels must be divisible by groups"
        );
        assert_eq!(
            out_channels % groups,
            0,
            "out_channels must be divisible by groups"
        );

        let (k_h, k_w) = kernel_size;

        // Kaiming/He initialization for ReLU networks
        let fan_in = in_channels * k_h * k_w / groups;
        let std = (2.0 / fan_in as f32).sqrt();
        let bound = std * (3.0_f32).sqrt(); // uniform distribution bounds

        let dist = Uniform::new_inclusive(-bound, bound);
        let mut rng = rand::thread_rng();

        let weight_data: Vec<f32> = (0..out_channels * in_channels * k_h * k_w / groups)
            .map(|_| dist.sample(&mut rng))
            .collect();

        let weight = Tensor::new(weight_data, &[out_channels, in_channels / groups, k_h, k_w])
            .requires_grad();

        let bias = if bias {
            Some(Tensor::new(vec![0.0; out_channels], &[out_channels]).requires_grad())
        } else {
            None
        };

        Conv2d {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Create a 3x3 conv layer with common defaults
    pub fn conv3x3(in_channels: usize, out_channels: usize, stride: usize, padding: usize) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (3, 3),
            Some((stride, stride)),
            Some((padding, padding)),
            None,
            None,
            true,
        )
    }

    /// Create a 1x1 conv layer for channel adjustment
    pub fn conv1x1(in_channels: usize, out_channels: usize) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (1, 1),
            None,
            None,
            None,
            None,
            true,
        )
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.groups == 1 {
            // Standard convolution
            input.conv2d(
                &self.weight,
                self.bias.as_ref(),
                self.stride,
                self.padding,
                self.dilation,
            )
        } else {
            // Grouped convolution - split input and weight, convolve separately, then concatenate
            let (_n, c_in, _h, _w) = (
                input.shape()[0],
                input.shape()[1],
                input.shape()[2],
                input.shape()[3],
            );
            let c_out = self.weight.shape()[0];
            let c_in_per_group = c_in / self.groups;
            let c_out_per_group = c_out / self.groups;

            let mut group_outputs = Vec::new();

            for g in 0..self.groups {
                // Extract input channels for this group
                let input_slice =
                    input.slice_channels(g * c_in_per_group, (g + 1) * c_in_per_group);

                // Extract weight channels for this group
                let weight_slice = self
                    .weight
                    .slice_output_channels(g * c_out_per_group, (g + 1) * c_out_per_group);

                // Extract bias for this group
                let bias_slice = self
                    .bias
                    .as_ref()
                    .map(|b| b.slice_1d(g * c_out_per_group, (g + 1) * c_out_per_group));

                // Convolve
                let group_out = input_slice.conv2d(
                    &weight_slice,
                    bias_slice.as_ref(),
                    self.stride,
                    self.padding,
                    self.dilation,
                );

                group_outputs.push(group_out);
            }

            // Concatenate along channel dimension
            Tensor::cat(&group_outputs, 1)
        }
    }

    fn quantize(&self, qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedConv2d {
            weight: self.weight.quantize(qconfig),
            bias: self.bias.as_ref().map(|b| b.quantize(qconfig)),
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}

/// Quantized Conv2d layer for inference
pub struct QuantizedConv2d {
    weight: crate::tensor::QuantizedTensor,
    bias: Option<crate::tensor::QuantizedTensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
}

impl QuantizedModule for QuantizedConv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight_f32 = self.weight.dequantize();
        let bias_f32 = self.bias.as_ref().map(|b| b.dequantize());

        if self.groups == 1 {
            // Standard convolution
            input.conv2d(
                &weight_f32,
                bias_f32.as_ref(),
                self.stride,
                self.padding,
                self.dilation,
            )
        } else {
            // Grouped convolution - split input and weight, convolve separately, then concatenate
            let (_n, c_in, _h, _w) = (
                input.shape()[0],
                input.shape()[1],
                input.shape()[2],
                input.shape()[3],
            );
            let c_out = weight_f32.shape()[0];
            let c_in_per_group = c_in / self.groups;
            let c_out_per_group = c_out / self.groups;

            let mut group_outputs = Vec::new();

            for g in 0..self.groups {
                // Extract input channels for this group
                let input_slice =
                    input.slice_channels(g * c_in_per_group, (g + 1) * c_in_per_group);

                // Extract weight channels for this group
                let weight_slice = weight_f32
                    .slice_output_channels(g * c_out_per_group, (g + 1) * c_out_per_group);

                // Extract bias for this group
                let bias_slice = bias_f32
                    .as_ref()
                    .map(|b| b.slice_1d(g * c_out_per_group, (g + 1) * c_out_per_group));

                // Convolve
                let group_out = input_slice.conv2d(
                    &weight_slice,
                    bias_slice.as_ref(),
                    self.stride,
                    self.padding,
                    self.dilation,
                );

                group_outputs.push(group_out);
            }

            // Concatenate along channel dimension
            Tensor::cat(&group_outputs, 1)
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Return empty for quantized modules since parameters are stored as QuantizedTensor
        vec![]
    }
}

/// Fused Conv2d + ReLU layer for better performance
#[derive(Debug)]
pub struct Conv2dReLU {
    conv: Conv2d,
}

impl Conv2dReLU {
    #[allow(clippy::too_many_arguments)] // mirrors the standard conv2d signature
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: bool,
    ) -> Self {
        Conv2dReLU {
            conv: Conv2d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
        }
    }

    pub fn conv3x3(in_channels: usize, out_channels: usize, stride: usize, padding: usize) -> Self {
        Conv2dReLU {
            conv: Conv2d::conv3x3(in_channels, out_channels, stride, padding),
        }
    }
}

impl Module for Conv2dReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Use fused conv+relu operation
        input.conv2d_relu(
            &self.conv.weight,
            self.conv.bias.as_ref(),
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
        )
    }

    fn quantize(&self, qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedConv2dReLU {
            conv: self.conv.quantize(qconfig),
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.conv.parameters()
    }
}

pub struct QuantizedConv2dReLU {
    conv: Box<dyn QuantizedModule>,
}

impl QuantizedModule for QuantizedConv2dReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.conv.forward(input).relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// 2D Max Pooling layer
#[derive(Debug)]
pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: Option<(usize, usize)>,
    pub padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
            padding: padding.unwrap_or((0, 0)),
        }
    }

    /// Common 2x2 max pooling
    pub fn new_2x2() -> Self {
        Self::new((2, 2), Some((2, 2)), None)
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.max_pool2d(self.kernel_size, self.stride, self.padding)
    }

    fn quantize(&self, _qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedMaxPool2d {
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Quantized MaxPool2d layer for inference
pub struct QuantizedMaxPool2d {
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
}

impl QuantizedModule for QuantizedMaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.max_pool2d(self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// 2D Average Pooling layer
#[derive(Debug)]
pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: Option<(usize, usize)>,
    pub padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> Self {
        AvgPool2d {
            kernel_size,
            stride,
            padding: padding.unwrap_or((0, 0)),
        }
    }

    /// Global average pooling - reduces spatial dimensions to 1x1
    pub fn global() -> Self {
        // This will need special handling in forward pass
        Self::new((0, 0), None, None)
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.kernel_size == (0, 0) {
            // Global average pooling
            let (_n, _c, h, w) = (
                input.shape()[0],
                input.shape()[1],
                input.shape()[2],
                input.shape()[3],
            );
            input.avg_pool2d((h, w), Some((1, 1)), (0, 0))
        } else {
            input.avg_pool2d(self.kernel_size, self.stride, self.padding)
        }
    }

    fn quantize(&self, _qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedAvgPool2d {
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Quantized AvgPool2d layer for inference
pub struct QuantizedAvgPool2d {
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
}

impl QuantizedModule for QuantizedAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.kernel_size == (0, 0) {
            // Global average pooling
            let (_n, _c, h, w) = (
                input.shape()[0],
                input.shape()[1],
                input.shape()[2],
                input.shape()[3],
            );
            input.avg_pool2d((h, w), Some((1, 1)), (0, 0))
        } else {
            input.avg_pool2d(self.kernel_size, self.stride, self.padding)
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Adaptive Average Pooling that outputs a specific size
#[derive(Debug)]
pub struct AdaptiveAvgPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        AdaptiveAvgPool2d { output_size }
    }

    pub fn global() -> Self {
        Self::new((1, 1))
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (_n, _c, h_in, w_in) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (h_out, w_out) = self.output_size;
        assert!(
            h_out > 0 && w_out > 0,
            "AdaptiveAvgPool2d: output size must be non-zero, got {:?}",
            self.output_size
        );
        // This lowers to a strided avg_pool, which can only hit the requested
        // output exactly when the input divides evenly. Otherwise the pool
        // silently produced a different shape than `output_size` advertised.
        assert!(
            h_in % h_out == 0 && w_in % w_out == 0,
            "AdaptiveAvgPool2d: input {}x{} is not divisible by output {}x{}",
            h_in,
            w_in,
            h_out,
            w_out
        );

        // Calculate kernel size and stride to achieve target output size
        let kernel_h = h_in / h_out;
        let kernel_w = w_in / w_out;

        input.avg_pool2d((kernel_h, kernel_w), Some((kernel_h, kernel_w)), (0, 0))
    }

    fn quantize(&self, _qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedAdaptiveAvgPool2d {
            output_size: self.output_size,
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Quantized AdaptiveAvgPool2d layer for inference
pub struct QuantizedAdaptiveAvgPool2d {
    output_size: (usize, usize),
}

impl QuantizedModule for QuantizedAdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (_n, _c, h_in, w_in) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (h_out, w_out) = self.output_size;
        assert!(
            h_out > 0 && w_out > 0,
            "AdaptiveAvgPool2d: output size must be non-zero, got {:?}",
            self.output_size
        );
        // This lowers to a strided avg_pool, which can only hit the requested
        // output exactly when the input divides evenly. Otherwise the pool
        // silently produced a different shape than `output_size` advertised.
        assert!(
            h_in % h_out == 0 && w_in % w_out == 0,
            "AdaptiveAvgPool2d: input {}x{} is not divisible by output {}x{}",
            h_in,
            w_in,
            h_out,
            w_out
        );

        // Calculate kernel size and stride to achieve target output size
        let kernel_h = h_in / h_out;
        let kernel_w = w_in / w_out;

        input.avg_pool2d((kernel_h, kernel_w), Some((kernel_h, kernel_w)), (0, 0))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Flatten layer to convert from 4D to 2D for fully connected layers
#[derive(Debug)]
pub struct Flatten {
    start_dim: usize,
}

impl Flatten {
    pub fn new(start_dim: Option<usize>) -> Self {
        Flatten {
            start_dim: start_dim.unwrap_or(1),
        }
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.flatten(self.start_dim)
    }

    fn quantize(&self, _qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedFlatten {
            start_dim: self.start_dim,
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Quantized Flatten layer for inference
pub struct QuantizedFlatten {
    start_dim: usize,
}

impl QuantizedModule for QuantizedFlatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.flatten(self.start_dim)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Dropout layer for regularization
#[derive(Debug)]
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be between 0 and 1"
        );
        Dropout { p, training: true }
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        if self.p == 1.0 {
            return Tensor::new(vec![0.0; input.numel()], input.shape());
        }

        // Create dropout mask
        let data = input.data();
        let mut rng = rand::thread_rng();
        let mut mask_data = vec![0.0; data.len()];
        let scale = 1.0 / (1.0 - self.p);

        for mask_val in mask_data.iter_mut() {
            if rng.r#gen::<f32>() > self.p {
                *mask_val = scale;
            }
        }

        let mask = Tensor::new(mask_data, input.shape());
        input * &mask
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

// Helper implementations for tensor operations needed by grouped convolution
impl Tensor {
    /// Extract a slice of channels from a 4D tensor
    pub fn slice_channels(&self, start: usize, end: usize) -> Tensor {
        assert_eq!(
            self.shape.len(),
            4,
            "slice_channels only works on 4D tensors"
        );
        assert!(start < end && end <= self.shape[1], "Invalid channel range");

        self.narrow(1, start, end - start)
    }

    /// Extract output channels from weight tensor
    pub fn slice_output_channels(&self, start: usize, end: usize) -> Tensor {
        assert_eq!(
            self.shape.len(),
            4,
            "slice_output_channels only works on 4D weight tensors"
        );
        assert!(
            start < end && end <= self.shape[0],
            "Invalid output channel range"
        );

        self.narrow(0, start, end - start)
    }

    /// Extract slice from 1D tensor (for bias)
    pub fn slice_1d(&self, start: usize, end: usize) -> Tensor {
        assert_eq!(self.shape.len(), 1, "slice_1d only works on 1D tensors");
        assert!(start < end && end <= self.shape[0], "Invalid range");

        self.narrow(0, start, end - start)
    }

    /// Concatenate tensors along specified dimension
    pub fn cat(tensors: &[Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty(), "Cannot concatenate empty tensor list");

        let first_shape = &tensors[0].shape;
        let ndim = first_shape.len();
        assert!(dim < ndim, "Concatenation dimension out of bounds");

        // Verify all tensors have compatible shapes
        let mut total_dim_size = 0;
        for tensor in tensors {
            assert_eq!(
                tensor.shape.len(),
                ndim,
                "All tensors must have same number of dimensions"
            );
            for (i, (&s1, &s2)) in first_shape.iter().zip(tensor.shape.iter()).enumerate() {
                if i != dim {
                    assert_eq!(
                        s1, s2,
                        "Tensor shapes must match except in concatenation dimension"
                    );
                }
            }
            total_dim_size += tensor.shape[dim];
        }

        // Calculate output shape
        let mut out_shape = first_shape.clone();
        out_shape[dim] = total_dim_size;

        // The outer dimensions (before `dim`) are traversed in lockstep across
        // the inputs; everything from `dim` onwards is contiguous per input.
        // Expressing it this way handles every rank and axis, where the old
        // hand-rolled cases covered only 2D and 4D-along-channels and bailed
        // out with `unimplemented!` otherwise.
        let outer: usize = first_shape[..dim].iter().product();
        let inner: usize = first_shape[dim + 1..].iter().product();

        let mut result_data = Vec::with_capacity(outer * total_dim_size * inner);
        // Per input, which output positions its elements landed in — used to
        // route gradients back without recomputing the layout.
        let mut source_positions: Vec<Vec<usize>> = tensors
            .iter()
            .map(|t| Vec::with_capacity(t.numel()))
            .collect();

        for o in 0..outer {
            for (ti, tensor) in tensors.iter().enumerate() {
                let block = tensor.shape[dim] * inner;
                let data = tensor.elements();
                let start = o * block;
                for &v in &data[start..start + block] {
                    source_positions[ti].push(result_data.len());
                    result_data.push(v);
                }
            }
        }

        let mut output = Tensor::new(result_data, &out_shape);

        // `cat` had no backward edge at all, so grouped convolution — its only
        // caller — could never propagate gradients past the concatenation.
        if let Some(anchor) = tensors.iter().find(|t| t.requires_grad) {
            output.requires_grad = true;
            let inputs: Vec<Tensor> = tensors.to_vec();
            let out = output.clone();

            // Anchor on a tensor that actually requires grad: `push_unary_op`
            // drops the node otherwise, and the first input may not be the one.
            crate::tape::Tape::push_unary_op(anchor, &output, move || {
                if let Some(gout) = out.grad.read().expect("grad RwLock poisoned").as_ref() {
                    for (tensor, positions) in inputs.iter().zip(source_positions.iter()) {
                        if !tensor.requires_grad {
                            continue;
                        }
                        let mut slot = tensor.grad.write().expect("grad RwLock poisoned");
                        let gin = slot.get_or_insert_with(|| vec![0.0; tensor.numel()]);
                        for (local, &pos) in positions.iter().enumerate() {
                            gin[local] += gout[pos];
                        }
                    }
                }
            });
        }

        output
    }
}
