use crate::{Tensor, QuantizationConfig};
use rand::{distributions::{Distribution, Uniform}, Rng};

/// Trait for any differentiable network component.
pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    
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

    fn quantize(&self, qconfig: &QuantizationConfig) -> Box<dyn QuantizedModule> {
        Box::new(QuantizedLinear {
            weight: self.weight.quantize(qconfig),
            bias: self.bias.as_ref().map(|b| b.quantize(qconfig)),
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
    weight: crate::tensor::QuantizedTensor,
    bias: Option<crate::tensor::QuantizedTensor>,
}

impl QuantizedModule for QuantizedLinear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight_f32 = self.weight.dequantize();
        let mut out = input.matmul(&weight_f32.transpose());
        
        if let Some(b) = &self.bias {
            let bias_f32 = b.dequantize();
            out = out.add_broadcast(&bias_f32);
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Return empty for quantized modules since parameters are stored as QuantizedTensor
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
            let (_n, c_in, _h, _w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
            let c_out = weight_f32.shape()[0];
            let c_in_per_group = c_in / self.groups;
            let c_out_per_group = c_out / self.groups;

            let mut group_outputs = Vec::new();

            for g in 0..self.groups {
                // Extract input channels for this group
                let input_slice = input.slice_channels(g * c_in_per_group, (g + 1) * c_in_per_group);

                // Extract weight channels for this group
                let weight_slice = weight_f32.slice_output_channels(g * c_out_per_group, (g + 1) * c_out_per_group);

                // Extract bias for this group
                let bias_slice = bias_f32.as_ref().map(|b| b.slice_1d(g * c_out_per_group, (g + 1) * c_out_per_group));

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

    fn parameters(&self) -> Vec<Tensor> {
        self.conv.parameters()
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
            let (_n, _c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
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

        // Calculate kernel size and stride to achieve target output size
        let kernel_h = h_in / h_out;
        let kernel_w = w_in / w_out;
        let stride_h = h_in / h_out;
        let stride_w = w_in / w_out;

        input.avg_pool2d((kernel_h, kernel_w), Some((stride_h, stride_w)), (0, 0))
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
        let (_n, _c, h_in, w_in) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        let (h_out, w_out) = self.output_size;

        // Calculate kernel size and stride to achieve target output size
        let kernel_h = h_in / h_out;
        let kernel_w = w_in / w_out;
        let stride_h = h_in / h_out;
        let stride_w = w_in / w_out;

        input.avg_pool2d((kernel_h, kernel_w), Some((stride_h, stride_w)), (0, 0))
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
            p >= 0.0 && p <= 1.0,
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
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        if self.p == 1.0 {
            return Tensor::new(vec![0.0; input.data().len()], input.shape());
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

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Basic CNN block: Conv -> BatchNorm -> ReLU (BatchNorm will be added later)
#[derive(Debug)]
pub struct BasicBlock {
    conv: Conv2d,
    // batchnorm: BatchNorm2d, // TODO: Add when BatchNorm2d is implemented
}

impl BasicBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        BasicBlock {
            conv: Conv2d::conv3x3(in_channels, out_channels, stride, 1),
            // batchnorm: BatchNorm2d::new(out_channels),
        }
    }
}

impl Module for BasicBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        let out = self.conv.forward(input);
        // let out = self.batchnorm.forward(&out); // TODO: Add when BatchNorm2d is implemented
        out.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        let params = self.conv.parameters();
        // params.extend(self.batchnorm.parameters()); // TODO: Add when BatchNorm2d is implemented
        params
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

        let (n, _c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let c_slice = end - start;

        let data = self.data();
        let mut result_data = Vec::new();

        for batch in 0..n {
            for ch in start..end {
                let base_idx = batch * self.shape[1] * h * w + ch * h * w;
                for spatial in 0..(h * w) {
                    result_data.push(data[base_idx + spatial]);
                }
            }
        }

        Tensor::new(result_data, &[n, c_slice, h, w])
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

        let (_c_out, c_in, k_h, k_w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let c_out_slice = end - start;

        let data = self.data();
        let mut result_data = Vec::new();

        for out_ch in start..end {
            let base_idx = out_ch * c_in * k_h * k_w;
            for i in 0..(c_in * k_h * k_w) {
                result_data.push(data[base_idx + i]);
            }
        }

        Tensor::new(result_data, &[c_out_slice, c_in, k_h, k_w])
    }

    /// Extract slice from 1D tensor (for bias)
    pub fn slice_1d(&self, start: usize, end: usize) -> Tensor {
        assert_eq!(self.shape.len(), 1, "slice_1d only works on 1D tensors");
        assert!(start < end && end <= self.shape[0], "Invalid range");

        let data = self.data();
        let result_data = data[start..end].to_vec();

        Tensor::new(result_data, &[end - start])
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

        // Concatenate data
        let mut result_data = Vec::new();

        // For efficiency, we'll implement this for common cases
        match ndim {
            2 => {
                if dim == 0 {
                    // Concatenate along rows
                    for tensor in tensors {
                        result_data.extend_from_slice(&tensor.data());
                    }
                } else {
                    // Concatenate along columns
                    let rows = first_shape[0];
                    // let col_offset = 0;

                    for row in 0..rows {
                        for tensor in tensors {
                            let cols = tensor.shape[1];
                            let tensor_data = tensor.data();
                            for col in 0..cols {
                                result_data.push(tensor_data[row * cols + col]);
                            }
                        }
                    }
                }
            }
            4 => {
                if dim == 1 {
                    // Concatenate along channel dimension (most common for CNNs)
                    let (n, _, h, w) = (
                        first_shape[0],
                        first_shape[1],
                        first_shape[2],
                        first_shape[3],
                    );

                    for batch in 0..n {
                        for tensor in tensors {
                            let c = tensor.shape[1];
                            let tensor_data = tensor.data();
                            for ch in 0..c {
                                for spatial in 0..(h * w) {
                                    let idx = batch * c * h * w + ch * h * w + spatial;
                                    result_data.push(tensor_data[idx]);
                                }
                            }
                        }
                    }
                } else {
                    // General case for other dimensions
                    unimplemented!("General 4D concatenation not implemented for dim != 1");
                }
            }
            _ => unimplemented!("Concatenation not implemented for {}D tensors", ndim),
        }

        Tensor::new(result_data, &out_shape)
    }
}
