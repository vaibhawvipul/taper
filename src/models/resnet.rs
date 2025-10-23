use crate::{Tensor, nn::{Module, Conv2d, MaxPool2d, AdaptiveAvgPool2d, Flatten, Linear}, norm::BatchNorm2d, activation::ReLU};

/// ResNet configuration for different variants
#[derive(Debug, Clone)]
pub struct ResNetConfig {
    pub layers: Vec<usize>,  // Number of blocks in each layer
    pub num_classes: usize,  // Number of output classes
    pub use_bottleneck: bool, // Whether to use BottleneckBlock (ResNet-50+) or BasicBlock (ResNet-18/34)
}

impl ResNetConfig {
    /// ResNet-18 configuration
    pub fn resnet18(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![2, 2, 2, 2],
            num_classes,
            use_bottleneck: false,
        }
    }

    /// ResNet-34 configuration
    pub fn resnet34(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![3, 4, 6, 3],
            num_classes,
            use_bottleneck: false,
        }
    }

    /// ResNet-50 configuration
    pub fn resnet50(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![3, 4, 6, 3],
            num_classes,
            use_bottleneck: true,
        }
    }

    /// ResNet-101 configuration
    pub fn resnet101(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![3, 4, 23, 3],
            num_classes,
            use_bottleneck: true,
        }
    }

    /// ResNet-152 configuration
    pub fn resnet152(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![3, 8, 36, 3],
            num_classes,
            use_bottleneck: true,
        }
    }
}

/// ResNet model implementation
pub struct ResNet {
    config: ResNetConfig,
    
    // Initial layers
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    
    // ResNet layers
    layer1: Vec<Box<dyn Module>>,
    layer2: Vec<Box<dyn Module>>,
    layer3: Vec<Box<dyn Module>>,
    layer4: Vec<Box<dyn Module>>,
    
    // Final layers
    avgpool: AdaptiveAvgPool2d,
    flatten: Flatten,
    fc: Linear,
}

impl ResNet {
    /// Create a new ResNet model with the given configuration
    pub fn new(config: ResNetConfig) -> Self {
        // TODO: Implement the constructor
        // This will include:
        // 1. Initial conv7x7 + BN + ReLU + MaxPool
        // 2. Creating the 4 ResNet layers using _make_layer
        // 3. Final adaptive avg pool + flatten + linear classifier
        
        todo!("Implement ResNet constructor")
    }

    /// Create a ResNet layer with the specified number of blocks
    /// 
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels  
    /// * `blocks` - Number of blocks in this layer
    /// * `stride` - Stride for the first block
    /// * `use_bottleneck` - Whether to use BottleneckBlock or BasicBlock
    fn _make_layer(
        &self,
        in_channels: usize,
        out_channels: usize,
        blocks: usize,
        stride: usize,
        use_bottleneck: bool,
    ) -> Vec<Box<dyn Module>> {
        // TODO: Implement layer creation
        // This will:
        // 1. Create the first block with the given stride
        // 2. Create remaining blocks with stride=1
        // 3. Return vector of blocks
        
        todo!("Implement _make_layer function")
    }
}

impl Module for ResNet {
    fn forward(&self, input: &Tensor) -> Tensor {
        // TODO: Implement forward pass
        // This will include:
        // 1. Initial conv + BN + ReLU + MaxPool
        // 2. Pass through layer1, layer2, layer3, layer4
        // 3. Adaptive avg pool + flatten + linear classifier
        
        todo!("Implement ResNet forward pass")
    }

    fn parameters(&self) -> Vec<Tensor> {
        // TODO: Implement parameter collection
        // This will collect parameters from all layers
        
        todo!("Implement ResNet parameters collection")
    }
}

/// Convenience functions for creating specific ResNet variants
impl ResNet {
    /// Create ResNet-18
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet18(num_classes))
    }

    /// Create ResNet-34  
    pub fn resnet34(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet34(num_classes))
    }

    /// Create ResNet-50
    pub fn resnet50(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet50(num_classes))
    }

    /// Create ResNet-101
    pub fn resnet101(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet101(num_classes))
    }

    /// Create ResNet-152
    pub fn resnet152(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet152(num_classes))
    }
}
