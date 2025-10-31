use crate::{Tensor, nn::{Module, Conv2d, MaxPool2d, AdaptiveAvgPool2d, Flatten, Linear}, norm::BatchNorm2d, activation::ReLU};

#[derive(Debug, Clone)]
pub struct ResNetConfig {
    pub layers: Vec<usize>,  // Number of blocks in each layer
    pub num_classes: usize,  // Number of output classes
    pub use_bottleneck: bool, // Whether to use BottleneckBlock (ResNet-50+) or BasicBlock (ResNet-18/34)
}

impl ResNetConfig {
    pub fn resnet18(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![2, 2, 2, 2],
            num_classes,
            use_bottleneck: false,
        }
    }

    pub fn resnet34(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![3, 4, 6, 3],
            num_classes,
            use_bottleneck: false,
        }
    }

    pub fn resnet50(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![3, 4, 6, 3],
            num_classes,
            use_bottleneck: true,
        }
    }

    pub fn resnet101(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![3, 4, 23, 3],
            num_classes,
            use_bottleneck: true,
        }
    }

    pub fn resnet152(num_classes: usize) -> Self {
        ResNetConfig {
            layers: vec![3, 8, 36, 3],
            num_classes,
            use_bottleneck: true,
        }
    }
}

pub struct ResNet {
    #[allow(dead_code)]
    config: ResNetConfig,
    
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    
    layer1: Vec<Box<dyn Module>>,
    layer2: Vec<Box<dyn Module>>,
    layer3: Vec<Box<dyn Module>>,
    layer4: Vec<Box<dyn Module>>,
    
    avgpool: AdaptiveAvgPool2d,
    flatten: Flatten,
    fc: Linear,
}

impl ResNet {
    pub fn new(config: ResNetConfig) -> Self {
        let conv1 = Conv2d::new(3, 64, (7, 7), Some((2, 2)), Some((3, 3)), None, None, false); // 7x7 conv, stride=2, padding=3
        let bn1 = BatchNorm2d::new(64);
        let relu = ReLU;
        let maxpool = MaxPool2d::new((3, 3), Some((2, 2)), Some((1, 1))); // 3x3 maxpool, stride=2, padding=1
        
        let mut in_channels = 64;
        let layer1 = Self::_make_layer(&config, in_channels, 64, config.layers[0], 1);
        in_channels = 64 * if config.use_bottleneck { 4 } else { 1 };
        
        let layer2 = Self::_make_layer(&config, in_channels, 128, config.layers[1], 2);
        in_channels = 128 * if config.use_bottleneck { 4 } else { 1 };
        
        let layer3 = Self::_make_layer(&config, in_channels, 256, config.layers[2], 2);
        in_channels = 256 * if config.use_bottleneck { 4 } else { 1 };
        
        let layer4 = Self::_make_layer(&config, in_channels, 512, config.layers[3], 2);
        
        let avgpool = AdaptiveAvgPool2d::new((1, 1)); // Global average pooling
        let flatten = Flatten::new(None);
        let fc = Linear::new(512 * if config.use_bottleneck { 4 } else { 1 }, config.num_classes, true);
        
        ResNet {
            config,
            conv1,
            bn1,
            relu,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            flatten,
            fc,
        }
    }

    /// Create ResNet for single-channel input (e.g., MNIST)
    pub fn new_single_channel(config: ResNetConfig) -> Self {
        // ResNet stem: initial conv7x7 + BN + ReLU + MaxPool for single channel
        let conv1 = Conv2d::new(1, 64, (7, 7), Some((2, 2)), Some((3, 3)), None, None, false); // 7x7 conv, stride=2, padding=3
        let bn1 = BatchNorm2d::new(64);
        let relu = ReLU;
        let maxpool = MaxPool2d::new((3, 3), Some((2, 2)), Some((1, 1))); // 3x3 maxpool, stride=2, padding=1
        
        // ResNet layers with standard channel progression
        let mut in_channels = 64;
        let layer1 = Self::_make_layer(&config, in_channels, 64, config.layers[0], 1);
        in_channels = 64 * if config.use_bottleneck { 4 } else { 1 };
        
        let layer2 = Self::_make_layer(&config, in_channels, 128, config.layers[1], 2);
        in_channels = 128 * if config.use_bottleneck { 4 } else { 1 };
        
        let layer3 = Self::_make_layer(&config, in_channels, 256, config.layers[2], 2);
        in_channels = 256 * if config.use_bottleneck { 4 } else { 1 };
        
        let layer4 = Self::_make_layer(&config, in_channels, 512, config.layers[3], 2);
        
        // Classifier block: adaptive avg pool + flatten + linear
        let avgpool = AdaptiveAvgPool2d::new((1, 1)); // Global average pooling
        let flatten = Flatten::new(None);
        let fc = Linear::new(512 * if config.use_bottleneck { 4 } else { 1 }, config.num_classes, true);
        
        ResNet {
            config,
            conv1,
            bn1,
            relu,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            flatten,
            fc,
        }
    }


    fn _make_layer(
        config: &ResNetConfig,
        in_channels: usize,
        out_channels: usize,
        blocks: usize,
        stride: usize,
    ) -> Vec<Box<dyn Module>> {
        let expansion = if config.use_bottleneck { 4 } else { 1 };
        let final_out_channels = out_channels * expansion;
        
        let mut layers: Vec<Box<dyn Module>> = Vec::new();
        
        if config.use_bottleneck {
            layers.push(Box::new(crate::nn::BottleneckBlock::new(in_channels, out_channels, stride)) as Box<dyn Module>);
        } else {
            layers.push(Box::new(crate::nn::BasicBlock::new(in_channels, out_channels, stride)) as Box<dyn Module>);
        }
        
        for _ in 1..blocks {
            if config.use_bottleneck {
                layers.push(Box::new(crate::nn::BottleneckBlock::new(final_out_channels, out_channels, 1)) as Box<dyn Module>);
            } else {
                layers.push(Box::new(crate::nn::BasicBlock::new(final_out_channels, out_channels, 1)) as Box<dyn Module>);
            }
        }
        
        layers
    }
}

impl Module for ResNet {
    fn forward(&self, input: &Tensor) -> Tensor {
        let conv1_out = self.conv1.forward(input);
        let bn1_out = self.bn1.forward(&conv1_out);
        let relu_out = bn1_out.relu();
        let maxpool_out = self.maxpool.forward(&relu_out);

        let mut layer1_out = maxpool_out;
        for layer in self.layer1.iter() {
            layer1_out = layer.forward(&layer1_out);
        }
        
        let mut layer2_out = layer1_out;
        for layer in self.layer2.iter() {
            layer2_out = layer.forward(&layer2_out);
        }

        let mut layer3_out = layer2_out;
        for layer in self.layer3.iter() {
            layer3_out = layer.forward(&layer3_out);
        }
        
        let mut layer4_out = layer3_out;
        for layer in self.layer4.iter() {
            layer4_out = layer.forward(&layer4_out);
        }

        let avgpool_out = self.avgpool.forward(&layer4_out);
        let flatten_out = self.flatten.forward(&avgpool_out);   
        
        self.fc.forward(&flatten_out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new(); 

        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.relu.parameters());
        params.extend(self.maxpool.parameters());
        params.extend(self.layer1.iter().flat_map(|layer| layer.parameters()).collect::<Vec<Tensor>>());
        params.extend(self.layer2.iter().flat_map(|layer| layer.parameters()).collect::<Vec<Tensor>>());
        params.extend(self.layer3.iter().flat_map(|layer| layer.parameters()).collect::<Vec<Tensor>>());
        params.extend(self.layer4.iter().flat_map(|layer| layer.parameters()).collect::<Vec<Tensor>>());
        params.extend(self.avgpool.parameters());
        params.extend(self.flatten.parameters());
        params.extend(self.fc.parameters());

        params
    }
}

impl ResNet {
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet18(num_classes))
    }

    pub fn resnet34(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet34(num_classes))
    }

    pub fn resnet50(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet50(num_classes))
    }

    pub fn resnet101(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet101(num_classes))
    }

    pub fn resnet152(num_classes: usize) -> Self {
        Self::new(ResNetConfig::resnet152(num_classes))
    }

    /// Create ResNet-18 for single-channel input (e.g., MNIST)
    pub fn resnet18_single_channel(num_classes: usize) -> Self {
        Self::new_single_channel(ResNetConfig::resnet18(num_classes))
    }

    /// Create ResNet-34 for single-channel input (e.g., MNIST)
    pub fn resnet34_single_channel(num_classes: usize) -> Self {
        Self::new_single_channel(ResNetConfig::resnet34(num_classes))
    }
}
