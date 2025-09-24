#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationConfig {
    pub enabled: bool,  
    pub quant_type: QuantizationType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationType {
    Int4, 
    Int8, 
    Float16, 
    BFloat16, 
    NF4,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationSchema {
    Uniform, 
    PerChannel, 
}

impl Default for QuantizationConfig { 
    fn default() -> Self {
        Self {
            enabled: false, 
            quant_type: QuantizationType::Int8,
        }
    }
}

impl QuantizationConfig {
    // new quantization config 
    pub fn new(enabled: bool, quant_type: QuantizationType) -> Self {
        Self {
            enabled, 
            quant_type,
        }
    }

    pub fn int8(enabled: bool) -> Self {
        Self::new(enabled, QuantizationType::Int8)
    }

    pub fn int4(enabled: bool) -> Self {
        Self::new(enabled, QuantizationType::Int4)
    }

    pub fn float16(enabled: bool) -> Self {
        Self::new(enabled, QuantizationType::Float16)
    }

    pub fn bfloat16(enabled: bool) -> Self {
        Self::new(enabled, QuantizationType::BFloat16)
    }

    pub fn nf4(enabled: bool) -> Self {
        Self::new(enabled, QuantizationType::NF4)
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    // updated compute range for multiple types. 
    pub fn compute_range(&self) -> Option<(i32, i32)> {
        match self.quant_type { 
            QuantizationType::Int8 => Some((-128, 127)),
            QuantizationType::Int4 => Some((-8, 7)),
            QuantizationType::Float16 => None, // Float types don't have discrete ranges
            QuantizationType::BFloat16 => None,
            QuantizationType::NF4 => Some((-8, 7)),
        }
    }

    pub fn bit_width(&self) -> u8 {
        match self.quant_type {
            QuantizationType::Int8 => 8, 
            QuantizationType::Int4 => 4, 
            QuantizationType::Float16 => 16, 
            QuantizationType::BFloat16 => 16, 
            QuantizationType::NF4 => 4, 
        }
    }

    pub fn is_integer(&self) -> bool {
        matches!(self.quant_type, QuantizationType::Int8 | QuantizationType::Int4 | QuantizationType::NF4)
    }

    pub fn is_float(&self) -> bool {
        matches!(self.quant_type, QuantizationType::BFloat16 | QuantizationType::Float16)
    }

    // per-channel quantization (using Int8 as default)
    pub fn per_channel(enabled: bool) -> Self {
        Self {
            enabled, 
            quant_type: QuantizationType::Int8, // Default to Int8 for per-channel
        }
    }

    // scale factor for quantization = (max - min) / (qmax - qmin);
    pub fn compute_scale(&self, min: f32, max: f32) -> Option<f32> {
        if let Some((qmin, qmax)) = self.compute_range() {
            Some((max - min) / (qmax - qmin) as f32)
        } else {
            None // Float types don't use scale/zero_point
        }
    }

    // calculate zero-point scale: -min / scale
    pub fn compute_zero_point(&self, min: f32, scale: f32) -> Option<i32> {
        if self.is_integer() {
            Some((-min / scale).round() as i32)
        } else {
            None // Float types don't use zero_point
        }
    }
}