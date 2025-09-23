#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationConfig {
    pub enabled: bool, 
    pub precision: u8, 
    pub schema: QuantizationSchema,
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
            precision: 8, 
            schema: QuantizationSchema::Uniform,
        }
    }
}

impl QuantizationConfig {
    // new quantization config 
    pub fn new(enabled: bool, precision: u8) -> Self {
        Self {
            enabled, 
            precision,
            schema: QuantizationSchema::Uniform,
        }
    }

    // per-channel qunatization. 
    pub fn per_channel(enabled: bool, precision: u8) -> Self {
        Self {
            enabled, 
            precision, 
            schema: QuantizationSchema::PerChannel,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    // compute the quantization range given the range. 
    pub fn compute_range(&self) -> (i32, i32) {
        let qmin = -(1i32 << (self.precision - 1)); 
        let qmax = (1i32 << (self.precision - 1)) - 1; 

        (qmin, qmax)
    }

    // scale factor for quantization = (max - min) / (qmax - qmin);
    pub fn compute_scale(&self, min: f32, max: f32) -> f32 {
        let (qmin, qmax) = self.compute_range(); 
        (max - min) / (qmax - qmin) as f32
    }

    // calculate zero-point scale: -min / scale
    pub fn compute_zero_point(&self, min: f32, scale: f32) -> i32 {
        (-min / scale).round() as i32
    }
}

// TESTING 
#[cfg(test)]
mod tests {
    use super::*; // go one module up 
    
    #[test]
    fn test_quantization_config_creation() {
        let cfg = QuantizationConfig::new(true, 8); 
        assert!(cfg.enabled); 
        assert_eq!(cfg.precision, 8); 
        assert_eq!(cfg.schema, QuantizationSchema::Uniform);
    }

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert!(!config.enabled);  // Fixed: should be false by default
        assert_eq!(config.precision, 8);
        assert_eq!(config.schema, QuantizationSchema::Uniform);
    }

    #[test]
    fn test_quant_range_8bit() {
        let cfg = QuantizationConfig::new(true, 8); 
        let (qmin, qmax) = cfg.compute_range(); 
        assert_eq!(qmin, -128); 
        assert_eq!(qmax, 127); 
    }

    #[test]
    fn test_quant_range_4bit() {
        let cfg = QuantizationConfig::new(true, 4);
        let (qmin, qmax) = cfg.compute_range();
        assert_eq!(qmin, -8); 
        assert_eq!(qmax, 7);  
    }

    #[test]
    fn test_calculate_scale() {
        let cfg = QuantizationConfig::new(true, 8); 
        let scale = 0.01; 
        let zero_point = cfg.compute_zero_point(-1.0, scale); 
        assert_eq!(zero_point, 100); 
    }
}