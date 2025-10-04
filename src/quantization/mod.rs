//! Quantization module for Taper deep learning library
//!
//! This module provides both Post-Training Quantization (PTQ) and
//! Quantization-Aware Training (QAT) capabilities.

pub mod config;
pub mod fake_quantize;
pub mod observers;
pub mod qat_config;
pub mod qat_layers;
pub mod qat_manager;

// Re-export main types for backward compatibility
pub use config::{QuantizationConfig, QuantizationSchema, QuantizationType};
pub use fake_quantize::FakeQuantize;
pub use qat_config::QATConfig;
pub use qat_manager::QATManager;

// Re-export QAT layers
pub use qat_layers::{QATConv2d, QATLinear, QATSequential};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_config_creation() {
        let config = QuantizationConfig::int8(true);
        assert!(config.is_enabled());
        assert_eq!(config.quant_type, QuantizationType::Int8);
    }

    #[test]
    fn test_qat_config_creation() {
        let quant_config = QuantizationConfig::int8(true);
        let qat_config = QATConfig::new(quant_config, 0.001, 5);
        assert!(qat_config.quant_config.is_enabled());
        assert_eq!(qat_config.learning_rate, 0.001);
        assert_eq!(qat_config.warmup_epochs, 5);
    }
}
