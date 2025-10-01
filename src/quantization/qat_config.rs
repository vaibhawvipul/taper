//! Quantization-Aware Training (QAT) configuration
//! 
//! This module provides configuration structures specifically for QAT,
//! extending the base quantization configuration with QAT-specific parameters.

use super::config::{QuantizationConfig, QuantizationType};

/// Configuration for Quantization-Aware Training
#[derive(Debug, Clone, PartialEq)]
pub struct QATConfig {
    /// Base quantization configuration
    pub quant_config: QuantizationConfig,
    /// Learning rate for QAT fine-tuning
    pub learning_rate: f32,
    /// Number of warmup epochs before applying quantization
    pub warmup_epochs: usize,
    /// Whether to freeze batch normalization layers during QAT
    pub freeze_bn: bool,
    /// Whether to enable quantization observers for statistics collection
    pub observer_enabled: bool,
    /// Whether to use per-channel quantization
    pub per_channel: bool,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
}

impl Default for QATConfig {
    fn default() -> Self {
        Self {
            quant_config: QuantizationConfig::int8(true),
            learning_rate: 0.001,
            warmup_epochs: 5,
            freeze_bn: true,
            observer_enabled: true,
            per_channel: false,
            symmetric: true,
        }
    }
}

impl QATConfig {
    /// Create a new QAT configuration
    pub fn new(quant_config: QuantizationConfig, learning_rate: f32, warmup_epochs: usize) -> Self {
        Self {
            quant_config,
            learning_rate,
            warmup_epochs,
            freeze_bn: true,
            observer_enabled: true,
            per_channel: false,
            symmetric: true,
        }
    }

    /// Create QAT config for Int8 quantization
    pub fn int8(learning_rate: f32, warmup_epochs: usize) -> Self {
        Self::new(QuantizationConfig::int8(true), learning_rate, warmup_epochs)
    }

    /// Create QAT config for Int4 quantization
    pub fn int4(learning_rate: f32, warmup_epochs: usize) -> Self {
        Self::new(QuantizationConfig::int4(true), learning_rate, warmup_epochs)
    }

    /// Create QAT config for Float16 quantization
    pub fn float16(learning_rate: f32, warmup_epochs: usize) -> Self {
        Self::new(QuantizationConfig::float16(true), learning_rate, warmup_epochs)
    }

    /// Enable per-channel quantization
    pub fn with_per_channel(mut self, enabled: bool) -> Self {
        self.per_channel = enabled;
        self
    }

    /// Enable symmetric quantization
    pub fn with_symmetric(mut self, enabled: bool) -> Self {
        self.symmetric = enabled;
        self
    }

    /// Enable quantization observers
    pub fn with_observers(mut self, enabled: bool) -> Self {
        self.observer_enabled = enabled;
        self
    }

    /// Freeze batch normalization layers
    pub fn with_freeze_bn(mut self, enabled: bool) -> Self {
        self.freeze_bn = enabled;
        self
    }

    /// Check if QAT is enabled
    pub fn is_enabled(&self) -> bool {
        self.quant_config.is_enabled()
    }

    /// Get the quantization type
    pub fn quant_type(&self) -> QuantizationType {
        self.quant_config.quant_type
    }

    /// Check if we're in warmup phase
    pub fn is_warmup(&self, current_epoch: usize) -> bool {
        current_epoch < self.warmup_epochs
    }

    /// Get the effective learning rate (may be different during warmup)
    pub fn get_effective_lr(&self, current_epoch: usize) -> f32 {
        if self.is_warmup(current_epoch) {
            // During warmup, use a reduced learning rate
            self.learning_rate * 0.1
        } else {
            self.learning_rate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_config_creation() {
        let config = QATConfig::int8(0.001, 5);
        assert!(config.is_enabled());
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.warmup_epochs, 5);
        assert!(config.freeze_bn);
        assert!(config.observer_enabled);
    }

    #[test]
    fn test_qat_config_builder() {
        let config = QATConfig::int8(0.001, 5)
            .with_per_channel(true)
            .with_symmetric(false)
            .with_observers(false)
            .with_freeze_bn(false);

        assert!(config.per_channel);
        assert!(!config.symmetric);
        assert!(!config.observer_enabled);
        assert!(!config.freeze_bn);
    }

    #[test]
    fn test_warmup_phase() {
        let config = QATConfig::int8(0.001, 5);
        
        assert!(config.is_warmup(0));
        assert!(config.is_warmup(4));
        assert!(!config.is_warmup(5));
        assert!(!config.is_warmup(10));
    }

    #[test]
    fn test_effective_learning_rate() {
        let config = QATConfig::int8(0.001, 5);
        
        // During warmup, should be reduced
        assert_eq!(config.get_effective_lr(0), 0.0001);
        assert_eq!(config.get_effective_lr(4), 0.0001);
        
        // After warmup, should be full rate
        assert_eq!(config.get_effective_lr(5), 0.001);
        assert_eq!(config.get_effective_lr(10), 0.001);
    }
}
