//! Quantization configuration and types
//!
//! This module contains the core quantization configuration structures
//! that define how quantization should be applied to models.

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
        matches!(
            self.quant_type,
            QuantizationType::Int8 | QuantizationType::Int4 | QuantizationType::NF4
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(
            self.quant_type,
            QuantizationType::BFloat16 | QuantizationType::Float16
        )
    }

    /// Scale factor for quantization: `(max - min) / (qmax - qmin)`.
    ///
    /// Returns `None` for float types, and for degenerate ranges that would
    /// produce a zero or non-finite scale — every caller divides by this, so
    /// handing back `0.0` turned every quantized value into an infinity.
    pub fn compute_scale(&self, min: f32, max: f32) -> Option<f32> {
        let (qmin, qmax) = self.compute_range()?;
        let scale = (max - min) / (qmax - qmin) as f32;
        (scale.is_finite() && scale > 0.0).then_some(scale)
    }

    /// The integer code that represents 0.0, i.e. `qmin - min / scale`.
    ///
    /// Anchoring at `qmin` is what maps `min` onto the bottom of the grid. The
    /// earlier `-min / scale` is only correct for an unsigned grid starting at
    /// zero; on int8's `[-128, 127]` it shifted everything up by 128 and
    /// clipped away half the representable range.
    pub fn compute_zero_point(&self, min: f32, scale: f32) -> Option<i32> {
        if !self.is_integer() || !scale.is_finite() || scale <= 0.0 {
            return None; // Float types don't use zero_point
        }
        let (qmin, qmax) = self.compute_range()?;
        Some(
            (qmin as f32 - min / scale)
                .round()
                .clamp(qmin as f32, qmax as f32) as i32,
        )
    }
}
