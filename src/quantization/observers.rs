//! Quantization observers for collecting statistics during QAT
//!
//! This module provides observers that collect statistics about tensor values
//! during training, which are used to determine optimal quantization parameters.

use crate::Tensor;

/// Tracks the running min and max across every tensor it observes.
///
/// The previous implementation stored a *per-element* min/max vector sized from
/// the first tensor it saw, so observing tensors of different lengths silently
/// ignored the tail of the larger ones — and the whole-tensor `global_min` /
/// `global_max` it exposed were the only things anyone read anyway.
#[derive(Debug, Clone)]
pub struct MinMaxObserver {
    min_value: f32,
    max_value: f32,
    /// Number of observations
    num_observations: usize,
    /// Whether the observer is enabled
    enabled: bool,
}

impl Default for MinMaxObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxObserver {
    /// Create a new MinMax observer
    pub fn new() -> Self {
        Self {
            min_value: f32::INFINITY,
            max_value: f32::NEG_INFINITY,
            num_observations: 0,
            enabled: true,
        }
    }

    /// Enable or disable the observer
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the observer is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Observe a tensor and update statistics
    pub fn observe(&mut self, tensor: &Tensor) {
        let data = tensor.data();
        self.observe_slice(&data);
    }

    /// Observe raw values and update statistics.
    ///
    /// Non-finite values are skipped: a single NaN would otherwise poison the
    /// range and every scale derived from it.
    pub fn observe_slice(&mut self, data: &[f32]) {
        if !self.enabled {
            return;
        }

        for &val in data {
            if val.is_finite() {
                self.min_value = self.min_value.min(val);
                self.max_value = self.max_value.max(val);
            }
        }

        self.num_observations += 1;
    }

    /// Get the global minimum value
    pub fn global_min(&self) -> f32 {
        self.min_value
    }

    /// Get the global maximum value
    pub fn global_max(&self) -> f32 {
        self.max_value
    }

    /// The observed `(min, max)`. Still `(inf, -inf)` if nothing finite was seen.
    pub fn range(&self) -> (f32, f32) {
        (self.min_value, self.max_value)
    }

    /// Whether any finite value has been observed.
    pub fn has_data(&self) -> bool {
        self.min_value.is_finite() && self.max_value.is_finite()
    }

    /// Get the number of observations
    pub fn num_observations(&self) -> usize {
        self.num_observations
    }

    /// Reset the observer
    pub fn reset(&mut self) {
        self.min_value = f32::INFINITY;
        self.max_value = f32::NEG_INFINITY;
        self.num_observations = 0;
    }

    /// Get statistics summary
    pub fn get_stats(&self) -> ObserverStats {
        ObserverStats {
            num_observations: self.num_observations,
            global_min: self.min_value,
            global_max: self.max_value,
            range: self.max_value - self.min_value,
        }
    }
}

/// Statistics collected by MinMax observer
#[derive(Debug, Clone, PartialEq)]
pub struct ObserverStats {
    pub num_observations: usize,
    pub global_min: f32,
    pub global_max: f32,
    pub range: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_observer() {
        let mut observer = MinMaxObserver::new();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);

        observer.observe(&tensor);

        assert_eq!(observer.num_observations(), 1);
        assert_eq!(observer.global_min(), 1.0);
        assert_eq!(observer.global_max(), 4.0);
    }

    #[test]
    fn observations_of_different_lengths_all_count() {
        let mut observer = MinMaxObserver::new();
        observer.observe(&Tensor::new(vec![0.0, 1.0], &[2]));
        // The tail of this longer tensor used to be dropped entirely.
        observer.observe(&Tensor::new(vec![0.5, 0.5, -9.0, 9.0], &[4]));

        assert_eq!(observer.range(), (-9.0, 9.0));
        assert_eq!(observer.num_observations(), 2);
    }

    #[test]
    fn non_finite_values_are_skipped() {
        let mut observer = MinMaxObserver::new();
        observer.observe_slice(&[1.0, f32::NAN, f32::INFINITY, 3.0]);
        assert_eq!(observer.range(), (1.0, 3.0));
    }

    #[test]
    fn empty_observer_reports_no_data() {
        let observer = MinMaxObserver::new();
        assert!(!observer.has_data());
    }
}
