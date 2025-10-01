//! Quantization observers for collecting statistics during QAT
//! 
//! This module provides observers that collect statistics about tensor values
//! during training, which are used to determine optimal quantization parameters.

use crate::Tensor;
use std::collections::HashMap;

/// Observer for collecting statistics about tensor values
#[derive(Debug, Clone)]
pub struct MinMaxObserver {
    /// Minimum values observed
    min_values: Vec<f32>,
    /// Maximum values observed
    max_values: Vec<f32>,
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
            min_values: Vec::new(),
            max_values: Vec::new(),
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
        if !self.enabled {
            return;
        }

        let data = tensor.data();
        
        if self.min_values.is_empty() {
            // Initialize with first observation
            self.min_values = data.iter().copied().collect();
            self.max_values = data.iter().copied().collect();
        } else {
            // Update min/max values
            for (i, &val) in data.iter().enumerate() {
                if i < self.min_values.len() {
                    self.min_values[i] = self.min_values[i].min(val);
                    self.max_values[i] = self.max_values[i].max(val);
                }
            }
        }
        
        self.num_observations += 1;
    }

    /// Get the minimum values observed
    pub fn min_values(&self) -> &[f32] {
        &self.min_values
    }

    /// Get the maximum values observed
    pub fn max_values(&self) -> &[f32] {
        &self.max_values
    }

    /// Get the global minimum value
    pub fn global_min(&self) -> f32 {
        self.min_values.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Get the global maximum value
    pub fn global_max(&self) -> f32 {
        self.max_values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Get the number of observations
    pub fn num_observations(&self) -> usize {
        self.num_observations
    }

    /// Reset the observer
    pub fn reset(&mut self) {
        self.min_values.clear();
        self.max_values.clear();
        self.num_observations = 0;
    }

    /// Get statistics summary
    pub fn get_stats(&self) -> ObserverStats {
        ObserverStats {
            num_observations: self.num_observations,
            global_min: self.global_min(),
            global_max: self.global_max(),
            range: self.global_max() - self.global_min(),
        }
    }
}

/// Observer for collecting histogram statistics
#[derive(Debug, Clone)]
pub struct HistogramObserver {
    /// Histogram bins
    bins: Vec<usize>,
    /// Bin edges
    bin_edges: Vec<f32>,
    /// Number of bins
    num_bins: usize,
    /// Number of observations
    num_observations: usize,
    /// Whether the observer is enabled
    enabled: bool,
}

impl HistogramObserver {
    /// Create a new histogram observer
    pub fn new(num_bins: usize) -> Self {
        Self {
            bins: vec![0; num_bins],
            bin_edges: Vec::new(),
            num_bins,
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

    /// Observe a tensor and update histogram
    pub fn observe(&mut self, tensor: &Tensor) {
        if !self.enabled {
            return;
        }

        let data = tensor.data();
        
        if self.bin_edges.is_empty() {
            // Initialize bin edges based on first observation
            let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            let range = max_val - min_val;
            let bin_width = range / self.num_bins as f32;
            
            self.bin_edges = (0..=self.num_bins)
                .map(|i| min_val + i as f32 * bin_width)
                .collect();
        }

        // Update histogram
        for &val in data.iter() {
            if let Some(bin_idx) = self.find_bin(val) {
                if bin_idx < self.bins.len() {
                    self.bins[bin_idx] += 1;
                }
            }
        }
        
        self.num_observations += 1;
    }

    /// Find the bin index for a value
    fn find_bin(&self, val: f32) -> Option<usize> {
        for (i, &edge) in self.bin_edges.iter().enumerate() {
            if val <= edge {
                return Some(i.saturating_sub(1));
            }
        }
        Some(self.num_bins - 1)
    }

    /// Get the histogram bins
    pub fn bins(&self) -> &[usize] {
        &self.bins
    }

    /// Get the bin edges
    pub fn bin_edges(&self) -> &[f32] {
        &self.bin_edges
    }

    /// Get the number of observations
    pub fn num_observations(&self) -> usize {
        self.num_observations
    }

    /// Reset the observer
    pub fn reset(&mut self) {
        self.bins.fill(0);
        self.bin_edges.clear();
        self.num_observations = 0;
    }

    /// Get statistics summary
    pub fn get_stats(&self) -> HistogramStats {
        let total_count: usize = self.bins.iter().sum();
        let mean_bin = if total_count > 0 {
            self.bins.iter().enumerate()
                .map(|(i, &count)| i * count)
                .sum::<usize>() as f32 / total_count as f32
        } else {
            0.0
        };

        HistogramStats {
            num_observations: self.num_observations,
            total_count,
            mean_bin,
            max_bin_count: self.bins.iter().max().copied().unwrap_or(0),
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

/// Statistics collected by Histogram observer
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramStats {
    pub num_observations: usize,
    pub total_count: usize,
    pub mean_bin: f32,
    pub max_bin_count: usize,
}

/// Manager for multiple observers
#[derive(Debug)]
pub struct ObserverManager {
    /// MinMax observers by name
    minmax_observers: HashMap<String, MinMaxObserver>,
    /// Histogram observers by name
    histogram_observers: HashMap<String, HistogramObserver>,
}

impl Default for ObserverManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ObserverManager {
    /// Create a new observer manager
    pub fn new() -> Self {
        Self {
            minmax_observers: HashMap::new(),
            histogram_observers: HashMap::new(),
        }
    }

    /// Add a MinMax observer
    pub fn add_minmax_observer(&mut self, name: &str) {
        self.minmax_observers.insert(name.to_string(), MinMaxObserver::new());
    }

    /// Add a histogram observer
    pub fn add_histogram_observer(&mut self, name: &str, num_bins: usize) {
        self.histogram_observers.insert(name.to_string(), HistogramObserver::new(num_bins));
    }

    /// Observe a tensor with a MinMax observer
    pub fn observe_minmax(&mut self, name: &str, tensor: &Tensor) {
        if let Some(observer) = self.minmax_observers.get_mut(name) {
            observer.observe(tensor);
        }
    }

    /// Observe a tensor with a histogram observer
    pub fn observe_histogram(&mut self, name: &str, tensor: &Tensor) {
        if let Some(observer) = self.histogram_observers.get_mut(name) {
            observer.observe(tensor);
        }
    }

    /// Get MinMax observer statistics
    pub fn get_minmax_stats(&self, name: &str) -> Option<ObserverStats> {
        self.minmax_observers.get(name).map(|obs| obs.get_stats())
    }

    /// Get histogram observer statistics
    pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        self.histogram_observers.get(name).map(|obs| obs.get_stats())
    }

    /// Reset all observers
    pub fn reset_all(&mut self) {
        for observer in self.minmax_observers.values_mut() {
            observer.reset();
        }
        for observer in self.histogram_observers.values_mut() {
            observer.reset();
        }
    }

    /// Get all observer names
    pub fn get_observer_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        names.extend(self.minmax_observers.keys().cloned());
        names.extend(self.histogram_observers.keys().cloned());
        names
    }
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
    fn test_histogram_observer() {
        let mut observer = HistogramObserver::new(10);
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        
        observer.observe(&tensor);
        
        assert_eq!(observer.num_observations(), 1);
        assert_eq!(observer.bins().len(), 10);
    }

    #[test]
    fn test_observer_manager() {
        let mut manager = ObserverManager::new();
        manager.add_minmax_observer("test");
        
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        manager.observe_minmax("test", &tensor);
        
        let stats = manager.get_minmax_stats("test").unwrap();
        assert_eq!(stats.num_observations, 1);
        assert_eq!(stats.global_min, 1.0);
        assert_eq!(stats.global_max, 3.0);
    }
}
