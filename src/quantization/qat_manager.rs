//! QAT Manager for managing Quantization-Aware Training modes
//! 
//! This module provides global state management for QAT, allowing
//! switching between training and evaluation modes across the entire model.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Global QAT manager for managing training/evaluation modes
#[derive(Debug, Clone)]
pub struct QATManager {
    /// Global QAT enabled state
    global_qat_enabled: Arc<RwLock<bool>>,
    /// Per-module QAT states
    module_states: Arc<RwLock<HashMap<String, bool>>>,
    /// Current training mode
    training_mode: Arc<RwLock<bool>>,
}

impl Default for QATManager {
    fn default() -> Self {
        Self::new()
    }
}

impl QATManager {
    /// Create a new QAT manager
    pub fn new() -> Self {
        Self {
            global_qat_enabled: Arc::new(RwLock::new(false)),
            module_states: Arc::new(RwLock::new(HashMap::new())),
            training_mode: Arc::new(RwLock::new(true)),
        }
    }

    /// Enable or disable QAT globally
    pub fn enable_qat(&self, enabled: bool) {
        if let Ok(mut state) = self.global_qat_enabled.write() {
            *state = enabled;
        }
    }

    /// Check if QAT is globally enabled
    pub fn is_qat_enabled(&self) -> bool {
        self.global_qat_enabled.read()
            .map(|state| *state)
            .unwrap_or(false)
    }

    /// Set QAT state for a specific module
    pub fn set_module_qat(&self, module_id: &str, enabled: bool) {
        if let Ok(mut states) = self.module_states.write() {
            states.insert(module_id.to_string(), enabled);
        }
    }

    /// Check if QAT is enabled for a specific module
    pub fn is_module_qat_enabled(&self, module_id: &str) -> bool {
        // Check global state first
        if !self.is_qat_enabled() {
            return false;
        }

        // Check module-specific state
        self.module_states.read()
            .map(|states| states.get(module_id).copied().unwrap_or(true))
            .unwrap_or(true) // Default to enabled if not specified
    }

    /// Set training mode
    pub fn set_training_mode(&self, training: bool) {
        if let Ok(mut mode) = self.training_mode.write() {
            *mode = training;
        }
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training_mode.read()
            .map(|mode| *mode)
            .unwrap_or(true)
    }

    /// Check if in evaluation mode
    pub fn is_eval(&self) -> bool {
        !self.is_training()
    }

    /// Reset all module states
    pub fn reset_module_states(&self) {
        if let Ok(mut states) = self.module_states.write() {
            states.clear();
        }
    }

    /// Get all module states
    pub fn get_module_states(&self) -> HashMap<String, bool> {
        self.module_states.read()
            .map(|states| states.clone())
            .unwrap_or_default()
    }

    /// Enable QAT for all modules
    pub fn enable_all_modules(&self) {
        if let Ok(mut states) = self.module_states.write() {
            for (_, enabled) in states.iter_mut() {
                *enabled = true;
            }
        }
    }

    /// Disable QAT for all modules
    pub fn disable_all_modules(&self) {
        if let Ok(mut states) = self.module_states.write() {
            for (_, enabled) in states.iter_mut() {
                *enabled = false;
            }
        }
    }

    /// Get QAT status summary
    pub fn get_status(&self) -> QATStatus {
        QATStatus {
            global_enabled: self.is_qat_enabled(),
            training_mode: self.is_training(),
            module_count: self.get_module_states().len(),
            enabled_modules: self.get_module_states()
                .values()
                .filter(|&&enabled| enabled)
                .count(),
        }
    }
}

/// QAT status information
#[derive(Debug, Clone, PartialEq)]
pub struct QATStatus {
    pub global_enabled: bool,
    pub training_mode: bool,
    pub module_count: usize,
    pub enabled_modules: usize,
}

impl QATStatus {
    /// Check if QAT is active (globally enabled and in training mode)
    pub fn is_active(&self) -> bool {
        self.global_enabled && self.training_mode
    }

    /// Get percentage of modules with QAT enabled
    pub fn enabled_percentage(&self) -> f32 {
        if self.module_count == 0 {
            0.0
        } else {
            (self.enabled_modules as f32) / (self.module_count as f32) * 100.0
        }
    }
}

/// Global QAT manager instance
lazy_static::lazy_static! {
    static ref GLOBAL_QAT_MANAGER: QATManager = QATManager::new();
}

/// Get the global QAT manager
pub fn get_global_qat_manager() -> &'static QATManager {
    &GLOBAL_QAT_MANAGER
}

/// Convenience functions for global QAT management
pub mod global {
    use super::*;

    /// Enable QAT globally
    pub fn enable_qat() {
        get_global_qat_manager().enable_qat(true);
    }

    /// Disable QAT globally
    pub fn disable_qat() {
        get_global_qat_manager().enable_qat(false);
    }

    /// Check if QAT is globally enabled
    pub fn is_qat_enabled() -> bool {
        get_global_qat_manager().is_qat_enabled()
    }

    /// Set QAT state for a specific module
    pub fn set_module_qat(module_id: &str, enabled: bool) {
        get_global_qat_manager().set_module_qat(module_id, enabled);
    }

    /// Check if QAT is enabled for a specific module
    pub fn is_module_qat_enabled(module_id: &str) -> bool {
        get_global_qat_manager().is_module_qat_enabled(module_id)
    }

    /// Set training mode
    pub fn set_training_mode(training: bool) {
        get_global_qat_manager().set_training_mode(training);
    }

    /// Check if in training mode
    pub fn is_training() -> bool {
        get_global_qat_manager().is_training()
    }

    /// Check if in evaluation mode
    pub fn is_eval() -> bool {
        get_global_qat_manager().is_eval()
    }

    /// Get QAT status
    pub fn get_status() -> QATStatus {
        get_global_qat_manager().get_status()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_manager_creation() {
        let manager = QATManager::new();
        assert!(!manager.is_qat_enabled());
        assert!(manager.is_training());
    }

    #[test]
    fn test_qat_manager_enable_disable() {
        let manager = QATManager::new();
        
        manager.enable_qat(true);
        assert!(manager.is_qat_enabled());
        
        manager.enable_qat(false);
        assert!(!manager.is_qat_enabled());
    }

    #[test]
    fn test_module_states() {
        let manager = QATManager::new();
        
        manager.enable_qat(true);
        manager.set_module_qat("linear1", true);
        manager.set_module_qat("linear2", false);
        
        assert!(manager.is_module_qat_enabled("linear1"));
        assert!(!manager.is_module_qat_enabled("linear2"));
        assert!(manager.is_module_qat_enabled("linear3")); // Default to enabled
    }

    #[test]
    fn test_training_mode() {
        let manager = QATManager::new();
        
        manager.set_training_mode(false);
        assert!(!manager.is_training());
        assert!(manager.is_eval());
        
        manager.set_training_mode(true);
        assert!(manager.is_training());
        assert!(!manager.is_eval());
    }

    #[test]
    fn test_qat_status() {
        let manager = QATManager::new();
        
        manager.enable_qat(true);
        manager.set_module_qat("linear1", true);
        manager.set_module_qat("linear2", false);
        
        let status = manager.get_status();
        assert!(status.global_enabled);
        assert!(status.training_mode);
        assert_eq!(status.module_count, 2);
        assert_eq!(status.enabled_modules, 1);
        assert_eq!(status.enabled_percentage(), 50.0);
    }

    #[test]
    fn test_global_functions() {
        global::enable_qat();
        assert!(global::is_qat_enabled());
        
        global::set_training_mode(false);
        assert!(!global::is_training());
        assert!(global::is_eval());
        
        global::disable_qat();
        assert!(!global::is_qat_enabled());
    }
}
