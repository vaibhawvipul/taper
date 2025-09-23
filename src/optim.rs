use crate::Tensor;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32, _momentum: Option<f32>) -> Self {
        // TODO: Implement momentum
        SGD { params, lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for param in &self.params {
            if let Some(grad) = param.grad() {
                let mut param_data = param.data_mut();
                let grad_data = grad.data();

                // Vanilla SGD update: param = param - lr * grad
                for i in 0..param_data.len() {
                    param_data[i] -= self.lr * grad_data[i];
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

/// Adam optimizer with momentum and adaptive learning rates
pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32), // (beta1, beta2)
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>, // First moment estimates
    v: Vec<Vec<f32>>, // Second moment estimates
    t: usize,         // Timestep
}

impl Adam {
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
    ) -> Self {
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);

        // Initialize moment buffers
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.data().len()]).collect();

        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.data().len()]).collect();

        Adam {
            params,
            lr,
            betas,
            eps,
            weight_decay,
            m,
            v,
            t: 0,
        }
    }

    pub fn step(&mut self) {
        self.t += 1;
        // let t = self.t as f32;

        // Bias correction
        let bias_correction1 = 1.0 - self.betas.0.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.betas.1.powi(self.t as i32);
        let step_size = self.lr * (bias_correction2.sqrt() / bias_correction1);

        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad_ref() {
                let mut data = param.data_mut();
                let m_t = &mut self.m[i];
                let v_t = &mut self.v[i];

                // Update biased first and second moment estimates
                for j in 0..data.len() {
                    let g = grad[j] + self.weight_decay * data[j]; // L2 regularization

                    // m_t = β1 * m_{t-1} + (1 - β1) * g_t
                    m_t[j] = self.betas.0 * m_t[j] + (1.0 - self.betas.0) * g;

                    // v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                    v_t[j] = self.betas.1 * v_t[j] + (1.0 - self.betas.1) * g * g;

                    // Update parameters
                    data[j] -= step_size * m_t[j] / (v_t[j].sqrt() + self.eps);
                }
            }
        }
    }

    pub fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    pub fn get_lr(&self) -> f32 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    adam: Adam,
}

impl AdamW {
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
    ) -> Self {
        AdamW {
            adam: Adam::new(params, lr, betas, eps, weight_decay),
        }
    }

    pub fn step(&mut self) {
        // AdamW applies weight decay directly to weights, not to gradients
        let weight_decay = self.adam.weight_decay;
        let lr = self.adam.lr;

        // Apply weight decay
        if weight_decay > 0.0 {
            for param in &self.adam.params {
                let mut data = param.data_mut();
                for w in data.iter_mut() {
                    *w *= 1.0 - lr * weight_decay;
                }
            }
        }

        // Set weight_decay to 0 temporarily for standard Adam update
        let original_wd = self.adam.weight_decay;
        self.adam.weight_decay = 0.0;
        self.adam.step();
        self.adam.weight_decay = original_wd;
    }

    pub fn zero_grad(&mut self) {
        self.adam.zero_grad();
    }

    pub fn get_lr(&self) -> f32 {
        self.adam.get_lr()
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.adam.set_lr(lr);
    }
}

/// Learning rate scheduler trait
pub trait LRScheduler {
    fn step(&mut self, metrics: Option<f32>);
    fn get_lr(&self) -> f32;
}

/// Step learning rate scheduler - reduces LR by gamma every step_size epochs
pub struct StepLR {
    // base_lr: f32,
    current_lr: f32,
    step_size: usize,
    gamma: f32,
    current_epoch: usize,
}

impl StepLR {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        StepLR {
            // base_lr,
            current_lr: base_lr,
            step_size,
            gamma,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, _metrics: Option<f32>) {
        self.current_epoch += 1;
        if self.current_epoch % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }
}

/// Exponential learning rate scheduler
pub struct ExponentialLR {
    // base_lr: f32,
    current_lr: f32,
    gamma: f32,
}

impl ExponentialLR {
    pub fn new(base_lr: f32, gamma: f32) -> Self {
        ExponentialLR {
            // base_lr,
            current_lr: base_lr,
            gamma,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, _metrics: Option<f32>) {
        self.current_lr *= self.gamma;
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }
}

/// Cosine annealing learning rate scheduler
pub struct CosineAnnealingLR {
    base_lr: f32,
    min_lr: f32,
    current_lr: f32,
    t_max: usize,
    current_epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, t_max: usize, min_lr: Option<f32>) -> Self {
        let min_lr = min_lr.unwrap_or(0.0);
        CosineAnnealingLR {
            base_lr,
            min_lr,
            current_lr: base_lr,
            t_max,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, _metrics: Option<f32>) {
        self.current_epoch += 1;

        let progress = (self.current_epoch as f32) / (self.t_max as f32);
        let cos_val = (1.0 + (progress * std::f32::consts::PI).cos()) / 2.0;

        self.current_lr = self.min_lr + (self.base_lr - self.min_lr) * cos_val;
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }
}

/// ReduceLROnPlateau - reduces learning rate when metric stops improving
pub struct ReduceLROnPlateau {
    current_lr: f32,
    factor: f32,
    patience: usize,
    min_lr: f32,
    mode: String, // "min" or "max"
    best_metric: f32,
    patience_counter: usize,
}

impl ReduceLROnPlateau {
    pub fn new(
        initial_lr: f32,
        factor: f32,
        patience: usize,
        min_lr: Option<f32>,
        mode: Option<String>,
    ) -> Self {
        let mode = mode.unwrap_or_else(|| "min".to_string());
        let best_metric = if mode == "min" {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };

        ReduceLROnPlateau {
            current_lr: initial_lr,
            factor,
            patience,
            min_lr: min_lr.unwrap_or(1e-6),
            mode,
            best_metric,
            patience_counter: 0,
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn step(&mut self, metrics: Option<f32>) {
        if let Some(metric) = metrics {
            let improved = if self.mode == "min" {
                metric < self.best_metric
            } else {
                metric > self.best_metric
            };

            if improved {
                self.best_metric = metric;
                self.patience_counter = 0;
            } else {
                self.patience_counter += 1;

                if self.patience_counter >= self.patience {
                    self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                    self.patience_counter = 0;
                    println!("Reducing learning rate to {:.6}", self.current_lr);
                }
            }
        }
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;

    #[test]
    fn test_adam_optimizer() {
        let _tape = Tape::reset();

        // Create some parameters
        let w = Tensor::randn(&[10, 10]).requires_grad();
        let b = Tensor::randn(&[10]).requires_grad();

        // Create optimizer
        let mut optimizer = Adam::new(vec![w.clone(), b.clone()], 0.001, None, None, None);

        // Simulate some gradients
        *w.grad.write().unwrap() = Some(vec![0.1; 100]);
        *b.grad.write().unwrap() = Some(vec![0.1; 10]);

        let w_before = w.data().clone();

        // Take optimization step
        optimizer.step();

        // Check that parameters changed
        let w_after = w.data();
        for (before, after) in w_before.iter().zip(w_after.iter()) {
            assert!((before - after).abs() > 1e-6);
        }

        // Check zero_grad
        optimizer.zero_grad();
        assert!(w.grad_ref().is_none());
        assert!(b.grad_ref().is_none());
    }

    #[test]
    fn test_lr_schedulers() {
        // Test StepLR
        let mut step_lr = StepLR::new(0.1, 3, 0.5);
        assert!((step_lr.get_lr() - 0.1).abs() < 1e-6);

        step_lr.step(None);
        step_lr.step(None);
        step_lr.step(None);
        assert!((step_lr.get_lr() - 0.05).abs() < 1e-6);

        // Test ExponentialLR
        let mut exp_lr = ExponentialLR::new(0.1, 0.9);
        exp_lr.step(None);
        assert!((exp_lr.get_lr() - 0.09).abs() < 1e-6);

        // Test CosineAnnealingLR
        let mut cos_lr = CosineAnnealingLR::new(0.1, 10, Some(0.01));
        let initial = cos_lr.get_lr();
        cos_lr.step(None);
        let after_one = cos_lr.get_lr();
        assert!(after_one < initial);

        // Test ReduceLROnPlateau
        let mut plateau_lr = ReduceLROnPlateau::new(0.1, 0.5, 2, None, Some("min".to_string()));

        // Simulate no improvement
        plateau_lr.step(Some(1.0));
        plateau_lr.step(Some(1.0));
        plateau_lr.step(Some(1.0)); // Should trigger reduction
        assert!((plateau_lr.get_lr() - 0.05).abs() < 1e-6);
    }
}
