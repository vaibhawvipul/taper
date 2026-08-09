//! Normalization layers.
//!
//! Both layers standardize a group of activations to zero mean and unit
//! variance, then apply a learnable per-feature scale and shift. They differ in
//! what they group over:
//!
//! - [`LayerNorm`] groups the trailing dimensions of a single sample, so it is
//!   independent of batch composition and behaves identically in training and
//!   evaluation.
//! - [`BatchNorm2d`] groups each channel across the batch and spatial extent,
//!   which means it *does* depend on batch composition — so it tracks running
//!   statistics during training and uses those at evaluation time.
//!
//! The forward and backward passes are fused rather than composed from
//! primitive ops: the normalization statistics depend on the input, so the
//! derivative does not factor into the elementwise ops the crate already has.
//! Both backwards are checked against finite differences in `tests/norm.rs`.

use crate::tape::Tape;
use crate::tensor::read_recovering;
use crate::{Tensor, nn::Module, ops};

/// The shared backward for a normalized group.
///
/// With `m` elements in a group, `xhat` the standardized values and `dxhat` the
/// incoming gradient already scaled by gamma:
///
/// ```text
/// dx = inv_std / m * (m * dxhat - Σ dxhat - xhat * Σ (dxhat * xhat))
/// ```
///
/// The two sums are what makes this irreducible to elementwise ops: every input
/// in the group influences every output through the mean and variance.
#[inline]
fn group_backward(dxhat: &[f32], xhat: &[f32], inv_std: f32, out: &mut [f32]) {
    let m = dxhat.len() as f32;
    let sum_dxhat: f32 = dxhat.iter().sum();
    let sum_dxhat_xhat: f32 = dxhat.iter().zip(xhat).map(|(d, h)| d * h).sum();

    for ((o, &d), &h) in out.iter_mut().zip(dxhat).zip(xhat) {
        *o = inv_std / m * (m * d - sum_dxhat - h * sum_dxhat_xhat);
    }
}

/// Mean and variance of a slice, in one pass over deviations for stability.
#[inline]
fn mean_var(values: &[f32]) -> (f32, f32) {
    let m = values.len() as f32;
    let mean = values.iter().sum::<f32>() / m;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / m;
    (mean, var)
}

/// Normalizes over the trailing dimensions of each sample.
///
/// Given an input whose trailing dimensions have `normalized_size` elements in
/// total, each such block is standardized independently, then scaled and
/// shifted by learnable per-element parameters.
#[derive(Debug)]
pub struct LayerNorm {
    /// Learnable scale, shape `[normalized_size]`.
    pub gamma: Tensor,
    /// Learnable shift, shape `[normalized_size]`.
    pub beta: Option<Tensor>,
    normalized_size: usize,
    eps: f32,
}

impl LayerNorm {
    /// `normalized_shape` gives the trailing dimensions to normalize over —
    /// `&[features]` for `[batch, features]`, `&[d_model]` for a transformer's
    /// `[batch, seq, d_model]`.
    pub fn new(normalized_shape: &[usize], affine: bool, eps: Option<f32>) -> Self {
        assert!(
            !normalized_shape.is_empty(),
            "LayerNorm: normalized_shape must not be empty"
        );
        let normalized_size: usize = normalized_shape.iter().product();
        assert!(
            normalized_size > 0,
            "LayerNorm: normalized_shape {normalized_shape:?} has no elements"
        );

        LayerNorm {
            gamma: Tensor::new(vec![1.0; normalized_size], &[normalized_size]).requires_grad(),
            beta: affine.then(|| {
                Tensor::new(vec![0.0; normalized_size], &[normalized_size]).requires_grad()
            }),
            normalized_size,
            eps: eps.unwrap_or(1e-5),
        }
    }

    /// Convenience for the common `[.., features]` case.
    pub fn with_features(features: usize) -> Self {
        Self::new(&[features], true, None)
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let size = self.normalized_size;
        assert!(
            input.numel().is_multiple_of(size),
            "LayerNorm: input of {} elements is not divisible by normalized size {size}",
            input.numel()
        );

        let values = input.to_vec();
        let rows = values.len() / size;
        let gamma = self.gamma.to_vec();
        let beta = self.beta.as_ref().map(|b| b.to_vec());

        // Keep the standardized values: the backward needs them, and
        // recomputing would mean a second pass over the input.
        let mut xhat = vec![0.0f32; values.len()];
        let mut inv_stds = Vec::with_capacity(rows);
        let mut out = vec![0.0f32; values.len()];

        for r in 0..rows {
            let span = r * size..(r + 1) * size;
            let (mean, var) = mean_var(&values[span.clone()]);
            let inv_std = 1.0 / (var + self.eps).sqrt();
            inv_stds.push(inv_std);

            for i in 0..size {
                let h = (values[span.start + i] - mean) * inv_std;
                xhat[span.start + i] = h;
                out[span.start + i] = h * gamma[i] + beta.as_ref().map_or(0.0, |b| b[i]);
            }
        }

        let mut output = Tensor::new(out, input.shape());

        let track = input.requires_grad
            || self.gamma.requires_grad
            || self.beta.as_ref().is_some_and(|b| b.requires_grad);
        if track {
            output.requires_grad = true;
            let x = input.clone();
            let gamma_t = self.gamma.clone();
            let beta_t = self.beta.clone();
            let out_t = output.clone();

            Tape::push_unary_op(input, &output, move || {
                let Some(gout) = read_recovering(&out_t.grad).as_ref().cloned() else {
                    return;
                };

                let mut dgamma = vec![0.0f32; size];
                let mut dbeta = vec![0.0f32; size];
                let mut dx = vec![0.0f32; gout.len()];
                let mut dxhat = vec![0.0f32; size];

                for (r, &inv_std) in inv_stds.iter().enumerate() {
                    let base = r * size;
                    for i in 0..size {
                        let g = gout[base + i];
                        dgamma[i] += g * xhat[base + i];
                        dbeta[i] += g;
                        dxhat[i] = g * gamma[i];
                    }
                    group_backward(
                        &dxhat,
                        &xhat[base..base + size],
                        inv_std,
                        &mut dx[base..base + size],
                    );
                }

                if x.requires_grad {
                    ops::accumulate_grad(&x, &dx);
                }
                if gamma_t.requires_grad {
                    ops::accumulate_grad(&gamma_t, &dgamma);
                }
                if let Some(b) = &beta_t
                    && b.requires_grad
                {
                    ops::accumulate_grad(b, &dbeta);
                }
            });
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.gamma.clone()];
        if let Some(b) = &self.beta {
            p.push(b.clone());
        }
        p
    }
}

/// Per-channel normalization over a 4D batch, `[N, C, H, W]`.
///
/// During training each channel is standardized using that batch's statistics,
/// and running estimates are updated. At evaluation the running estimates are
/// used instead, so a prediction depends only on its own input — normalizing
/// with batch statistics at inference would make results depend on whatever
/// else shared the batch, and would divide by ~`eps` for a batch of one.
#[derive(Debug)]
pub struct BatchNorm2d {
    /// Learnable scale, shape `[channels]`.
    pub gamma: Tensor,
    /// Learnable shift, shape `[channels]`.
    pub beta: Tensor,
    /// Running mean, shape `[channels]`. Not learnable: updated by observation,
    /// never by the optimizer.
    running_mean: Tensor,
    /// Running variance, shape `[channels]`.
    running_var: Tensor,
    channels: usize,
    momentum: f32,
    eps: f32,
    training: bool,
}

impl BatchNorm2d {
    pub fn new(channels: usize) -> Self {
        Self::with_config(channels, 0.1, 1e-5)
    }

    /// `momentum` weights each batch's contribution to the running estimates.
    pub fn with_config(channels: usize, momentum: f32, eps: f32) -> Self {
        assert!(channels > 0, "BatchNorm2d: channels must be non-zero");
        assert!(
            (0.0..=1.0).contains(&momentum),
            "BatchNorm2d: momentum must be in [0, 1], got {momentum}"
        );

        BatchNorm2d {
            gamma: Tensor::new(vec![1.0; channels], &[channels]).requires_grad(),
            beta: Tensor::new(vec![0.0; channels], &[channels]).requires_grad(),
            // Start as a standard normal, so an untrained layer in eval mode is
            // a no-op rather than a division by zero.
            running_mean: Tensor::new(vec![0.0; channels], &[channels]),
            running_var: Tensor::new(vec![1.0; channels], &[channels]),
            channels,
            momentum,
            eps,
            training: true,
        }
    }

    pub fn is_training(&self) -> bool {
        self.training
    }

    /// The running mean tracked so far.
    pub fn running_mean(&self) -> &Tensor {
        &self.running_mean
    }

    /// The running variance tracked so far.
    pub fn running_var(&self) -> &Tensor {
        &self.running_var
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape().len(),
            4,
            "BatchNorm2d expects [N, C, H, W], got {:?}",
            input.shape()
        );
        let (n, c, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        assert_eq!(
            c, self.channels,
            "BatchNorm2d: layer has {} channels but input has {c}",
            self.channels
        );

        let values = input.to_vec();
        let gamma = self.gamma.to_vec();
        let beta = self.beta.to_vec();
        let spatial = h * w;
        let group = n * spatial; // elements per channel

        if self.training {
            assert!(
                group > 1,
                "BatchNorm2d: training needs more than one element per channel, \
                 got batch {n} of {h}x{w}"
            );
        }

        // Positions belonging to channel `ch`, in NCHW order.
        let channel_indices = |ch: usize| {
            (0..n).flat_map(move |b| {
                let base = b * c * spatial + ch * spatial;
                base..base + spatial
            })
        };

        let mut xhat = vec![0.0f32; values.len()];
        let mut out = vec![0.0f32; values.len()];
        let mut inv_stds = Vec::with_capacity(c);

        for ch in 0..c {
            let (mean, var) = if self.training {
                let slice: Vec<f32> = channel_indices(ch).map(|i| values[i]).collect();
                let (mean, var) = mean_var(&slice);

                // Bessel-corrected variance goes into the running estimate, so
                // it is an unbiased estimator of the population variance, while
                // the biased one is used to normalize this batch.
                let unbiased = var * group as f32 / (group as f32 - 1.0);
                let mut rm = self.running_mean.data_mut();
                let mut rv = self.running_var.data_mut();
                rm[ch] = (1.0 - self.momentum) * rm[ch] + self.momentum * mean;
                rv[ch] = (1.0 - self.momentum) * rv[ch] + self.momentum * unbiased;

                (mean, var)
            } else {
                (
                    self.running_mean.to_vec()[ch],
                    self.running_var.to_vec()[ch],
                )
            };

            let inv_std = 1.0 / (var + self.eps).sqrt();
            inv_stds.push(inv_std);

            for i in channel_indices(ch) {
                let hval = (values[i] - mean) * inv_std;
                xhat[i] = hval;
                out[i] = hval * gamma[ch] + beta[ch];
            }
        }

        let mut output = Tensor::new(out, input.shape());

        let track = input.requires_grad || self.gamma.requires_grad || self.beta.requires_grad;
        if track {
            output.requires_grad = true;
            let x = input.clone();
            let gamma_t = self.gamma.clone();
            let beta_t = self.beta.clone();
            let out_t = output.clone();
            let training = self.training;

            Tape::push_unary_op(input, &output, move || {
                let Some(gout) = read_recovering(&out_t.grad).as_ref().cloned() else {
                    return;
                };

                let mut dgamma = vec![0.0f32; c];
                let mut dbeta = vec![0.0f32; c];
                let mut dx = vec![0.0f32; gout.len()];

                for ch in 0..c {
                    let positions: Vec<usize> = (0..n)
                        .flat_map(|b| {
                            let base = b * c * spatial + ch * spatial;
                            base..base + spatial
                        })
                        .collect();

                    let mut dxhat = Vec::with_capacity(positions.len());
                    let mut hats = Vec::with_capacity(positions.len());
                    for &i in &positions {
                        dgamma[ch] += gout[i] * xhat[i];
                        dbeta[ch] += gout[i];
                        dxhat.push(gout[i] * gamma[ch]);
                        hats.push(xhat[i]);
                    }

                    if training {
                        // The batch statistics depend on x, so every element in
                        // the channel contributes to every other's gradient.
                        let mut grads = vec![0.0f32; positions.len()];
                        group_backward(&dxhat, &hats, inv_stds[ch], &mut grads);
                        for (&i, g) in positions.iter().zip(grads) {
                            dx[i] = g;
                        }
                    } else {
                        // Running statistics are constants here, so the
                        // derivative is a plain scale.
                        for (&i, d) in positions.iter().zip(dxhat) {
                            dx[i] = d * inv_stds[ch];
                        }
                    }
                }

                if x.requires_grad {
                    ops::accumulate_grad(&x, &dx);
                }
                if gamma_t.requires_grad {
                    ops::accumulate_grad(&gamma_t, &dgamma);
                }
                if beta_t.requires_grad {
                    ops::accumulate_grad(&beta_t, &dbeta);
                }
            });
        }

        output
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Running statistics are deliberately absent: handing them to an
        // optimizer would have it apply momentum and weight decay to what are
        // observations, not weights. They are exposed as buffers instead.
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn buffers(&self) -> Vec<Tensor> {
        vec![self.running_mean.clone(), self.running_var.clone()]
    }
}
