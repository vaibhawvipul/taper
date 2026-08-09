//! Verify a backward pass against finite differences.
//!
//! Every differentiable operation in this crate carries a hand-written
//! backward, and a hand-written backward that is quietly wrong is the failure
//! mode this library has hit most often: convolution once trained for an entire
//! epoch without its weights receiving a gradient at all, because two ops
//! rebuilt their outputs in a way that dropped the backward edge. Nothing about
//! that shows up as an error — the loss still falls, because other layers
//! compensate.
//!
//! This module makes the check cheap enough to apply to everything, including
//! your own [`Module`](crate::nn::Module) implementations:
//!
//! ```
//! use taper::{Tensor, gradcheck};
//!
//! let x = Tensor::new(vec![0.3, -1.2, 0.8], &[3]).requires_grad();
//! gradcheck::check(&[x.clone()], |t| t[0].exp()).expect("exp gradient is wrong");
//! ```
//!
//! The comparison is a central difference, so it is accurate to roughly `eps²`
//! and is evaluated in `f32`. That sets a floor on how tight the tolerance can
//! be; see [`GradCheck`] to loosen or tighten it.

use std::fmt;

use crate::Tensor;
use crate::tape::{self, Tape};

/// One element whose analytic gradient disagreed with the numeric one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mismatch {
    /// Index into the `inputs` slice.
    pub input: usize,
    /// Flat index of the element within that input.
    pub element: usize,
    pub analytic: f32,
    pub numeric: f32,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "input {} element {}: analytic {} vs numeric {} (off by {:.3e})",
            self.input,
            self.element,
            self.analytic,
            self.numeric,
            (self.analytic - self.numeric).abs()
        )
    }
}

/// Every disagreement found by a check.
#[derive(Debug, Clone, PartialEq)]
pub struct GradError {
    pub mismatches: Vec<Mismatch>,
    /// How many gradient elements were compared in total.
    pub checked: usize,
}

impl fmt::Display for GradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} of {} gradient elements disagree with finite differences:",
            self.mismatches.len(),
            self.checked
        )?;
        // A wrong backward usually breaks many elements at once; a handful is
        // enough to identify it.
        for m in self.mismatches.iter().take(8) {
            writeln!(f, "  {m}")?;
        }
        if self.mismatches.len() > 8 {
            write!(f, "  … and {} more", self.mismatches.len() - 8)?;
        }
        Ok(())
    }
}

impl std::error::Error for GradError {}

/// Tolerances for a gradient check.
#[derive(Debug, Clone, Copy)]
pub struct GradCheck {
    /// Perturbation size for the central difference.
    pub eps: f32,
    /// Absolute tolerance.
    pub atol: f32,
    /// Relative tolerance, scaled by the larger of the two gradients.
    pub rtol: f32,
}

impl Default for GradCheck {
    fn default() -> Self {
        // `eps` is a compromise: smaller values are a better approximation of
        // the derivative but lose more to f32 cancellation in `f(x+h) - f(x-h)`.
        GradCheck {
            eps: 1e-2,
            atol: 2e-2,
            rtol: 2e-2,
        }
    }
}

/// Deterministic weights in `[-1, 1)`, avoiding a uniform reduction.
///
/// Reducing the output with a plain sum would give every element the same
/// upstream gradient, which lets errors that are antisymmetric across a group
/// cancel out — exactly the shape of a mistake a normalization backward makes.
fn probe_weights(n: usize) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            // xorshift64*, so the sequence is fixed and failures reproduce.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40; // 24 bits
            (bits as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

impl GradCheck {
    /// Compare the analytic gradient of `f` at `inputs` against finite differences.
    ///
    /// `f` must be a pure function of `inputs` and may return a tensor of any
    /// shape; it is reduced to a scalar internally with fixed weights.
    ///
    /// # Panics
    /// If an input is not a dense `f32` tensor with `requires_grad` set — the
    /// check has to perturb individual elements in place.
    pub fn check<F>(&self, inputs: &[Tensor], f: F) -> Result<(), GradError>
    where
        F: Fn(&[Tensor]) -> Tensor,
    {
        assert!(!inputs.is_empty(), "gradcheck: no inputs to differentiate");
        for (i, t) in inputs.iter().enumerate() {
            assert!(
                t.requires_grad,
                "gradcheck: input {i} does not require grad, so it has no gradient to check"
            );
        }

        // Shape the reduction to whatever `f` produces.
        let weights = {
            let _guard = tape::no_grad();
            probe_weights(f(inputs).numel())
        };

        let objective = |inputs: &[Tensor]| -> f32 {
            let _guard = tape::no_grad();
            f(inputs)
                .to_vec()
                .iter()
                .zip(&weights)
                .map(|(o, w)| o * w)
                .sum()
        };

        // Analytic pass.
        Tape::reset();
        for t in inputs {
            t.zero_grad();
        }
        let out = f(inputs);
        let weighted = &out * &Tensor::new(weights.clone(), out.shape());
        weighted.sum(None, false).backward();

        let analytic: Vec<Vec<f32>> = inputs
            .iter()
            .map(|t| {
                t.grad_ref()
                    .map(|g| g.to_vec())
                    // No gradient at all is the exact bug this exists to catch,
                    // so treat it as zeros rather than skipping the input.
                    .unwrap_or_else(|| vec![0.0; t.numel()])
            })
            .collect();

        // Numeric pass, one element at a time.
        let mut mismatches = Vec::new();
        let mut checked = 0usize;

        for (ti, tensor) in inputs.iter().enumerate() {
            for (element, &a) in analytic[ti].iter().enumerate() {
                let original = tensor.data()[element];

                tensor.data_mut()[element] = original + self.eps;
                let plus = objective(inputs);
                tensor.data_mut()[element] = original - self.eps;
                let minus = objective(inputs);
                tensor.data_mut()[element] = original;

                let numeric = (plus - minus) / (2.0 * self.eps);
                checked += 1;

                let allowed = self.atol + self.rtol * a.abs().max(numeric.abs());
                if !(a - numeric).abs().le(&allowed) {
                    mismatches.push(Mismatch {
                        input: ti,
                        element,
                        analytic: a,
                        numeric,
                    });
                }
            }
        }

        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(GradError {
                mismatches,
                checked,
            })
        }
    }
}

/// Check `f`'s gradient with the default tolerances.
///
/// See [`GradCheck`] when an operation needs looser ones — anything built on
/// `exp` over a wide range, for instance.
pub fn check<F>(inputs: &[Tensor], f: F) -> Result<(), GradError>
where
    F: Fn(&[Tensor]) -> Tensor,
{
    GradCheck::default().check(inputs, f)
}
