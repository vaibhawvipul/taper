//! Normalization layers.
//!
//! Both backwards are hand-derived rather than composed from primitive ops, so
//! each is checked against finite differences — the same discipline that caught
//! conv2d having no gradient path at all.

use taper::nn::Module;
use taper::norm::{BatchNorm2d, LayerNorm};
use taper::{Tape, Tensor};

/// Central-difference gradient of `objective` at `values`.
fn numeric_gradient(values: &[f32], objective: impl Fn(&[f32]) -> f32) -> Vec<f32> {
    let eps = 1e-2;
    (0..values.len())
        .map(|i| {
            let mut plus = values.to_vec();
            plus[i] += eps;
            let mut minus = values.to_vec();
            minus[i] -= eps;
            (objective(&plus) - objective(&minus)) / (2.0 * eps)
        })
        .collect()
}

/// Deterministic, non-degenerate values.
fn sample(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.13)
        .collect()
}

// --- LayerNorm ---

#[test]
fn layer_norm_standardizes_each_row() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[2, 4]);
    let ln = LayerNorm::new(&[4], true, None);
    let y = ln.forward(&x).to_vec();

    // With gamma=1, beta=0 each row has zero mean and unit variance.
    for row in y.chunks(4) {
        let mean = row.iter().sum::<f32>() / 4.0;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "row mean {mean} should be ~0");
        assert!((var - 1.0).abs() < 1e-3, "row variance {var} should be ~1");
    }

    // Both rows are the same pattern at different scale, so they standardize
    // to the same values — that is the point of normalizing per sample.
    for i in 0..4 {
        assert!((y[i] - y[4 + i]).abs() < 1e-4);
    }
}

#[test]
fn layer_norm_is_independent_of_the_other_rows() {
    let ln = LayerNorm::new(&[3], true, None);
    let alone = ln
        .forward(&Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]))
        .to_vec();
    let batched = ln
        .forward(&Tensor::new(
            vec![1.0, 2.0, 3.0, 100.0, -50.0, 7.0],
            &[2, 3],
        ))
        .to_vec();

    assert_eq!(alone.len(), 3);
    for i in 0..3 {
        assert!((alone[i] - batched[i]).abs() < 1e-5);
    }
}

#[test]
fn layer_norm_applies_its_affine_parameters() {
    let ln = LayerNorm::new(&[2], true, None);
    ln.gamma.data_mut().copy_from_slice(&[2.0, 3.0]);
    ln.beta
        .as_ref()
        .unwrap()
        .data_mut()
        .copy_from_slice(&[1.0, -1.0]);

    let y = ln.forward(&Tensor::new(vec![0.0, 1.0], &[1, 2])).to_vec();
    // Standardized row is [-1, 1]; scaled and shifted gives [-1, 2].
    assert!((y[0] - (-1.0)).abs() < 1e-3, "got {y:?}");
    assert!((y[1] - 2.0).abs() < 1e-3, "got {y:?}");
}

#[test]
fn layer_norm_gradients_match_finite_differences() {
    let shape = [3usize, 4];
    let x = sample(12);
    let coefficients = sample(12).iter().map(|v| v + 0.7).collect::<Vec<_>>();

    let ln = LayerNorm::new(&[4], true, None);
    ln.gamma.data_mut().copy_from_slice(&[1.3, -0.7, 0.5, 2.0]);
    ln.beta
        .as_ref()
        .unwrap()
        .data_mut()
        .copy_from_slice(&[0.1, 0.2, -0.3, 0.4]);

    // Scalar objective so a single backward gives every partial derivative.
    let objective = |values: &[f32]| -> f32 {
        let out = ln.forward(&Tensor::new(values.to_vec(), &shape));
        out.to_vec()
            .iter()
            .zip(&coefficients)
            .map(|(o, c)| o * c)
            .sum()
    };

    Tape::reset();
    let xt = Tensor::new(x.clone(), &shape).requires_grad();
    let out = ln.forward(&xt);
    let weighted = &out * &Tensor::new(coefficients.clone(), &shape);
    weighted.sum(None, false).backward();

    let analytic = xt.grad_ref().expect("no input gradient");
    let numeric = numeric_gradient(&x, objective);
    for i in 0..x.len() {
        assert!(
            (analytic[i] - numeric[i]).abs() < 2e-2,
            "input grad {i}: analytic {} vs numeric {}",
            analytic[i],
            numeric[i]
        );
    }

    // And the affine parameters.
    let gamma_now = ln.gamma.to_vec();
    let gamma_numeric = numeric_gradient(&gamma_now, |g| {
        ln.gamma.data_mut().copy_from_slice(g);
        let v = objective(&x);
        ln.gamma.data_mut().copy_from_slice(&gamma_now);
        v
    });
    let gamma_analytic = ln.gamma.grad_ref().expect("no gamma gradient");
    for i in 0..gamma_now.len() {
        assert!(
            (gamma_analytic[i] - gamma_numeric[i]).abs() < 2e-2,
            "gamma grad {i}: analytic {} vs numeric {}",
            gamma_analytic[i],
            gamma_numeric[i]
        );
    }
}

// --- BatchNorm2d ---

#[test]
fn batch_norm_standardizes_each_channel() {
    let (n, c, h, w) = (2usize, 2, 2, 2);
    let x = Tensor::new(sample(n * c * h * w), &[n, c, h, w]);
    let bn = BatchNorm2d::new(c);
    let y = bn.forward(&x).to_vec();

    let spatial = h * w;
    for ch in 0..c {
        let vals: Vec<f32> = (0..n)
            .flat_map(|b| {
                let base = b * c * spatial + ch * spatial;
                (base..base + spatial).map(|i| y[i])
            })
            .collect();
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
        assert!(mean.abs() < 1e-4, "channel {ch} mean {mean}");
        assert!((var - 1.0).abs() < 1e-2, "channel {ch} variance {var}");
    }
}

/// The property PR-style BatchNorm without running statistics gets wrong: at
/// inference a prediction must not depend on what else shares the batch.
#[test]
fn eval_mode_predictions_do_not_depend_on_the_batch() {
    let (c, h, w) = (2usize, 2, 2);
    let mut bn = BatchNorm2d::new(c);

    // Train briefly so the running statistics are populated.
    let train_x = Tensor::new(sample(4 * c * h * w), &[4, c, h, w]);
    for _ in 0..5 {
        bn.forward(&train_x);
    }
    bn.set_training(false);

    let one = sample(c * h * w);
    let alone = bn
        .forward(&Tensor::new(one.clone(), &[1, c, h, w]))
        .to_vec();

    // The same sample, now sharing a batch with wildly different data.
    let mut together = one.clone();
    together.extend(sample(c * h * w).iter().map(|v| v * 50.0 + 30.0));
    let batched = bn.forward(&Tensor::new(together, &[2, c, h, w])).to_vec();

    for i in 0..one.len() {
        assert!(
            (alone[i] - batched[i]).abs() < 1e-5,
            "element {i} changed with batch composition: {} vs {}",
            alone[i],
            batched[i]
        );
    }
}

#[test]
fn eval_mode_works_with_a_batch_of_one() {
    let (c, h, w) = (2usize, 2, 2);
    let mut bn = BatchNorm2d::new(c);
    bn.set_training(false);

    // Batch statistics would give zero variance here and divide by ~eps.
    let y = bn.forward(&Tensor::new(sample(c * h * w), &[1, c, h, w]));
    assert!(
        y.to_vec().iter().all(|v| v.is_finite()),
        "batch-of-one inference produced non-finite values: {:?}",
        y.to_vec()
    );
}

#[test]
fn running_statistics_track_the_data_and_are_not_parameters() {
    let (n, c, h, w) = (4usize, 2, 2, 2);
    let bn = BatchNorm2d::new(c);

    // Start at the standard-normal defaults.
    assert_eq!(bn.running_mean().to_vec(), vec![0.0, 0.0]);
    assert_eq!(bn.running_var().to_vec(), vec![1.0, 1.0]);

    // Data with a large positive offset should drag the running mean up.
    let shifted: Vec<f32> = sample(n * c * h * w).iter().map(|v| v + 10.0).collect();
    let x = Tensor::new(shifted, &[n, c, h, w]);
    for _ in 0..20 {
        bn.forward(&x);
    }
    assert!(
        bn.running_mean().to_vec().iter().all(|m| *m > 5.0),
        "running mean did not track the data: {:?}",
        bn.running_mean().to_vec()
    );

    // They must not be handed to an optimizer.
    assert_eq!(
        bn.parameters().len(),
        2,
        "only gamma and beta are learnable"
    );
    assert_eq!(
        bn.buffers().len(),
        2,
        "running mean and variance are buffers"
    );
}

#[test]
fn batch_norm_gradients_match_finite_differences() {
    let (n, c, h, w) = (3usize, 2, 2, 2);
    let shape = [n, c, h, w];
    let count = n * c * h * w;
    let x = sample(count);
    let coefficients: Vec<f32> = sample(count).iter().map(|v| v + 0.4).collect();

    let bn = BatchNorm2d::new(c);
    bn.gamma.data_mut().copy_from_slice(&[1.4, -0.6]);
    bn.beta.data_mut().copy_from_slice(&[0.25, -0.1]);

    let objective = |values: &[f32]| -> f32 {
        let out = bn.forward(&Tensor::new(values.to_vec(), &shape));
        out.to_vec()
            .iter()
            .zip(&coefficients)
            .map(|(o, cf)| o * cf)
            .sum()
    };

    Tape::reset();
    let xt = Tensor::new(x.clone(), &shape).requires_grad();
    let out = bn.forward(&xt);
    let weighted = &out * &Tensor::new(coefficients.clone(), &shape);
    weighted.sum(None, false).backward();

    let analytic = xt.grad_ref().expect("no input gradient");
    let numeric = numeric_gradient(&x, objective);
    for i in 0..count {
        assert!(
            (analytic[i] - numeric[i]).abs() < 2e-2,
            "input grad {i}: analytic {} vs numeric {}",
            analytic[i],
            numeric[i]
        );
    }

    let gamma_now = bn.gamma.to_vec();
    let gamma_numeric = numeric_gradient(&gamma_now, |g| {
        bn.gamma.data_mut().copy_from_slice(g);
        let v = objective(&x);
        bn.gamma.data_mut().copy_from_slice(&gamma_now);
        v
    });
    let gamma_analytic = bn.gamma.grad_ref().expect("no gamma gradient");
    for i in 0..c {
        assert!(
            (gamma_analytic[i] - gamma_numeric[i]).abs() < 5e-2,
            "gamma grad {i}: analytic {} vs numeric {}",
            gamma_analytic[i],
            gamma_numeric[i]
        );
    }
}

#[test]
fn eval_mode_gradients_match_finite_differences() {
    // With running statistics held constant the derivative is a plain scale,
    // which is a different code path from the training one.
    let (n, c, h, w) = (2usize, 2, 2, 2);
    let shape = [n, c, h, w];
    let count = n * c * h * w;
    let x = sample(count);
    let coefficients: Vec<f32> = sample(count).iter().map(|v| v + 0.9).collect();

    let mut bn = BatchNorm2d::new(c);
    bn.forward(&Tensor::new(sample(count), &shape)); // populate running stats
    bn.set_training(false);

    let objective = |values: &[f32]| -> f32 {
        let out = bn.forward(&Tensor::new(values.to_vec(), &shape));
        out.to_vec()
            .iter()
            .zip(&coefficients)
            .map(|(o, cf)| o * cf)
            .sum()
    };

    Tape::reset();
    let xt = Tensor::new(x.clone(), &shape).requires_grad();
    let out = bn.forward(&xt);
    let weighted = &out * &Tensor::new(coefficients.clone(), &shape);
    weighted.sum(None, false).backward();

    let analytic = xt.grad_ref().unwrap();
    let numeric = numeric_gradient(&x, objective);
    for i in 0..count {
        assert!(
            (analytic[i] - numeric[i]).abs() < 2e-2,
            "eval grad {i}: analytic {} vs numeric {}",
            analytic[i],
            numeric[i]
        );
    }
}

#[test]
#[should_panic(expected = "more than one element per channel")]
fn training_rejects_a_degenerate_batch() {
    let bn = BatchNorm2d::new(2);
    // One element per channel: the variance is identically zero.
    bn.forward(&Tensor::new(vec![1.0, 2.0], &[1, 2, 1, 1]));
}

/// A model reloaded without its running statistics normalizes with reset ones
/// and infers differently, so buffers have to be part of the checkpoint.
#[test]
fn running_statistics_survive_a_checkpoint() {
    use taper::nn::Sequential;
    use taper::safetensors;

    let (n, c, h, w) = (4usize, 2, 2, 2);
    let mut path = std::env::temp_dir();
    path.push(format!("taper_bn_{}.safetensors", std::process::id()));

    let mut trained = Sequential::new(vec![Box::new(BatchNorm2d::new(c))]);
    let data = Tensor::new(
        sample(n * c * h * w).iter().map(|v| v + 4.0).collect(),
        &[n, c, h, w],
    );
    for _ in 0..10 {
        trained.forward(&data);
    }
    trained.set_training(false);

    let probe = Tensor::new(sample(c * h * w), &[1, c, h, w]);
    let expected = trained.forward(&probe).to_vec();

    safetensors::save_module(&trained, &path).unwrap();

    // A fresh layer starts from the standard-normal defaults, so it must differ
    // before loading and agree exactly afterwards.
    let mut restored = Sequential::new(vec![Box::new(BatchNorm2d::new(c))]);
    restored.set_training(false);
    assert_ne!(restored.forward(&probe).to_vec(), expected);

    safetensors::load_module(&restored, &path).unwrap();
    let after = restored.forward(&probe).to_vec();
    for (a, e) in after.iter().zip(&expected) {
        assert!(
            (a - e).abs() < 1e-6,
            "reloaded model infers differently: {a} vs {e}"
        );
    }

    std::fs::remove_file(&path).ok();
}
