use crate::tape::Tape;
use crate::{Tensor, ops};

/// Binary Cross Entropy loss (mean reduction)
/// L = -mean( y*log(p) + (1-y)*log(1-p) )
pub fn bce_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let eps: f32 = 1e-7;
    let p = predictions.elements();
    let t = targets.elements();
    assert_eq!(
        p.len(),
        t.len(),
        "bce_loss: predictions and targets must match in length"
    );

    // forward
    let mut acc = 0.0f32;
    for i in 0..p.len() {
        let pi = p[i].clamp(eps, 1.0 - eps);
        let yi = t[i];
        acc -= yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln();
    }
    let mut out = Tensor::scalar(acc / p.len() as f32);

    // backward
    if predictions.requires_grad || targets.requires_grad {
        out.requires_grad = true;

        let preds = predictions.clone();
        let targs = targets.clone();
        let out_clone = out.clone();
        let n = p.len();

        Tape::push_binary_op(predictions, targets, &out, move || {
            if let Some(gout) = out_clone.grad.read().unwrap().as_ref() {
                let g = gout[0]; // scalar chain multiplier from upstream

                let pdat = preds.elements();
                let tdat = targs.elements();

                // dL/dp_i = -( y/p - (1-y)/(1-p) ) / N
                if preds.requires_grad {
                    let mut slot = preds.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; n]);
                    }
                    let gp = slot.as_mut().unwrap();
                    for i in 0..n {
                        let pi = pdat[i].clamp(1e-7, 1.0 - 1e-7);
                        let yi = tdat[i];
                        gp[i] += g * (-(yi / pi - (1.0 - yi) / (1.0 - pi))) / (n as f32);
                    }
                }

                // Optional (usually false for labels):
                // dL/dy_i = -ln(p_i) + ln(1 - p_i) = ln((1-p)/p), divided by N
                if targs.requires_grad {
                    let mut slot = targs.grad.write().unwrap();
                    if slot.is_none() {
                        *slot = Some(vec![0.0; n]);
                    }
                    let gy = slot.as_mut().unwrap();
                    for i in 0..n {
                        let pi = pdat[i].clamp(1e-7, 1.0 - 1e-7);
                        gy[i] += g * ((1.0 - pi).ln() - pi.ln()) / (n as f32);
                    }
                }
            }
        });
    }

    out
}

/// Mean Squared Error loss (mean reduction)
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let diff = predictions - targets;
    let squared = &diff * &diff;
    squared.mean()
}

pub fn softmax(x: &Tensor, dim: i32) -> Tensor {
    let ndim = x.shape().len() as i32;
    let dim = if dim < 0 { ndim + dim } else { dim } as usize;
    assert_eq!(
        dim,
        x.shape().len() - 1,
        "Only last-dim softmax is supported; got dim={dim} for shape {:?}",
        x.shape()
    );

    // Subtract max for numerical stability (broadcasting: [B,C] - [B,1])
    let (max_vals, _) = x.max(Some(dim));
    let x_shifted = x.sub_broadcast_rows(&max_vals);

    // exp(x - max(x))
    let exp_x = x_shifted.exp();

    // sum(exp(x - max(x)))  keepdim=true -> [B,1]
    let sum_exp = exp_x.sum(Some(dim), true);

    // softmax = exp(x - max(x)) / sum(exp(x - max(x)))  broadcasting: [B,C] / [B,1]
    exp_x.div_broadcast_rows(&sum_exp)
}

/// Log-softmax for numerical stability in cross-entropy
pub fn log_softmax(x: &Tensor, dim: i32) -> Tensor {
    let ndim = x.shape().len() as i32;
    let dim = if dim < 0 {
        (ndim + dim) as usize
    } else {
        dim as usize
    };
    assert_eq!(
        dim,
        x.shape().len() - 1,
        "Only last-dim log_softmax is supported; got dim={dim} for shape {:?}",
        x.shape()
    );

    // Numerically stable: subtract rowwise max
    // max_vals: [B,1]
    let (max_vals, _) = x.max(Some(dim));
    let x_shifted = x.sub_broadcast_rows(&max_vals); // [B,C]

    // log(sum(exp(x_shifted))) : [B,1]
    let sum_exp = x_shifted.exp().sum(Some(dim), /*keepdim=*/ true);
    let log_sum = sum_exp.log();

    // [B,C] - [B,1]  -> [B,C]
    x_shifted.sub_broadcast_rows(&log_sum)
}

/// Cross-entropy loss for multi-class classification
///
/// Arguments:
/// - logits: [batch_size, num_classes] - raw network outputs (before softmax)
/// - targets: [batch_size] - class indices (0 to num_classes-1)
///
/// Returns:
/// - loss: scalar tensor
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    // Accept [B] or [B,1] targets
    assert!(
        targets.shape().len() == 1 || (targets.shape().len() == 2 && targets.shape()[1] == 1),
        "Targets must be [B] or [B,1]"
    );
    assert_eq!(logits.shape().len(), 2, "Logits must be [B,C]");
    assert_eq!(
        logits.shape()[0],
        targets.shape()[0],
        "Batch sizes must match"
    );

    let b = logits.shape()[0];
    let c = logits.shape()[1];

    // log p = log_softmax(logits)
    let logp = log_softmax(logits, -1);
    let lp = logp.elements();
    let t = targets.elements();

    // NLL loss (mean)
    let mut acc = 0.0f32;
    for i in 0..b {
        let cls = t[i] as usize; // works for [B] or [B,1] (len is B)
        assert!(cls < c, "Target class {} out of bounds for {}", cls, c);
        acc -= lp[i * c + cls];
    }
    let mut out = Tensor::scalar(acc / b as f32);

    // backward: ∂L/∂logits = softmax - one_hot(target)
    if logits.requires_grad {
        out.requires_grad = true;
        let logits_c = logits.clone();
        let out_c = out.clone();

        // Precompute softmax - one_hot here, in the forward pass. Calling
        // `logp.exp()` from inside the closure ran a tape-recording op *during*
        // backward, appending nodes to the tape being replayed and leaking a
        // tensor per backward call.
        let mut base_grad: Vec<f32> = lp.iter().map(|v| v.exp()).collect();
        for i in 0..b {
            base_grad[i * c + t[i] as usize] -= 1.0;
        }
        drop(lp);
        drop(t);

        Tape::push_unary_op(logits, &out, move || {
            if let Some(g) = out_c.grad.read().expect("grad RwLock poisoned").as_ref() {
                // scale by upstream scalar / batch
                let scale = g[0] / b as f32;
                let grad: Vec<f32> = base_grad.iter().map(|v| v * scale).collect();
                ops::accumulate_grad(&logits_c, &grad);
            }
        });
    }

    out
}

/// Cross-entropy loss with one-hot encoded targets
///
/// Arguments:
/// - logits: [batch_size, num_classes] - raw network outputs
/// - targets: [batch_size, num_classes] - one-hot encoded targets
pub fn cross_entropy_loss_onehot(logits: &Tensor, targets: &Tensor) -> Tensor {
    assert_eq!(
        logits.shape(),
        targets.shape(),
        "Logits and targets shapes must match"
    );
    assert_eq!(logits.shape().len(), 2, "Must be 2D tensors");

    let batch_size = logits.shape()[0];

    // Compute log_softmax
    let log_probs = log_softmax(logits, -1);

    // Cross entropy: -sum(targets * log_probs) / batch_size
    let elementwise_prod = targets * &log_probs;
    let sum = elementwise_prod.sum(None, false);
    let loss_val = -sum.data()[0] / batch_size as f32;

    let mut loss = Tensor::scalar(loss_val);

    if logits.requires_grad {
        loss.requires_grad = true;
        let logits_clone = logits.clone();
        let loss_out = loss.clone();

        // As in `cross_entropy_loss`, softmax - targets is computed here rather
        // than inside the closure: recomputing `log_softmax` during backward
        // both re-entered the tape and repeated the whole forward pass.
        let base_grad: Vec<f32> = log_probs
            .data()
            .iter()
            .zip(targets.data().iter())
            .map(|(&lp, &t)| lp.exp() - t)
            .collect();

        Tape::push_unary_op(logits, &loss, move || {
            // A read lock is enough here; taking the write lock serialized
            // backward passes against every concurrent reader for no reason.
            if let Some(gloss) = loss_out.grad.read().expect("grad RwLock poisoned").as_ref() {
                // Gradient: (softmax - targets) * grad_loss / batch_size
                let scale = gloss[0] / batch_size as f32;
                let grad_data: Vec<f32> = base_grad.iter().map(|g| g * scale).collect();

                crate::ops::accumulate_grad(&logits_clone, &grad_data);
            }
        });
    }

    loss
}

/// Convert class indices to one-hot encoding
pub fn one_hot(indices: &Tensor, num_classes: usize) -> Tensor {
    assert_eq!(indices.shape().len(), 1, "Indices must be 1D");

    let batch_size = indices.shape()[0];
    let indices_data = indices.data();

    let mut one_hot_data = vec![0.0f32; batch_size * num_classes];

    for b in 0..batch_size {
        let class_idx = indices_data[b] as usize;
        assert!(
            class_idx < num_classes,
            "Index {} out of bounds for {} classes",
            class_idx,
            num_classes
        );
        one_hot_data[b * num_classes + class_idx] = 1.0;
    }

    Tensor::new(one_hot_data, &[batch_size, num_classes])
}

/// Number of rows whose argmax matches the target class.
///
/// Prefer this over [`accuracy`] when totalling across batches: converting to a
/// ratio and back (`(acc * batch_size) as usize`) truncates and undercounts.
pub fn correct_count(predictions: &Tensor, targets: &Tensor) -> usize {
    assert_eq!(
        predictions.shape()[0],
        targets.shape()[0],
        "Batch sizes must match"
    );

    // argmax is a tape-recording op; without this guard every metric call
    // during evaluation appended a node that lived until the next Tape::reset.
    let _guard = crate::tape::no_grad();

    let pred_classes = predictions.argmax(Some(1));
    let pred_data = pred_classes.data();
    let target_data = targets.data();

    pred_data
        .iter()
        .zip(target_data.iter())
        .filter(|(p, t)| (*p - *t).abs() < 1e-6)
        .count()
}

/// Compute accuracy between predictions and targets
pub fn accuracy(predictions: &Tensor, targets: &Tensor) -> f32 {
    let total = targets.shape()[0];
    if total == 0 {
        return 0.0;
    }
    correct_count(predictions, targets) as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;

    #[test]
    fn test_softmax() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &[2, 3]);
        let y = softmax(&x, -1);

        // Check that softmax sums to 1 along last dimension
        let sums = y.sum(Some(1), false);
        for &s in sums.data().iter() {
            assert!((s - 1.0).abs() < 1e-6);
        }

        // Check that all values are positive
        for &v in y.data().iter() {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_cross_entropy_gradient() {
        Tape::reset();

        // Simple 2-class problem
        let logits = Tensor::new(vec![2.0, 1.0, -1.0, 3.0], &[2, 2]).requires_grad();
        let targets = Tensor::new(vec![0.0, 1.0], &[2]);

        let loss = cross_entropy_loss(&logits, &targets);
        loss.backward();

        // Check that gradient exists
        assert!(logits.grad_ref().is_some());

        // Gradient should push correct class up, wrong class down
        let grad = logits.grad_ref().unwrap();
        // For first sample (target=0), grad[0] should be negative (correct class)
        // For second sample (target=1), grad[3] should be negative (correct class)
        assert!(
            grad[0] < 0.0,
            "Gradient for correct class should be negative"
        );
        assert!(
            grad[3] < 0.0,
            "Gradient for correct class should be negative"
        );
    }

    #[test]
    fn test_one_hot() {
        let indices = Tensor::new(vec![0.0, 2.0, 1.0], &[3]);
        let one_hot_result = one_hot(&indices, 3);

        let expected = vec![
            1.0, 0.0, 0.0, // class 0
            0.0, 0.0, 1.0, // class 2
            0.0, 1.0, 0.0, // class 1
        ];

        for (a, b) in one_hot_result.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_accuracy() {
        let predictions = Tensor::new(
            vec![
                0.1, 0.9, // predicts class 1
                0.8, 0.2, // predicts class 0
                0.3, 0.7, // predicts class 1
            ],
            &[3, 2],
        );

        let targets = Tensor::new(vec![1.0, 0.0, 0.0], &[3]);

        let acc = accuracy(&predictions, &targets);
        assert!((acc - 2.0 / 3.0).abs() < 1e-6, "Expected 66.67% accuracy");
    }
}
