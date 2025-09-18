use crate::tape::Tape;
use crate::ops::accumulate_grad;
use crate::Tensor;

/// Binary Cross Entropy loss for binary classification
pub fn bce_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    // BCE = -mean(y * log(p) + (1-y) * log(1-p))
    let eps = 1e-7;
    
    let pred_data = predictions.data();
    let target_data = targets.data();
    
    let mut loss = 0.0;
    for i in 0..pred_data.len() {
        let p = pred_data[i].max(eps).min(1.0 - eps);
        let y = target_data[i];
        loss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
    }
    
    let mut output = Tensor::scalar(loss / pred_data.len() as f32);

    if predictions.requires_grad {
        output.requires_grad = true;

        let preds = predictions.clone();
        let targs = targets.clone();
        let out = output.clone();

        Tape::push_binary_op(predictions, targets, &output, move || {
            if let Some(grad_output) = out.grad.borrow().as_ref() {
                let grad_scale = grad_output.data()[0] / preds.data().len() as f32;

                let grad_data: Vec<f32> = preds.data()
                    .iter()
                    .zip(targs.data().iter())
                    .map(|(&p, &y)| {
                        let p_safe = p.max(eps).min(1.0 - eps);
                        grad_scale * (p_safe - y) / (p_safe * (1.0 - p_safe))
                    })
                    .collect();

                let mut grad_tensor = Tensor::new(grad_data, &preds.shape);
                grad_tensor.requires_grad = false;
                accumulate_grad(&preds, &grad_tensor);
            }
        });
    }

    output
}

/// Mean Squared Error loss
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let diff = predictions - targets;
    let squared = &diff * &diff;
    squared.mean()
}
