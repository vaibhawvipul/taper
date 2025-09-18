use crate::Tensor;
use crate::tape::Tape;

/// Binary Cross Entropy loss (mean reduction)
/// L = -mean( y*log(p) + (1-y)*log(1-p) )
pub fn bce_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let eps: f32 = 1e-7;
    let p = predictions.data();
    let t = targets.data();
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
            if let Some(gout) = out_clone.grad.borrow().as_ref() {
                let g = gout[0]; // scalar chain multiplier from upstream

                let pdat = preds.data();
                let tdat = targs.data();

                // dL/dp_i = -( y/p - (1-y)/(1-p) ) / N
                if preds.requires_grad {
                    let mut slot = preds.grad.borrow_mut();
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
                    let mut slot = targs.grad.borrow_mut();
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
