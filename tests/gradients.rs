//! Finite-difference verification of every hand-written backward pass.
//!
//! A backward that is quietly wrong is this crate's most common defect, and it
//! does not surface as a failure — the loss still falls, because other layers
//! compensate. So each differentiable operation gets checked here rather than
//! only having its forward behaviour asserted.
//!
//! `FakeQuantize` is deliberately absent: its straight-through estimator is not
//! the true derivative of rounding (which is zero almost everywhere), so
//! disagreeing with finite differences is the intended behaviour.

use taper::Tensor;
use taper::gradcheck::{self, GradCheck};
use taper::nn::{
    AdaptiveAvgPool2d, AvgPool2d, Conv2d, Flatten, Linear, MaxPool2d, Module, Sequential,
};
use taper::norm::{BatchNorm2d, LayerNorm};
use taper::tensor::DType;

/// Deterministic, non-degenerate values away from kinks like ReLU's at 0.
fn sample(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = ((i * 37 % 23) as f32 - 11.0) * 0.11;
            // Nudge anything near zero so ReLU/abs-style kinks don't make the
            // numeric derivative meaningless.
            if v.abs() < 0.05 { v + 0.37 } else { v }
        })
        .collect()
}

fn t(shape: &[usize]) -> Tensor {
    Tensor::new(sample(shape.iter().product()), shape).requires_grad()
}

/// Strictly positive, for log / division / sqrt.
fn positive(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let values = (0..n).map(|i| 0.4 + (i % 7) as f32 * 0.23).collect();
    Tensor::new(values, shape).requires_grad()
}

macro_rules! assert_grad {
    ($inputs:expr, $f:expr) => {
        if let Err(e) = gradcheck::check($inputs, $f) {
            panic!("{e}");
        }
    };
    ($inputs:expr, $f:expr, $cfg:expr) => {
        if let Err(e) = $cfg.check($inputs, $f) {
            panic!("{e}");
        }
    };
}

// --- elementwise operators ---

#[test]
fn add_sub_mul_div() {
    let a = t(&[3, 4]);
    let b = positive(&[3, 4]);

    assert_grad!(&[a.clone(), b.clone()], |x| &x[0] + &x[1]);
    assert_grad!(&[a.clone(), b.clone()], |x| &x[0] - &x[1]);
    assert_grad!(&[a.clone(), b.clone()], |x| &x[0] * &x[1]);
    assert_grad!(&[a, b], |x| &x[0] / &x[1]);
}

#[test]
fn elementwise_ops_under_broadcasting() {
    let m = t(&[3, 4]);
    let row = t(&[4]);
    let col = t(&[3, 1]);

    assert_grad!(&[m.clone(), row.clone()], |x| &x[0] + &x[1]);
    assert_grad!(&[m.clone(), col.clone()], |x| &x[0] * &x[1]);
    assert_grad!(&[col.clone(), row.clone()], |x| &x[0] * &x[1]);
    assert_grad!(&[m, col], |x| &x[0] - &x[1]);
}

#[test]
fn unary_math() {
    let x = t(&[2, 5]);
    let p = positive(&[2, 5]);

    assert_grad!(std::slice::from_ref(&x), |v| v[0].relu());
    assert_grad!(std::slice::from_ref(&x), |v| v[0].sigmoid());
    assert_grad!(std::slice::from_ref(&x), |v| v[0].exp());
    assert_grad!(std::slice::from_ref(&p), |v| v[0].log());
    assert_grad!(std::slice::from_ref(&p), |v| v[0].pow(3.0));
    assert_grad!(std::slice::from_ref(&p), |v| v[0].sqrt());
    assert_grad!(std::slice::from_ref(&x), |v| v[0].exp().sigmoid());
    assert_grad!(std::slice::from_ref(&p), |v| v[0].log().exp());
}

// --- reductions ---

#[test]
fn reductions() {
    let x = t(&[3, 4]);

    assert_grad!(std::slice::from_ref(&x), |v| v[0].mean());
    assert_grad!(std::slice::from_ref(&x), |v| v[0].sum(None, false));
    assert_grad!(std::slice::from_ref(&x), |v| v[0].sum(Some(1), true));
    assert_grad!(std::slice::from_ref(&x), |v| v[0].sum(Some(1), false));
    assert_grad!(std::slice::from_ref(&x), |v| v[0].sum(Some(0), true));
    assert_grad!(std::slice::from_ref(&x), |v| v[0].sum(Some(0), false));
}

#[test]
fn reductions_over_rank_three() {
    let x = t(&[2, 3, 4]);
    for dim in 0..3 {
        assert_grad!(std::slice::from_ref(&x), move |v| v[0].sum(Some(dim), true));
        assert_grad!(std::slice::from_ref(&x), move |v| v[0]
            .sum(Some(dim), false));
    }
}

#[test]
fn max_routes_gradient_to_the_argmax() {
    let x = t(&[2, 3, 4]);

    assert_grad!(std::slice::from_ref(&x), |v| v[0].max(None).0);
    for dim in 0..3 {
        assert_grad!(std::slice::from_ref(&x), move |v| v[0].max(Some(dim)).0);
    }
}

// --- shape and layout ---

#[test]
fn shape_ops() {
    let x = t(&[3, 4]);
    let cube = t(&[2, 3, 4]);

    assert_grad!(std::slice::from_ref(&x), |v| v[0].transpose());
    assert_grad!(std::slice::from_ref(&x), |v| v[0].reshape(&[4, 3]));
    assert_grad!(std::slice::from_ref(&x), |v| v[0].contiguous());
    assert_grad!(std::slice::from_ref(&x), |v| v[0].transpose().contiguous());
    assert_grad!(std::slice::from_ref(&cube), |v| v[0].flatten(1));
    assert_grad!(std::slice::from_ref(&cube), |v| v[0].unsqueeze(0));
    assert_grad!(std::slice::from_ref(&x), |v| v[0].view(&[2, 6]));
    assert_grad!(std::slice::from_ref(&cube), |v| v[0].narrow(1, 1, 2));
}

#[test]
fn expand_sums_over_stretched_axes() {
    assert_grad!(&[t(&[1, 4])], |v| v[0].expand(&[3, 4]));
    assert_grad!(&[t(&[3, 1])], |v| v[0].expand(&[3, 5]));
    assert_grad!(&[t(&[4])], |v| v[0].expand(&[2, 4]));
}

#[test]
fn cat_splits_gradient_across_inputs() {
    let a = t(&[1, 2, 2, 2]);
    let b = t(&[1, 3, 2, 2]);
    assert_grad!(&[a.clone(), b.clone()], |v| Tensor::cat(
        &[v[0].clone(), v[1].clone()],
        1
    ));

    let p = t(&[2, 3]);
    let q = t(&[2, 3]);
    assert_grad!(&[p, q], |v| Tensor::cat(&[v[0].clone(), v[1].clone()], 0));
}

#[test]
fn dtype_cast_passes_gradient_through() {
    // The cast is element-wise with unit derivative, but bf16 quantizes the
    // forward value, so this needs a looser tolerance than the default.
    let cfg = GradCheck {
        eps: 1e-1,
        atol: 0.2,
        rtol: 0.2,
    };
    assert_grad!(&[t(&[3, 3])], |v| v[0].to_dtype(DType::BF16), cfg);
    assert_grad!(&[t(&[3, 3])], |v| v[0].to_dtype(DType::F32));
}

// --- broadcasting helpers ---

#[test]
fn row_broadcast_helpers() {
    let m = t(&[3, 4]);
    let bias = t(&[4]);
    let rows = positive(&[3, 1]);

    assert_grad!(&[m.clone(), bias], |v| v[0].add_broadcast(&v[1]));
    assert_grad!(&[m.clone(), rows.clone()], |v| v[0]
        .sub_broadcast_rows(&v[1]));
    assert_grad!(&[m, rows], |v| v[0].div_broadcast_rows(&v[1]));
}

// --- matmul, across every operand layout ---

#[test]
fn matmul() {
    let a = t(&[3, 4]);
    let b = t(&[4, 2]);

    assert_grad!(&[a.clone(), b.clone()], |v| v[0].matmul(&v[1]));
    // Transposed views feed GEMM directly under its trans flag, which is a
    // different code path in both the forward and the two backward products.
    assert_grad!(&[a.clone(), b.clone()], |v| v[0]
        .transpose()
        .transpose()
        .matmul(&v[1]));
    assert_grad!(&[a.clone(), b.clone()], |v| v[0]
        .matmul(&v[1].transpose().transpose()));

    let c = t(&[4, 3]);
    assert_grad!(&[c, b], |v| v[0].transpose().matmul(&v[1]));
}

// --- losses ---

#[test]
fn losses() {
    let logits = t(&[4, 3]);
    let targets = Tensor::new(vec![0.0, 2.0, 1.0, 2.0], &[4]);
    let one_hot = Tensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[4, 3],
    );

    assert_grad!(std::slice::from_ref(&logits), move |v| {
        taper::loss::cross_entropy_loss(&v[0], &targets)
    });
    assert_grad!(std::slice::from_ref(&logits), move |v| {
        taper::loss::cross_entropy_loss_onehot(&v[0], &one_hot)
    });

    let preds = Tensor::new(vec![0.2, 0.7, 0.45, 0.9], &[4]).requires_grad();
    let labels = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], &[4]);
    assert_grad!(
        std::slice::from_ref(&preds),
        move |v| taper::loss::bce_loss(&v[0], &labels)
    );

    let y = t(&[3, 2]);
    let target = Tensor::new(sample(6), &[3, 2]);
    assert_grad!(std::slice::from_ref(&y), move |v| taper::loss::mse_loss(
        &v[0], &target
    ));
}

#[test]
fn softmax_and_log_softmax() {
    let x = t(&[3, 4]);
    assert_grad!(std::slice::from_ref(&x), |v| taper::loss::softmax(
        &v[0], -1
    ));
    assert_grad!(std::slice::from_ref(&x), |v| taper::loss::log_softmax(
        &v[0], -1
    ));
}

// --- convolution and pooling ---

#[test]
fn conv2d_across_configurations() {
    let x = t(&[2, 2, 5, 5]);
    let w = t(&[3, 2, 3, 3]);
    let bias = t(&[3]);

    assert_grad!(&[x.clone(), w.clone()], |v| v[0].conv2d(
        &v[1],
        None,
        (1, 1),
        (1, 1),
        (1, 1)
    ));
    assert_grad!(&[x.clone(), w.clone(), bias.clone()], |v| v[0].conv2d(
        &v[1],
        Some(&v[2]),
        (1, 1),
        (1, 1),
        (1, 1)
    ));
    // Strided, unpadded, and dilated all take different index paths.
    assert_grad!(&[x.clone(), w.clone()], |v| v[0].conv2d(
        &v[1],
        None,
        (2, 2),
        (0, 0),
        (1, 1)
    ));
    assert_grad!(&[x.clone(), w.clone()], |v| v[0].conv2d(
        &v[1],
        None,
        (1, 1),
        (2, 2),
        (2, 2)
    ));
    // 1x1, which used to have its own (incorrect) im2col path.
    let w1 = t(&[2, 2, 1, 1]);
    assert_grad!(&[x, w1], |v| v[0].conv2d(
        &v[1],
        None,
        (1, 1),
        (0, 0),
        (1, 1)
    ));
}

/// Grouped convolution slices its input and weights per group and concatenates
/// the results, so the gradient has to reach *every* group — not just the first.
#[test]
fn grouped_conv2d() {
    for groups in [1usize, 2, 4] {
        let conv = Conv2d::new(4, 4, (3, 3), None, Some((1, 1)), None, Some(groups), false);
        let x = t(&[1, 4, 5, 5]);
        // The layer's own weight, so the gradient lands where it is read from.
        let w = conv.weight.clone();

        if let Err(e) = gradcheck::check(&[x, w], |v| conv.forward(&v[0])) {
            panic!("groups={groups}: {e}");
        }
    }
}

/// ReLU is not differentiable at 0, so a central difference across a
/// pre-activation nearer than `eps` to the kink averages two different
/// derivatives and matches neither.
///
/// A bias shifts the pre-activations without touching `dz/dx` or `dz/dw`, so it
/// moves the whole output clear of the kink while leaving gradient magnitudes
/// alone — unlike scaling the input, which inflates them and trades a kink
/// problem for an f32 precision one. Both sides of the kink get their own case.
#[test]
fn conv2d_relu_on_each_side_of_the_kink() {
    let x = t(&[2, 2, 5, 5]);
    let w = t(&[3, 2, 3, 3]);
    let eps = GradCheck::default().eps;
    let margin = 10.0 * eps;

    let pre = x.conv2d(&w, None, (1, 1), (1, 1), (1, 1)).to_vec();
    let lowest = pre.iter().fold(f32::INFINITY, |m, v| m.min(*v));
    let highest = pre.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));

    // Fully inside ReLU's active region: every output passes through.
    let up = Tensor::new(vec![margin - lowest; 3], &[3]).requires_grad();
    assert_grad!(&[x.clone(), w.clone(), up], |v| v[0].conv2d_relu(
        &v[1],
        Some(&v[2]),
        (1, 1),
        (1, 1),
        (1, 1)
    ));

    // Fully inside the dead region: every gradient is exactly zero, which is a
    // distinct thing to get wrong from getting the active branch right.
    let down = Tensor::new(vec![-margin - highest; 3], &[3]).requires_grad();
    let out = x.conv2d_relu(&w, Some(&down), (1, 1), (1, 1), (1, 1));
    assert!(
        out.to_vec().iter().all(|v| *v == 0.0),
        "expected the dead region, got non-zero outputs"
    );
    assert_grad!(&[x, w, down], |v| v[0].conv2d_relu(
        &v[1],
        Some(&v[2]),
        (1, 1),
        (1, 1),
        (1, 1)
    ));
}

#[test]
fn pooling() {
    let x = t(&[2, 2, 4, 4]);

    assert_grad!(std::slice::from_ref(&x), |v| v[0].max_pool2d(
        (2, 2),
        Some((2, 2)),
        (0, 0)
    ));
    assert_grad!(std::slice::from_ref(&x), |v| v[0].avg_pool2d(
        (2, 2),
        Some((2, 2)),
        (0, 0)
    ));
    // Overlapping windows make one input contribute to several outputs.
    assert_grad!(std::slice::from_ref(&x), |v| v[0].avg_pool2d(
        (2, 2),
        Some((1, 1)),
        (0, 0)
    ));
    assert_grad!(std::slice::from_ref(&x), |v| v[0].max_pool2d(
        (2, 2),
        Some((1, 1)),
        (0, 0)
    ));
    assert_grad!(std::slice::from_ref(&x), |v| MaxPool2d::new_2x2()
        .forward(&v[0]));
    assert_grad!(std::slice::from_ref(&x), |v| AvgPool2d::new(
        (2, 2),
        None,
        None
    )
    .forward(&v[0]));
    assert_grad!(std::slice::from_ref(&x), |v| AdaptiveAvgPool2d::new((2, 2))
        .forward(&v[0]));
}

// --- layers ---

#[test]
fn linear_layer() {
    let layer = Linear::new(4, 3, true);
    let x = t(&[2, 4]);
    let w = layer.weight.clone();
    let b = layer.bias.clone().unwrap();

    assert_grad!(&[x, w, b], |v| {
        v[0].matmul(&v[1].transpose()).add_broadcast(&v[2])
    });
}

#[test]
fn layer_norm() {
    let ln = LayerNorm::new(&[4], true, None);
    let x = t(&[3, 4]);
    let gamma = ln.gamma.clone();
    let beta = ln.beta.clone().unwrap();
    gamma.data_mut().copy_from_slice(&[1.3, -0.7, 0.5, 2.0]);
    beta.data_mut().copy_from_slice(&[0.1, 0.2, -0.3, 0.4]);

    assert_grad!(&[x, gamma, beta], |v| ln.forward(&v[0]));
}

#[test]
fn batch_norm_training_and_eval() {
    let x = t(&[3, 2, 2, 2]);

    let bn = BatchNorm2d::new(2);
    let gamma = bn.gamma.clone();
    let beta = bn.beta.clone();
    assert_grad!(&[x.clone(), gamma, beta], |v| bn.forward(&v[0]));

    let mut eval_bn = BatchNorm2d::new(2);
    eval_bn.forward(&Tensor::new(sample(24), &[3, 2, 2, 2]));
    eval_bn.set_training(false);
    let g2 = eval_bn.gamma.clone();
    let b2 = eval_bn.beta.clone();
    assert_grad!(&[x, g2, b2], |v| eval_bn.forward(&v[0]));
}

/// A composed network, to check that the chain agrees and not only each op
/// alone.
///
/// Deliberately built from smooth layers. ReLU and max-pooling are piecewise
/// linear, and a central difference that straddles a kink — a pre-activation
/// near zero, or two values close enough that the argmax flips — averages two
/// different derivatives and matches neither. Both are verified individually
/// above with inputs chosen to clear their kinks.
#[test]
fn a_whole_smooth_cnn() {
    let model = Sequential::new(vec![
        Box::new(Conv2d::new(
            1,
            3,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            None,
            None,
            true,
        )),
        Box::new(BatchNorm2d::new(3)),
        Box::new(AvgPool2d::new((2, 2), Some((2, 2)), None)),
        Box::new(Flatten::new(Some(1))),
        Box::new(Linear::new(3 * 2 * 2, 2, true)),
        Box::new(taper::activation::Sigmoid),
    ]);

    let x = t(&[2, 1, 4, 4]);
    // Slightly looser: f32 error accumulates through six layers.
    let cfg = GradCheck {
        eps: 1e-2,
        atol: 5e-2,
        rtol: 5e-2,
    };
    assert_grad!(std::slice::from_ref(&x), |v| model.forward(&v[0]), cfg);
}

#[test]
fn an_mlp_through_cross_entropy() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(5, 8, true)),
        Box::new(taper::activation::ReLU),
        Box::new(Linear::new(8, 3, true)),
    ]);
    let x = t(&[4, 5]);
    let targets = Tensor::new(vec![0.0, 2.0, 1.0, 0.0], &[4]);

    assert_grad!(std::slice::from_ref(&x), move |v| {
        taper::loss::cross_entropy_loss(&model.forward(&v[0]), &targets)
    });
}

/// The check must actually fail on a wrong gradient, or it proves nothing.
#[test]
fn gradcheck_catches_a_broken_backward() {
    use taper::ops;
    use taper::tape::Tape;

    // A "double" op whose backward wrongly reports 1 instead of 2.
    fn broken_double(x: &Tensor) -> Tensor {
        let mut out = Tensor::new(x.to_vec().iter().map(|v| v * 2.0).collect(), x.shape());
        out.requires_grad = true;
        let input = x.clone();
        let o = out.clone();
        Tape::push_unary_op(x, &out, move || {
            if let Some(g) = o.grad.read().unwrap().as_ref() {
                ops::accumulate_grad(&input, g); // should be scaled by 2
            }
        });
        out
    }

    let err = gradcheck::check(&[t(&[4])], |v| broken_double(&v[0]))
        .expect_err("gradcheck passed a knowingly wrong backward");
    assert_eq!(err.mismatches.len(), 4, "{err}");
}

/// An op that silently drops its backward edge is the real historical bug, so
/// the check has to flag a missing gradient rather than skipping the input.
#[test]
fn gradcheck_catches_a_severed_graph() {
    let severed = |x: &Tensor| Tensor::new(x.to_vec(), x.shape());

    let err = gradcheck::check(&[t(&[3])], |v| severed(&v[0]))
        .expect_err("gradcheck passed an op with no backward edge at all");
    assert_eq!(err.mismatches.len(), 3, "{err}");
}
