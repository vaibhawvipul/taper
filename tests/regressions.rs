//! Regression tests: each case here failed before the corresponding fix.

use taper::nn::{Conv2d, Dropout, Linear, Module, Sequential};
use taper::optim::Adam;
use taper::quantization::{QuantizationConfig, QuantizationType};
use taper::train::Trainer;
use taper::{Tape, Tensor};

/// Node ids are indices starting at 0, so 0 could not double as "this tensor is
/// not an op output". A graph whose output was the first recorded op got no
/// gradients at all, silently.
#[test]
fn first_op_after_reset_still_backpropagates() {
    Tape::reset();
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = a.matmul(&b);
    c.backward();

    let grad = a.grad_ref().expect("single-op graph produced no gradient");
    assert_eq!(grad.len(), 4);
}

/// `max(Some(d))` advanced its output stride only for dimensions below `d`, so
/// for any non-last `d` on a rank-3+ tensor it collapsed distinct outputs onto
/// the same slot and left the rest at -inf.
#[test]
fn max_over_a_middle_dimension_is_correct() {
    let x = Tensor::new((0..24).map(|i| i as f32).collect(), &[2, 3, 4]);

    let (values, indices) = x.max(Some(1));
    assert_eq!(values.shape(), &[2, 1, 4]);
    assert_eq!(
        values.data().as_slice(),
        &[8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0]
    );
    // The maximum along dim 1 always sits at the last index.
    assert!(indices.data().iter().all(|&i| i == 2.0));

    // And over the first dimension.
    let (values, _) = x.max(Some(0));
    assert_eq!(values.shape(), &[1, 3, 4]);
    assert_eq!(
        values.data().as_slice(),
        &[
            12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0
        ]
    );
}

#[test]
fn max_over_a_middle_dimension_routes_gradients_to_the_argmax() {
    Tape::reset();
    let x = Tensor::new((0..24).map(|i| i as f32).collect(), &[2, 3, 4]).requires_grad();
    let (values, _) = x.max(Some(1));
    values.backward();

    let grad = x.grad_ref().unwrap();
    for b in 0..2 {
        for h in 0..3 {
            for w in 0..4 {
                let expected = if h == 2 { 1.0 } else { 0.0 };
                assert_eq!(
                    grad[b * 12 + h * 4 + w],
                    expected,
                    "gradient at [{b},{h},{w}]"
                );
            }
        }
    }
}

/// `cat` and the slice helpers built their outputs with a bare `Tensor::new`,
/// leaving grouped convolution with no path back to its weights.
#[test]
fn grouped_conv_reaches_its_weights() {
    Tape::reset();
    let conv = Conv2d::new(4, 4, (3, 3), None, Some((1, 1)), None, Some(2), true);
    let input = Tensor::new(vec![0.5; 4 * 6 * 6], &[1, 4, 6, 6]).requires_grad();

    let out = conv.forward(&input);
    assert_eq!(out.shape(), &[1, 4, 6, 6]);

    let loss = out.sum(None, false);
    loss.backward();

    let wgrad = conv
        .weight
        .grad_ref()
        .expect("grouped conv produced no weight gradient");
    assert!(
        wgrad.iter().any(|g| g.abs() > 1e-6),
        "weight gradient is all zeros"
    );
    assert!(
        input
            .grad_ref()
            .is_some_and(|g| g.iter().any(|v| *v != 0.0)),
        "grouped conv produced no input gradient"
    );
}

#[test]
fn cat_splits_gradients_back_to_each_input() {
    Tape::reset();
    let a = Tensor::new(vec![1.0, 2.0], &[1, 2, 1, 1]).requires_grad();
    let b = Tensor::new(vec![3.0], &[1, 1, 1, 1]).requires_grad();

    let joined = Tensor::cat(&[a.clone(), b.clone()], 1);
    assert_eq!(joined.shape(), &[1, 3, 1, 1]);
    assert_eq!(joined.data().as_slice(), &[1.0, 2.0, 3.0]);

    joined.backward();
    assert_eq!(a.grad_ref().unwrap().as_slice(), &[1.0, 1.0]);
    assert_eq!(b.grad_ref().unwrap().as_slice(), &[1.0]);
}

/// `Module::forward` takes `&self`, so without `set_training` a Dropout buried
/// in a Sequential could never be switched off for evaluation.
#[test]
fn dropout_can_be_disabled_through_sequential() {
    let mut model = Sequential::new(vec![Box::new(Dropout::new(0.5))]);
    let input = Tensor::new(vec![1.0; 512], &[512]);

    model.set_training(false);
    let eval_out = model.forward(&input);
    assert!(
        eval_out.data().iter().all(|&v| v == 1.0),
        "dropout still sampled a mask in eval mode"
    );

    model.set_training(true);
    let train_out = model.forward(&input);
    assert!(
        train_out.data().contains(&0.0),
        "dropout dropped nothing in training mode"
    );
}

/// Evaluation used to record a backward closure per op per batch, each holding
/// its inputs alive, with nothing clearing the tape until the next epoch.
#[test]
fn inference_under_no_grad_records_nothing() {
    Tape::reset();
    let model = Sequential::new(vec![Box::new(Linear::new(8, 4, true))]);
    let input = Tensor::new(vec![0.25; 16], &[2, 8]);

    {
        let _guard = taper::tape::no_grad();
        for _ in 0..10 {
            let _ = model.forward(&input);
        }
        assert_eq!(Tape::len(), 0, "no_grad still grew the tape");
    }

    let _ = model.forward(&input);
    assert!(Tape::len() > 0, "recording did not resume after the guard");
}

fn round_trip(config: QuantizationConfig, values: Vec<f32>) -> Vec<f32> {
    let t = Tensor::new(values, &[4]);
    t.quantize(&config).dequantize().data().clone()
}

/// The int8 zero point was derived as `-min/scale`, which is only right for an
/// unsigned grid; on `[-128, 127]` it clipped away half the dynamic range.
#[test]
fn int8_quantization_covers_the_full_range() {
    let values = vec![-4.0, -1.0, 1.0, 4.0];
    let out = round_trip(QuantizationConfig::int8(true), values.clone());

    // 8 units of range over 255 codes: well under 0.05 per step.
    for (o, v) in out.iter().zip(values.iter()) {
        assert!(
            (o - v).abs() < 0.05,
            "int8 round trip lost {v} -> {o} (expected < 0.05 error)"
        );
    }
}

/// Int4, BFloat16 and NF4 all used to `unimplemented!()` — a panic reachable
/// from the public `Tensor::quantize`.
#[test]
fn every_quantization_type_round_trips() {
    let values = vec![-1.0, -0.25, 0.5, 2.0];

    for (quant_type, tolerance) in [
        (QuantizationType::Int8, 0.02),
        (QuantizationType::Int4, 0.25),
        (QuantizationType::Float16, 0.001),
        (QuantizationType::BFloat16, 0.02),
        (QuantizationType::NF4, 0.25),
    ] {
        let config = QuantizationConfig::new(true, quant_type);
        let out = round_trip(config, values.clone());

        assert_eq!(out.len(), values.len(), "{quant_type:?} changed length");
        for (o, v) in out.iter().zip(values.iter()) {
            assert!(
                (o - v).abs() <= tolerance,
                "{quant_type:?}: {v} round-tripped to {o}, error exceeds {tolerance}"
            );
        }
    }
}

#[test]
fn quantizing_an_odd_length_tensor_preserves_every_element() {
    // 4-bit types pack two values per byte; an odd count must not lose the tail.
    let values: Vec<f32> = (0..7).map(|i| i as f32 * 0.3 - 1.0).collect();
    for quant_type in [QuantizationType::Int4, QuantizationType::NF4] {
        let t = Tensor::new(values.clone(), &[7]);
        let out = t
            .quantize(&QuantizationConfig::new(true, quant_type))
            .dequantize();
        assert_eq!(out.shape(), &[7], "{quant_type:?} lost the shape");
        assert_eq!(out.data().len(), 7, "{quant_type:?} lost an element");
    }
}

#[test]
fn quantizing_a_constant_tensor_stays_finite() {
    for quant_type in [
        QuantizationType::Int8,
        QuantizationType::Int4,
        QuantizationType::NF4,
        QuantizationType::Float16,
        QuantizationType::BFloat16,
    ] {
        let t = Tensor::new(vec![0.0; 4], &[4]);
        let out = t
            .quantize(&QuantizationConfig::new(true, quant_type))
            .dequantize();
        assert!(
            out.data().iter().all(|v| v.is_finite()),
            "{quant_type:?} produced a non-finite value for an all-zero tensor"
        );
    }
}

/// `save_checkpoint` shipped without a loader, so checkpoints were write-only.
#[test]
fn checkpoints_round_trip() {
    let path = std::env::temp_dir().join("taper_checkpoint_roundtrip.txt");
    let path = path.to_str().unwrap();

    let model = Sequential::new(vec![Box::new(Linear::new(4, 3, true))]);
    let params = model.parameters();
    let saved: Vec<f32> = params[0].data().clone();
    let optimizer = Adam::new(params, 0.001, None, None, None);
    let trainer = Trainer::new(Box::new(model), optimizer, None);
    trainer.save_checkpoint(path).unwrap();

    // A second model with independently initialized weights.
    let other = Sequential::new(vec![Box::new(Linear::new(4, 3, true))]);
    let other_params = other.parameters();
    let before: Vec<f32> = other_params[0].data().clone();
    assert_ne!(before, saved, "models happened to initialize identically");

    let optimizer = Adam::new(other.parameters(), 0.001, None, None, None);
    let mut other_trainer = Trainer::new(Box::new(other), optimizer, None);
    other_trainer.load_checkpoint(path).unwrap();

    assert_eq!(
        other_trainer.model.parameters()[0].data().clone(),
        saved,
        "loaded weights do not match the saved ones"
    );

    // Shape mismatches are reported, not silently accepted.
    let wrong = Sequential::new(vec![Box::new(Linear::new(8, 3, true))]);
    let optimizer = Adam::new(wrong.parameters(), 0.001, None, None, None);
    let mut wrong_trainer = Trainer::new(Box::new(wrong), optimizer, None);
    assert!(wrong_trainer.load_checkpoint(path).is_err());

    std::fs::remove_file(path).ok();
}

/// Element-wise ops only checked flat length, so `[2,3] op [3,2]` produced a
/// result mislabelled with the left operand's shape.
#[test]
#[should_panic(expected = "do not broadcast")]
fn elementwise_ops_reject_mismatched_shapes() {
    let a = Tensor::new(vec![1.0; 6], &[2, 3]);
    let b = Tensor::new(vec![1.0; 6], &[3, 2]);
    let _ = &a + &b;
}

#[test]
#[should_panic(expected = "do not fill shape")]
fn tensor_new_rejects_a_shape_that_does_not_match_the_data() {
    let _ = Tensor::new(vec![1.0, 2.0, 3.0], &[2, 2]);
}

/// Naive reference convolution, written for clarity rather than speed.
fn conv2d_reference(
    x: &[f32],
    (n, c_in, h_in, w_in): (usize, usize, usize, usize),
    w: &[f32],
    (c_out, k_h, k_w): (usize, usize, usize),
    (stride_h, stride_w): (usize, usize),
    (pad_h, pad_w): (usize, usize),
    (dil_h, dil_w): (usize, usize),
) -> (Vec<f32>, usize, usize) {
    let h_out = (h_in + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1;
    let mut out = vec![0.0; n * c_out * h_out * w_out];

    for b in 0..n {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0.0;
                    for ic in 0..c_in {
                        for kh in 0..k_h {
                            let ih = (oh * stride_h + kh * dil_h) as isize - pad_h as isize;
                            if ih < 0 || ih >= h_in as isize {
                                continue;
                            }
                            for kw in 0..k_w {
                                let iw = (ow * stride_w + kw * dil_w) as isize - pad_w as isize;
                                if iw < 0 || iw >= w_in as isize {
                                    continue;
                                }
                                acc += x[b * c_in * h_in * w_in
                                    + ic * h_in * w_in
                                    + ih as usize * w_in
                                    + iw as usize]
                                    * w[oc * c_in * k_h * k_w + ic * k_h * k_w + kh * k_w + kw];
                            }
                        }
                    }
                    out[b * c_out * h_out * w_out + oh * w_out + ow + oc * h_out * w_out] = acc;
                }
            }
        }
    }
    (out, h_out, w_out)
}

/// conv2d flattened the weight straight to `[K, C_out]`, which transposes the
/// filter axis against the patch axis; and the 1x1 im2col path was a memcpy
/// that only matched the column layout when the spatial size was 1.
/// `(input NCHW, (C_out, K_h, K_w), stride, padding, dilation)`
type ConvCase = (
    (usize, usize, usize, usize),
    (usize, usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize),
);

#[test]
fn conv2d_matches_a_naive_reference() {
    let cases: &[ConvCase] = &[
        ((1, 2, 5, 5), (3, 3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 3, 6, 6), (4, 3, 3), (2, 2), (1, 1), (1, 1)),
        ((1, 3, 7, 7), (2, 1, 1), (1, 1), (0, 0), (1, 1)), // the broken 1x1 path
        ((1, 2, 8, 8), (3, 3, 3), (1, 1), (2, 2), (2, 2)), // dilated + padded
        ((1, 1, 5, 5), (2, 2, 4), (1, 2), (1, 0), (1, 1)), // non-square kernel
    ];

    for &((n, c_in, h_in, w_in), (c_out, k_h, k_w), stride, padding, dilation) in cases {
        let x: Vec<f32> = (0..n * c_in * h_in * w_in)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.1)
            .collect();
        let w: Vec<f32> = (0..c_out * c_in * k_h * k_w)
            .map(|i| ((i * 17 % 13) as f32 - 6.0) * 0.05)
            .collect();

        let (expected, h_out, w_out) = conv2d_reference(
            &x,
            (n, c_in, h_in, w_in),
            &w,
            (c_out, k_h, k_w),
            stride,
            padding,
            dilation,
        );

        let xt = Tensor::new(x, &[n, c_in, h_in, w_in]);
        let wt = Tensor::new(w, &[c_out, c_in, k_h, k_w]);
        let got = xt.conv2d(&wt, None, stride, padding, dilation);

        assert_eq!(got.shape(), &[n, c_out, h_out, w_out]);
        for (i, (g, e)) in got.data().iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "conv2d mismatch at {i} for shape {:?} k{k_h}x{k_w} s{stride:?} p{padding:?} d{dilation:?}: {g} vs {e}",
                [n, c_in, h_in, w_in],
            );
        }
    }
}

/// `im2col` and `transpose_4d` both rebuilt their outputs with a bare
/// `Tensor::new`, so conv2d had no backward edge at all: only the conv *bias*
/// ever learned, and the weights stayed at their initialization forever.
#[test]
fn conv2d_gradients_match_finite_differences() {
    let shape = [1usize, 2, 4, 4];
    let wshape = [2usize, 2, 3, 3];
    let x: Vec<f32> = (0..32)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.1)
        .collect();
    let w: Vec<f32> = (0..36).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1).collect();

    // Scalar objective: sum of a weighted convolution output.
    let coefficients: Vec<f32> = (0..32).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
    let objective = |w: &[f32]| -> f32 {
        let xt = Tensor::new(x.clone(), &shape);
        let wt = Tensor::new(w.to_vec(), &wshape);
        let out = xt.conv2d(&wt, None, (1, 1), (1, 1), (1, 1));
        out.data()
            .iter()
            .zip(coefficients.iter())
            .map(|(o, c)| o * c)
            .sum()
    };

    Tape::reset();
    let xt = Tensor::new(x.clone(), &shape).requires_grad();
    let wt = Tensor::new(w.clone(), &wshape).requires_grad();
    let out = xt.conv2d(&wt, None, (1, 1), (1, 1), (1, 1));
    let weighted = &out * &Tensor::new(coefficients.clone(), out.shape());
    weighted.sum(None, false).backward();

    let analytic = wt.grad_ref().expect("no weight gradient");
    let eps = 1e-2;
    for i in 0..w.len() {
        let mut plus = w.clone();
        plus[i] += eps;
        let mut minus = w.clone();
        minus[i] -= eps;
        let numeric = (objective(&plus) - objective(&minus)) / (2.0 * eps);

        assert!(
            (analytic[i] - numeric).abs() < 1e-2,
            "weight gradient {i}: analytic {} vs numeric {numeric}",
            analytic[i]
        );
    }

    assert!(
        xt.grad_ref().is_some_and(|g| g.iter().any(|v| *v != 0.0)),
        "conv2d produced no input gradient"
    );
}

/// backward() used to replay every node recorded up to the output's id, so two
/// independent graphs sharing a tape would each run the other's backward pass.
#[test]
fn independent_graphs_on_one_tape_do_not_contaminate_each_other() {
    Tape::reset();
    let a = Tensor::new(vec![2.0], &[1]).requires_grad();
    let b = Tensor::new(vec![3.0], &[1]).requires_grad();

    // Two graphs, recorded interleaved onto the same tape.
    let ga = &a * &a; // depends only on a
    let gb = &b * &b; // depends only on b

    ga.backward();
    assert_eq!(a.grad_ref().unwrap().as_slice(), &[4.0], "d(a²)/da = 2a");
    assert!(
        b.grad_ref().is_none(),
        "differentiating a's graph also ran b's backward"
    );

    gb.backward();
    assert_eq!(b.grad_ref().unwrap().as_slice(), &[6.0], "d(b²)/db = 2b");
}

/// Only the ancestors of the output should run. cross_entropy_loss builds a
/// log_softmax chain it then bypasses, so those nodes are dead weight.
#[test]
fn backward_skips_operations_the_output_does_not_depend_on() {
    Tape::reset();
    let x = Tensor::new(vec![1.0, 2.0], &[2]).requires_grad();

    let used = &x + &x;
    // A side branch that nothing downstream consumes.
    let _unused = (&x * &x).exp().log();
    let recorded = Tape::len();

    used.backward();

    assert!(
        recorded > 2,
        "expected the unused branch to be on the tape, got {recorded} nodes"
    );
    // d(2x)/dx = 2 exactly; had the side branch run, it would have added to x.
    assert_eq!(x.grad_ref().unwrap().as_slice(), &[2.0, 2.0]);
}
