//! NumPy-rule broadcasting, implemented as stride-0 views.

use taper::tensor::broadcast_shape;
use taper::{Tape, Tensor};

#[test]
fn shape_rules_match_numpy() {
    let cases: &[(&[usize], &[usize], Option<&[usize]>)] = &[
        (&[3, 4], &[3, 4], Some(&[3, 4])),
        (&[3, 4], &[4], Some(&[3, 4])), // right-aligned, rank lifted
        (&[3, 1], &[1, 4], Some(&[3, 4])), // both stretch
        (&[5, 1, 4], &[3, 1], Some(&[5, 3, 4])), // mixed rank
        (&[3, 4], &[1], Some(&[3, 4])), // scalar-ish
        (&[3, 4], &[3, 2], None),       // incompatible
        (&[3, 4], &[2, 3, 5], None),
    ];

    for (a, b, want) in cases {
        let got = broadcast_shape(a, b);
        match want {
            Some(w) => assert_eq!(got.as_deref(), Some(*w), "{a:?} vs {b:?}"),
            None => assert!(got.is_none(), "{a:?} vs {b:?} should not broadcast"),
        }
        // Broadcasting is symmetric.
        assert_eq!(
            broadcast_shape(a, b).is_some(),
            broadcast_shape(b, a).is_some()
        );
    }
}

#[test]
fn expand_is_a_stride_zero_view() {
    let row = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]);
    let wide = row.expand(&[4, 3]);

    assert_eq!(wide.shape(), &[4, 3]);
    // The stretched axis reads the same element every step.
    assert_eq!(wide.stride(), &[0, 1]);
    assert_eq!(
        wide.to_vec(),
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn expand_lifts_rank_from_the_left() {
    let v = Tensor::new(vec![1.0, 2.0], &[2]);
    let lifted = v.expand(&[3, 2]);
    assert_eq!(lifted.stride(), &[0, 1]);
    assert_eq!(lifted.to_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn operators_broadcast_a_row_against_a_matrix() {
    let m = Tensor::new((0..6).map(|i| i as f32).collect(), &[2, 3]);
    let row = Tensor::new(vec![10.0, 20.0, 30.0], &[3]);

    let sum = &m + &row;
    assert_eq!(sum.shape(), &[2, 3]);
    assert_eq!(sum.to_vec(), vec![10.0, 21.0, 32.0, 13.0, 24.0, 35.0]);
}

#[test]
fn operators_broadcast_a_column_against_a_matrix() {
    let m = Tensor::new((0..6).map(|i| i as f32).collect(), &[2, 3]);
    let col = Tensor::new(vec![10.0, 20.0], &[2, 1]);

    let sum = &m + &col;
    assert_eq!(sum.to_vec(), vec![10.0, 11.0, 12.0, 23.0, 24.0, 25.0]);
}

#[test]
fn operators_broadcast_both_operands_at_once() {
    let col = Tensor::new(vec![1.0, 2.0, 3.0], &[3, 1]);
    let row = Tensor::new(vec![10.0, 20.0], &[1, 2]);

    let outer = &col * &row;
    assert_eq!(outer.shape(), &[3, 2]);
    assert_eq!(outer.to_vec(), vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
}

/// The whole point of stride 0: a value read many times must collect the
/// gradient from every position that read it.
#[test]
fn expand_sums_gradients_over_stretched_axes() {
    Tape::reset();
    let row = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).requires_grad();
    let wide = row.expand(&[4, 3]);
    wide.backward();

    // Each element was read by 4 rows.
    assert_eq!(row.grad_ref().unwrap().as_slice(), &[4.0, 4.0, 4.0]);
}

#[test]
fn broadcast_add_reduces_gradient_to_the_bias_shape() {
    Tape::reset();
    let m = Tensor::new(vec![0.0; 6], &[2, 3]).requires_grad();
    let bias = Tensor::new(vec![0.0, 0.0, 0.0], &[3]).requires_grad();

    let out = &m + &bias;
    out.backward();

    // The matrix sees ones; the bias collects one per row.
    assert_eq!(m.grad_ref().unwrap().as_slice(), &[1.0; 6]);
    assert_eq!(bias.grad_ref().unwrap().as_slice(), &[2.0, 2.0, 2.0]);
}

#[test]
fn broadcast_mul_routes_gradients_through_both_operands() {
    Tape::reset();
    let col = Tensor::new(vec![1.0, 2.0], &[2, 1]).requires_grad();
    let row = Tensor::new(vec![3.0, 4.0, 5.0], &[1, 3]).requires_grad();

    let outer = &col * &row;
    outer.backward();

    // d/dcol = sum over the row; d/drow = sum over the column.
    assert_eq!(col.grad_ref().unwrap().as_slice(), &[12.0, 12.0]);
    assert_eq!(row.grad_ref().unwrap().as_slice(), &[3.0, 3.0, 3.0]);
}

/// The three hand-written helpers still work, now on the general machinery.
#[test]
fn the_original_broadcast_helpers_still_agree() {
    let m = Tensor::new((0..6).map(|i| i as f32).collect(), &[2, 3]);
    let bias = Tensor::new(vec![10.0, 20.0, 30.0], &[3]);
    assert_eq!(m.add_broadcast(&bias).to_vec(), (&m + &bias).to_vec());

    let rows = Tensor::new(vec![1.0, 2.0], &[2, 1]);
    assert_eq!(
        m.sub_broadcast_rows(&rows).to_vec(),
        vec![-1.0, 0.0, 1.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        m.div_broadcast_rows(&rows).to_vec(),
        vec![0.0, 1.0, 2.0, 1.5, 2.0, 2.5]
    );
}

#[test]
#[should_panic(expected = "do not broadcast")]
fn incompatible_shapes_are_rejected() {
    let a = Tensor::new(vec![0.0; 6], &[2, 3]);
    let b = Tensor::new(vec![0.0; 8], &[2, 4]);
    let _ = &a + &b;
}
