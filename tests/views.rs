//! Strided-view semantics: layout metadata, zero-copy shape ops, and the
//! guarantee that a view is never silently read as if it were dense.

use taper::{Tape, Tensor};

fn m23() -> Tensor {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
}

#[test]
fn a_fresh_tensor_is_contiguous() {
    let t = m23();
    assert!(t.is_contiguous());
    assert_eq!(t.stride(), &[3, 1]);
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn transpose_is_a_zero_copy_view() {
    let t = m23().transpose();

    assert_eq!(t.shape(), &[3, 2]);
    // Shape and stride are swapped rather than the data being rearranged.
    assert_eq!(t.stride(), &[1, 3]);
    assert!(!t.is_contiguous());

    // Logical order still reads out correctly.
    assert_eq!(t.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn transposing_twice_returns_the_original_layout() {
    let t = m23().transpose().transpose();
    assert_eq!(t.shape(), &[2, 3]);
    assert!(t.is_contiguous());
    assert_eq!(t.to_vec(), m23().to_vec());
}

/// Reading a view through the raw-slice accessor would return the base
/// tensor's elements in the wrong order — a plausible-looking wrong answer.
#[test]
#[should_panic(expected = "non-contiguous view")]
fn raw_data_access_on_a_view_is_rejected() {
    let t = m23().transpose();
    drop(t.data());
}

#[test]
fn contiguous_materializes_a_view_in_logical_order() {
    let view = m23().transpose();
    let packed = view.contiguous();

    assert!(packed.is_contiguous());
    assert_eq!(packed.shape(), &[3, 2]);
    // Now safe to read as a dense slice.
    assert_eq!(packed.data().as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn contiguous_on_an_already_dense_tensor_is_a_no_op() {
    let t = m23();
    let c = t.contiguous();
    assert_eq!(c.to_vec(), t.to_vec());
    assert!(c.is_contiguous());
}

/// matmul feeds transposed views straight to GEMM under its trans flag. Each
/// combination has to agree with the same product computed from dense copies.
#[test]
fn matmul_agrees_across_every_operand_layout() {
    let a = Tensor::new((0..6).map(|i| i as f32 + 1.0).collect(), &[2, 3]);
    let b = Tensor::new((0..12).map(|i| i as f32 * 0.5 - 2.0).collect(), &[3, 4]);

    let reference = a.matmul(&b).to_vec();

    // Same products, but reached through transposed views of the operands.
    let a_via_view = a.transpose().transpose();
    let b_via_view = b.transpose().transpose();

    assert_eq!(a_via_view.matmul(&b).to_vec(), reference);
    assert_eq!(a.matmul(&b_via_view).to_vec(), reference);
    assert_eq!(a_via_view.matmul(&b_via_view).to_vec(), reference);

    // A genuinely column-major operand: bᵀ is [4,3], so (bᵀ)ᵀ · nothing —
    // instead build aᵀ [3,2] and check aᵀᵀ·b once more against the dense path.
    let a_t = a.transpose(); // [3,2], stride [1,3] — column-major
    let a_t_dense = a_t.contiguous();
    assert_eq!(
        a_t.matmul(&Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]))
            .to_vec(),
        a_t_dense
            .matmul(&Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]))
            .to_vec()
    );
}

#[test]
fn matmul_through_a_transposed_view_still_backpropagates() {
    Tape::reset();
    let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).requires_grad();
    let x = Tensor::new(vec![1.0, 1.0, 1.0], &[1, 3]);

    // The shape `Linear::forward` uses: x @ wᵀ where wᵀ is a view.
    let y = x.matmul(&w.transpose().transpose());
    assert_eq!(y.shape(), &[1, 2]);
    y.backward();

    let g = w.grad_ref().expect("no gradient through the view");
    assert_eq!(g.len(), 6);
    assert!(g.iter().any(|v| *v != 0.0));
}

/// The gradient of a transposed view must land on the base in base order.
#[test]
fn transpose_backward_maps_gradients_back() {
    Tape::reset();
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).requires_grad();
    let t = a.transpose();

    // Weight each transposed position distinctly so a mis-mapping shows up.
    let weights = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let scaled = &t * &weights;
    scaled.sum(None, false).backward();

    // grad[i][j] == weights[j][i]
    let g = a.grad_ref().unwrap();
    assert_eq!(g.as_slice(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn elementwise_ops_accept_views() {
    let a = m23().transpose();
    let b = m23().transpose();
    let sum = &a + &b;
    assert_eq!(sum.shape(), &[3, 2]);
    assert_eq!(sum.to_vec(), vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
}

#[test]
fn narrow_is_a_view_with_an_offset() {
    let t = Tensor::new((0..12).map(|i| i as f32).collect(), &[3, 4]);
    let mid = t.narrow(0, 1, 2);

    assert_eq!(mid.shape(), &[2, 4]);
    // Strides are untouched; only the start moved.
    assert_eq!(mid.stride(), &[4, 1]);
    assert_eq!(mid.to_vec(), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn narrow_along_an_inner_axis_is_strided() {
    let t = Tensor::new((0..12).map(|i| i as f32).collect(), &[3, 4]);
    let cols = t.narrow(1, 1, 2);

    assert_eq!(cols.shape(), &[3, 2]);
    assert!(!cols.is_contiguous(), "an inner window must stay strided");
    assert_eq!(cols.to_vec(), vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
}

#[test]
fn narrow_scatters_gradients_to_the_right_window() {
    Tape::reset();
    let t = Tensor::new((0..12).map(|i| i as f32).collect(), &[3, 4]).requires_grad();
    let cols = t.narrow(1, 1, 2);
    cols.backward();

    // Only the selected columns receive gradient.
    let g = t.grad_ref().unwrap();
    let expected: Vec<f32> = (0..12)
        .map(|i| if i % 4 == 1 || i % 4 == 2 { 1.0 } else { 0.0 })
        .collect();
    assert_eq!(g.as_slice(), expected.as_slice());
}

#[test]
fn reshape_of_a_contiguous_tensor_is_a_view() {
    let t = Tensor::new((0..6).map(|i| i as f32).collect(), &[2, 3]);
    let r = t.reshape(&[3, 2]);

    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.stride(), &[2, 1]);
    assert!(r.is_contiguous());
    assert_eq!(r.to_vec(), t.to_vec());
}

/// A view's logical order is not its memory order, so reshaping one has to
/// pack first — otherwise the result would reinterpret the base's layout.
#[test]
fn reshape_of_a_view_packs_first() {
    let t = Tensor::new((0..6).map(|i| i as f32).collect(), &[2, 3]);
    let r = t.transpose().reshape(&[2, 3]);

    assert!(r.is_contiguous());
    // Transposed order is [0,3,1,4,2,5]; reshaping regroups that, not the base.
    assert_eq!(r.to_vec(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn slicing_helpers_produce_views_that_still_backpropagate() {
    Tape::reset();
    let w = Tensor::new((0..36).map(|i| i as f32 * 0.1).collect(), &[4, 1, 3, 3]).requires_grad();
    let half = w.slice_output_channels(2, 4);

    assert_eq!(half.shape(), &[2, 1, 3, 3]);
    assert_eq!(half.to_vec()[0], w.to_vec()[18]);

    half.backward();
    let g = w.grad_ref().unwrap();
    assert!(
        g[..18].iter().all(|v| *v == 0.0),
        "unselected filters got gradient"
    );
    assert!(
        g[18..].iter().all(|v| *v == 1.0),
        "selected filters missed gradient"
    );
}
