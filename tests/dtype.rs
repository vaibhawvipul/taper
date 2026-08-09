//! Narrow storage dtypes. Computation stays in f32; the dtype controls what a
//! tensor *stores*, not the precision it is evaluated at.

use taper::tensor::{DType, Storage};
use taper::{Tape, Tensor};

#[test]
fn tensors_are_f32_by_default() {
    let t = Tensor::new(vec![1.0, 2.0], &[2]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.storage_bytes(), 8);
}

#[test]
fn narrow_dtypes_shrink_storage() {
    let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.25).collect();
    let f32_t = Tensor::new(values, &[64]);
    assert_eq!(f32_t.storage_bytes(), 256);

    // Half the bytes for the 16-bit types, a quarter for u8.
    assert_eq!(f32_t.to_dtype(DType::BF16).storage_bytes(), 128);
    assert_eq!(f32_t.to_dtype(DType::F16).storage_bytes(), 128);
    assert_eq!(f32_t.to_dtype(DType::I32).storage_bytes(), 256);
    assert_eq!(f32_t.to_dtype(DType::U8).storage_bytes(), 64);
}

#[test]
fn casting_round_trips_within_the_format_precision() {
    let values = vec![-2.5, -0.125, 0.0, 1.75, 100.0];
    let t = Tensor::new(values.clone(), &[5]);

    for (dtype, tolerance) in [(DType::BF16, 1.0), (DType::F16, 0.05)] {
        let back = t.to_dtype(dtype).to_dtype(DType::F32);
        assert_eq!(back.dtype(), DType::F32);
        for (got, want) in back.to_vec().iter().zip(values.iter()) {
            assert!(
                (got - want).abs() <= tolerance,
                "{dtype:?}: {want} round-tripped to {got}"
            );
        }
    }
}

#[test]
fn integer_storage_holds_exact_values() {
    // Class labels are the motivating case: they are integers, not floats.
    let labels = Tensor::from_storage(Storage::I32(vec![0, 3, 9, 7]), &[4]);
    assert_eq!(labels.dtype(), DType::I32);
    assert_eq!(labels.to_vec(), vec![0.0, 3.0, 9.0, 7.0]);
}

#[test]
fn u8_storage_holds_exact_values() {
    let mask = Tensor::from_storage(Storage::U8(vec![0, 1, 1, 0]), &[2, 2]);
    assert_eq!(mask.dtype(), DType::U8);
    assert_eq!(mask.to_vec(), vec![0.0, 1.0, 1.0, 0.0]);
    assert_eq!(mask.storage_bytes(), 4);
}

/// Ops read through `elements()`, which widens whatever the storage holds, so a
/// narrow tensor participates in arithmetic without any special casing.
#[test]
fn arithmetic_works_across_dtypes() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).to_dtype(DType::BF16);
    let b = Tensor::from_storage(Storage::I32(vec![10, 20, 30, 40]), &[4]);

    let sum = &a + &b;
    // The result is computed and stored in f32.
    assert_eq!(sum.dtype(), DType::F32);
    assert_eq!(sum.to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn matmul_accepts_narrow_operands() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let reference = a.matmul(&b).to_vec();

    // bf16 holds these exactly, so the product must be identical.
    let narrow = a.to_dtype(DType::BF16);
    assert_eq!(narrow.matmul(&b).to_vec(), reference);
    assert_eq!(a.matmul(&b.to_dtype(DType::BF16)).to_vec(), reference);
}

/// The cast is element-wise with unit derivative, so gradients pass through.
#[test]
fn casting_passes_gradients_through() {
    Tape::reset();
    let w = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).requires_grad();
    let narrow = w.to_dtype(DType::BF16);
    narrow.backward();

    assert_eq!(w.grad_ref().unwrap().as_slice(), &[1.0, 1.0, 1.0]);
}

/// `data()` hands out a raw f32 slice, which only makes sense for f32 storage.
#[test]
#[should_panic(expected = "needs f32 storage")]
fn raw_f32_access_to_narrow_storage_is_rejected() {
    let t = Tensor::new(vec![1.0, 2.0], &[2]).to_dtype(DType::BF16);
    drop(t.data());
}

#[test]
fn casting_to_the_same_dtype_is_a_no_op() {
    let t = Tensor::new(vec![1.0, 2.0], &[2]);
    let same = t.to_dtype(DType::F32);
    assert_eq!(same.to_vec(), t.to_vec());
    assert_eq!(same.dtype(), DType::F32);
}

#[test]
fn narrow_views_still_read_in_logical_order() {
    let t = Tensor::new((0..6).map(|i| i as f32).collect(), &[2, 3]).to_dtype(DType::F16);
    let viewed = t.transpose();

    assert_eq!(viewed.dtype(), DType::F16);
    assert_eq!(viewed.to_vec(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}
