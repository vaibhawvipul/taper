//! Recoverable failure: shape problems reported as errors, and a panic in one
//! thread not permanently disabling a tensor.

use taper::Tensor;
use taper::error::ShapeError;

#[test]
fn try_new_reports_a_length_mismatch() {
    let err = Tensor::try_new(vec![1.0, 2.0, 3.0], &[2, 2]).unwrap_err();
    assert!(
        matches!(
            err,
            ShapeError::Length {
                expected: 4,
                got: 3,
                ..
            }
        ),
        "{err:?}"
    );
    // The message a service would log.
    assert!(err.to_string().contains("do not fill shape"), "{err}");
}

#[test]
fn try_new_accepts_a_valid_shape() {
    let t = Tensor::try_new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(t.shape(), &[2, 2]);
}

#[test]
fn try_reshape_reports_an_element_count_mismatch() {
    let t = Tensor::new(vec![0.0; 6], &[2, 3]);
    assert!(t.try_reshape(&[4, 2]).is_err());
    assert!(t.try_reshape(&[3, 2]).is_ok());
    assert!(t.try_reshape(&[6]).is_ok());
}

#[test]
fn try_matmul_reports_bad_inner_dimensions() {
    let a = Tensor::new(vec![0.0; 6], &[2, 3]);
    let b = Tensor::new(vec![0.0; 8], &[4, 2]);

    let err = a.try_matmul(&b).unwrap_err();
    assert!(matches!(err, ShapeError::Mismatch { .. }), "{err:?}");

    assert!(a.try_matmul(&Tensor::new(vec![0.0; 6], &[3, 2])).is_ok());
}

#[test]
fn try_matmul_reports_a_bad_rank() {
    let a = Tensor::new(vec![0.0; 8], &[2, 2, 2]);
    let b = Tensor::new(vec![0.0; 4], &[2, 2]);

    let err = a.try_matmul(&b).unwrap_err();
    assert!(matches!(err, ShapeError::Rank { .. }), "{err:?}");
}

#[test]
fn fallible_operators_report_incompatible_shapes() {
    let a = Tensor::new(vec![0.0; 6], &[2, 3]);
    let bad = Tensor::new(vec![0.0; 8], &[2, 4]);
    let broadcastable = Tensor::new(vec![0.0; 3], &[3]);

    for result in [
        a.try_add(&bad),
        a.try_sub(&bad),
        a.try_mul(&bad),
        a.try_div(&bad),
    ] {
        let err = result.unwrap_err();
        assert!(
            matches!(err, ShapeError::NotBroadcastable { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("do not broadcast"), "{err}");
    }

    // And the compatible case still works through the same path.
    assert_eq!(a.try_add(&broadcastable).unwrap().shape(), &[2, 3]);
}

/// The fallible and panicking forms share one implementation, so a shape the
/// `try_` form accepts must not panic in the operator, and vice versa.
#[test]
fn the_two_forms_agree() {
    let cases: &[(&[usize], &[usize])] = &[
        (&[2, 3], &[2, 3]),
        (&[2, 3], &[3]),
        (&[2, 3], &[2, 1]),
        (&[2, 3], &[2, 4]),
        (&[2, 3], &[5]),
    ];

    for (lhs, rhs) in cases {
        let a = Tensor::new(vec![1.0; lhs.iter().product()], lhs);
        let b = Tensor::new(vec![1.0; rhs.iter().product()], rhs);

        let fallible_ok = a.try_add(&b).is_ok();
        let panicking_ok = std::panic::catch_unwind(|| {
            let _ = &a + &b;
        })
        .is_ok();

        assert_eq!(
            fallible_ok, panicking_ok,
            "try_add and `+` disagree on {lhs:?} + {rhs:?}"
        );
    }
}

/// A panic while a tensor's lock was held used to poison it: every later access
/// panicked, so one bad request took a parameter out of service for the life of
/// the process. The guarded value is a plain numeric buffer, so recovering
/// cannot be unsound.
#[test]
fn a_panic_does_not_permanently_disable_a_tensor() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let shared = t.clone();

    // Panic while holding the write lock.
    let poisoned = std::thread::spawn(move || {
        let _guard = shared.data_mut();
        panic!("simulated failure mid-operation");
    })
    .join();
    assert!(poisoned.is_err(), "the worker was supposed to panic");

    // The tensor is still usable from this thread.
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.data()[0], 1.0);
    t.data_mut()[0] = 9.0;
    assert_eq!(t.to_vec()[0], 9.0);

    // And it still participates in operations.
    let doubled = &t + &t;
    assert_eq!(doubled.to_vec()[0], 18.0);
}

#[test]
fn a_panic_does_not_disable_a_tensors_gradient() {
    let t = Tensor::new(vec![1.0, 2.0], &[2]).requires_grad();
    let shared = t.clone();

    let _ = std::thread::spawn(move || {
        let _guard = shared.grad.write().unwrap();
        panic!("simulated failure while updating a gradient");
    })
    .join();

    // Reading and accumulating both still work.
    assert!(t.grad_ref().is_none());
    taper::ops::accumulate_grad(&t, &[1.0, 1.0]);
    assert_eq!(t.grad_ref().unwrap().as_slice(), &[1.0, 1.0]);
}

#[test]
fn shape_errors_are_std_errors() {
    fn boxed() -> Result<Tensor, Box<dyn std::error::Error>> {
        // `?` has to work through the standard error trait for these to be
        // usable in ordinary application code.
        Ok(Tensor::try_new(vec![1.0], &[2])?)
    }
    assert!(boxed().is_err());
}
