use taper::{Tape, Tensor};

#[inline]
fn s(t: &Tensor) -> f32 {
    t.data()[0]
}

macro_rules! g {
    ($t:expr) => {
        $t.grad().map(|t| t.data()[0]).unwrap_or(0.0)
    };
}

#[test]
fn mul_grads() {
    let _tape = Tape::new();
    let x = Tensor::scalar(2.0).requires_grad();
    let y = Tensor::scalar(3.0).requires_grad();
    let z = &x * &y;
    z.backward();

    assert!((s(&z) - 6.0).abs() < 1e-6);
    assert!((g!(x) - 3.0).abs() < 1e-6);
    assert!((g!(y) - 2.0).abs() < 1e-6);
}

#[test]
fn compound_affine() {
    let _tape = Tape::new();
    let a = Tensor::scalar(2.0).requires_grad();
    let b = Tensor::scalar(3.0).requires_grad();
    let c = &a * &b + &a; // c = a*b + a
    c.backward();

    assert!((s(&c) - 8.0).abs() < 1e-6);
    assert!((g!(a) - 4.0).abs() < 1e-6); // b + 1
    assert!((g!(b) - 2.0).abs() < 1e-6); // a
}

#[test]
fn matmul_shapes_and_grads() {
    let _tape = Tape::new();

    // [2x3] @ [3x2] -> [2x2]
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], &[2, 3]).requires_grad();
    let b = Tensor::new(vec![7., 8., 9., 10., 11., 12.], &[3, 2]).requires_grad();

    let c = a.matmul(&b);
    assert_eq!(c.shape(), &[2, 2]);

    c.backward();

    // sanity: gradients exist and have expected shapes
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(ga.shape(), &[2, 3]);
    assert_eq!(gb.shape(), &[3, 2]);

    // light numeric checks (spot-check a couple entries)
    let cd = c.data();
    assert!(cd.len() == 4);
    // expected C (row-major): [[58, 64], [139, 154]]
    assert!((cd[0] - 58.0).abs() < 1e-4);
    assert!((cd[3] - 154.0).abs() < 1e-4);
}
