use std::time::Instant;

use taper::{
    Tape, Tensor,
    loss::{accuracy, cross_entropy_loss, log_softmax, softmax},
};

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
    let _tape = Tape::reset();
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
    let _tape = Tape::reset();
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
    let _tape = Tape::reset();

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

#[test]
fn verify_simd_is_working() {
    println!("\n=== SIMD Feature Detection ===");

    // Check CPU features
    #[cfg(target_arch = "x86_64")]
    {
        println!("Architecture: x86_64");
        println!("  SSE: {}", is_x86_feature_detected!("sse"));
        println!("  AVX: {}", is_x86_feature_detected!("avx"));
        println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        println!("  FMA: {}", is_x86_feature_detected!("fma"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("Architecture: aarch64 (ARM NEON)");
    }

    println!("\n=== SIMD Performance Test ===");

    // Test with a size that shows SIMD benefits
    let size = 100_000;
    let iterations = 100;

    // Create test tensors
    let a = Tensor::randn(&[size]);
    let b = Tensor::randn(&[size]);

    // Warm-up
    for _ in 0..10 {
        let _ = &a + &b;
    }

    // Benchmark element-wise addition
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = &a + &b;
    }
    let elapsed = start.elapsed();

    let ops = (iterations * size) as f64;
    let gflops = ops / elapsed.as_secs_f64() / 1e9;

    println!("Vector addition ({} elements):", size);
    println!("  Time: {:.2} ms", elapsed.as_millis());
    println!("  Performance: {:.2} GFLOPS", gflops);

    // Benchmark element-wise multiplication
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = &a * &b;
    }
    let elapsed = start.elapsed();

    let gflops = ops / elapsed.as_secs_f64() / 1e9;
    println!("\nVector multiplication ({} elements):", size);
    println!("  Time: {:.2} ms", elapsed.as_millis());
    println!("  Performance: {:.2} GFLOPS", gflops);

    // Test ReLU (another SIMD operation)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.relu();
    }
    let elapsed = start.elapsed();

    let gflops = ops / elapsed.as_secs_f64() / 1e9;
    println!("\nReLU activation ({} elements):", size);
    println!("  Time: {:.2} ms", elapsed.as_millis());
    println!("  Performance: {:.2} GFLOPS", gflops);

    // Matrix multiplication test
    let mat_size = 256;
    let mat_a = Tensor::randn(&[mat_size, mat_size]);
    let mat_b = Tensor::randn(&[mat_size, mat_size]);

    let start = Instant::now();
    let _ = mat_a.matmul(&mat_b);
    let elapsed = start.elapsed();

    let flops = 2.0 * (mat_size as f64).powi(3);
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    println!("\nMatrix multiplication ({}x{}):", mat_size, mat_size);
    println!("  Time: {:.2} ms", elapsed.as_millis());
    println!("  Performance: {:.2} GFLOPS", gflops);

    // Compare with scalar baseline
    println!("\n=== SIMD vs Scalar Comparison ===");

    let test_size = 10_000;
    let test_iters = 1000;

    // Scalar implementation
    let vec_a: Vec<f32> = (0..test_size).map(|i| i as f32 * 0.1).collect();
    let vec_b: Vec<f32> = (0..test_size).map(|i| i as f32 * 0.2).collect();

    let start = Instant::now();
    for _ in 0..test_iters {
        let mut result = vec![0.0f32; test_size];
        for i in 0..test_size {
            result[i] = vec_a[i] + vec_b[i];
        }
        // Prevent optimization
        std::hint::black_box(&result);
    }
    let scalar_time = start.elapsed();

    // SIMD implementation (through Tensor)
    let tensor_a = Tensor::new(vec_a.clone(), &[test_size]);
    let tensor_b = Tensor::new(vec_b.clone(), &[test_size]);

    let start = Instant::now();
    for _ in 0..test_iters {
        let result = &tensor_a + &tensor_b;
        std::hint::black_box(&result);
    }
    let simd_time = start.elapsed();

    let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();

    println!("Scalar time: {:.2} ms", scalar_time.as_millis());
    println!("SIMD time: {:.2} ms", simd_time.as_millis());
    println!("Speedup: {:.2}x", speedup);

    // Verdict
    println!("\n=== Verdict ===");
    if speedup > 2.0 {
        println!("SIMD is working fine! {:.1}x speedup", speedup);
    } else if speedup > 1.5 {
        println!("SIMD is working! {:.1}x speedup", speedup);
    } else if speedup > 1.2 {
        println!(
            "Modest SIMD benefit ({:.1}x). May be memory-bound.",
            speedup
        );
    } else {
        println!(
            "SIMD doesn't appear to be active (only {:.1}x speedup)",
            speedup
        );
        println!("  Try: RUSTFLAGS=\"-C target-cpu=native\" cargo test --release");
    }

    // Assert reasonable performance (adjust threshold as needed)
    assert!(
        speedup > 1.2,
        "Expected SIMD speedup > 1.2x, got {:.2}x",
        speedup
    );
}

#[test]
fn benchmark_with_mimalloc_and_simd() {
    use std::time::Instant;

    println!("\n=== Performance with MiMalloc + SIMD ===");

    // Allocation-heavy test
    let start = Instant::now();
    for _ in 0..1000 {
        let a = Tensor::randn(&[1000]);
        let b = Tensor::randn(&[1000]);
        let _ = &a + &b;
        let _ = &a * &b;
        let _ = a.relu();
    }
    let elapsed = start.elapsed();

    println!(
        "1000 iterations of mixed ops: {:.2} ms",
        elapsed.as_millis()
    );
    println!("Per iteration: {:.2} μs", elapsed.as_micros() / 1000);

    // Matrix multiplication (allocation + compute intensive)
    let sizes = vec![128, 256, 512];
    for size in sizes {
        let a = Tensor::randn(&[size, size]);
        let b = Tensor::randn(&[size, size]);

        let start = Instant::now();
        let _ = a.matmul(&b);
        let elapsed = start.elapsed();

        println!("{}x{} matmul: {:.2} ms", size, size, elapsed.as_millis());
    }
}

#[test]
fn test_reshape_operations() {
    // Test basic reshape
    let x = Tensor::new((0..12).map(|i| i as f32).collect(), &[3, 4]);
    let reshaped = x.reshape(&[2, 6]);
    assert_eq!(reshaped.shape(), &[2, 6]);

    // Test flatten
    let flattened = x.flatten(0);
    assert_eq!(flattened.shape(), &[12]);

    // Partial flatten
    let x3d = Tensor::new((0..24).map(|i| i as f32).collect(), &[2, 3, 4]);
    let flat = x3d.flatten(1);
    assert_eq!(flat.shape(), &[2, 12]);

    // Test squeeze
    let x_squeezable = Tensor::new(vec![1.0, 2.0], &[1, 2, 1]);
    let squeezed = x_squeezable.squeeze(None);
    assert_eq!(squeezed.shape(), &[2]);

    // Test unsqueeze
    let x_1d = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
    let unsqueezed = x_1d.unsqueeze(0);
    assert_eq!(unsqueezed.shape(), &[1, 3]);

    let unsqueezed2 = x_1d.unsqueeze(1);
    assert_eq!(unsqueezed2.shape(), &[3, 1]);
}

#[test]
fn test_reshape_gradients() {
    let _tape = Tape::reset();

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
    let reshaped = x.reshape(&[4]);
    let sum = reshaped.sum(None, false);

    sum.backward();

    let grad = x.grad_ref().unwrap();
    // All elements should have gradient of 1
    for &g in grad.iter() {
        assert!((g - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_sum_operations() {
    // Test sum all
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let sum_all = x.sum(None, false);
    assert_eq!(sum_all.shape(), &[1]);
    assert!((sum_all.data()[0] - 21.0).abs() < 1e-6);

    // Test sum along dimension 0
    let sum_dim0 = x.sum(Some(0), false);
    assert_eq!(sum_dim0.shape(), &[3]);
    let expected = vec![5.0, 7.0, 9.0]; // [1+4, 2+5, 3+6]
    for (a, b) in sum_dim0.data().iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6);
    }

    // Test sum along dimension 1
    let sum_dim1 = x.sum(Some(1), false);
    assert_eq!(sum_dim1.shape(), &[2]);
    let expected = vec![6.0, 15.0]; // [1+2+3, 4+5+6]
    for (a, b) in sum_dim1.data().iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6);
    }

    // Test keepdim
    let sum_keepdim = x.sum(Some(1), true);
    assert_eq!(sum_keepdim.shape(), &[2, 1]);
}

#[test]
fn test_sum_gradients() {
    let _tape = Tape::reset();

    // Test gradient flow through sum
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
    let sum = x.sum(Some(1), false); // Sum along columns
    let loss = sum.sum(None, false); // Total sum

    loss.backward();

    let grad = x.grad_ref().unwrap();
    // All elements should have gradient of 1
    for &g in grad.iter() {
        assert!((g - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_max_operations() {
    let x = Tensor::new(vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0], &[2, 3]);

    // Max along dimension 0
    let (max_vals, max_indices) = x.max(Some(0));
    assert_eq!(max_vals.shape(), &[1, 3]);
    assert!((max_vals.data()[0] - 4.0).abs() < 1e-6); // max(1, 4)
    assert!((max_vals.data()[1] - 6.0).abs() < 1e-6); // max(3, 6)
    assert!((max_vals.data()[2] - 5.0).abs() < 1e-6); // max(2, 5)

    // Check indices
    assert!((max_indices.data()[0] - 1.0).abs() < 1e-6); // index 1 (second row)
    assert!((max_indices.data()[1] - 1.0).abs() < 1e-6); // index 1 (second row)
    assert!((max_indices.data()[2] - 1.0).abs() < 1e-6); // index 1 (second row)

    // Argmax
    let argmax = x.argmax(Some(1));
    assert_eq!(argmax.shape(), &[2, 1]);
    assert!((argmax.data()[0] - 1.0).abs() < 1e-6); // index 1 (value 3)
    assert!((argmax.data()[1] - 1.0).abs() < 1e-6); // index 1 (value 6)
}

#[test]
fn test_exp_log_operations() {
    let x = Tensor::new(vec![0.0, 1.0, 2.0], &[3]);

    // Test exp
    let exp_x = x.exp();
    assert!((exp_x.data()[0] - 1.0).abs() < 1e-6); // e^0 = 1
    assert!((exp_x.data()[1] - 2.71828).abs() < 1e-2); // e^1 ≈ 2.718
    assert!((exp_x.data()[2] - 7.38906).abs() < 1e-2); // e^2 ≈ 7.389

    // Test log
    let log_exp = exp_x.log();
    for (a, b) in log_exp.data().iter().zip(x.data().iter()) {
        assert!((a - b).abs() < 1e-5); // log(exp(x)) = x
    }

    // Test pow and sqrt
    let x2 = Tensor::new(vec![1.0, 4.0, 9.0], &[3]);
    let sqrt_x = x2.sqrt();
    assert!((sqrt_x.data()[0] - 1.0).abs() < 1e-6);
    assert!((sqrt_x.data()[1] - 2.0).abs() < 1e-6);
    assert!((sqrt_x.data()[2] - 3.0).abs() < 1e-6);

    let squared = sqrt_x.pow(2.0);
    for (a, b) in squared.data().iter().zip(x2.data().iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_exp_log_gradients() {
    let _tape = Tape::reset();

    // Test exp gradient
    let x = Tensor::new(vec![1.0, 2.0], &[2]).requires_grad();
    let exp_x = x.exp();
    let sum = exp_x.sum(None, false);
    sum.backward();

    let grad = x.grad_ref().unwrap();
    let exp_vals = x.exp();
    for (g, e) in grad.iter().zip(exp_vals.data().iter()) {
        assert!((g - e).abs() < 1e-5); // d/dx e^x = e^x
    }

    // Test log gradient
    let _tape = Tape::reset();
    let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).requires_grad();
    let log_x = x.log();
    let sum = log_x.sum(None, false);
    sum.backward();

    let grad = x.grad_ref().unwrap();
    for (g, x_val) in grad.iter().zip(x.data().iter()) {
        assert!((g - 1.0 / x_val).abs() < 1e-5); // d/dx ln(x) = 1/x
    }
}

#[test]
fn test_softmax_cross_entropy() {
    // Test softmax
    let logits = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &[2, 3]);
    let probs = softmax(&logits, -1);

    // Check that each row sums to 1
    let row_sums = probs.sum(Some(1), false);
    for &s in row_sums.data().iter() {
        assert!((s - 1.0).abs() < 1e-6);
    }

    // Test cross-entropy loss
    let _tape = Tape::reset();
    let logits = Tensor::new(vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0], &[2, 3]).requires_grad();
    let targets = Tensor::new(vec![0.0, 2.0], &[2]); // First sample -> class 0, second -> class 2

    let loss = cross_entropy_loss(&logits, &targets);
    assert!(loss.data()[0] > 0.0); // Loss should be positive

    loss.backward();
    assert!(logits.grad_ref().is_some());
}

#[test]
fn test_mnist_simulation() {
    // Simulate a mini MNIST-like scenario
    let _tape = Tape::reset();

    // Mock data: 4 samples, 784 features (28x28 flattened)
    let batch_size = 4;
    let input_size = 784;
    let num_classes = 10;

    // Random input (normally would be MNIST images)
    let x = Tensor::randn(&[batch_size, input_size]);

    // Simple linear model (no hidden layers for this test)
    let w = Tensor::randn(&[num_classes, input_size]).requires_grad();
    let b = Tensor::randn(&[num_classes]).requires_grad();

    // Forward pass
    let logits = x.matmul(&w.transpose()).add_broadcast(&b);

    // Random targets
    let targets = Tensor::new(vec![3.0, 7.0, 1.0, 9.0], &[batch_size]);

    // Compute loss
    let loss = cross_entropy_loss(&logits, &targets);

    // Check that we can compute gradients
    loss.backward();

    assert!(w.grad_ref().is_some());
    assert!(b.grad_ref().is_some());

    // Check accuracy computation
    let acc = accuracy(&logits, &targets);
    assert!(acc >= 0.0 && acc <= 1.0);

    println!(
        "Mini MNIST test - Loss: {:.4}, Accuracy: {:.2}%",
        loss.data()[0],
        acc * 100.0
    );
}

#[test]
fn test_numerical_stability() {
    // Test that softmax is numerically stable with large values
    let x = Tensor::new(vec![1000.0, 1001.0, 1002.0], &[1, 3]);
    let probs = softmax(&x, -1);

    // Should not produce NaN or Inf
    for &p in probs.data().iter() {
        assert!(!p.is_nan());
        assert!(!p.is_infinite());
        assert!(p >= 0.0 && p <= 1.0);
    }

    // Test log_softmax stability
    let log_probs = log_softmax(&x, -1);
    for &lp in log_probs.data().iter() {
        assert!(!lp.is_nan());
        assert!(!lp.is_infinite());
    }
}
