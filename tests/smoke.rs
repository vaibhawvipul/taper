use std::time::Instant;

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
        println!("Modest SIMD benefit ({:.1}x). May be memory-bound.", speedup);
    } else {
        println!("SIMD doesn't appear to be active (only {:.1}x speedup)", speedup);
        println!("  Try: RUSTFLAGS=\"-C target-cpu=native\" cargo test --release");
    }

    // Assert reasonable performance (adjust threshold as needed)
    assert!(speedup > 1.2, "Expected SIMD speedup > 1.2x, got {:.2}x", speedup);
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

    println!("1000 iterations of mixed ops: {:.2} ms", elapsed.as_millis());
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
