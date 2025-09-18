// build.rs
fn main() {
    // Link Apple Accelerate when the feature is enabled
    #[cfg(all(feature = "blas-accelerate", target_os = "macos"))]
    {
        // Provides cblas_* symbols on macOS
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
