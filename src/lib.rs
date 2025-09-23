pub mod activation;
pub mod gemm;
pub mod loss;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod tape;
pub mod tensor;
pub mod train;
pub mod quantization; 
pub mod data;

pub use gemm::{n, sgemm_rowmajor, t};
pub use quantization::QuantizationConfig;
pub use tape::Tape;
pub use tensor::Tensor;
pub use train::{Metrics, Trainer};
