// `blas-src` carries the native link directives for the selected BLAS vendor,
// but it exports no items we call — we go through `cblas-sys` instead. Without
// an explicit reference rustc drops it as an unused dependency and its
// `-l` flags go with it, so the build gets the library's search path but never
// links it: `undefined symbol: cblas_sgemm`.
#[cfg(feature = "blas")]
use blas_src as _;

pub mod activation;
pub mod data;
pub mod gemm;
pub mod gradcheck;
pub mod loss;
pub mod nn;
pub mod norm;
pub mod ops;
pub mod optim;
pub mod quantization;
pub mod safetensors;
pub mod tape;
pub mod tensor;
pub mod train;

pub use gemm::{n, sgemm_rowmajor, t};
pub use quantization::{QATConfig, QATManager, QuantizationConfig, QuantizationType};
pub use tape::Tape;
pub use tensor::{QuantizedTensor, Tensor};
pub use train::{Metrics, Trainer};
