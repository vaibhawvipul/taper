//! Recoverable errors from tensor operations.
//!
//! Most of this crate's preconditions are programmer errors — an internal
//! invariant that a caller cannot influence — and those keep panicking, which
//! is idiomatic. But shapes are frequently *data*-derived: a feature count read
//! from a CSV header, a sequence length from a request. Aborting the process
//! over one malformed input is not acceptable in a service, so every operation
//! whose validity depends on shapes has a `try_` form returning [`ShapeError`].
//!
//! The panicking forms are thin wrappers over the fallible ones, so there is a
//! single implementation and the two cannot drift apart.
//!
//! ```
//! use taper::Tensor;
//!
//! // Recover instead of aborting.
//! let malformed = Tensor::try_new(vec![1.0, 2.0, 3.0], &[2, 2]);
//! assert!(malformed.is_err());
//!
//! let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
//! let b = Tensor::new(vec![1.0; 6], &[3, 2]);
//! match a.try_matmul(&b) {
//!     Ok(product) => println!("{:?}", product.shape()),
//!     Err(e) => eprintln!("bad request: {e}"),
//! }
//! ```

use std::fmt;

/// A shape precondition that a caller can plausibly hit with real data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// A buffer's length does not match the shape it was given.
    Length {
        op: &'static str,
        expected: usize,
        got: usize,
        shape: Vec<usize>,
    },
    /// Two shapes had to be equal and were not.
    Mismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    /// Two shapes do not broadcast against each other.
    NotBroadcastable {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    /// An operation needed a particular rank.
    Rank {
        op: &'static str,
        expected: &'static str,
        got: Vec<usize>,
    },
    /// An index or range fell outside a dimension.
    OutOfBounds {
        op: &'static str,
        dim: usize,
        requested: usize,
        limit: usize,
    },
    /// A dimension was zero where the operation needs a positive extent.
    ZeroExtent {
        op: &'static str,
        what: &'static str,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Wording is load-bearing: `expect` on these produces the panic
            // messages the crate has always emitted.
            ShapeError::Length {
                op,
                expected,
                got,
                shape,
            } => write!(
                f,
                "{op}: {got} values do not fill shape {shape:?} ({expected} needed)"
            ),
            ShapeError::Mismatch { op, lhs, rhs } => {
                write!(f, "{op}: shapes must match ({lhs:?} vs {rhs:?})")
            }
            ShapeError::NotBroadcastable { op, lhs, rhs } => {
                write!(f, "{op}: shapes {lhs:?} and {rhs:?} do not broadcast")
            }
            ShapeError::Rank { op, expected, got } => {
                write!(f, "{op}: expected {expected}, got shape {got:?}")
            }
            ShapeError::OutOfBounds {
                op,
                dim,
                requested,
                limit,
            } => write!(
                f,
                "{op}: {requested} is out of range for dimension {dim} of length {limit}"
            ),
            ShapeError::ZeroExtent { op, what } => write!(f, "{op}: {what} must be non-zero"),
        }
    }
}

impl std::error::Error for ShapeError {}

/// Shorthand for fallible tensor operations.
pub type Result<T> = std::result::Result<T, ShapeError>;
