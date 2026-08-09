//! Datasets and batching.
//!
//! [`Dataset`] is the only thing training needs from your data. Implement it
//! and every batching, shuffling and training utility in the crate works — see
//! [`TensorDataset`] for the common case of data already in memory.

pub mod mnist;

use crate::Tensor;

/// A finite, indexable collection of `(input, target)` samples.
///
/// Batching is expressed as a gather over indices rather than one sample at a
/// time, so implementations can copy a whole batch in one pass — the MNIST one
/// does it in parallel.
pub trait Dataset {
    /// Number of samples.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gather `indices` into a batch of `(inputs, targets)`.
    ///
    /// Both tensors must have `indices.len()` as their leading dimension.
    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor);
}

/// Data already resident in memory, as a pair of stacked tensors.
///
/// `inputs` and `targets` share a leading sample dimension; everything after it
/// is per-sample and copied through unchanged, so any feature shape works —
/// `[N, features]` for tabular data, `[N, C, H, W]` for images.
///
/// ```
/// # use taper::{Tensor, data::{Dataset, TensorDataset}};
/// let x = Tensor::new((0..12).map(|i| i as f32).collect(), &[4, 3]);
/// let y = Tensor::new(vec![0.0, 1.0, 0.0, 1.0], &[4]);
/// let dataset = TensorDataset::new(x, y);
///
/// let (bx, by) = dataset.get_batch(&[0, 2]);
/// assert_eq!(bx.shape(), &[2, 3]);
/// assert_eq!(by.shape(), &[2]);
/// ```
pub struct TensorDataset {
    inputs: Tensor,
    targets: Tensor,
}

impl TensorDataset {
    /// # Panics
    /// If the two tensors disagree on the number of samples.
    pub fn new(inputs: Tensor, targets: Tensor) -> Self {
        assert!(
            !inputs.shape().is_empty() && !targets.shape().is_empty(),
            "TensorDataset: inputs and targets need a leading sample dimension"
        );
        assert_eq!(
            inputs.shape()[0],
            targets.shape()[0],
            "TensorDataset: {} inputs but {} targets",
            inputs.shape()[0],
            targets.shape()[0]
        );
        Self { inputs, targets }
    }

    pub fn inputs(&self) -> &Tensor {
        &self.inputs
    }

    pub fn targets(&self) -> &Tensor {
        &self.targets
    }
}

/// Copy the samples at `indices` out of a stacked `[N, ...]` tensor.
fn gather_samples(source: &Tensor, indices: &[usize]) -> Tensor {
    let per_sample: usize = source.shape()[1..].iter().product();
    let values = source.to_vec();

    let mut out = Vec::with_capacity(indices.len() * per_sample);
    for &i in indices {
        let start = i * per_sample;
        out.extend_from_slice(&values[start..start + per_sample]);
    }

    let mut shape = vec![indices.len()];
    shape.extend_from_slice(&source.shape()[1..]);
    Tensor::new(out, &shape)
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.inputs.shape()[0]
    }

    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        (
            gather_samples(&self.inputs, indices),
            gather_samples(&self.targets, indices),
        )
    }
}

/// Iterates a [`Dataset`] in batches, optionally reshuffling each epoch.
///
/// The final batch is short when the dataset does not divide evenly.
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current: usize,
}

impl<D: Dataset> DataLoader<D> {
    /// # Panics
    /// If `batch_size` is zero.
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        assert!(batch_size > 0, "DataLoader: batch_size must be non-zero");

        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());
        }

        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            current: 0,
        }
    }

    /// Rewind to the start, reshuffling if this loader shuffles.
    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            self.indices.shuffle(&mut rand::thread_rng());
        }
    }

    /// Number of batches per pass, counting a short final batch.
    pub fn num_batches(&self) -> usize {
        self.dataset.len().div_ceil(self.batch_size)
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn dataset(&self) -> &D {
        &self.dataset
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.dataset.len());
        let batch = self.dataset.get_batch(&self.indices[self.current..end]);
        self.current = end;

        Some(batch)
    }
}
