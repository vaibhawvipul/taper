//! Training on arbitrary data.
//!
//! `DataLoader` used to hold a concrete `MNISTDataset` and `Trainer` took only
//! that type, so the crate's headline training API worked on exactly one
//! dataset. These tests pin the general case.

use std::collections::HashSet;

use taper::Tensor;
use taper::activation::ReLU;
use taper::data::{DataLoader, Dataset, TensorDataset};
use taper::nn::{Linear, Module, Sequential};
use taper::optim::Adam;
use taper::train::Trainer;

/// A synthetic two-class problem with a feature width that is deliberately not 784.
fn synthetic(samples: usize, features: usize) -> TensorDataset {
    let mut x = Vec::with_capacity(samples * features);
    let mut y = Vec::with_capacity(samples);

    for i in 0..samples {
        let positive = i % 2 == 0;
        for f in 0..features {
            // Two well-separated clusters, with a little deterministic jitter.
            let base = if positive { 1.0 } else { -1.0 };
            let jitter = ((i * 7 + f * 13) % 11) as f32 * 0.01;
            x.push(base + jitter);
        }
        y.push(if positive { 1.0 } else { 0.0 });
    }

    TensorDataset::new(
        Tensor::new(x, &[samples, features]),
        Tensor::new(y, &[samples]),
    )
}

#[test]
fn tensor_dataset_handles_arbitrary_feature_widths() {
    let dataset = synthetic(10, 5);
    assert_eq!(dataset.len(), 10);

    let (x, y) = dataset.get_batch(&[0, 3, 7]);
    assert_eq!(x.shape(), &[3, 5]);
    assert_eq!(y.shape(), &[3]);
}

#[test]
fn tensor_dataset_preserves_per_sample_shape() {
    // Image-shaped samples: everything after the leading dimension passes through.
    let images = Tensor::new(vec![0.0; 4 * 3 * 8 * 8], &[4, 3, 8, 8]);
    let labels = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], &[4]);
    let dataset = TensorDataset::new(images, labels);

    let (x, y) = dataset.get_batch(&[1, 2]);
    assert_eq!(x.shape(), &[2, 3, 8, 8]);
    assert_eq!(y.to_vec(), vec![1.0, 2.0]);
}

#[test]
fn tensor_dataset_gathers_the_requested_samples() {
    let x = Tensor::new((0..12).map(|i| i as f32).collect(), &[4, 3]);
    let y = Tensor::new(vec![10.0, 11.0, 12.0, 13.0], &[4]);
    let dataset = TensorDataset::new(x, y);

    let (bx, by) = dataset.get_batch(&[3, 0]);
    assert_eq!(bx.to_vec(), vec![9.0, 10.0, 11.0, 0.0, 1.0, 2.0]);
    assert_eq!(by.to_vec(), vec![13.0, 10.0]);
}

#[test]
#[should_panic(expected = "but")]
fn tensor_dataset_rejects_mismatched_sample_counts() {
    let x = Tensor::new(vec![0.0; 6], &[3, 2]);
    let y = Tensor::new(vec![0.0; 2], &[2]);
    let _ = TensorDataset::new(x, y);
}

#[test]
fn dataloader_covers_every_sample_exactly_once() {
    // 10 samples in batches of 4 -> 4 + 4 + 2.
    let mut loader = DataLoader::new(synthetic(10, 3), 4, false);
    assert_eq!(loader.num_batches(), 3);

    let sizes: Vec<usize> = loader.by_ref().map(|(x, _)| x.shape()[0]).collect();
    assert_eq!(sizes, vec![4, 4, 2]);

    // And the same total after a reset.
    loader.reset();
    let total: usize = loader.map(|(x, _)| x.shape()[0]).sum();
    assert_eq!(total, 10);
}

#[test]
fn shuffling_reorders_without_losing_samples() {
    // Distinct first features let us identify each sample after shuffling.
    let dataset = TensorDataset::new(
        Tensor::new((0..8).map(|i| i as f32).collect(), &[8, 1]),
        Tensor::new(vec![0.0; 8], &[8]),
    );
    let mut loader = DataLoader::new(dataset, 8, true);

    let seen: HashSet<u32> = loader
        .by_ref()
        .flat_map(|(x, _)| x.to_vec())
        .map(|v| v as u32)
        .collect();
    assert_eq!(seen, (0..8).collect::<HashSet<u32>>());
}

#[test]
#[should_panic(expected = "batch_size must be non-zero")]
fn dataloader_rejects_a_zero_batch_size() {
    let _ = DataLoader::new(synthetic(4, 2), 0, false);
}

/// A user-defined dataset needs nothing but the trait — no crate changes.
struct Countdown {
    samples: usize,
}

impl Dataset for Countdown {
    fn len(&self) -> usize {
        self.samples
    }

    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        let x: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
        let y: Vec<f32> = indices.iter().map(|&i| (i % 2) as f32).collect();
        (
            Tensor::new(x, &[indices.len(), 1]),
            Tensor::new(y, &[indices.len()]),
        )
    }
}

#[test]
fn a_custom_dataset_works_with_the_loader() {
    let mut loader = DataLoader::new(Countdown { samples: 5 }, 2, false);
    assert_eq!(loader.num_batches(), 3);

    let (x, y) = loader.next().unwrap();
    assert_eq!(x.to_vec(), vec![0.0, 1.0]);
    assert_eq!(y.to_vec(), vec![0.0, 1.0]);
}

/// The point of the whole change: `Trainer` trains on data that is not MNIST.
#[test]
fn trainer_learns_a_non_mnist_dataset() {
    let features = 6;
    let model = Sequential::new(vec![
        Box::new(Linear::new(features, 16, true)),
        Box::new(ReLU),
        Box::new(Linear::new(16, 2, true)),
    ]);
    let optimizer = Adam::new(model.parameters(), 0.01, None, None, None);
    let mut trainer = Trainer::new(Box::new(model), optimizer, None);

    let mut train = DataLoader::new(synthetic(256, features), 32, true);

    let (first_loss, _) = trainer.train_epoch(&mut train);
    for _ in 0..4 {
        trainer.train_epoch(&mut train);
    }
    let (last_loss, last_acc) = trainer.train_epoch(&mut train);

    assert!(
        last_loss < first_loss,
        "loss did not fall: {first_loss} -> {last_loss}"
    );
    assert!(
        last_acc > 0.9,
        "separable problem should be learned, got {last_acc}"
    );
}

/// `fit` is the headline entry point, so it has to be dataset-generic too.
#[test]
fn fit_runs_end_to_end_on_a_custom_dataset() {
    let features = 4;
    let model = Sequential::new(vec![
        Box::new(Linear::new(features, 8, true)),
        Box::new(ReLU),
        Box::new(Linear::new(8, 2, true)),
    ]);
    let optimizer = Adam::new(model.parameters(), 0.01, None, None, None);
    let mut trainer = Trainer::new(Box::new(model), optimizer, None).with_early_stop(None);

    let mut train = DataLoader::new(synthetic(128, features), 16, true);
    let mut val = DataLoader::new(synthetic(64, features), 16, false);

    trainer.fit(&mut train, &mut val, 3, false);

    assert_eq!(trainer.metrics.train_loss.len(), 3);
    assert!(
        trainer.metrics.val_acc.last().copied().unwrap_or(0.0) > 0.9,
        "validation accuracy did not reach 90%: {:?}",
        trainer.metrics.val_acc
    );
}
