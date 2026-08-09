use crate::data::mnist::DataLoader;
use crate::loss::{correct_count, cross_entropy_loss};
use crate::optim::{Adam, LRScheduler};
use crate::{Tape, nn::Module};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

/// Training metrics tracking
#[derive(Clone, Default)]
pub struct Metrics {
    pub train_loss: Vec<f32>,
    pub train_acc: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub val_acc: Vec<f32>,
    pub epoch_times: Vec<f32>,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn print_last(&self) {
        if let (Some(train_loss), Some(train_acc), Some(val_loss), Some(val_acc)) = (
            self.train_loss.last(),
            self.train_acc.last(),
            self.val_loss.last(),
            self.val_acc.last(),
        ) {
            println!(
                "Train Loss: {:.4} | Train Acc: {:.2}% | Val Loss: {:.4} | Val Acc: {:.2}%",
                train_loss,
                train_acc * 100.0,
                val_loss,
                val_acc * 100.0
            );
        }
    }

    pub fn plot_summary(&self) {
        println!("\nTraining Summary:");
        println!("{}", "=".repeat(50));

        if !self.train_acc.is_empty() {
            let best_train_acc = self.train_acc.iter().copied().fold(0.0f32, f32::max);
            let best_val_acc = self.val_acc.iter().copied().fold(0.0f32, f32::max);
            let final_train_acc = self.train_acc.last().unwrap();
            let final_val_acc = self.val_acc.last().unwrap();

            println!("Best Train Accuracy: {:.2}%", best_train_acc * 100.0);
            println!("Best Val Accuracy: {:.2}%", best_val_acc * 100.0);
            println!("Final Train Accuracy: {:.2}%", final_train_acc * 100.0);
            println!("Final Val Accuracy: {:.2}%", final_val_acc * 100.0);

            if !self.epoch_times.is_empty() {
                let total_time: f32 = self.epoch_times.iter().sum();
                let avg_time = total_time / self.epoch_times.len() as f32;
                println!("Total Training Time: {:.2}s", total_time);
                println!("Average Epoch Time: {:.2}s", avg_time);
            }
        }

        println!("{}", "=".repeat(50));
    }
}

/// Trainer class that handles the training loop
pub struct Trainer {
    pub model: Box<dyn Module>,
    pub optimizer: Adam,
    pub scheduler: Option<Box<dyn LRScheduler>>,
    pub metrics: Metrics,
    pub device: String, // For future GPU support
    /// Stop [`Trainer::fit`] once validation accuracy reaches this value.
    /// `None` runs every requested epoch.
    pub early_stop_val_acc: Option<f32>,
}

impl Trainer {
    pub fn new(
        model: Box<dyn Module>,
        optimizer: Adam,
        scheduler: Option<Box<dyn LRScheduler>>,
    ) -> Self {
        Trainer {
            model,
            optimizer,
            scheduler,
            metrics: Metrics::new(),
            device: "cpu".to_string(),
            early_stop_val_acc: Some(0.99),
        }
    }

    /// Set (or clear, with `None`) the validation-accuracy early-stop threshold.
    pub fn with_early_stop(mut self, val_acc: Option<f32>) -> Self {
        self.early_stop_val_acc = val_acc;
        self
    }

    /// Train for one epoch
    pub fn train_epoch(&mut self, dataloader: &mut DataLoader) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_samples = 0;

        dataloader.reset();
        let num_batches = dataloader.num_batches();

        for (batch_idx, (images, labels)) in dataloader.enumerate() {
            // Reset tape for new computation graph
            Tape::reset();

            // Forward pass
            let logits = self.model.forward(&images);
            let loss = cross_entropy_loss(&logits, &labels);

            // Calculate accuracy. Counting directly avoids the ratio round trip
            // `(acc * batch_size) as usize`, whose truncation undercounted.
            let batch_size = images.shape()[0];
            total_correct += correct_count(&logits, &labels);
            total_samples += batch_size;

            // Backward pass
            loss.backward();

            // Update weights
            self.optimizer.step();
            self.optimizer.zero_grad();

            total_loss += loss.data()[0];

            // Print progress every 10 batches
            if batch_idx % 10 == 0 {
                print!(
                    "\rBatch [{}/{}] Loss: {:.4}",
                    batch_idx + 1,
                    num_batches,
                    loss.data()[0]
                );
            }
        }

        let avg_loss = total_loss / num_batches as f32;
        let avg_acc = total_correct as f32 / total_samples as f32;

        (avg_loss, avg_acc)
    }

    /// Evaluate on validation/test set
    pub fn evaluate(&self, dataloader: &mut DataLoader) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_samples = 0;

        dataloader.reset();
        let num_batches = dataloader.num_batches();

        // No gradient computation needed for evaluation. Without this guard the
        // model's parameters still require grad, so every batch appended
        // backward closures — each holding its inputs alive — to the tape, and
        // nothing cleared them until the next training epoch.
        let _guard = crate::tape::no_grad();

        for (images, labels) in dataloader {
            let logits = self.model.forward(&images);
            let loss = cross_entropy_loss(&logits, &labels);

            let batch_size = images.shape()[0];
            total_correct += correct_count(&logits, &labels);
            total_samples += batch_size;

            total_loss += loss.data()[0];
        }

        let avg_loss = total_loss / num_batches as f32;
        let avg_acc = if total_samples == 0 {
            0.0
        } else {
            total_correct as f32 / total_samples as f32
        };

        (avg_loss, avg_acc)
    }

    /// Main training loop
    pub fn fit(
        &mut self,
        train_loader: &mut DataLoader,
        val_loader: &mut DataLoader,
        epochs: usize,
        verbose: bool,
    ) {
        println!("Starting training for {} epochs", epochs);
        println!("{}", "=".repeat(60));

        let pb = if verbose {
            let pb = ProgressBar::new(epochs as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-")
            );
            Some(pb)
        } else {
            None
        };

        for epoch in 0..epochs {
            let epoch_start = Instant::now();

            if verbose {
                println!("\nEpoch {}/{}", epoch + 1, epochs);
            }

            // Training phase
            self.model.set_training(true);
            let (train_loss, train_acc) = self.train_epoch(train_loader);

            // Validation phase
            self.model.set_training(false);
            let (val_loss, val_acc) = self.evaluate(val_loader);
            self.model.set_training(true);

            // Update learning rate
            if let Some(scheduler) = &mut self.scheduler {
                scheduler.step(Some(val_loss));
                let new_lr = scheduler.get_lr();
                self.optimizer.set_lr(new_lr);
            }

            // Record metrics
            self.metrics.train_loss.push(train_loss);
            self.metrics.train_acc.push(train_acc);
            self.metrics.val_loss.push(val_loss);
            self.metrics.val_acc.push(val_acc);
            self.metrics
                .epoch_times
                .push(epoch_start.elapsed().as_secs_f32());

            // Print epoch summary
            if verbose {
                println!(
                    "\nEpoch {} - Train Loss: {:.4} | Train Acc: {:.2}% | Val Loss: {:.4} | Val Acc: {:.2}% | Time: {:.2}s",
                    epoch + 1,
                    train_loss,
                    train_acc * 100.0,
                    val_loss,
                    val_acc * 100.0,
                    self.metrics.epoch_times.last().unwrap()
                );

                if let Some(scheduler) = &self.scheduler {
                    println!("   Learning Rate: {:.6}", scheduler.get_lr());
                }
            }

            if let Some(ref pb) = pb {
                pb.inc(1);
            }

            // Early stopping check (optional)
            if let Some(threshold) = self.early_stop_val_acc
                && val_acc > threshold
            {
                println!(
                    "\nReached {:.1}% validation accuracy! Stopping early.",
                    threshold * 100.0
                );
                break;
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message("Training complete!");
        }

        // Print final summary
        self.metrics.plot_summary();
    }

    /// Save model checkpoint.
    ///
    /// Format: parameter count, then per parameter a shape line
    /// (`rank dim0 dim1 …`) followed by one value per line.
    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let params = self.model.parameters();
        // Unbuffered writes cost one syscall per float; MNIST MLP checkpoints
        // are ~100k of them.
        let mut file = BufWriter::new(File::create(path)?);

        // Simple format: write number of parameters, then each parameter's data
        writeln!(file, "{}", params.len())?;

        for param in params {
            let data = param.data();
            let shape = param.shape();

            // Write shape
            write!(file, "{}", shape.len())?;
            for dim in shape {
                write!(file, " {}", dim)?;
            }
            writeln!(file)?;

            // Write data
            for value in data.iter() {
                writeln!(file, "{}", value)?;
            }
        }

        file.flush()
    }

    /// Load parameters previously written by [`Trainer::save_checkpoint`] into
    /// this trainer's model, in `parameters()` order.
    ///
    /// The saver shipped without a counterpart, so checkpoints could be written
    /// but never restored.
    pub fn load_checkpoint(&mut self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{BufRead, BufReader, Error, ErrorKind};

        let file = BufReader::new(File::open(path)?);
        let mut lines = file.lines();
        let malformed = |msg: String| -> Error {
            Error::new(ErrorKind::InvalidData, format!("checkpoint: {msg}"))
        };

        let mut next = |what: &str| -> std::io::Result<String> {
            lines
                .next()
                .transpose()?
                .ok_or_else(|| malformed(format!("unexpected end of file reading {what}")))
        };

        let count: usize = next("parameter count")?
            .trim()
            .parse()
            .map_err(|e| malformed(format!("bad parameter count: {e}")))?;

        let params = self.model.parameters();
        if count != params.len() {
            return Err(malformed(format!(
                "holds {count} parameters but the model has {}",
                params.len()
            )));
        }

        for (i, param) in params.iter().enumerate() {
            let shape_line = next(&format!("shape of parameter {i}"))?;
            let dims: Vec<usize> = shape_line
                .split_whitespace()
                .skip(1) // leading rank
                .map(|d| d.parse::<usize>())
                .collect::<Result<_, _>>()
                .map_err(|e| malformed(format!("bad shape for parameter {i}: {e}")))?;

            if dims != param.shape() {
                return Err(malformed(format!(
                    "parameter {i} has shape {dims:?} but the model expects {:?}",
                    param.shape()
                )));
            }

            let numel: usize = dims.iter().product();
            let mut values = Vec::with_capacity(numel);
            for j in 0..numel {
                let value = next(&format!("value {j} of parameter {i}"))?;
                values.push(
                    value
                        .trim()
                        .parse::<f32>()
                        .map_err(|e| malformed(format!("bad value in parameter {i}: {e}")))?,
                );
            }

            param.data_mut().copy_from_slice(&values);
        }

        Ok(())
    }
}

/// Helper function to create and train a model quickly
pub fn quick_train_mnist(
    model: Box<dyn Module>,
    train_loader: &mut DataLoader,
    val_loader: &mut DataLoader,
    epochs: usize,
    learning_rate: f32,
) -> Metrics {
    let params = model.parameters();
    let optimizer = Adam::new(params, learning_rate, None, None, Some(1e-4));

    let scheduler = Box::new(crate::optim::StepLR::new(learning_rate, 10, 0.5));

    let mut trainer = Trainer::new(model, optimizer, Some(scheduler));
    trainer.fit(train_loader, val_loader, epochs, true);

    trainer.metrics
}

/// Utility function to test model on a few samples
pub fn test_samples(model: &dyn Module, dataloader: &mut DataLoader, num_samples: usize) {
    println!("\nTesting on {} samples:", num_samples);
    println!("{}", "-".repeat(40));

    dataloader.reset();
    let _guard = crate::tape::no_grad();

    if let Some((images, labels)) = dataloader.next() {
        let n = num_samples.min(images.shape()[0]);

        // Get predictions for first n samples
        let logits = model.forward(&images);
        let predictions = logits.argmax(Some(1));

        let pred_data = predictions.data();
        let label_data = labels.data();
        let image_data = images.data();

        for i in 0..n {
            let predicted = pred_data[i] as usize;
            let actual = label_data[i] as usize;

            println!(
                "Sample {}: Predicted = {}, Actual = {} {}",
                i,
                predicted,
                actual,
                if predicted == actual {
                    "CORRECT"
                } else {
                    "WRONG"
                }
            );

            // Optional: Print a mini visualization of the digit
            if i < 3 {
                print_digit(&image_data[i * 784..(i + 1) * 784]);
            }
        }
    }
}

/// ASCII visualization of MNIST digit
fn print_digit(pixels: &[f32]) {
    println!("\n");
    for row in 0..28 {
        for col in 0..28 {
            let pixel = pixels[row * 28 + col];
            let c = if pixel > 0.75 {
                '█'
            } else if pixel > 0.5 {
                '▓'
            } else if pixel > 0.25 {
                '▒'
            } else if pixel > 0.0 {
                '░'
            } else {
                ' '
            };
            print!("{}", c);
        }
        println!();
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::activation::ReLU;
    use crate::nn::{Linear, Sequential};

    #[test]
    fn test_trainer_basic() {
        // Create a simple model
        let model = Sequential::new(vec![
            Box::new(Linear::new(784, 128, true)),
            Box::new(ReLU),
            Box::new(Linear::new(128, 10, true)),
        ]);

        // Create mock data
        let images = Tensor::randn(&[100, 784]);
        let labels = Tensor::new((0..100).map(|i| (i % 10) as f32).collect(), &[100]);

        let dataset = crate::data::mnist::MNISTDataset {
            images,
            labels,
            train: true,
        };

        let mut train_loader = DataLoader::new(dataset, 32, true);

        // Test one epoch of training
        let params = model.parameters();
        let optimizer = Adam::new(params, 0.001, None, None, None);
        let mut trainer = Trainer::new(Box::new(model), optimizer, None);

        let (loss, acc) = trainer.train_epoch(&mut train_loader);

        assert!(loss > 0.0);
        assert!((0.0..=1.0).contains(&acc));
    }
}
