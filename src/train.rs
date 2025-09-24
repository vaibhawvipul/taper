use crate::data::mnist::DataLoader;
use crate::loss::{accuracy, cross_entropy_loss};
use crate::optim::{Adam, LRScheduler};
use crate::{Tape, nn::Module};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

/// Training metrics tracking
#[derive(Clone)]
pub struct Metrics {
    pub train_loss: Vec<f32>,
    pub train_acc: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub val_acc: Vec<f32>,
    pub epoch_times: Vec<f32>,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            train_loss: Vec::new(),
            train_acc: Vec::new(),
            val_loss: Vec::new(),
            val_acc: Vec::new(),
            epoch_times: Vec::new(),
        }
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

            if self.epoch_times.len() > 0 {
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
        }
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

            // Calculate accuracy
            let acc = accuracy(&logits, &labels);
            let batch_size = images.shape()[0];
            total_correct += (acc * batch_size as f32) as usize;
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

        // No gradient computation needed for evaluation
        for (images, labels) in dataloader {
            let logits = self.model.forward(&images);
            let loss = cross_entropy_loss(&logits, &labels);

            let acc = accuracy(&logits, &labels);
            let batch_size = images.shape()[0];
            total_correct += (acc * batch_size as f32) as usize;
            total_samples += batch_size;

            total_loss += loss.data()[0];
        }

        let avg_loss = total_loss / num_batches as f32;
        let avg_acc = total_correct as f32 / total_samples as f32;

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
            let (train_loss, train_acc) = self.train_epoch(train_loader);

            // Validation phase
            let (val_loss, val_acc) = self.evaluate(val_loader);

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
            if val_acc > 0.99 {
                println!("\nReached 99% validation accuracy! Stopping early.");
                break;
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message("Training complete!");
        }

        // Print final summary
        self.metrics.plot_summary();
    }

    /// Save model checkpoint (basic version - can be enhanced)
    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let params = self.model.parameters();
        let mut file = File::create(path)?;

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
pub fn test_samples(model: &Box<dyn Module>, dataloader: &mut DataLoader, num_samples: usize) {
    println!("\nTesting on {} samples:", num_samples);
    println!("{}", "-".repeat(40));

    dataloader.reset();

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
                if predicted == actual { "CORRECT" } else { "WRONG" }
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
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
