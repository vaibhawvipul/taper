// examples/train_mnist.rs - Complete working MNIST training example

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::time::Instant;
use taper::Tape;
use taper::activation::ReLU;
use taper::data::mnist::{DataLoader, MNISTDataset};
use taper::loss::{accuracy, cross_entropy_loss};
use taper::nn::{Linear, Module, Sequential};
use taper::optim::Adam;
use taper::train::Trainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MNIST Neural Network Training\n");

    // Load MNIST dataset
    println!("Loading MNIST dataset...");
    let train_dataset = MNISTDataset::new(true, None)?;
    let test_dataset = MNISTDataset::new(false, None)?;

    println!("Training set: {} samples", train_dataset.len());
    println!("Test set: {} samples\n", test_dataset.len());

    // Create data loaders with smaller batch size for stability
    let batch_size = 256; // Smaller batch size to avoid dimension issues
    let mut train_loader = DataLoader::new(train_dataset, batch_size, true);
    let mut test_loader = DataLoader::new(test_dataset, batch_size, false);

    // Build model - Simple MLP
    println!("Building model...");
    let model = Sequential::new(vec![
        Box::new(Linear::new(784, 128, true)), // Input layer
        Box::new(ReLU),
        Box::new(Linear::new(128, 64, true)), // Hidden layer
        Box::new(ReLU),
        Box::new(Linear::new(64, 10, true)), // Output layer (10 classes)
    ]);

    // Get model parameters for optimizer
    let params = model.parameters();
    println!(
        "Total parameters: {}",
        params.iter().map(|p| p.data().len()).sum::<usize>()
    );

    // Create optimizer with conservative learning rate
    let learning_rate = 0.001;
    let optimizer = Adam::new(params, learning_rate, None, None, Some(0.0001));

    // Create trainer
    let mut trainer = Trainer::new(
        Box::new(model),
        optimizer,
        None, // No scheduler for now
    );

    // Training settings
    let epochs = 10; // Start with fewer epochs for testing

    println!("\nTraining Configuration:");
    println!("   Batch size: {}", batch_size);
    println!("   Learning rate: {}", learning_rate);
    println!("   Epochs: {}", epochs);
    println!("\n{}\n", "=".repeat(60));

    // Training loop with manual implementation for better control
    for epoch in 1..=epochs {
        let epoch_start = Instant::now();

        println!("Epoch {}/{}", epoch, epochs);

        // Training phase
        trainer
            .model
            .parameters()
            .iter()
            .for_each(|p| p.zero_grad());

        let mut train_loss = 0.0;
        let mut train_correct = 0;
        let mut train_total = 0;

        train_loader.reset();
        let num_batches = train_loader.num_batches();

        for (batch_idx, (images, labels)) in train_loader.by_ref().enumerate() {
            // Create new tape for this batch
            Tape::reset();

            // Ensure shapes are correct
            assert_eq!(
                images.shape(),
                &[batch_size.min(images.shape()[0]), 784],
                "Image batch shape mismatch"
            );

            // Forward pass
            let logits = trainer.model.forward(&images);

            // Ensure output shape is correct
            assert_eq!(logits.shape()[1], 10, "Output should have 10 classes");

            // Compute loss
            let loss = cross_entropy_loss(&logits, &labels);

            // Compute accuracy
            let batch_acc = accuracy(&logits, &labels);
            train_correct += (batch_acc * labels.shape()[0] as f32) as usize;
            train_total += labels.shape()[0];

            // Backward pass
            loss.backward();

            // Update weights
            trainer.optimizer.step();
            trainer.optimizer.zero_grad();

            train_loss += loss.data()[0];

            // Progress update
            if (batch_idx + 1) % 100 == 0 || batch_idx == num_batches - 1 {
                print!(
                    "\r   Batch [{}/{}] Loss: {:.4}, Acc: {:.2}%",
                    batch_idx + 1,
                    num_batches,
                    loss.data()[0],
                    100.0 * train_correct as f32 / train_total as f32
                );
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
        }

        let avg_train_loss = train_loss / num_batches as f32;
        let train_accuracy = train_correct as f32 / train_total as f32;

        println!(); // New line after progress bar

        // Validation phase
        print!("   Evaluating...");
        use std::io::{self, Write};
        io::stdout().flush().unwrap();

        let mut val_loss = 0.0;
        let mut val_correct = 0;
        let mut val_total = 0;

        test_loader.reset();
        let num_val_batches = test_loader.num_batches();

        for (images, labels) in &mut test_loader {
            Tape::reset();

            // No tape needed for evaluation
            let logits = trainer.model.forward(&images);
            let loss = cross_entropy_loss(&logits, &labels);

            let batch_acc = accuracy(&logits, &labels);
            val_correct += (batch_acc * labels.shape()[0] as f32) as usize;
            val_total += labels.shape()[0];

            val_loss += loss.data()[0];
        }

        let avg_val_loss = val_loss / num_val_batches as f32;
        let val_accuracy = val_correct as f32 / val_total as f32;

        let epoch_time = epoch_start.elapsed().as_secs_f32();

        // Print epoch summary
        println!("\rEpoch {} complete:", epoch);
        println!(
            "   Train Loss: {:.4} | Train Acc: {:.2}%",
            avg_train_loss,
            train_accuracy * 100.0
        );
        println!(
            "   Val Loss: {:.4}   | Val Acc: {:.2}%",
            avg_val_loss,
            val_accuracy * 100.0
        );
        println!("   Time: {:.2}s", epoch_time);
        println!();

        // Early stopping if we reach good accuracy
        if val_accuracy > 0.98 {
            println!(
                "🎉 Reached {:.2}% validation accuracy! Stopping early.",
                val_accuracy * 100.0
            );
            break;
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Training Complete!");

    // Test on a few samples
    println!("\nTesting on sample images:");
    test_loader.reset();

    if let Some((images, labels)) = test_loader.next() {
        let predictions = trainer.model.forward(&images);
        let pred_classes = predictions.argmax(Some(1));

        for i in 0..5.min(images.shape()[0]) {
            let predicted = pred_classes.data()[i] as u8;
            let actual = labels.data()[i] as u8;

            println!(
                "Sample {}: Predicted={}, Actual={} {}",
                i + 1,
                predicted,
                actual,
                if predicted == actual {
                    "Correct"
                } else {
                    "Wrong"
                }
            );
        }
    }

    Ok(())
}
