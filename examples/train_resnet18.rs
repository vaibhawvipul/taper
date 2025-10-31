// ResNet-18 training example on MNIST dataset
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::time::Instant;
use taper::{Tape, Tensor};
use taper::data::mnist::{DataLoader, MNISTDataset};
use taper::loss::{accuracy, cross_entropy_loss};
use taper::models::ResNet;
use taper::nn::Module;
use taper::optim::Adam;
use taper::train::Trainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ResNet-18 MNIST Training\n");

    // Load MNIST dataset
    println!("Loading MNIST dataset...");
    let train_dataset = MNISTDataset::new(true, None)?;
    let test_dataset = MNISTDataset::new(false, None)?;

    println!("Training set: {} samples", train_dataset.len());
    println!("Test set: {} samples\n", test_dataset.len());

    // Create data loaders
    let batch_size = 32; // Smaller batch for faster iteration during debugging
    let mut train_loader = DataLoader::new(train_dataset, batch_size, true);
    let mut test_loader = DataLoader::new(test_dataset, batch_size, false);

    // Build ResNet-18 model for MNIST (10 classes) with single-channel input
    println!("Building ResNet-18 model...");
    let model = ResNet::resnet18_single_channel(10);

    // Count parameters
    let params = model.parameters();
    let total_params: usize = params.iter().map(|p| p.data().len()).sum();
    println!("Total parameters: {}", total_params);

    // Create optimizer with higher learning rate for ResNet
    let learning_rate = 0.01; // Increased LR for faster convergence
    let optimizer = Adam::new(params, learning_rate, None, None, Some(0.0001));

    // Create learning rate scheduler
    let scheduler = Box::new(taper::optim::StepLR::new(learning_rate, 5, 0.5));
    
    // Create trainer
    let mut trainer = Trainer::new(Box::new(model), optimizer, Some(scheduler));

    // Training settings
    let epochs = 5; // Reduced epochs for faster testing  
    let log_interval = 1; // Log every batch for debugging

    println!("\nTraining Configuration:");
    println!("   Model: ResNet-18");
    println!("   Batch size: {}", batch_size);
    println!("   Learning rate: {}", learning_rate);
    println!("   Epochs: {}", epochs);
    println!("\n{}\n", "=".repeat(60));

    // Training loop
    let total_start = Instant::now();

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
            let batch_start = Instant::now();

            // Reshape and normalize images from [B, 784] to [B, 1, 28, 28] for single-channel ResNet
            let batch_size = images.shape()[0];
            let images_4d = images.reshape(&[batch_size, 1, 28, 28]);
            // Note: MNIST data loader already returns normalized data [0,1] range

            // Forward pass
            let logits = trainer.model.forward(&images_4d);

            // Compute loss
            let loss = cross_entropy_loss(&logits, &labels);

            // Compute accuracy
            let batch_acc = accuracy(&logits, &labels);
            train_correct += (batch_acc * labels.shape()[0] as f32) as usize;
            train_total += labels.shape()[0];

            // Backward pass
            loss.backward();

            // Gradient clipping to prevent explosion (relaxed from 1.0 to 10.0)
            let params = trainer.model.parameters();
            taper::ops::clip_grad_norm(&params, 10.0); // Clip gradients to max norm of 10.0

            // Update weights
            trainer.optimizer.step();
            trainer.optimizer.zero_grad();
            
            // Clear tape after backward pass
            Tape::reset();

            train_loss += loss.data()[0];

            // Progress update
            if (batch_idx + 1) % log_interval == 0 || batch_idx == num_batches - 1 {
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

            // Reshape for single-channel ResNet (already normalized by dataloader)
            let batch_size = images.shape()[0];
            let images_4d = images.reshape(&[batch_size, 1, 28, 28]);

            let logits = trainer.model.forward(&images_4d);
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
        if val_accuracy > 0.99 {
            println!(
                "Reached {:.2}% validation accuracy! Stopping early.",
                val_accuracy * 100.0
            );
            break;
        }
    }

    let total_time = total_start.elapsed();
    println!("\n{}", "=".repeat(60));
    println!(
        "Training Complete! Total time: {:.2}s",
        total_time.as_secs_f32()
    );

    // Test on a few samples
    println!("\nTesting ResNet-18 on sample images:");
    test_loader.reset();

    if let Some((images, labels)) = test_loader.next() {
        let batch_size = images.shape()[0].min(5);
        let images_4d = images.reshape(&[images.shape()[0], 1, 28, 28]);
        
        let predictions = trainer.model.forward(&images_4d);
        let pred_classes = predictions.argmax(Some(1));

        for i in 0..batch_size {
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


