use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::time::Instant;
use taper::Tape;
use taper::activation::ReLU;
use taper::data::mnist::{DataLoader, MNISTDataset};
use taper::loss::{accuracy, cross_entropy_loss};
use taper::nn::{Conv2dReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Module, Sequential, Flatten};
use taper::optim::Adam;
use taper::train::Trainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CNN MNIST Training with Performance Optimization\n");

    // Load MNIST dataset
    println!("Loading MNIST dataset...");
    let train_dataset = MNISTDataset::new(true, None)?;
    let test_dataset = MNISTDataset::new(false, None)?;

    println!("Training set: {} samples", train_dataset.len());
    println!("Test set: {} samples\n", test_dataset.len());

    // Create data loaders with optimal batch size for CNN
    let batch_size = 256; // Optimal for CNN training
    let mut train_loader = DataLoader::new(train_dataset, batch_size, true);
    let mut test_loader = DataLoader::new(test_dataset, batch_size, false);

    // Build CNN model optimized for MNIST
    println!("Building optimized CNN model...");
    
    // High-performance CNN with fusion operations
    let model = Sequential::new(vec![
        // First conv block with fused conv+relu
        Box::new(Conv2dReLU::new(
            1, 32, (3, 3),           // 28x28x1 -> 28x28x32
            Some((1, 1)),            // stride
            Some((1, 1)),            // padding
            None, None, true
        )),
        Box::new(Conv2dReLU::new(
            32, 32, (3, 3),          // 28x28x32 -> 28x28x32
            Some((1, 1)),
            Some((1, 1)),
            None, None, true
        )),
        Box::new(MaxPool2d::new((2, 2), Some((2, 2)), None)), // 28x28x32 -> 14x14x32

        // Second conv block
        Box::new(Conv2dReLU::new(
            32, 64, (3, 3),          // 14x14x32 -> 14x14x64
            Some((1, 1)),
            Some((1, 1)),
            None, None, true
        )),
        Box::new(Conv2dReLU::new(
            64, 64, (3, 3),          // 14x14x64 -> 14x14x64
            Some((1, 1)),
            Some((1, 1)),
            None, None, true
        )),
        Box::new(MaxPool2d::new((2, 2), Some((2, 2)), None)), // 14x14x64 -> 7x7x64

        // Third conv block
        Box::new(Conv2dReLU::new(
            64, 128, (3, 3),         // 7x7x64 -> 7x7x128
            Some((1, 1)),
            Some((1, 1)),
            None, None, true
        )),

        // Global average pooling instead of flatten to reduce parameters
        Box::new(AdaptiveAvgPool2d::global()), // 7x7x128 -> 1x1x128
        Box::new(Flatten::new(Some(1))),       // 128


        // Classifier
        Box::new(Linear::new(128, 128, true)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 64, true)),
        Box::new(ReLU),
        Box::new(Linear::new(64, 10, true)),   // 10 classes
    ]);

    // Count parameters
    let params = model.parameters();
    let total_params: usize = params.iter().map(|p| p.data().len()).sum();
    println!("Total parameters: {}", total_params);

    // Create optimizer with learning rate scheduling
    let mut learning_rate = 0.01;  // Much lower starting LR
    let optimizer = Adam::new(params, learning_rate, None, None, Some(0.0001));

    // Create trainer
    let mut trainer = Trainer::new(
        Box::new(model),
        optimizer,
        None,
    );

    // Training settings optimized for fast convergence
    let epochs = 50;
    let log_interval = 50;

    println!("\nTraining Configuration:");
    println!("   Batch size: {}", batch_size);
    println!("   Learning rate: {}", learning_rate);
    println!("   Epochs: {}", epochs);
    println!("\n{}\n", "=".repeat(60));

    // Training loop with performance monitoring
    let total_start = Instant::now();
    
    for epoch in 1..=epochs {
        let epoch_start = Instant::now();

        println!("Epoch {}/{}", epoch, epochs);

        // Decay learning rate
        if epoch % 5 == 0 && epoch >= 5 {
            learning_rate *= 0.8;
            println!("   Reducing learning rate to {:.6}", learning_rate);
            trainer.optimizer.set_lr(learning_rate);
        }

        // Training phase
        trainer.model.parameters().iter().for_each(|p| p.zero_grad());

        let mut train_loss = 0.0;
        let mut train_correct = 0;
        let mut train_total = 0;
        let mut batch_times = Vec::new();

        train_loader.reset();
        let num_batches = train_loader.num_batches();

        for (batch_idx, (images, labels)) in train_loader.by_ref().enumerate() {
            let batch_start = Instant::now();

            // Clear tape for new computation
            Tape::reset();

            // Reshape images from [B, 784] to [B, 1, 28, 28] for CNN
            let batch_size = images.shape()[0];
            let images_4d = images.reshape(&[batch_size, 1, 28, 28]);

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

            // Update weights
            trainer.optimizer.step();
            trainer.optimizer.zero_grad();

            train_loss += loss.data()[0];
            batch_times.push(batch_start.elapsed().as_millis());

            // Progress update
            if (batch_idx + 1) % log_interval == 0 || batch_idx == num_batches - 1 {
                let avg_batch_time = batch_times.iter().sum::<u128>() / batch_times.len() as u128;
                print!(
                    "\r   Batch [{}/{}] Loss: {:.4}, Acc: {:.2}%, Avg Batch Time: {}ms",
                    batch_idx + 1,
                    num_batches,
                    loss.data()[0],
                    100.0 * train_correct as f32 / train_total as f32,
                    avg_batch_time
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

        let val_start = Instant::now();
        let mut val_loss = 0.0;
        let mut val_correct = 0;
        let mut val_total = 0;

        test_loader.reset();
        let num_val_batches = test_loader.num_batches();

        for (images, labels) in &mut test_loader {
            Tape::reset();

            // Reshape for CNN
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
        let val_time = val_start.elapsed().as_millis();

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
        println!("   Time: {:.2}s (Val: {}ms)", epoch_time, val_time);
        
        // Performance metrics
        let throughput = train_total as f32 / epoch_time;
        println!("   Throughput: {:.0} samples/sec", throughput);
        println!();

        // Early stopping if we reach excellent accuracy
        if val_accuracy > 0.995 {
            println!(
                "Reached {:.2}% validation accuracy! Stopping early.",
                val_accuracy * 100.0
            );
            break;
        }
    }

    let total_time = total_start.elapsed();
    println!("\n{}", "=".repeat(60));
    println!("Training Complete! Total time: {:.2}s", total_time.as_secs_f32());

    // Final test on some samples
    println!("\nTesting CNN on sample images:");
    test_loader.reset();

    if let Some((images, labels)) = test_loader.next() {
        let batch_size = images.shape()[0].min(10);
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
                if predicted == actual { "Correct" } else { "False" }
            );
        }
    }
    Ok(())
}

// Performance benchmark function
#[allow(dead_code)]
fn benchmark_conv_performance() {
    println!("Benchmarking convolution performance...");

    // Create test data
    let input = taper::Tensor::randn(&[32, 64, 56, 56]); // Typical ResNet intermediate size
    let weight = taper::Tensor::randn(&[128, 64, 3, 3]);
    let bias = Some(taper::Tensor::randn(&[128]));

    let num_iterations = 100;
    let start = Instant::now();

    for _ in 0..num_iterations {
        let _output = input.conv2d(
            &weight,
            bias.as_ref(),
            (1, 1), // stride
            (1, 1), // padding
            (1, 1), // dilation
        );
    }

    let elapsed = start.elapsed();
    let avg_time = elapsed.as_millis() as f32 / num_iterations as f32;
    let throughput = (32.0 * 128.0 * 56.0 * 56.0 * 64.0 * 3.0 * 3.0) / (avg_time / 1000.0); // FLOPS

    println!("Average conv2d time: {:.2}ms", avg_time);
    println!("Estimated throughput: {:.2} GFLOPS", throughput / 1e9);
}
