//! Quantization-Aware Training (QAT) Example
//!
//! This example demonstrates how to use QAT in the Taper library.
//! It shows the complete workflow from training with fake quantization
//! to deploying a quantized model.

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::time::Instant;
use taper::Tape;
use taper::activation::ReLU;
use taper::data::mnist::{DataLoader, MNISTDataset};
use taper::loss::{accuracy, cross_entropy_loss};
use taper::nn::{AdaptiveAvgPool2d, Flatten, MaxPool2d, Module};
use taper::optim::Adam;
use taper::quantization::qat_manager::global;
use taper::quantization::{QATConfig, QATConv2d, QATLinear, QATSequential};
use taper::train::Trainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Quantization-Aware Training (QAT) Example");
    println!("This example demonstrates QAT training and deployment.\n");

    // Load MNIST dataset
    println!("Loading MNIST dataset...");
    let train_dataset = MNISTDataset::new(true, None)?;
    let test_dataset = MNISTDataset::new(false, None)?;

    println!("Training set: {} samples", train_dataset.len());
    println!("Test set: {} samples\n", test_dataset.len());

    // Create data loaders
    let mut train_loader = DataLoader::new(train_dataset, 64, true);
    let mut test_loader = DataLoader::new(test_dataset, 64, false);

    // Create QAT configuration
    let qat_config = QATConfig::int8(0.001, 5)
        .with_per_channel(false)
        .with_symmetric(true)
        .with_observers(true);

    println!("QAT Configuration:");
    println!("  Quantization Type: {:?}", qat_config.quant_type());
    println!("  Learning Rate: {:.4}", qat_config.learning_rate);
    println!("  Warmup Epochs: {}", qat_config.warmup_epochs);
    println!("  Per-Channel: {}", qat_config.per_channel);
    println!("  Symmetric: {}", qat_config.symmetric);
    println!();

    // Build QAT-aware CNN model
    println!("Building QAT-aware CNN model...");
    let model = QATSequential::new(
        vec![
            // First conv block
            Box::new(QATConv2d::new(
                1,
                32,
                (3, 3),
                Some((1, 1)),
                Some((1, 1)),
                None,
                None,
                true,
                &qat_config,
                Some("conv1".to_string()),
            )),
            Box::new(QATConv2d::new(
                32,
                32,
                (3, 3),
                Some((1, 1)),
                Some((1, 1)),
                None,
                None,
                true,
                &qat_config,
                Some("conv2".to_string()),
            )),
            Box::new(MaxPool2d::new((2, 2), Some((2, 2)), None)), // 28x28 -> 14x14
            // Second conv block
            Box::new(QATConv2d::new(
                32,
                64,
                (3, 3),
                Some((1, 1)),
                Some((1, 1)),
                None,
                None,
                true,
                &qat_config,
                Some("conv3".to_string()),
            )),
            Box::new(QATConv2d::new(
                64,
                64,
                (3, 3),
                Some((1, 1)),
                Some((1, 1)),
                None,
                None,
                true,
                &qat_config,
                Some("conv4".to_string()),
            )),
            Box::new(MaxPool2d::new((2, 2), Some((2, 2)), None)), // 14x14 -> 7x7
            // Third conv block
            Box::new(QATConv2d::new(
                64,
                128,
                (3, 3),
                Some((1, 1)),
                Some((1, 1)),
                None,
                None,
                true,
                &qat_config,
                Some("conv5".to_string()),
            )),
            // Global average pooling
            Box::new(AdaptiveAvgPool2d::global()), // 7x7x128 -> 1x1x128
            Box::new(Flatten::new(Some(1))),       // 128
            // Classifier
            Box::new(QATLinear::new(
                128,
                128,
                true,
                &qat_config,
                Some("linear1".to_string()),
            )),
            Box::new(ReLU),
            Box::new(QATLinear::new(
                128,
                64,
                true,
                &qat_config,
                Some("linear2".to_string()),
            )),
            Box::new(ReLU),
            Box::new(QATLinear::new(
                64,
                10,
                true,
                &qat_config,
                Some("linear3".to_string()),
            )), // 10 classes
        ],
        qat_config.clone(),
        Some("main_model".to_string()),
    );

    // Count parameters
    let params = model.parameters();
    let total_params: usize = params.iter().map(|p| p.data().len()).sum();
    println!("Total parameters: {}", total_params);

    // Create optimizer
    let optimizer = Adam::new(params, 0.01, None, None, Some(0.0001));

    // Create trainer
    let mut trainer = Trainer::new(Box::new(model), optimizer, None);

    println!("\n{}\n", "=".repeat(60));
    println!("Step 1: QAT Training (3 epochs)...");

    // Enable QAT globally
    global::enable_qat();
    global::set_training_mode(true);

    // QAT training loop
    for epoch in 1..=3 {
        let epoch_start = Instant::now();

        println!("Epoch {}/3", epoch);

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
            Tape::reset();

            // Reshape images from [B, 784] to [B, 1, 28, 28] for CNN
            let batch_size = images.shape()[0];
            let images_4d = images.reshape(&[batch_size, 1, 28, 28]);

            // Forward pass (with fake quantization)
            let logits = trainer.model.forward(&images_4d);

            // Compute loss
            let loss = cross_entropy_loss(&logits, &labels);

            // Compute accuracy
            let batch_acc = accuracy(&logits, &labels);
            train_correct += (batch_acc * labels.shape()[0] as f32) as usize;
            train_total += labels.shape()[0];

            // Backward pass (with straight-through estimator)
            loss.backward();

            // Update weights
            trainer.optimizer.step();
            trainer.optimizer.zero_grad();

            train_loss += loss.data()[0];

            // Progress update
            if (batch_idx + 1) % 50 == 0 || batch_idx == num_batches - 1 {
                print!(
                    "\r   Batch [{}/{}] Loss: {:.4}, Acc: {:.2}%",
                    batch_idx + 1,
                    num_batches,
                    loss.data()[0],
                    100.0 * train_correct as f32 / train_total as f32,
                );
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
        }

        let avg_train_loss = train_loss / num_batches as f32;
        let train_accuracy = train_correct as f32 / train_total as f32;

        println!(); // New line after progress bar

        // Quick validation
        print!("   Evaluating...");
        use std::io::{self, Write};
        io::stdout().flush().unwrap();

        let mut val_correct = 0;
        let mut val_total = 0;

        test_loader.reset();
        for (images, labels) in &mut test_loader {
            Tape::reset();

            let batch_size = images.shape()[0];
            let images_4d = images.reshape(&[batch_size, 1, 28, 28]);

            let logits = trainer.model.forward(&images_4d);
            let batch_acc = accuracy(&logits, &labels);
            val_correct += (batch_acc * labels.shape()[0] as f32) as usize;
            val_total += labels.shape()[0];
        }

        let val_accuracy = val_correct as f32 / val_total as f32;
        let epoch_time = epoch_start.elapsed().as_secs_f32();

        println!("\rEpoch {} complete:", epoch);
        println!(
            "   Train Loss: {:.4} | Train Acc: {:.2}%",
            avg_train_loss,
            train_accuracy * 100.0
        );
        println!(
            "   Val Acc: {:.2}% | Time: {:.2}s",
            val_accuracy * 100.0,
            epoch_time
        );
        println!();
    }

    println!("{}\n", "=".repeat(60));
    println!("Step 2: Converting to quantized model...");

    // Switch to evaluation mode (disable fake quantization)
    global::set_training_mode(false);

    // Test the model in evaluation mode (no fake quantization)
    println!("Testing model in evaluation mode...");
    test_loader.reset();
    let (images, labels) = test_loader.next().unwrap();
    let batch_size = images.shape()[0];
    let images_4d = images.reshape(&[batch_size, 1, 28, 28]);

    let test_start = Instant::now();
    let eval_predictions = trainer.model.forward(&images_4d);
    let eval_time = test_start.elapsed();

    let eval_acc = accuracy(&eval_predictions, &labels);
    println!(
        "Evaluation mode - Accuracy: {:.2}%, Time: {:.2}ms",
        eval_acc * 100.0,
        eval_time.as_millis()
    );

    // Show QAT status
    let status = global::get_status();
    println!("\nQAT Status:");
    println!("  Global QAT Enabled: {}", status.global_enabled);
    println!("  Training Mode: {}", status.training_mode);
    println!("  Module Count: {}", status.module_count);
    println!("  Enabled Modules: {}", status.enabled_modules);
    println!("  Enabled Percentage: {:.1}%", status.enabled_percentage());

    // Show some sample predictions
    println!("\nSample predictions (first 5):");
    let pred_classes = eval_predictions.argmax(Some(1));
    for i in 0..5.min(batch_size) {
        let predicted = pred_classes.data()[i] as u8;
        let actual = labels.data()[i] as u8;
        println!(
            "   Sample {}: Predicted={}, Actual={} {}",
            i + 1,
            predicted,
            actual,
            if predicted == actual {
                "CORRECT"
            } else {
                "WRONG"
            }
        );
    }

    println!("\n{}\n", "=".repeat(60));
    println!("QAT Example completed successfully!");
    println!("The model has been trained with fake quantization and is ready for deployment.");

    Ok(())
}
