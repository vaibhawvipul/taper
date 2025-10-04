use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::time::Instant;
use taper::QuantizationConfig;
use taper::Tape;
use taper::activation::ReLU;
use taper::data::mnist::{DataLoader, MNISTDataset};
use taper::loss::{accuracy, cross_entropy_loss};
use taper::nn::{AdaptiveAvgPool2d, Conv2dReLU, Flatten, Linear, MaxPool2d, Module, Sequential};
use taper::optim::Adam;
use taper::train::Trainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Model Quantization Example");
    println!("This example shows how to train a model and then quantize it for inference.\n");

    // Load MNIST dataset
    println!("Loading MNIST dataset...");
    let train_dataset = MNISTDataset::new(true, None)?;
    let test_dataset = MNISTDataset::new(false, None)?;

    println!("Training set: {} samples", train_dataset.len());
    println!("Test set: {} samples\n", test_dataset.len());

    // Create data loaders
    let mut train_loader = DataLoader::new(train_dataset, 64, true);
    let mut test_loader = DataLoader::new(test_dataset, 64, false);

    // Build CNN model
    println!("Building CNN model...");
    let model = Sequential::new(vec![
        // First conv block
        Box::new(Conv2dReLU::new(
            1,
            32,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            None,
            None,
            true,
        )),
        Box::new(Conv2dReLU::new(
            32,
            32,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            None,
            None,
            true,
        )),
        Box::new(MaxPool2d::new((2, 2), Some((2, 2)), None)), // 28x28 -> 14x14
        // Second conv block
        Box::new(Conv2dReLU::new(
            32,
            64,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            None,
            None,
            true,
        )),
        Box::new(Conv2dReLU::new(
            64,
            64,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            None,
            None,
            true,
        )),
        Box::new(MaxPool2d::new((2, 2), Some((2, 2)), None)), // 14x14 -> 7x7
        // Third conv block
        Box::new(Conv2dReLU::new(
            64,
            128,
            (3, 3),
            Some((1, 1)),
            Some((1, 1)),
            None,
            None,
            true,
        )),
        // Global average pooling
        Box::new(AdaptiveAvgPool2d::global()), // 7x7x128 -> 1x1x128
        Box::new(Flatten::new(Some(1))),       // 128
        // Classifier
        Box::new(Linear::new(128, 128, true)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 64, true)),
        Box::new(ReLU),
        Box::new(Linear::new(64, 10, true)), // 10 classes
    ]);

    // Count parameters
    let params = model.parameters();
    let total_params: usize = params.iter().map(|p| p.data().len()).sum();
    println!("Total parameters: {}", total_params);

    // Create optimizer
    let optimizer = Adam::new(params, 0.01, None, None, Some(0.0001));

    // Create trainer
    let mut trainer = Trainer::new(Box::new(model), optimizer, None);

    println!("\n{}\n", "=".repeat(60));
    println!("Step 1: Training the model (10 epochs)...");

    // Quick training (10 epochs)
    for epoch in 1..=2 {
        let epoch_start = Instant::now();

        println!("Epoch {}/10", epoch);

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
    println!("Step 2: Quantizing the trained model...");

    // Create quantization configs
    let int8_config = QuantizationConfig::int8(true);
    let float16_config = QuantizationConfig::float16(true);

    // Quantize the model
    println!("Quantizing model to Int8...");
    let quantize_start = Instant::now();
    let quantized_model_int8 = trainer.model.quantize(&int8_config);
    let quantize_time = quantize_start.elapsed();
    println!(
        "Int8 quantization completed in {:.2}ms",
        quantize_time.as_millis()
    );

    println!("Quantizing model to Float16...");
    let quantize_start = Instant::now();
    let quantized_model_f16 = trainer.model.quantize(&float16_config);
    let quantize_time = quantize_start.elapsed();
    println!(
        "Float16 quantization completed in {:.2}ms",
        quantize_time.as_millis()
    );

    println!("\n{}\n", "=".repeat(60));
    println!("Step 3: Testing quantized models on full test set...");

    // Diagnostic: Check if Int8 and Float16 produce different outputs
    println!("\nDiagnostic: Comparing Int8 and Float16 outputs...");
    test_loader.reset();
    let (sample_images, _sample_labels) = test_loader.next().unwrap();
    let sample_batch_size = sample_images.shape()[0];
    let sample_images_4d = sample_images.reshape(&[sample_batch_size, 1, 28, 28]);

    let int8_sample = quantized_model_int8.forward(&sample_images_4d);
    let f16_sample = quantized_model_f16.forward(&sample_images_4d);

    let int8_data = int8_sample.data();
    let f16_data = f16_sample.data();
    let mut max_diff = 0.0f32;
    let mut total_diff = 0.0f32;

    for (a, b) in int8_data.iter().zip(f16_data.iter()) {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        total_diff += diff;
    }

    println!("Max difference: {:.6}", max_diff);
    println!("Avg difference: {:.6}", total_diff / int8_data.len() as f32);

    if max_diff < 1e-6 {
        println!(
            "WARNING: Outputs are nearly identical! Quantization may not be working correctly."
        );
    } else {
        println!("Outputs differ correctly.\n");
    }

    // Test original model on full test set
    println!("Testing original model on full test set...");
    let mut original_correct = 0;
    let mut original_total = 0;
    let test_start = Instant::now();

    test_loader.reset();
    for (images, labels) in &mut test_loader {
        Tape::reset();
        let batch_size = images.shape()[0];
        let images_4d = images.reshape(&[batch_size, 1, 28, 28]);
        let predictions = trainer.model.forward(&images_4d);
        let acc = accuracy(&predictions, &labels);
        original_correct += (acc * batch_size as f32) as usize;
        original_total += batch_size;
    }
    let original_time = test_start.elapsed();
    let original_acc = original_correct as f32 / original_total as f32;

    println!(
        "Original - Accuracy: {:.2}% ({}/{}), Time: {}ms",
        original_acc * 100.0,
        original_correct,
        original_total,
        original_time.as_millis()
    );

    // Test Int8 quantized model on full test set
    println!("Testing Int8 quantized model on full test set...");
    let mut int8_correct = 0;
    let mut int8_total = 0;
    let test_start = Instant::now();

    test_loader.reset();
    for (images, labels) in &mut test_loader {
        Tape::reset();
        let batch_size = images.shape()[0];
        let images_4d = images.reshape(&[batch_size, 1, 28, 28]);
        let predictions = quantized_model_int8.forward(&images_4d);
        let acc = accuracy(&predictions, &labels);
        int8_correct += (acc * batch_size as f32) as usize;
        int8_total += batch_size;
    }
    let int8_time = test_start.elapsed();
    let int8_acc = int8_correct as f32 / int8_total as f32;

    println!(
        "Int8 - Accuracy: {:.2}% ({}/{}), Time: {}ms",
        int8_acc * 100.0,
        int8_correct,
        int8_total,
        int8_time.as_millis()
    );

    // Test Float16 quantized model on full test set
    println!("Testing Float16 quantized model on full test set...");
    let mut f16_correct = 0;
    let mut f16_total = 0;
    let test_start = Instant::now();

    test_loader.reset();
    for (images, labels) in &mut test_loader {
        Tape::reset();
        let batch_size = images.shape()[0];
        let images_4d = images.reshape(&[batch_size, 1, 28, 28]);
        let predictions = quantized_model_f16.forward(&images_4d);
        let acc = accuracy(&predictions, &labels);
        f16_correct += (acc * batch_size as f32) as usize;
        f16_total += batch_size;
    }
    let f16_time = test_start.elapsed();
    let f16_acc = f16_correct as f32 / f16_total as f32;

    println!(
        "Float16 - Accuracy: {:.2}% ({}/{}), Time: {}ms",
        f16_acc * 100.0,
        f16_correct,
        f16_total,
        f16_time.as_millis()
    );

    println!("\n{}\n", "=".repeat(60));
    println!("Quantization Summary:");
    println!(
        "   Original model:     {:.2}% accuracy, {}ms",
        original_acc * 100.0,
        original_time.as_millis()
    );
    println!(
        "   Int8 quantized:     {:.2}% accuracy, {}ms (Δ {:.2}%)",
        int8_acc * 100.0,
        int8_time.as_millis(),
        (int8_acc - original_acc) * 100.0
    );
    println!(
        "   Float16 quantized:  {:.2}% accuracy, {}ms (Δ {:.2}%)",
        f16_acc * 100.0,
        f16_time.as_millis(),
        (f16_acc - original_acc) * 100.0
    );

    // Show if accuracies are suspiciously identical
    if (int8_acc - f16_acc).abs() < 1e-6 {
        println!("\nNOTE: Int8 and Float16 have identical accuracy. This may indicate:");
        println!("  - Small test set size leading to identical results");
        println!("  - Shared cached weights (check implementation)");
    }

    println!("\nNote: This is storage quantization for model compression.");
    println!("Inference speed is similar to original (or slightly slower).");
    println!("Benefits: 4x smaller model size, minimal accuracy loss.");

    let original_size = total_params * 4; // f32 = 4 bytes
    let int8_size = total_params * 1; // int8 = 1 byte
    let f16_size = total_params * 2; // float16 = 2 bytes

    println!("Model size comparison:");
    println!(
        "  Original (f32):  {:.2} MB",
        original_size as f32 / 1_000_000.0
    );
    println!(
        "  Int8 quantized:  {:.2} MB ({}x smaller)",
        int8_size as f32 / 1_000_000.0,
        original_size / int8_size
    );
    println!(
        "  Float16:         {:.2} MB ({}x smaller)",
        f16_size as f32 / 1_000_000.0,
        original_size / f16_size
    );

    Ok(())
}
