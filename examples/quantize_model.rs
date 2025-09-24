use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::time::Instant;
use taper::Tape;
use taper::activation::ReLU;
use taper::data::mnist::{DataLoader, MNISTDataset};
use taper::loss::{accuracy, cross_entropy_loss};
use taper::nn::{Conv2dReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Module, Sequential, Flatten, QuantizedModule};
use taper::optim::Adam;
use taper::train::Trainer;
use taper::{QuantizationConfig, quantization::QuantizationType};

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
        Box::new(Conv2dReLU::new(1, 32, (3, 3), Some((1, 1)), Some((1, 1)), None, None, true)),
        Box::new(Conv2dReLU::new(32, 32, (3, 3), Some((1, 1)), Some((1, 1)), None, None, true)),
        Box::new(MaxPool2d::new((2, 2), Some((2, 2)), None)), // 28x28 -> 14x14

        // Second conv block
        Box::new(Conv2dReLU::new(32, 64, (3, 3), Some((1, 1)), Some((1, 1)), None, None, true)),
        Box::new(Conv2dReLU::new(64, 64, (3, 3), Some((1, 1)), Some((1, 1)), None, None, true)),
        Box::new(MaxPool2d::new((2, 2), Some((2, 2)), None)), // 14x14 -> 7x7

        // Third conv block
        Box::new(Conv2dReLU::new(64, 128, (3, 3), Some((1, 1)), Some((1, 1)), None, None, true)),

        // Global average pooling
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

    // Create optimizer
    let optimizer = Adam::new(params, 0.01, None, None, Some(0.0001));

    // Create trainer
    let mut trainer = Trainer::new(
        Box::new(model),
        optimizer,
        None,
    );

    println!("\n{}\n", "=".repeat(60));
    println!("Step 1: Training the model (2 epochs)...");

    // Quick training (2 epochs)
    for epoch in 1..=2 {
        let epoch_start = Instant::now();

        println!("Epoch {}/2", epoch);

        // Training phase
        trainer.model.parameters().iter().for_each(|p| p.zero_grad());

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
    println!("Int8 quantization completed in {:.2}ms", quantize_time.as_millis());

    println!("Quantizing model to Float16...");
    let quantize_start = Instant::now();
    let quantized_model_f16 = trainer.model.quantize(&float16_config);
    let quantize_time = quantize_start.elapsed();
    println!("Float16 quantization completed in {:.2}ms", quantize_time.as_millis());

    println!("\n{}\n", "=".repeat(60));
    println!("Step 3: Testing quantized models...");

    // Test original model
    println!("Testing original model...");
    test_loader.reset();
    let (images, labels) = test_loader.next().unwrap();
    let batch_size = images.shape()[0];
    let images_4d = images.reshape(&[batch_size, 1, 28, 28]);

    let test_start = Instant::now();
    let original_predictions = trainer.model.forward(&images_4d);
    let original_time = test_start.elapsed();

    let original_acc = accuracy(&original_predictions, &labels);
    println!("Original model - Accuracy: {:.2}%, Time: {:.2}ms", 
             original_acc * 100.0, original_time.as_millis());

    // Test Int8 quantized model
    println!("Testing Int8 quantized model...");
    let test_start = Instant::now();
    let int8_predictions = quantized_model_int8.forward(&images_4d);
    let int8_time = test_start.elapsed();

    let int8_acc = accuracy(&int8_predictions, &labels);
    println!("Int8 quantized - Accuracy: {:.2}%, Time: {:.2}ms", 
             int8_acc * 100.0, int8_time.as_millis());

    // Test Float16 quantized model
    println!("Testing Float16 quantized model...");
    let test_start = Instant::now();
    let f16_predictions = quantized_model_f16.forward(&images_4d);
    let f16_time = test_start.elapsed();

    let f16_acc = accuracy(&f16_predictions, &labels);
    println!("Float16 quantized - Accuracy: {:.2}%, Time: {:.2}ms", 
             f16_acc * 100.0, f16_time.as_millis());

    println!("\n{}\n", "=".repeat(60));
    println!("Quantization Summary:");
    println!("   Original model:  {:.2}% accuracy, {:.2}ms", original_acc * 100.0, original_time.as_millis());
    println!("   Int8 quantized:  {:.2}% accuracy, {:.2}ms", int8_acc * 100.0, int8_time.as_millis());
    println!("   Float16 quantized: {:.2}% accuracy, {:.2}ms", f16_acc * 100.0, f16_time.as_millis());

    // Show some sample predictions
    println!("\nSample predictions (first 5):");
    let pred_classes = int8_predictions.argmax(Some(1));
    for i in 0..5.min(batch_size) {
        let predicted = pred_classes.data()[i] as u8;
        let actual = labels.data()[i] as u8;
        println!("   Sample {}: Predicted={}, Actual={} {}", 
                 i + 1, predicted, actual, 
                 if predicted == actual { "CORRECT" } else { "WRONG" });
    }

    Ok(())
}
