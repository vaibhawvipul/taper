use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::time::Instant;
use std::env;
use taper::Tape;
use taper::activation::ReLU;
use taper::data::mnist::{DataLoader, MNISTDataset};
use taper::loss::{accuracy, cross_entropy_loss};
use taper::nn::{Conv2dReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Module, Sequential, Flatten};
use taper::optim::Adam;
use taper::train::Trainer;
use taper::{QuantizationConfig, quantization::QuantizationType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let (quantize, quant_type, epochs, batch_size, lr) = parse_args(&args);

    println!("🚀 Quantized CNN MNIST Training");
    println!("Quantization: {} ({})", if quantize { "Enabled" } else { "Disabled" }, 
             if quantize { format!("{:?}", quant_type) } else { "N/A".to_string() });
    println!("Epochs: {}, Batch Size: {}, Learning Rate: {}\n", epochs, batch_size, lr);

    // Load MNIST dataset
    println!("Loading MNIST dataset...");
    let train_dataset = MNISTDataset::new(true, None)?;
    let test_dataset = MNISTDataset::new(false, None)?;

    println!("Training set: {} samples", train_dataset.len());
    println!("Test set: {} samples\n", test_dataset.len());

    // Create data loaders
    let mut train_loader = DataLoader::new(train_dataset, batch_size, true);
    let mut test_loader = DataLoader::new(test_dataset, batch_size, false);

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

    // Create quantization config
    let qconfig = QuantizationConfig::new(quantize, quant_type);

    // Create optimizer
    let optimizer = Adam::new(params, lr, None, None, Some(0.0001));

    // Create trainer
    let mut trainer = Trainer::new(
        Box::new(model),
        optimizer,
        None,
    );

    println!("\n{}\n", "=".repeat(60));

    // Training loop
    let total_start = Instant::now();
    
    for epoch in 1..=epochs {
        let epoch_start = Instant::now();

        println!("Epoch {}/{}", epoch, epochs);

        // Training phase
        trainer.model.parameters().iter().for_each(|p| p.zero_grad());

        let mut train_loss = 0.0;
        let mut train_correct = 0;
        let mut train_total = 0;

        train_loader.reset();
        let num_batches = train_loader.num_batches();

        for (batch_idx, (images, labels)) in train_loader.by_ref().enumerate() {
            let batch_start = Instant::now();
            Tape::reset();

            // Reshape images from [B, 784] to [B, 1, 28, 28] for CNN
            let batch_size = images.shape()[0];
            let images_4d = images.reshape(&[batch_size, 1, 28, 28]);
            let reshape_time = batch_start.elapsed();

            // Forward pass - use quantized if enabled
            let forward_start = Instant::now();
            let logits = if quantize {
                trainer.model.forward_quantized(&images_4d, &qconfig)
            } else {
                trainer.model.forward(&images_4d)
            };
            let forward_time = forward_start.elapsed();

            // Compute loss
            let loss_start = Instant::now();
            let loss = cross_entropy_loss(&logits, &labels);
            let loss_time = loss_start.elapsed();

            // Compute accuracy
            let batch_acc = accuracy(&logits, &labels);
            train_correct += (batch_acc * labels.shape()[0] as f32) as usize;
            train_total += labels.shape()[0];

            // Backward pass
            let backward_start = Instant::now();
            loss.backward();
            let backward_time = backward_start.elapsed();

            // Update weights
            let update_start = Instant::now();
            trainer.optimizer.step();
            trainer.optimizer.zero_grad();
            let update_time = update_start.elapsed();

            train_loss += loss.data()[0];
            let total_batch_time = batch_start.elapsed();

            // Progress update with timing info
            if (batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1 {
                print!(
                    "\r   Batch [{}/{}] Loss: {:.4}, Acc: {:.2}% | Times: Fwd:{:.1}ms Bwd:{:.1}ms Upd:{:.1}ms Total:{:.1}ms",
                    batch_idx + 1,
                    num_batches,
                    loss.data()[0],
                    100.0 * train_correct as f32 / train_total as f32,
                    forward_time.as_millis(),
                    backward_time.as_millis(),
                    update_time.as_millis(),
                    total_batch_time.as_millis(),
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

            // Use quantized forward pass for validation too
            let logits = if quantize {
                trainer.model.forward_quantized(&images_4d, &qconfig)
            } else {
                trainer.model.forward(&images_4d)
            };

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
        
        // Use quantized forward pass for final test
        let predictions = if quantize {
            trainer.model.forward_quantized(&images_4d, &qconfig)
        } else {
            trainer.model.forward(&images_4d)
        };
        
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

    // Print quantization summary
    if quantize {
        println!("\n📊 Quantization Summary:");
        println!("   Type: {:?}", quant_type);
        println!("   Bit width: {}", qconfig.bit_width());
        println!("   Integer quantization: {}", qconfig.is_integer());
        println!("   Float quantization: {}", qconfig.is_float());
    }

    Ok(())
}

fn parse_args(args: &[String]) -> (bool, QuantizationType, usize, usize, f32) {
    let mut quantize = false;
    let mut quant_type = QuantizationType::Int8;
    let mut epochs = 10;
    let mut batch_size = 256;
    let mut lr = 0.01;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--quantize" | "-q" => {
                quantize = true;
                // Check if next arg specifies quantization type
                if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                    quant_type = match args[i].as_str() {
                        "int8" => QuantizationType::Int8,
                        "int4" => QuantizationType::Int4,
                        "float16" => QuantizationType::Float16,
                        "bfloat16" => QuantizationType::BFloat16,
                        "nf4" => QuantizationType::NF4,
                        _ => {
                            eprintln!("Unknown quantization type: {}. Using Int8.", args[i]);
                            QuantizationType::Int8
                        }
                    };
                }
            }
            "--epochs" | "-e" => {
                if i + 1 < args.len() {
                    i += 1;
                    epochs = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid epochs value: {}. Using default: 10", args[i]);
                        10
                    });
                }
            }
            "--batch-size" | "-b" => {
                if i + 1 < args.len() {
                    i += 1;
                    batch_size = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid batch size: {}. Using default: 256", args[i]);
                        256
                    });
                }
            }
            "--lr" | "-l" => {
                if i + 1 < args.len() {
                    i += 1;
                    lr = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid learning rate: {}. Using default: 0.01", args[i]);
                        0.01
                    });
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                eprintln!("Use --help for usage information");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    (quantize, quant_type, epochs, batch_size, lr)
}

fn print_help() {
    println!("Quantized CNN MNIST Training");
    println!();
    println!("Usage: cargo run --example train_quantized_cnn [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -q, --quantize [TYPE]    Enable quantization (int8|int4|float16|bfloat16|nf4)");
    println!("  -e, --epochs NUM         Number of training epochs (default: 10)");
    println!("  -b, --batch-size NUM     Batch size (default: 256)");
    println!("  -l, --lr RATE            Learning rate (default: 0.01)");
    println!("  -h, --help               Show this help message");
    println!();
    println!("Examples:");
    println!("  cargo run --example train_quantized_cnn");
    println!("  cargo run --example train_quantized_cnn --quantize int8 --epochs 20");
    println!("  cargo run --example train_quantized_cnn -q float16 -e 15 -b 128 -l 0.005");
}
