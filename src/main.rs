use taper::activation::Sigmoid;
use taper::loss::bce_loss;
use taper::nn::{Linear, Module, Sequential};
use taper::optim::{Optimizer, SGD};
use taper::{Tape, Tensor, QuantizationConfig, QuantizationType};
use std::env;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let (quantize, quant_type) = parse_xor_args(&args);

    println!("🧠 XOR Neural Network Training");
    if quantize {
        println!("Quantization: Enabled ({:?})", quant_type);
    } else {
        println!("Quantization: Disabled");
    }
    println!();

    // XOR data
    let x_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y_data = vec![0.0, 1.0, 1.0, 0.0];

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4, true)),
        Box::new(Sigmoid),
        Box::new(Linear::new(4, 1, true)),
        Box::new(Sigmoid),
    ]);

    // Create quantization config
    let qconfig = QuantizationConfig::new(quantize, quant_type);

    let params = model.parameters();
    let mut opt = SGD::new(params, 0.10, None);

    let epochs = 50_000usize;

    for epoch in 0..epochs {
        Tape::reset(); // clear tape for new forward/backward pass

        let x = Tensor::new(x_data.clone(), &[4, 2]);
        let y = Tensor::new(y_data.clone(), &[4, 1]);

        // Use quantized forward pass if enabled
        let yhat = if quantize {
            model.forward_quantized(&x, &qconfig)
        } else {
            model.forward(&x)
        };
        
        let loss = bce_loss(&yhat, &y);

        loss.backward();
        opt.step();
        opt.zero_grad();

        if epoch % 1000 == 0 {
            println!("iteration {:4}: Loss = {:.4}", epoch, loss.data()[0]);
        }
    }

    // final eval
    let _tape = Tape::reset();
    let x = Tensor::new(x_data, &[4, 2]);
    
    // Use quantized forward pass for final evaluation
    let yhat = if quantize {
        model.forward_quantized(&x, &qconfig)
    } else {
        model.forward(&x)
    };
    
    let p = yhat.data();

    println!(
        "\n[0,0]->{:.3}\n[0,1]->{:.3}\n[1,0]->{:.3}\n[1,1]->{:.3}",
        p[0], p[1], p[2], p[3]
    );

    let ok = (p[0] < 0.5) && (p[1] > 0.5) && (p[2] > 0.5) && (p[3] < 0.5);
    println!("{}", if ok { "learned XOR" } else { "not yet" });

    // Print quantization summary
    if quantize {
        println!("\n📊 Quantization Summary:");
        println!("   Type: {:?}", quant_type);
        println!("   Bit width: {}", qconfig.bit_width());
        println!("   Integer quantization: {}", qconfig.is_integer());
        println!("   Float quantization: {}", qconfig.is_float());
    }
}

fn parse_xor_args(args: &[String]) -> (bool, QuantizationType) {
    let mut quantize = false;
    let mut quant_type = QuantizationType::Int8;

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
            "--help" | "-h" => {
                print_xor_help();
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

    (quantize, quant_type)
}

fn print_xor_help() {
    println!("XOR Neural Network Training");
    println!();
    println!("Usage: cargo run [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -q, --quantize [TYPE]    Enable quantization (int8|int4|float16|bfloat16|nf4)");
    println!("  -h, --help               Show this help message");
    println!();
    println!("Examples:");
    println!("  cargo run");
    println!("  cargo run --quantize int8");
    println!("  cargo run -q float16");
}
