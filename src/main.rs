use taper::activation::Sigmoid;
use taper::loss::bce_loss;
use taper::nn::{Linear, Module, Sequential};
use taper::optim::{Optimizer, SGD};
use taper::{Tape, Tensor};
use std::env;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    println!("🧠 XOR Neural Network Training");
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


    let params = model.parameters();
    let mut opt = SGD::new(params, 0.10, None);

    let epochs = 50_000usize;

    for epoch in 0..epochs {
        Tape::reset(); // clear tape for new forward/backward pass

        let x = Tensor::new(x_data.clone(), &[4, 2]);
        let y = Tensor::new(y_data.clone(), &[4, 1]);

        // Use regular forward pass (quantization happens after training)
        let yhat = model.forward(&x);
        
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
    
    // Use regular forward pass for final evaluation
    let yhat = model.forward(&x);
    
    let p = yhat.data();

    println!(
        "\n[0,0]->{:.3}\n[0,1]->{:.3}\n[1,0]->{:.3}\n[1,1]->{:.3}",
        p[0], p[1], p[2], p[3]
    );

    let ok = (p[0] < 0.5) && (p[1] > 0.5) && (p[2] > 0.5) && (p[3] < 0.5);
    println!("{}", if ok { "learned XOR" } else { "not yet" });

}

