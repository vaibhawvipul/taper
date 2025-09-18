use taper::{Tape, Tensor};
use taper::nn::{Linear, Module, Sequential};
use taper::activation::Sigmoid;
use taper::optim::{Optimizer, SGD};
use taper::loss::bce_loss;

fn main() {
    // XOR
    let x_data = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ];
    let y_data = vec![0.0, 1.0, 1.0, 0.0];

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4, true)),
        Box::new(Sigmoid),
        Box::new(Linear::new(4, 1, true)),
        Box::new(Sigmoid),
    ]);

    let params = model.parameters();
    let mut opt = SGD::new(params, 0.10, None);

    let epochs = 10_000usize;

    for epoch in 0..epochs {
        let _tape = Tape::new();

        let x = Tensor::new(x_data.clone(), &[4, 2]);
        let y = Tensor::new(y_data.clone(), &[4, 1]);

        let yhat = model.forward(&x);
        let loss = bce_loss(&yhat, &y);

        loss.backward();
        opt.step();
        opt.zero_grad();

        if epoch % 500 == 0 {
            println!("iteration {:4}: Loss = {:.4}", epoch, loss.data()[0]);
        }
    }

    // final eval
    let _tape = Tape::new();
    let x = Tensor::new(x_data, &[4, 2]);
    let yhat = model.forward(&x);
    let p = yhat.data();

    println!(
        "\n[0,0]->{:.3}\n[0,1]->{:.3}\n[1,0]->{:.3}\n[1,1]->{:.3}",
        p[0], p[1], p[2], p[3]
    );

    let ok = (p[0] < 0.5) && (p[1] > 0.5) && (p[2] > 0.5) && (p[3] < 0.5);
    println!("{}", if ok { "learned XOR" } else { "not yet" });
}
