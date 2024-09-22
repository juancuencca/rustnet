use rustnet::nn::loss::MeanSquaredError;
use rustnet::nn::{activations::Sigmoid, linear::Linear, sequential::Sequential};
use rustnet::matrix::Matrix;

fn main() {
    let seed = Some(42);

    let mut sequential = Sequential::new();
    sequential.add_layer(Box::new(Linear::new(2, 2, seed)));
    sequential.add_layer(Box::new(Sigmoid::new()));
    sequential.add_layer(Box::new(Linear::new(2, 1, seed)));
    sequential.add_layer(Box::new(Sigmoid::new()));

    let x = Matrix::new((4, 2), vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ]);

    let y = Matrix::new((4, 1), vec![
        0.0,
        1.0,
        1.0,
        1.0,
    ]);

    let predictions = sequential.forward(&x);

    let mse_loss = MeanSquaredError::apply(&y, &predictions);
    println!("Loss: {mse_loss}"); 
}