use rustnet::nn::{
    loss::{
        MeanAbsoluteError,
        MeanSquaredError, 
        LossFunction
    },
    activations::Sigmoid, 
    linear::Linear, 
    sequential::Sequential
};
use rustnet::matrix::Matrix;

fn main() {
    let seed = Some(42);

    let mut sequential = Sequential::new();
    sequential.add_layer(Box::new(Linear::new(2, 2, seed)));
    sequential.add_layer(Box::new(Sigmoid::new()));
    sequential.add_layer(Box::new(Linear::new(2, 1, seed)));
    sequential.add_layer(Box::new(Sigmoid::new()));

    let train_x = Matrix::new((4, 2), vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ]);

    let train_y = Matrix::new((4, 1), vec![
        0.0,
        1.0,
        1.0,
        1.0,
    ]);

    let num_epochs = 5000;

    for epoch in 0..num_epochs {
        let predictions = sequential.forward(&train_x);
        let loss = MeanSquaredError::calculate(&train_y, &predictions);
        
        let loss_grad = MeanSquaredError::gradient(&train_y, &predictions); 
        sequential.backward(&loss_grad);
        
        if epoch % 500 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss);
        }
    }

    let predictions = sequential.forward(&train_x);
    println!("{predictions:#?}");
}
