use rustnet::nn::{
    activations::Sigmoid, linear::Linear, loss::{
        LossFunction, MeanAbsoluteError, MeanSquaredError
    }, sequential::Sequential
};
use rustnet::matrix::Matrix;

pub struct ModelV0 {
    sequential: Sequential,
}

impl ModelV0 {
    pub fn new(input_size: usize, hidden_units: usize, output_size: usize, seed: Option<u64>) -> ModelV0 {
        let mut sequential = Sequential::new();
        sequential.add_layer(Box::new(Linear::new(input_size, hidden_units, seed)));
        sequential.add_layer(Box::new(Sigmoid::new()));
        sequential.add_layer(Box::new(Linear::new(hidden_units, output_size, seed)));
        sequential.add_layer(Box::new(Sigmoid::new()));

        ModelV0 { 
            sequential 
        }
    }

    pub fn train<F: LossFunction>(&mut self, train_x: &Matrix, train_y: &Matrix, epochs: usize, _loss_function: F) {
        for epoch in 0..epochs {
            let predictions = self.sequential.forward(&train_x);    
            let loss = F::calculate(&train_y, &predictions);
        
            let loss_grad = F::gradient(&train_y, &predictions);
            
            self.sequential.backward(&loss_grad);
            
            if epoch % (epochs / 10) == 0 {
                println!("Epoch {}: Loss = {}", epoch, loss);
            }
        }
    }

    pub fn predict(&mut self, train_x: &Matrix) -> Matrix {
        self.sequential.forward(&train_x)
    }
}



fn main() {
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

    let mut ModelV0 = ModelV0::new(2, 2, 1,Some(42));

    ModelV0.train(&train_x, &train_y, 5000, MeanSquaredError);

    ModelV0.train(&train_x, &train_y, 5000, MeanAbsoluteError);

    // let predictions = ModelV0.predict(&train_x);
    // println!("{predictions:#?}");
}
