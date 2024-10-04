use rustnet::nn::{activations::Sigmoid, linear::Linear, loss::{Loss, BCELoss}, sequential::Sequential};
use rustnet::matrix::Matrix;

const L_RNG: f64 = 0.0;
const R_RNG: f64 = 1.0;

pub struct GatesModel {
    sequential: Sequential,
}

impl GatesModel {
    pub fn new(input_size: usize, hidden_units: usize, output_size: usize, lr: f64, seed: Option<u64>) -> GatesModel {
        let mut sequential = Sequential::new();

        sequential.add_layer(Box::new(Linear::new(input_size, hidden_units, L_RNG, R_RNG, lr, seed)));
        sequential.add_layer(Box::new(Linear::new(hidden_units, hidden_units, L_RNG, R_RNG, lr, seed)));
        sequential.add_layer(Box::new(Linear::new(hidden_units, hidden_units, L_RNG, R_RNG, lr, seed)));
        sequential.add_layer(Box::new(Linear::new(hidden_units, output_size, L_RNG, R_RNG, lr, seed)));
        sequential.add_layer(Box::new(Sigmoid::new()));

        GatesModel { 
            sequential 
        }
    }

    pub fn train(&mut self, train_x: &Matrix, train_y: &Matrix, epochs: usize, loss_fn: &mut dyn Loss) {
        for epoch in 0..epochs {
            let predictions = self.sequential.forward(&train_x);    
            let loss = loss_fn.compute(&train_y, &predictions);
        
            let loss_grad = loss_fn.gradient();
            
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
    let train_x = Matrix::new(4, 2, vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ]);

    let train_y = Matrix::new(4, 1, vec![
        0.0,
        1.0,
        1.0,
        1.0,
    ]);
    
    let input_size = train_x.cols;
    let hidden_units = 2;
    let output_size = train_y.cols;
    let learning_rate = 1.0;
    let seed = Some(42);

    let mut model = GatesModel::new(input_size, hidden_units, output_size, learning_rate, seed);
    
    model.train(&train_x, &train_y, 100, &mut BCELoss::new());

    println!("{:?}", model.predict(&train_x));
}
