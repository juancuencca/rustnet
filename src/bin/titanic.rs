use std::{path::Path, error::Error, process};
use rustnet::matrix::Matrix;
use rustnet::metrics::accuracy;
use rustnet::nn::{sequential::Sequential, activations::{Sigmoid}, linear::Linear, loss::{BinaryCrossEntropy, LossFunction}};

fn main() {
    let train_path = Path::new("C:/Users/juan9/Documents/Education/Code/ML/datasets/titanic/titanic_train_46cols.csv");
    let result = read_csv(train_path);

    if let Err(err) = result {
        eprintln!("error running read csv: {}", err);
        process::exit(1);
    }

    if let Ok((train_x, train_y)) = result {
        let mut model_v0 = ModelV0::new(train_x.cols, 7, train_y.cols, 1.0, Some(42));
        model_v0.train(&train_x, &train_y, 5000);
    }  
}

struct ModelV0 {
    sequential: Sequential,
}

impl ModelV0 {
    fn new(input_size: usize, hidden_units: usize, output_size:usize, lr: f64, seed: Option<u64>) -> ModelV0 {
        let mut sequential = Sequential::new();
        let (l_range, r_range) = (-0.5, 0.5);
        sequential.add_layer(Box::new(Linear::new(input_size, hidden_units, lr, (l_range, r_range), seed)));
        sequential.add_layer(Box::new(Sigmoid::new()));
        sequential.add_layer(Box::new(Linear::new(hidden_units, output_size, lr, (l_range, r_range), seed)));
        sequential.add_layer(Box::new(Sigmoid::new()));
        
        ModelV0 {
            sequential
        }
    }

    fn train(&mut self, train_x: &Matrix, train_y: &Matrix, epochs: usize) {
        for epoch in 0..epochs {
            let predictions = self.sequential.forward(&train_x);    
            let loss_grad = BinaryCrossEntropy::gradient(&train_y, &predictions);
            self.sequential.backward(&loss_grad);
            
            if epoch % (epochs / 10) == 0 {
                let loss = BinaryCrossEntropy::calculate(&train_y, &predictions);
                
                let pred_values = round(&predictions.values, 0.5);
                let acc = accuracy(&train_y.values, &pred_values);

                println!("Epoch {}: Loss = {} Accuracy: {}", epoch, loss, acc);
            }
        }
    }
}

fn round(values: &[f64], threshold: f64) -> Vec<f64> {
    values.iter().map(|&x| { if x < threshold { 0.0 } else { 1.0 } }).collect()
}

fn read_csv(path: &Path) -> Result<(Matrix, Matrix), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;

    let mut x_values = Vec::new(); 
    let mut y_values = Vec::new();

    for result in rdr.records() {
        let record = result?;

        for (i, item) in record.iter().enumerate() {
            let item_f64 = item.parse::<f64>()?;
            if i == 0 {
                y_values.push(item_f64);
                continue; 
            }
            x_values.push(item_f64);
        } 
    }

    let x_matrix = Matrix::new((x_values.len() / 45, 45), x_values); 
    let y_matrix = Matrix::new((y_values.len(), 1), y_values);

    Ok((x_matrix, y_matrix))
}