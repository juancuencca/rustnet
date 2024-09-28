use std::{error::Error, path::Path, process};
use rustnet::{matrix::Matrix, nn::{activations::{Sigmoid, Softmax}, linear::Linear, loss::{CrossEntropy, MulticlassLossFunction}, sequential::Sequential}};

struct ModelV0 {
    sequential: Sequential,
}

impl ModelV0 {
    fn new(input_size: usize, hidden_units: usize, output_size:usize, lr: f64, seed: Option<u64>) -> ModelV0 {
        let mut sequential = Sequential::new();
        let (l_range, r_range) = (0.0, 1.0);

        sequential.add_layer(Box::new(Linear::new(input_size, hidden_units, lr, (l_range, r_range), seed)));
        sequential.add_layer(Box::new(Sigmoid::new()));
        sequential.add_layer(Box::new(Linear::new(hidden_units, output_size, lr, (l_range, r_range), seed)));
        sequential.add_layer(Box::new(Softmax::new()));
        
        ModelV0 {
            sequential
        }
    }

    fn train(&mut self, train_x: &Matrix, train_y: &Matrix, epochs: usize) {
        for epoch in 0..epochs {
            let predictions = self.sequential.forward(&train_x);    
            let loss_grad = CrossEntropy::gradient(&train_y, &predictions);
            self.sequential.backward(&loss_grad);
            
            if epoch % (epochs / 10) == 0 {
                let loss = CrossEntropy::calculate(&train_y, &predictions);

                let y_true = one_hot_reverse(&train_y.values, train_y.cols);
                let y_pred = one_hot_reverse(&predictions.values, predictions.cols);
                let acc = accuracy(&y_true, &y_pred);

                println!("Epoch {}: Loss = {} Accuracy: {}", epoch, loss, acc);
            }
        }
    }

    fn test(&mut self, test_x: &Matrix, test_y: &Matrix) {        
        let predictions = self.sequential.forward(&test_x);    

        let loss = CrossEntropy::calculate(&test_y, &predictions);

        let y_true = one_hot_reverse(&test_y.values, test_y.cols);
        let y_pred = one_hot_reverse(&predictions.values, predictions.cols);
        let acc = accuracy(&y_true, &y_pred);

        println!("Loss = {} Accuracy: {}", loss, acc);
    }
}

fn main() {
    let train_path = Path::new("C:/Users/juan9/Documents/Education/Code/ML/datasets/mnist_csv/mnist_train.csv");
    let result = read_csv(train_path, Some(10));

    if let Ok((train_x, train_y)) = result {
        let mut model_v0 = ModelV0::new(train_x.cols, 8, train_y.cols, 1.0, Some(42));
        model_v0.train(&train_x, &train_y, 10);

        let test_path = Path::new("C:/Users/juan9/Documents/Education/Code/ML/datasets/mnist_csv/mnist_test.csv");
        if let Ok((test_x, test_y)) = read_csv(test_path, None) {
            model_v0.test(&test_x, &test_y);
        }
    } else if let Err(err) = result {
        println!("error running example: {}", err);
        process::exit(1);
    } 
}

fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "array's length mismatch");
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(x1, x2)| x1 == x2).count();
        
    correct as f64 / y_true.len() as f64
}

fn one_hot_reverse(values: &[f64], chunk_size: usize) -> Vec<f64> {
    let mut new_values = Vec::new();
    
    for chunk in values.chunks(chunk_size) {
        let mut max_value = f64::NEG_INFINITY;
        let mut index = 0;
        
        for (i, &value) in chunk.iter().enumerate() {
            if value > max_value {
                max_value = value;
                index = i;
            } 
        }
        new_values.push(index as f64);
    }
    
    return new_values;
}

fn one_hot(items: &[f64]) -> Vec<f64> {
    let size = items.len() * 10; 
    let mut one_hot = vec![0.0; size];

    for &item in items {
        let index = item as usize;

        if index < size {
            one_hot[index] = 1.0;
        }
    }

    return one_hot;
}

fn read_csv(path: &Path, max_records: Option<usize>) -> Result<(Matrix, Matrix), Box<dyn Error>> {
    let (x_values, y_values) = get_data(path, max_records)?;

    let x_matrix = Matrix::new((x_values.len() / 784, 784), x_values); 
    let y_matrix = Matrix::new((y_values.len() / 10, 10), y_values);

    Ok((x_matrix, y_matrix))
}

fn get_data(path: &Path, max_records: Option<usize>) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut x_values = Vec::new(); 
    let mut y_values = Vec::new();

    for (n, result) in rdr.records().enumerate() {
        let record = result?;
        for (i, item) in record.iter().enumerate() {
            let item_f64 = item.parse::<f64>()?;
            if i == 0 {
                y_values.push(item_f64);
                continue; 
            }
            x_values.push(item_f64);
        } 

        if let Some(max_records) = max_records {
            if n == max_records - 1 {
                break;    
            }
        }
    }

    let one_hot_values = one_hot(&y_values);
    
    Ok((x_values, one_hot_values))
}