use std::{error::Error, path::Path, process};
use rustnet::{matrix::Matrix, nn::{activations::{Relu, Sigmoid, Softmax}, linear::Linear, loss::{CrossEntropy, MulticlassLossFunction}, sequential::Sequential}};

fn main() {
    let path = Path::new("C:/Users/juan9/Documents/Education/Code/ML/datasets/mnist_csv/mnist_train.csv");
    let result = read_from_path(path, 10000);

    if let Ok((train_x, train_y)) = result {
        
        let seed = Some(42);
        let mut sequential = Sequential::new();
        sequential.add_layer(Box::new(Linear::new(784, 8, seed)));
        sequential.add_layer(Box::new(Relu::new()));
        sequential.add_layer(Box::new(Linear::new(8, 10, seed)));
        sequential.add_layer(Box::new(Softmax::new()));

        // let predictions = sequential.forward(&train_x);
    
        // println!("rows: {}", predictions.rows);
        // println!("cols: {}", predictions.cols);
        // println!("first 20 values: {:?}", &predictions.values[0..20]);

        let epochs = 10;

        for epoch in 0..epochs {
            let predictions = sequential.forward(&train_x);    
            let loss = CrossEntropy::calculate(&train_y, &predictions);
        
            let loss_grad = CrossEntropy::gradient(&train_y, &predictions);
            
            sequential.backward(&loss_grad);
            
            if epoch % (epochs / 10) == 0 {
                println!("Epoch {}: Loss = {}", epoch, loss);
            }
        }

    } else if let Err(err) = result {
        println!("error running example: {}", err);
        process::exit(1);
    } 


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

fn read_from_path(path: &Path, max_records: usize) -> Result<(Matrix, Matrix), Box<dyn Error>> {
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
            x_values.push(item_f64 / 255.0);
        } 

        if n == max_records - 1 {
            break;    
        }
    }

    let one_hot_values = one_hot(&y_values);
 
    Ok((
        Matrix::new((x_values.len() / 784, 784), x_values),
        Matrix::new((one_hot_values.len() / 10, 10), one_hot_values),
    ))
}