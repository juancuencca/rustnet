use std::{error::Error, path::Path, process};
use rustnet::matrix::Matrix;

fn main() {
    let path = Path::new("C:/Users/juan9/Documents/Education/Code/ML/datasets/mnist_csv/mnist_train.csv");
    let result = read_from_path(path, 5);

    if let Ok((train_x, train_y)) = result {
        println!("Train x");
        println!("rows: {}", train_x.rows);
        println!("cols: {}", train_x.cols);
        println!("values lenght: {}", train_x.values.len());
        
        println!("\nTrain y");
        println!("rows: {}", train_y.rows);
        println!("cols: {}", train_y.cols);
        println!("values lenght: {}", train_y.values.len());

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
            x_values.push(item_f64);
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