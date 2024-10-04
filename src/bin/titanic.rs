use std::{path::Path, error::Error, process};
use rustnet::matrix::Matrix;
use rustnet::metrics::{accuracy, sensitivity, false_positive_rate, specificity};
use rustnet::nn::{sequential::Sequential, activations::Sigmoid, linear::Linear, loss::{Loss, BCELoss, AUCReshaping}};

const L_RNG: f64 = -0.5;
const R_RNG: f64 = 0.5;

struct TitanicModel {
    sequential: Sequential,
}

impl TitanicModel {
    fn new(input_size: usize, hidden_units: usize, output_size:usize, lr: f64, seed: Option<u64>) -> TitanicModel {
        let mut sequential = Sequential::new();
        sequential.add_layer(Box::new(Linear::new(input_size, hidden_units, lr, (L_RNG, R_RNG), seed)));
        sequential.add_layer(Box::new(Sigmoid::new()));
        sequential.add_layer(Box::new(Linear::new(hidden_units, output_size, lr, (L_RNG, R_RNG), seed)));
        sequential.add_layer(Box::new(Sigmoid::new()));
        
        TitanicModel {
            sequential
        }
    }

    fn train(&mut self, train_x: &Matrix, train_y: &Matrix, epochs: usize, loss_fn: &mut dyn Loss) {
        for epoch in 0..epochs {
            
            let predictions = self.sequential.forward(&train_x);    
            
            let loss = loss_fn.compute(&train_y, &predictions);
            let loss_grad = loss_fn.gradient();

            self.sequential.backward(&loss_grad);
            
            if epoch % (epochs / 10) == 0 {
                let pred_values = round(&predictions.values, 0.5);
                let acc = accuracy(&train_y.values, &pred_values);
                let sensitivity = sensitivity(&train_y.values, &pred_values);
                let specificity = specificity(&train_y.values, &pred_values);
                
                println!("Epoch {}: Loss = {} Accuracy = {} Sensitivity = {} Specificity = {}", epoch, loss, acc, sensitivity, specificity);
            }
        }
    }

    fn calculate_roc(&mut self, train_x: &Matrix, train_y: &Matrix, n_threshold: usize) -> (Vec<f64>, Vec<f64>) {
        let predictions = self.sequential.forward(&train_x);

        let threshold_values = (0..=n_threshold).map(|i| i as f64 / n_threshold as f64).collect::<Vec<f64>>(); 
        let mut tpr_values = vec![];
        let mut fpr_values = vec![];

        for &threshold in threshold_values.iter() {
            let pred_values = round(&predictions.values, threshold);

            let tpr = sensitivity(&train_y.values, &pred_values);
            let fpr = false_positive_rate(&train_y.values, &pred_values);

            tpr_values.push(tpr);
            fpr_values.push(fpr);
        }

        (tpr_values, fpr_values)
    }
}


fn main() {
    let train_path = Path::new("C:/Users/juan9/Documents/Education/Code/ML/datasets/titanic/titanic_train_46cols.csv");
    let result = read_csv(train_path);

    if let Err(err) = result {
        eprintln!("error running read csv: {}", err);
        process::exit(1);
    }

    if let Ok((train_x, train_y)) = result {
        let mut auc_loss_fn = AUCReshaping::new(1e-4, 0.98);   
        let mut bce_loss_fn = BCELoss::new();

        let epochs = 2000;

        let input_size = train_x.cols;
        let hidden_units = 4;
        let output_size = train_y.cols;
        let lr = 1e-1;
        let seed = Some(42);

        // Model using bce
        let mut model = TitanicModel::new(input_size, hidden_units, output_size, lr, seed);
        model.train(&train_x, &train_y, epochs, &mut bce_loss_fn);
        let (tpr, fpr) = model.calculate_roc(&train_x, &train_y, 10);
        println!("tpr: {:?}", tpr);
        println!("fpr: {:?}", fpr);

        // Model using auc reshaping
        let mut model = TitanicModel::new(input_size, hidden_units, output_size, lr, seed);
        model.train(&train_x, &train_y, epochs, &mut auc_loss_fn);
        let (tpr, fpr) = model.calculate_roc(&train_x, &train_y, 10);
        println!("tpr: {:?}", tpr);
        println!("fpr: {:?}", fpr);
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