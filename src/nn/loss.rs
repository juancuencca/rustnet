use crate::matrix::Matrix;
use std::f64::EPSILON; 

pub trait LossFunction {
    fn loss(target: f64, predicted: f64) -> f64;
    fn derivative(target: f64, predicted: f64) -> f64;

    fn calculate(target: &Matrix, predicted: &Matrix) -> f64 {
        let n = target.values.len() as f64;
        let values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&y_true, &y_pred)| Self::loss(y_true, y_pred))
            .collect::<Vec<f64>>();

        values.iter().fold(0.0, |acc, x| acc + (x / n))
    }

    fn gradient(target: &Matrix, predicted: &Matrix) -> Matrix {
        let n = target.values.len() as f64;

        let grad_values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&y_true, &y_pred)| Self::derivative(y_true, y_pred) / n)
            .collect::<Vec<f64>>();

        Matrix::new((grad_values.len(), 1), grad_values)
    }
}

pub struct MeanSquaredError;

impl LossFunction for MeanSquaredError {
    fn loss(target: f64, predicted: f64) -> f64 {
        (target - predicted).powi(2)
    }

    fn derivative(target: f64, predicted: f64) -> f64 {
        2.0 * (predicted - target) 
    }
}

pub struct MeanAbsoluteError;

impl LossFunction for MeanAbsoluteError {
    fn loss(target: f64, predicted: f64) -> f64 {
        (target - predicted).abs()
    }

    fn derivative(target: f64, predicted: f64) -> f64 {
        if predicted > target { 1.0 } 
        else if predicted < target { -1.0 } 
        else { 0.0 } 
    }
}

pub struct BinaryCrossEntropy;

impl LossFunction for BinaryCrossEntropy {
    fn loss(target: f64, predicted: f64) -> f64 {
        let predicted = predicted.clamp(EPSILON, 1.0 - EPSILON);

        -target * predicted.ln() - (1.0 - target) * (1.0 - predicted).ln()
    }

    fn derivative(target: f64, predicted: f64) -> f64 {
        let predicted = predicted.clamp(EPSILON, 1.0 - EPSILON);

        (predicted - target) / (predicted * (1.0 - predicted))
    }
}

pub trait MulticlassLossFunction {
    fn loss(target: &[f64], predicted: &[f64]) -> f64;
    fn derivative(target: &[f64], predicted: &[f64]) -> Vec<f64>;

    fn calculate(target: &Matrix, predicted: &Matrix) -> f64 {
        let n = target.values.len() as f64;

        // For each row (representing one sample), calculate the loss for the corresponding target and predicted row
        let values = target.values
            .chunks(target.cols) // Split by rows
            .zip(predicted.values.chunks(predicted.cols)) // Corresponding rows in predicted
            .map(|(y_true_row, y_pred_row)| Self::loss(y_true_row, y_pred_row))
            .collect::<Vec<f64>>();

        // Sum the individual losses and average over the number of samples
        values.iter().fold(0.0, |acc, x| acc + (x / n))
    }

    fn gradient(target: &Matrix, predicted: &Matrix) -> Matrix {
        let n = target.values.len() as f64;

        // For each row (representing one sample), calculate the gradient for the corresponding target and predicted row
        let grad_values = target.values
            .chunks(target.cols)
            .zip(predicted.values.chunks(predicted.cols))
            .flat_map(|(y_true_row, y_pred_row)| {
                Self::derivative(y_true_row, y_pred_row)
                    .iter()
                    .map(|grad| grad / n)
                    .collect::<Vec<f64>>() // Flatten back into a vector
            })
            .collect::<Vec<f64>>();

        Matrix::new((grad_values.len() / target.cols, target.cols), grad_values)
    }
}



pub struct CrossEntropy;

impl MulticlassLossFunction for CrossEntropy {
    // fn loss(target: &[f64], predicted: &[f64]) -> f64 {
    //     let mut loss = 0.0;
    //     for (t, p) in target.iter().zip(predicted.iter()) {
    //         let p = p.clamp(EPSILON, 1.0 - EPSILON); 
    //         loss += -t * p.ln();
    //     }
    //     loss
    // }

    fn loss(target: &[f64], predicted: &[f64]) -> f64 {
        target.iter().zip(predicted.iter())
            .map(|(&t, &p)| {
                let p = p.clamp(std::f64::EPSILON, 1.0 - std::f64::EPSILON);
                -t * p.ln()
            })
            .sum()
    }

    fn derivative(target: &[f64], predicted: &[f64]) -> Vec<f64> {
        target.iter().zip(predicted.iter())
            .map(|(t, p)| p - t)
            .collect()
    }
}