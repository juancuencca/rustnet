use crate::matrix::Matrix;

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

    fn derivative(y_pred: f64, y_true: f64) -> f64 {
        2.0 * (y_true - y_pred) 
    }
}