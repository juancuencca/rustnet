use crate::matrix::Matrix;

pub struct MeanSquaredError;

impl MeanSquaredError {
    fn calculate(target: f64, predicted: f64) -> f64 {
        (target - predicted).powi(2)
    }

    fn mse_derivative(y_pred: f64, y_true: f64) -> f64 {
        2.0 * (y_true - y_pred) 
    }

    pub fn compute_gradient(target: &Matrix, predicted: &Matrix) -> Matrix {
        let n = target.values.len() as f64;
    
        let grad_values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&y_true, &y_pred)| Self::mse_derivative(y_true, y_pred) / n)
            .collect::<Vec<f64>>();
    
        Matrix::new((grad_values.len(), 1), grad_values)
    }

    pub fn apply(target: &Matrix, predicted: &Matrix) -> f64 {
        assert_eq!(target.rows, predicted.rows, "Error: missmatch matrix dimension");
        assert_eq!(target.cols, predicted.cols, "Error: missmatch matrix dimension");

        let n = target.values.len();

        let values = (0..n)
            .into_iter()
            .map(|i| Self::calculate(target.values[i], predicted.values[i]))
            .collect::<Vec<f64>>();

        values
            .iter()
            .fold(0.0, |acc, x| acc + (x / n as f64))
    }
}