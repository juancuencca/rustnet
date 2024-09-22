use crate::matrix::Matrix;

pub struct MeanSquaredError;

impl MeanSquaredError {
    fn calculate(target: f64, predicted: f64) -> f64 {
        (target - predicted).powi(2)
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